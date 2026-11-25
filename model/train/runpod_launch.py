"""Train the classifier on a RunPod GPU pod, over SSH, with no network volume.

Flow: create a pod (runpod SDK) -> copy the committed code (`git archive HEAD`) and the enhanced
data over scp -> run train.py detached under `timeout` -> copy `checkpoints/<run>/` back ->
terminate the pod. Settings come from `.env` (see example.env, RUNPOD_* and TRAIN_*).

    uv run python -m model.train.runpod_launch up                   # dry run: print the request
    uv run python -m model.train.runpod_launch run --yes -- --epochs 1 --limit-train 2000
    uv run python -m model.train.runpod_launch up --yes             # step by step instead:
    uv run python -m model.train.runpod_launch push
    uv run python -m model.train.runpod_launch train -- --epochs 3
    uv run python -m model.train.runpod_launch status | logs
    uv run python -m model.train.runpod_launch pull
    uv run python -m model.train.runpod_launch down

Spending: nothing is created without `--yes`. A pod whose hourly rate is above
RUNPOD_MAX_HOURLY_USD is terminated at once. Training is stopped after RUNPOD_MAX_HOURS. `run`
always terminates the pod at the end, also on errors and Ctrl+C. After step-by-step use,
`down` is your job: a pod keeps billing until it is terminated.

Memory: RUNPOD_MIN_RAM_GB and RUNPOD_MIN_VCPU are placement minimums, not caps. GPU memory is
bounded by the GPU type and the TRAIN_* settings (batch size, max length, gradient
checkpointing), which are passed to train.py as flags.

Secrets: the RunPod API key stays on this machine (it is not put in the pod env). Only the SSH
public key is sent. `git archive HEAD` uploads committed files only, so `.env` and `gitignore/`
never leave the machine.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from model.architecture import ENCODER_MODEL
from model.train.enhance_pipeline import load_dotenv

# Pinned image from the RunPod docs. torch itself comes from uv.lock, with its own CUDA runtime;
# the image only provides Python, sshd and the PUBLIC_KEY setup.
DEFAULT_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
# torch 2.15 from PyPI ships CUDA 13 libraries, so the host driver must support CUDA 13.
DEFAULT_ALLOWED_CUDA = ("13.0", "13.1", "13.2")
UV_VERSION = "0.13.9"  # same uv as the dev machine
REMOTE_DIR = "/root/havelai"
DATA_DIR = Path("model/train/data/enhanced")
DATA_FILES = ("train.jsonl", "test.jsonl")  # train.jsonl is required
CHECKPOINTS_DIR = Path("model/train/checkpoints")
STATE_FILE = Path("gitignore/temp/runpod_pod.json")
KNOWN_HOSTS = Path("gitignore/temp/runpod_known_hosts")
AUTO_SSH_KEY = Path("gitignore/temp/runpod_ed25519")  # the launcher's own key, gitignored
POD_NAME = "havelai-train"
CLOUD_TYPES = ("SECURE", "COMMUNITY", "ALL")
FAILED_STATUSES = ("EXITED", "ERROR", "TERMINATED")
SETUP_ALLOWANCE_S = 3600  # time for uv sync and the model download, on top of RUNPOD_MAX_HOURS


class LaunchError(RuntimeError):
    """A launch step failed; the message says what to do."""


def say(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


Runner = Callable[..., subprocess.CompletedProcess]


def run_command(
    argv: Sequence[str], input: bytes | None = None, capture: bool = False
) -> subprocess.CompletedProcess:
    return subprocess.run(list(argv), input=input, capture_output=capture, check=True)


# --------------------------------------------------------------------------- settings


@dataclass(frozen=True)
class LaunchSettings:
    api_key: str = ""
    gpu_type: str = "NVIDIA GeForce RTX 4090"
    gpu_count: int = 1
    cloud_type: str = "SECURE"
    image: str = DEFAULT_IMAGE
    container_disk_gb: int = 30
    min_ram_gb: int = 16
    min_vcpu: int = 4
    data_center: str | None = None
    allowed_cuda: tuple[str, ...] = DEFAULT_ALLOWED_CUDA
    max_hours: float = 6.0
    max_hourly_usd: float = 1.0
    ssh_key: Path = AUTO_SSH_KEY  # generated on the first `--yes` unless RUNPOD_SSH_KEY is set
    # training memory
    batch_size: int = 16
    eval_batch_size: int = 32
    grad_accum: int = 1
    max_length: int = 512
    gradient_checkpointing: bool = False
    cuda_alloc_conf: str = "expandable_segments:True"


def _get(env: Mapping[str, str], key: str) -> str | None:
    value = env.get(key, "").strip()
    return value or None


def _int(env: Mapping[str, str], key: str, default: int, minimum: int = 1) -> int:
    raw = _get(env, key)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise LaunchError(f"{key} must be an integer, got {raw!r}") from None
    if value < minimum:
        raise LaunchError(f"{key} must be at least {minimum}, got {value}")
    return value


def _float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = _get(env, key)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise LaunchError(f"{key} must be a number, got {raw!r}") from None
    if not value > 0:
        raise LaunchError(f"{key} must be above 0, got {value}")
    return value


def _bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = _get(env, key)
    if raw is None:
        return default
    if raw.lower() in ("1", "true", "yes", "on"):
        return True
    if raw.lower() in ("0", "false", "no", "off"):
        return False
    raise LaunchError(f"{key} must be 0 or 1, got {raw!r}")


def settings_from_env(env: Mapping[str, str]) -> LaunchSettings:
    """Read RUNPOD_* and TRAIN_* settings. Empty or missing values use the defaults."""
    d = LaunchSettings()
    cloud = (_get(env, "RUNPOD_CLOUD_TYPE") or d.cloud_type).upper()
    if cloud not in CLOUD_TYPES:
        raise LaunchError(f"RUNPOD_CLOUD_TYPE must be one of {', '.join(CLOUD_TYPES)}")
    image = _get(env, "RUNPOD_IMAGE") or d.image
    tag = image.rsplit("/", 1)[-1].partition(":")[2]
    if not tag or tag == "latest":
        raise LaunchError("RUNPOD_IMAGE must have a pinned tag, not :latest or no tag")
    cuda_raw = env.get("RUNPOD_ALLOWED_CUDA")
    allowed_cuda = (
        d.allowed_cuda
        if cuda_raw is None
        else tuple(v.strip() for v in cuda_raw.split(",") if v.strip())
    )
    ssh_key = _get(env, "RUNPOD_SSH_KEY")
    return LaunchSettings(
        api_key=_get(env, "RUNPOD_API_KEY") or "",
        gpu_type=_get(env, "RUNPOD_GPU_TYPE") or d.gpu_type,
        gpu_count=_int(env, "RUNPOD_GPU_COUNT", d.gpu_count),
        cloud_type=cloud,
        image=image,
        container_disk_gb=_int(env, "RUNPOD_CONTAINER_DISK_GB", d.container_disk_gb, minimum=10),
        min_ram_gb=_int(env, "RUNPOD_MIN_RAM_GB", d.min_ram_gb),
        min_vcpu=_int(env, "RUNPOD_MIN_VCPU", d.min_vcpu),
        data_center=_get(env, "RUNPOD_DATA_CENTER"),
        allowed_cuda=allowed_cuda,
        max_hours=_float(env, "RUNPOD_MAX_HOURS", d.max_hours),
        max_hourly_usd=_float(env, "RUNPOD_MAX_HOURLY_USD", d.max_hourly_usd),
        ssh_key=Path(ssh_key).expanduser() if ssh_key else d.ssh_key,
        batch_size=_int(env, "TRAIN_BATCH_SIZE", d.batch_size),
        eval_batch_size=_int(env, "TRAIN_EVAL_BATCH_SIZE", d.eval_batch_size),
        grad_accum=_int(env, "TRAIN_GRAD_ACCUM", d.grad_accum),
        max_length=_int(env, "TRAIN_MAX_LENGTH", d.max_length, minimum=8),
        gradient_checkpointing=_bool(env, "TRAIN_GRADIENT_CHECKPOINTING", d.gradient_checkpointing),
        cuda_alloc_conf=_get(env, "PYTORCH_CUDA_ALLOC_CONF") or d.cuda_alloc_conf,
    )


def require_api_key(settings: LaunchSettings) -> None:
    if not settings.api_key:
        raise LaunchError("RUNPOD_API_KEY is not set (see example.env).")


def _public_path(key: Path) -> Path:
    return key.with_name(key.name + ".pub")


def generate_ssh_key(key: Path, runner: Runner = run_command) -> None:
    """Create a passphrase-less ed25519 key pair used only for RunPod pods."""
    key.parent.mkdir(parents=True, exist_ok=True)
    runner(["ssh-keygen", "-t", "ed25519", "-N", "", "-q", "-C", "havelai-runpod", "-f", str(key)])
    if not key.is_file() or not _public_path(key).is_file():
        raise LaunchError(f"ssh-keygen did not create {key}")
    key.chmod(0o600)  # ssh refuses private keys readable by others
    say(f"created SSH key {key} (used only for RunPod pods, gitignored)")


def read_public_key(
    settings: LaunchSettings, create: bool = False, runner: Runner = run_command
) -> str:
    """The public half of the SSH key. With `create`, the launcher's own key (the default, when
    RUNPOD_SSH_KEY is not set) is generated if missing; a key you set yourself never is."""
    key, pub = settings.ssh_key, _public_path(settings.ssh_key)
    if not key.is_file() or not pub.is_file():
        if create and key == AUTO_SSH_KEY:
            generate_ssh_key(key, runner)
        elif key == AUTO_SSH_KEY:
            raise LaunchError(f"no SSH key yet: {key} is created on the first `--yes`.")
        else:
            raise LaunchError(
                f"SSH key {key} (and {pub.name}) not found. Create it with "
                f"`ssh-keygen -t ed25519 -f {key}`, or leave RUNPOD_SSH_KEY empty to let the "
                "launcher create its own."
            )
    return pub.read_text().strip()


# --------------------------------------------------------------------------- pure builders


def pod_request(settings: LaunchSettings, public_key: str, name: str = POD_NAME) -> dict[str, Any]:
    """Keyword arguments for `runpod.create_pod`. No API key in `env`, no persistent volume."""
    request: dict[str, Any] = {
        "name": name,
        "image_name": settings.image,
        "gpu_type_id": settings.gpu_type,
        "gpu_count": settings.gpu_count,
        "cloud_type": settings.cloud_type,
        "container_disk_in_gb": settings.container_disk_gb,
        "volume_in_gb": 0,
        "min_memory_in_gb": settings.min_ram_gb,
        "min_vcpu_count": settings.min_vcpu,
        "ports": "22/tcp",
        "start_ssh": True,
        "env": {
            "PUBLIC_KEY": public_key,
            "PYTORCH_CUDA_ALLOC_CONF": settings.cuda_alloc_conf,
        },
    }
    if settings.data_center:
        request["data_center_id"] = settings.data_center
    if settings.allowed_cuda:
        request["allowed_cuda_versions"] = list(settings.allowed_cuda)
    return request


def display_request(request: Mapping[str, Any]) -> str:
    """The request as JSON for the console, with the public key shortened."""
    shown = json.loads(json.dumps(request))
    key = shown.get("env", {}).get("PUBLIC_KEY")
    if key:
        shown["env"]["PUBLIC_KEY"] = f"{key[:24]}... ({len(key)} chars)"
    return json.dumps(shown, indent=2)


def ssh_endpoint(pod: Mapping[str, Any] | None) -> tuple[str, int] | None:
    """(host, port) for direct SSH to the pod, or None while it is not up yet.

    Same lookup as the SDK's own `get_pod_ssh_ip_port`: `ssh.direct` first, then the
    `runtime.ports` entry whose `private` port is 22 (REST v2 field names)."""
    if not pod:
        return None
    direct = (pod.get("ssh") or {}).get("direct") or {}
    if direct.get("host") and direct.get("port"):
        return str(direct["host"]), int(direct["port"])
    runtime = pod.get("runtime") or {}
    for port in runtime.get("ports") or []:
        if not isinstance(port, Mapping) or port.get("private") != 22:
            continue
        ip, public = port.get("ip"), port.get("public")
        if ip and public:
            return str(ip), int(public)
    return None


def train_flags(settings: LaunchSettings) -> list[str]:
    flags = [
        "--batch-size", str(settings.batch_size),
        "--eval-batch-size", str(settings.eval_batch_size),
        "--grad-accum", str(settings.grad_accum),
        "--max-length", str(settings.max_length),
    ]  # fmt: skip
    if settings.gradient_checkpointing:
        flags.append("--gradient-checkpointing")
    return flags


def default_run_name(now: datetime | None = None) -> str:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"{ENCODER_MODEL.split('/')[-1]}-runpod-{stamp}"


def remote_train_script(
    settings: LaunchSettings, run_name: str, extra_args: Sequence[str] = ()
) -> str:
    """Bash script run on the pod. Writes its exit code to run.exit whatever happens.

    124 in run.exit means RUNPOD_MAX_HOURS was reached (train.py got SIGINT first, so it still
    exported its partial metrics); 3 means the pod has no usable CUDA."""
    seconds = int(settings.max_hours * 3600)
    args = [*train_flags(settings), *extra_args, "--no-progress", "--run-name", run_name]
    cuda_check = (
        "import sys, torch; ok = torch.cuda.is_available(); "
        'print("torch", torch.__version__, "cuda", torch.version.cuda, "available", ok); '
        "sys.exit(0 if ok else 3)"
    )
    uv = shlex.quote(f"uv=={UV_VERSION}")
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "# written by model/train/runpod_launch.py",
            "set -uo pipefail",
            f"cd {shlex.quote(REMOTE_DIR)}",
            "trap 'echo $? > run.exit' EXIT",
            "set -e",
            f"export PYTORCH_CUDA_ALLOC_CONF={shlex.quote(settings.cuda_alloc_conf)}",
            'export PATH="$HOME/.local/bin:$PATH"',
            f"python3 -m pip install --quiet {uv} "
            f"|| python3 -m pip install --quiet --break-system-packages {uv}",
            "uv sync --frozen --no-dev --extra model",
            f"uv run --no-sync python -c {shlex.quote(cuda_check)}",
            f"timeout --signal=INT --kill-after=300 {seconds} "
            f"uv run --no-sync python -m model.train.train {shlex.join(args)}",
            "",
        ]
    )


# --------------------------------------------------------------------------- ssh


@dataclass
class Remote:
    host: str
    port: int
    key: Path
    known_hosts: Path = KNOWN_HOSTS
    user: str = "root"
    runner: Runner = run_command

    def _options(self) -> list[str]:
        return [
            "-i", str(self.key),
            "-o", "IdentitiesOnly=yes",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={self.known_hosts}",
            "-o", "ConnectTimeout=20",
            "-o", "ServerAliveInterval=30",
        ]  # fmt: skip

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"

    def ssh_argv(self, command: str) -> list[str]:
        return ["ssh", *self._options(), "-p", str(self.port), self.target, command]

    def upload_argv(self, local: Sequence[Path], remote_dir: str) -> list[str]:
        files = [str(p) for p in local]
        return [
            "scp",
            *self._options(),
            "-P",
            str(self.port),
            *files,
            f"{self.target}:{remote_dir}/",
        ]

    def download_argv(self, remote_path: str, local_dir: Path) -> list[str]:
        return [
            "scp", "-r", *self._options(), "-P", str(self.port),
            f"{self.target}:{remote_path}", str(local_dir),
        ]  # fmt: skip

    def _call(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        self.known_hosts.parent.mkdir(parents=True, exist_ok=True)
        return self.runner(argv, **kwargs)

    def ssh(self, command: str, input: bytes | None = None, capture: bool = False):
        return self._call(self.ssh_argv(command), input=input, capture=capture)

    def upload(self, local: Sequence[Path], remote_dir: str):
        return self._call(self.upload_argv(local, remote_dir))

    def download(self, remote_path: str, local_dir: Path):
        return self._call(self.download_argv(remote_path, local_dir))


# --------------------------------------------------------------------------- state


def save_state(state: Mapping[str, Any], path: Path = STATE_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(state), indent=2))


def load_state(path: Path = STATE_FILE) -> dict[str, Any]:
    if not path.is_file():
        raise LaunchError(f"No pod recorded in {path}. Run `up --yes` first.")
    return json.loads(path.read_text())


def remote_from_state(
    state: Mapping[str, Any], settings: LaunchSettings, runner: Runner = run_command
) -> Remote:
    if not state.get("host") or not state.get("port"):
        raise LaunchError("The recorded pod has no SSH endpoint. Run `status` or `down`.")
    return Remote(state["host"], int(state["port"]), settings.ssh_key, runner=runner)


# --------------------------------------------------------------------------- steps


def check_cost(pod: Mapping[str, Any], max_hourly_usd: float) -> float | None:
    """Return the pod's hourly rate; raise if it is above the limit."""
    cost = pod.get("cost")
    if isinstance(cost, int | float) and cost > max_hourly_usd:
        raise LaunchError(
            f"pod costs ${cost:.3f}/h, above RUNPOD_MAX_HOURLY_USD=${max_hourly_usd:.2f}"
        )
    return float(cost) if isinstance(cost, int | float) else None


def wait_for_ssh(
    sdk: Any,
    pod_id: str,
    timeout_s: float = 900,
    interval_s: float = 10,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[dict[str, Any], tuple[str, int]]:
    """Poll until the pod is RUNNING with a public port 22."""
    deadline = clock() + timeout_s
    last = None
    while True:
        pod = sdk.get_pod(pod_id)
        if pod is None:
            raise LaunchError(f"pod {pod_id} no longer exists")
        status = pod.get("status")
        if status != last:
            say(f"pod {pod_id}: {status}")
            last = status
        if status in FAILED_STATUSES:
            raise LaunchError(f"pod {pod_id} is {status}")
        endpoint = ssh_endpoint(pod)
        if status == "RUNNING" and endpoint:
            return pod, endpoint
        if clock() >= deadline:
            raise LaunchError(f"pod {pod_id} did not expose SSH within {timeout_s:.0f}s")
        sleep(interval_s)


def wait_for_sshd(remote: Remote, attempts: int = 12, sleep: Callable[[float], None] = time.sleep):
    """The port is mapped before sshd accepts connections: retry a no-op for a while."""
    for i in range(attempts):
        try:
            remote.ssh("true")
            return
        except subprocess.CalledProcessError:
            if i == attempts - 1:
                raise LaunchError("SSH did not accept the key. Check RUNPOD_SSH_KEY.") from None
            sleep(10)


def up(
    settings: LaunchSettings,
    sdk: Any,
    yes: bool,
    state_path: Path = STATE_FILE,
    runner: Runner = run_command,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any] | None:
    """Create the pod and wait for SSH. Without `yes`, print the request and create nothing."""
    try:
        public_key = read_public_key(settings, create=yes, runner=runner)
    except LaunchError as exc:
        if yes:
            raise
        say(f"note: {exc}")
        public_key = "<SSH public key>"
    request = pod_request(settings, public_key)
    say("create_pod request:\n" + display_request(request))
    if not yes:
        say("dry run: no pod created. Pass --yes to create it (this starts billing).")
        return None
    require_api_key(settings)
    if state_path.is_file():
        raise LaunchError(f"A pod is already recorded in {state_path}. Run `down` first.")
    pod = sdk.create_pod(**request)
    pod_id = pod["id"]
    state: dict[str, Any] = {"pod_id": pod_id, "created": datetime.now(UTC).isoformat()}
    save_state(state, state_path)  # recorded before anything else can fail, so `down` works
    say(f"created pod {pod_id}")
    try:
        state["cost_per_hour"] = check_cost(pod, settings.max_hourly_usd)
        pod, (host, port) = wait_for_ssh(sdk, pod_id, sleep=sleep)
        state["cost_per_hour"] = check_cost(pod, settings.max_hourly_usd)
        state.update(host=host, port=port)
        save_state(state, state_path)
        wait_for_sshd(Remote(host, port, settings.ssh_key, runner=runner), sleep=sleep)
    except BaseException:
        say(f"setup failed: terminating pod {pod_id}")
        down(sdk, state_path)
        raise
    rate = state["cost_per_hour"]
    say(f"pod {pod_id} ready at {host}:{port}" + (f", ${rate:.3f}/h" if rate else ""))
    return state


def push(remote: Remote, data_dir: Path = DATA_DIR, runner: Runner = run_command) -> None:
    """Upload committed code (git archive HEAD) and the enhanced data."""
    dirty = runner(["git", "status", "--porcelain"], capture=True).stdout
    if dirty.strip():
        say("warning: uncommitted changes are NOT uploaded; only HEAD is (git archive).")
    archive = runner(["git", "archive", "--format=tar", "HEAD"], capture=True).stdout
    remote.ssh(f"mkdir -p {REMOTE_DIR} && tar -x -C {REMOTE_DIR}", input=archive)
    say(f"code: uploaded HEAD ({len(archive) / 1e6:.1f} MB tar)")
    files = [data_dir / name for name in DATA_FILES if (data_dir / name).is_file()]
    if not any(f.name == "train.jsonl" for f in files):
        raise LaunchError(f"{data_dir / 'train.jsonl'} not found. Run the enhancement first.")
    remote_data = f"{REMOTE_DIR}/{data_dir.as_posix()}"
    remote.ssh(f"mkdir -p {shlex.quote(remote_data)}")
    remote.upload(files, remote_data)
    say("data: uploaded " + ", ".join(f.name for f in files))


def start_training(
    remote: Remote, settings: LaunchSettings, run_name: str, extra_args: Sequence[str] = ()
) -> None:
    script = remote_train_script(settings, run_name, extra_args)
    remote.ssh(f"cat > {REMOTE_DIR}/run_train.sh", input=script.encode())
    remote.ssh(
        f"cd {REMOTE_DIR} && rm -f run.exit && nohup bash run_train.sh > run.out 2>&1 < /dev/null &"
    )
    say(f"training started on the pod: run {run_name} (stops after {settings.max_hours:g} h)")


def remote_exit_code(remote: Remote) -> int | None:
    out = remote.ssh(f"cat {REMOTE_DIR}/run.exit 2>/dev/null || true", capture=True).stdout
    text = out.decode().strip() if isinstance(out, bytes) else str(out or "").strip()
    return int(text) if text.lstrip("-").isdigit() else None


def tail(remote: Remote, lines: int = 40) -> str:
    out = remote.ssh(f"tail -n {int(lines)} {REMOTE_DIR}/run.out 2>/dev/null || true", capture=True)
    return out.stdout.decode(errors="replace") if isinstance(out.stdout, bytes) else ""


def wait_for_exit(
    remote: Remote,
    timeout_s: float,
    interval_s: float = 60,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    max_failures: int = 10,
) -> int:
    """Poll run.exit. Up to `max_failures` SSH errors in a row (a network blip) are tolerated."""
    deadline = clock() + timeout_s
    failures = 0
    while True:
        try:
            code = remote_exit_code(remote)
            if code is not None:
                return code
            last = tail(remote, 1).strip()
            failures = 0
        except subprocess.CalledProcessError:
            failures += 1
            if failures >= max_failures:
                raise
            say(f"  ssh poll failed ({failures}/{max_failures}), retrying")
            last = ""
        if clock() >= deadline:
            raise LaunchError(f"training did not finish within {timeout_s / 3600:.1f} h")
        if last:
            say(f"  {last[-200:]}")
        sleep(interval_s)


def pull(remote: Remote, run_name: str, dest: Path = CHECKPOINTS_DIR) -> Path:
    """Copy checkpoints/<run>/ and the pod's run.out back. A missing run dir is reported."""
    dest.mkdir(parents=True, exist_ok=True)
    local = dest / run_name
    try:
        remote.download(f"{REMOTE_DIR}/{CHECKPOINTS_DIR.as_posix()}/{run_name}", dest)
    except subprocess.CalledProcessError:
        say(f"warning: no checkpoints/{run_name} on the pod (the run failed before writing it)")
    local.mkdir(parents=True, exist_ok=True)
    try:
        remote.download(f"{REMOTE_DIR}/run.out", local / "pod_run.out")
    except subprocess.CalledProcessError:
        say("warning: could not copy run.out")
    say(f"pulled to {local}")
    return local


def down(sdk: Any, state_path: Path = STATE_FILE) -> None:
    state = load_state(state_path)
    sdk.terminate_pod(state["pod_id"])
    state_path.unlink(missing_ok=True)
    say(f"terminated pod {state['pod_id']}")


def run_all(
    settings: LaunchSettings,
    sdk: Any,
    yes: bool,
    run_name: str,
    extra_args: Sequence[str] = (),
    state_path: Path = STATE_FILE,
    runner: Runner = run_command,
    sleep: Callable[[float], None] = time.sleep,
    dest: Path = CHECKPOINTS_DIR,
) -> int:
    """up -> push -> train -> wait -> pull, and always terminate the pod at the end."""
    state = up(settings, sdk, yes, state_path, runner, sleep)
    if state is None:  # dry run
        say("remote script:\n" + remote_train_script(settings, run_name, extra_args))
        return 0
    remote = remote_from_state(state, settings, runner)
    started = False
    try:
        push(remote, runner=runner)
        start_training(remote, settings, run_name, extra_args)
        started = True
        code = wait_for_exit(remote, settings.max_hours * 3600 + SETUP_ALLOWANCE_S, sleep=sleep)
        say(f"training exited with code {code}")
        return code
    finally:
        try:
            if started:
                pull(remote, run_name, dest)
        finally:
            down(sdk, state_path)


# --------------------------------------------------------------------------- CLI


def parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    """Arguments after `--` are passed to train.py unchanged."""
    argv = list(argv)
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1 :]
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("up", "run"):
        s = sub.add_parser(name)
        s.add_argument("--yes", action="store_true", help="really create the pod (billing)")
        if name == "run":
            s.add_argument("--run-name")
    sub.add_parser("push")
    s = sub.add_parser("train")
    s.add_argument("--run-name")
    sub.add_parser("status")
    s = sub.add_parser("logs")
    s.add_argument("-n", "--lines", type=int, default=40)
    s = sub.add_parser("pull")
    s.add_argument("--run-name", help="default: the last run started with `train`")
    sub.add_parser("down")
    return p.parse_args(argv), extra


def load_sdk(settings: LaunchSettings) -> Any:
    try:
        import runpod
    except ImportError:
        raise LaunchError("runpod is not installed: uv sync --extra model --group cloud") from None
    runpod.api_key = settings.api_key
    return runpod


def main(argv: Sequence[str] | None = None) -> int:
    args, extra = parse_args(sys.argv[1:] if argv is None else argv)
    load_dotenv()
    try:
        settings = settings_from_env(os.environ)
        return _dispatch(args, extra, settings)
    except LaunchError as exc:
        say(f"error: {exc}")
        return 2
    except subprocess.CalledProcessError as exc:
        say(f"error: {shlex.join(map(str, exc.cmd[:1]))} exited with {exc.returncode}")
        return 2


def _dispatch(args: argparse.Namespace, extra: list[str], settings: LaunchSettings) -> int:
    cmd = args.command
    if cmd == "up":
        up(settings, load_sdk(settings) if args.yes else None, args.yes)
        return 0
    if cmd == "run":
        sdk = load_sdk(settings) if args.yes else None
        return run_all(settings, sdk, args.yes, args.run_name or default_run_name(), extra)
    if cmd == "down":
        require_api_key(settings)
        down(load_sdk(settings))
        return 0
    state = load_state()
    if cmd == "status":
        require_api_key(settings)
        pod = load_sdk(settings).get_pod(state["pod_id"])
        status = pod.get("status") if pod else "not found"
        say(f"pod {state['pod_id']}: {status}, cost {pod.get('cost') if pod else '?'} $/h")
        if pod and status == "RUNNING" and state.get("run_name"):
            code = remote_exit_code(remote_from_state(state, settings))
            say(f"run {state['run_name']}: " + ("running" if code is None else f"exited {code}"))
        return 0
    remote = remote_from_state(state, settings)
    if cmd == "push":
        push(remote)
    elif cmd == "train":
        state["run_name"] = args.run_name or default_run_name()
        start_training(remote, settings, state["run_name"], extra)
        save_state(state)
    elif cmd == "logs":
        print(tail(remote, args.lines), end="")
    elif cmd == "pull":
        run_name = args.run_name or state.get("run_name")
        if not run_name:
            raise LaunchError("No run recorded. Pass --run-name.")
        pull(remote, run_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
