"""Offline tests for the RunPod launcher: no network, no ssh, a fake SDK and a fake runner."""

import hashlib
import shlex
import subprocess
from pathlib import Path

import pytest

from model.train import runpod_launch as rl
from model.train import train as tr


def _settings(tmp_path: Path, **overrides) -> rl.LaunchSettings:
    key = tmp_path / "id_test"
    key.write_text("private")
    key.with_name("id_test.pub").write_text("ssh-ed25519 AAAAPUBLIC test@host\n")
    values = {"api_key": "rpa_SECRET", "ssh_key": key, **overrides}
    return rl.LaunchSettings(**values)


RUNNING_POD = {
    "id": "pod1",
    "status": "RUNNING",
    "cost": 0.69,
    "runtime": {"ports": [{"ip": "1.2.3.4", "public": 40022, "private": 22, "type": "tcp"}]},
}


class FakeSdk:
    """`existing`: what get_pod returns before any polling (for reused pods)."""

    def __init__(self, created=None, polled=None, existing=None):
        self.created = created or {"id": "pod1", "status": "PROVISIONING", "cost": 0.69}
        self.existing = existing
        self.polled = list(polled or [RUNNING_POD])
        self.requests, self.terminated, self.stopped, self.resumed = [], [], [], []

    def create_pod(self, **kwargs):
        self.requests.append(kwargs)
        return self.created

    def get_pod(self, pod_id):
        if self.existing is not None:
            pod, self.existing = self.existing, None
            return pod or None
        return self.polled.pop(0) if len(self.polled) > 1 else self.polled[0]

    def stop_pod(self, pod_id):
        self.stopped.append(pod_id)

    def resume_pod(self, pod_id):
        self.resumed.append(pod_id)

    def terminate_pod(self, pod_id):
        self.terminated.append(pod_id)


class FakeRunner:
    """Records commands. `fail_on` makes a command containing that text raise."""

    def __init__(self, fail_on=None, exit_code=b"0\n", dirty=b"", hashes=b"", busy=False):
        self.calls = []
        self.fail_on = fail_on
        self.exit_code = exit_code
        self.dirty = dirty
        self.hashes = hashes
        self.busy = busy

    def __call__(self, argv, input=None, capture=False):
        self.calls.append((list(argv), input))
        joined = " ".join(map(str, argv))
        if self.fail_on and self.fail_on in joined:
            raise subprocess.CalledProcessError(1, argv)
        out = b""
        if argv[:2] == ["git", "status"]:
            out = self.dirty
        elif argv[:2] == ["git", "archive"]:
            out = b"TAR"
        elif joined.startswith("ssh") and "pgrep" in joined:
            out = b"busy\n" if self.busy else b""
        elif joined.startswith("ssh") and "sha256sum" in joined:
            out = self.hashes
        elif "run.exit" in joined and joined.startswith("ssh") and "cat " in joined:
            out = self.exit_code
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr=b"")

    def commands(self):
        return [" ".join(map(str, argv)) for argv, _ in self.calls]


def _data(tmp_path: Path) -> Path:
    d = tmp_path / "enhanced"
    d.mkdir()
    (d / "train.jsonl").write_text("{}\n")
    return d


def _up(tmp_path, sdk, runner=None, **overrides):
    return rl.up(
        _settings(tmp_path, **overrides), sdk, True, tmp_path / "s.json",
        runner or FakeRunner(), sleep=lambda s: None,
    )  # fmt: skip


# ---- settings


def test_settings_defaults_when_env_is_empty():
    s = rl.settings_from_env({})
    assert s.gpu_type == "NVIDIA GeForce RTX 4090"
    assert s.cloud_type == "SECURE"
    assert s.api_key == "" and s.pod_id is None
    assert s.allowed_cuda == rl.DEFAULT_ALLOWED_CUDA
    assert (s.container_disk_gb, s.volume_gb) == (20, 30)
    assert s.gradient_checkpointing is False


def test_settings_parse_values():
    s = rl.settings_from_env(
        {
            "RUNPOD_API_KEY": "k",
            "RUNPOD_POD_ID": " abc123 ",
            "RUNPOD_GPU_TYPE": "NVIDIA RTX A4000",
            "RUNPOD_CLOUD_TYPE": "community",
            "RUNPOD_CONTAINER_DISK_GB": "40",
            "RUNPOD_VOLUME_GB": "50",
            "RUNPOD_MIN_RAM_GB": "8",
            "RUNPOD_ALLOWED_CUDA": "",
            "RUNPOD_MAX_HOURS": "1.5",
            "RUNPOD_SSH_KEY": "~/k",
            "TRAIN_BATCH_SIZE": "4",
            "TRAIN_GRAD_ACCUM": "4",
            "TRAIN_GRADIENT_CHECKPOINTING": "1",
        }
    )
    assert s.pod_id == "abc123"
    assert s.cloud_type == "COMMUNITY"
    assert (s.container_disk_gb, s.volume_gb, s.min_ram_gb) == (40, 50, 8)
    assert s.allowed_cuda == ()  # empty means no CUDA filter
    assert s.max_hours == 1.5
    assert s.ssh_key == Path("~/k").expanduser()
    assert (s.batch_size, s.grad_accum, s.gradient_checkpointing) == (4, 4, True)


@pytest.mark.parametrize(
    "env",
    [
        {"RUNPOD_GPU_COUNT": "two"},
        {"RUNPOD_GPU_COUNT": "0"},
        {"RUNPOD_CONTAINER_DISK_GB": "5"},
        {"RUNPOD_VOLUME_GB": "5"},
        {"RUNPOD_CLOUD_TYPE": "CHEAP"},
        {"RUNPOD_IMAGE": "runpod/pytorch:latest"},
        {"RUNPOD_IMAGE": "runpod/pytorch"},
        {"RUNPOD_MAX_HOURLY_USD": "-1"},
        {"TRAIN_GRADIENT_CHECKPOINTING": "maybe"},
        {"RUNPOD_NETWORK_VOLUME_ID": "vol123"},  # needs RUNPOD_DATA_CENTER
    ],
)
def test_bad_settings_fail_clearly(env):
    with pytest.raises(rl.LaunchError):
        rl.settings_from_env(env)


def test_missing_api_key_is_an_error():
    with pytest.raises(rl.LaunchError, match="RUNPOD_API_KEY"):
        rl.require_api_key(rl.LaunchSettings())


# ---- ssh key


class KeygenRunner(FakeRunner):
    """Fakes ssh-keygen by writing the two files it would create."""

    def __call__(self, argv, input=None, capture=False):
        if argv[0] == "ssh-keygen":
            key = Path(argv[argv.index("-f") + 1])
            key.write_text("private")
            key.with_name(key.name + ".pub").write_text("ssh-ed25519 AAAAGEN havelai-runpod\n")
        return super().__call__(argv, input, capture)


def test_default_key_is_the_launchers_own():
    assert rl.settings_from_env({}).ssh_key == rl.AUTO_SSH_KEY
    assert rl.settings_from_env({"RUNPOD_SSH_KEY": ""}).ssh_key == rl.AUTO_SSH_KEY


def test_own_key_is_created_on_yes_only(tmp_path, monkeypatch):
    auto = tmp_path / "temp" / "runpod_ed25519"
    monkeypatch.setattr(rl, "AUTO_SSH_KEY", auto)
    s, runner = rl.LaunchSettings(ssh_key=auto), KeygenRunner()
    with pytest.raises(rl.LaunchError, match="first `--yes`"):
        rl.read_public_key(s, create=False, runner=runner)
    assert runner.calls == []
    assert rl.read_public_key(s, create=True, runner=runner) == "ssh-ed25519 AAAAGEN havelai-runpod"
    argv = runner.calls[0][0]
    assert argv[:3] == ["ssh-keygen", "-t", "ed25519"] and argv[argv.index("-N") + 1] == ""
    assert auto.stat().st_mode & 0o077 == 0, "private key must not be readable by others"
    rl.read_public_key(s, create=True, runner=runner)
    assert len(runner.calls) == 1, "an existing key is reused, not regenerated"


def test_a_key_you_set_is_never_generated(tmp_path):
    s, runner = rl.LaunchSettings(ssh_key=tmp_path / "mine"), KeygenRunner()
    with pytest.raises(rl.LaunchError, match="not found"):
        rl.read_public_key(s, create=True, runner=runner)
    assert runner.calls == []


def test_up_with_yes_creates_the_own_key_then_the_pod(tmp_path, monkeypatch):
    auto = tmp_path / "runpod_ed25519"
    monkeypatch.setattr(rl, "AUTO_SSH_KEY", auto)
    sdk, runner = FakeSdk(), KeygenRunner()
    s = rl.LaunchSettings(api_key="k", ssh_key=auto)
    rl.up(s, sdk, True, tmp_path / "s.json", runner, sleep=lambda s: None)
    assert sdk.requests[0]["env"]["PUBLIC_KEY"] == "ssh-ed25519 AAAAGEN havelai-runpod"
    assert any(f"-i {auto}" in c for c in runner.commands())


# ---- pod request


def test_pod_request_sizes_the_pod_and_keeps_the_api_key_local(tmp_path):
    s = _settings(tmp_path, data_center="EU-RO-1")
    req = rl.pod_request(s, "ssh-ed25519 AAAA")
    assert req["gpu_type_id"] == "NVIDIA GeForce RTX 4090"
    assert req["gpu_count"] == 1
    assert req["container_disk_in_gb"] == 20
    assert req["min_memory_in_gb"] == 16 and req["min_vcpu_count"] == 4
    assert req["ports"] == "22/tcp" and req["start_ssh"] is True
    assert req["data_center_id"] == "EU-RO-1"
    assert req["allowed_cuda_versions"] == list(rl.DEFAULT_ALLOWED_CUDA)
    assert req["env"]["PUBLIC_KEY"] == "ssh-ed25519 AAAA"
    assert "rpa_SECRET" not in repr(req)
    # options the SDK 2.x rejects are never set
    for name in ("support_public_ip", "country_code", "min_download", "min_upload"):
        assert name not in req


def test_pod_request_matches_the_sdk_signature(tmp_path):
    inspect = pytest.importorskip("inspect")
    ctl = pytest.importorskip("runpod.api.ctl_commands")
    params = inspect.signature(ctl.create_pod).parameters
    s = _settings(tmp_path, data_center="X", network_volume_id="v")
    assert set(rl.pod_request(s, "k")) <= set(params)


def test_workspace_is_a_pod_volume_by_default(tmp_path):
    req = rl.pod_request(_settings(tmp_path), "k")
    assert req["volume_in_gb"] == 30 and req["volume_mount_path"] == "/workspace"
    assert "network_volume_id" not in req


def test_workspace_is_the_network_volume_when_set():
    s = rl.settings_from_env(
        {"RUNPOD_NETWORK_VOLUME_ID": "vol123", "RUNPOD_DATA_CENTER": "EU-RO-1"}
    )
    req = rl.pod_request(s, "k")
    assert req["network_volume_id"] == "vol123" and req["data_center_id"] == "EU-RO-1"
    assert req["volume_in_gb"] == 0 and req["volume_mount_path"] == "/workspace"


def test_display_request_shortens_the_public_key():
    shown = rl.display_request({"env": {"PUBLIC_KEY": "ssh-ed25519 " + "A" * 200}})
    assert "A" * 100 not in shown and "chars" in shown


# ---- ssh endpoint


def test_ssh_endpoint_from_runtime_ports():
    assert rl.ssh_endpoint(RUNNING_POD) == ("1.2.3.4", 40022)


def test_ssh_endpoint_prefers_direct_ssh():
    pod = {**RUNNING_POD, "ssh": {"direct": {"host": "5.6.7.8", "port": "2222"}}}
    assert rl.ssh_endpoint(pod) == ("5.6.7.8", 2222)


@pytest.mark.parametrize(
    "pod",
    [
        None,
        {"status": "PROVISIONING", "runtime": None},
        {"runtime": {"ports": [{"ip": "1.2.3.4", "public": 8888, "private": 8888}]}},
        {"runtime": {"ports": [{"ip": None, "public": 1, "private": 22}]}},
    ],
)
def test_ssh_endpoint_is_none_until_port_22_is_mapped(pod):
    assert rl.ssh_endpoint(pod) is None


# ---- training command


def test_train_flags():
    s = rl.LaunchSettings(batch_size=4, eval_batch_size=8, grad_accum=4, max_length=256)
    assert rl.train_flags(s) == [
        "--batch-size", "4", "--eval-batch-size", "8", "--grad-accum", "4", "--max-length", "256",
    ]  # fmt: skip
    assert rl.train_flags(rl.LaunchSettings(gradient_checkpointing=True))[-1] == (
        "--gradient-checkpointing"
    )


def test_train_flags_are_accepted_by_train_py():
    s = rl.LaunchSettings(batch_size=4, grad_accum=2, max_length=128, gradient_checkpointing=True)
    cfg = tr.config_from_args(tr.parse_args(rl.train_flags(s)))
    assert (cfg.batch_size, cfg.grad_accum, cfg.max_length) == (4, 2, 128)
    assert cfg.gradient_checkpointing is True
    assert tr.config_from_args(tr.parse_args([])).gradient_checkpointing is False


def test_remote_script_quotes_arguments_and_bounds_time():
    s = rl.LaunchSettings(max_hours=2)
    script = rl.remote_train_script(s, "run-1", ["--limit-train", "2000", "--x; rm -rf /"])
    assert "set -e" in script and "trap 'echo $? > run.exit' EXIT" in script
    assert "uv sync --frozen --no-dev --extra model" in script
    assert "timeout --signal=INT --kill-after=300 7200 " in script
    assert shlex.quote("--x; rm -rf /") in script
    assert "--run-name run-1" in script and "--no-progress" in script
    assert "torch.cuda.is_available()" in script
    assert "RUNPOD_API_KEY" not in script


def test_remote_script_keeps_everything_under_workspace():
    script = rl.remote_train_script(rl.LaunchSettings(), "r")
    for line in (
        f"export UV_CACHE_DIR={rl.CACHE_DIR}/uv",
        f"export PIP_CACHE_DIR={rl.CACHE_DIR}/pip",
        f"export HF_HOME={rl.CACHE_DIR}/huggingface",
        f"export UV_PROJECT_ENVIRONMENT={rl.REMOTE_VENV}",
    ):
        assert line in script
    assert f"--data-dir {rl.REMOTE_DATA} --out-dir {rl.REMOTE_RUNS}" in script
    for path in (rl.REMOTE_DIR, rl.REMOTE_VENV, rl.REMOTE_DATA, rl.REMOTE_RUNS, rl.CACHE_DIR):
        assert path.startswith("/workspace/"), "must survive a stop"
    # caches and venv are set before anything downloads
    assert script.index("UV_CACHE_DIR") < script.index("pip install") < script.index("uv sync")


def test_remote_script_args_are_accepted_by_train_py():
    """The data/out dirs and flags the launcher adds parse with train.py's own parser."""
    s = rl.LaunchSettings()
    args = shlex.split(rl.remote_train_script(s, "r", ["--epochs", "1"]).splitlines()[-1])
    cfg_args = tr.parse_args(args[args.index("model.train.train") + 1 :])
    assert str(cfg_args.data_dir) == rl.REMOTE_DATA and str(cfg_args.out_dir) == rl.REMOTE_RUNS
    assert cfg_args.epochs == 1


# ---- up: create, reuse, resume


def test_up_without_yes_creates_nothing(tmp_path):
    sdk = FakeSdk()
    assert rl.up(_settings(tmp_path), sdk, yes=False, state_path=tmp_path / "s.json") is None
    assert sdk.requests == [] and sdk.resumed == [] and not (tmp_path / "s.json").exists()


def test_up_creates_a_pod_and_records_it(tmp_path, capsys):
    sdk = FakeSdk()
    state = _up(tmp_path, sdk)
    assert state["pod_id"] == "pod1" and (state["host"], state["port"]) == ("1.2.3.4", 40022)
    assert rl.load_state(tmp_path / "s.json")["pod_id"] == "pod1"
    assert sdk.terminated == [] and sdk.stopped == []
    assert "RUNPOD_POD_ID=pod1" in capsys.readouterr().err


def test_up_reuses_the_recorded_pod_instead_of_creating(tmp_path):
    rl.save_state({"pod_id": "pod1"}, tmp_path / "s.json")
    sdk = FakeSdk(existing=RUNNING_POD)
    _up(tmp_path, sdk)
    assert sdk.requests == [] and sdk.resumed == []


def test_up_resumes_a_stopped_pod(tmp_path):
    sdk = FakeSdk(existing={"id": "pod1", "status": "EXITED"}, polled=[
        {"id": "pod1", "status": "EXITED"},  # still reported as stopped right after resume
        RUNNING_POD,
    ])  # fmt: skip
    state = _up(tmp_path, sdk, pod_id="pod1")
    assert sdk.resumed == ["pod1"] and sdk.requests == []
    assert state["host"] == "1.2.3.4"


def test_runpod_pod_id_wins_over_the_recorded_pod(tmp_path):
    rl.save_state({"pod_id": "old", "host": "9.9.9.9", "port": 1}, tmp_path / "s.json")
    sdk = FakeSdk(existing=RUNNING_POD)
    state = _up(tmp_path, sdk, pod_id="pod1")
    assert state["pod_id"] == "pod1" and state["host"] == "1.2.3.4"


@pytest.mark.parametrize("existing", [{}, {"id": "pod1", "status": "TERMINATED"}])
def test_a_pod_id_that_no_longer_exists_fails_clearly(tmp_path, existing):
    sdk = FakeSdk(existing=existing)
    with pytest.raises(rl.LaunchError, match="RUNPOD_POD_ID"):
        _up(tmp_path, sdk, pod_id="gone")
    assert sdk.requests == [], "never silently create a second pod"


def test_up_terminates_a_new_pod_above_the_hourly_limit(tmp_path):
    sdk = FakeSdk(created={"id": "pod1", "cost": 3.5})
    with pytest.raises(rl.CostLimitError):
        _up(tmp_path, sdk, max_hourly_usd=1.0)
    assert sdk.terminated == ["pod1"] and not (tmp_path / "s.json").exists()


def test_up_stops_a_reused_pod_above_the_hourly_limit(tmp_path):
    sdk = FakeSdk(
        existing={"id": "pod1", "status": "EXITED"}, polled=[{**RUNNING_POD, "cost": 3.5}]
    )
    with pytest.raises(rl.CostLimitError):
        _up(tmp_path, sdk, pod_id="pod1", max_hourly_usd=1.0)
    assert sdk.stopped == ["pod1"] and sdk.terminated == []


def test_up_stops_a_pod_that_fails_to_start(tmp_path):
    sdk = FakeSdk(polled=[{"id": "pod1", "status": "ERROR"}])
    with pytest.raises(rl.LaunchError, match="ERROR"):
        _up(tmp_path, sdk)
    assert sdk.stopped == ["pod1"] and sdk.terminated == []
    assert rl.load_state(tmp_path / "s.json")["pod_id"] == "pod1", "kept for the next run"


# ---- push


def test_push_replaces_the_code_and_uploads_the_data(tmp_path, capsys):
    runner = FakeRunner(dirty=b" M model/train/train.py\n")
    remote = rl.Remote("1.2.3.4", 40022, tmp_path / "k", tmp_path / "kh", runner=runner)
    rl.push(remote, _data(tmp_path), runner=runner)
    cmds = runner.commands()
    assert any(c.startswith("git archive --format=tar HEAD") for c in cmds)
    tar_call = next(c for c in runner.calls if "tar -x" in " ".join(c[0]))
    assert tar_call[1] == b"TAR" and f"rm -rf {rl.REMOTE_DIR}" in " ".join(tar_call[0])
    assert any(c.startswith("scp") and "train.jsonl" in c and rl.REMOTE_DATA in c for c in cmds)
    assert "uncommitted changes are NOT uploaded" in capsys.readouterr().err


def test_push_skips_data_that_is_already_on_the_pod(tmp_path, capsys):
    data = _data(tmp_path)
    digest = hashlib.sha256((data / "train.jsonl").read_bytes()).hexdigest()
    runner = FakeRunner(hashes=f"{digest}  train.jsonl\n".encode())
    remote = rl.Remote("h", 1, tmp_path / "k", tmp_path / "kh", runner=runner)
    rl.push(remote, data, runner=runner)
    assert not any(c.startswith("scp") for c in runner.commands())
    assert "nothing to upload" in capsys.readouterr().err


def test_push_refuses_while_a_training_job_runs(tmp_path):
    runner = FakeRunner(busy=True)
    remote = rl.Remote("h", 1, tmp_path / "k", tmp_path / "kh", runner=runner)
    with pytest.raises(rl.LaunchError, match="still running"):
        rl.push(remote, _data(tmp_path), runner=runner)
    assert not any("rm -rf" in c for c in runner.commands())


def test_push_needs_train_jsonl(tmp_path):
    runner = FakeRunner()
    remote = rl.Remote("h", 1, tmp_path / "k", tmp_path / "kh", runner=runner)
    (tmp_path / "empty").mkdir()
    with pytest.raises(rl.LaunchError, match="train.jsonl"):
        rl.push(remote, tmp_path / "empty", runner=runner)


def test_ssh_options_use_the_key_and_a_private_known_hosts(tmp_path):
    remote = rl.Remote("1.2.3.4", 40022, Path("/k"), tmp_path / "kh", runner=FakeRunner())
    argv = remote.ssh_argv("true")
    assert argv[:3] == ["ssh", "-i", "/k"]
    assert f"UserKnownHostsFile={tmp_path / 'kh'}" in argv
    assert argv[-4:] == ["-p", "40022", "root@1.2.3.4", "true"]
    assert "-P" in remote.upload_argv([Path("a")], "/r")


# ---- stop / down


def test_stop_keeps_the_pod_recorded_but_forgets_its_address(tmp_path):
    state_path = tmp_path / "s.json"
    rl.save_state({"pod_id": "pod1", "host": "h", "port": 1}, state_path)
    sdk = FakeSdk()
    rl.stop(sdk, state_path)
    assert sdk.stopped == ["pod1"]
    assert rl.load_state(state_path) == {"pod_id": "pod1"}


def test_down_terminates_and_forgets(tmp_path):
    state_path = tmp_path / "s.json"
    rl.save_state({"pod_id": "pod1"}, state_path)
    sdk = FakeSdk()
    rl.down(sdk, state_path)
    assert sdk.terminated == ["pod1"] and not state_path.exists()


# ---- run_all


@pytest.fixture
def in_project(tmp_path, monkeypatch):
    """Run in an empty folder with only model/train/data/enhanced/train.jsonl."""
    monkeypatch.chdir(tmp_path)
    data = tmp_path / rl.DATA_DIR
    data.mkdir(parents=True)
    (data / "train.jsonl").write_text("{}\n")
    return tmp_path


def _run(tmp_path, sdk, runner, **kwargs):
    return rl.run_all(
        _settings(tmp_path), sdk, True, "run-1", kwargs.pop("extra", ()),
        state_path=tmp_path / "s.json", runner=runner, sleep=lambda s: None,
        dest=tmp_path / "ckpt", **kwargs,
    )  # fmt: skip


def test_run_all_trains_pulls_and_stops_the_pod(tmp_path, in_project):
    sdk, runner = FakeSdk(), FakeRunner(exit_code=b"0\n")
    assert _run(tmp_path, sdk, runner, extra=["--epochs", "1"]) == 0
    assert sdk.stopped == ["pod1"] and sdk.terminated == []
    assert rl.load_state(tmp_path / "s.json")["pod_id"] == "pod1", "kept for the next run"
    cmds = runner.commands()
    assert any("run_train.sh" in c and "nohup" in c for c in cmds)
    assert any(c.startswith("scp -r") and f"{rl.REMOTE_RUNS}/run-1" in c for c in cmds)


def test_run_all_can_terminate_instead(tmp_path, in_project):
    sdk = FakeSdk()
    _run(tmp_path, sdk, FakeRunner(), terminate=True)
    assert sdk.terminated == ["pod1"] and sdk.stopped == []
    assert not (tmp_path / "s.json").exists()


@pytest.mark.parametrize("fail_on", ["git archive", "nohup"])
def test_run_all_stops_the_pod_when_a_step_fails(tmp_path, in_project, fail_on):
    sdk = FakeSdk()
    with pytest.raises(subprocess.CalledProcessError):
        _run(tmp_path, sdk, FakeRunner(fail_on=fail_on))
    assert sdk.stopped == ["pod1"] and sdk.terminated == []


def test_run_all_dry_run_prints_and_does_nothing(tmp_path, capsys):
    sdk = FakeSdk()
    assert rl.run_all(_settings(tmp_path), sdk, False, "run-1", state_path=tmp_path / "s") == 0
    assert sdk.requests == [] and sdk.stopped == [] and sdk.resumed == []
    err = capsys.readouterr().err
    assert "dry run" in err and "remote script" in err and "stopped (kept)" in err


# ---- waiting


def test_wait_for_exit_times_out(tmp_path):
    runner = FakeRunner(exit_code=b"")
    remote = rl.Remote("h", 1, tmp_path / "k", tmp_path / "kh", runner=runner)
    ticks = iter(range(0, 10_000, 100))
    with pytest.raises(rl.LaunchError, match="did not finish"):
        rl.wait_for_exit(remote, 250, sleep=lambda s: None, clock=lambda: next(ticks))


class FlakyRunner(FakeRunner):
    """SSH fails `failures` times, then works."""

    def __init__(self, failures, **kwargs):
        super().__init__(**kwargs)
        self.failures = failures

    def __call__(self, argv, input=None, capture=False):
        if self.failures > 0:
            self.failures -= 1
            raise subprocess.CalledProcessError(255, argv)
        return super().__call__(argv, input, capture)


def test_wait_for_exit_tolerates_a_few_failed_polls(tmp_path):
    remote = rl.Remote("h", 1, tmp_path / "k", tmp_path / "kh", runner=FlakyRunner(3))
    assert rl.wait_for_exit(remote, 10_000, sleep=lambda s: None, max_failures=4) == 0


def test_wait_for_exit_gives_up_after_too_many_failed_polls(tmp_path):
    remote = rl.Remote("h", 1, tmp_path / "k", tmp_path / "kh", runner=FlakyRunner(5))
    with pytest.raises(subprocess.CalledProcessError):
        rl.wait_for_exit(remote, 10_000, sleep=lambda s: None, max_failures=4)


# ---- CLI


def test_parse_args_passes_everything_after_double_dash_to_train():
    args, extra = rl.parse_args(["run", "--yes", "--terminate", "--", "--epochs", "1", "--yes"])
    assert args.command == "run" and args.yes is True and args.terminate is True
    assert extra == ["--epochs", "1", "--yes"]
    assert rl.parse_args(["stop"])[0].command == "stop"
