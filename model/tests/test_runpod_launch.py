"""Offline tests for the RunPod launcher: no network, no ssh, a fake SDK and a fake runner."""

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
    def __init__(self, created=None, polled=None):
        self.created = created or {"id": "pod1", "status": "PROVISIONING", "cost": 0.69}
        self.polled = list(polled or [RUNNING_POD])
        self.requests = []
        self.terminated = []

    def create_pod(self, **kwargs):
        self.requests.append(kwargs)
        return self.created

    def get_pod(self, pod_id):
        return self.polled.pop(0) if len(self.polled) > 1 else self.polled[0]

    def terminate_pod(self, pod_id):
        self.terminated.append(pod_id)


class FakeRunner:
    """Records commands. `fail_on` makes a command containing that text raise."""

    def __init__(self, fail_on=None, exit_code=b"0\n", dirty=b""):
        self.calls = []
        self.fail_on = fail_on
        self.exit_code = exit_code
        self.dirty = dirty

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


# ---- settings


def test_settings_defaults_when_env_is_empty():
    s = rl.settings_from_env({})
    assert s.gpu_type == "NVIDIA GeForce RTX 4090"
    assert s.cloud_type == "SECURE"
    assert s.api_key == ""
    assert s.allowed_cuda == rl.DEFAULT_ALLOWED_CUDA
    assert s.gradient_checkpointing is False


def test_settings_parse_values():
    s = rl.settings_from_env(
        {
            "RUNPOD_API_KEY": "k",
            "RUNPOD_GPU_TYPE": "NVIDIA RTX A4000",
            "RUNPOD_CLOUD_TYPE": "community",
            "RUNPOD_CONTAINER_DISK_GB": "40",
            "RUNPOD_MIN_RAM_GB": "8",
            "RUNPOD_ALLOWED_CUDA": "",
            "RUNPOD_MAX_HOURS": "1.5",
            "RUNPOD_SSH_KEY": "~/k",
            "TRAIN_BATCH_SIZE": "4",
            "TRAIN_GRAD_ACCUM": "4",
            "TRAIN_GRADIENT_CHECKPOINTING": "1",
        }
    )
    assert s.cloud_type == "COMMUNITY"
    assert s.container_disk_gb == 40 and s.min_ram_gb == 8
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
        {"RUNPOD_CLOUD_TYPE": "CHEAP"},
        {"RUNPOD_IMAGE": "runpod/pytorch:latest"},
        {"RUNPOD_IMAGE": "runpod/pytorch"},
        {"RUNPOD_MAX_HOURLY_USD": "-1"},
        {"TRAIN_GRADIENT_CHECKPOINTING": "maybe"},
    ],
)
def test_bad_settings_fail_clearly(env):
    with pytest.raises(rl.LaunchError):
        rl.settings_from_env(env)


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


def test_missing_api_key_is_an_error():
    with pytest.raises(rl.LaunchError, match="RUNPOD_API_KEY"):
        rl.require_api_key(rl.LaunchSettings())


# ---- pod request


def test_pod_request_sizes_the_pod_and_keeps_the_api_key_local(tmp_path):
    s = _settings(tmp_path, data_center="EU-RO-1")
    req = rl.pod_request(s, "ssh-ed25519 AAAA")
    assert req["gpu_type_id"] == "NVIDIA GeForce RTX 4090"
    assert req["gpu_count"] == 1
    assert req["container_disk_in_gb"] == 30
    assert req["volume_in_gb"] == 0
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
    assert set(rl.pod_request(_settings(tmp_path, data_center="X"), "k")) <= set(params)


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


# ---- steps with a fake SDK and runner


def test_up_without_yes_creates_nothing(tmp_path):
    sdk = FakeSdk()
    assert rl.up(_settings(tmp_path), sdk, yes=False, state_path=tmp_path / "s.json") is None
    assert sdk.requests == [] and not (tmp_path / "s.json").exists()


def test_up_creates_waits_and_records_the_pod(tmp_path):
    sdk, runner, state_path = FakeSdk(), FakeRunner(), tmp_path / "s.json"
    state = rl.up(_settings(tmp_path), sdk, True, state_path, runner, sleep=lambda s: None)
    assert state["pod_id"] == "pod1" and (state["host"], state["port"]) == ("1.2.3.4", 40022)
    assert rl.load_state(state_path)["pod_id"] == "pod1"
    assert sdk.terminated == []


def test_up_terminates_a_pod_above_the_hourly_limit(tmp_path):
    sdk, state_path = FakeSdk(created={"id": "pod1", "cost": 3.5}), tmp_path / "s.json"
    s = _settings(tmp_path, max_hourly_usd=1.0)
    with pytest.raises(rl.LaunchError, match="RUNPOD_MAX_HOURLY_USD"):
        rl.up(s, sdk, True, state_path, FakeRunner(), sleep=lambda s: None)
    assert sdk.terminated == ["pod1"] and not state_path.exists()


def test_up_terminates_a_pod_that_fails_to_start(tmp_path):
    sdk = FakeSdk(polled=[{"id": "pod1", "status": "ERROR"}])
    with pytest.raises(rl.LaunchError, match="ERROR"):
        rl.up(_settings(tmp_path), sdk, True, tmp_path / "s.json", FakeRunner(), lambda s: None)
    assert sdk.terminated == ["pod1"]


def test_up_refuses_a_second_pod(tmp_path):
    state_path = tmp_path / "s.json"
    rl.save_state({"pod_id": "old"}, state_path)
    with pytest.raises(rl.LaunchError, match="down"):
        rl.up(_settings(tmp_path), FakeSdk(), True, state_path, FakeRunner())


def test_push_sends_head_only_and_the_data(tmp_path, capsys):
    runner = FakeRunner(dirty=b" M model/train/train.py\n")
    remote = rl.Remote("1.2.3.4", 40022, tmp_path / "k", tmp_path / "kh", runner=runner)
    rl.push(remote, _data(tmp_path), runner=runner)
    cmds = runner.commands()
    assert any(c.startswith("git archive --format=tar HEAD") for c in cmds)
    tar_call = next(c for c in runner.calls if "tar -x" in " ".join(c[0]))
    assert tar_call[1] == b"TAR"
    assert any(c.startswith("scp") and "train.jsonl" in c for c in cmds)
    assert "uncommitted changes are NOT uploaded" in capsys.readouterr().err


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


@pytest.fixture
def in_project(tmp_path, monkeypatch):
    """Run in an empty folder with only model/train/data/enhanced/train.jsonl."""
    monkeypatch.chdir(tmp_path)
    data = tmp_path / rl.DATA_DIR
    data.mkdir(parents=True)
    (data / "train.jsonl").write_text("{}\n")
    return tmp_path


def test_run_all_trains_pulls_and_terminates(tmp_path, in_project):
    sdk, runner = FakeSdk(), FakeRunner(exit_code=b"0\n")
    code = rl.run_all(
        _settings(tmp_path), sdk, True, "run-1", ["--epochs", "1"],
        state_path=tmp_path / "s.json", runner=runner, sleep=lambda s: None,
        dest=tmp_path / "ckpt",
    )  # fmt: skip
    assert code == 0
    assert sdk.terminated == ["pod1"]
    cmds = runner.commands()
    assert any("run_train.sh" in c and "nohup" in c for c in cmds)
    assert any(c.startswith("scp -r") and "checkpoints/run-1" in c for c in cmds)


@pytest.mark.parametrize("fail_on", ["git archive", "nohup"])
def test_run_all_terminates_the_pod_when_a_step_fails(tmp_path, in_project, fail_on):
    sdk, runner = FakeSdk(), FakeRunner(fail_on=fail_on)
    with pytest.raises(subprocess.CalledProcessError):
        rl.run_all(
            _settings(tmp_path), sdk, True, "run-1",
            state_path=tmp_path / "s.json", runner=runner, sleep=lambda s: None,
            dest=tmp_path / "ckpt",
        )  # fmt: skip
    assert sdk.terminated == ["pod1"]
    assert not (tmp_path / "s.json").exists()


def test_run_all_dry_run_prints_and_creates_nothing(tmp_path, capsys):
    sdk = FakeSdk()
    assert rl.run_all(_settings(tmp_path), sdk, False, "run-1", state_path=tmp_path / "s") == 0
    assert sdk.requests == []
    err = capsys.readouterr().err
    assert "dry run" in err and "remote script" in err


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


def test_parse_args_passes_everything_after_double_dash_to_train():
    args, extra = rl.parse_args(["run", "--yes", "--", "--epochs", "1", "--yes"])
    assert args.command == "run" and args.yes is True
    assert extra == ["--epochs", "1", "--yes"]
