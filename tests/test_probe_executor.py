import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from evalyn.adapters.base import ProbeArtifact
from evalyn.adapters.pytest import PytestAdapter
from evalyn.adapters.vitest import VitestAdapter
from evalyn.engine.models import Revision
from evalyn.execution import (
    CommandResult,
    ImageIdentity,
    ImageIdentityResolution,
    OciContainerRuntime,
)
from evalyn.execution.structured_results import (
    PYTEST_REPORTER_SOURCE,
    PYTEST_RESULT_PREFIX,
    parse_structured_result,
)
from evalyn.probe_executor import (
    ContainerProbeExecutor,
    _looks_invalid,
    _stable_failure_signature,
)


def artifact(source: str = "def test_one():\n    assert True\n") -> ProbeArtifact:
    return PytestAdapter().build_artifact(source)


def test_materialize_uses_adapter_owned_contained_path(tmp_path: Path) -> None:
    item = artifact()

    target = ContainerProbeExecutor._materialize(tmp_path, item)

    assert target.is_relative_to(tmp_path)
    assert target.read_text(encoding="utf-8") == item.source


def test_materialize_rejects_conflicting_content(tmp_path: Path) -> None:
    item = artifact()
    target = tmp_path / item.relative_path
    target.parent.mkdir(parents=True)
    target.write_text("different", encoding="utf-8")

    with pytest.raises(ValueError, match="different content"):
        ContainerProbeExecutor._materialize(tmp_path, item)


def test_materializes_content_checked_trusted_pytest_reporter(tmp_path: Path) -> None:
    target = ContainerProbeExecutor._materialize_pytest_reporter(tmp_path)

    assert target.read_text(encoding="utf-8") == PYTEST_REPORTER_SOURCE
    assert target.stat().st_mode & 0o222 == 0


def test_invalid_output_detection() -> None:
    assert _looks_invalid("pytest", "ERROR collecting test_file.py")
    assert _looks_invalid("vitest", "Failed to parse source")
    assert not _looks_invalid("pytest", "AssertionError: expected 1")


def test_default_pytest_setup_uses_workspace_venv(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\nversion='1'\n")
    executor = ContainerProbeExecutor(
        PytestAdapter(), engine="docker", run_dir=tmp_path / "artifacts"
    )

    commands = executor._default_setup(tmp_path)

    assert commands[0] == ("python", "-m", "venv", ".evalyn/venv")
    assert commands[-1][-2:] == ("-e", ".")


def test_setup_environment_requires_every_allowlisted_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NPM_TOKEN", raising=False)
    executor = ContainerProbeExecutor(
        PytestAdapter(),
        engine="docker",
        run_dir=tmp_path / "artifacts",
        setup_env_names=("NPM_TOKEN",),
    )

    with pytest.raises(RuntimeError, match="NPM_TOKEN"):
        executor._setup_secret_environment()


def test_setup_environment_reads_only_named_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NPM_TOKEN", "package-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "model-secret")
    executor = ContainerProbeExecutor(
        PytestAdapter(),
        engine="docker",
        run_dir=tmp_path / "artifacts",
        setup_env_names=("NPM_TOKEN",),
    )

    assert executor._setup_secret_environment() == {"NPM_TOKEN": "package-secret"}


def test_execution_budget_exhaustion_is_fail_closed(tmp_path: Path) -> None:
    executor = ContainerProbeExecutor(
        PytestAdapter(),
        engine="docker",
        run_dir=tmp_path / "artifacts",
        timeout_seconds=1,
    )
    executor._deadline = 0

    with pytest.raises(TimeoutError, match="budget"):
        executor.remaining_timeout()


def test_pytest_failure_signature_ignores_only_timing_summary() -> None:
    first = _stable_failure_signature(
        "pytest",
        "E assert 106 == 105\n===== 1 failed in 0.02s =====\n",
        "",
    )
    repeated = _stable_failure_signature(
        "pytest",
        "E assert 106 == 105\n1 failed in 0.19s\n",
        "",
    )
    different = _stable_failure_signature(
        "pytest",
        "E assert 107 == 105\n===== 1 failed in 0.02s =====\n",
        "",
    )

    assert first == repeated
    assert first != different


def test_vitest_failure_signature_ignores_timing_lines() -> None:
    first = _stable_failure_signature(
        "vitest",
        "expected 106 to be 105\n Start at 12:01:02\n Duration 80ms\n",
        "",
    )
    repeated = _stable_failure_signature(
        "vitest",
        "expected 106 to be 105\n Start at 12:02:03\n Duration 120ms\n",
        "",
    )

    assert first == repeated


def test_parses_pytest_machine_result_and_stable_failure_signature() -> None:
    payload = (
        PYTEST_RESULT_PREFIX + '{"collection_errors":[],"framework":"pytest","tests":['
        '{"detail":"assert 2 == 1","nodeid":"test_probe.py::test_value",'
        '"outcome":"failed"}],"version":1}'
    )

    result = parse_structured_result("pytest", f"human output\n{payload}\n")

    assert result is not None
    assert result.total == 1
    assert result.failed == 1
    assert len(result.failure_signature) == 64


def test_pytest_parser_rejects_duplicate_reporter_markers() -> None:
    payload = (
        PYTEST_RESULT_PREFIX + '{"collection_errors":[],"framework":"pytest","tests":[],'
        '"version":1}'
    )

    assert parse_structured_result("pytest", f"{payload}\n{payload}\n") is None


def test_parses_vitest_json_result() -> None:
    output = (
        '{"numTotalTests":1,"numPassedTests":0,"numFailedTests":1,'
        '"numPendingTests":0,"testResults":[{"assertionResults":['
        '{"status":"failed","fullName":"value remains one",'
        '"failureMessages":["expected 2 to be 1"]}]}]}'
    )

    result = parse_structured_result("vitest", output)

    assert result is not None
    assert result.failed == 1
    assert result.metadata()["result_protocol"] == "evalyn.test-result.v1"


def test_vitest_suite_import_failure_is_structured_framework_evidence() -> None:
    output = (
        '{"numTotalTests":0,"numPassedTests":0,"numFailedTests":0,'
        '"numPendingTests":0,"testResults":[{"status":"failed",'
        '"message":"Cannot import changed module","assertionResults":[]}]}'
    )

    result = parse_structured_result("vitest", output)

    assert result is not None
    assert result.collection_errors == 1
    assert len(result.failure_signature) == 64


def test_structured_parser_rejects_contradictory_aggregate_counts() -> None:
    output = (
        '{"numTotalTests":2,"numPassedTests":1,"numFailedTests":0,'
        '"numPendingTests":0,"testResults":[]}'
    )

    assert parse_structured_result("vitest", output) is None


@pytest.mark.parametrize(
    ("adapter", "output"),
    (("pytest", "not structured"), ("vitest", "{bad json")),
)
def test_rejects_missing_or_malformed_machine_results(adapter: str, output: str) -> None:
    assert parse_structured_result(adapter, output) is None


def test_verified_environment_identity_pins_probe_image_and_metadata(tmp_path: Path) -> None:
    executor = ContainerProbeExecutor(
        PytestAdapter(), engine="docker", run_dir=tmp_path / "artifacts"
    )
    resolution = ImageIdentityResolution.resolved(ImageIdentity("sha256:" + "a" * 64))
    fingerprint = OciContainerRuntime(
        "python:3.12-slim",
        engine="docker",
        network="none",
        pull_policy="missing",
        workspace_read_only=True,
    ).environment_fingerprint(resolution)

    executor._adopt_probe_fingerprint(fingerprint)

    assert executor.execution_image == "sha256:" + "a" * 64
    assert executor.environment_fingerprint is not None
    assert executor.environment_fingerprint["complete"] is True
    metadata = executor._environment_metadata()
    assert metadata["environment_integrity"] == "verified"
    assert metadata["environment_fingerprint_sha256"].startswith("sha256:")


def test_setup_runs_only_after_the_declared_tag_is_resolved_and_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolution = ImageIdentityResolution.resolved(ImageIdentity("sha256:" + "a" * 64))
    fingerprint = OciContainerRuntime(
        "python:3.12-slim",
        engine="docker",
        network="none",
        pull_policy="missing",
        workspace_read_only=True,
    ).environment_fingerprint(resolution)
    calls: list[tuple[str, str, tuple[str, ...]]] = []

    class Runner:
        def __init__(self, workspace, *, image, network, **_kwargs):
            self.workspace = Path(workspace)
            self.image = image
            self.network = network

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def run(self, argv, **_kwargs):
            calls.append((self.image, self.network, tuple(argv)))
            stdout = ""
            if self.network == "none":
                attestation = next(
                    self.workspace.glob(".evalyn/probes/support/workspace-attestation-*")
                )
                stdout = hashlib.sha256(attestation.read_bytes()).hexdigest() + "\n"
            return SimpleNamespace(
                succeeded=True,
                error=None,
                stderr="",
                stdout=stdout,
                output_truncated=False,
            )

        def resolve_environment_fingerprint(self):
            return fingerprint

    monkeypatch.setattr("evalyn.probe_executor.ContainerRunner", Runner)
    executor = ContainerProbeExecutor(
        PytestAdapter(),
        engine="docker",
        run_dir=tmp_path / "artifacts",
        setup_argv=("python", "-m", "pip", "--version"),
    )

    executor._prepare_workspace(tmp_path)

    assert len(calls) == 2
    assert calls[0][0:2] == ("python:3.12-slim", "none")
    assert calls[0][2][0:2] == ("python", "-c")
    assert calls[1] == (
        "sha256:" + "a" * 64,
        "bridge",
        ("python", "-m", "pip", "--version"),
    )


def test_saved_environment_identity_and_endpoint_scope_must_match(tmp_path: Path) -> None:
    resolution = ImageIdentityResolution.resolved(ImageIdentity("sha256:" + "a" * 64))
    fingerprint = OciContainerRuntime(
        "python:3.12-slim", engine="docker", network="none"
    ).environment_fingerprint(resolution)
    wrong_image = ContainerProbeExecutor(
        PytestAdapter(),
        engine="docker",
        run_dir=tmp_path / "image",
        expected_image_identity="sha256:" + "b" * 64,
    )
    wrong_endpoint = ContainerProbeExecutor(
        PytestAdapter(),
        engine="docker",
        run_dir=tmp_path / "endpoint",
        expected_endpoint_scope="remote",
    )

    with pytest.raises(RuntimeError, match="image does not match"):
        wrong_image._adopt_probe_fingerprint(fingerprint)
    with pytest.raises(RuntimeError, match="endpoint scope"):
        wrong_endpoint._adopt_probe_fingerprint(fingerprint)


def test_workspace_attestation_fails_closed_and_removes_random_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Runner:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def run(self, *_args, **_kwargs):
            return SimpleNamespace(
                succeeded=True,
                error=None,
                stderr="",
                stdout="wrong-workspace\n",
                output_truncated=False,
            )

    monkeypatch.setattr("evalyn.probe_executor.ContainerRunner", Runner)
    executor = ContainerProbeExecutor(
        PytestAdapter(), engine="docker", run_dir=tmp_path / "artifacts"
    )

    with pytest.raises(RuntimeError, match="workspace attestation"):
        executor._resolve_probe_environment(tmp_path)

    assert not tuple(tmp_path.glob(".evalyn/probes/support/workspace-attestation-*"))


def test_missing_environment_identity_is_explicitly_incomplete(tmp_path: Path) -> None:
    executor = ContainerProbeExecutor(
        PytestAdapter(), engine="docker", run_dir=tmp_path / "artifacts"
    )

    assert executor._environment_metadata() == {"environment_integrity": "incomplete"}


def test_collection_preflight_commands_are_framework_owned(tmp_path: Path) -> None:
    pytest_executor = ContainerProbeExecutor(
        PytestAdapter(), engine="docker", run_dir=tmp_path / "pytest-artifacts"
    )
    pytest_artifact = artifact()
    pytest_command = pytest_executor._collection_command(pytest_artifact)
    vitest_executor = ContainerProbeExecutor(
        VitestAdapter(), engine="docker", run_dir=tmp_path / "vitest-artifacts"
    )
    vitest_artifact = VitestAdapter().build_artifact(
        "import { expect, test } from 'vitest';\ntest('ok', () => expect(1).toBe(1));"
    )

    assert "--collect-only" in pytest_command
    assert pytest_command[-1] == pytest_artifact.relative_path.as_posix()
    assert vitest_executor._collection_command(vitest_artifact) == (
        "npx",
        "--no-install",
        "vitest",
        "list",
        vitest_artifact.relative_path.as_posix(),
    )


@pytest.mark.parametrize(
    ("side", "expected"),
    (("base", "invalid_probe"), ("candidate", "failed")),
)
def test_collection_failure_is_causal_only_after_a_valid_base_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    side: str,
    expected: str,
) -> None:
    workspace = tmp_path / side
    workspace.mkdir()
    machine_result = (
        PYTEST_RESULT_PREFIX + '{"collection_errors":["ImportError: changed module"],'
        '"framework":"pytest","tests":[],"version":1}\n'
    )
    command_result = CommandResult(
        argv=("python",),
        cwd=workspace,
        exit_code=2,
        stdout=machine_result,
        stderr="ERROR collecting probe",
        duration_seconds=0.1,
        timed_out=False,
    )

    class Runner:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def run(self, *_args, **_kwargs):
            return command_result

    monkeypatch.setattr("evalyn.probe_executor.ContainerRunner", Runner)
    executor = ContainerProbeExecutor(
        PytestAdapter(), engine="docker", run_dir=tmp_path / "artifacts"
    )
    executor._prepared.add(workspace.resolve())
    executor._labels[workspace.resolve()] = side
    if side == "candidate":
        executor._collection_validated.add(artifact().probe_id)
    result = executor.execute(
        Revision(side, workspace),
        artifact(),
        repetition=1,
    )

    assert result.status.value == expected
    assert bool(result.failure_signature) is (side == "candidate")


def test_truncated_machine_output_is_infrastructure_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    command_result = CommandResult(
        argv=("python",),
        cwd=workspace,
        exit_code=1,
        stdout="partial",
        stderr="",
        duration_seconds=0.1,
        timed_out=False,
        stdout_truncated=True,
    )

    class Runner:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def run(self, *_args, **_kwargs):
            return command_result

    monkeypatch.setattr("evalyn.probe_executor.ContainerRunner", Runner)
    executor = ContainerProbeExecutor(
        PytestAdapter(), engine="docker", run_dir=tmp_path / "artifacts"
    )
    executor._prepared.add(workspace.resolve())
    executor._labels[workspace.resolve()] = "candidate"

    result = executor.execute(Revision("candidate", workspace), artifact(), repetition=1)

    assert result.status.value == "infra_error"
    assert result.metadata["result_protocol"] == "truncated"
