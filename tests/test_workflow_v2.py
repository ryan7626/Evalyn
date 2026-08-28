import subprocess
from pathlib import Path

import pytest

import evalyn.workflow as workflow_module
from evalyn.engine.models import ExecutionResult, ExecutionStatus
from evalyn.execution.worktree import WorktreeSession
from evalyn.llm.base import BaseLLM
from evalyn.settings import load_settings
from evalyn.workflow import (
    CheckError,
    CheckOptions,
    _read_file_prefix,
    _settings_snapshot,
    _workspace_diff,
    inspect_check,
    run_check,
)


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    target = tmp_path / "repo"
    target.mkdir()
    git(target, "init", "-b", "main")
    git(target, "config", "user.email", "test@example.com")
    git(target, "config", "user.name", "Test")
    (target / "pyproject.toml").write_text(
        "[project]\nname='fixture'\nversion='1'\n[tool.pytest.ini_options]\n",
        encoding="utf-8",
    )
    (target / "docs").mkdir()
    (target / "docs" / "api.md").write_text(
        "The public documentation guarantees the value remains one.\n",
        encoding="utf-8",
    )
    (target / "value.py").write_text("def value(): return 1\n", encoding="utf-8")
    git(target, "add", ".")
    git(target, "commit", "-m", "base")
    return target


class FakeLLM(BaseLLM):
    def complete(self, system_prompt, user_prompt, *, max_tokens=4096):
        raise AssertionError("not used")

    def complete_json(self, system_prompt, user_prompt, *, max_tokens=4096):
        return {
            "probes": [
                {
                    "title": "value remains one",
                    "source": "from value import value\ndef test_value():\n    assert value() == 1",
                    "oracle": "the public documentation guarantees the value remains one",
                    "oracle_provenance": "documentation",
                    "changed_file": "value.py",
                    "changed_symbol": "value",
                }
            ]
        }


class FakeExecutor:
    def __init__(self):
        self.setup_errors = {}

    def prepare(self, revisions):
        self.base_sha = revisions[0][1].sha

    def execute(self, revision, artifact, *, repetition):
        if revision.sha == self.base_sha:
            return ExecutionResult.passed()
        return ExecutionResult.failed(failure_signature="AssertionError:value")


def factory(*_args):
    return FakeExecutor()


class InvalidProbeExecutor(FakeExecutor):
    def execute(self, revision, artifact, *, repetition):
        return ExecutionResult(
            status=ExecutionStatus.INVALID_PROBE,
            exit_code=2,
            stderr="probe collection failed",
        )


def test_full_workflow_writes_gating_evidence_without_touching_checkout(
    repo: Path, tmp_path: Path
) -> None:
    git(repo, "checkout", "-b", "feature")
    (repo / "value.py").write_text("def value(): return 2\n", encoding="utf-8")
    git(repo, "commit", "-am", "change")
    before_status = git(repo, "status", "--porcelain=v1")
    before_worktrees = git(repo, "worktree", "list", "--porcelain")

    outcome = run_check(
        CheckOptions(repo, range_spec="main...HEAD", state_root=tmp_path / "state"),
        llm=FakeLLM(),
        executor_factory=factory,
    )

    assert len(outcome.gating_findings) == 1
    assert outcome.report_paths["sarif"].is_file()
    assert outcome.run.manifest["status"] == "complete"
    configuration = outcome.run.manifest["configuration"]
    assert configuration["limits"]["repeat"] == 3
    assert configuration["adapters"][0]["image"].startswith("python:")
    assert git(repo, "status", "--porcelain=v1") == before_status
    assert git(repo, "worktree", "list", "--porcelain") == before_worktrees


def test_invalid_probe_evidence_marks_run_incomplete(repo: Path, tmp_path: Path) -> None:
    git(repo, "checkout", "-b", "feature")
    (repo / "value.py").write_text("def value(): return 2\n", encoding="utf-8")
    git(repo, "commit", "-am", "change")

    outcome = run_check(
        CheckOptions(repo, range_spec="main...HEAD", state_root=tmp_path / "state"),
        llm=FakeLLM(),
        executor_factory=lambda *_args: InvalidProbeExecutor(),
    )

    assert outcome.incomplete is True
    assert outcome.run.manifest["status"] == "incomplete"
    assert outcome.gating_findings == ()


def test_builtin_provider_timeout_uses_remaining_check_budget(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    git(repo, "checkout", "-b", "feature")
    (repo / "value.py").write_text("def value(): return 2\n", encoding="utf-8")
    git(repo, "commit", "-am", "change")
    observed: list[float] = []

    def fake_create_provider(_settings, *, timeout_seconds=120):
        observed.append(timeout_seconds)
        return FakeLLM()

    monkeypatch.setattr(workflow_module, "create_provider", fake_create_provider)

    run_check(
        CheckOptions(
            repo,
            range_spec="main...HEAD",
            state_root=tmp_path / "state",
            budget_seconds=7,
        ),
        executor_factory=factory,
    )

    assert len(observed) == 1
    assert 0 < observed[0] <= 7


def test_settings_snapshot_can_pin_the_resolved_container_runtime(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    snapshot = _settings_snapshot(load_settings(repo), ("pytest",), runtime="docker")

    assert snapshot["sandbox"]["runtime"] == "docker"


def test_dirty_only_candidate_gets_content_addressed_identity(repo: Path, tmp_path: Path) -> None:
    original = (repo / "value.py").read_bytes()
    (repo / "value.py").write_text("def value(): return 2\n", encoding="utf-8")
    dirty = (repo / "value.py").read_bytes()

    outcome = run_check(
        CheckOptions(
            repo,
            range_spec="HEAD...HEAD",
            include_dirty=True,
            state_root=tmp_path / "state",
        ),
        llm=FakeLLM(),
        executor_factory=factory,
    )

    assert "+dirty." in outcome.run.manifest["candidate_id"]
    assert (repo / "value.py").read_bytes() == dirty
    assert (repo / "value.py").read_bytes() != original


def test_dirty_checkout_requires_explicit_snapshot(repo: Path) -> None:
    (repo / "value.py").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(CheckError, match="--include-dirty"):
        inspect_check(CheckOptions(repo, range_spec="HEAD...HEAD"))


def test_setup_failure_is_incomplete_and_preserves_checkout(repo: Path, tmp_path: Path) -> None:
    git(repo, "checkout", "-b", "feature")
    (repo / "value.py").write_text("def value(): return 2\n", encoding="utf-8")
    git(repo, "commit", "-am", "change")

    class BrokenExecutor(FakeExecutor):
        def prepare(self, revisions):
            self.setup_errors = {revisions[0][1].workspace: "failed"}

    with pytest.raises(CheckError, match="dependency setup failed") as raised:
        run_check(
            CheckOptions(repo, range_spec="main", state_root=tmp_path / "state"),
            llm=FakeLLM(),
            executor_factory=lambda *_args: BrokenExecutor(),
        )

    assert raised.value.run_path is not None
    assert git(repo, "status", "--porcelain=v1") == ""


def test_context_file_prefix_read_never_buffers_the_entire_file(tmp_path: Path) -> None:
    source = tmp_path / "large-context.txt"
    source.write_bytes(b"x" * 1_000_000)

    data, truncated = _read_file_prefix(source, 128)

    assert data == b"x" * 128
    assert truncated is True


def test_workspace_diff_stream_truncates_a_large_git_diff(repo: Path, tmp_path: Path) -> None:
    base_sha = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "-b", "large-diff")
    (repo / "value.py").write_text("x = " + repr("x" * 1_000_000), encoding="utf-8")
    git(repo, "commit", "-am", "large diff")
    candidate_sha = git(repo, "rev-parse", "HEAD")

    with WorktreeSession(
        repo,
        baseline_revision=base_sha,
        candidate_revision=candidate_sha,
        temp_parent=tmp_path / "sessions",
    ) as worktrees:
        rendered = _workspace_diff(worktrees, ("value.py",))

    body, marker = rendered.split("\n\n", 1)
    assert len(body) <= 120_000
    assert marker == "[truncated: diff exceeded safe input limit]"
