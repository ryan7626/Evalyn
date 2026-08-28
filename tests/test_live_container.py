"""Opt-in live smoke coverage for the OCI boundary."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from evalyn.adapters.pytest import PytestAdapter
from evalyn.engine.models import ExecutionStatus, Revision
from evalyn.execution import ContainerRunner
from evalyn.probe_executor import ContainerProbeExecutor
from evalyn.run_store import default_state_root
from evalyn.workflow import CheckOptions, run_check


class RecordedProbeProvider:
    def complete(self, system_prompt, user_prompt, *, max_tokens=4096):
        raise AssertionError("the live workflow requests structured output")

    def complete_json(self, system_prompt, user_prompt, *, max_tokens=4096):
        return {
            "probes": [
                {
                    "title": "service fee remains five cents",
                    "source": (
                        "from fee import total\n\n"
                        "def test_service_fee_contract():\n"
                        "    assert total(100) == 105\n"
                    ),
                    "oracle": "service fee remains five cents",
                    "oracle_provenance": "user_invariant",
                    "changed_file": "fee.py",
                    "changed_symbol": "total",
                }
            ]
        }


@pytest.mark.live_container
@pytest.mark.skipif(
    os.environ.get("EVALYN_LIVE_CONTAINER") != "1",
    reason="set EVALYN_LIVE_CONTAINER=1 to exercise the local Docker daemon",
)
def test_python_probe_runs_offline_in_restricted_container() -> None:
    parent = default_state_root() / "live-container-tests"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="workspace-", dir=parent) as raw_workspace:
        workspace = Path(raw_workspace)
        with ContainerRunner(
            workspace,
            image="python:3.12-slim",
            engine="docker",
            network="none",
            pull_policy="never",
            default_timeout_seconds=60,
        ) as runner:
            result = runner.run(
                [
                    "python",
                    "-c",
                    "from pathlib import Path; "
                    "Path('.evalyn-live-smoke').write_text('ok'); print('container-ok')",
                ]
            )
            invocation = runner.last_invocation

        assert result.succeeded, result.error or result.stderr
        assert result.stdout.strip() == "container-ok"
        assert (workspace / ".evalyn-live-smoke").read_text(encoding="utf-8") == "ok"
        assert invocation is not None and invocation.containerized
        assert invocation.argv[invocation.argv.index("--network") + 1] == "none"
        assert "--read-only" in invocation.argv
        assert "no-new-privileges" in invocation.argv


@pytest.mark.live_container
@pytest.mark.skipif(
    os.environ.get("EVALYN_LIVE_CONTAINER") != "1",
    reason="set EVALYN_LIVE_CONTAINER=1 to exercise the local Docker daemon",
)
def test_pytest_adapter_prepares_then_executes_generated_probe() -> None:
    parent = default_state_root() / "live-container-tests"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="probe-", dir=parent) as raw_workspace:
        workspace = Path(raw_workspace)
        (workspace / "pricing.py").write_text("PRICE = 42\n", encoding="utf-8")
        adapter = PytestAdapter()
        artifact = adapter.build_artifact(
            "from pathlib import Path\n\n"
            "import pytest\n\n"
            "from pricing import PRICE\n\n"
            "def test_price():\n"
            "    assert PRICE == 42\n"
            "    with pytest.raises(OSError):\n"
            "        Path('pricing.py').write_text('PRICE = 0\\n')\n"
        )
        executor = ContainerProbeExecutor(
            adapter,
            engine="docker",
            run_dir=workspace / "artifacts",
            timeout_seconds=120,
        )
        revision = Revision("a" * 40, workspace)

        executor.prepare((("candidate", revision),))
        result = executor.execute(revision, artifact, repetition=1)

        assert executor.setup_errors == {}
        assert result.status == ExecutionStatus.PASSED, result.stderr
        assert result.metadata["log_path"].endswith("run-1.log")
        assert (workspace / "pricing.py").read_text(encoding="utf-8") == "PRICE = 42\n"


@pytest.mark.live_container
@pytest.mark.skipif(
    os.environ.get("EVALYN_LIVE_CONTAINER") != "1",
    reason="set EVALYN_LIVE_CONTAINER=1 to exercise the local Docker daemon",
)
def test_full_diff_workflow_emits_stable_finding_without_touching_checkout() -> None:
    parent = default_state_root() / "live-container-tests"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="workflow-", dir=parent) as raw:
        root = Path(raw)
        repo = root / "repo"
        repo.mkdir()
        _git(repo, "init", "--quiet")
        _git(repo, "config", "user.name", "Evalyn Test")
        _git(repo, "config", "user.email", "evalyn@example.invalid")
        (repo / "fee.py").write_text("def total(cents):\n    return cents + 5\n", encoding="utf-8")
        _git(repo, "add", "fee.py")
        _git(repo, "commit", "--quiet", "-m", "base")
        base_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
        (repo / "fee.py").write_text("def total(cents):\n    return cents + 6\n", encoding="utf-8")
        _git(repo, "add", "fee.py")
        _git(repo, "commit", "--quiet", "-m", "candidate")
        candidate_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

        outcome = run_check(
            CheckOptions(
                repo_path=repo,
                range_spec=f"{base_sha}...{candidate_sha}",
                adapters=("pytest",),
                max_tests=1,
                repeat=2,
                budget_seconds=180,
                runtime="docker",
                state_root=root / "state",
                intent=("service fee remains five cents",),
            ),
            llm=RecordedProbeProvider(),
        )

        diagnostics = [
            {
                "classification": assessment.classification,
                "base": [record.result for record in assessment.base_runs],
                "candidate": [record.result for record in assessment.candidate_runs],
            }
            for result in outcome.results
            for assessment in result.assessments
        ]
        assert len(outcome.findings) == 1, diagnostics
        assert len(outcome.gating_findings) == 1
        assert outcome.findings[0].classification.kind.value == "stable_change"
        assert outcome.report_paths["sarif"].is_file()
        assert _git(repo, "status", "--porcelain").stdout == ""
        assert not (repo / ".evalyn").exists()


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
