"""Tests for runner.py orchestration."""

from evalyn.config import Config
from evalyn.models import TestResult
from evalyn.runner.runner import run_execution_phase, _execute_category
from evalyn.state import AgentState


class TestRunExecutionPhaseDryRun:
    def test_returns_preview_result_without_timestamp_error(self, tmp_path):
        run_dir = tmp_path / ".agent" / "runs" / "2026-01-01_000000"
        run_dir.mkdir(parents=True)
        (run_dir / "plan.md").write_text(
            """\
# Evalyn Test Plan

### [REQUIRED] Unit Tests

- **Suggested commands**:
  - `pytest tests/`

### [OPTIONAL] Docs

- **Suggested commands**:
  - `pytest tests/docs/`

Status: APPROVED
""",
            encoding="utf-8",
        )

        result = run_execution_phase(
            Config(repo_path=tmp_path, claude_md_content="# Test Repo"),
            AgentState(),
            llm=None,
            dry_run=True,
        )

        assert result is not None
        assert result.timestamp == "2026-01-01_000000"
        assert result.branch == "(dry-run)"
        assert [(c.name, c.status) for c in result.category_results] == [
            ("Unit Tests", "skipped"),
            ("Docs", "skipped"),
        ]
        assert "would execute" in result.category_results[0].error_message
        assert "not required" in result.category_results[1].error_message


class TestExecuteCategoryRepairModes:
    def test_test_only_mode_marks_failure_without_repair(self, tmp_path, monkeypatch):
        def fake_run_test_command(**kwargs):
            return TestResult(
                category=kwargs["category"],
                command=kwargs["command"],
                exit_code=1,
                stdout="",
                stderr="failed",
                duration_seconds=0.1,
                status="failed",
            )

        monkeypatch.setattr(
            "evalyn.runner.runner.run_test_command",
            fake_run_test_command,
        )
        config = Config(repo_path=tmp_path, claude_md_content="# Test Repo")
        config.repair_mode = "test_only"

        result = _execute_category(
            config=config,
            llm=None,
            rollback=object(),
            repo_path=tmp_path,
            category_name="Unit Tests",
            commands=["pytest tests/"],
            relevant_files=[],
            category_metadata={"risk_level": "high"},
            run_dir=tmp_path,
            priority="required",
        )

        assert result.status == "failed"
        assert result.repair_outcome == "disabled"
        assert result.fix_attempts == []
        assert result.risk_level == "high"
