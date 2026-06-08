"""Tests for reporter/reporter.py — report generation, icons, and formatting."""

import json
import tempfile
from pathlib import Path

import pytest

from evalyn.models import CategoryResult, FixAttempt, RunResult, TestResult
from evalyn.reporter.reporter import (
    _build_json_report,
    _format_duration,
    _render_category_result,
    _render_markdown_report,
    generate_reports,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_run_dir(tmp: Path) -> Path:
    run_dir = tmp / "runs" / "2026-01-01_000000"
    run_dir.mkdir(parents=True)
    return run_dir


def _make_run_result(run_dir: Path) -> RunResult:
    return RunResult(
        timestamp="2026-01-01_000000",
        branch="evalyn/test",
        run_dir=run_dir,
    )


def _make_test_result(status: str = "passed") -> TestResult:
    return TestResult(
        category="Auth",
        command="pytest tests/",
        exit_code=0 if status == "passed" else 1,
        stdout="ok",
        stderr="",
        duration_seconds=1.5,
        status=status,
    )


# ── _format_duration ─────────────────────────────────────────────────────────


class TestFormatDuration:
    def test_seconds(self):
        assert _format_duration(45) == "45 seconds"

    def test_one_minute(self):
        assert _format_duration(60) == "1m 0s"

    def test_minutes_and_seconds(self):
        assert _format_duration(90) == "1m 30s"

    def test_hours(self):
        assert _format_duration(3661) == "1h 1m"

    def test_zero_seconds(self):
        assert _format_duration(0) == "0 seconds"


# ── _render_category_result ──────────────────────────────────────────────────


class TestRenderCategoryResult:
    def test_passed_uses_checkmark_icon(self):
        cat = CategoryResult(name="Auth", status="passed")
        lines = _render_category_result(cat)
        assert any("✅" in l for l in lines)

    def test_failed_uses_cross_icon(self):
        cat = CategoryResult(name="Auth", status="failed")
        lines = _render_category_result(cat)
        assert any("❌" in l for l in lines)

    def test_uncertain_uses_question_icon(self):
        cat = CategoryResult(name="Auth", status="uncertain")
        lines = _render_category_result(cat)
        assert any("❓" in l for l in lines)

    def test_skipped_uses_skip_icon(self):
        cat = CategoryResult(name="Auth", status="skipped")
        lines = _render_category_result(cat)
        assert any("⏭" in l for l in lines)

    def test_error_message_included(self):
        cat = CategoryResult(name="Auth", status="uncertain", error_message="timeout quirk")
        lines = _render_category_result(cat)
        assert any("timeout quirk" in l for l in lines)

    def test_test_result_command_shown(self):
        cat = CategoryResult(name="Auth", status="passed")
        cat.test_results.append(_make_test_result("passed"))
        lines = _render_category_result(cat)
        assert any("pytest" in l for l in lines)

    def test_failed_test_shows_failure_summary(self):
        cat = CategoryResult(name="Auth", status="failed")
        tr = _make_test_result("failed")
        tr.stderr = "AssertionError: expected True"
        cat.test_results.append(tr)
        lines = _render_category_result(cat)
        assert any("AssertionError" in l for l in lines)

    def test_applied_fix_shown(self):
        cat = CategoryResult(name="Auth", status="passed")
        fix = FixAttempt(
            file_path="auth.py",
            original_content="a",
            replacement_content="b",
            explanation="fixed missing import",
            applied=True,
        )
        cat.fix_attempts.append(fix)
        lines = _render_category_result(cat)
        assert any("auth.py" in l for l in lines)

    def test_repaired_pass_has_review_next_action(self):
        cat = CategoryResult(
            name="Auth",
            status="passed",
            repair_outcome="applied_passed",
        )
        lines = _render_category_result(cat)
        assert any("Review the unstaged fix" in l for l in lines)

    def test_rich_metadata_shown(self):
        cat = CategoryResult(
            name="Auth",
            status="passed",
            priority="required",
            risk_level="high",
            affected_surface="Sessions",
            success_criteria="Auth tests pass.",
            plan_reason="Login can break.",
        )
        lines = _render_category_result(cat)
        assert any("Risk: high" in l for l in lines)
        assert any("Sessions" in l for l in lines)

    def test_rejected_fix_shown(self):
        cat = CategoryResult(name="Auth", status="uncertain")
        fix = FixAttempt(
            file_path="auth.py",
            original_content="a",
            replacement_content="b",
            explanation="attempt",
            applied=False,
            rejected_reason="LLM declined",
        )
        cat.fix_attempts.append(fix)
        lines = _render_category_result(cat)
        assert any("LLM declined" in l for l in lines)


# ── generate_reports ─────────────────────────────────────────────────────────


class TestGenerateReports:
    def test_creates_markdown_and_json(self, tmp_path):
        run_result = _make_run_result(_make_run_dir(tmp_path))
        md_path, json_path = generate_reports(run_result)
        assert md_path.is_file()
        assert json_path.is_file()

    def test_json_is_valid(self, tmp_path):
        run_result = _make_run_result(_make_run_dir(tmp_path))
        _, json_path = generate_reports(run_result)
        data = json.loads(json_path.read_text())
        assert "summary" in data
        assert "categories" in data
        assert "timestamp" in data

    def test_json_counts_correct(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        run_result = _make_run_result(run_dir)
        run_result.category_results.append(CategoryResult(name="A", status="passed"))
        run_result.category_results.append(CategoryResult(name="B", status="failed"))
        run_result.category_results.append(CategoryResult(name="C", status="uncertain"))
        run_result.category_results.append(CategoryResult(name="D", status="skipped"))
        _, json_path = generate_reports(run_result)
        data = json.loads(json_path.read_text())
        assert data["summary"]["passed"] == 1
        assert data["summary"]["failed"] == 1
        assert data["summary"]["uncertain"] == 1
        assert data["summary"]["skipped"] == 1

    def test_markdown_contains_timestamp(self, tmp_path):
        run_result = _make_run_result(_make_run_dir(tmp_path))
        md_path, _ = generate_reports(run_result)
        assert "2026-01-01_000000" in md_path.read_text()

    def test_markdown_contains_branch(self, tmp_path):
        run_result = _make_run_result(_make_run_dir(tmp_path))
        md_path, _ = generate_reports(run_result)
        assert "evalyn/test" in md_path.read_text()

    def test_markdown_contains_morning_decision(self, tmp_path):
        run_result = _make_run_result(_make_run_dir(tmp_path))
        run_result.category_results.append(CategoryResult(name="A", status="passed"))
        md_path, _ = generate_reports(run_result)
        content = md_path.read_text()
        assert "Morning Decision" in content
        assert "No immediate action" in content

    def test_json_branch_recorded(self, tmp_path):
        run_result = _make_run_result(_make_run_dir(tmp_path))
        _, json_path = generate_reports(run_result)
        data = json.loads(json_path.read_text())
        assert data["branch"] == "evalyn/test"

    def test_no_files_modified_section(self, tmp_path):
        run_result = _make_run_result(_make_run_dir(tmp_path))
        md_path, _ = generate_reports(run_result)
        assert "No files were modified" in md_path.read_text()

    def test_files_modified_listed(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        run_result = _make_run_result(run_dir)
        cat = CategoryResult(name="A", status="passed")
        fix = FixAttempt(
            file_path="src/auth.py",
            original_content="a",
            replacement_content="b",
            explanation="fix",
            applied=True,
        )
        cat.fix_attempts.append(fix)
        run_result.category_results.append(cat)
        md_path, _ = generate_reports(run_result)
        assert "src/auth.py" in md_path.read_text()

    def test_rolled_back_fix_not_listed_as_modified(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        run_result = _make_run_result(run_dir)
        cat = CategoryResult(name="A", status="uncertain")
        fix = FixAttempt(
            file_path="src/auth.py",
            original_content="a",
            replacement_content="b",
            explanation="fix",
            applied=True,
            rolled_back=True,
        )
        cat.fix_attempts.append(fix)
        run_result.category_results.append(cat)
        md_path, _ = generate_reports(run_result)
        content = md_path.read_text()
        assert "No files were modified" in content
