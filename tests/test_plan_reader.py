"""Tests for runner/plan_reader.py — plan parsing, filtering, and file extraction."""

import textwrap
from pathlib import Path

import pytest

from evalyn.runner.plan_reader import (
    parse_plan_categories,
    parse_user_notes,
    apply_category_filters,
    extract_files_for_category,
    extract_category_metadata,
)


@pytest.fixture
def plan(tmp_path):
    """Helper: write a plan.md with given content and return the path."""
    def _write(content: str) -> Path:
        p = tmp_path / "plan.md"
        p.write_text(textwrap.dedent(content), encoding="utf-8")
        return p
    return _write


# ── extract_files_for_category ───────────────────────────────────────────────


class TestExtractFilesForCategory:
    def test_extracts_backtick_files(self, plan):
        p = plan("""\
            ### [REQUIRED] Auth Tests

            - **Relevant files**: `src/auth.ts`, `src/session.ts`
            - **Suggested commands**: `npm test`
        """)
        files = extract_files_for_category(p, "Auth Tests")
        assert "src/auth.ts" in files
        assert "src/session.ts" in files

    def test_returns_empty_when_category_not_found(self, plan):
        p = plan("""\
            ### [REQUIRED] Auth Tests

            - **Relevant files**: `src/auth.ts`
        """)
        files = extract_files_for_category(p, "Nonexistent Category")
        assert files == []

    def test_does_not_match_description_lines(self, plan):
        """BUG-7 regression: lines with 'files' but not 'relevant files' must not match."""
        p = plan("""\
            ### [REQUIRED] Auth Tests

            - **Description**: Changes made to several files in auth module.
            - **Relevant files**: `src/auth.ts`
            - **Suggested commands**: `pytest`
        """)
        files = extract_files_for_category(p, "Auth Tests")
        # Should only find the relevant files line, not the description line
        assert "src/auth.ts" in files
        # Must not pick up extension-less words from the description line
        assert len(files) == 1

    def test_empty_relevant_files_line(self, plan):
        p = plan("""\
            ### [REQUIRED] Auth Tests

            - **Relevant files**: _none specified_
        """)
        # No backtick-wrapped file paths on that line
        files = extract_files_for_category(p, "Auth Tests")
        assert files == []

    def test_case_insensitive_category_match(self, plan):
        p = plan("""\
            ### [REQUIRED] Auth Tests

            - **Relevant files**: `src/auth.ts`
        """)
        # Lowercase match
        files = extract_files_for_category(p, "auth tests")
        assert "src/auth.ts" in files


# ── extract_category_metadata ────────────────────────────────────────────────


class TestExtractCategoryMetadata:
    def test_extracts_rich_plan_metadata(self, plan):
        p = plan("""\
            ### [REQUIRED] Auth Tests

            - **Description**: Auth behavior changed.
            - **Risk level**: high
            - **Why this matters**: Login can break.
            - **Affected surface**: Session handling
            - **Relevant files**: `src/auth.ts`
            - **Success criteria**: Auth tests pass.
            - **Suggested commands**:
              - `pytest tests/auth.py`
        """)

        metadata = extract_category_metadata(p, "Auth Tests")

        assert metadata["priority"] == "required"
        assert metadata["description"] == "Auth behavior changed."
        assert metadata["risk_level"] == "high"
        assert metadata["plan_reason"] == "Login can break."
        assert metadata["affected_surface"] == "Session handling"
        assert metadata["success_criteria"] == "Auth tests pass."

    def test_missing_metadata_returns_empty_fields(self, plan):
        p = plan("""\
            ### [REQUIRED] Auth Tests

            - `pytest`
        """)

        metadata = extract_category_metadata(p, "Missing")

        assert metadata["priority"] == ""
        assert metadata["description"] == ""


# ── parse_plan_categories (public-name re-test for plan_reader module) ───────


class TestParsePlanCategoriesModule:
    """Smoke tests to confirm plan_reader exports the same logic as before."""

    def test_parses_required(self, plan):
        p = plan("""\
            ### [REQUIRED] Database

            - `pytest tests/db/`
        """)
        cats = parse_plan_categories(p)
        assert len(cats) == 1
        assert cats[0] == ("Database", ["pytest tests/db/"], "required")

    def test_parses_optional(self, plan):
        p = plan("""\
            ### [OPTIONAL] Docs

            - `npm run build:docs`
        """)
        cats = parse_plan_categories(p)
        assert cats[0][2] == "optional"

    def test_parses_skip(self, plan):
        p = plan("""\
            ### [SKIP] Flaky Suite

            - `npm test`
        """)
        cats = parse_plan_categories(p)
        assert cats[0][2] == "skip"

    def test_no_commands(self, plan):
        p = plan("""\
            ### [REQUIRED] Empty

            Some description only.
        """)
        cats = parse_plan_categories(p)
        assert cats[0][1] == []

    def test_metadata_backticks_are_not_parsed_as_commands(self, plan):
        p = plan("""\
            ### [REQUIRED] Auth Tests

            - **Success criteria**: `pytest tests/auth.py` passes cleanly.
            - **Suggested commands**:
              - `pytest tests/unit.py`
        """)
        cats = parse_plan_categories(p)
        assert cats[0][1] == ["pytest tests/unit.py"]


# ── apply_category_filters (spot checks for public rename) ───────────────────


class TestApplyCategoryFiltersModule:
    def test_required_only_by_default(self):
        cats = [
            ("A", ["cmd"], "required"),
            ("B", ["cmd"], "optional"),
        ]
        to_run, skipped = apply_category_filters(cats)
        assert [n for n, _, _ in to_run] == ["A"]
        assert any(n == "B" for n, _ in skipped)

    def test_cli_filter_takes_precedence(self):
        cats = [
            ("A", ["cmd"], "required"),
            ("B", ["cmd"], "optional"),
        ]
        to_run, _ = apply_category_filters(cats, cli_filter=["B"])
        assert [n for n, _, _ in to_run] == ["B"]


# ── parse_user_notes (spot checks) ───────────────────────────────────────────


class TestParseUserNotesModule:
    def test_only_run_directive(self, plan):
        p = plan("Status: APPROVED\n\nONLY RUN Auth Tests\n")
        d = parse_user_notes(p)
        assert "Auth Tests" in d["only_run"]

    def test_skip_directive(self, plan):
        p = plan("Status: APPROVED\n\nSKIP Flaky Suite\n")
        d = parse_user_notes(p)
        assert "Flaky Suite" in d["skip"]

    def test_max_categories(self, plan):
        p = plan("Status: APPROVED\n\nMAX CATEGORIES 2\n")
        d = parse_user_notes(p)
        assert d["max_categories"] == 2
