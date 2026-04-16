"""Tests for the v2 smart plan parsing and category filtering logic."""

import textwrap
from pathlib import Path

import pytest

from evalyn.runner.plan_reader import (
    parse_plan_categories as _parse_plan_categories,
    parse_user_notes as _parse_user_notes,
    apply_category_filters as _apply_category_filters,
)


@pytest.fixture
def tmp_plan(tmp_path):
    """Helper to create a plan.md with given content."""
    def _write(content: str) -> Path:
        plan = tmp_path / "plan.md"
        plan.write_text(textwrap.dedent(content), encoding="utf-8")
        return plan
    return _write


# ── _parse_plan_categories ──────────────────────────────────────────


class TestParsePlanCategories:
    def test_parses_required_and_optional(self, tmp_plan):
        plan = tmp_plan("""\
            ### [REQUIRED] Auth Tests
            - `npm test -- --grep auth`

            ### [OPTIONAL] Docs Build
            - `npm run build:docs`
        """)
        cats = _parse_plan_categories(plan)
        assert len(cats) == 2
        assert cats[0] == ("Auth Tests", ["npm test -- --grep auth"], "required")
        assert cats[1] == ("Docs Build", ["npm run build:docs"], "optional")

    def test_parses_skip_marker(self, tmp_plan):
        plan = tmp_plan("""\
            ### [REQUIRED] Keep This
            - `pytest tests/`

            ### [SKIP] Drop This
            - `npm test`

            ### [REQUIRED] Also Keep
            - `npm run lint`
        """)
        cats = _parse_plan_categories(plan)
        assert len(cats) == 3
        names = [c[0] for c in cats]
        assert "Keep This" in names
        assert "Drop This" in names
        assert "Also Keep" in names
        # Verify priorities
        priorities = {c[0]: c[2] for c in cats}
        assert priorities["Drop This"] == "skip"
        assert priorities["Keep This"] == "required"

    def test_case_insensitive_markers(self, tmp_plan):
        plan = tmp_plan("""\
            ### [required] Lower Case
            - `make test`

            ### [Skip] Mixed Case
            - `npm test`
        """)
        cats = _parse_plan_categories(plan)
        assert cats[0][2] == "required"
        assert cats[1][2] == "skip"

    def test_categories_with_no_commands(self, tmp_plan):
        plan = tmp_plan("""\
            ### [REQUIRED] Empty Category

            Some description but no commands.

            ### [REQUIRED] Has Commands
            - `pytest`
        """)
        cats = _parse_plan_categories(plan)
        assert len(cats) == 2
        assert cats[0] == ("Empty Category", [], "required")
        assert cats[1] == ("Has Commands", ["pytest"], "required")


# ── _parse_user_notes ───────────────────────────────────────────────


class TestParseUserNotes:
    def test_only_run_single(self, tmp_plan):
        plan = tmp_plan("""\
            Status: APPROVED

            ONLY RUN Auth Tests
        """)
        d = _parse_user_notes(plan)
        assert d["only_run"] == ["Auth Tests"]

    def test_only_run_multiple_comma_separated(self, tmp_plan):
        plan = tmp_plan("""\
            Status: APPROVED

            ONLY RUN Auth Tests, Photo Upload
        """)
        d = _parse_user_notes(plan)
        assert d["only_run"] == ["Auth Tests", "Photo Upload"]

    def test_skip_directive(self, tmp_plan):
        plan = tmp_plan("""\
            Status: APPROVED

            SKIP Marble API Integration
        """)
        d = _parse_user_notes(plan)
        assert d["skip"] == ["Marble API Integration"]

    def test_max_categories(self, tmp_plan):
        plan = tmp_plan("""\
            Status: APPROVED

            MAX CATEGORIES 2
        """)
        d = _parse_user_notes(plan)
        assert d["max_categories"] == 2

    def test_ignores_html_comments(self, tmp_plan):
        plan = tmp_plan("""\
            Status: APPROVED

            <!-- ONLY RUN Should Be Ignored -->
            SKIP Real Directive
        """)
        d = _parse_user_notes(plan)
        assert d["only_run"] == []
        assert d["skip"] == ["Real Directive"]

    def test_user_notes_section_header(self, tmp_plan):
        plan = tmp_plan("""\
            ## User Notes

            ONLY RUN My Category
        """)
        d = _parse_user_notes(plan)
        assert d["only_run"] == ["My Category"]

    def test_no_approved_or_notes_section(self, tmp_plan):
        plan = tmp_plan("""\
            Status: PENDING

            Some random text
        """)
        d = _parse_user_notes(plan)
        assert d["only_run"] == []
        assert d["skip"] == []
        assert d["max_categories"] is None

    def test_multiple_directives(self, tmp_plan):
        plan = tmp_plan("""\
            Status: APPROVED

            SKIP Category A
            SKIP Category B
            MAX CATEGORIES 3
        """)
        d = _parse_user_notes(plan)
        assert d["skip"] == ["Category A", "Category B"]
        assert d["max_categories"] == 3


# ── _apply_category_filters ─────────────────────────────────────────


class TestApplyCategoryFilters:
    def _cats(self, *items):
        """Helper to build category tuples: (name, commands, priority)."""
        return list(items)

    def test_default_runs_only_required(self):
        cats = self._cats(
            ("A", ["cmd1"], "required"),
            ("B", ["cmd2"], "optional"),
            ("C", ["cmd3"], "required"),
        )
        to_run, skipped = _apply_category_filters(cats)
        assert [n for n, _, _ in to_run] == ["A", "C"]
        assert [n for n, _ in skipped] == ["B"]

    def test_skip_marker_always_skips(self):
        cats = self._cats(
            ("A", ["cmd1"], "required"),
            ("B", ["cmd2"], "skip"),
        )
        to_run, skipped = _apply_category_filters(cats)
        assert [n for n, _, _ in to_run] == ["A"]
        assert skipped[0] == ("B", "[SKIP] marker in plan")

    def test_cli_filter_overrides_everything(self):
        cats = self._cats(
            ("A", ["cmd1"], "required"),
            ("B", ["cmd2"], "required"),
            ("C", ["cmd3"], "optional"),
        )
        to_run, skipped = _apply_category_filters(
            cats, cli_filter=["C"]
        )
        assert [n for n, _, _ in to_run] == ["C"]
        assert len(skipped) == 2

    def test_cli_filter_still_respects_skip_marker(self):
        cats = self._cats(
            ("A", ["cmd1"], "skip"),
            ("B", ["cmd2"], "required"),
        )
        to_run, skipped = _apply_category_filters(
            cats, cli_filter=["A", "B"]
        )
        # A is [SKIP] in the plan header, should still be skipped
        assert [n for n, _, _ in to_run] == ["B"]
        assert ("A", "[SKIP] marker in plan") in skipped

    def test_only_run_directive(self):
        cats = self._cats(
            ("A", ["cmd1"], "required"),
            ("B", ["cmd2"], "required"),
            ("C", ["cmd3"], "required"),
        )
        directives = {"only_run": ["B"], "skip": [], "max_categories": None}
        to_run, skipped = _apply_category_filters(cats, directives=directives)
        assert [n for n, _, _ in to_run] == ["B"]
        assert len(skipped) == 2

    def test_skip_directive(self):
        cats = self._cats(
            ("A", ["cmd1"], "required"),
            ("B", ["cmd2"], "required"),
        )
        directives = {"only_run": [], "skip": ["A"], "max_categories": None}
        to_run, skipped = _apply_category_filters(cats, directives=directives)
        assert [n for n, _, _ in to_run] == ["B"]
        assert ("A", "SKIP directive in user notes") in skipped

    def test_max_categories_limit(self):
        cats = self._cats(
            ("A", ["cmd1"], "required"),
            ("B", ["cmd2"], "required"),
            ("C", ["cmd3"], "required"),
        )
        directives = {"only_run": [], "skip": [], "max_categories": 1}
        to_run, skipped = _apply_category_filters(cats, directives=directives)
        assert [n for n, _, _ in to_run] == ["A"]
        skip_names = [n for n, _ in skipped]
        assert "B" in skip_names
        assert "C" in skip_names

    def test_case_insensitive_matching(self):
        cats = self._cats(
            ("UI Enhancements", ["cmd1"], "required"),
        )
        to_run, skipped = _apply_category_filters(
            cats, cli_filter=["ui enhancements"]
        )
        assert [n for n, _, _ in to_run] == ["UI Enhancements"]

    def test_only_run_forces_optional(self):
        cats = self._cats(
            ("A", ["cmd1"], "optional"),
            ("B", ["cmd2"], "required"),
        )
        directives = {"only_run": ["A"], "skip": [], "max_categories": None}
        to_run, skipped = _apply_category_filters(cats, directives=directives)
        assert [n for n, _, _ in to_run] == ["A"]

    def test_empty_categories_returns_empty(self):
        to_run, skipped = _apply_category_filters([])
        assert to_run == []
        assert skipped == []
