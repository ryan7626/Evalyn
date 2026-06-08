"""Report generator for Evalyn.

Produces markdown and JSON reports from run results.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from evalyn.models import RunResult, CategoryResult
from evalyn.utils.log import get_logger

logger = get_logger("reporter")


def generate_reports(result: RunResult) -> tuple[Path, Path]:
    """Generate both markdown and JSON reports for a run.

    Args:
        result: The completed RunResult.

    Returns:
        Tuple of (markdown_path, json_path).
    """
    md_path = result.run_dir / "report.md"
    json_path = result.run_dir / "report.json"

    md_content = _render_markdown_report(result)
    md_path.write_text(md_content, encoding="utf-8")
    logger.info(f"Markdown report: {md_path}")

    json_content = _build_json_report(result)
    json_path.write_text(
        json.dumps(json_content, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(f"JSON report: {json_path}")

    return md_path, json_path


def _render_markdown_report(result: RunResult) -> str:
    """Render the morning markdown report."""
    lines = [
        f"# Evalyn Morning Report — {result.timestamp}",
        "",
        "## Summary",
        "",
        f"- ✅ **{result.passed_count}** categories passed",
        f"- ❌ **{result.failed_count}** categories failed",
        f"- ❓ **{result.uncertain_count}** categories uncertain",
        f"- ⏭ **{result.skipped_count}** categories skipped",
        f"- ⏱️ Total runtime: {_format_duration(result.total_duration_seconds)}",
        f"- 🌿 Branch: `{result.branch}`",
        "",
        "## Morning Decision",
        "",
    ]

    if result.category_results:
        for cat in result.category_results:
            lines.append(f"- **{cat.name}**: {_next_action_for_category(cat)}")
    else:
        lines.append("_No categories were recorded for this run._")

    lines.extend([
        "",
        "---",
        "",
        "## Results by Category",
        "",
    ])

    for cat in result.category_results:
        lines.extend(_render_category_result(cat))
        lines.append("")

    # Files modified section
    lines.extend(_render_modifications(result))

    # Patches section
    lines.extend(_render_patches(result))

    return "\n".join(lines)


def _render_category_result(cat: CategoryResult) -> list[str]:
    """Render a single category's result as markdown."""
    icon = {"passed": "✅", "failed": "❌", "uncertain": "❓", "skipped": "⏭"}.get(cat.status, "❓")
    status_label = cat.status.upper()

    lines = [
        f"### {icon} {cat.name} ({status_label})",
        "",
    ]

    details = []
    if cat.priority:
        details.append(f"Priority: {cat.priority.upper()}")
    if cat.risk_level:
        details.append(f"Risk: {cat.risk_level}")
    if cat.affected_surface:
        details.append(f"Surface: {cat.affected_surface}")
    if cat.repair_outcome and cat.repair_outcome != "none":
        details.append(f"Repair: {_repair_outcome_label(cat.repair_outcome)}")

    for detail in details:
        lines.append(f"- {detail}")

    if cat.description:
        lines.append(f"- Plan: {cat.description}")
    if cat.plan_reason:
        lines.append(f"- Why it matters: {cat.plan_reason}")
    if cat.success_criteria:
        lines.append(f"- Success criteria: {cat.success_criteria}")

    lines.append(f"- Next action: {_next_action_for_category(cat)}")
    lines.append("")

    # Test results
    for test in cat.test_results:
        test_icon = "✅" if test.is_success else "❌"
        lines.append(
            f"- {test_icon} `{test.command}` — "
            f"{test.status} in {test.duration_seconds:.1f}s"
        )

        if not test.is_success and test.failure_summary:
            # Indent failure details
            for detail_line in test.failure_summary.split("\n")[:5]:
                lines.append(f"  > {detail_line}")

    # Fix attempts
    if cat.fix_attempts:
        lines.append("")
        lines.append("**Fix attempts:**")
        for i, fix in enumerate(cat.fix_attempts, 1):
            context = _fix_context(fix)
            if fix.applied:
                lines.append(
                    f"  {i}. Applied fix to `{fix.file_path}`{context}: {fix.explanation}"
                )
            elif fix.suggested_only:
                patch_note = (
                    f" (patch: `{Path(fix.suggested_patch_path).name}`)"
                    if fix.suggested_patch_path else ""
                )
                lines.append(
                    f"  {i}. Suggested fix to `{fix.file_path}`{context}{patch_note}: "
                    f"{fix.explanation}"
                )
            elif fix.rejected_reason:
                lines.append(
                    f"  {i}. Rejected ({fix.rejected_reason}){context}: {fix.explanation}"
                )
            else:
                lines.append(f"  {i}. No fix{context}: {fix.explanation}")

    # Error message
    if cat.error_message:
        lines.append("")
        lines.append(f"> ⚠️ {cat.error_message}")

    return lines


def _render_modifications(result: RunResult) -> list[str]:
    """Render the files modified section."""
    lines = [
        "---",
        "",
        "## Files Modified",
        "",
    ]

    modified_files = set()
    for cat in result.category_results:
        for fix in cat.fix_attempts:
            if fix.applied and not fix.rolled_back and not fix.suggested_only and fix.file_path:
                modified_files.add(fix.file_path)

    if modified_files:
        for f in sorted(modified_files):
            lines.append(f"- `{f}`")
    else:
        lines.append("_No files were modified during this run._")

    lines.append("")
    return lines


def _render_patches(result: RunResult) -> list[str]:
    """Render the patches section."""
    lines = [
        "---",
        "",
        "## Patches",
        "",
    ]

    patch_entries = []
    for label, patches_dir in (
        ("Rollback snapshot", result.run_dir / "patches"),
        ("Suggested repair", result.run_dir / "suggested-patches"),
    ):
        if not patches_dir.is_dir():
            continue
        for patch in sorted(patches_dir.glob("*.patch")):
            patch_entries.append(f"- {label}: `{patch.name}`")

    if patch_entries:
        lines.extend(patch_entries)
    else:
        lines.append("_No patches created._")

    lines.append("")
    return lines


def _build_json_report(result: RunResult) -> dict:
    """Build the JSON report structure."""
    return {
        "timestamp": result.timestamp,
        "branch": result.branch,
        "run_dir": str(result.run_dir),
        "summary": {
            "passed": result.passed_count,
            "failed": result.failed_count,
            "uncertain": result.uncertain_count,
            "skipped": result.skipped_count,
            "repaired": sum(
                1 for c in result.category_results
                if c.repair_outcome == "applied_passed"
            ),
            "total_duration_seconds": round(result.total_duration_seconds, 1),
        },
        "categories": [
            {
                "name": cat.name,
                "status": cat.status,
                "priority": cat.priority,
                "risk_level": cat.risk_level,
                "description": cat.description,
                "affected_surface": cat.affected_surface,
                "success_criteria": cat.success_criteria,
                "plan_reason": cat.plan_reason,
                "repair_outcome": cat.repair_outcome,
                "next_action": _next_action_for_category(cat),
                "error_message": cat.error_message,
                "test_results": [
                    {
                        "command": t.command,
                        "status": t.status,
                        "exit_code": t.exit_code,
                        "duration_seconds": round(t.duration_seconds, 1),
                    }
                    for t in cat.test_results
                ],
                "fix_attempts": [
                    {
                        "file_path": f.file_path,
                        "explanation": f.explanation,
                        "confidence": f.confidence,
                        "diagnosis": f.diagnosis,
                        "applied": f.applied,
                        "passed": f.passed,
                        "rolled_back": f.rolled_back,
                        "suggested_only": f.suggested_only,
                        "suggested_patch_path": f.suggested_patch_path,
                        "rejected_reason": f.rejected_reason,
                    }
                    for f in cat.fix_attempts
                ],
            }
            for cat in result.category_results
        ],
    }


def _format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.0f} seconds"
    minutes = int(seconds // 60)
    remaining = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {remaining}s"
    hours = int(minutes // 60)
    minutes = minutes % 60
    return f"{hours}h {minutes}m"


def _next_action_for_category(cat: CategoryResult) -> str:
    """Return the human decision a report reader should make next."""
    if cat.next_action:
        return cat.next_action
    if cat.status == "passed" and cat.repair_outcome == "applied_passed":
        return "Review the unstaged fix before accepting it."
    if cat.status == "passed":
        return "No immediate action."
    if cat.status == "failed":
        return "Inspect the failing test and decide whether to fix manually."
    if cat.status == "skipped":
        return "No action unless this category should be included."
    if cat.repair_outcome == "suggested":
        return "Review the suggested patch and apply it manually if appropriate."
    return "Inspect the failure and repair attempts before changing code."


def _repair_outcome_label(outcome: str) -> str:
    labels = {
        "not_needed": "not needed",
        "disabled": "disabled",
        "suggested": "patch suggested",
        "declined": "no valid fix",
        "applied_failed": "applied then rolled back",
        "applied_passed": "passed after repair",
    }
    return labels.get(outcome, outcome.replace("_", " "))


def _fix_context(fix) -> str:
    parts = []
    if fix.confidence and fix.confidence != "unknown":
        parts.append(f"confidence: {fix.confidence}")
    if fix.diagnosis and fix.diagnosis != "unknown":
        parts.append(f"diagnosis: {fix.diagnosis.replace('_', ' ')}")
    if fix.rolled_back:
        parts.append("rolled back")
    return f" ({', '.join(parts)})" if parts else ""
