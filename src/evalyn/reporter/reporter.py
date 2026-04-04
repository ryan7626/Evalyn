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
        f"- ⏱️ Total runtime: {_format_duration(result.total_duration_seconds)}",
        f"- 🌿 Branch: `{result.branch}`",
        "",
        "---",
        "",
        "## Results by Category",
        "",
    ]

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
    icon = {"passed": "✅", "failed": "❌", "uncertain": "❓"}.get(cat.status, "❓")
    status_label = cat.status.upper()

    lines = [
        f"### {icon} {cat.name} ({status_label})",
        "",
    ]

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
            if fix.applied:
                lines.append(
                    f"  {i}. Applied fix to `{fix.file_path}`: {fix.explanation}"
                )
            elif fix.rejected_reason:
                lines.append(
                    f"  {i}. Rejected ({fix.rejected_reason}): {fix.explanation}"
                )
            else:
                lines.append(f"  {i}. No fix: {fix.explanation}")

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
            if fix.applied and fix.file_path:
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

    patches_dir = result.run_dir / "patches"
    if patches_dir.is_dir():
        patches = sorted(patches_dir.glob("*.patch"))
        if patches:
            for p in patches:
                lines.append(f"- `{p.name}`")
        else:
            lines.append("_No patches created._")
    else:
        lines.append("_No patches directory._")

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
            "total_duration_seconds": round(result.total_duration_seconds, 1),
        },
        "categories": [
            {
                "name": cat.name,
                "status": cat.status,
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
                        "applied": f.applied,
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
