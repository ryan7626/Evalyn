"""Planner orchestration for Evalyn.

Coordinates the full planning phase: analyze changes, infer categories,
generate a plan markdown file for human review.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from evalyn.config import Config
from evalyn.llm.base import BaseLLM
from evalyn.planner.diff_analyzer import ChangeSet, analyze_changes
from evalyn.planner.category_inference import TestCategory, infer_categories
from evalyn.state import AgentState, save_state
from evalyn.utils.log import get_logger, setup_logging

logger = get_logger("planner")

PLAN_TEMPLATE = """\
# Evalyn Test Plan — {timestamp}

## Changes Detected

- **Commits**: {num_commits} commit(s) since `{since_sha}`
- **Files changed**: {num_files}
- **HEAD**: `{head_sha}`

### Commit Log

{commit_log}

### Changed Files

{files_list}

### Directories Affected

{dirs_list}

---

## Test Categories

{categories_section}

---

## Approval

<!-- To approve this plan, change PENDING to APPROVED below -->

Status: PENDING

---

## User Notes

<!-- Add directives below to control which categories run. -->
<!-- Examples (remove the comment markers to activate): -->
<!-- ONLY RUN UI Enhancements, Photo Upload Functionality -->
<!-- SKIP Spatial Detection Fixes -->
<!-- MAX CATEGORIES 2 -->
<!-- You can also change any category header from [REQUIRED] to [SKIP] above. -->
"""

CATEGORY_TEMPLATE = """\
### [{priority}] {name}

- **Description**: {description}
- **Risk level**: {risk_level}
- **Why this matters**: {reason}
- **Affected surface**: {affected_surface}
- **Relevant files**: {relevant_files}
- **Success criteria**: {success_criteria}
- **Suggested commands**:
{commands_list}
"""


def run_planning_phase(
    config: Config,
    state: AgentState,
    llm: BaseLLM,
) -> Optional[Path]:
    """Execute the full planning phase.

    1. Analyze git changes since last run
    2. Infer test categories via LLM
    3. Generate plan.md in a timestamped run directory
    4. Save updated state

    Args:
        config: Resolved configuration.
        state: Current agent state.
        llm: LLM provider instance.

    Returns:
        Path to the generated plan.md, or None if no changes detected.
    """
    repo = config.repo_path

    # Create timestamped run directory
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    run_dir = config.runs_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Set up file logging for this run
    setup_logging(run_dir)
    logger.info(f"Starting planning phase — run directory: {run_dir}")

    # Step 1: Analyze changes
    changeset = analyze_changes(
        repo,
        since_sha=state.last_run_commit,
        max_diff_lines=config.max_output_lines * 2,
    )

    if changeset.is_empty:
        logger.info("No changes detected since last run. Nothing to plan.")
        return None

    # Step 2: Infer test categories via LLM
    categories = infer_categories(
        llm=llm,
        changeset=changeset,
        claude_md_content=config.claude_md_content,
    )

    if not categories:
        logger.warning("No test categories inferred. Generating plan with notice.")

    # Step 3: Generate plan.md
    plan_path = run_dir / "plan.md"
    plan_content = _render_plan(changeset, categories, timestamp)
    plan_path.write_text(plan_content, encoding="utf-8")

    logger.info(f"Plan generated: {plan_path}")

    # Step 4: Save state (note: we do NOT update last_run_commit here —
    # that happens when the run phase actually executes)
    state.last_run_timestamp = timestamp
    save_state(repo, state)

    return plan_path


def _render_plan(
    changeset: ChangeSet,
    categories: list[TestCategory],
    timestamp: str,
) -> str:
    """Render the plan markdown content."""
    # Commit log
    commit_log = "\n".join(
        f"- `{c.sha[:8]}` {c.message} — {c.author} ({c.timestamp})"
        for c in changeset.commits
    ) or "_No commits found_"

    # Files list
    files_list = "\n".join(
        f"- `{f}`" for f in changeset.files_changed
    ) or "_No files changed_"

    # Directories
    dirs_list = "\n".join(
        f"- `{d}/` ({len(files)} files)"
        for d, files in sorted(changeset.files_by_directory.items())
    ) or "_None_"

    # Categories section
    if categories:
        categories_section = "\n".join(
            _render_category(cat) for cat in categories
        )
    else:
        categories_section = (
            "_No test categories were inferred. "
            "This may indicate the changes don't map to testable areas, "
            "or the LLM could not determine appropriate categories._\n\n"
            "You may manually add categories below before approving."
        )

    return PLAN_TEMPLATE.format(
        timestamp=timestamp,
        num_commits=len(changeset.commits),
        since_sha=changeset.since_sha[:8],
        num_files=len(changeset.files_changed),
        head_sha=changeset.head_sha[:8],
        commit_log=commit_log,
        files_list=files_list,
        dirs_list=dirs_list,
        categories_section=categories_section,
    )


def _render_category(cat: TestCategory) -> str:
    """Render a single test category as markdown."""
    relevant = ", ".join(f"`{f}`" for f in cat.relevant_files) or "_none specified_"

    if cat.suggested_test_commands:
        commands = "\n".join(
            f"  - `{cmd}`" for cmd in cat.suggested_test_commands
        )
    else:
        commands = "  - _No test commands suggested — manual review needed_"

    return CATEGORY_TEMPLATE.format(
        priority=cat.priority.upper(),
        name=cat.name,
        description=cat.description,
        risk_level=cat.risk_level,
        reason=cat.reason or "Directly related to the changed files.",
        affected_surface=cat.affected_surface or "Unspecified",
        relevant_files=relevant,
        success_criteria=cat.success_criteria or "Suggested commands pass without new failures.",
        commands_list=commands,
    )


def check_plan_approved(plan_path: Path) -> bool:
    """Check if a plan.md has been approved.

    Looks for 'Status: APPROVED' in the plan file.
    """
    if not plan_path.is_file():
        return False

    content = plan_path.read_text(encoding="utf-8")
    # Look for the approval marker (case-insensitive)
    for line in content.split("\n"):
        stripped = line.strip().lower()
        if stripped.startswith("status:") and "approved" in stripped:
            return True
    return False
