"""Plan file parsing helpers for the Evalyn runner.

Extracted from runner.py to keep orchestration separate from plan-file
parsing, filtering, and stale-state recovery.
"""

import re
from pathlib import Path
from typing import Optional

from evalyn.state import AgentState, save_state
from evalyn.utils.git import (
    branch_exists,
    checkout_branch,
    delete_branch,
    get_current_branch,
    reset_hard,
)
from evalyn.utils.log import get_logger

logger = get_logger("runner.plan_reader")


def parse_plan_categories(plan_path: Path) -> list[tuple[str, list[str], str]]:
    """Parse test categories and their commands from a plan.md file.

    Returns list of (category_name, [commands], priority).
    Priority is one of: 'required', 'optional', 'skip'.
    """
    content = plan_path.read_text(encoding="utf-8")
    categories = []

    # Match category headers: ### [REQUIRED|OPTIONAL|SKIP] Category Name
    category_pattern = re.compile(
        r"^###\s+\[(REQUIRED|OPTIONAL|SKIP)\]\s+(.+)$",
        re.MULTILINE | re.IGNORECASE,
    )

    matches = list(category_pattern.finditer(content))

    for i, match in enumerate(matches):
        priority = match.group(1).lower()
        name = match.group(2).strip()

        # Extract section content between this match and the next
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        section = content[start:end]

        # Find commands (lines starting with `- ` that contain backtick-wrapped commands)
        commands = []
        for line in section.split("\n"):
            if "relevant files" in line.lower() or "description" in line.lower():
                continue
            # Match lines like: - `npm test -- --grep auth`
            cmd_match = re.search(r"`([^`]+)`", line)
            if cmd_match and line.strip().startswith("-"):
                cmd = cmd_match.group(1)
                # Skip if it looks like a file path rather than a command
                if not cmd.startswith("/") and not cmd.startswith("src/"):
                    commands.append(cmd)

        categories.append((name, commands, priority))

    return categories


def parse_user_notes(plan_path: Path) -> dict:
    """Parse the User Notes section for actionable directives.

    Supported directives:
      ONLY RUN <category>[, <category2>, ...]
      SKIP <category>
      MAX CATEGORIES <N>

    Returns a dict with keys: 'only_run', 'skip', 'max_categories'.
    """
    content = plan_path.read_text(encoding="utf-8")
    directives: dict = {
        "only_run": [],
        "skip": [],
        "max_categories": None,
    }

    # Find content after "Status: APPROVED" or "## User Notes"
    notes_start = None
    for marker in ["## User Notes", "Status: APPROVED"]:
        idx = content.lower().find(marker.lower())
        if idx != -1:
            notes_start = idx + len(marker)
            break

    if notes_start is None:
        return directives

    notes_section = content[notes_start:]

    for line in notes_section.split("\n"):
        stripped = line.strip()
        # Skip HTML comments and empty lines
        if not stripped or stripped.startswith("<!--"):
            continue

        upper = stripped.upper()

        # ONLY RUN <name>, <name2>
        if upper.startswith("ONLY RUN "):
            names_str = stripped[len("ONLY RUN "):]
            names = [n.strip() for n in names_str.split(",") if n.strip()]
            directives["only_run"].extend(names)
            logger.info(f"User directive: ONLY RUN {names}")

        # SKIP <name>
        elif upper.startswith("SKIP "):
            name = stripped[len("SKIP "):].strip()
            if name:
                directives["skip"].append(name)
                logger.info(f"User directive: SKIP {name}")

        # MAX CATEGORIES <N>
        elif upper.startswith("MAX CATEGORIES "):
            try:
                n = int(stripped[len("MAX CATEGORIES "):])
                directives["max_categories"] = n
                logger.info(f"User directive: MAX CATEGORIES {n}")
            except ValueError:
                logger.warning(f"Invalid MAX CATEGORIES value: {stripped}")

    return directives


def apply_category_filters(
    all_categories: list[tuple[str, list[str], str]],
    *,
    cli_filter: Optional[list[str]] = None,
    directives: Optional[dict] = None,
) -> tuple[list[tuple[str, list[str], str]], list[tuple[str, str]]]:
    """Filter categories based on CLI flags, user notes, and plan markers.

    Priority order:
      1. CLI --category filter (highest priority)
      2. User notes directives (ONLY RUN, SKIP, MAX CATEGORIES)
      3. Plan markers ([SKIP] header, [OPTIONAL] excluded by default)

    Args:
        all_categories: Parsed categories as (name, commands, priority).
        cli_filter: Category names from --category CLI flag.
        directives: Parsed user notes directives dict.

    Returns:
        Tuple of:
          - categories to execute: list of (name, commands, priority)
          - skipped categories: list of (name, reason)
    """
    if directives is None:
        directives = {"only_run": [], "skip": [], "max_categories": None}

    to_execute: list[tuple[str, list[str], str]] = []
    skipped: list[tuple[str, str]] = []

    # Normalize names for case-insensitive matching
    cli_names = [n.lower() for n in (cli_filter or [])]
    only_run_names = [n.lower() for n in directives.get("only_run", [])]
    skip_names = [n.lower() for n in directives.get("skip", [])]

    for name, commands, priority in all_categories:
        name_lower = name.lower()

        # 1. [SKIP] in header — always skip
        if priority == "skip":
            skipped.append((name, "[SKIP] marker in plan"))
            continue

        # 2. CLI --category filter takes highest priority
        if cli_names:
            if name_lower in cli_names:
                to_execute.append((name, commands, priority))
            else:
                skipped.append((name, "not in --category filter"))
            continue

        # 3. ONLY RUN directive — if present, only named categories run
        if only_run_names:
            if name_lower in only_run_names:
                to_execute.append((name, commands, priority))
            else:
                skipped.append((name, "not in ONLY RUN directive"))
            continue

        # 4. SKIP directive — skip named categories
        if name_lower in skip_names:
            skipped.append((name, "SKIP directive in user notes"))
            continue

        # 5. Default: only execute REQUIRED categories
        if priority == "required":
            to_execute.append((name, commands, priority))
        else:
            skipped.append((name, f"[{priority.upper()}] — not required"))

    # 6. Apply MAX CATEGORIES limit
    max_cats = directives.get("max_categories")
    if max_cats is not None and len(to_execute) > max_cats:
        overflow = to_execute[max_cats:]
        to_execute = to_execute[:max_cats]
        for name, _, _ in overflow:
            skipped.append((name, f"exceeded MAX CATEGORIES {max_cats}"))

    return to_execute, skipped


def extract_files_for_category(plan_path: Path, category_name: str) -> list[str]:
    """Extract relevant file paths for a category from the plan.

    Looks for the 'Relevant files' line in the category section.
    Only matches lines that explicitly contain 'relevant files', not any
    line with the word 'files' (which would be too greedy).
    """
    content = plan_path.read_text(encoding="utf-8")
    files = []

    # Find the category section
    pattern = re.compile(
        rf"###\s+\[(?:REQUIRED|OPTIONAL|SKIP)\]\s+{re.escape(category_name)}",
        re.MULTILINE | re.IGNORECASE,
    )
    match = pattern.search(content)
    if not match:
        return files

    # Look for relevant files line in the section
    section_start = match.end()
    section = content[section_start:section_start + 1000]  # Look ahead

    for line in section.split("\n"):
        if "relevant files" in line.lower():  # BUG-7 fix: no longer matches any line with "files"
            # Extract backtick-wrapped file paths
            file_matches = re.findall(r"`([^`]+\.\w+)`", line)
            files.extend(file_matches)

    return files


def recover_stale_state(
    repo_path: Path,
    state: AgentState,
    target_branch: str,
    fallback_branch: str,
) -> None:
    """Detect and recover from stale state before creating a new branch.

    Handles two cases:
      1. state.json stuck at 'in_progress' from a killed process
      2. A branch named `target_branch` already exists (from a previous run)

    Args:
        repo_path: Target repository path.
        state: Current agent state (will be mutated and saved if recovered).
        target_branch: The branch name we're about to create.
        fallback_branch: The branch to checkout if we need to clean up.
    """
    # Case 1: Orphaned in_progress state
    if state.last_run_status == "in_progress":
        logger.warning(
            f"Previous run was interrupted (status: in_progress). "
            f"Cleaning up stale branch '{state.current_branch}'..."
        )
        stale_branch = state.current_branch
        if stale_branch and branch_exists(repo_path, stale_branch):
            current = get_current_branch(repo_path)
            if current == stale_branch:
                reset_hard(repo_path)
                checkout_branch(repo_path, fallback_branch)
            delete_branch(repo_path, stale_branch, force=True)
            logger.info(f"Deleted orphaned branch '{stale_branch}'")

        state.last_run_status = "recovered"
        state.current_branch = None
        save_state(repo_path, state)

    # Case 2: Target branch already exists (stale from a completed/aborted run)
    if branch_exists(repo_path, target_branch):
        logger.warning(
            f"Stale branch '{target_branch}' already exists. Removing it..."
        )
        current = get_current_branch(repo_path)
        if current == target_branch:
            reset_hard(repo_path)
            checkout_branch(repo_path, fallback_branch)
        delete_branch(repo_path, target_branch, force=True)
        logger.info(f"Removed stale branch '{target_branch}'")
