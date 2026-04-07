"""Run phase orchestration for Evalyn.

The core execution loop: validate plan, create branch, run tests per category,
attempt fixes on failure, manage rollback, and trigger reporting.
"""

import re
import time
from pathlib import Path
from typing import Optional

from evalyn.config import Config
from evalyn.llm.base import BaseLLM
from evalyn.models import CategoryResult, RunResult
from evalyn.planner.planner import check_plan_approved
from evalyn.runner.fixer import attempt_fix
from evalyn.runner.rollback import RollbackManager
from evalyn.runner.test_executor import run_test_command
from evalyn.services.manager import ServiceManager
from evalyn.state import AgentState, save_state, get_latest_run_dir
from evalyn.utils.git import (
    create_branch,
    get_head_sha,
    get_current_branch,
    checkout_branch,
    delete_branch,
    branch_exists,
    reset_hard,
)
from evalyn.utils.log import get_logger, setup_logging

logger = get_logger("runner")


def run_execution_phase(
    config: Config,
    state: AgentState,
    llm: BaseLLM,
    *,
    category_filter: Optional[list[str]] = None,
    dry_run: bool = False,
) -> Optional[RunResult]:
    """Execute the full run phase.

    1. Find and validate the approved plan
    2. Create a new branch
    3. Execute tests per category with fix loop
    4. Update state and produce results

    Args:
        config: Resolved configuration.
        state: Current agent state.
        llm: LLM provider instance.

    Returns:
        RunResult with all execution details, or None if plan is not approved.
    """
    start_time = time.monotonic()

    # Find the latest run directory (where the plan should be)
    run_dir = get_latest_run_dir(config.repo_path)
    if run_dir is None:
        logger.error("No run directory found. Run 'evalyn plan' first.")
        return None

    plan_path = run_dir / "plan.md"
    if not plan_path.is_file():
        logger.error(f"No plan.md found in {run_dir}. Run 'evalyn plan' first.")
        return None

    # Check approval
    if not check_plan_approved(plan_path):
        logger.error(
            "Plan is not approved. Edit the plan.md and change "
            "'Status: PENDING' to 'Status: APPROVED' before running."
        )
        return None

    # Set up logging for this run
    setup_logging(run_dir)
    logger.info("Plan approved — starting execution phase")

    # Parse categories from the approved plan
    all_categories = _parse_plan_categories(plan_path)
    if not all_categories:
        logger.error("No test categories found in the approved plan.")
        return None

    # Parse user notes for directives
    user_directives = _parse_user_notes(plan_path)

    # Apply filtering: CLI --category > user notes > default (all REQUIRED)
    categories, skipped = _apply_category_filters(
        all_categories,
        cli_filter=category_filter,
        directives=user_directives,
    )

    if not categories and not skipped:
        logger.error("No executable categories after applying filters.")
        return None

    logger.info(f"Categories to execute: {len(categories)}")
    for name, _ in categories:
        logger.info(f"  ▶ {name}")
    if skipped:
        logger.info(f"Categories skipped: {len(skipped)}")
        for name, reason in skipped:
            logger.info(f"  ⏭ {name} — {reason}")

    # Dry run: print what would happen and exit
    if dry_run:
        logger.info("Dry run — no branch created, no tests executed.")
        return None

    # Create a new branch for this run
    timestamp = run_dir.name
    branch_name = f"evalyn/{timestamp}"

    original_branch = get_current_branch(config.repo_path)

    # Pre-flight: detect and recover from stale state
    _recover_stale_state(config.repo_path, state, branch_name, original_branch)

    create_branch(config.repo_path, branch_name)
    logger.info(f"Created branch: {branch_name}")

    # Update state to in-progress
    state.last_run_status = "in_progress"
    state.current_branch = branch_name
    state.last_run_commit = get_head_sha(config.repo_path)
    save_state(config.repo_path, state)

    # Initialize rollback manager
    rollback = RollbackManager(run_dir, config.repo_path)

    # Start required services
    svc_manager = ServiceManager(config.repo_path, run_dir)
    if config.services:
        logger.info(f"Starting {len(config.services)} configured services")
        svc_manager.start_services(config.services)
    
    # Execute each category
    run_result = RunResult(
        timestamp=timestamp,
        branch=branch_name,
        run_dir=run_dir,
    )

    # Record skipped categories in the result
    for skip_name, skip_reason in skipped:
        run_result.category_results.append(CategoryResult(
            name=skip_name,
            status="skipped",
            error_message=skip_reason,
        ))

    try:
        for cat_name, cat_commands in categories:
            logger.info(f"\n{'='*60}")
            logger.info(f"Category: {cat_name}")
            logger.info(f"{'='*60}")

            cat_result = _execute_category(
                config=config,
                llm=llm,
                rollback=rollback,
                repo_path=config.repo_path,
                category_name=cat_name,
                commands=cat_commands,
                relevant_files=_extract_files_for_category(plan_path, cat_name),
            )
            run_result.category_results.append(cat_result)

            logger.info(f"Category '{cat_name}': {cat_result.status.upper()}")

    except BaseException as e:
        # An unexpected exception or KeyboardInterrupt occurred.
        # Ensure we discard the messy branch to return to target's clean original state.
        logger.warning(f"Run aborted due to '{type(e).__name__}'. Rolling back strictly to '{original_branch}'...")
        try:
            reset_hard(config.repo_path)
            checkout_branch(config.repo_path, original_branch)
            delete_branch(config.repo_path, branch_name, force=True)
            logger.info(f"Safely removed aborted branch '{branch_name}' and restored '{original_branch}'.")
        except Exception as rollback_err:
            logger.error(f"Failed to cleanly rollback branch during abort: {rollback_err}")
            
        state.last_run_status = "aborted"
        save_state(config.repo_path, state)
        raise e

    finally:
        # Always stop services we started
        if svc_manager.started_count > 0:
            logger.info("Stopping agent-started services")
            svc_manager.stop_services()

    # Finalize
    run_result.total_duration_seconds = time.monotonic() - start_time

    # Generate reports (lazy import to avoid circular dep)
    try:
        from evalyn.reporter.reporter import generate_reports
        md_path, json_path = generate_reports(run_result)
        logger.info(f"Reports generated: {md_path}, {json_path}")
    except Exception as e:
        logger.error(f"Report generation failed: {e}")

    # Update state
    state.last_run_status = "completed"
    state.last_run_timestamp = timestamp
    state.run_history.append({
        "timestamp": timestamp,
        "branch": branch_name,
        "passed": run_result.passed_count,
        "failed": run_result.failed_count,
        "uncertain": run_result.uncertain_count,
        "skipped": run_result.skipped_count,
        "duration_seconds": round(run_result.total_duration_seconds, 1),
    })
    save_state(config.repo_path, state)

    logger.info(f"\nExecution complete in {run_result.total_duration_seconds:.1f}s")
    logger.info(
        f"Results: {run_result.passed_count} passed, "
        f"{run_result.failed_count} failed, "
        f"{run_result.uncertain_count} uncertain"
    )

    return run_result


def _execute_category(
    config: Config,
    llm: BaseLLM,
    rollback: RollbackManager,
    repo_path: Path,
    category_name: str,
    commands: list[str],
    relevant_files: list[str],
) -> CategoryResult:
    """Execute all test commands for a single category with fix loop."""

    cat_result = CategoryResult(name=category_name, status="passed")

    for command in commands:
        # Run the test
        test_result = run_test_command(
            repo_path=repo_path,
            command=command,
            category=category_name,
            timeout_seconds=config.test_timeout_seconds,
            max_output_lines=config.max_output_lines,
        )
        cat_result.test_results.append(test_result)

        if test_result.is_success:
            continue

        # Test failed — enter fix loop
        logger.warning(f"Test failed: {command}")

        fixed = False
        for attempt_num in range(1, config.max_fix_retries + 1):
            logger.info(
                f"Fix attempt {attempt_num}/{config.max_fix_retries} "
                f"for '{category_name}'"
            )

            # Snapshot before fix
            rollback.create_snapshot(f"pre_fix_{category_name}_{attempt_num}")

            # Ask LLM for a fix
            fix = attempt_fix(
                llm=llm,
                repo_path=repo_path,
                category=category_name,
                command=command,
                exit_code=test_result.exit_code,
                stdout=test_result.stdout,
                stderr=test_result.stderr,
                claude_md_content=config.claude_md_content,
                relevant_files=relevant_files,
            )
            cat_result.fix_attempts.append(fix)

            if not fix.applied:
                logger.warning(f"Fix not applied: {fix.rejected_reason or fix.explanation}")
                rollback.rollback_last()
                continue

            # Re-run the test
            retest_result = run_test_command(
                repo_path=repo_path,
                command=command,
                category=category_name,
                timeout_seconds=config.test_timeout_seconds,
                max_output_lines=config.max_output_lines,
            )
            cat_result.test_results.append(retest_result)

            if retest_result.is_success:
                logger.info(f"Fix successful on attempt {attempt_num}!")
                fixed = True
                break
            else:
                logger.warning(
                    f"Fix attempt {attempt_num} did not resolve the failure"
                )
                # Rollback the failed fix
                rollback.rollback_last()

                # Update test_result for the next iteration's context
                test_result = retest_result

        if not fixed:
            cat_result.status = "uncertain"
            cat_result.error_message = (
                f"Failed after {config.max_fix_retries} fix attempts. "
                f"Marked as uncertain for human review."
            )
            logger.warning(
                f"Category '{category_name}' marked UNCERTAIN after "
                f"{config.max_fix_retries} fix attempts"
            )

    return cat_result


def _parse_plan_categories(plan_path: Path) -> list[tuple[str, list[str], str]]:
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

        # Find commands (lines starting with ` - ` that contain backtick-wrapped commands)
        commands = []
        for line in section.split("\n"):
            # Match lines like: - `npm test -- --grep auth`
            cmd_match = re.search(r"`([^`]+)`", line)
            if cmd_match and line.strip().startswith("-"):
                cmd = cmd_match.group(1)
                # Skip if it looks like a file path rather than a command
                if not cmd.startswith("/") and not cmd.startswith("src/"):
                    commands.append(cmd)

        categories.append((name, commands, priority))

    return categories


def _parse_user_notes(plan_path: Path) -> dict:
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


def _apply_category_filters(
    all_categories: list[tuple[str, list[str], str]],
    *,
    cli_filter: Optional[list[str]] = None,
    directives: Optional[dict] = None,
) -> tuple[list[tuple[str, list[str]]], list[tuple[str, str]]]:
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
          - categories to execute: list of (name, commands)
          - skipped categories: list of (name, reason)
    """
    if directives is None:
        directives = {"only_run": [], "skip": [], "max_categories": None}

    to_execute: list[tuple[str, list[str]]] = []
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
                to_execute.append((name, commands))
            else:
                skipped.append((name, "not in --category filter"))
            continue

        # 3. ONLY RUN directive — if present, only named categories run
        if only_run_names:
            if name_lower in only_run_names:
                to_execute.append((name, commands))
            else:
                skipped.append((name, "not in ONLY RUN directive"))
            continue

        # 4. SKIP directive — skip named categories
        if name_lower in skip_names:
            skipped.append((name, "SKIP directive in user notes"))
            continue

        # 5. Default: only execute REQUIRED categories
        if priority == "required":
            to_execute.append((name, commands))
        else:
            skipped.append((name, f"[{priority.upper()}] — not required"))

    # 6. Apply MAX CATEGORIES limit
    max_cats = directives.get("max_categories")
    if max_cats is not None and len(to_execute) > max_cats:
        overflow = to_execute[max_cats:]
        to_execute = to_execute[:max_cats]
        for name, _ in overflow:
            skipped.append((name, f"exceeded MAX CATEGORIES {max_cats}"))

    return to_execute, skipped


def _extract_files_for_category(plan_path: Path, category_name: str) -> list[str]:
    """Extract relevant file paths for a category from the plan.

    Looks for the 'Relevant files' line in the category section.
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
        if "relevant files" in line.lower() or "files" in line.lower():
            # Extract backtick-wrapped file paths
            file_matches = re.findall(r"`([^`]+\.\w+)`", line)
            files.extend(file_matches)

    return files
