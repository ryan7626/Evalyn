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
from evalyn.utils.git import create_branch, get_head_sha, get_current_branch
from evalyn.utils.log import get_logger, setup_logging

logger = get_logger("runner")


def run_execution_phase(
    config: Config,
    state: AgentState,
    llm: BaseLLM,
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
    categories = _parse_plan_categories(plan_path)
    if not categories:
        logger.error("No test categories found in the approved plan.")
        return None

    logger.info(f"Found {len(categories)} categories to execute")

    # Create a new branch for this run
    timestamp = run_dir.name
    branch_name = f"evalyn/{timestamp}"

    original_branch = get_current_branch(config.repo_path)
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


def _parse_plan_categories(plan_path: Path) -> list[tuple[str, list[str]]]:
    """Parse test categories and their commands from a plan.md file.

    Returns list of (category_name, [commands]).
    """
    content = plan_path.read_text(encoding="utf-8")
    categories = []

    # Match category headers: ### [REQUIRED] Category Name or ### [OPTIONAL] Category Name
    category_pattern = re.compile(
        r"^###\s+\[(REQUIRED|OPTIONAL)\]\s+(.+)$",
        re.MULTILINE,
    )

    matches = list(category_pattern.finditer(content))

    for i, match in enumerate(matches):
        priority = match.group(1)
        name = match.group(2).strip()

        # Only execute REQUIRED categories
        if priority.upper() != "REQUIRED":
            continue

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

        if commands:
            categories.append((name, commands))

    return categories


def _extract_files_for_category(plan_path: Path, category_name: str) -> list[str]:
    """Extract relevant file paths for a category from the plan.

    Looks for the 'Relevant files' line in the category section.
    """
    content = plan_path.read_text(encoding="utf-8")
    files = []

    # Find the category section
    pattern = re.compile(
        rf"###\s+\[(?:REQUIRED|OPTIONAL)\]\s+{re.escape(category_name)}",
        re.MULTILINE,
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
