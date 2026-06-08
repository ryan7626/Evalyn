"""Run phase orchestration for Evalyn.

The core execution loop: validate plan, create branch, run tests per category,
attempt fixes on failure, manage rollback, and trigger reporting.
"""

import signal
import sys
import re
import time
from pathlib import Path
from typing import Optional

from evalyn.config import Config
from evalyn.llm.base import BaseLLM
from evalyn.models import CategoryResult, RunResult
from evalyn.planner.planner import check_plan_approved
from evalyn.runner.fixer import generate_fix, apply_fix, write_suggested_patch
from evalyn.runner.plan_reader import (
    parse_plan_categories,
    parse_user_notes,
    apply_category_filters,
    extract_files_for_category,
    extract_category_metadata,
    recover_stale_state,
)
from evalyn.runner.rollback import RollbackManager
from evalyn.runner.test_executor import run_test_command
from evalyn.runner.test_generator import generate_ephemeral_test
from evalyn.services.manager import ServiceManager
from evalyn.state import AgentState, save_state, get_latest_run_dir
from evalyn.utils.git import (
    create_branch,
    get_head_sha,
    get_current_branch,
    checkout_branch,
    delete_branch,
    reset_hard,
)
from evalyn.utils.log import get_logger, setup_logging

logger = get_logger("runner")


def _sigterm_handler(signum, frame):
    """Convert SIGTERM to KeyboardInterrupt so the rollback logic in
    run_execution_phase fires instead of the process being killed silently."""
    raise KeyboardInterrupt(f"Signal {signum} (SIGTERM) received")


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
        category_filter: Optional list of category names to run (from --category CLI flag).
        dry_run: If True, parse plan and show what would run without executing.

    Returns:
        RunResult with all execution details, or None if plan is not approved.
    """
    start_time = time.monotonic()

    # Find the latest run directory (where the plan should be)
    run_dir = get_latest_run_dir(config.repo_path)
    if run_dir is None:
        logger.error(
            "No run directory found. "
            "Run `evalyn plan --repo <path>` first to generate a test plan."
        )
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
    all_categories = parse_plan_categories(plan_path)
    if not all_categories:
        logger.error("No test categories found in the approved plan.")
        return None
    priority_by_name = {name: priority for name, _, priority in all_categories}

    # Parse user notes for directives
    user_directives = parse_user_notes(plan_path)

    # Apply filtering: CLI --category > user notes > default (all REQUIRED)
    categories, skipped = apply_category_filters(
        all_categories,
        cli_filter=category_filter,
        directives=user_directives,
    )

    if not categories and not skipped:
        logger.error("No executable categories after applying filters.")
        return None

    logger.info(f"Categories to execute: {len(categories)}")
    for name, _, priority in categories:
        logger.info(f"  ▶ [{priority.upper()}] {name}")
    if skipped:
        logger.info(f"Categories skipped: {len(skipped)}")
        for name, reason in skipped:
            logger.info(f"  ⏭ {name} — {reason}")

    timestamp = run_dir.name

    # Dry run: show what would execute and return a summary result
    if dry_run:
        logger.info("Dry run — no branch created, no tests executed.")
        dry_result = RunResult(timestamp=timestamp, branch="(dry-run)", run_dir=run_dir)
        for cat_name, _, priority in categories:
            metadata = extract_category_metadata(plan_path, cat_name)
            dry_result.category_results.append(CategoryResult(
                name=cat_name,
                status="skipped",
                **_category_metadata_kwargs(metadata, priority),
                error_message=f"Dry run — would execute [{priority.upper()}]",
                next_action="Run without --dry-run to execute this category.",
            ))
        for skip_name, reason in skipped:
            metadata = extract_category_metadata(plan_path, skip_name)
            dry_result.category_results.append(CategoryResult(
                name=skip_name,
                status="skipped",
                **_category_metadata_kwargs(
                    metadata,
                    priority_by_name.get(skip_name, metadata.get("priority", "")),
                ),
                error_message=reason,
                next_action="No action unless this category should be included.",
            ))
        return dry_result

    # Register SIGTERM handler so an OS kill triggers the rollback path
    _prev_sigterm = signal.signal(signal.SIGTERM, _sigterm_handler)

    # Create a new branch for this run
    branch_name = f"evalyn/{timestamp}"

    original_branch = get_current_branch(config.repo_path)
    if original_branch == branch_name:
        # We started exactly on the branch we're trying to recreate
        # Fall back to main/master to prevent infinite branching loops
        try:
            checkout_branch(config.repo_path, "main")
            original_branch = "main"
        except Exception:
            checkout_branch(config.repo_path, "master")
            original_branch = "master"

    # Pre-flight: detect and recover from stale state
    recover_stale_state(config.repo_path, state, branch_name, original_branch)

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
    base_url = None
    if config.services:
        logger.info(f"Starting {len(config.services)} configured services")
        svc_manager.start_services(config.services)
        base_url = svc_manager.get_base_url()

    # Execute each category
    run_result = RunResult(
        timestamp=timestamp,
        branch=branch_name,
        run_dir=run_dir,
    )

    # Record skipped categories in the result
    for skip_name, skip_reason in skipped:
        metadata = extract_category_metadata(plan_path, skip_name)
        run_result.category_results.append(CategoryResult(
            name=skip_name,
            status="skipped",
            **_category_metadata_kwargs(
                metadata,
                priority_by_name.get(skip_name, metadata.get("priority", "")),
            ),
            error_message=skip_reason,
            next_action="No action unless this category should be included.",
        ))

    try:
        for cat_name, cat_commands, priority in categories:
            logger.info(f"\n{'='*60}")
            logger.info(f"Category: [{priority.upper()}] {cat_name}")
            logger.info(f"{'='*60}")

            cat_result = _execute_category(
                config=config,
                llm=llm,
                rollback=rollback,
                repo_path=config.repo_path,
                category_name=cat_name,
                commands=cat_commands,
                relevant_files=extract_files_for_category(plan_path, cat_name),
                category_metadata=extract_category_metadata(plan_path, cat_name),
                run_dir=run_dir,
                priority=priority,
                base_url=base_url,
            )
            run_result.category_results.append(cat_result)

            logger.info(f"Category '{cat_name}': {cat_result.status.upper()}")

    except BaseException as e:
        # An unexpected exception or KeyboardInterrupt occurred.
        # Ensure we discard the messy branch to return to target's clean original state.
        logger.warning(
            f"Run aborted due to '{type(e).__name__}'. "
            f"Rolling back strictly to '{original_branch}'..."
        )
        try:
            reset_hard(config.repo_path)
            checkout_branch(config.repo_path, original_branch)
            delete_branch(config.repo_path, branch_name, force=True)
            logger.info(
                f"Safely removed aborted branch '{branch_name}' "
                f"and restored '{original_branch}'."
            )
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

    # Update state — derive status from actual outcomes
    if run_result.failed_count > 0 or run_result.uncertain_count > 0:
        final_status = "completed_with_issues"
    else:
        final_status = "completed"

    state.last_run_status = final_status
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
    # Cap history to last 50 runs to prevent unbounded growth
    state.run_history = state.run_history[-50:]
    save_state(config.repo_path, state)

    logger.info(f"\nExecution complete in {run_result.total_duration_seconds:.1f}s")
    logger.info(
        f"Results: {run_result.passed_count} passed, "
        f"{run_result.failed_count} failed, "
        f"{run_result.uncertain_count} uncertain"
    )

    # Restore original SIGTERM handler now that rollback window has passed
    signal.signal(signal.SIGTERM, _prev_sigterm)

    return run_result


def _execute_category(
    config: Config,
    llm: BaseLLM,
    rollback: RollbackManager,
    repo_path: Path,
    category_name: str,
    commands: list[str],
    relevant_files: list[str],
    category_metadata: dict[str, str],
    run_dir: Path,
    priority: str,
    base_url: str | None = None,
) -> CategoryResult:
    """Execute all test commands for a single category with fix loop."""

    cat_result = CategoryResult(
        name=category_name,
        status="passed",
        **_category_metadata_kwargs(category_metadata, priority),
    )

    if priority == "required" and not commands:
        logger.info(
            f"Required category '{category_name}' lacks test coverage. "
            "Generating ephemeral test suite..."
        )
        try:
            ephemeral_test = generate_ephemeral_test(
                llm=llm,
                category_name=category_name,
                relevant_files=relevant_files,
                claude_md_content=config.claude_md_content,
                run_dir=run_dir,
                base_url=base_url,
            )
            # Create the file
            generated_tests_dir = run_dir / "generated-tests"
            generated_tests_dir.mkdir(parents=True, exist_ok=True)

            # Flatten name to be safe
            safe_filename = (
                ephemeral_test
                .get("filename", f"{category_name.replace(' ', '_')}.test")
                .replace("/", "_")
            )
            test_file_path = generated_tests_dir / safe_filename
            test_file_path.write_text(ephemeral_test.get("content", ""), encoding="utf-8")

            # Log it
            cat_result.generated_tests.append(str(test_file_path))
            logger.info(f"Generated test file: {test_file_path}")

            raw_cmd = ephemeral_test.get(
                "run_command",
                f"echo 'No command provided for {test_file_path}'"
            )
            raw_file = ephemeral_test.get("filename", "")
            if raw_file and raw_file in raw_cmd:
                raw_cmd = raw_cmd.replace(raw_file, str(test_file_path.absolute()))

            # Use the active interpreter so venv packages are available
            if raw_cmd.startswith("python3 "):
                raw_cmd = raw_cmd.replace("python3 ", f"{sys.executable} ", 1)
            elif raw_cmd.startswith("python "):
                raw_cmd = raw_cmd.replace("python ", f"{sys.executable} ", 1)

            commands = [raw_cmd]

        except Exception as e:
            logger.error(f"Failed to generate ephemeral test: {e}")
            cat_result.status = "uncertain"
            cat_result.repair_outcome = "none"
            cat_result.next_action = "Add a concrete test command or inspect why test generation failed."
            cat_result.error_message = (
                f"Missing required test coverage and generation failed: {e}"
            )
            return cat_result

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

        if config.repair_mode == "test_only":
            cat_result.status = "failed"
            cat_result.repair_outcome = "disabled"
            cat_result.next_action = "Inspect the failing test; repair was disabled by config."
            cat_result.error_message = (
                "Repair mode is test_only, so Evalyn did not attempt a fix."
            )
            break

        if config.repair_mode == "suggest_patches":
            _suggest_patch_for_failure(
                config=config,
                llm=llm,
                repo_path=repo_path,
                category_name=category_name,
                command=command,
                test_result=test_result,
                relevant_files=relevant_files,
                run_dir=run_dir,
                cat_result=cat_result,
            )
            break

        fixed = False
        previous_attempts_context = ""
        base_test_result = test_result  # Keep original baseline test output

        # Bounded Autoresearch Loop
        for attempt_num in range(1, config.max_fix_retries + 1):
            logger.info(
                f"Fix attempt hypothesis {attempt_num}/{config.max_fix_retries} "
                f"for '{category_name}'"
            )

            # Snapshot before fix
            rollback.create_snapshot(f"pre_fix_{category_name}_{attempt_num}")

            # Ask LLM for a fix hypothesis
            fix = generate_fix(
                llm=llm,
                repo_path=repo_path,
                category=category_name,
                command=command,
                exit_code=base_test_result.exit_code,
                stdout=base_test_result.stdout,
                stderr=base_test_result.stderr,
                claude_md_content=config.claude_md_content,
                relevant_files=relevant_files,
                previous_attempts_context=previous_attempts_context,
            )
            cat_result.fix_attempts.append(fix)

            if not apply_fix(fix, repo_path):
                logger.warning(f"Fix validation/apply failed: {fix.rejected_reason or fix.explanation}")
                rollback.rollback_last()
                previous_attempts_context += (
                    f"### Attempt {attempt_num} Failed to Apply\n"
                    f"Reason: {fix.rejected_reason}\n\n"
                )
                continue

            # Re-run the test to evaluate hypothesis
            retest_result = run_test_command(
                repo_path=repo_path,
                command=command,
                category=category_name,
                timeout_seconds=config.test_timeout_seconds,
                max_output_lines=config.max_output_lines,
            )
            cat_result.test_results.append(retest_result)

            fix.passed = retest_result.is_success

            if fix.passed:
                logger.info(f"Fix successful on attempt {attempt_num}! Fast-exiting the bounded search.")
                fix.is_best_candidate = True
                cat_result.repair_outcome = "applied_passed"
                cat_result.next_action = "Review the unstaged fix before accepting it."
                fixed = True
                # Leave snapshot/applied fix intact
                break
            else:
                logger.warning(
                    f"Fix hypothesis {attempt_num} did not resolve the failure "
                    f"(Diff size: {fix.diff_size})"
                )
                # Rollback precisely to baseline for a clean next attempt
                rollback.rollback_last()
                fix.rolled_back = True
                cat_result.repair_outcome = "applied_failed"

                # Feedback loop: Store the hypothesis and its specific error
                previous_attempts_context += (
                    f"### Attempt {attempt_num} Failed (Rolled Back)\n"
                    f"You tried the following replacement:\n```\n{fix.replacement_content}\n```\n"
                    f"It resulted in this error:\n```\n{retest_result.failure_summary}\n```\n"
                    f"Do NOT re-suggest this approach.\n\n"
                )

        if not fixed:
            cat_result.status = "uncertain"
            if cat_result.repair_outcome == "none":
                cat_result.repair_outcome = (
                    "declined" if cat_result.fix_attempts else "none"
                )
            cat_result.next_action = "Inspect the failing test and rejected or rolled-back fix attempts."
            cat_result.error_message = (
                f"Failed after {config.max_fix_retries} fix attempts. "
                f"Marked as uncertain for human review."
            )
            logger.warning(
                f"Category '{category_name}' marked UNCERTAIN after "
                f"{config.max_fix_retries} fix attempts"
            )
            # Stop processing further commands in this category — the
            # category is already compromised; running more commands would
            # produce misleading partial results.
            break

    if cat_result.status == "passed" and cat_result.generated_tests:
        logger.info(f"Category '{category_name}' passed. Cleaning up ephemeral tests...")
        for gen_test in cat_result.generated_tests:
            try:
                Path(gen_test).unlink(missing_ok=True)
                logger.info(f"Removed ephemeral test: {gen_test}")
            except Exception as e:
                logger.warning(f"Failed to remove ephemeral test {gen_test}: {e}")

    if cat_result.status == "passed" and cat_result.repair_outcome == "none":
        cat_result.repair_outcome = "not_needed"

    return cat_result


def _suggest_patch_for_failure(
    config: Config,
    llm: BaseLLM,
    repo_path: Path,
    category_name: str,
    command: str,
    test_result,
    relevant_files: list[str],
    run_dir: Path,
    cat_result: CategoryResult,
) -> None:
    """Generate patch suggestions without changing the working tree."""
    previous_attempts_context = ""
    wrote_patch = False

    for attempt_num in range(1, config.max_fix_retries + 1):
        logger.info(
            f"Patch suggestion hypothesis {attempt_num}/{config.max_fix_retries} "
            f"for '{category_name}'"
        )
        fix = generate_fix(
            llm=llm,
            repo_path=repo_path,
            category=category_name,
            command=command,
            exit_code=test_result.exit_code,
            stdout=test_result.stdout,
            stderr=test_result.stderr,
            claude_md_content=config.claude_md_content,
            relevant_files=relevant_files,
            previous_attempts_context=previous_attempts_context,
        )
        cat_result.fix_attempts.append(fix)

        if fix.rejected_reason:
            previous_attempts_context += (
                f"### Attempt {attempt_num} Failed to Produce a Valid Patch\n"
                f"Reason: {fix.rejected_reason}\n\n"
            )
            continue

        wrote_patch = write_suggested_patch(
            fix,
            repo_path,
            run_dir / "suggested-patches",
            f"{category_name}_{attempt_num}",
        )
        if wrote_patch:
            break

        previous_attempts_context += (
            f"### Attempt {attempt_num} Failed to Write Patch\n"
            f"Reason: {fix.rejected_reason}\n\n"
        )

    cat_result.status = "uncertain"
    cat_result.repair_outcome = "suggested" if wrote_patch else "declined"
    if wrote_patch:
        cat_result.next_action = "Review the suggested patch and apply it manually if appropriate."
        cat_result.error_message = (
            "Repair mode is suggest_patches, so Evalyn wrote a patch suggestion "
            "without changing the working tree."
        )
    else:
        cat_result.next_action = "Inspect the failure; Evalyn could not produce a valid patch suggestion."
        cat_result.error_message = (
            "Repair mode is suggest_patches, but no valid patch suggestion was produced."
        )


def _category_metadata_kwargs(
    metadata: dict[str, str] | None,
    priority: str = "",
) -> dict[str, str]:
    """Build CategoryResult metadata kwargs from plan metadata."""
    metadata = metadata or {}
    return {
        "priority": priority or metadata.get("priority", ""),
        "risk_level": metadata.get("risk_level", ""),
        "description": metadata.get("description", ""),
        "affected_surface": metadata.get("affected_surface", ""),
        "success_criteria": metadata.get("success_criteria", ""),
        "plan_reason": metadata.get("plan_reason", ""),
    }
