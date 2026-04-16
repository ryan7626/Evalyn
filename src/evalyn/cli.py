"""Evalyn CLI — entry point for plan, run, and status commands."""

import os
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evalyn import __version__
from evalyn.config import load_config, ConfigError
from evalyn.llm.claude_provider import ClaudeProvider, ClaudeProviderError
from evalyn.llm.openrouter_provider import OpenRouterProvider, OpenRouterProviderError
from evalyn.planner.planner import run_planning_phase
from evalyn.runner.runner import run_execution_phase
from evalyn.state import load_state
from evalyn.utils.log import get_console_logger

load_dotenv()

console = Console()


def _create_llm(config):
    """Create the LLM provider based on config, then env-var fallback."""
    # Explicit config.llm_provider wins; otherwise use env vars to pick.
    want_openrouter = (
        config.llm_provider == "openrouter"
        or (
            os.environ.get("OPENROUTER_API_KEY")
            and not os.environ.get("ANTHROPIC_API_KEY")
        )
    )

    if want_openrouter:
        try:
            model = config.llm_model
            if "/" not in model:
                model = "openai/gpt-4o-mini"
            console.print(f"[dim italic]Using OpenRouter ({model}) for LLM API calls.[/]")
            return OpenRouterProvider(model=model)
        except OpenRouterProviderError as e:
            console.print(f"[bold red]OpenRouter error:[/] {e}")
            raise SystemExit(1)

    try:
        return ClaudeProvider(model=config.llm_model)
    except ClaudeProviderError as e:
        console.print(f"[bold red]LLM error:[/] {e}")
        raise SystemExit(1)


def validate_repo(repo_path: Path) -> Path:
    """Validate that the given path is a repo with a CLAUDE.md at root.

    Accepts CLAUDE.md, claude.md, or Claude.md to work on case-sensitive
    Linux filesystems (e.g. Jetson Nano).
    """
    repo = repo_path.resolve()
    if not repo.is_dir():
        raise click.BadParameter(f"Directory does not exist: {repo}")
    for name in ("CLAUDE.md", "claude.md", "Claude.md"):
        if (repo / name).is_file():
            return repo
    raise click.BadParameter(
        f"No CLAUDE.md found at repo root: {repo}\n"
        "Evalyn requires a CLAUDE.md context file in the target repository."
    )


@click.group()
@click.version_option(version=__version__, prog_name="evalyn")
def main():
    """Evalyn — Overnight autonomous codebase testing and repair agent."""
    pass


@main.command()
@click.option(
    "--repo",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=".",
    help="Path to the target repository (default: current directory).",
)
@click.pass_context
def plan(ctx, repo: Path):
    """Analyze changes and generate a test plan for approval."""
    repo = validate_repo(repo)
    logger = get_console_logger("plan")

    try:
        config = load_config(repo)
    except ConfigError as e:
        console.print(f"[bold red]Config error:[/] {e}")
        raise SystemExit(1)

    state = load_state(repo)
    logger.info(f"Target repo: {repo}")
    logger.info(f"Last run commit: {state.last_run_commit or 'none (first run)'}")

    llm = _create_llm(config)

    plan_path = run_planning_phase(config, state, llm)

    if plan_path is None:
        console.print(
            "\n[bold green]No changes detected since last run.[/] "
            "Nothing to plan."
        )
    else:
        console.print(f"\n[bold green]Plan generated:[/] {plan_path}")
        
        while True:
            choice = click.prompt(
                "\n[bold cyan]What would you like to do?[/]\n"
                "  [y] Approve and execute plan now\n"
                "  [e] Edit plan.md in my default editor first\n"
                "  [n] Exit and run later\n"
                "Your choice",
                type=click.Choice(['y', 'e', 'n'], case_sensitive=False),
                default='y',
                show_choices=False
            ).lower()

            if choice == 'e':
                click.edit(filename=str(plan_path))
                continue
            
            elif choice == 'y':
                content = plan_path.read_text(encoding="utf-8")
                if "Status: PENDING" in content:
                    plan_path.write_text(
                        content.replace("Status: PENDING", "Status: APPROVED"),
                        encoding="utf-8"
                    )
                console.print("\n[bold green]Plan approved. Passing to execution...[/]\n")
                ctx.invoke(run, repo=repo, categories=(), dry_run=False)
                break
                
            elif choice == 'n':
                console.print("\n[dim]Run this plan later with:[/]")
                console.print(f"  [bold cyan]evalyn run --repo {repo}[/]")
                break


@main.command()
@click.option(
    "--repo",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=".",
    help="Path to the target repository (default: current directory).",
)
@click.option(
    "--category",
    "categories",
    multiple=True,
    help="Run only the named category (repeatable). Overrides plan directives.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Parse the plan and show what would run without executing.",
)
def run(repo: Path, categories: tuple[str, ...], dry_run: bool):
    """Execute the approved test plan."""
    repo = validate_repo(repo)
    logger = get_console_logger("run")

    try:
        config = load_config(repo)
    except ConfigError as e:
        console.print(f"[bold red]Config error:[/] {e}")
        raise SystemExit(1)

    state = load_state(repo)
    logger.info(f"Target repo: {repo}")

    llm = _create_llm(config)

    category_filter = list(categories) if categories else None
    result = run_execution_phase(
        config, state, llm,
        category_filter=category_filter,
        dry_run=dry_run,
    )

    if result is None:
        if dry_run:
            # dry_run should return a RunResult — if we get None, plan had no categories
            console.print("\n[bold yellow]Dry run complete.[/] No executable categories found.")
            return
        console.print(
            "\n[bold red]Run could not start.[/] "
            "Check the errors above."
        )
        raise SystemExit(1)

    if dry_run:
        from rich.table import Table as RichTable
        preview = RichTable(title="Categories That Would Execute", show_header=True)
        preview.add_column("Category", style="bold")
        preview.add_column("Priority", style="cyan")
        preview.add_column("Note")
        for cat in result.category_results:
            note = cat.error_message or ""
            priority = "SKIP" if "not in" in note or "exceeded" in note else note.replace("Dry run — would execute [", "").rstrip("]")
            preview.add_row(cat.name, priority, "(skipped)" if "skip" in note.lower() or "not in" in note else "✓ would run")
        console.print("\n")
        console.print(preview)
        console.print("\n[bold green]Dry run complete.[/] No tests were executed.")
        return

    # Print summary
    summary_text = (
        f"  ✅ Passed:    {result.passed_count}\n"
        f"  ❌ Failed:    {result.failed_count}\n"
        f"  ❓ Uncertain: {result.uncertain_count}\n"
        f"  ⏭️  Skipped:   {result.skipped_count}\n"
        f"  ⏱️  Duration:  {result.total_duration_seconds:.1f}s\n\n"
        f"  Branch: [bold cyan]{result.branch}[/]\n"
        f"  Report: [bold]{result.run_dir / 'report.md'}[/]"
    )
    console.print(Panel(summary_text, title=f"Run Complete — {result.run_dir}", border_style="green"))


@main.command()
@click.option(
    "--repo",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=".",
    help="Path to the target repository (default: current directory).",
)
def status(repo: Path):
    """Show current agent state and latest run info."""
    repo = validate_repo(repo)

    try:
        config = load_config(repo)
    except ConfigError as e:
        console.print(f"[bold red]Config error:[/] {e}")
        raise SystemExit(1)

    state = load_state(repo)

    table = Table(title="Evalyn Agent Status", show_header=False, box=None)
    table.add_row("Repository:", str(repo))

    # Humanize the raw timestamp string
    ts_raw = state.last_run_timestamp
    ts_display = "never"
    if ts_raw:
        try:
            from datetime import datetime
            dt = datetime.strptime(ts_raw, "%Y-%m-%d_%H%M%S")
            ts_display = dt.strftime("%B %d, %Y at %I:%M %p UTC")
        except ValueError:
            ts_display = ts_raw
    table.add_row("Last run:", ts_display)

    table.add_row("Last commit:", str(state.last_run_commit or 'n/a'))

    status_str = state.last_run_status or 'n/a'
    if status_str in ('completed', 'passed'):
        status_str = f"[green]{status_str}[/]"
    elif status_str == 'completed_with_issues':
        status_str = f"[yellow]{status_str}[/]"
    elif status_str in ('failed', 'aborted'):
        status_str = f"[red]{status_str}[/]"
    elif status_str == 'in_progress':
        status_str = f"[cyan]{status_str}[/]"
    elif status_str == 'recovered':
        status_str = f"[dim]{status_str}[/]"
    table.add_row("Last status:", status_str)

    table.add_row("Agent branch:", str(state.current_branch or 'n/a'))
    table.add_row("Total runs:", str(len(state.run_history)))

    console.print("\n")
    console.print(table)
    console.print("\n")


@main.command()
@click.option(
    "--repo",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=".",
    help="Path to the target repository (default: current directory).",
)
@click.option(
    "--all",
    "clean_all",
    is_flag=True,
    default=False,
    help="Remove all evalyn/* branches and reset agent state.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Skip confirmation prompt.",
)
def clean(repo: Path, clean_all: bool, force: bool):
    """Clean up stale evalyn branches and reset agent state."""
    repo = validate_repo(repo)

    from evalyn.utils.git import (
        list_branches,
        get_current_branch,
        checkout_branch,
        delete_branch,
        reset_hard,
    )
    from evalyn.state import reset_state

    branches = list_branches(repo, "evalyn/*")
    state = load_state(repo)

    if not branches and state.last_run_status not in ("in_progress", "aborted"):
        console.print("\n[bold green]Nothing to clean.[/] No stale branches found.")
        return

    # Show what we found
    console.print("\n[bold]Evalyn Cleanup[/]")
    console.print(f"  Repository: {repo}")
    if branches:
        console.print(f"  Stale branches ({len(branches)}):")
        for b in branches:
            console.print(f"    • {b}")
    if state.last_run_status in ("in_progress", "aborted"):
        console.print(f"  State: [yellow]{state.last_run_status}[/] (will be reset)")

    if not clean_all:
        console.print("\n[dim]Use --all to remove all branches and reset state.[/]")
        return

    # Confirm
    if not force:
        click.confirm("\nProceed with cleanup?", abort=True)

    # Clean branches
    current = get_current_branch(repo)
    deleted = 0
    for b in branches:
        try:
            if current == b:
                reset_hard(repo)
                # Try main, then master, then give up
                checked_out = False
                for default_branch in ("main", "master"):
                    try:
                        checkout_branch(repo, default_branch)
                        current = default_branch
                        checked_out = True
                        break
                    except Exception:
                        continue
                if not checked_out:
                    console.print(f"  [yellow]⚠[/] Could not determine default branch — skipping deletion of {b}")
                    continue
            delete_branch(repo, b, force=True)
            deleted += 1
            console.print(f"  [red]✗[/] Deleted branch: {b}")
        except Exception as e:
            console.print(f"  [yellow]⚠[/] Could not delete {b}: {e}")

    # Reset state
    reset_state(repo)
    console.print(f"\n[bold green]Cleanup complete.[/] Deleted {deleted} branch(es), state reset.")


# ---------------------------------------------------------------------------
# Templates for `evalyn init`
# ---------------------------------------------------------------------------

_CLAUDE_MD_TEMPLATE = """\
# Project Context

> Fill this in — Evalyn reads this file as the source of truth for your repo.

## Overview

<!-- Brief description of the project and its purpose -->

## Stack

<!-- e.g. Next.js, TypeScript, Python, Postgres -->

## Test Commands

<!-- How to run your existing tests. Examples:
- Unit tests: `pytest tests/`
- Frontend:   `npm test`
- Type check: `npx tsc --noEmit`
-->

## Services

<!-- Services that must be running for integration tests. Example:
services:
  - name: backend
    start: npm run dev
    port: 3000
    health_check: http://localhost:3000/health
-->
"""

_CONFIG_YAML_TEMPLATE = """\
# Evalyn per-repo configuration (all settings optional)

# LLM settings
# llm:
#   provider: claude           # claude | openrouter
#   model: claude-3-5-sonnet-20241022

# Execution settings
# execution:
#   max_fix_retries: 3         # fix hypotheses tried per failing test
#   test_timeout_seconds: 300  # per-command timeout
#   max_output_lines: 200      # max output lines sent to LLM

# Services to start before testing
# services:
#   - name: backend
#     start: npm run dev
#     port: 3000
#     health_check: http://localhost:3000/health
"""


@main.command()
@click.option(
    "--repo",
    type=click.Path(file_okay=False, path_type=Path),
    default=".",
    help="Path to the target repository to initialize (default: current directory).",
)
def init(repo: Path):
    """Scaffold a target repository for use with Evalyn.

    Creates CLAUDE.md and .agent/config.yaml if they don't exist.
    """
    repo = repo.resolve()
    if not repo.is_dir():
        console.print(f"[bold red]Error:[/] Directory does not exist: {repo}")
        raise SystemExit(1)

    created: list[str] = []

    # --- CLAUDE.md ---
    existing_md = next(
        (repo / n for n in ("CLAUDE.md", "claude.md", "Claude.md") if (repo / n).is_file()),
        None,
    )
    if existing_md:
        console.print(f"[dim]✓ {existing_md.name} already exists — skipping.[/]")
    else:
        target = repo / "CLAUDE.md"
        target.write_text(_CLAUDE_MD_TEMPLATE, encoding="utf-8")
        created.append("CLAUDE.md")
        console.print(f"[bold green]✓[/] Created [bold]CLAUDE.md[/]")

    # --- .agent/config.yaml ---
    agent_dir = repo / ".agent"
    agent_dir.mkdir(exist_ok=True)
    config_yaml = agent_dir / "config.yaml"
    if config_yaml.is_file():
        console.print("[dim]✓ .agent/config.yaml already exists — skipping.[/]")
    else:
        config_yaml.write_text(_CONFIG_YAML_TEMPLATE, encoding="utf-8")
        created.append(".agent/config.yaml")
        console.print("[bold green]✓[/] Created [bold].agent/config.yaml[/]")

    if created:
        console.print(
            f"\n[bold green]Repository initialized![/] "
            f"Edit [bold]CLAUDE.md[/] to describe your project, then run:\n"
            f"  [bold cyan]evalyn plan --repo {repo}[/]"
        )
    else:
        console.print(
            "\n[dim]Nothing to do — repository is already initialized.[/]\n"
            f"Run [bold cyan]evalyn plan --repo {repo}[/] to generate a test plan."
        )
