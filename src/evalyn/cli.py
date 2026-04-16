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
    """Create the LLM provider based on config."""
    if os.environ.get("OPENROUTER_API_KEY"):
        try:
            # OpenRouter model strings are typically prefixed with the provider
            model = config.llm_model
            if not ("/" in model):
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
    """Validate that the given path is a repo with a CLAUDE.md at root."""
    repo = repo_path.resolve()
    if not repo.is_dir():
        raise click.BadParameter(f"Directory does not exist: {repo}")
    claude_md = repo / "CLAUDE.md"
    if not claude_md.is_file():
        raise click.BadParameter(
            f"No CLAUDE.md found at repo root: {repo}\n"
            "Evalyn requires a CLAUDE.md context file in the target repository."
        )
    return repo


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
            console.print("\n[bold green]Dry run complete.[/] No tests were executed.")
            return
        console.print(
            "\n[bold red]Run could not start.[/] "
            "Check the errors above."
        )
        raise SystemExit(1)

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
    table.add_row("Last run:", str(state.last_run_timestamp or 'never'))
    table.add_row("Last commit:", str(state.last_run_commit or 'n/a'))
    
    status_str = state.last_run_status or 'n/a'
    if status_str == 'passed':
        status_str = f"[green]{status_str}[/]"
    elif status_str == 'failed':
        status_str = f"[red]{status_str}[/]"
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
                checkout_branch(repo, "main")
                current = "main"
            delete_branch(repo, b, force=True)
            deleted += 1
            console.print(f"  [red]✗[/] Deleted branch: {b}")
        except Exception as e:
            console.print(f"  [yellow]⚠[/] Could not delete {b}: {e}")

    # Reset state
    reset_state(repo)
    console.print(f"\n[bold green]Cleanup complete.[/] Deleted {deleted} branch(es), state reset.")
