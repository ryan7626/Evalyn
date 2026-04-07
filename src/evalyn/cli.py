"""Evalyn CLI — entry point for plan, run, and status commands."""

from pathlib import Path

import click
from rich.console import Console

from evalyn import __version__
from evalyn.config import load_config, ConfigError
from evalyn.state import load_state
from evalyn.llm.claude_provider import ClaudeProvider, ClaudeProviderError
from evalyn.planner.planner import run_planning_phase
from evalyn.runner.runner import run_execution_phase
from evalyn.utils.log import get_console_logger

console = Console()


from evalyn.llm.openrouter_provider import OpenRouterProvider, OpenRouterProviderError
import os
from dotenv import load_dotenv

load_dotenv()

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
def plan(repo: Path):
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
        console.print(
            "\n[bold]Next steps:[/]\n"
            "  1. Review the plan file above\n"
            "  2. Edit categories if needed\n"
            "  3. Change 'Status: PENDING' to 'Status: APPROVED'\n"
            f"  4. Run: [bold cyan]evalyn run --repo {repo}[/]"
        )


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
    console.print(f"\n[bold]Run Complete[/] — {result.run_dir}")
    console.print(
        f"  ✅ Passed:    {result.passed_count}\n"
        f"  ❌ Failed:    {result.failed_count}\n"
        f"  ❓ Uncertain: {result.uncertain_count}\n"
        f"  ⏭️  Skipped:   {result.skipped_count}\n"
        f"  ⏱️  Duration:  {result.total_duration_seconds:.1f}s"
    )
    console.print(
        f"\n  Branch: [bold cyan]{result.branch}[/]\n"
        f"  Report: [bold]{result.run_dir / 'report.md'}[/]"
    )


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

    console.print("\n[bold]Evalyn Agent Status[/]")
    console.print(f"  Repository:     {repo}")
    console.print(f"  Last run:       {state.last_run_timestamp or 'never'}")
    console.print(f"  Last commit:    {state.last_run_commit or 'n/a'}")
    console.print(f"  Last status:    {state.last_run_status or 'n/a'}")
    console.print(f"  Agent branch:   {state.current_branch or 'n/a'}")
    console.print(f"  Total runs:     {len(state.run_history)}")
    console.print()
