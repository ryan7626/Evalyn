"""Evalyn v0.2 command-line interface."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from evalyn import __version__
from evalyn.adapters.registry import default_registry
from evalyn.execution import RuntimeUnavailableError, resolve_container_engine
from evalyn.llm import create_provider
from evalyn.operations import (
    OperationError,
    reproduce_saved_finding,
    triage_saved_finding,
    validate_finding_id,
)
from evalyn.revisions import RevisionError, validate_git_repository
from evalyn.run_store import RunStore
from evalyn.settings import SettingsError, load_settings, render_default_config
from evalyn.workflow import CheckError, CheckOptions, inspect_check, run_check

load_dotenv()


@dataclass(frozen=True)
class CLIContext:
    repo: Path
    config: Path | None
    state_root: Path | None
    console: Console


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--repo", type=click.Path(path_type=Path, file_okay=False), default=".", show_default=True
)
@click.option("--config", type=click.Path(path_type=Path, dir_okay=False))
@click.option(
    "--state-root",
    type=click.Path(path_type=Path, file_okay=False),
    envvar="EVALYN_STATE_HOME",
    help="External Evalyn state/artifact root.",
)
@click.option("--no-color", is_flag=True)
@click.option("-v", "--verbose", count=True)
@click.version_option(version=__version__, prog_name="evalyn")
@click.pass_context
def main(
    ctx: click.Context,
    repo: Path,
    config: Path | None,
    state_root: Path | None,
    no_color: bool,
    verbose: int,
) -> None:
    """Generate disposable tests for a Git diff and surface stable regressions."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING)
    ctx.obj = CLIContext(
        repo=repo.expanduser(),
        config=config.expanduser() if config else None,
        state_root=state_root.expanduser() if state_root else None,
        console=Console(color_system=None if no_color else "auto"),
    )


@main.command()
@click.option("--force", is_flag=True)
@click.option("--non-interactive", is_flag=True)
@click.pass_obj
def init(context: CLIContext, force: bool, non_interactive: bool) -> None:
    """Detect supported test frameworks and create .evalyn.yml."""
    del non_interactive
    try:
        repo = validate_git_repository(context.repo)
    except RevisionError as exc:
        _usage_error(context, str(exc))
    path = repo / ".evalyn.yml"
    if path.exists() and not force:
        raise click.ClickException(f"{path} already exists; use --force to replace it")
    adapters = [adapter.name for adapter in default_registry().detect(repo)]
    try:
        RunStore.write_text(path, render_default_config(adapters), mode=0o644)
    except (OSError, ValueError) as exc:
        raise click.ClickException(f"Could not write {path}: {exc}") from exc
    context.console.print(f"[green]Created[/] {path}")
    if not adapters:
        context.console.print("[yellow]No pytest or Vitest project detected.[/]")


@main.command()
@click.option(
    "--format", "output_format", type=click.Choice(["console", "json"]), default="console"
)
@click.option("--check-provider", is_flag=True)
@click.pass_obj
def doctor(context: CLIContext, output_format: str, check_provider: bool) -> None:
    """Validate Git, configuration, adapters, runtime, and provider settings."""
    checks: list[dict[str, str]] = []
    failed = False
    try:
        repo = validate_git_repository(context.repo)
        checks.append({"name": "git", "status": "ok", "detail": str(repo)})
        settings = load_settings(repo, context.config)
        checks.append({"name": "config", "status": "ok", "detail": "schema v1"})
        adapters = default_registry().detect(repo)
        failed = failed or not adapters
        checks.append(
            {
                "name": "adapters",
                "status": "ok" if adapters else "error",
                "detail": ", ".join(adapter.name for adapter in adapters) or "none detected",
            }
        )
        preferred = None if settings.sandbox.runtime == "auto" else settings.sandbox.runtime
        engine = resolve_container_engine(preferred)
        checks.append({"name": "container", "status": "ok", "detail": engine})
        provider_detail = f"{settings.provider.kind}/{settings.provider.model}"
        if check_provider:
            reply = create_provider(settings.provider).complete(
                "This is a connectivity check. Reply with OK.", "OK", max_tokens=8
            )
            provider_detail += f" — replied {reply.strip()[:20]!r}"
        else:
            provider_detail += " — connectivity not requested"
        checks.append({"name": "provider", "status": "ok", "detail": provider_detail})
    except Exception as exc:
        failed = True
        checks.append(
            {"name": "doctor", "status": "error", "detail": f"{type(exc).__name__}: {exc}"}
        )
    if output_format == "json":
        click.echo(json.dumps({"checks": checks, "ok": not failed}, indent=2))
    else:
        table = Table(title="Evalyn doctor")
        table.add_column("Check")
        table.add_column("Status")
        table.add_column("Detail")
        for item in checks:
            style = "green" if item["status"] == "ok" else "red"
            table.add_row(item["name"], f"[{style}]{item['status']}[/]", item["detail"])
        context.console.print(table)
    if failed:
        raise SystemExit(3)


@main.command()
@click.argument("range_spec", required=False)
@click.option("--include-dirty", is_flag=True)
@click.option("--intent", multiple=True)
@click.option(
    "--intent-file",
    multiple=True,
    type=click.Path(path_type=Path, dir_okay=False, exists=True),
)
@click.option("--adapter", "adapters", multiple=True, type=click.Choice(["pytest", "vitest"]))
@click.option("--include", "include_globs", multiple=True)
@click.option("--exclude", "exclude_globs", multiple=True)
@click.option("--max-tests", type=click.IntRange(1, 50))
@click.option("--repeat", type=click.IntRange(2, 10))
@click.option("--budget", callback=lambda c, p, v: _parse_duration(v))
@click.option("--runtime", type=click.Choice(["auto", "docker", "podman"]))
@click.option("--provider", "provider_kind", type=click.Choice(["anthropic", "openai-compatible"]))
@click.option("--model")
@click.option(
    "--fail-on", type=click.Choice(["likely", "never"]), default="likely", show_default=True
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["console", "json", "markdown", "sarif"]),
    default="console",
    show_default=True,
)
@click.option("--output", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--dry-run", is_flag=True)
@click.pass_obj
def check(
    context: CLIContext,
    range_spec: str | None,
    include_dirty: bool,
    intent: tuple[str, ...],
    intent_file: tuple[Path, ...],
    adapters: tuple[str, ...],
    include_globs: tuple[str, ...],
    exclude_globs: tuple[str, ...],
    max_tests: int | None,
    repeat: int | None,
    budget: int | None,
    runtime: str | None,
    provider_kind: str | None,
    model: str | None,
    fail_on: str,
    output_format: str,
    output: Path | None,
    dry_run: bool,
) -> None:
    """Check a Git range for stable, evidence-backed behavior regressions."""
    if output is not None:
        try:
            output = _resolve_output_destination(context.repo, output)
        except (OSError, ValueError) as exc:
            _usage_error(context, str(exc))
    intent_values = list(intent)
    for path in intent_file:
        try:
            text = RunStore.read_bytes(path, max_bytes=100_000).decode("utf-8")
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise click.BadParameter(f"Could not read bounded intent file {path}: {exc}") from exc
        intent_values.append(text)
    options = CheckOptions(
        repo_path=context.repo,
        range_spec=range_spec,
        include_dirty=include_dirty,
        intent=tuple(intent_values),
        adapters=adapters,
        include_globs=include_globs,
        exclude_globs=exclude_globs,
        max_tests=max_tests,
        repeat=repeat,
        budget_seconds=budget,
        runtime=runtime,
        provider_kind=provider_kind,
        model=model,
        config_path=context.config,
        state_root=context.state_root,
    )
    if dry_run:
        try:
            inspection = inspect_check(options)
        except (CheckError, RevisionError, SettingsError, ValueError) as exc:
            _usage_error(context, str(exc))
        payload = {
            "base": inspection.revisions.base_sha,
            "candidate": inspection.revisions.candidate_sha,
            "dirty": inspection.dirty,
            "adapters": list(inspection.adapters),
            "changed_files": list(inspection.changed_files),
        }
        if output_format == "json":
            click.echo(json.dumps(payload, indent=2))
        else:
            context.console.print("[bold]Evalyn dry run[/]")
            context.console.print(f"Base:      {payload['base']}")
            context.console.print(f"Candidate: {payload['candidate']}")
            context.console.print(f"Adapters:  {', '.join(payload['adapters'])}")
            context.console.print(f"Files:     {len(payload['changed_files'])}")
        return
    try:
        outcome = run_check(options)
    except CheckError as exc:
        suffix = f"\nArtifacts: {exc.run_path}" if exc.run_path else ""
        context.console.print(f"[red]Incomplete:[/] {exc}{suffix}")
        raise SystemExit(3) from None
    except (RevisionError, SettingsError, ValueError) as exc:
        _usage_error(context, str(exc))
    except Exception as exc:
        context.console.print(f"[red]Incomplete:[/] {type(exc).__name__}: {exc}")
        raise SystemExit(3) from None
    try:
        _emit_check_output(context, outcome, output_format, output)
    except Exception as exc:
        context.console.print(
            f"[red]Incomplete:[/] Could not emit check output: {type(exc).__name__}: {exc}"
        )
        raise SystemExit(3) from None
    if outcome.incomplete:
        raise SystemExit(3)
    if fail_on == "likely" and outcome.gating_findings:
        raise SystemExit(1)


@main.command()
@click.argument("identifier", required=False)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["console", "json", "markdown", "sarif"]),
    default="console",
)
@click.pass_obj
def show(context: CLIContext, identifier: str | None, output_format: str) -> None:
    """Show the latest run, a run ID, or a finding ID."""
    try:
        repo = validate_git_repository(context.repo)
        store = RunStore(repo, context.state_root)
    except (RevisionError, ValueError) as exc:
        _usage_error(context, str(exc))
    if identifier and identifier.upper().startswith("EVL-"):
        try:
            identifier = validate_finding_id(identifier)
        except OperationError as exc:
            _usage_error(context, str(exc))
        located = store.find_finding(identifier)
        if located is None:
            raise click.ClickException(f"Finding not found: {identifier}")
        _emit_value(context, located[1], output_format)
        return
    run = store.get(identifier)
    if run is None:
        raise click.ClickException("No matching Evalyn run found")
    path = run.path / {"json": "findings.json", "sarif": "evalyn.sarif"}.get(
        output_format, "report.md"
    )
    if not path.is_file():
        context.console.print(f"[red]Incomplete:[/] Run output is missing: {path}")
        raise SystemExit(3)
    try:
        text = RunStore.read_bytes(path, max_bytes=32 * 1024 * 1024).decode("utf-8")
        context.console.print(Markdown(text)) if output_format == "console" else click.echo(text)
    except Exception as exc:
        context.console.print(
            f"[red]Incomplete:[/] Could not read run output: {type(exc).__name__}: {exc}"
        )
        raise SystemExit(3) from None


@main.command()
@click.argument("finding_id")
@click.option("--side", type=click.Choice(["base", "candidate", "both"]), default="both")
@click.option(
    "--repeat",
    type=click.IntRange(1, 10),
    help="Override the saved run's repetition count.",
)
@click.option("--runtime", type=click.Choice(["docker", "podman"]))
@click.pass_obj
def reproduce(
    context: CLIContext,
    finding_id: str,
    side: str,
    repeat: int | None,
    runtime: str | None,
) -> None:
    """Re-run the exact stored probe against its pinned revisions."""
    try:
        finding_id = validate_finding_id(finding_id)
    except OperationError as exc:
        _usage_error(context, str(exc))
    try:
        repo = validate_git_repository(context.repo)
        result = reproduce_saved_finding(
            RunStore(repo, context.state_root),
            finding_id,
            side=side,
            repeat=repeat,
            runtime=runtime,
        )
    except (RevisionError, OperationError, RuntimeUnavailableError, ValueError) as exc:
        context.console.print(f"[red]Incomplete:[/] {exc}")
        raise SystemExit(3) from None
    context.console.print(f"[bold]{finding_id}[/]")
    if result.base:
        context.console.print("Base:      " + ", ".join(item.status.value for item in result.base))
    if result.candidate:
        context.console.print(
            "Candidate: " + ", ".join(item.status.value for item in result.candidate)
        )


@main.command()
@click.argument("finding_id")
@click.option("--confirm", is_flag=True)
@click.option("--dismiss", is_flag=True)
@click.option("--uncertain", is_flag=True)
@click.option("--reason", required=True)
@click.pass_obj
def triage(
    context: CLIContext,
    finding_id: str,
    confirm: bool,
    dismiss: bool,
    uncertain: bool,
    reason: str,
) -> None:
    """Record a human disposition for a finding."""
    try:
        finding_id = validate_finding_id(finding_id)
    except OperationError as exc:
        _usage_error(context, str(exc))
    selections = [
        name
        for name, enabled in (
            ("confirmed", confirm),
            ("dismissed", dismiss),
            ("uncertain", uncertain),
        )
        if enabled
    ]
    if len(selections) != 1:
        raise click.UsageError("Choose exactly one of --confirm, --dismiss, or --uncertain")
    try:
        repo = validate_git_repository(context.repo)
        run, _finding = triage_saved_finding(
            RunStore(repo, context.state_root), finding_id, status=selections[0], reason=reason
        )
    except (RevisionError, OperationError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    context.console.print(f"[green]Updated[/] {finding_id} in {run.path}")


@main.command()
@click.argument("finding_id")
@click.option("--emit-patch", type=click.Path(path_type=Path, dir_okay=False), required=True)
@click.option("--runtime", type=click.Choice(["docker", "podman"]))
@click.pass_obj
def repair(context: CLIContext, finding_id: str, emit_patch: Path, runtime: str | None) -> None:
    """EXPERIMENTAL: validate a model repair and emit a patch without applying it."""
    from evalyn.repair_v2 import RepairError, repair_saved_finding

    try:
        finding_id = validate_finding_id(finding_id)
    except OperationError as exc:
        _usage_error(context, str(exc))
    try:
        repo = validate_git_repository(context.repo)
        result = repair_saved_finding(
            RunStore(repo, context.state_root),
            finding_id,
            output_path=emit_patch,
            runtime=runtime,
            config_path=context.config,
        )
    except (RevisionError, RepairError, ValueError) as exc:
        context.console.print(f"[red]Repair not emitted:[/] {exc}")
        raise SystemExit(3) from None
    context.console.print(f"[green]Verified experimental patch:[/] {result}")


@main.command(name="runs")
@click.option("--limit", type=click.IntRange(1, 1000), default=20, show_default=True)
@click.option(
    "--format", "output_format", type=click.Choice(["console", "json"]), default="console"
)
@click.pass_obj
def list_runs(context: CLIContext, limit: int, output_format: str) -> None:
    """List stored runs for this repository."""
    try:
        repo = validate_git_repository(context.repo)
        runs = RunStore(repo, context.state_root).list_runs(limit)
    except (RevisionError, ValueError) as exc:
        _usage_error(context, str(exc))
    if output_format == "json":
        click.echo(json.dumps([run.manifest for run in runs], indent=2))
        return
    table = Table(title="Evalyn runs")
    for heading in ("Run", "Status", "Base", "Candidate", "Findings"):
        table.add_column(heading)
    for run in runs:
        table.add_row(
            run.run_id,
            str(run.manifest.get("status", "unknown")),
            str(run.manifest.get("base", ""))[:12],
            str(run.manifest.get("candidate_id", run.manifest.get("candidate", "")))[:20],
            str(run.manifest.get("finding_count", 0)),
        )
    context.console.print(table)


@main.command()
@click.option("--runs-before", type=click.IntRange(0, 100_000), default=30, show_default=True)
@click.pass_obj
def clean(context: CLIContext, runs_before: int) -> None:
    """Remove old external artifacts; never reset or clean a Git checkout."""
    try:
        repo = validate_git_repository(context.repo)
        removed = RunStore(repo, context.state_root).clean_before(runs_before)
    except (RevisionError, ValueError) as exc:
        _usage_error(context, str(exc))
    noun = "directory" if removed == 1 else "directories"
    context.console.print(f"Removed {removed} owned run artifact {noun}.")


@main.command()
@click.pass_obj
def demo(context: CLIContext) -> None:
    """Run the bundled no-key regression demonstration."""
    from evalyn.demo import DemoError, run_demo

    try:
        result = run_demo(state_root=context.state_root)
        report = RunStore.read_bytes(result.report_path, max_bytes=32 * 1024 * 1024).decode("utf-8")
    except (DemoError, OSError, UnicodeDecodeError, ValueError) as exc:
        context.console.print(f"[red]Demo incomplete:[/] {exc}")
        raise SystemExit(3) from None
    context.console.print(Markdown(report))
    context.console.print(f"\nArtifacts: {result.run_path}")


def _emit_check_output(
    context: CLIContext, outcome, output_format: str, output: Path | None
) -> None:
    key = {"console": "markdown", "markdown": "markdown", "json": "json", "sarif": "sarif"}[
        output_format
    ]
    path = _validated_report_path(outcome, key)
    payload = RunStore.read_bytes(path, max_bytes=32 * 1024 * 1024)
    if output is not None:
        destination = _resolve_output_destination(context.repo, output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination = _resolve_output_destination(context.repo, destination)
        _copy_file_atomically(payload, destination)
    text = payload.decode("utf-8")
    if output_format == "console":
        context.console.print(Markdown(text))
        context.console.print(f"Artifacts: {outcome.run.path}")
    else:
        click.echo(text)


def _validated_report_path(outcome, key: str) -> Path:
    """Return an owned regular report file from the completed run."""
    path = Path(outcome.report_paths[key])
    if path.is_symlink():
        raise OSError(f"Run report cannot be a symlink: {path}")
    run_root = Path(outcome.run.path).resolve(strict=True)
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(run_root)
    except ValueError as exc:
        raise OSError(f"Run report escaped its artifact directory: {path}") from exc
    if not resolved.is_file():
        raise OSError(f"Run report is not a regular file: {path}")
    return resolved


def _resolve_output_destination(repo_path: Path, output: Path) -> Path:
    """Resolve output inside the repo; absolute paths outside it are rejected."""
    repo = repo_path.expanduser().resolve(strict=True)
    if not repo.is_dir():
        raise ValueError(f"Target repository is not a directory: {repo}")
    requested = output.expanduser()
    candidate = requested if requested.is_absolute() else repo / requested
    candidate = Path(os.path.abspath(candidate))
    _reject_symlink_components(candidate)
    destination = candidate.resolve(strict=False)
    try:
        relative = destination.relative_to(repo)
    except ValueError as exc:
        kind = "absolute" if requested.is_absolute() else "relative"
        raise ValueError(f"{kind} output must resolve inside the target repository") from exc
    if not relative.parts or relative.parts[0].casefold() == ".git":
        raise ValueError("output cannot target the repository root or Git metadata")
    if destination.exists() and not destination.is_file():
        raise ValueError("output destination must name a regular file")
    if destination.parent.exists() and not destination.parent.is_dir():
        raise ValueError("output parent must be a directory")
    return destination


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"output path cannot contain symlinks: {current}")


def _copy_file_atomically(payload: bytes, destination: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as writer:
            writer.write(payload)
            writer.flush()
            os.fsync(writer.fileno())
        temporary.replace(destination)
    except BaseException:
        with suppress(OSError):
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def _emit_value(context: CLIContext, value: dict, output_format: str) -> None:
    if output_format in {"json", "sarif"}:
        click.echo(json.dumps(value, indent=2))
        return
    rendered = "\n".join(
        (
            f"# {value.get('finding_id', 'Finding')}",
            "",
            str(value.get("title", "")),
            "",
            f"- Semantic state: `{value.get('semantic_state', 'unknown')}`",
            f"- Triage: `{value.get('triage_status', 'unreviewed')}`",
            f"- Oracle: {value.get('oracle', '')}",
            f"- Reproduce: `evalyn reproduce {value.get('finding_id', '')}`",
        )
    )
    context.console.print(Markdown(rendered)) if output_format == "console" else click.echo(
        rendered
    )


def _usage_error(context: CLIContext, message: str) -> None:
    context.console.print(f"[red]Error:[/] {message}")
    raise SystemExit(2)


def _parse_duration(value: str | None) -> int | None:
    if value is None:
        return None
    match = re.fullmatch(r"([1-9][0-9]*)([smh]?)", value.strip().casefold())
    if not match:
        raise click.BadParameter("duration must look like 90s, 10m, or 1h")
    return int(match.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600}[match.group(2)]


if __name__ == "__main__":
    main()
