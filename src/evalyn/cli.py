"""Evalyn v0.2 command-line interface."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
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
from evalyn.artifacts import (
    ArtifactExportError,
    build_finding_bundle,
    promote_finding_source,
    write_bundle_exclusive,
)
from evalyn.execution import RuntimeUnavailableError, resolve_container_engine
from evalyn.execution.git_safety import sanitized_git_command, sanitized_git_environment
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


@dataclass(frozen=True)
class CLIContext:
    repo: Path
    config: Path | None
    state_root: Path | None
    console: Console
    allow_remote_container: bool = False


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
@click.option(
    "--env-file",
    type=click.Path(path_type=Path, dir_okay=False, exists=True),
    help="Explicit credential file outside the target repository.",
)
@click.option("--no-color", is_flag=True)
@click.option(
    "--allow-remote-container",
    is_flag=True,
    help="Explicitly permit a tcp:// or ssh:// OCI daemon endpoint for this invocation.",
)
@click.option("-v", "--verbose", count=True)
@click.version_option(version=__version__, prog_name="evalyn")
@click.pass_context
def main(
    ctx: click.Context,
    repo: Path,
    config: Path | None,
    state_root: Path | None,
    env_file: Path | None,
    no_color: bool,
    allow_remote_container: bool,
    verbose: int,
) -> None:
    """Generate disposable tests for a Git diff and surface stable regressions."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING)
    if env_file is not None:
        _load_external_env_file(repo, env_file)
    ctx.obj = CLIContext(
        repo=repo.expanduser(),
        config=config.expanduser() if config else None,
        state_root=state_root.expanduser() if state_root else None,
        console=Console(color_system=None if no_color else "auto"),
        allow_remote_container=allow_remote_container,
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
    repo: Path | None = None
    settings = None

    def record(name: str, action) -> object | None:
        nonlocal failed
        try:
            detail, value = action()
        except Exception as exc:
            failed = True
            checks.append(
                {"name": name, "status": "error", "detail": f"{type(exc).__name__}: {exc}"}
            )
            return None
        checks.append({"name": name, "status": "ok", "detail": detail})
        return value

    repo_value = record(
        "git",
        lambda: (
            str(validated := validate_git_repository(context.repo)),
            validated,
        ),
    )
    # Keep independent checks useful even when Git validation failed.
    repo = repo_value if isinstance(repo_value, Path) else context.repo.expanduser().resolve()

    settings = record(
        "config",
        lambda: ("schema v1", load_settings(repo, context.config)),
    )
    adapter_value = record(
        "adapters",
        lambda: _doctor_adapters(repo),
    )
    del adapter_value

    preferred = None
    if settings is not None:
        preferred = None if settings.sandbox.runtime == "auto" else settings.sandbox.runtime
    record(
        "container",
        lambda: (
            str(engine := resolve_container_engine(preferred)),
            engine,
        ),
    )

    if settings is None:
        checks.append(
            {
                "name": "provider",
                "status": "skipped",
                "detail": "configuration is unavailable",
            }
        )
    elif check_provider:
        record("provider", lambda: _doctor_provider(settings))
    else:
        checks.append(
            {
                "name": "provider",
                "status": "not_checked",
                "detail": (
                    f"{settings.provider.kind}/{settings.provider.model} — use --check-provider"
                ),
            }
        )
    if output_format == "json":
        click.echo(json.dumps({"checks": checks, "ok": not failed}, indent=2))
    else:
        table = Table(title="Evalyn doctor")
        table.add_column("Check")
        table.add_column("Status")
        table.add_column("Detail")
        for item in checks:
            style = (
                "green"
                if item["status"] == "ok"
                else ("yellow" if item["status"] in {"skipped", "not_checked"} else "red")
            )
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
        allow_remote_container=context.allow_remote_container,
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
        try:
            _emit_dry_run(context, payload, output_format, output)
        except (OSError, ValueError) as exc:
            context.console.print(f"[red]Incomplete:[/] Could not emit dry-run output: {exc}")
            raise SystemExit(3) from None
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
            allow_remote_endpoint=context.allow_remote_container,
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
    context.console.print(f"Verdict:   {result.verdict}")
    if result.incomplete:
        raise SystemExit(3)
    if not result.reproduced:
        raise SystemExit(1)


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
@click.option(
    "--output",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="New .zip path inside the target repository.",
)
@click.pass_obj
def bundle(context: CLIContext, finding_id: str, output: Path) -> None:
    """Export one finding as a deterministic, integrity-manifested ZIP."""
    try:
        finding_id = validate_finding_id(finding_id)
    except OperationError as exc:
        _usage_error(context, str(exc))
    try:
        repo = validate_git_repository(context.repo)
        destination = _resolve_output_destination(repo, output)
        if destination.suffix.casefold() != ".zip":
            raise ArtifactExportError("Evidence bundle output must use a .zip extension")
        payload = build_finding_bundle(RunStore(repo, context.state_root), finding_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination = _resolve_output_destination(repo, destination)
        write_bundle_exclusive(destination, payload)
    except (RevisionError, OperationError, ArtifactExportError, OSError, ValueError) as exc:
        context.console.print(f"[red]Bundle not emitted:[/] {exc}")
        raise SystemExit(3) from None
    context.console.print(f"[green]Evidence bundle:[/] {destination}")


@main.command()
@click.argument("finding_id")
@click.option(
    "--output",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="New permanent test path inside the target repository.",
)
@click.pass_obj
def promote(context: CLIContext, finding_id: str, output: Path) -> None:
    """Promote a maintainer-confirmed generated probe to a normal test."""
    try:
        finding_id = validate_finding_id(finding_id)
    except OperationError as exc:
        _usage_error(context, str(exc))
    try:
        repo = validate_git_repository(context.repo)
        destination = _resolve_output_destination(repo, output)
        promoted = promote_finding_source(
            RunStore(repo, context.state_root),
            finding_id,
            destination,
        )
    except (RevisionError, OperationError, ArtifactExportError, OSError, ValueError) as exc:
        context.console.print(f"[red]Test not promoted:[/] {exc}")
        raise SystemExit(3) from None
    context.console.print(f"[green]Promoted test:[/] {promoted}")


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
            allow_remote_endpoint=context.allow_remote_container,
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


def _doctor_adapters(repo: Path) -> tuple[str, tuple[object, ...]]:
    adapters = tuple(default_registry().detect(repo))
    if not adapters:
        raise RuntimeError("no pytest or Vitest adapter detected")
    return ", ".join(adapter.name for adapter in adapters), adapters


def _doctor_provider(settings) -> tuple[str, object]:
    reply = create_provider(settings.provider).complete(
        "This is a connectivity check. Reply with OK.", "OK", max_tokens=8
    )
    detail = f"{settings.provider.kind}/{settings.provider.model} — replied {reply.strip()[:20]!r}"
    return detail, reply


def _emit_dry_run(
    context: CLIContext,
    payload: dict[str, object],
    output_format: str,
    output: Path | None,
) -> None:
    if output_format == "json":
        rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    elif output_format == "sarif":
        rendered = (
            json.dumps(
                {
                    "version": "2.1.0",
                    "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
                    "runs": [
                        {
                            "tool": {"driver": {"name": "Evalyn dry run"}},
                            "invocations": [
                                {
                                    "executionSuccessful": True,
                                    "properties": payload,
                                }
                            ],
                            "results": [],
                        }
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    else:
        changed_files = payload["changed_files"]
        assert isinstance(changed_files, list)
        adapters = payload["adapters"]
        assert isinstance(adapters, list)
        lines = [
            "# Evalyn dry run",
            "",
            f"- Base: `{payload['base']}`",
            f"- Candidate: `{payload['candidate']}`",
            f"- Dirty overlay: `{str(payload['dirty']).lower()}`",
            f"- Adapters: {', '.join(str(item) for item in adapters) or 'none'}",
            f"- Changed files: {len(changed_files)}",
            "",
        ]
        lines.extend(f"  - `{path}`" for path in changed_files)
        rendered = "\n".join(lines) + "\n"

    if output is not None:
        destination = _resolve_output_destination(context.repo, output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination = _resolve_output_destination(context.repo, destination)
        _copy_file_atomically(rendered.encode("utf-8"), destination)
    if output_format == "console":
        context.console.print(Markdown(rendered))
    else:
        click.echo(rendered, nl=False)


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
    if destination.exists():
        raise ValueError("output destination already exists; Evalyn will not overwrite it")
    if destination.parent.exists() and not destination.parent.is_dir():
        raise ValueError("output parent must be a directory")
    if _is_tracked_output(repo, relative.as_posix()):
        raise ValueError("output cannot target a tracked repository path")
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
        # A hard-link publishes the fully flushed temporary inode while failing
        # atomically if another process created the destination after validation.
        os.link(temporary, destination, follow_symlinks=False)
        temporary.unlink()
    except BaseException:
        with suppress(OSError):
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def _is_tracked_output(repo: Path, relative: str) -> bool:
    try:
        completed = subprocess.run(
            sanitized_git_command(repo, "ls-files", "--", relative),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            env=sanitized_git_environment(),
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"could not verify whether output is tracked: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(
            "could not verify whether output is tracked" + (f": {detail}" if detail else "")
        )
    return bool(completed.stdout.strip())


def _load_external_env_file(repo_path: Path, env_file: Path) -> None:
    try:
        repo = repo_path.expanduser().resolve(strict=True)
        if env_file.is_symlink():
            raise click.BadParameter("--env-file cannot be a symlink")
        resolved = env_file.expanduser().resolve(strict=True)
        if resolved.is_relative_to(repo):
            raise click.BadParameter("--env-file must be outside the target repository")
    except click.BadParameter:
        raise
    except (OSError, ValueError) as exc:
        raise click.BadParameter(f"Could not validate --env-file: {exc}") from exc
    load_dotenv(resolved, override=False)


def _emit_value(context: CLIContext, value: dict, output_format: str) -> None:
    if output_format == "sarif":
        click.echo(json.dumps(_stored_finding_sarif(value), indent=2))
        return
    if output_format == "json":
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


def _stored_finding_sarif(finding: dict) -> dict[str, object]:
    """Render one stored finding as an actual SARIF log for ``show``."""
    should_gate = finding.get("should_gate") is True
    semantic_state = str(finding.get("semantic_state", "uncertain_change"))
    triage_status = str(finding.get("triage_status", "unreviewed"))
    level = (
        "error"
        if should_gate
        else (
            "note"
            if triage_status == "dismissed" or semantic_state == "intended_change"
            else "warning"
        )
    )
    rule_id = "evalyn/likely-regression" if should_gate else "evalyn/behavior-change"
    result: dict[str, object] = {
        "ruleId": rule_id,
        "level": level,
        "message": {"text": f"{finding.get('title', '')} ({finding.get('finding_id', 'unknown')})"},
        "properties": {
            "findingId": finding.get("finding_id", ""),
            "semanticState": semantic_state,
            "oracleProvenance": finding.get("oracle_provenance", ""),
            "triageStatus": triage_status,
            "evidence": finding.get("evidence"),
        },
    }
    metadata = finding.get("probe_metadata")
    changed_file = metadata.get("changed_file") if isinstance(metadata, dict) else None
    if isinstance(changed_file, str) and changed_file:
        result["locations"] = [{"physicalLocation": {"artifactLocation": {"uri": changed_file}}}]
    return {
        "version": "2.1.0",
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "Evalyn",
                        "rules": [
                            {
                                "id": rule_id,
                                "shortDescription": {"text": "Evalyn differential finding"},
                            }
                        ],
                    }
                },
                "results": [result],
            }
        ],
    }


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
