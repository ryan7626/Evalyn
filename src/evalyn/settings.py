"""Configuration for Evalyn's diff-regression workflow.

The v0.2 configuration is intentionally small and optional.  Repository
configuration supplies trusted adapter commands; generated model output never
does.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from evalyn.run_store import RunStore

MAX_CONFIG_BYTES = 256 * 1024
MAX_TESTS = 100
MAX_REPEAT = 10
MAX_BUDGET_SECONDS = 24 * 60 * 60


class SettingsError(ValueError):
    """Raised when ``.evalyn.yml`` contains unsupported configuration."""


@dataclass(frozen=True)
class ProviderSettings:
    kind: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    base_url: str | None = None


@dataclass(frozen=True)
class AdapterSettings:
    name: str
    image: str | None = None
    setup_argv: tuple[str, ...] = ()
    suite_argv: tuple[str, ...] = ()


@dataclass(frozen=True)
class SandboxSettings:
    runtime: str = "auto"
    setup_env: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvalynSettings:
    repo_path: Path
    provider: ProviderSettings = field(default_factory=ProviderSettings)
    adapters: tuple[AdapterSettings, ...] = ()
    sandbox: SandboxSettings = field(default_factory=SandboxSettings)
    max_tests: int = 12
    repeat: int = 3
    budget_seconds: int = 600


def resolve_settings_path(repo_path: Path, config_path: Path | None = None) -> Path:
    """Resolve the selected configuration without accepting linked components."""
    repo = repo_path.resolve()
    requested = config_path.expanduser() if config_path is not None else repo / ".evalyn.yml"
    candidate = requested if requested.is_absolute() else repo / requested
    path = Path(os.path.abspath(os.fspath(candidate)))
    _reject_linked_config_path(path)
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError:
        try:
            resolved = path.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise SettingsError(f"Could not resolve configuration path {path}: {exc}") from exc
    except (OSError, RuntimeError) as exc:
        raise SettingsError(f"Could not resolve configuration path {path}: {exc}") from exc
    # Recheck after resolution so a component swapped to a link while resolving
    # is rejected before the path is used for either trust classification or IO.
    _reject_linked_config_path(path)
    return resolved


def load_settings(repo_path: Path, config_path: Path | None = None) -> EvalynSettings:
    """Load optional schema-version-1 repository configuration."""
    repo = repo_path.resolve()
    path = resolve_settings_path(repo, config_path)
    if not path.exists():
        if config_path is not None:
            raise SettingsError(f"Explicit configuration does not exist: {path}")
        return EvalynSettings(repo_path=repo)
    operator_config = not path.is_relative_to(repo)

    try:
        payload = RunStore.read_bytes(path, max_bytes=MAX_CONFIG_BYTES)
        raw = yaml.safe_load(payload.decode("utf-8")) or {}
    except (OSError, UnicodeDecodeError, ValueError, yaml.YAMLError) as exc:
        raise SettingsError(f"Could not read {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise SettingsError(".evalyn.yml must contain a mapping")
    unknown = set(raw) - {"version", "provider", "adapters", "sandbox", "limits"}
    if unknown:
        raise SettingsError(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
    version = raw.get("version", 1)
    if not isinstance(version, int) or isinstance(version, bool) or version != 1:
        raise SettingsError("Only .evalyn.yml schema version 1 is supported")

    provider = _provider_settings(
        raw.get("provider", {}),
        allow_custom_endpoint=operator_config,
    )
    adapters = _adapter_settings(raw.get("adapters", []))
    sandbox = _sandbox_settings(
        raw.get("sandbox", {}),
        allow_setup_environment=operator_config,
    )
    limits = _mapping(raw.get("limits", {}), "limits")
    unknown_limits = set(limits) - {"max_tests", "repeat", "budget_seconds"}
    if unknown_limits:
        raise SettingsError(f"Unknown limits keys: {', '.join(sorted(unknown_limits))}")
    max_tests = _positive_int(
        limits.get("max_tests", 12),
        "limits.max_tests",
        maximum=MAX_TESTS,
    )
    repeat = _positive_int(
        limits.get("repeat", 3),
        "limits.repeat",
        maximum=MAX_REPEAT,
    )
    if repeat < 2:
        raise SettingsError("limits.repeat must be at least 2")
    budget_seconds = _positive_int(
        limits.get("budget_seconds", 600),
        "limits.budget_seconds",
        maximum=MAX_BUDGET_SECONDS,
    )
    return EvalynSettings(
        repo_path=repo,
        provider=provider,
        adapters=adapters,
        sandbox=sandbox,
        max_tests=max_tests,
        repeat=repeat,
        budget_seconds=budget_seconds,
    )


def _reject_linked_config_path(path: Path) -> None:
    for component in (path, *path.parents):
        if component.is_symlink():
            raise SettingsError(f"Configuration path cannot contain a symlink: {path}")
        is_junction = getattr(component, "is_junction", lambda: False)
        if is_junction():
            raise SettingsError(f"Configuration path cannot contain a junction: {path}")


def render_default_config(adapters: list[str]) -> str:
    """Return a minimal, reviewable configuration discovered by ``init``."""
    selected = adapters or ["pytest"]
    lines = [
        "version: 1",
        "provider:",
        "  kind: anthropic",
        "  model: claude-sonnet-4-20250514",
        "sandbox:",
        "  runtime: auto",
        "adapters:",
    ]
    for name in selected:
        image = {
            "pytest": "python:3.12-slim",
            "vitest": "node:22-bookworm-slim",
        }[name]
        lines.extend(
            (
                f"  - name: {name}",
                f"    image: {image}",
                "    setup_argv: []",
                "    suite_argv: []",
            )
        )
    lines.extend(("limits:", "  max_tests: 12", "  repeat: 3", "  budget_seconds: 600", ""))
    return "\n".join(lines)


def _provider_settings(value: Any, *, allow_custom_endpoint: bool) -> ProviderSettings:
    raw = _mapping(value, "provider")
    unknown = set(raw) - {"kind", "model", "base_url"}
    if unknown:
        raise SettingsError(f"Unknown provider keys: {', '.join(sorted(unknown))}")
    kind = raw.get("kind", "anthropic")
    if not isinstance(kind, str):
        raise SettingsError("provider.kind must be a string")
    if kind not in {"anthropic", "openai-compatible"}:
        raise SettingsError("provider.kind must be anthropic or openai-compatible")
    default_model = "claude-sonnet-4-20250514" if kind == "anthropic" else "gpt-4.1-mini"
    model = raw.get("model", default_model)
    if not isinstance(model, str) or not model.strip() or len(model) > 256:
        raise SettingsError("provider.model must be a nonempty string of at most 256 characters")
    base_url = raw.get("base_url")
    if base_url is not None and (not isinstance(base_url, str) or not base_url.strip()):
        raise SettingsError("provider.base_url must be a nonempty string")
    if base_url is not None and not allow_custom_endpoint:
        raise SettingsError(
            "provider.base_url is operator policy and requires an explicit "
            "configuration outside the repository"
        )
    return ProviderSettings(
        kind=kind,
        model=model,
        base_url=base_url,
    )


def _adapter_settings(value: Any) -> tuple[AdapterSettings, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise SettingsError("adapters must be a list")
    result: list[AdapterSettings] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        raw = _mapping(item, f"adapters[{index}]")
        unknown = set(raw) - {"name", "image", "setup_argv", "suite_argv"}
        if unknown:
            raise SettingsError(f"Unknown adapters[{index}] keys: {', '.join(sorted(unknown))}")
        name = raw.get("name", "")
        if not isinstance(name, str):
            raise SettingsError(f"adapters[{index}].name must be a string")
        if name not in {"pytest", "vitest"}:
            raise SettingsError(f"adapters[{index}].name must be pytest or vitest")
        if name in seen:
            raise SettingsError(f"Duplicate adapter: {name}")
        seen.add(name)
        result.append(
            AdapterSettings(
                name=name,
                image=_container_image(raw.get("image"), f"adapters[{index}].image"),
                setup_argv=_argv(raw.get("setup_argv", []), f"adapters[{index}].setup_argv"),
                suite_argv=_argv(raw.get("suite_argv", []), f"adapters[{index}].suite_argv"),
            )
        )
    return tuple(result)


def _sandbox_settings(value: Any, *, allow_setup_environment: bool) -> SandboxSettings:
    raw = _mapping(value, "sandbox")
    unknown = set(raw) - {"runtime", "setup_env"}
    if unknown:
        raise SettingsError(f"Unknown sandbox keys: {', '.join(sorted(unknown))}")
    runtime = raw.get("runtime", "auto")
    if not isinstance(runtime, str):
        raise SettingsError("sandbox.runtime must be a string")
    if runtime not in {"auto", "docker", "podman"}:
        raise SettingsError("sandbox.runtime must be auto, docker, or podman")
    setup_env = _argv(raw.get("setup_env", []), "sandbox.setup_env")
    for name in setup_env:
        if not name.replace("_", "").isalnum() or name[0].isdigit():
            raise SettingsError(f"Invalid environment name in sandbox.setup_env: {name!r}")
        if not _is_package_setup_environment(name):
            raise SettingsError(
                "sandbox.setup_env only accepts package-registry variables; "
                f"unsupported name: {name!r}"
            )
    if setup_env and not allow_setup_environment:
        raise SettingsError(
            "sandbox.setup_env forwards operator secrets and requires an explicit "
            "configuration outside the repository"
        )
    return SandboxSettings(runtime=runtime, setup_env=setup_env)


def _is_package_setup_environment(name: str) -> bool:
    exact = {
        "BUNDLE_GEMS__CONTRIBSYS__COM",
        "CARGO_REGISTRY_TOKEN",
        "COMPOSER_AUTH",
        "NODE_AUTH_TOKEN",
        "NPM_TOKEN",
        "PIP_EXTRA_INDEX_URL",
        "PIP_INDEX_URL",
        "UV_EXTRA_INDEX_URL",
        "UV_INDEX_URL",
        "YARN_NPM_AUTH_TOKEN",
    }
    prefixes = ("BUNDLE_", "CARGO_REGISTRIES_", "NUGET_", "POETRY_HTTP_BASIC_")
    return name in exact or name.startswith(prefixes)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SettingsError(f"{label} must be a mapping")
    return value


def _argv(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(part, str) for part in value):
        raise SettingsError(f"{label} must be a list of strings")
    if any(not part or "\x00" in part for part in value):
        raise SettingsError(f"{label} contains an empty or invalid argument")
    return tuple(value)


def _container_image(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SettingsError(f"{label} must be a string")
    image = value.strip()
    if (
        not image
        or image != value
        or image.startswith("-")
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/@-]{0,511}", image)
    ):
        raise SettingsError(f"{label} is not a valid OCI image reference")
    return image


def _positive_int(value: Any, label: str, *, maximum: int | None = None) -> int:
    """Validate an actual YAML integer without accepting coercible values."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise SettingsError(f"{label} must be a positive integer")
    if maximum is not None and value > maximum:
        raise SettingsError(f"{label} must be at most {maximum}")
    return value
