"""Configuration for Evalyn's diff-regression workflow.

The v0.2 configuration is intentionally small and optional.  Repository
configuration supplies trusted adapter commands; generated model output never
does.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from evalyn.run_store import RunStore

_MAX_CONFIG_BYTES = 256 * 1024


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


def load_settings(repo_path: Path, config_path: Path | None = None) -> EvalynSettings:
    """Load optional schema-version-1 repository configuration."""
    repo = repo_path.resolve()
    path = config_path.expanduser() if config_path else repo / ".evalyn.yml"
    if not path.is_absolute():
        path = repo / path
    if path.is_symlink():
        raise SettingsError(f"Configuration path cannot be a symlink: {path}")
    if not path.exists():
        return EvalynSettings(repo_path=repo)

    try:
        payload = RunStore.read_bytes(path, max_bytes=_MAX_CONFIG_BYTES)
        raw = yaml.safe_load(payload.decode("utf-8")) or {}
    except (OSError, UnicodeDecodeError, ValueError, yaml.YAMLError) as exc:
        raise SettingsError(f"Could not read {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise SettingsError(".evalyn.yml must contain a mapping")
    unknown = set(raw) - {"version", "provider", "adapters", "sandbox", "limits"}
    if unknown:
        raise SettingsError(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
    if raw.get("version", 1) != 1:
        raise SettingsError("Only .evalyn.yml schema version 1 is supported")

    provider = _provider_settings(raw.get("provider", {}))
    adapters = _adapter_settings(raw.get("adapters", []))
    sandbox = _sandbox_settings(raw.get("sandbox", {}))
    limits = _mapping(raw.get("limits", {}), "limits")
    max_tests = _positive_int(limits.get("max_tests", 12), "limits.max_tests")
    repeat = _positive_int(limits.get("repeat", 3), "limits.repeat")
    if repeat < 2:
        raise SettingsError("limits.repeat must be at least 2")
    if repeat > 10:
        raise SettingsError("limits.repeat must be at most 10")
    budget_seconds = _positive_int(limits.get("budget_seconds", 600), "limits.budget_seconds")
    return EvalynSettings(
        repo_path=repo,
        provider=provider,
        adapters=adapters,
        sandbox=sandbox,
        max_tests=max_tests,
        repeat=repeat,
        budget_seconds=budget_seconds,
    )


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


def _provider_settings(value: Any) -> ProviderSettings:
    raw = _mapping(value, "provider")
    unknown = set(raw) - {"kind", "model", "base_url"}
    if unknown:
        raise SettingsError(f"Unknown provider keys: {', '.join(sorted(unknown))}")
    kind = str(raw.get("kind", "anthropic"))
    if kind not in {"anthropic", "openai-compatible"}:
        raise SettingsError("provider.kind must be anthropic or openai-compatible")
    default_model = "claude-sonnet-4-20250514" if kind == "anthropic" else "gpt-4.1-mini"
    return ProviderSettings(
        kind=kind,
        model=str(raw.get("model", default_model)),
        base_url=str(raw["base_url"]) if raw.get("base_url") else None,
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
        name = str(raw.get("name", ""))
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


def _sandbox_settings(value: Any) -> SandboxSettings:
    raw = _mapping(value, "sandbox")
    unknown = set(raw) - {"runtime", "setup_env"}
    if unknown:
        raise SettingsError(f"Unknown sandbox keys: {', '.join(sorted(unknown))}")
    runtime = str(raw.get("runtime", "auto"))
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


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise SettingsError(f"{label} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise SettingsError(f"{label} must be a positive integer") from exc
    if result <= 0:
        raise SettingsError(f"{label} must be a positive integer")
    return result
