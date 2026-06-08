"""Configuration loading for Evalyn.

Reads the target repo's CLAUDE.md for context and optional .agent/config.yaml
for per-repo overrides.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


class ConfigError(Exception):
    """Raised when configuration is invalid or cannot be loaded."""
    pass


@dataclass
class Config:
    """Resolved configuration for an Evalyn run."""

    # Target repo
    repo_path: Path
    claude_md_content: str

    # LLM settings
    llm_provider: str = "claude"
    llm_model: str = "claude-3-5-sonnet-20241022"

    # Execution settings
    max_fix_retries: int = 3
    test_timeout_seconds: int = 300
    max_output_lines: int = 200  # max test output lines to send to LLM
    repair_mode: str = "apply_unstaged"

    # Service settings
    services: list[dict] = field(default_factory=list)

    # Paths (derived)
    agent_dir: Path = field(init=False)
    runs_dir: Path = field(init=False)

    def __post_init__(self):
        self.agent_dir = self.repo_path / ".agent"
        self.runs_dir = self.agent_dir / "runs"


def load_config(repo_path: Path) -> Config:
    """Load configuration from the target repository.

    Reads CLAUDE.md for repo context, then applies any overrides from
    .agent/config.yaml if it exists.

    Args:
        repo_path: Absolute path to the target repository root.

    Returns:
        Resolved Config object.

    Raises:
        ConfigError: If CLAUDE.md is missing or config.yaml is malformed.
    """
    repo_path = repo_path.resolve()

    # Read CLAUDE.md (required) — try common casings for Linux compatibility
    claude_md_path = None
    for name in ("CLAUDE.md", "claude.md", "Claude.md"):
        candidate = repo_path / name
        if candidate.is_file():
            claude_md_path = candidate
            break

    if claude_md_path is None:
        raise ConfigError(
            f"CLAUDE.md not found at {repo_path}. "
            "Evalyn requires a CLAUDE.md context file in the target repository."
        )

    try:
        claude_md_content = claude_md_path.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError(f"Failed to read {claude_md_path.name}: {e}")

    if not claude_md_content.strip():
        raise ConfigError(f"{claude_md_path.name} is empty: {claude_md_path}")

    # Start with defaults
    config = Config(
        repo_path=repo_path,
        claude_md_content=claude_md_content,
    )

    # Apply overrides from .agent/config.yaml if it exists
    config_yaml_path = repo_path / ".agent" / "config.yaml"
    if config_yaml_path.is_file():
        try:
            raw = yaml.safe_load(config_yaml_path.read_text(encoding="utf-8"))
        except (yaml.YAMLError, OSError) as e:
            raise ConfigError(f"Failed to parse .agent/config.yaml: {e}")

        if raw and isinstance(raw, dict):
            _apply_overrides(config, raw)

    return config


def _apply_overrides(config: Config, overrides: dict) -> None:
    """Apply config.yaml overrides to the Config object."""
    llm = overrides.get("llm", {})
    if isinstance(llm, dict):
        if "provider" in llm:
            config.llm_provider = str(llm["provider"])
        if "model" in llm:
            config.llm_model = str(llm["model"])

    execution = overrides.get("execution", {})
    if isinstance(execution, dict):
        for key, attr in (
            ("max_fix_retries", "max_fix_retries"),
            ("test_timeout_seconds", "test_timeout_seconds"),
            ("max_output_lines", "max_output_lines"),
        ):
            if key in execution:
                try:
                    setattr(config, attr, int(execution[key]))
                except (TypeError, ValueError):
                    raise ConfigError(
                        f"Invalid value for {key} in .agent/config.yaml: "
                        f"{execution[key]!r} (expected an integer)"
                    )
        if "repair_mode" in execution:
            repair_mode = str(execution["repair_mode"])
            allowed = {"apply_unstaged", "suggest_patches", "test_only"}
            if repair_mode not in allowed:
                choices = ", ".join(sorted(allowed))
                raise ConfigError(
                    f"Invalid value for repair_mode in .agent/config.yaml: "
                    f"{repair_mode!r} (expected one of: {choices})"
                )
            config.repair_mode = repair_mode

    services = overrides.get("services", [])
    if isinstance(services, list):
        config.services = services
