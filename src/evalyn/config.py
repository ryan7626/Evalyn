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

    # Read CLAUDE.md (required)
    claude_md_path = repo_path / "CLAUDE.md"
    if not claude_md_path.is_file():
        raise ConfigError(f"CLAUDE.md not found at {claude_md_path}")

    try:
        claude_md_content = claude_md_path.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError(f"Failed to read CLAUDE.md: {e}")

    if not claude_md_content.strip():
        raise ConfigError(f"CLAUDE.md is empty: {claude_md_path}")

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
        if "max_fix_retries" in execution:
            config.max_fix_retries = int(execution["max_fix_retries"])
        if "test_timeout_seconds" in execution:
            config.test_timeout_seconds = int(execution["test_timeout_seconds"])
        if "max_output_lines" in execution:
            config.max_output_lines = int(execution["max_output_lines"])

    services = overrides.get("services", [])
    if isinstance(services, list):
        config.services = services
