"""Tests for config.py — load_config, _apply_overrides, and Config defaults."""

import tempfile
from pathlib import Path

import pytest

from evalyn.config import Config, ConfigError, load_config, _apply_overrides

SAMPLE_CLAUDE_MD = "# My Project\n\nThis is the project context.\n"


@pytest.fixture
def repo(tmp_path):
    """A tmp directory containing a valid CLAUDE.md."""
    (tmp_path / "CLAUDE.md").write_text(SAMPLE_CLAUDE_MD, encoding="utf-8")
    return tmp_path


class TestLoadConfig:
    def test_loads_successfully(self, repo):
        config = load_config(repo)
        assert isinstance(config, Config)
        assert config.repo_path == repo.resolve()
        assert "My Project" in config.claude_md_content

    def test_agent_dir_derived(self, repo):
        config = load_config(repo)
        assert config.agent_dir == config.repo_path / ".agent"
        assert config.runs_dir == config.repo_path / ".agent" / "runs"

    def test_missing_claude_md_raises(self, tmp_path):
        with pytest.raises(ConfigError, match="CLAUDE.md not found"):
            load_config(tmp_path)

    def test_empty_claude_md_raises(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("   \n  \n", encoding="utf-8")
        with pytest.raises(ConfigError, match="empty"):
            load_config(tmp_path)

    def test_default_model_is_non_empty_string(self, repo):
        config = load_config(repo)
        assert isinstance(config.llm_model, str)
        assert len(config.llm_model) > 0

    def test_default_provider_is_claude(self, repo):
        config = load_config(repo)
        assert config.llm_provider == "claude"

    def test_default_retries(self, repo):
        config = load_config(repo)
        assert config.max_fix_retries == 3

    def test_applies_yaml_overrides(self, repo):
        agent_dir = repo / ".agent"
        agent_dir.mkdir()
        (agent_dir / "config.yaml").write_text(
            "llm:\n  model: my-custom-model\nexecution:\n  max_fix_retries: 7\n",
            encoding="utf-8",
        )
        config = load_config(repo)
        assert config.llm_model == "my-custom-model"
        assert config.max_fix_retries == 7

    def test_bad_yaml_raises(self, repo):
        agent_dir = repo / ".agent"
        agent_dir.mkdir()
        (agent_dir / "config.yaml").write_text("{unclosed: [bracket\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="config.yaml"):
            load_config(repo)

    def test_no_yaml_file_uses_defaults(self, repo):
        # No .agent/config.yaml — should use defaults without error
        config = load_config(repo)
        assert config.max_fix_retries == 3

    def test_empty_yaml_uses_defaults(self, repo):
        agent_dir = repo / ".agent"
        agent_dir.mkdir()
        (agent_dir / "config.yaml").write_text("", encoding="utf-8")
        config = load_config(repo)
        assert config.max_fix_retries == 3

    def test_services_default_empty(self, repo):
        config = load_config(repo)
        assert config.services == []


class TestApplyOverrides:
    def _make_config(self) -> Config:
        tmp = tempfile.mkdtemp()
        return Config(repo_path=Path(tmp), claude_md_content="# test")

    def test_llm_provider_overridden(self):
        config = self._make_config()
        _apply_overrides(config, {"llm": {"provider": "openrouter"}})
        assert config.llm_provider == "openrouter"

    def test_llm_model_overridden(self):
        config = self._make_config()
        _apply_overrides(config, {"llm": {"model": "gpt-4o"}})
        assert config.llm_model == "gpt-4o"

    def test_max_fix_retries_overridden(self):
        config = self._make_config()
        _apply_overrides(config, {"execution": {"max_fix_retries": 9}})
        assert config.max_fix_retries == 9

    def test_timeout_overridden(self):
        config = self._make_config()
        _apply_overrides(config, {"execution": {"test_timeout_seconds": 60}})
        assert config.test_timeout_seconds == 60

    def test_max_output_lines_overridden(self):
        config = self._make_config()
        _apply_overrides(config, {"execution": {"max_output_lines": 50}})
        assert config.max_output_lines == 50

    def test_services_overridden(self):
        config = self._make_config()
        svcs = [{"name": "web", "command": "npm run dev", "port": 3000}]
        _apply_overrides(config, {"services": svcs})
        assert config.services == svcs

    def test_empty_overrides_no_change(self):
        config = self._make_config()
        original_model = config.llm_model
        original_retries = config.max_fix_retries
        _apply_overrides(config, {})
        assert config.llm_model == original_model
        assert config.max_fix_retries == original_retries

    def test_non_dict_llm_section_ignored(self):
        config = self._make_config()
        original = config.llm_model
        _apply_overrides(config, {"llm": "not-a-dict"})
        assert config.llm_model == original

    def test_non_list_services_ignored(self):
        config = self._make_config()
        _apply_overrides(config, {"services": "not-a-list"})
        assert config.services == []
