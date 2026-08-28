from pathlib import Path

import pytest

from evalyn.settings import SettingsError, load_settings, render_default_config


def test_missing_config_uses_safe_defaults(tmp_path: Path) -> None:
    settings = load_settings(tmp_path)

    assert settings.repo_path == tmp_path.resolve()
    assert settings.provider.kind == "anthropic"
    assert settings.repeat == 3
    assert settings.sandbox.runtime == "auto"


def test_loads_provider_adapters_and_limits(tmp_path: Path) -> None:
    image = "ghcr.io/example/tests@sha256:" + "a" * 64
    (tmp_path / ".evalyn.yml").write_text(
        f"""\
version: 1
provider:
  kind: openai-compatible
  model: local-model
  base_url: http://localhost:11434/v1
sandbox:
  runtime: docker
  setup_env: [NPM_TOKEN]
adapters:
  - name: vitest
    image: {image}
    setup_argv: [npm, ci]
    suite_argv: [npm, test]
limits:
  max_tests: 5
  repeat: 2
  budget_seconds: 30
""",
        encoding="utf-8",
    )

    settings = load_settings(tmp_path)

    assert settings.provider.model == "local-model"
    assert settings.adapters[0].image == image
    assert settings.adapters[0].setup_argv == ("npm", "ci")
    assert settings.sandbox.setup_env == ("NPM_TOKEN",)
    assert settings.max_tests == 5


@pytest.mark.parametrize(
    "body, message",
    [
        ("version: 2\n", "schema version 1"),
        ("version: 1\nunknown: true\n", "Unknown configuration keys"),
        ("version: 1\nlimits:\n  repeat: 1\n", "at least 2"),
        ("version: 1\nadapters:\n  - name: jest\n", "pytest or vitest"),
        ("version: 1\nsandbox:\n  runtime: host\n", "auto, docker, or podman"),
        (
            "version: 1\nadapters:\n  - name: pytest\n    image: 'bad image'\n",
            "OCI image reference",
        ),
    ],
)
def test_rejects_unsupported_configuration(tmp_path: Path, body: str, message: str) -> None:
    (tmp_path / ".evalyn.yml").write_text(body, encoding="utf-8")

    with pytest.raises(SettingsError, match=message):
        load_settings(tmp_path)


def test_default_config_is_round_trippable(tmp_path: Path) -> None:
    (tmp_path / ".evalyn.yml").write_text(
        render_default_config(["pytest", "vitest"]), encoding="utf-8"
    )

    settings = load_settings(tmp_path)

    assert [adapter.name for adapter in settings.adapters] == ["pytest", "vitest"]


def test_rejects_model_credentials_in_setup_environment(tmp_path: Path) -> None:
    (tmp_path / ".evalyn.yml").write_text(
        "version: 1\nsandbox:\n  setup_env: [OPENAI_API_KEY]\n",
        encoding="utf-8",
    )

    with pytest.raises(SettingsError, match="package-registry"):
        load_settings(tmp_path)


def test_rejects_more_than_ten_repetitions(tmp_path: Path) -> None:
    (tmp_path / ".evalyn.yml").write_text(
        "version: 1\nlimits:\n  repeat: 11\n",
        encoding="utf-8",
    )

    with pytest.raises(SettingsError, match="at most 10"):
        load_settings(tmp_path)


def test_rejects_linked_or_oversized_configuration(tmp_path: Path) -> None:
    outside = tmp_path / "outside.yml"
    outside.write_text("version: 1\n", encoding="utf-8")
    config = tmp_path / ".evalyn.yml"
    config.symlink_to(outside)

    with pytest.raises(SettingsError, match="symlink"):
        load_settings(tmp_path)

    config.unlink()
    config.write_text("#" + ("x" * (256 * 1024)), encoding="utf-8")
    with pytest.raises(SettingsError, match="too large"):
        load_settings(tmp_path)
