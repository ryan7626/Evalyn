from pathlib import Path

import pytest

from evalyn.run_store import RunStore
from evalyn.settings import SettingsError, load_settings, render_default_config


def test_missing_config_uses_safe_defaults(tmp_path: Path) -> None:
    settings = load_settings(tmp_path)

    assert settings.repo_path == tmp_path.resolve()
    assert settings.provider.kind == "anthropic"
    assert settings.repeat == 3
    assert settings.sandbox.runtime == "auto"


def test_loads_provider_adapters_and_limits(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    image = "ghcr.io/example/tests@sha256:" + "a" * 64
    config = tmp_path / "operator.yml"
    config.write_text(
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

    settings = load_settings(repo, config)

    assert settings.provider.model == "local-model"
    assert settings.adapters[0].image == image
    assert settings.adapters[0].setup_argv == ("npm", "ci")
    assert settings.sandbox.setup_env == ("NPM_TOKEN",)
    assert settings.max_tests == 5


@pytest.mark.parametrize(
    ("body", "message"),
    (
        (
            "version: 1\nprovider:\n  kind: openai-compatible\n"
            "  base_url: https://models.example/v1\n",
            "operator policy",
        ),
        (
            "version: 1\nsandbox:\n  setup_env: [NPM_TOKEN]\n",
            "forwards operator secrets",
        ),
    ),
)
def test_repository_config_cannot_select_credential_destinations(
    tmp_path: Path, body: str, message: str
) -> None:
    (tmp_path / ".evalyn.yml").write_text(body, encoding="utf-8")

    with pytest.raises(SettingsError, match=message):
        load_settings(tmp_path)


def test_explicit_missing_config_is_not_silently_ignored(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match="does not exist"):
        load_settings(tmp_path, tmp_path.parent / "missing-operator.yml")


@pytest.mark.parametrize(
    "body",
    (
        "version: 1\nprovider:\n  kind: true\n",
        "version: 1\nprovider:\n  model: [not, a, string]\n",
        "version: 1\nsandbox:\n  runtime: 1\n",
        "version: 1\nadapters:\n  - name: true\n",
    ),
)
def test_named_configuration_fields_require_exact_strings(tmp_path: Path, body: str) -> None:
    (tmp_path / ".evalyn.yml").write_text(body, encoding="utf-8")

    with pytest.raises(SettingsError, match="string"):
        load_settings(tmp_path)


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


@pytest.mark.parametrize(
    "field, value",
    [
        ("max_tests", "'5'"),
        ("max_tests", "5.0"),
        ("max_tests", "true"),
        ("repeat", "3.0"),
        ("budget_seconds", "'600'"),
    ],
)
def test_limits_require_exact_yaml_integers(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    (tmp_path / ".evalyn.yml").write_text(
        f"version: 1\nlimits:\n  {field}: {value}\n",
        encoding="utf-8",
    )

    with pytest.raises(SettingsError, match="positive integer"):
        load_settings(tmp_path)


@pytest.mark.parametrize(
    "field, value, maximum",
    [("max_tests", 101, 100), ("budget_seconds", 86_401, 86_400)],
)
def test_limits_have_resource_safety_upper_bounds(
    tmp_path: Path,
    field: str,
    value: int,
    maximum: int,
) -> None:
    (tmp_path / ".evalyn.yml").write_text(
        f"version: 1\nlimits:\n  {field}: {value}\n",
        encoding="utf-8",
    )

    with pytest.raises(SettingsError, match=f"at most {maximum}"):
        load_settings(tmp_path)


@pytest.mark.parametrize("version", ["true", "1.0", "'1'"])
def test_schema_version_requires_the_exact_integer_one(tmp_path: Path, version: str) -> None:
    (tmp_path / ".evalyn.yml").write_text(f"version: {version}\n", encoding="utf-8")

    with pytest.raises(SettingsError, match="schema version 1"):
        load_settings(tmp_path)


def test_rejects_unknown_limit_keys(tmp_path: Path) -> None:
    (tmp_path / ".evalyn.yml").write_text(
        "version: 1\nlimits:\n  unlimited: 1\n",
        encoding="utf-8",
    )

    with pytest.raises(SettingsError, match="Unknown limits keys"):
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


def test_rejects_configuration_with_linked_parent(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "operator.yml").write_text("version: 1\n", encoding="utf-8")
    linked_parent = repo / "linked"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")

    with pytest.raises(SettingsError, match="symlink"):
        load_settings(repo, Path("linked/operator.yml"))


def test_load_reads_the_resolved_configuration_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    config = repo / "operator.yml"
    config.write_text("version: 1\n", encoding="utf-8")
    observed: list[Path] = []
    real_read_bytes = RunStore.read_bytes

    def recording_read(path: Path, *, max_bytes: int) -> bytes:
        observed.append(path)
        return real_read_bytes(path, max_bytes=max_bytes)

    monkeypatch.setattr(RunStore, "read_bytes", recording_read)

    load_settings(repo, Path("unused/../operator.yml"))

    assert observed == [config.resolve()]
