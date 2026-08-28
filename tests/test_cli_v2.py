import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import evalyn.cli as cli_module
from evalyn.cli import _resolve_output_destination, main


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test")
    (repo / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    (repo / "value.py").write_text("VALUE = 1\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "base")
    return repo


def test_help_exposes_diff_workflow_and_removes_legacy_commands() -> None:
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "check" in result.output
    assert "reproduce" in result.output
    assert "plan" not in result.output


def test_init_writes_optional_schema_v1_config(tmp_path: Path) -> None:
    repo = repository(tmp_path)

    result = CliRunner().invoke(main, ["--repo", str(repo), "init"])

    assert result.exit_code == 0
    assert (repo / ".evalyn.yml").read_text().startswith("version: 1\n")
    assert not (repo / "CLAUDE.md").exists()
    assert not (repo / ".agent").exists()


def test_init_force_does_not_follow_a_config_symlink(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    victim = tmp_path / "victim.yml"
    victim.write_text("keep\n", encoding="utf-8")
    (repo / ".evalyn.yml").symlink_to(victim)

    result = CliRunner().invoke(main, ["--repo", str(repo), "init", "--force"])

    assert result.exit_code == 1
    assert victim.read_text(encoding="utf-8") == "keep\n"


def test_check_dry_run_needs_no_key_or_container(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    git(repo, "checkout", "-b", "feature")
    (repo / "value.py").write_text("VALUE = 2\n")
    git(repo, "commit", "-am", "change")

    result = CliRunner().invoke(
        main,
        ["--repo", str(repo), "check", "main...HEAD", "--dry-run", "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["adapters"] == ["pytest"]
    assert payload["changed_files"] == ["value.py"]


def test_dirty_dry_run_requires_explicit_opt_in(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    (repo / "value.py").write_text("VALUE = 2\n")

    result = CliRunner().invoke(main, ["--repo", str(repo), "check", "--dry-run"])

    assert result.exit_code == 2
    assert "--include-dirty" in result.output


def test_budget_rejects_ambiguous_duration(tmp_path: Path) -> None:
    repo = repository(tmp_path)

    result = CliRunner().invoke(
        main, ["--repo", str(repo), "check", "--budget", "tomorrow", "--dry-run"]
    )

    assert result.exit_code == 2
    assert "90s, 10m, or 1h" in result.output


def test_check_rejects_relative_output_escape_before_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = repository(tmp_path)

    def unexpected_run(_options):
        raise AssertionError("unsafe output should be rejected before the check runs")

    monkeypatch.setattr(cli_module, "run_check", unexpected_run)
    result = CliRunner().invoke(
        main,
        ["--repo", str(repo), "check", "--output", "../escape.sarif"],
    )

    assert result.exit_code == 2
    assert "inside the target repository" in result.output
    assert not (tmp_path / "escape.sarif").exists()


def test_output_destination_allows_only_absolute_paths_inside_repo(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    inside = repo / "artifacts" / "result.sarif"

    assert _resolve_output_destination(repo, inside) == inside
    with pytest.raises(ValueError, match="absolute output must resolve inside"):
        _resolve_output_destination(repo, tmp_path / "outside.sarif")
    with pytest.raises(ValueError, match="Git metadata"):
        _resolve_output_destination(repo, repo / ".GIT" / "result.sarif")


def test_check_rejects_symlink_output_before_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = repository(tmp_path)
    outside = tmp_path / "outside.sarif"
    outside.write_text("preserve me", encoding="utf-8")
    (repo / "result.sarif").symlink_to(outside)

    def unexpected_run(_options):
        raise AssertionError("symlink output should be rejected before the check runs")

    monkeypatch.setattr(cli_module, "run_check", unexpected_run)
    result = CliRunner().invoke(
        main,
        ["--repo", str(repo), "check", "--output", "result.sarif"],
    )

    assert result.exit_code == 2
    assert "cannot contain symlinks" in result.output
    assert outside.read_text(encoding="utf-8") == "preserve me"


def test_post_run_report_read_failure_is_incomplete_even_with_findings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = repository(tmp_path)
    run_path = tmp_path / "run"
    run_path.mkdir()
    outcome = SimpleNamespace(
        run=SimpleNamespace(path=run_path),
        report_paths={"markdown": run_path / "missing-report.md"},
        incomplete=False,
        gating_findings=(object(),),
    )
    monkeypatch.setattr(cli_module, "run_check", lambda _options: outcome)

    result = CliRunner().invoke(main, ["--repo", str(repo), "check"])

    assert result.exit_code == 3
    assert "Incomplete" in result.output
    assert "Could not emit check output" in result.output


def test_incomplete_check_exit_wins_over_finding_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = repository(tmp_path)
    outcome = SimpleNamespace(incomplete=True, gating_findings=(object(),))
    monkeypatch.setattr(cli_module, "run_check", lambda _options: outcome)
    monkeypatch.setattr(cli_module, "_emit_check_output", lambda *_args: None)

    result = CliRunner().invoke(main, ["--repo", str(repo), "check"])

    assert result.exit_code == 3


@pytest.mark.parametrize(
    "command",
    (
        ("show", "EVL-../../.."),
        ("reproduce", "EVL-../../.."),
        ("triage", "EVL-../../..", "--confirm", "--reason", "unsafe"),
        ("repair", "EVL-../../..", "--emit-patch", "repair.patch"),
    ),
)
def test_cli_rejects_invalid_finding_ids(tmp_path: Path, command: tuple[str, ...]) -> None:
    repo = repository(tmp_path)

    result = CliRunner().invoke(main, ["--repo", str(repo), *command])

    assert result.exit_code == 2
    assert "finding_id must match EVL-XXXXXXXX" in result.output
