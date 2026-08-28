"""Tests for fail-closed workspace path handling."""

from pathlib import Path

import pytest

from evalyn.execution.errors import PathPolicyError
from evalyn.execution.paths import WorkspacePathPolicy


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "src" / "other.py").write_text("VALUE = 2\n", encoding="utf-8")
    (root / "tests").mkdir()
    (root / "tests" / "helpers.py").write_text("# helper\n", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("secret\n", encoding="utf-8")
    return root


@pytest.fixture
def policy(workspace: Path) -> WorkspacePathPolicy:
    return WorkspacePathPolicy(
        workspace,
        tracked_paths=["src/app.py", "src/other.py", "tests/helpers.py", ".git/config"],
        context_allowlist=["src/app.py"],
        fix_allowlist=["src/app.py"],
    )


@pytest.mark.parametrize(
    "unsafe",
    ["../secret.txt", "src/../../secret.txt", "/etc/hosts", "C:\\secret.txt", "bad\x00.py"],
)
def test_rejects_absolute_traversal_and_nul(
    policy: WorkspacePathPolicy,
    unsafe: str,
) -> None:
    with pytest.raises(PathPolicyError):
        policy.resolve_read(unsafe)


def test_context_requires_tracked_and_explicitly_allowed_file(
    policy: WorkspacePathPolicy,
) -> None:
    assert policy.resolve_context("src/app.py").name == "app.py"

    with pytest.raises(PathPolicyError, match="context allowlist"):
        policy.resolve_context("src/other.py")


def test_fix_target_requires_fix_allowlist(policy: WorkspacePathPolicy) -> None:
    assert policy.resolve_fix_target("src/app.py").name == "app.py"

    with pytest.raises(PathPolicyError, match="fix allowlist"):
        policy.resolve_fix_target("src/other.py")


@pytest.mark.parametrize("protected", [".git/config", "tests/helpers.py"])
def test_context_and_fix_targets_reject_protected_paths(
    policy: WorkspacePathPolicy,
    protected: str,
) -> None:
    with pytest.raises(PathPolicyError, match="Protected"):
        policy.resolve_context(protected)
    with pytest.raises(PathPolicyError, match="Protected"):
        policy.resolve_fix_target(protected)


def test_rejects_symlink_even_when_target_stays_inside_workspace(
    workspace: Path,
    policy: WorkspacePathPolicy,
) -> None:
    link = workspace / "src" / "link.py"
    try:
        link.symlink_to(workspace / "src" / "app.py")
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(PathPolicyError, match="Symlink"):
        policy.resolve_read("src/link.py")


def test_rejects_symlink_escape(workspace: Path, policy: WorkspacePathPolicy) -> None:
    outside = workspace.parent / "outside.txt"
    outside.write_text("do not read\n", encoding="utf-8")
    link = workspace / "src" / "outside.py"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(PathPolicyError, match="escapes workspace"):
        policy.resolve_read("src/outside.py")


def test_generated_test_path_ignores_directory_shaped_stem(
    policy: WorkspacePathPolicy,
) -> None:
    generated = policy.generated_test_path("../../Auth / Browser", suffix=".PY")

    assert generated.parent == policy.workspace_root / ".evalyn-generated"
    assert generated.name == "Auth_Browser.py"


def test_fix_resolution_fails_closed_without_allowlists(workspace: Path) -> None:
    policy = WorkspacePathPolicy(workspace)

    with pytest.raises(PathPolicyError, match="tracked-file"):
        policy.resolve_fix_target("src/app.py")
