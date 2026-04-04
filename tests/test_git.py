"""Tests for evalyn.utils.git module."""

import subprocess
from pathlib import Path

import pytest

from evalyn.utils.git import (
    get_current_branch,
    get_head_sha,
    get_commits_since,
    get_changed_files,
    create_branch,
    checkout_branch,
    create_patch,
    apply_patch,
    reset_hard,
    GitError,
)


@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary git repository with an initial commit."""
    repo = tmp_path / "test_repo"
    repo.mkdir()

    subprocess.run(["git", "init"], cwd=repo, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo, capture_output=True,
    )

    # Create initial commit
    (repo / "README.md").write_text("# Test Repo")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo, capture_output=True,
    )

    return repo


class TestGetCurrentBranch:
    def test_returns_branch_name(self, git_repo):
        branch = get_current_branch(git_repo)
        # Could be "main" or "master" depending on git config
        assert branch in ("main", "master")


class TestGetHeadSha:
    def test_returns_sha(self, git_repo):
        sha = get_head_sha(git_repo)
        assert len(sha) == 40
        assert all(c in "0123456789abcdef" for c in sha)


class TestGetCommitsSince:
    def test_no_new_commits(self, git_repo):
        sha = get_head_sha(git_repo)
        commits = get_commits_since(git_repo, sha)
        assert commits == []

    def test_finds_new_commits(self, git_repo):
        initial_sha = get_head_sha(git_repo)

        # Make a new commit
        (git_repo / "new_file.txt").write_text("hello")
        subprocess.run(["git", "add", "."], cwd=git_repo, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Add new file"],
            cwd=git_repo, capture_output=True,
        )

        commits = get_commits_since(git_repo, initial_sha)
        assert len(commits) == 1
        assert commits[0].message == "Add new file"


class TestGetChangedFiles:
    def test_detects_changed_files(self, git_repo):
        initial_sha = get_head_sha(git_repo)

        (git_repo / "new.txt").write_text("new")
        subprocess.run(["git", "add", "."], cwd=git_repo, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Add"],
            cwd=git_repo, capture_output=True,
        )

        files = get_changed_files(git_repo, initial_sha)
        assert "new.txt" in files


class TestBranching:
    def test_create_and_checkout_branch(self, git_repo):
        create_branch(git_repo, "test-branch")
        assert get_current_branch(git_repo) == "test-branch"

        # Switch back
        original = "main" if "main" in subprocess.run(
            ["git", "branch"], cwd=git_repo, capture_output=True, text=True
        ).stdout else "master"
        checkout_branch(git_repo, original)
        assert get_current_branch(git_repo) == original


class TestPatchOperations:
    def test_create_and_apply_patch(self, git_repo, tmp_path):
        # Make uncommitted changes
        (git_repo / "README.md").write_text("# Modified")
        subprocess.run(["git", "add", "."], cwd=git_repo, capture_output=True)

        # Create patch (stored outside repo so git clean doesn't remove it)
        patch_path = tmp_path / "patches" / "test.patch"
        created = create_patch(git_repo, patch_path)
        assert created is True
        assert patch_path.is_file()

        # Reset and apply
        reset_hard(git_repo)
        assert (git_repo / "README.md").read_text() == "# Test Repo"

        apply_patch(git_repo, patch_path)
        assert (git_repo / "README.md").read_text() == "# Modified"

    def test_no_patch_when_clean(self, git_repo):
        patch_path = git_repo / "empty.patch"
        created = create_patch(git_repo, patch_path)
        assert created is False


class TestResetHard:
    def test_discards_changes(self, git_repo):
        (git_repo / "README.md").write_text("# Modified")
        reset_hard(git_repo)
        assert (git_repo / "README.md").read_text() == "# Test Repo"


class TestGitError:
    def test_raises_on_invalid_repo(self, tmp_path):
        with pytest.raises(GitError):
            get_head_sha(tmp_path)
