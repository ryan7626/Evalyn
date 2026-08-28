"""Tests for read-only Git range parsing and immutable resolution."""

from dataclasses import dataclass, field

import pytest

from evalyn.engine.ranges import (
    GitRangeResolver,
    RangeMode,
    RangeResolutionError,
    parse_range_spec,
)

CANDIDATE_SHA = "c" * 40
NAMED_BASE_SHA = "b" * 40
MERGE_BASE_SHA = "a" * 40


@dataclass
class FakeGit:
    responses: dict[tuple[str, ...], str]
    calls: list[tuple[tuple[str, ...], bool]] = field(default_factory=list)

    def run(self, repo_path, *args, check=True):
        self.calls.append((args, check))
        return self.responses.get(args, "")


def test_parse_triple_dot_uses_merge_base():
    requested = parse_range_spec("main...feature")

    assert requested.base_ref == "main"
    assert requested.candidate_ref == "feature"
    assert requested.mode == RangeMode.MERGE_BASE


def test_parse_double_dot_uses_named_base():
    requested = parse_range_spec("release..HEAD")

    assert requested.mode == RangeMode.DIRECT


def test_parse_single_ref_defaults_candidate_to_head():
    requested = parse_range_spec("origin/main")

    assert requested.base_ref == "origin/main"
    assert requested.candidate_ref == "HEAD"
    assert requested.mode == RangeMode.MERGE_BASE


@pytest.mark.parametrize("spec", ["", "...HEAD", "main...", "a...b...c", "a..b..c"])
def test_rejects_malformed_ranges(spec):
    with pytest.raises(RangeResolutionError):
        parse_range_spec(spec)


def test_resolves_merge_base_to_immutable_shas(tmp_path):
    git = FakeGit(
        {
            ("rev-parse", "--verify", "feature^{commit}"): CANDIDATE_SHA,
            ("rev-parse", "--verify", "main^{commit}"): NAMED_BASE_SHA,
            ("merge-base", NAMED_BASE_SHA, CANDIDATE_SHA): MERGE_BASE_SHA,
        }
    )

    result = GitRangeResolver(git).resolve(tmp_path, "main...feature")

    assert result.base_sha == MERGE_BASE_SHA
    assert result.candidate_sha == CANDIDATE_SHA
    assert result.mode == RangeMode.MERGE_BASE


def test_direct_range_does_not_call_merge_base(tmp_path):
    git = FakeGit(
        {
            ("rev-parse", "--verify", "HEAD^{commit}"): CANDIDATE_SHA,
            ("rev-parse", "--verify", "base^{commit}"): NAMED_BASE_SHA,
        }
    )

    result = GitRangeResolver(git).resolve(tmp_path, "base..HEAD")

    assert result.base_sha == NAMED_BASE_SHA
    assert all(call[0][0] != "merge-base" for call in git.calls)


def test_default_range_prefers_configured_upstream(tmp_path):
    git = FakeGit(
        {
            (
                "rev-parse",
                "--abbrev-ref",
                "--symbolic-full-name",
                "@{upstream}",
            ): "origin/trunk",
            ("rev-parse", "--verify", "HEAD^{commit}"): CANDIDATE_SHA,
            ("rev-parse", "--verify", "origin/trunk^{commit}"): NAMED_BASE_SHA,
            ("merge-base", NAMED_BASE_SHA, CANDIDATE_SHA): MERGE_BASE_SHA,
        }
    )

    result = GitRangeResolver(git).resolve(tmp_path)

    assert result.base_ref == "origin/trunk"
    assert result.base_sha == MERGE_BASE_SHA


def test_same_resolved_commit_is_rejected(tmp_path):
    same_sha = "d" * 40
    git = FakeGit(
        {
            ("rev-parse", "--verify", "HEAD^{commit}"): same_sha,
            ("rev-parse", "--verify", "base^{commit}"): same_sha,
        }
    )

    with pytest.raises(RangeResolutionError, match="same commit"):
        GitRangeResolver(git).resolve(tmp_path, "base..HEAD")
