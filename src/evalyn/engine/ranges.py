"""Read-only Git revision-range parsing and resolution."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from evalyn.execution.bounded import run_bounded_process
from evalyn.execution.git_safety import sanitized_git_command, sanitized_git_environment

_GIT_OUTPUT_LIMIT = 64 * 1024
_GIT_OID = re.compile(r"[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?\Z")


class RangeResolutionError(ValueError):
    """Raised when a requested Git range cannot be resolved safely."""


class RangeMode(str, Enum):
    """Whether a range uses its named base or the refs' merge base."""

    DIRECT = "direct"
    MERGE_BASE = "merge_base"


@dataclass(frozen=True)
class RequestedRange:
    base_ref: str
    candidate_ref: str
    mode: RangeMode


@dataclass(frozen=True)
class ResolvedRange:
    base_ref: str
    candidate_ref: str
    base_sha: str
    candidate_sha: str
    mode: RangeMode


class GitRunner(Protocol):
    def run(
        self,
        repo_path: Path,
        *args: str,
        check: bool = True,
    ) -> str:
        """Run a read-only Git command and return stripped stdout."""


class SubprocessGitRunner:
    """Minimal read-only Git runner used by the default resolver."""

    def run(
        self,
        repo_path: Path,
        *args: str,
        check: bool = True,
    ) -> str:
        try:
            result = run_bounded_process(
                sanitized_git_command(repo_path, *args),
                timeout=30,
                env=sanitized_git_environment(),
                max_stdout_bytes=_GIT_OUTPUT_LIMIT,
                max_stderr_bytes=_GIT_OUTPUT_LIMIT,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            raise RangeResolutionError(f"could not execute git: {exc}") from exc
        if result.stdout_truncated or result.stderr_truncated:
            raise RangeResolutionError("git command output exceeded 64 KiB")
        stdout = result.stdout.decode("utf-8", errors="replace").strip()
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        if check and result.returncode != 0:
            raise RangeResolutionError(stderr or "git command failed")
        return stdout


def parse_range_spec(spec: str, *, default_head: str = "HEAD") -> RequestedRange:
    """Parse ``base``, ``base..head``, or ``base...head`` syntax."""
    value = spec.strip()
    if not value:
        raise RangeResolutionError("range cannot be empty")
    if "..." in value:
        parts = value.split("...")
        if len(parts) != 2 or not all(parts):
            raise RangeResolutionError("range must contain exactly one base and head")
        return RequestedRange(parts[0], parts[1], RangeMode.MERGE_BASE)
    if ".." in value:
        parts = value.split("..")
        if len(parts) != 2 or not all(parts):
            raise RangeResolutionError("range must contain exactly one base and head")
        return RequestedRange(parts[0], parts[1], RangeMode.DIRECT)
    if not default_head.strip():
        raise RangeResolutionError("default head cannot be empty")
    return RequestedRange(value, default_head, RangeMode.MERGE_BASE)


class GitRangeResolver:
    """Resolve user refs to immutable base and candidate commit SHAs."""

    _FALLBACK_BASE_REFS = ("origin/main", "main", "origin/master", "master")

    def __init__(self, git: GitRunner | None = None) -> None:
        self.git = git or SubprocessGitRunner()

    def resolve(
        self,
        repo_path: Path,
        spec: str | None = None,
        *,
        head: str = "HEAD",
    ) -> ResolvedRange:
        requested = (
            parse_range_spec(spec, default_head=head)
            if spec is not None
            else RequestedRange(self._default_base(repo_path), head, RangeMode.MERGE_BASE)
        )
        candidate_sha = self._resolve_commit(repo_path, requested.candidate_ref)
        if requested.mode == RangeMode.MERGE_BASE:
            named_base_sha = self._resolve_commit(repo_path, requested.base_ref)
            base_sha = self.git.run(
                repo_path,
                "merge-base",
                named_base_sha,
                candidate_sha,
            )
            if _GIT_OID.fullmatch(base_sha) is None:
                raise RangeResolutionError("base and candidate have no merge base")
        else:
            base_sha = self._resolve_commit(repo_path, requested.base_ref)

        if base_sha == candidate_sha:
            raise RangeResolutionError("base and candidate resolve to the same commit")
        return ResolvedRange(
            base_ref=requested.base_ref,
            candidate_ref=requested.candidate_ref,
            base_sha=base_sha,
            candidate_sha=candidate_sha,
            mode=requested.mode,
        )

    def _resolve_commit(self, repo_path: Path, ref: str) -> str:
        if not ref or ref.startswith("-") or any(character.isspace() for character in ref):
            raise RangeResolutionError(f"invalid Git ref: {ref!r}")
        value = self.git.run(repo_path, "rev-parse", "--verify", f"{ref}^{{commit}}")
        if _GIT_OID.fullmatch(value) is None:
            raise RangeResolutionError(f"ref did not resolve to a commit: {ref}")
        return value

    def _default_base(self, repo_path: Path) -> str:
        upstream = self.git.run(
            repo_path,
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
            check=False,
        )
        if upstream:
            return upstream

        for ref in self._FALLBACK_BASE_REFS:
            resolved = self.git.run(
                repo_path,
                "rev-parse",
                "--verify",
                f"{ref}^{{commit}}",
                check=False,
            )
            if resolved:
                return ref
        raise RangeResolutionError(
            "could not infer a base ref; pass an explicit range such as main...HEAD"
        )
