"""Disposable dual-revision Git worktrees for safe Evalyn execution."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Literal

from evalyn.execution.bounded import (
    BoundedProcessResult as _BoundedProcessResult,
)
from evalyn.execution.bounded import run_bounded_process as _run_bounded_process
from evalyn.execution.errors import WorktreeCleanupError, WorktreeError

RevisionRole = Literal["baseline", "candidate"]

_GIT_STDOUT_LIMIT = 8 * 1024 * 1024
_GIT_STDERR_LIMIT = 256 * 1024
_MANIFEST_LIMIT = 64 * 1024


@dataclass(frozen=True)
class DirtyOverlay:
    """Snapshot of user changes copied into the candidate worktree."""

    tracked_patch_bytes: int = 0
    untracked_paths: tuple[str, ...] = ()
    content_sha256: str = ""

    @property
    def is_empty(self) -> bool:
        return self.tracked_patch_bytes == 0 and not self.untracked_paths


@dataclass(frozen=True)
class WorktreePair:
    """Resolved immutable revisions and their disposable checkout paths."""

    baseline_sha: str
    candidate_sha: str
    baseline_path: Path
    candidate_path: Path


class WorktreeSession:
    """Create and own detached baseline/candidate worktrees.

    The source checkout is never checked out, reset, cleaned, or otherwise
    mutated. When ``include_dirty_overlay`` is enabled, tracked and untracked
    user changes are copied into the candidate worktree while baseline remains
    pinned to its commit.
    """

    MANIFEST_NAME = ".evalyn-worktree-session.json"

    def __init__(
        self,
        repo_path: Path,
        *,
        baseline_revision: str,
        candidate_revision: str = "HEAD",
        include_dirty_overlay: bool = False,
        temp_parent: Path | None = None,
        max_overlay_bytes: int = 50 * 1024 * 1024,
        max_candidate_diff_bytes: int = 50 * 1024 * 1024,
        git_timeout_seconds: float = 60.0,
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.baseline_revision = baseline_revision
        self.candidate_revision = candidate_revision
        self.include_dirty_overlay = include_dirty_overlay
        self.temp_parent = Path(temp_parent).resolve() if temp_parent else None
        self.max_overlay_bytes = max(0, int(max_overlay_bytes))
        self.max_candidate_diff_bytes = max(0, int(max_candidate_diff_bytes))
        self.git_timeout_seconds = max(1.0, float(git_timeout_seconds))
        self.session_id = uuid.uuid4().hex

        self.session_root: Path | None = None
        self.manifest_path: Path | None = None
        self.pair: WorktreePair | None = None
        self.overlay = DirtyOverlay()
        self.cleanup_errors: tuple[str, ...] = ()
        self._created_paths: list[Path] = []
        self._overlay_patch = b""
        self._overlay_untracked: dict[str, tuple[bytes, int]] = {}
        self._hooks_path: Path | None = None
        self._filter_names: tuple[str, ...] = ()
        self._active = False

    @property
    def baseline_path(self) -> Path:
        return self._require_pair().baseline_path

    @property
    def candidate_path(self) -> Path:
        return self._require_pair().candidate_path

    @property
    def baseline_sha(self) -> str:
        return self._require_pair().baseline_sha

    @property
    def candidate_sha(self) -> str:
        return self._require_pair().candidate_sha

    @property
    def is_active(self) -> bool:
        return self._active

    def start(self) -> WorktreeSession:
        """Resolve revisions, create both worktrees, and apply any overlay."""
        if self._active:
            return self
        self._validate_repository()

        baseline_sha = self._resolve_revision(self.baseline_revision)
        candidate_sha = self._resolve_revision(self.candidate_revision)
        head_sha = self._resolve_revision("HEAD")
        if self.include_dirty_overlay and candidate_sha != head_sha:
            raise WorktreeError("A dirty overlay can only be applied to the source checkout's HEAD")

        if self.include_dirty_overlay:
            self._capture_dirty_overlay()

        if self.temp_parent is not None:
            self.temp_parent.mkdir(parents=True, exist_ok=True)
        root = Path(
            tempfile.mkdtemp(
                prefix="evalyn-worktrees-",
                dir=str(self.temp_parent) if self.temp_parent else None,
            )
        ).resolve()
        self.session_root = root
        self._hooks_path = root / ".empty-hooks"
        self._hooks_path.mkdir(mode=0o700)
        self._filter_names = self._configured_filter_names()
        self.manifest_path = root / self.MANIFEST_NAME
        baseline_path = root / "baseline"
        candidate_path = root / "candidate"
        self.pair = WorktreePair(
            baseline_sha=baseline_sha,
            candidate_sha=candidate_sha,
            baseline_path=baseline_path,
            candidate_path=candidate_path,
        )
        self._write_manifest("preparing")

        try:
            self._add_worktree(baseline_path, baseline_sha)
            self._add_worktree(candidate_path, candidate_sha)
            if self.include_dirty_overlay:
                self._apply_overlay()
            self._active = True
            self._write_manifest("active")
            return self
        except BaseException as exc:
            cleanup_errors = self._cleanup_owned_paths()
            detail = f" Cleanup errors: {'; '.join(cleanup_errors)}" if cleanup_errors else ""
            if isinstance(exc, WorktreeError):
                raise WorktreeError(f"{exc}{detail}") from exc
            raise WorktreeError(f"Failed to prepare worktrees: {exc}.{detail}") from exc

    def restore_candidate(self) -> None:
        """Discard candidate mutations and recreate its starting overlay."""
        pair = self._require_pair()
        self._git_in(pair.candidate_path, "reset", "--hard", pair.candidate_sha)
        # This is intentionally scoped to an owned disposable worktree.
        self._git_in(pair.candidate_path, "clean", "-fdx")
        if self.include_dirty_overlay:
            self._apply_overlay()

    def tracked_files(self, role: RevisionRole = "candidate") -> tuple[str, ...]:
        """List tracked paths for building a :class:`WorkspacePathPolicy`."""
        path = self.path_for(role)
        output = self._git_in_bytes(path, "ls-files", "-z")
        return tuple(
            item.decode("utf-8", errors="surrogateescape") for item in output.split(b"\0") if item
        )

    def path_for(self, role: RevisionRole) -> Path:
        pair = self._require_pair()
        if role == "baseline":
            return pair.baseline_path
        if role == "candidate":
            return pair.candidate_path
        raise ValueError(f"Unknown revision role: {role!r}")

    def candidate_diff(self, paths: Iterable[str] | None = None) -> bytes:
        """Return a binary patch from candidate SHA to current candidate state.

        This includes the dirty overlay when one was requested. Integrators that
        need a repair-only patch should pass the validated fix ledger as
        ``paths`` and retain the overlay manifest alongside the artifact.
        """
        pair = self._require_pair()
        args = [
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--binary",
            "--full-index",
            pair.candidate_sha,
        ]
        if paths is not None:
            safe_paths = [_validate_git_relative(path) for path in paths]
            args.extend(["--", *safe_paths])
        return self._git_in_bytes(
            pair.candidate_path,
            *args,
            max_stdout_bytes=self.max_candidate_diff_bytes,
            stdout_limit_message=(f"Candidate diff exceeds {self.max_candidate_diff_bytes} bytes"),
        )

    def write_candidate_patch(
        self,
        output_path: Path,
        *,
        paths: Iterable[str] | None = None,
    ) -> Path | None:
        """Atomically persist the current candidate diff, if non-empty."""
        patch = self.candidate_diff(paths)
        if not patch:
            return None
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(patch)
                handle.flush()
                os.fsync(handle.fileno())
            Path(tmp_name).replace(output)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        return output

    def cleanup(self) -> None:
        """Remove only worktrees registered by this session."""
        if self.session_root is None:
            return
        self._write_manifest("cleaning")
        errors = self._cleanup_owned_paths()
        self.cleanup_errors = tuple(errors)
        self._active = False
        if errors:
            self._write_manifest("cleanup_failed")
            raise WorktreeCleanupError("; ".join(errors))

    def __enter__(self) -> WorktreeSession:
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> bool:
        try:
            self.cleanup()
        except WorktreeCleanupError:
            if exc_type is None:
                raise
        return False

    def _validate_repository(self) -> None:
        if not self.repo_path.is_dir():
            raise WorktreeError(f"Repository does not exist: {self.repo_path}")
        if self.temp_parent is not None:
            try:
                self.temp_parent.relative_to(self.repo_path)
            except ValueError:
                pass
            else:
                raise WorktreeError("temp_parent must be outside the source checkout")
        top = self._git("rev-parse", "--show-toplevel").decode().strip()
        if Path(top).resolve() != self.repo_path:
            raise WorktreeError(
                f"repo_path must be the Git worktree root ({top}), got {self.repo_path}"
            )

    def _resolve_revision(self, revision: str) -> str:
        if not revision or revision.startswith("-"):
            raise WorktreeError(f"Invalid revision: {revision!r}")
        return self._git("rev-parse", "--verify", f"{revision}^{{commit}}").decode().strip()

    def _capture_dirty_overlay(self) -> None:
        patch = self._git(
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--binary",
            "--full-index",
            "HEAD",
            max_stdout_bytes=self.max_overlay_bytes,
            stdout_limit_message=f"Dirty overlay exceeds {self.max_overlay_bytes} bytes",
        )
        raw_untracked = self._git("ls-files", "--others", "--exclude-standard", "-z")
        untracked_paths = [
            item.decode("utf-8", errors="surrogateescape")
            for item in raw_untracked.split(b"\0")
            if item
        ]

        total_bytes = len(patch)
        if total_bytes > self.max_overlay_bytes:
            raise WorktreeError(f"Dirty overlay exceeds {self.max_overlay_bytes} bytes")
        copied: dict[str, tuple[bytes, int]] = {}
        digest = hashlib.sha256()
        digest.update(patch)
        for relative in untracked_paths:
            safe_relative = _validate_git_relative(relative)
            source = self.repo_path.joinpath(*PurePosixPath(safe_relative).parts)
            if source.is_symlink() or not source.is_file():
                raise WorktreeError(
                    f"Dirty overlay only supports regular untracked files: {relative!r}"
                )
            try:
                source.resolve().relative_to(self.repo_path)
            except ValueError as exc:
                raise WorktreeError(f"Untracked path escapes repository: {relative!r}") from exc
            remaining = self.max_overlay_bytes - total_bytes
            data, mode = self._read_untracked_file(
                source,
                relative,
                max_bytes=remaining,
                changed_message="Source checkout changed while capturing dirty overlay",
            )
            total_bytes += len(data)
            copied[safe_relative] = (data, mode)
            digest.update(safe_relative.encode("utf-8", errors="surrogateescape"))
            digest.update(b"\0")
            digest.update(mode.to_bytes(2, byteorder="big"))
            digest.update(data)

        # Editors are not governed by the Evalyn run lock. Fail instead of
        # silently creating a torn overlay if the source checkout changes while
        # this snapshot is being captured.
        patch_after = self._git(
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--binary",
            "--full-index",
            "HEAD",
            max_stdout_bytes=self.max_overlay_bytes,
            stdout_limit_message=f"Dirty overlay exceeds {self.max_overlay_bytes} bytes",
        )
        untracked_after = self._git("ls-files", "--others", "--exclude-standard", "-z")
        if patch_after != patch or untracked_after != raw_untracked:
            raise WorktreeError("Source checkout changed while capturing dirty overlay")
        for relative, (data, mode) in copied.items():
            source = self.repo_path.joinpath(*PurePosixPath(relative).parts)
            try:
                current_data, current_mode = self._read_untracked_file(
                    source,
                    relative,
                    max_bytes=len(data),
                    changed_message="Source checkout changed while capturing dirty overlay",
                )
                changed = current_data != data or current_mode != mode
            except WorktreeError:
                changed = True
            if changed:
                raise WorktreeError("Source checkout changed while capturing dirty overlay")

        self._overlay_patch = patch
        self._overlay_untracked = copied
        self.overlay = DirtyOverlay(
            tracked_patch_bytes=len(patch),
            untracked_paths=tuple(copied),
            content_sha256=digest.hexdigest(),
        )

    def _read_untracked_file(
        self,
        source: Path,
        relative: str,
        *,
        max_bytes: int,
        changed_message: str,
    ) -> tuple[bytes, int]:
        """Read one stable regular file without ever buffering past ``max_bytes``."""
        try:
            before = source.stat(follow_symlinks=False)
        except OSError as exc:
            raise WorktreeError(changed_message) from exc
        if not stat.S_ISREG(before.st_mode):
            raise WorktreeError(
                f"Dirty overlay only supports regular untracked files: {relative!r}"
            )
        if before.st_size > max_bytes:
            raise WorktreeError(f"Dirty overlay exceeds {self.max_overlay_bytes} bytes")

        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(source, flags)
        except OSError as exc:
            raise WorktreeError(changed_message) from exc

        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
                before.st_dev,
                before.st_ino,
            ):
                raise WorktreeError(changed_message)
            with os.fdopen(fd, "rb") as handle:
                fd = -1
                data = handle.read(max_bytes + 1)
                after = os.fstat(handle.fileno())
        except OSError as exc:
            raise WorktreeError(changed_message) from exc
        finally:
            if fd >= 0:
                os.close(fd)

        if len(data) > max_bytes:
            raise WorktreeError(f"Dirty overlay exceeds {self.max_overlay_bytes} bytes")
        if (
            (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino)
            or after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
            or len(data) != opened.st_size
        ):
            raise WorktreeError(changed_message)
        return data, stat.S_IMODE(opened.st_mode)

    def _apply_overlay(self) -> None:
        pair = self._require_pair()
        if self._overlay_patch:
            self._git_in_bytes(
                pair.candidate_path,
                "apply",
                "--binary",
                "--whitespace=nowarn",
                "-",
                input_bytes=self._overlay_patch,
            )
        for relative, (content, mode) in self._overlay_untracked.items():
            target = pair.candidate_path.joinpath(*PurePosixPath(relative).parts)
            try:
                target.resolve(strict=False).relative_to(pair.candidate_path)
            except ValueError as exc:
                raise WorktreeError(f"Overlay target escapes candidate: {relative!r}") from exc
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            target.chmod(mode)

    def _add_worktree(self, path: Path, revision: str) -> None:
        self._git("worktree", "add", "--detach", str(path), revision)
        self._created_paths.append(path)
        self._write_manifest("preparing")

    def _cleanup_owned_paths(self) -> list[str]:
        errors: list[str] = []
        root = self.session_root
        if root is None:
            return errors

        remaining_paths: list[Path] = []
        for path in reversed(self._created_paths):
            if not self._is_owned_worktree_path(path):
                errors.append(f"Refused to remove unowned path: {path}")
                remaining_paths.append(path)
                continue
            result = self._git_result("worktree", "remove", "--force", str(path))
            if result.returncode != 0 and path.exists():
                errors.append(
                    f"Could not remove {path}: {result.stderr.decode(errors='replace').strip()}"
                )
                remaining_paths.append(path)
        self._created_paths = list(reversed(remaining_paths))

        prune = self._git_result("worktree", "prune")
        if prune.returncode != 0:
            errors.append(
                f"Could not prune worktrees: {prune.stderr.decode(errors='replace').strip()}"
            )

        if not errors and self._hooks_path is not None:
            try:
                self._hooks_path.rmdir()
            except FileNotFoundError:
                pass
            except OSError as exc:
                errors.append(f"Could not remove empty hooks directory: {exc}")

        if not errors:
            unexpected = []
            try:
                unexpected = [
                    child
                    for child in root.iterdir()
                    if self.manifest_path is None or child != self.manifest_path
                ]
            except FileNotFoundError:
                unexpected = []
            if unexpected:
                errors.append(
                    f"Session root contains unowned files: "
                    f"{', '.join(str(path) for path in unexpected)}"
                )

        if not errors:
            if self.manifest_path is not None:
                self.manifest_path.unlink(missing_ok=True)
            try:
                root.rmdir()
            except FileNotFoundError:
                pass
            except OSError as exc:
                errors.append(f"Could not remove session root {root}: {exc}")

        if errors and self.manifest_path is not None and self.pair is not None:
            self._write_manifest("cleanup_failed")

        return errors

    def _is_owned_worktree_path(self, path: Path) -> bool:
        root = self.session_root
        if root is None or self.manifest_path is None or not self.manifest_path.is_file():
            return False
        try:
            path.resolve(strict=False).relative_to(root.resolve())
            with self.manifest_path.open("rb") as handle:
                manifest_bytes = handle.read(_MANIFEST_LIMIT + 1)
            if len(manifest_bytes) > _MANIFEST_LIMIT:
                return False
            raw = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, OSError, json.JSONDecodeError):
            return False
        return (
            isinstance(raw, dict)
            and raw.get("session_id") == self.session_id
            and raw.get("repository") == str(self.repo_path)
        )

    def _write_manifest(self, status: str) -> None:
        if self.manifest_path is None or self.pair is None:
            return
        data = {
            "schema_version": 1,
            "session_id": self.session_id,
            "repository": str(self.repo_path),
            "pid": os.getpid(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "baseline_sha": self.pair.baseline_sha,
            "candidate_sha": self.pair.candidate_sha,
            "baseline_path": str(self.pair.baseline_path),
            "candidate_path": str(self.pair.candidate_path),
            "created_paths": [str(path) for path in self._created_paths],
            "dirty_overlay": {
                "tracked_patch_bytes": self.overlay.tracked_patch_bytes,
                "untracked_paths": list(self.overlay.untracked_paths),
                "content_sha256": self.overlay.content_sha256,
            },
        }
        self.manifest_path.write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _require_pair(self) -> WorktreePair:
        if self.pair is None:
            raise WorktreeError("Worktree session has not been started")
        return self.pair

    def _git(
        self,
        *args: str,
        max_stdout_bytes: int = _GIT_STDOUT_LIMIT,
        stdout_limit_message: str | None = None,
    ) -> bytes:
        result = self._git_result(*args, max_stdout_bytes=max_stdout_bytes)
        if result.stdout_truncated:
            raise WorktreeError(
                stdout_limit_message
                or f"git {' '.join(args)} output exceeds {max_stdout_bytes} bytes"
            )
        if result.stderr_truncated:
            raise WorktreeError(f"git {' '.join(args)} stderr exceeds {_GIT_STDERR_LIMIT} bytes")
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            raise WorktreeError(f"git {' '.join(args)} failed: {stderr}")
        return result.stdout

    def _git_result(
        self,
        *args: str,
        max_stdout_bytes: int = _GIT_STDOUT_LIMIT,
    ) -> _BoundedProcessResult:
        env = self._git_environment()
        argv = ["git", *self._git_safety_config(), "-C", str(self.repo_path), *args]
        try:
            return _run_bounded_process(
                argv,
                env=env,
                timeout=self.git_timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise WorktreeError(f"Unable to run git {' '.join(args)}: {exc}") from exc

    def _git_in(self, path: Path, *args: str) -> bytes:
        return self._git_in_bytes(path, *args)

    def _git_in_bytes(
        self,
        path: Path,
        *args: str,
        input_bytes: bytes | None = None,
        max_stdout_bytes: int = _GIT_STDOUT_LIMIT,
        stdout_limit_message: str | None = None,
    ) -> bytes:
        pair = self._require_pair()
        if path not in {pair.baseline_path, pair.candidate_path}:
            raise WorktreeError(f"Refused Git operation outside owned worktrees: {path}")
        env = self._git_environment()
        argv = ["git", *self._git_safety_config(), "-C", str(path), *args]
        try:
            result = _run_bounded_process(
                argv,
                env=env,
                timeout=self.git_timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                input_bytes=input_bytes,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise WorktreeError(f"Unable to run git {' '.join(args)}: {exc}") from exc
        if result.stdout_truncated:
            raise WorktreeError(
                stdout_limit_message
                or f"git {' '.join(args)} output exceeds {max_stdout_bytes} bytes"
            )
        if result.stderr_truncated:
            raise WorktreeError(f"git {' '.join(args)} stderr exceeds {_GIT_STDERR_LIMIT} bytes")
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            raise WorktreeError(f"git {' '.join(args)} failed in {path}: {stderr}")
        return result.stdout

    def _configured_filter_names(self) -> tuple[str, ...]:
        result = self._git_result(
            "config",
            "--includes",
            "--name-only",
            "--get-regexp",
            r"^filter\..*\.(clean|smudge|process|required)$",
        )
        if result.returncode not in {0, 1}:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise WorktreeError(f"Could not inspect configured Git filters: {detail}")
        names: set[str] = set()
        for raw_key in result.stdout.decode("utf-8", errors="strict").splitlines():
            match = re.fullmatch(
                r"filter\.([A-Za-z0-9][A-Za-z0-9._-]{0,100})\."
                r"(?:clean|smudge|process|required)",
                raw_key,
                flags=re.IGNORECASE,
            )
            if match is None:
                raise WorktreeError(f"Unsafe configured Git filter name: {raw_key!r}")
            names.add(match.group(1))
        return tuple(sorted(names, key=str.casefold))

    def _git_safety_config(self) -> list[str]:
        values = ["-c", "core.fsmonitor=false", "-c", "submodule.recurse=false"]
        if self._hooks_path is not None:
            values.extend(("-c", f"core.hooksPath={self._hooks_path}"))
        for name in self._filter_names:
            values.extend(
                (
                    "-c",
                    f"filter.{name}.clean=",
                    "-c",
                    f"filter.{name}.smudge=",
                    "-c",
                    f"filter.{name}.process=",
                    "-c",
                    f"filter.{name}.required=false",
                )
            )
        return values

    def _git_environment(self) -> dict[str, str]:
        home = self.session_root or self.temp_parent or Path(tempfile.gettempdir())
        return {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "HOME": str(home),
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": os.environ.get("PATH", os.defpath),
        }


def _validate_git_relative(path: str) -> str:
    raw = str(path).replace("\\", "/")
    posix = PurePosixPath(raw)
    if not raw or posix.is_absolute() or "\x00" in raw or ".." in posix.parts:
        raise WorktreeError(f"Unsafe Git-relative path: {path!r}")
    parts = tuple(part for part in posix.parts if part not in {"", "."})
    if not parts:
        raise WorktreeError(f"Unsafe Git-relative path: {path!r}")
    return PurePosixPath(*parts).as_posix()
