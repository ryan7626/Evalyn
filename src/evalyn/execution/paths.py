"""Fail-closed path validation for disposable execution workspaces."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path, PurePosixPath, PureWindowsPath

from evalyn.execution.errors import PathPolicyError

DEFAULT_PROTECTED_COMPONENTS = frozenset(
    {
        ".git",
        ".agent",
        ".evalyn-generated",
        ".github",
        ".venv",
        "venv",
        "node_modules",
        "vendor",
        "generated",
        "dist",
        "build",
        "tests",
        "test",
        "__tests__",
    }
)

_SECRET_NAMES = frozenset(
    {
        ".netrc",
        ".npmrc",
        ".pypirc",
        "credentials",
        "credentials.json",
        "id_rsa",
        "id_ed25519",
    }
)

_LOCKFILE_NAMES = frozenset(
    {
        "package-lock.json",
        "pnpm-lock.yaml",
        "poetry.lock",
        "pipfile.lock",
        "yarn.lock",
    }
)


class WorkspacePathPolicy:
    """Resolve workspace paths without trusting model- or plan-provided text.

    ``tracked_paths`` should come from ``git ls-files`` for the pinned candidate
    revision. Context and fix allowlists are intentionally separate: a file may
    be safe to show to a model without being safe for autonomous modification.
    """

    def __init__(
        self,
        workspace_root: Path,
        *,
        tracked_paths: Iterable[str | Path] | None = None,
        context_allowlist: Iterable[str | Path] | None = None,
        fix_allowlist: Iterable[str | Path] | None = None,
        protected_components: Iterable[str] = DEFAULT_PROTECTED_COMPONENTS,
    ) -> None:
        root = Path(workspace_root).resolve()
        if not root.is_dir():
            raise PathPolicyError(f"Workspace root is not a directory: {root}")
        self.workspace_root = root
        self.tracked_paths = _normalize_set(tracked_paths)
        self.context_allowlist = _normalize_set(context_allowlist)
        self.fix_allowlist = _normalize_set(fix_allowlist)
        self.protected_components = frozenset(
            component.casefold() for component in protected_components
        )

    def resolve_read(
        self,
        path: str | Path,
        *,
        require_tracked: bool = False,
        allowed_paths: Iterable[str | Path] | None = None,
    ) -> Path:
        """Resolve an existing regular file for reading."""
        relative, resolved = self._resolve(path, must_exist=True)
        if not resolved.is_file():
            raise PathPolicyError(f"Read target is not a regular file: {relative}")
        self._reject_symlinks(relative)
        if require_tracked:
            self._require_membership(relative, self.tracked_paths, "tracked-file")
        allowed = _normalize_set(allowed_paths)
        if allowed is not None:
            self._require_membership(relative, allowed, "read allowlist")
        return resolved

    def resolve_context(self, path: str | Path) -> Path:
        """Resolve a tracked, explicitly permitted LLM-context file."""
        relative, _ = self._resolve(path, must_exist=True)
        self._reject_protected(relative)
        self._require_membership(relative, self.tracked_paths, "tracked-file")
        if self.context_allowlist is not None:
            self._require_membership(
                relative,
                self.context_allowlist,
                "context allowlist",
            )
        return self.resolve_read(relative, require_tracked=True)

    def resolve_write(
        self,
        path: str | Path,
        *,
        protect_sensitive: bool = True,
        must_exist: bool = False,
    ) -> Path:
        """Resolve a contained write target without following symlinks."""
        relative, resolved = self._resolve(path, must_exist=must_exist)
        self._reject_symlinks(relative)
        if protect_sensitive:
            self._reject_protected(relative)
        if must_exist and not resolved.is_file():
            raise PathPolicyError(f"Write target is not a regular file: {relative}")
        return resolved

    def resolve_fix_target(self, path: str | Path) -> Path:
        """Resolve a tracked source file approved for patch-only repair."""
        relative, _ = self._resolve(path, must_exist=True)
        self._reject_protected(relative)
        self._require_membership(relative, self.tracked_paths, "tracked-file")
        self._require_membership(relative, self.fix_allowlist, "fix allowlist")
        return self.resolve_write(relative, must_exist=True)

    def generated_test_path(self, stem: str, *, suffix: str = ".py") -> Path:
        """Return an Evalyn-owned test path, ignoring all directory suggestions.

        Callers provide a semantic stem such as the category name, not a path
        supplied by the model. The directory remains disposable with the
        worktree and cannot overlap repair targets.
        """
        if not re.fullmatch(r"\.[A-Za-z0-9]{1,12}", suffix):
            raise PathPolicyError(f"Invalid generated-test suffix: {suffix!r}")
        safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", str(stem)).strip("_-")[:80]
        if not safe_stem:
            safe_stem = "generated_test"
        generated_root = self.workspace_root / ".evalyn-generated"
        return generated_root / f"{safe_stem}{suffix.lower()}"

    def relative(self, path: str | Path) -> str:
        """Return a canonical POSIX relative path after containment checks."""
        relative, _ = self._resolve(path, must_exist=False)
        return relative

    def _resolve(self, path: str | Path, *, must_exist: bool) -> tuple[str, Path]:
        relative = _normalize_relative(path)
        candidate = self.workspace_root.joinpath(*PurePosixPath(relative).parts)
        try:
            resolved = candidate.resolve(strict=must_exist)
        except (OSError, RuntimeError) as exc:
            raise PathPolicyError(f"Cannot resolve workspace path {relative!r}: {exc}") from exc
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError as exc:
            raise PathPolicyError(f"Path escapes workspace: {path!r}") from exc
        return relative, resolved

    def _reject_symlinks(self, relative: str) -> None:
        cursor = self.workspace_root
        for component in PurePosixPath(relative).parts:
            cursor = cursor / component
            if cursor.is_symlink():
                raise PathPolicyError(f"Symlink paths are not allowed: {relative}")

    def _reject_protected(self, relative: str) -> None:
        posix = PurePosixPath(relative)
        folded_parts = tuple(part.casefold() for part in posix.parts)
        protected = next(
            (part for part in folded_parts if part in self.protected_components),
            None,
        )
        if protected is not None:
            raise PathPolicyError(f"Protected path component {protected!r} in {relative!r}")

        name = posix.name.casefold()
        if (
            name.startswith(".env")
            or name in _SECRET_NAMES
            or name in _LOCKFILE_NAMES
            or name.endswith(".lock")
            or name.startswith("test_")
            or name.endswith("_test.py")
            or ".test." in name
            or ".spec." in name
        ):
            raise PathPolicyError(f"Protected file cannot be used here: {relative!r}")

    @staticmethod
    def _require_membership(
        relative: str,
        candidates: frozenset[str] | None,
        policy_name: str,
    ) -> None:
        if candidates is None:
            raise PathPolicyError(f"No {policy_name} was configured")
        if relative not in candidates:
            raise PathPolicyError(f"Path is not in the {policy_name}: {relative!r}")


def _normalize_set(
    paths: Iterable[str | Path] | None,
) -> frozenset[str] | None:
    if paths is None:
        return None
    return frozenset(_normalize_relative(path) for path in paths)


def _normalize_relative(path: str | Path) -> str:
    raw = str(path)
    if not raw or "\x00" in raw:
        raise PathPolicyError("Path must be a non-empty string without NUL bytes")

    # Treat both slash styles as separators even on POSIX so Windows-shaped
    # traversal cannot become an innocent-looking filename.
    normalized_slashes = raw.replace("\\", "/")
    windows_path = PureWindowsPath(raw)
    posix_path = PurePosixPath(normalized_slashes)
    if windows_path.is_absolute() or windows_path.drive or posix_path.is_absolute():
        raise PathPolicyError(f"Absolute paths are not allowed: {raw!r}")
    if any(part == ".." for part in posix_path.parts):
        raise PathPolicyError(f"Parent traversal is not allowed: {raw!r}")

    parts = tuple(part for part in posix_path.parts if part not in {"", "."})
    if not parts:
        raise PathPolicyError(f"Path does not name a file: {raw!r}")
    return PurePosixPath(*parts).as_posix()
