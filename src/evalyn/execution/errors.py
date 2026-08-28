"""Exceptions raised by Evalyn's isolated execution foundation."""


class ExecutionSafetyError(RuntimeError):
    """Base class for failures that prevent safe execution."""


class RunLockError(ExecutionSafetyError):
    """Raised when another process owns the repository run lock."""


class WorktreeError(ExecutionSafetyError):
    """Raised when an isolated Git worktree cannot be prepared."""


class WorktreeCleanupError(WorktreeError):
    """Raised when one or more owned worktrees could not be removed."""


class PathPolicyError(ExecutionSafetyError, ValueError):
    """Raised when an execution path violates workspace policy."""


class RuntimeUnavailableError(ExecutionSafetyError):
    """Raised when a requested execution runtime is unavailable."""
