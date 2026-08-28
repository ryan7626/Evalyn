"""Safe execution primitives for Evalyn's differential probe runner.

Integrators should compose these APIs in this order:
``RepoRunLock`` -> ``WorktreeSession`` -> path/runtime policy ->
``CommandRunner``.
"""

from evalyn.execution.container import ContainerRunner, resolve_container_engine
from evalyn.execution.errors import (
    ExecutionSafetyError,
    PathPolicyError,
    RunLockError,
    RuntimeUnavailableError,
    WorktreeCleanupError,
    WorktreeError,
)
from evalyn.execution.lock import RepoRunLock, read_lock_metadata
from evalyn.execution.paths import WorkspacePathPolicy
from evalyn.execution.process import (
    CommandResult,
    CommandRunner,
    validate_environment_overrides,
)
from evalyn.execution.runtime import (
    ExecutionRuntime,
    HostRuntime,
    OciContainerRuntime,
    RuntimeInvocation,
)
from evalyn.execution.worktree import DirtyOverlay, WorktreePair, WorktreeSession

__all__ = [
    "CommandResult",
    "CommandRunner",
    "ContainerRunner",
    "DirtyOverlay",
    "ExecutionRuntime",
    "ExecutionSafetyError",
    "HostRuntime",
    "OciContainerRuntime",
    "PathPolicyError",
    "RepoRunLock",
    "RunLockError",
    "RuntimeInvocation",
    "RuntimeUnavailableError",
    "WorkspacePathPolicy",
    "WorktreeCleanupError",
    "WorktreeError",
    "WorktreePair",
    "WorktreeSession",
    "read_lock_metadata",
    "resolve_container_engine",
    "validate_environment_overrides",
]
