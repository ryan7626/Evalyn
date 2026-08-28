"""Base contract for framework-specific generated-probe adapters.

Adapters are the trust boundary between model-authored probe source and command
execution.  The model never selects a path or command: both are derived here
from a content-addressed probe identifier.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath

_SAFE_PROBE_ID = re.compile(r"^[a-f0-9]{12,64}$")


class AdapterError(ValueError):
    """Raised when an adapter cannot safely construct a probe artifact."""


@dataclass(frozen=True)
class ProbeArtifact:
    """Immutable, adapter-owned representation of a generated test probe."""

    probe_id: str
    adapter: str
    source: str
    relative_path: PurePosixPath
    command: tuple[str, ...]
    source_sha256: str
    media_type: str

    def __post_init__(self) -> None:
        if not _SAFE_PROBE_ID.fullmatch(self.probe_id):
            raise AdapterError("probe_id must be a 12-64 character lowercase hex digest")
        if not self.adapter.strip():
            raise AdapterError("adapter name cannot be empty")
        if not self.source.strip():
            raise AdapterError("probe source cannot be empty")
        if self.relative_path.is_absolute() or ".." in self.relative_path.parts:
            raise AdapterError("probe path must be relative and cannot traverse parents")
        if not self.relative_path.parts:
            raise AdapterError("probe path cannot be empty")
        if not self.command or any(not part for part in self.command):
            raise AdapterError("probe command must contain non-empty argv entries")
        actual_digest = sha256(self.source.encode("utf-8")).hexdigest()
        if self.source_sha256 != actual_digest:
            raise AdapterError("source_sha256 does not match probe source")


class TestAdapter(ABC):
    """Deterministically turns model-authored source into a runnable artifact."""

    name: str
    media_type: str

    @abstractmethod
    def detect(self, repo_path: Path) -> bool:
        """Return whether this framework appears to be configured in a repo."""

    @abstractmethod
    def probe_path(self, probe_id: str) -> PurePosixPath:
        """Return an adapter-owned safe relative path for a probe."""

    @abstractmethod
    def probe_command(self, relative_path: PurePosixPath) -> tuple[str, ...]:
        """Return an argv tuple that executes only the supplied probe path."""

    def validate_source(self, source: str) -> None:
        """Reject invalid or assertion-free generated source."""
        if not source.strip():
            raise AdapterError("probe source cannot be empty")

    def build_artifact(self, source: str) -> ProbeArtifact:
        """Build a content-addressed artifact without accepting path or command."""
        normalized_source = source.rstrip() + "\n"
        self.validate_source(normalized_source)

        source_digest = sha256(normalized_source.encode("utf-8")).hexdigest()
        probe_id = source_digest[:16]
        relative_path = self.probe_path(probe_id)
        command = self.probe_command(relative_path)
        return ProbeArtifact(
            probe_id=probe_id,
            adapter=self.name,
            source=normalized_source,
            relative_path=relative_path,
            command=command,
            source_sha256=source_digest,
            media_type=self.media_type,
        )
