"""Fail-closed OCI image identity and execution-environment fingerprints."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from evalyn.execution.process import CommandResult

_SHA256_IMAGE_ID = re.compile(r"(?:sha256:)?([0-9a-f]{64})")


class ImageIdentityStatus(str, Enum):
    """Whether an immutable identity was obtained from the OCI engine."""

    RESOLVED = "resolved"
    INCOMPLETE = "incomplete"


class ImageIdentityError(str, Enum):
    """Stable, non-sensitive reasons an image identity is incomplete."""

    NOT_INSPECTED = "not_inspected"
    NO_SUCCESSFUL_LAUNCH = "no_successful_launch"
    INSPECTION_FAILED = "inspection_failed"
    INSPECTION_TIMED_OUT = "inspection_timed_out"
    INSPECTION_TRUNCATED = "inspection_truncated"
    MALFORMED_INSPECTION = "malformed_inspection"


@dataclass(frozen=True, slots=True)
class ImageIdentity:
    """An immutable, content-addressed identity reported by the OCI engine."""

    value: str
    kind: str = "oci_image_id"

    def __post_init__(self) -> None:
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.value):
            raise ValueError("OCI image identity must be a canonical sha256 digest")
        if self.kind != "oci_image_id":
            raise ValueError("Unsupported OCI image identity kind")

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "value": self.value}


@dataclass(frozen=True, slots=True)
class ImageIdentityResolution:
    """The result of a bounded, local-endpoint image inspection."""

    status: ImageIdentityStatus
    identity: ImageIdentity | None = None
    error: ImageIdentityError | None = None

    def __post_init__(self) -> None:
        resolved = self.status == ImageIdentityStatus.RESOLVED
        if resolved != (self.identity is not None) or resolved == (self.error is not None):
            raise ValueError("Resolved image identity state is inconsistent")

    @classmethod
    def resolved(cls, identity: ImageIdentity) -> ImageIdentityResolution:
        return cls(ImageIdentityStatus.RESOLVED, identity=identity)

    @classmethod
    def incomplete(cls, error: ImageIdentityError) -> ImageIdentityResolution:
        return cls(ImageIdentityStatus.INCOMPLETE, error=error)

    @property
    def complete(self) -> bool:
        return self.status == ImageIdentityStatus.RESOLVED

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "identity": self.identity.to_dict() if self.identity is not None else None,
            "error": self.error.value if self.error is not None else None,
        }


@dataclass(frozen=True, slots=True)
class SecurityPolicy:
    """Security-relevant OCI launch settings, without host-specific values."""

    capabilities_dropped: tuple[str, ...]
    no_new_privileges: bool
    pids_limit: int
    memory_limit: str
    cpu_limit: str
    run_as_current_user: bool
    tmpfs_size: str

    def to_dict(self) -> dict[str, object]:
        return {
            "capabilities_dropped": list(self.capabilities_dropped),
            "no_new_privileges": self.no_new_privileges,
            "pids_limit": self.pids_limit,
            "memory_limit": self.memory_limit,
            "cpu_limit": self.cpu_limit,
            "run_as_current_user": self.run_as_current_user,
            "tmpfs_size": self.tmpfs_size,
        }


@dataclass(frozen=True, slots=True)
class EnvironmentFingerprint:
    """Serializable evidence about the exact image and its launch policy.

    ``fingerprint_sha256`` is intentionally absent while image inspection is
    incomplete. This prevents callers from treating a hash of a mutable image
    tag as verified environment identity.
    """

    engine: str
    declared_image: str
    image_resolution: ImageIdentityResolution
    endpoint_scope: str
    network_policy: str
    root_filesystem_read_only: bool
    workspace_read_only: bool
    pull_policy: str
    security_policy: SecurityPolicy
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.endpoint_scope not in {"local", "remote"}:
            raise ValueError("endpoint_scope must be local or remote")

    @property
    def complete(self) -> bool:
        return self.image_resolution.complete

    @property
    def resolved_image_identity(self) -> str | None:
        identity = self.image_resolution.identity
        return identity.value if identity is not None else None

    @property
    def fingerprint_sha256(self) -> str | None:
        if not self.complete:
            return None
        canonical = json.dumps(
            self._identity_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        return f"sha256:{hashlib.sha256(canonical).hexdigest()}"

    def to_dict(self) -> dict[str, object]:
        payload = self._identity_payload()
        payload.update(
            {
                "complete": self.complete,
                "fingerprint_sha256": self.fingerprint_sha256,
            }
        )
        return payload

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "engine": self.engine,
            "endpoint_scope": self.endpoint_scope,
            "declared_image": self.declared_image,
            "resolved_image_identity": self.resolved_image_identity,
            "image_identity": self.image_resolution.to_dict(),
            "network_policy": self.network_policy,
            "read_only_policy": {
                "root_filesystem": self.root_filesystem_read_only,
                "workspace": self.workspace_read_only,
            },
            "pull_policy": self.pull_policy,
            "security_policy": self.security_policy.to_dict(),
        }


def parse_image_identity_inspection(result: CommandResult) -> ImageIdentityResolution:
    """Validate one bounded ``image inspect --format {{.Id}}`` result.

    Only a complete, successful response containing exactly one full SHA-256
    image ID is accepted. Diagnostic output is deliberately not retained in
    the returned value because it can contain daemon endpoints or host paths.
    """

    if result.timed_out:
        return ImageIdentityResolution.incomplete(ImageIdentityError.INSPECTION_TIMED_OUT)
    if result.output_truncated:
        return ImageIdentityResolution.incomplete(ImageIdentityError.INSPECTION_TRUNCATED)
    if not result.succeeded:
        return ImageIdentityResolution.incomplete(ImageIdentityError.INSPECTION_FAILED)

    value = _strip_one_line_ending(result.stdout)
    matched = _SHA256_IMAGE_ID.fullmatch(value)
    if matched is None:
        return ImageIdentityResolution.incomplete(ImageIdentityError.MALFORMED_INSPECTION)
    return ImageIdentityResolution.resolved(ImageIdentity(f"sha256:{matched.group(1)}"))


def normalized_engine_name(engine: str) -> str:
    """Return a useful engine label without serializing its host path."""

    name = Path(engine).name.casefold()
    if name.endswith(".exe"):
        name = name[:-4]
    return name if name in {"docker", "podman"} else "custom-oci"


def _strip_one_line_ending(value: str) -> str:
    if value.endswith("\n"):
        value = value[:-1]
        if value.endswith("\r"):
            value = value[:-1]
    return value
