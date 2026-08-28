"""Deterministic Vitest adapter."""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath

from evalyn.adapters.base import AdapterError, TestAdapter
from evalyn.run_store import RunStore


class VitestAdapter(TestAdapter):
    """Create isolated Vitest probe files and fixed argv commands."""

    name = "vitest"
    media_type = "text/typescript"

    _CONFIG_NAMES = (
        "vitest.config.js",
        "vitest.config.mjs",
        "vitest.config.cjs",
        "vitest.config.ts",
        "vitest.workspace.js",
        "vitest.workspace.ts",
    )

    def detect(self, repo_path: Path) -> bool:
        if any((repo_path / name).is_file() for name in self._CONFIG_NAMES):
            return True

        package_json = repo_path / "package.json"
        if not package_json.is_file():
            return False
        try:
            package = json.loads(
                RunStore.read_bytes(package_json, max_bytes=1024 * 1024).decode("utf-8")
            )
        except (json.JSONDecodeError, OSError, UnicodeDecodeError, ValueError):
            return False

        dependencies: dict[str, object] = {}
        for key in ("dependencies", "devDependencies", "peerDependencies"):
            value = package.get(key, {})
            if isinstance(value, dict):
                dependencies.update(value)
        if "vitest" in dependencies:
            return True

        scripts = package.get("scripts", {})
        return isinstance(scripts, dict) and any(
            "vitest" in str(command).lower() for command in scripts.values()
        )

    def probe_path(self, probe_id: str) -> PurePosixPath:
        return PurePosixPath(".evalyn", "probes", "vitest", f"evalyn.{probe_id}.test.ts")

    def validate_source(self, source: str) -> None:
        super().validate_source(source)
        if re.search(
            r"(?:from\s+|require\s*\()['\"](?:node:)?(?:child_process|net|http|https|dgram)",
            source,
        ):
            raise AdapterError("Vitest probe imports a prohibited process/network module")
        if not re.search(r"\bexpect\s*\([^\n]+\)\s*\.[A-Za-z]+\s*\(", source):
            raise AdapterError("Vitest probe must contain an expect assertion")

    def probe_command(self, relative_path: PurePosixPath) -> tuple[str, ...]:
        return (
            "npx",
            "--no-install",
            "vitest",
            "run",
            relative_path.as_posix(),
            "--reporter=verbose",
            "--no-cache",
        )
