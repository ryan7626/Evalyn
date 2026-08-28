"""Deterministic pytest adapter."""

from __future__ import annotations

import ast
from pathlib import Path, PurePosixPath

from evalyn.adapters.base import AdapterError, TestAdapter
from evalyn.run_store import RunStore


class PytestAdapter(TestAdapter):
    """Create isolated pytest probe files and fixed argv commands."""

    name = "pytest"
    media_type = "text/x-python"

    def __init__(self, python_executable: str = "python") -> None:
        if not python_executable.strip():
            raise ValueError("python_executable cannot be empty")
        self.python_executable = python_executable

    def detect(self, repo_path: Path) -> bool:
        if (repo_path / "pytest.ini").is_file() or (repo_path / "conftest.py").is_file():
            return True

        pyproject = repo_path / "pyproject.toml"
        if pyproject.is_file():
            try:
                text = RunStore.read_bytes(pyproject, max_bytes=1024 * 1024).decode(
                    "utf-8", errors="replace"
                )
            except (OSError, ValueError):
                text = ""
            text = text.lower()
            if "[tool.pytest" in text or "pytest" in text:
                return True

        return any(repo_path.glob("tests/test_*.py")) or any(repo_path.glob("test_*.py"))

    def probe_path(self, probe_id: str) -> PurePosixPath:
        return PurePosixPath(".evalyn", "probes", "pytest", f"test_evalyn_{probe_id}.py")

    def validate_source(self, source: str) -> None:
        super().validate_source(source)
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise AdapterError(f"pytest probe is not valid Python: {exc.msg}") from exc
        banned_modules = {"subprocess", "socket", "requests", "urllib", "http", "ftplib"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(
                alias.name.split(".")[0] in banned_modules for alias in node.names
            ):
                raise AdapterError("pytest probe imports a prohibited process/network module")
            if (
                isinstance(node, ast.ImportFrom)
                and (node.module or "").split(".")[0] in banned_modules
            ):
                raise AdapterError("pytest probe imports a prohibited process/network module")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"system", "popen", "spawn", "fork", "execv", "execve"}
            ):
                raise AdapterError("pytest probe calls a prohibited process API")
        has_assertion = any(isinstance(node, ast.Assert) for node in ast.walk(tree))
        has_pytest_context = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "pytest"
            and node.func.attr in {"raises", "warns"}
            for node in ast.walk(tree)
        )
        if not has_assertion and not has_pytest_context:
            raise AdapterError("pytest probe must contain a meaningful assertion")

    def probe_command(self, relative_path: PurePosixPath) -> tuple[str, ...]:
        return (
            self.python_executable,
            "-m",
            "pytest",
            "-q",
            "--disable-warnings",
            "--maxfail=1",
            "-p",
            "no:cacheprovider",
            relative_path.as_posix(),
        )
