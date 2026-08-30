"""Build deterministic, provenance-rich context for a repository change."""

from __future__ import annotations

import ast
import os
import re
import shlex
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath

from evalyn.analysis.models import ChangeGraph, ChangeTarget, ContextEvidence, OracleReference

_HUNK = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)
_TS_DECLARATION = re.compile(
    r"^\s*(?:export\s+(?:default\s+)?)?(?:declare\s+)?(?:async\s+)?"
    r"(?:(?P<kind>function|class|interface|type|enum|const|let|var)\s+)"
    r"(?P<name>[A-Za-z_$][\w$]*)"
)
_TEST_COMPONENTS = frozenset({"test", "tests", "__tests__", "spec", "specs"})
_SCHEMA_COMPONENTS = frozenset({"schema", "schemas", "openapi", "swagger"})
_DOCUMENTATION_SUFFIXES = frozenset({".md", ".mdx", ".rst", ".txt"})
_TEST_SUFFIXES = frozenset({".cjs", ".js", ".jsx", ".mjs", ".py", ".ts", ".tsx"})
_SCHEMA_SUFFIXES = frozenset(
    {".avsc", ".gql", ".graphql", ".json", ".proto", ".yaml", ".yml", ".xsd"}
)
_IMPLEMENTATION_SUFFIXES = frozenset(
    {".cjs", ".js", ".jsx", ".mjs", ".mts", ".py", ".pyx", ".ts", ".tsx"}
)
_MANIFEST_NAMES = frozenset(
    {
        "package.json",
        "package-lock.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "bun.lockb",
        "pyproject.toml",
        "setup.cfg",
        "setup.py",
        "requirements.txt",
        "requirements-dev.txt",
        "poetry.lock",
        "pdm.lock",
        "uv.lock",
        "pytest.ini",
        "tox.ini",
        "tsconfig.json",
        "vitest.config.js",
        "vitest.config.ts",
        "vitest.workspace.js",
        "vitest.workspace.ts",
    }
)
_IGNORED_DIRECTORIES = frozenset(
    {
        ".git",
        ".evalyn",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "coverage",
        "dist",
        "node_modules",
        "target",
        "vendor",
    }
)
_ALLOWED_CONTEXT_SUFFIXES = (
    _DOCUMENTATION_SUFFIXES
    | _TEST_SUFFIXES
    | _SCHEMA_SUFFIXES
    | _IMPLEMENTATION_SUFFIXES
    | frozenset({".cfg", ".ini", ".lock", ".pyi", ".toml"})
)
_ORACLE_CATEGORIES = frozenset({"existing_test", "documentation", "api_schema", "type_contract"})
_MAX_DEFINITION_CHARS = 8_000


@dataclass(frozen=True)
class _HunkRange:
    old_start: int
    old_count: int
    new_start: int
    new_count: int


@dataclass(frozen=True)
class _DiffChange:
    old_path: str
    new_path: str
    hunks: tuple[_HunkRange, ...]
    renamed: bool = False


@dataclass(frozen=True)
class _Symbol:
    name: str
    kind: str
    start: int
    end: int
    definition: str


@dataclass(frozen=True)
class _EvidenceCandidate:
    path: str
    side: str
    category: str
    content: str
    score: int
    target_ids: tuple[str, ...]
    reasons: tuple[str, ...]


def build_change_graph(
    base_root: Path,
    candidate_root: Path,
    *,
    changed_paths: Iterable[str] = (),
    diff: str = "",
    max_context_files: int = 64,
    max_context_chars: int = 80_000,
    max_file_chars: int = 20_000,
    repository_paths: Iterable[str] | None = None,
) -> ChangeGraph:
    """Return a deterministic graph without consulting model-authored metadata."""
    return ChangeGraphBuilder(
        base_root,
        candidate_root,
        max_context_files=max_context_files,
        max_context_chars=max_context_chars,
        max_file_chars=max_file_chars,
        repository_paths=repository_paths,
    ).build(changed_paths=changed_paths, diff=diff)


class ChangeGraphBuilder:
    """Discover changed symbols and rank repository context deterministically."""

    def __init__(
        self,
        base_root: Path,
        candidate_root: Path,
        *,
        max_context_files: int = 64,
        max_context_chars: int = 80_000,
        max_file_chars: int = 20_000,
        max_inventory_files: int = 5_000,
        repository_paths: Iterable[str] | None = None,
    ) -> None:
        self.base_root = Path(base_root).resolve()
        self.candidate_root = Path(candidate_root).resolve()
        if not self.base_root.is_dir() or not self.candidate_root.is_dir():
            raise ValueError("base_root and candidate_root must be existing directories")
        if max_context_files < 1 or max_context_chars < 1 or max_file_chars < 1:
            raise ValueError("context limits must be positive")
        if max_inventory_files < 1:
            raise ValueError("max_inventory_files must be positive")
        self.max_context_files = max_context_files
        self.max_context_chars = max_context_chars
        self.max_file_chars = max_file_chars
        self.max_inventory_files = max_inventory_files
        self.repository_paths = (
            tuple(sorted({_safe_relative_path(str(path)) for path in repository_paths}))
            if repository_paths is not None
            else None
        )

    def build(self, *, changed_paths: Iterable[str] = (), diff: str = "") -> ChangeGraph:
        parsed = _parse_diff(diff)
        changes = _merge_changes(parsed, changed_paths, self.base_root, self.candidate_root)
        if not changes:
            raise ValueError("change graph requires a changed path or parseable diff")
        targets = self._build_targets(changes)
        context = self._build_context(changes, targets)
        oracles = _build_oracles(context, targets)
        paths = tuple(sorted({target.path for target in targets}))
        return ChangeGraph(
            targets=targets,
            context=context,
            oracles=oracles,
            changed_paths=paths,
            diff_sha256=sha256(diff.encode("utf-8")).hexdigest(),
        )

    def _build_targets(self, changes: tuple[_DiffChange, ...]) -> tuple[ChangeTarget, ...]:
        targets: list[ChangeTarget] = []
        for change in changes:
            base_text = _read_repository_text(self.base_root, change.old_path, max_chars=250_000)
            candidate_text = _read_repository_text(
                self.candidate_root, change.new_path, max_chars=250_000
            )
            change_kind = _change_kind(change, base_text is not None, candidate_text is not None)
            path = change.new_path or change.old_path
            old_ranges = _side_ranges(change.hunks, "base")
            new_ranges = _side_ranges(change.hunks, "candidate")
            file_id = _identifier(
                "target", path, change_kind, "file", change.old_path, change.new_path
            )
            targets.append(
                ChangeTarget(
                    target_id=file_id,
                    path=path,
                    change_kind=change_kind,
                    base_path=change.old_path,
                    candidate_path=change.new_path,
                    base_start_line=_range_start(old_ranges),
                    base_end_line=_range_end(old_ranges),
                    candidate_start_line=_range_start(new_ranges),
                    candidate_end_line=_range_end(new_ranges),
                    base_definition=_bounded_text(base_text or "", _MAX_DEFINITION_CHARS),
                    candidate_definition=_bounded_text(candidate_text or "", _MAX_DEFINITION_CHARS),
                )
            )
            base_symbols = _extract_symbols(change.old_path, base_text or "")
            candidate_symbols = _extract_symbols(change.new_path, candidate_text or "")
            symbol_keys = {
                (symbol.name, symbol.kind)
                for symbol in base_symbols
                if not old_ranges or _overlaps(symbol, old_ranges)
            }
            symbol_keys.update(
                (symbol.name, symbol.kind)
                for symbol in candidate_symbols
                if not new_ranges or _overlaps(symbol, new_ranges)
            )
            base_by_key = {(symbol.name, symbol.kind): symbol for symbol in base_symbols}
            candidate_by_key = {(symbol.name, symbol.kind): symbol for symbol in candidate_symbols}
            for name, kind in sorted(symbol_keys):
                old_symbol = base_by_key.get((name, kind))
                new_symbol = candidate_by_key.get((name, kind))
                target_id = _identifier(
                    "target",
                    path,
                    change_kind,
                    kind,
                    name,
                    str(old_symbol.start if old_symbol else 0),
                    str(new_symbol.start if new_symbol else 0),
                )
                targets.append(
                    ChangeTarget(
                        target_id=target_id,
                        path=path,
                        change_kind=change_kind,
                        symbol=name,
                        symbol_kind=kind,
                        base_path=change.old_path,
                        candidate_path=change.new_path,
                        base_start_line=old_symbol.start if old_symbol else None,
                        base_end_line=old_symbol.end if old_symbol else None,
                        candidate_start_line=new_symbol.start if new_symbol else None,
                        candidate_end_line=new_symbol.end if new_symbol else None,
                        base_definition=old_symbol.definition if old_symbol else "",
                        candidate_definition=new_symbol.definition if new_symbol else "",
                    )
                )
        return tuple(
            sorted(
                targets,
                key=lambda target: (
                    target.path,
                    target.symbol_kind != "file",
                    target.candidate_start_line or target.base_start_line or 0,
                    target.symbol,
                    target.target_id,
                ),
            )
        )

    def _build_context(
        self,
        changes: tuple[_DiffChange, ...],
        targets: tuple[ChangeTarget, ...],
    ) -> tuple[ContextEvidence, ...]:
        if self.repository_paths is None:
            base_files = _repository_inventory(self.base_root, self.max_inventory_files)
            candidate_files = _repository_inventory(self.candidate_root, self.max_inventory_files)
        else:
            base_files = self.repository_paths[: self.max_inventory_files]
            candidate_files = base_files
        changed_base = {change.old_path for change in changes if change.old_path}
        changed_candidate = {change.new_path for change in changes if change.new_path}
        all_paths = sorted(
            set(base_files) | set(candidate_files) | changed_base | changed_candidate
        )
        candidates: list[_EvidenceCandidate] = []
        for path in all_paths:
            base_content = _read_repository_text(
                self.base_root, path, max_chars=self.max_file_chars
            )
            candidate_content = _read_repository_text(
                self.candidate_root, path, max_chars=self.max_file_chars
            )
            if base_content is None and candidate_content is None:
                continue
            category = classify_context_path(path)
            changed = path in changed_base or path in changed_candidate
            if base_content is not None and base_content == candidate_content:
                item = _score_context(path, "both", category, base_content, changed, targets)
                if item is not None:
                    candidates.append(item)
                continue
            if base_content is not None:
                item = _score_context(
                    path, "base", category, base_content, path in changed_base, targets
                )
                if item is not None:
                    candidates.append(item)
            if candidate_content is not None:
                item = _score_context(
                    path,
                    "candidate",
                    category,
                    candidate_content,
                    path in changed_candidate,
                    targets,
                )
                if item is not None:
                    candidates.append(item)
        candidates.sort(key=lambda item: (-item.score, item.path, item.side, item.category))
        selected: list[ContextEvidence] = []
        remaining = self.max_context_chars
        for candidate in candidates:
            if len(selected) >= self.max_context_files or remaining <= 0:
                break
            content = candidate.content[:remaining]
            if not content:
                continue
            digest = sha256(content.encode("utf-8")).hexdigest()
            evidence_id = _identifier(
                "evidence", candidate.side, candidate.path, candidate.category, digest
            )
            reasons = candidate.reasons
            if len(content) < len(candidate.content):
                reasons = (*reasons, "context_budget_truncated")
            selected.append(
                ContextEvidence(
                    evidence_id=evidence_id,
                    path=candidate.path,
                    side=candidate.side,
                    category=candidate.category,
                    content=content,
                    content_sha256=digest,
                    score=candidate.score,
                    rank=len(selected) + 1,
                    target_ids=candidate.target_ids,
                    reasons=reasons,
                )
            )
            remaining -= len(content)
        return tuple(selected)


def classify_context_path(path: str) -> str:
    """Classify a path conservatively; repository text cannot choose its category."""
    normalized = PurePosixPath(path)
    parts = tuple(part.casefold() for part in normalized.parts)
    name = normalized.name.casefold()
    suffixes = tuple(suffix.casefold() for suffix in normalized.suffixes)
    suffix = suffixes[-1] if suffixes else ""
    if (
        (any(part in _TEST_COMPONENTS for part in parts[:-1]) and suffix in _TEST_SUFFIXES)
        or name.startswith("test_")
        or name.endswith(
            (
                "_test.py",
                "_spec.py",
                ".test.js",
                ".test.jsx",
                ".test.ts",
                ".test.tsx",
                ".spec.js",
                ".spec.jsx",
                ".spec.ts",
                ".spec.tsx",
            )
        )
    ):
        return "existing_test"
    if name.endswith((".d.ts", ".d.mts", ".d.cts", ".pyi")):
        return "type_contract"
    if (
        (any(part in _SCHEMA_COMPONENTS for part in parts[:-1]) and suffix in _SCHEMA_SUFFIXES)
        or ".schema." in name
        or name.startswith(("openapi.", "swagger."))
        or suffix in {".graphql", ".gql", ".proto", ".xsd"}
    ):
        return "api_schema"
    if ("docs" in parts[:-1] and suffix in _DOCUMENTATION_SUFFIXES) or (
        name.startswith("readme") and suffix in _DOCUMENTATION_SUFFIXES
    ):
        return "documentation"
    if name in _MANIFEST_NAMES:
        return "manifest"
    return "implementation"


def _parse_diff(diff: str) -> tuple[_DiffChange, ...]:
    changes: list[_DiffChange] = []
    old_path = ""
    new_path = ""
    hunks: list[_HunkRange] = []
    renamed = False

    def finish() -> None:
        nonlocal old_path, new_path, hunks, renamed
        if old_path or new_path:
            changes.append(_DiffChange(old_path, new_path, tuple(hunks), renamed))
        old_path = ""
        new_path = ""
        hunks = []
        renamed = False

    for line in diff.splitlines():
        if line.startswith("diff --git "):
            finish()
            try:
                fields = shlex.split(line)
            except ValueError:
                fields = line.split()
            if len(fields) >= 4:
                old_path = _diff_path(fields[2], "a/")
                new_path = _diff_path(fields[3], "b/")
            continue
        if line.startswith("rename from "):
            old_path = _safe_path_or_empty(line[len("rename from ") :])
            renamed = True
            continue
        if line.startswith("rename to "):
            new_path = _safe_path_or_empty(line[len("rename to ") :])
            renamed = True
            continue
        if line.startswith("--- "):
            old_path = _diff_path(line[4:].split("\t", 1)[0], "a/")
            continue
        if line.startswith("+++ "):
            new_path = _diff_path(line[4:].split("\t", 1)[0], "b/")
            continue
        match = _HUNK.match(line)
        if match:
            hunks.append(
                _HunkRange(
                    old_start=int(match.group("old_start")),
                    old_count=int(match.group("old_count") or "1"),
                    new_start=int(match.group("new_start")),
                    new_count=int(match.group("new_count") or "1"),
                )
            )
    finish()
    return tuple(changes)


def _merge_changes(
    parsed: tuple[_DiffChange, ...],
    changed_paths: Iterable[str],
    base_root: Path,
    candidate_root: Path,
) -> tuple[_DiffChange, ...]:
    by_path: dict[str, _DiffChange] = {}
    for change in parsed:
        canonical = change.new_path or change.old_path
        if canonical:
            by_path[canonical] = change
    for raw_path in changed_paths:
        path = _safe_relative_path(str(raw_path))
        if path in by_path:
            continue
        in_base = _is_repository_file(base_root, path)
        in_candidate = _is_repository_file(candidate_root, path)
        by_path[path] = _DiffChange(path if in_base else "", path if in_candidate else "", ())
    return tuple(by_path[path] for path in sorted(by_path))


def _change_kind(change: _DiffChange, in_base: bool, in_candidate: bool) -> str:
    renamed_paths = change.old_path and change.new_path and change.old_path != change.new_path
    if change.renamed or renamed_paths:
        return "renamed"
    if in_candidate and not in_base:
        return "added"
    if in_base and not in_candidate:
        return "deleted"
    return "modified"


def _extract_symbols(path: str, text: str) -> tuple[_Symbol, ...]:
    suffix = PurePosixPath(path).suffix.casefold()
    if suffix in {".py", ".pyi", ".pyx"}:
        return _python_symbols(text)
    if suffix in {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts"}:
        return _typescript_symbols(text)
    return ()


def _python_symbols(text: str) -> tuple[_Symbol, ...]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return _python_symbols_fallback(text)
    lines = text.splitlines()
    symbols: list[_Symbol] = []

    def visit(nodes: list[ast.stmt], parents: tuple[str, ...] = ()) -> None:
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = ".".join((*parents, node.name))
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                start = node.lineno
                end = getattr(node, "end_lineno", None) or start
                definition = "\n".join(lines[start - 1 : end])[:_MAX_DEFINITION_CHARS]
                symbols.append(_Symbol(name, kind, start, end, definition))
                visit(node.body, (*parents, node.name))

    visit(tree.body)
    return tuple(symbols)


def _python_symbols_fallback(text: str) -> tuple[_Symbol, ...]:
    lines = text.splitlines()
    symbols: list[_Symbol] = []
    declaration = re.compile(r"^\s*(?:async\s+)?(?P<kind>def|class)\s+(?P<name>[A-Za-z_]\w*)")
    for index, line in enumerate(lines, 1):
        match = declaration.match(line)
        if match:
            symbols.append(
                _Symbol(
                    match.group("name"),
                    "function" if match.group("kind") == "def" else "class",
                    index,
                    index,
                    line[:_MAX_DEFINITION_CHARS],
                )
            )
    return tuple(symbols)


def _typescript_symbols(text: str) -> tuple[_Symbol, ...]:
    lines = text.splitlines()
    symbols: list[_Symbol] = []
    for index, line in enumerate(lines, 1):
        match = _TS_DECLARATION.match(line)
        if not match:
            continue
        kind = match.group("kind")
        end = _typescript_declaration_end(lines, index)
        normalized_kind = {
            "const": "variable",
            "let": "variable",
            "var": "variable",
            "interface": "type",
            "type": "type",
        }.get(kind, kind)
        symbols.append(
            _Symbol(
                match.group("name"),
                normalized_kind,
                index,
                end,
                "\n".join(lines[index - 1 : end])[:_MAX_DEFINITION_CHARS],
            )
        )
    return tuple(symbols)


def _typescript_declaration_end(lines: list[str], start: int) -> int:
    balance = 0
    saw_brace = False
    for index in range(start - 1, min(len(lines), start + 500)):
        line = lines[index]
        balance += line.count("{") - line.count("}")
        saw_brace = saw_brace or "{" in line
        if saw_brace and balance <= 0:
            return index + 1
        if not saw_brace and ";" in line:
            return index + 1
    return start


def _score_context(
    path: str,
    side: str,
    category: str,
    content: str,
    changed: bool,
    targets: tuple[ChangeTarget, ...],
) -> _EvidenceCandidate | None:
    associated: list[tuple[int, ChangeTarget, tuple[str, ...]]] = []
    for target in targets:
        reasons: list[str] = []
        proximity = _path_proximity(path, target.path)
        if path in {target.path, target.base_path, target.candidate_path}:
            reasons.append("changed_path")
            proximity = max(proximity, 250)
        if _references_symbol(content, target.symbol):
            reasons.append("symbol_reference")
        if _references_module(path, content, target.path):
            reasons.append("module_import_or_call")
        if reasons or (category == "existing_test" and proximity > 0):
            associated.append((proximity, target, tuple(reasons)))

    include = changed or category == "manifest" or bool(associated)
    if category == "documentation" and PurePosixPath(path).name.casefold().startswith("readme"):
        include = True
    if not include:
        return None

    category_score = {
        "existing_test": 520,
        "type_contract": 470,
        "api_schema": 450,
        "documentation": 380,
        "manifest": 280,
        "implementation": 180,
    }[category]
    score = category_score + (1_000 if changed else 0)
    reasons: list[str] = []
    if changed:
        reasons.append("changed_path")
    target_ids: set[str] = set()
    if associated:
        best_proximity = max(item[0] for item in associated)
        score += min(best_proximity, 250)
        reasons.append("path_proximity")
        for _, target, match_reasons in associated:
            target_ids.add(target.target_id)
            for reason in match_reasons:
                if reason not in reasons:
                    reasons.append(reason)
                if reason == "module_import_or_call":
                    score += 130
                elif reason == "symbol_reference":
                    score += 90
    if category == "manifest":
        reasons.append("project_manifest")
        target_ids.update(target.target_id for target in targets)
    if side in {"base", "both"}:
        score += 5
        reasons.append("base_visible")
    score -= min(80, len(content) // 2_000)
    return _EvidenceCandidate(
        path=path,
        side=side,
        category=category,
        content=content,
        score=score,
        target_ids=tuple(sorted(target_ids)),
        reasons=tuple(dict.fromkeys(reasons)),
    )


def _build_oracles(
    context: tuple[ContextEvidence, ...], targets: tuple[ChangeTarget, ...]
) -> tuple[OracleReference, ...]:
    by_id = {target.target_id: target for target in targets}
    oracles: list[OracleReference] = []
    for evidence in context:
        if evidence.category not in _ORACLE_CATEGORIES or evidence.side not in {"base", "both"}:
            continue
        related = tuple(by_id[target_id] for target_id in evidence.target_ids if target_id in by_id)
        excerpt = _oracle_excerpt(evidence.category, evidence.content, related)
        if len(re.sub(r"\s+", " ", excerpt).strip()) < 10:
            continue
        excerpt_digest = sha256(excerpt.encode("utf-8")).hexdigest()
        oracle_id = _identifier("oracle", evidence.evidence_id, excerpt_digest)
        oracles.append(
            OracleReference(
                oracle_id=oracle_id,
                category=evidence.category,
                excerpt=excerpt,
                excerpt_sha256=excerpt_digest,
                source_path=evidence.path,
                source_side=evidence.side,
                evidence_id=evidence.evidence_id,
                target_ids=evidence.target_ids,
            )
        )
    return tuple(oracles)


def _oracle_excerpt(category: str, content: str, targets: tuple[ChangeTarget, ...]) -> str:
    lines = content.splitlines()
    tokens = tuple(
        token.casefold()
        for target in targets
        for token in (target.symbol.rsplit(".", 1)[-1], PurePosixPath(target.path).stem)
        if len(token) >= 3
    )
    preferred_markers = {
        "existing_test": ("assert ", "expect(", "expect (", "raises(", "toequal("),
        "type_contract": ("def ", "class ", "interface ", "type ", ":"),
        "api_schema": ("required", "properties", "type", "enum", "schema"),
        "documentation": ("must", "should", "returns", "raises", "invalid", "default"),
    }[category]
    best_index: int | None = None
    for index, line in enumerate(lines):
        normalized = line.strip().casefold()
        if not normalized:
            continue
        has_token = any(token in normalized for token in tokens)
        has_marker = any(marker in normalized for marker in preferred_markers)
        if has_token and has_marker:
            best_index = index
            break
        if best_index is None and has_marker:
            best_index = index
    if best_index is None:
        best_index = next((index for index, line in enumerate(lines) if line.strip()), 0)
    start = max(0, best_index - 1)
    end = min(len(lines), best_index + 2)
    return "\n".join(lines[start:end]).strip()[:500]


def _repository_inventory(root: Path, limit: int) -> tuple[str, ...]:
    paths: list[str] = []
    for current, directories, files in os.walk(root, followlinks=False):
        directories[:] = sorted(
            directory
            for directory in directories
            if directory not in _IGNORED_DIRECTORIES
            and not (Path(current) / directory).is_symlink()
        )
        current_path = Path(current)
        for name in sorted(files):
            path = current_path / name
            if path.is_symlink():
                continue
            relative = path.relative_to(root).as_posix()
            is_context = (
                name.casefold() in _MANIFEST_NAMES
                or path.suffix.casefold() in _ALLOWED_CONTEXT_SUFFIXES
            )
            if is_context:
                paths.append(relative)
                if len(paths) >= limit:
                    return tuple(paths)
    return tuple(paths)


def _read_repository_text(root: Path, relative: str, *, max_chars: int) -> str | None:
    if not relative:
        return None
    safe = _safe_relative_path(relative)
    path = root.joinpath(*PurePosixPath(safe).parts)
    if not path.is_file() or path.is_symlink():
        return None
    try:
        data = path.read_bytes()[: max_chars * 4 + 1]
    except OSError:
        return None
    if b"\x00" in data:
        return None
    return data.decode("utf-8", errors="replace")[:max_chars]


def _is_repository_file(root: Path, relative: str) -> bool:
    path = root.joinpath(*PurePosixPath(relative).parts)
    return path.is_file() and not path.is_symlink()


def _safe_relative_path(value: str) -> str:
    if "\\" in value:
        raise ValueError(f"unsafe repository path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe repository path: {value!r}")
    return path.as_posix()


def _safe_path_or_empty(value: str) -> str:
    value = value.strip()
    if not value or value == "/dev/null":
        return ""
    return _safe_relative_path(value)


def _diff_path(value: str, prefix: str) -> str:
    value = value.strip().strip('"')
    if value == "/dev/null":
        return ""
    if value.startswith(prefix):
        value = value[len(prefix) :]
    try:
        return _safe_relative_path(value)
    except ValueError:
        return ""


def _side_ranges(hunks: tuple[_HunkRange, ...], side: str) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    for hunk in hunks:
        start, count = (
            (hunk.old_start, hunk.old_count) if side == "base" else (hunk.new_start, hunk.new_count)
        )
        if count > 0:
            ranges.append((start, start + count - 1))
    return tuple(ranges)


def _range_start(ranges: tuple[tuple[int, int], ...]) -> int | None:
    return min((start for start, _ in ranges), default=None)


def _range_end(ranges: tuple[tuple[int, int], ...]) -> int | None:
    return max((end for _, end in ranges), default=None)


def _overlaps(symbol: _Symbol, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(symbol.start <= end and start <= symbol.end for start, end in ranges)


def _path_proximity(left: str, right: str) -> int:
    left_path = PurePosixPath(left)
    right_path = PurePosixPath(right)
    common = 0
    for left_part, right_part in zip(left_path.parts[:-1], right_path.parts[:-1], strict=False):
        if left_part.casefold() != right_part.casefold():
            break
        common += 1
    left_stem = _normalized_stem(left_path.name)
    right_stem = _normalized_stem(right_path.name)
    stem_bonus = 120 if left_stem == right_stem else 0
    if left_stem and right_stem and (left_stem in right_stem or right_stem in left_stem):
        stem_bonus = max(stem_bonus, 80)
    distance = abs(len(left_path.parts) - len(right_path.parts))
    return max(0, common * 35 + stem_bonus - distance * 5)


def _normalized_stem(name: str) -> str:
    value = PurePosixPath(name).stem.casefold()
    for affix in ("test_", "_test", ".test", "_spec", ".spec"):
        value = value.replace(affix, "")
    return value


def _references_symbol(content: str, symbol: str) -> bool:
    if not symbol:
        return False
    name = symbol.rsplit(".", 1)[-1]
    if len(name) < 3:
        return False
    return re.search(rf"(?<![\w$]){re.escape(name)}(?![\w$])", content) is not None


def _references_module(context_path: str, content: str, target_path: str) -> bool:
    target = PurePosixPath(target_path)
    stem = target.stem
    module_parts = list(target.with_suffix("").parts)
    if module_parts and module_parts[0] in {"src", "lib"}:
        module_parts = module_parts[1:]
    dotted = ".".join(module_parts)
    slash = "/".join(module_parts)
    patterns = (
        rf"\b(?:from|import)\s+{re.escape(dotted)}(?:\b|\s+import)",
        rf"\bfrom\s+['\"][^'\"]*{re.escape(slash)}['\"]",
        rf"\brequire\s*\(\s*['\"][^'\"]*{re.escape(stem)}['\"]\s*\)",
    )
    return context_path != target_path and any(re.search(pattern, content) for pattern in patterns)


def _bounded_text(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[:limit]


def _identifier(prefix: str, *parts: str) -> str:
    digest = sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"
