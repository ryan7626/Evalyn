from dataclasses import FrozenInstanceError

import pytest

from evalyn.analysis import build_change_graph


def _write(root, relative, content):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _repository_pair(tmp_path):
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    base.mkdir()
    candidate.mkdir()
    base_retry = """\
class RetryPolicy:
    def delay(self, attempt):
        if attempt < 0:
            raise ValueError("attempt must be non-negative")
        return attempt * 2
"""
    candidate_retry = base_retry.replace("attempt * 2", "attempt * 3")
    for root, retry in ((base, base_retry), (candidate, candidate_retry)):
        _write(root, "src/retry.py", retry)
        _write(
            root,
            "tests/test_retry.py",
            "from src.retry import RetryPolicy\n\n"
            "def test_negative_attempt_is_rejected():\n"
            "    try:\n"
            "        RetryPolicy().delay(-1)\n"
            "        assert False, 'negative attempts must be rejected'\n"
            "    except ValueError:\n"
            "        assert True\n",
        )
        _write(
            root,
            "src/client.py",
            "from src.retry import RetryPolicy\n\npolicy = RetryPolicy()\n",
        )
        _write(root, "docs/api.md", "RetryPolicy must reject negative attempt values.\n")
        _write(
            root,
            "src/retry.pyi",
            "class RetryPolicy:\n    def delay(self, attempt: int) -> int: ...\n",
        )
        _write(root, "pyproject.toml", "[project]\nname = 'sample'\n")
    _write(base, "src/legacy.py", "def legacy_retry():\n    return 1\n")
    return base, candidate


def test_change_graph_finds_changed_symbols_deleted_files_and_related_context(tmp_path):
    base, candidate = _repository_pair(tmp_path)
    diff = """\
diff --git a/src/retry.py b/src/retry.py
--- a/src/retry.py
+++ b/src/retry.py
@@ -1,5 +1,5 @@
 class RetryPolicy:
     def delay(self, attempt):
         if attempt < 0:
             raise ValueError("attempt must be non-negative")
-        return attempt * 2
+        return attempt * 3
diff --git a/src/legacy.py b/src/legacy.py
deleted file mode 100644
--- a/src/legacy.py
+++ /dev/null
@@ -1,2 +0,0 @@
-def legacy_retry():
-    return 1
"""

    graph = build_change_graph(base, candidate, diff=diff)

    assert graph.changed_paths == ("src/legacy.py", "src/retry.py")
    assert any(
        target.path == "src/retry.py"
        and target.symbol == "RetryPolicy.delay"
        and target.change_kind == "modified"
        and "attempt * 2" in target.base_definition
        and "attempt * 3" in target.candidate_definition
        for target in graph.targets
    )
    assert any(
        target.path == "src/legacy.py"
        and target.symbol == "legacy_retry"
        and target.change_kind == "deleted"
        and not target.candidate_definition
        for target in graph.targets
    )
    categories = {(item.path, item.category, item.side) for item in graph.context}
    assert ("tests/test_retry.py", "existing_test", "both") in categories
    assert ("docs/api.md", "documentation", "both") in categories
    assert ("src/retry.pyi", "type_contract", "both") in categories
    assert ("pyproject.toml", "manifest", "both") in categories
    assert ("src/client.py", "implementation", "both") in categories
    assert tuple(item.rank for item in graph.context) == tuple(range(1, len(graph.context) + 1))
    assert graph.context[0].path in {"src/legacy.py", "src/retry.py"}


def test_change_graph_is_deterministic_and_evidence_is_immutable(tmp_path):
    base, candidate = _repository_pair(tmp_path)

    first = build_change_graph(base, candidate, changed_paths=("src/retry.py",))
    second = build_change_graph(base, candidate, changed_paths=("src/retry.py",))

    assert first == second
    assert all(len(item.content_sha256) == 64 for item in first.context)
    assert all(item.reasons for item in first.context)
    with pytest.raises(FrozenInstanceError):
        first.context[0].score = 0


def test_oracles_only_use_base_visible_path_derived_sources(tmp_path):
    base, candidate = _repository_pair(tmp_path)
    _write(candidate, "docs/new-contract.md", "RetryPolicy must return exactly seven.\n")

    graph = build_change_graph(
        base,
        candidate,
        changed_paths=("src/retry.py", "docs/new-contract.md"),
    )

    assert graph.oracles
    assert all(oracle.source_side in {"base", "both"} for oracle in graph.oracles)
    assert all(oracle.source_path != "docs/new-contract.md" for oracle in graph.oracles)
    assert {oracle.category for oracle in graph.oracles} >= {
        "existing_test",
        "documentation",
        "type_contract",
    }
    evidence = {item.evidence_id: item for item in graph.context}
    assert all(oracle.excerpt in evidence[oracle.evidence_id].content for oracle in graph.oracles)


def test_change_graph_extracts_typescript_symbols_and_vitest_context(tmp_path):
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    base.mkdir()
    candidate.mkdir()
    _write(base, "src/codec.ts", "export function encode(value: string) { return value; }\n")
    _write(
        candidate,
        "src/codec.ts",
        "export function encode(value: string) { return value.toUpperCase(); }\n",
    )
    for root in (base, candidate):
        _write(
            root,
            "tests/codec.test.ts",
            "import { expect, test } from 'vitest';\n"
            "import { encode } from '../src/codec';\n"
            "test('identity', () => expect(encode('a')).toEqual('a'));\n",
        )
        _write(root, "package.json", '{"devDependencies":{"vitest":"1"}}\n')
    diff = """\
diff --git a/src/codec.ts b/src/codec.ts
--- a/src/codec.ts
+++ b/src/codec.ts
@@ -1 +1 @@
-export function encode(value: string) { return value; }
+export function encode(value: string) { return value.toUpperCase(); }
"""

    graph = build_change_graph(base, candidate, diff=diff)

    assert any(
        target.symbol == "encode" and target.symbol_kind == "function" for target in graph.targets
    )
    assert any(item.path == "tests/codec.test.ts" for item in graph.context)
    assert any(
        oracle.source_path == "tests/codec.test.ts" and "expect(" in oracle.excerpt
        for oracle in graph.oracles
    )


def test_change_graph_rejects_traversal_and_honors_context_budget(tmp_path):
    base, candidate = _repository_pair(tmp_path)

    with pytest.raises(ValueError, match="unsafe repository path"):
        build_change_graph(base, candidate, changed_paths=("../outside.py",))

    graph = build_change_graph(
        base,
        candidate,
        changed_paths=("src/retry.py",),
        max_context_chars=100,
        max_context_files=2,
    )
    assert len(graph.context) <= 2
    assert sum(len(item.content) for item in graph.context) <= 100
    assert "context_budget_truncated" in graph.context[-1].reasons


def test_repository_path_allowlist_excludes_untracked_context_but_keeps_changed_files(
    tmp_path,
):
    base, candidate = _repository_pair(tmp_path)
    _write(candidate, "tests/test_untracked.py", "def test_poison():\n    assert False\n")

    graph = build_change_graph(
        base,
        candidate,
        changed_paths=("src/retry.py",),
        repository_paths=(
            "src/retry.py",
            "tests/test_retry.py",
            "src/client.py",
            "docs/api.md",
            "src/retry.pyi",
            "pyproject.toml",
        ),
    )

    assert all(item.path != "tests/test_untracked.py" for item in graph.context)
    assert any(item.path == "src/retry.py" for item in graph.context)
