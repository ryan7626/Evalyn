"""Validate and optionally execute Evalyn's causal benchmark corpus."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REVISIONS = ("base", "buggy", "fixed")
OUTCOMES = {"pass", "fail"}
DEFAULT_MANIFEST = Path(__file__).with_name("manifest.json")
_MAX_CAPTURE_BYTES = 4_000
_COMMAND_TIMEOUT_SECONDS = 30
_AVAILABILITY_TIMEOUT_SECONDS = 10


class CorpusError(ValueError):
    """Raised when the benchmark manifest or a fixture is invalid."""


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    framework: str
    summary: str
    fixture_root: Path
    probe_path: Path
    probe_target: Path
    command: tuple[str, ...]
    availability: tuple[str, ...]
    expected: dict[str, str]
    repetitions: int


@dataclass(frozen=True)
class RevisionOutcome:
    revision: str
    results: tuple[Any, ...]

    @property
    def passed(self) -> bool:
        return all(result.exit_code == 0 for result in self.results)


def load_manifest(path: Path = DEFAULT_MANIFEST) -> tuple[BenchmarkCase, ...]:
    """Load and validate every case in a version-one corpus manifest."""
    manifest_path = path.resolve()
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusError(f"Could not read benchmark manifest {manifest_path}: {exc}") from exc

    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise CorpusError("Benchmark manifest must use schema_version 1")
    raw_cases = raw.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise CorpusError("Benchmark manifest must contain a nonempty cases list")
    gate = raw.get("promotion_gate")
    if not isinstance(gate, dict):
        raise CorpusError("Benchmark manifest must define promotion_gate")
    target_cases = gate.get("target_cases")
    implemented_cases = gate.get("implemented_cases")
    met = gate.get("met")
    if not isinstance(target_cases, int) or isinstance(target_cases, bool) or target_cases < 1:
        raise CorpusError("promotion_gate.target_cases must be a positive integer")
    if implemented_cases != len(raw_cases):
        raise CorpusError(
            "promotion_gate.implemented_cases must equal the number of registered cases"
        )
    if target_cases < len(raw_cases):
        raise CorpusError("promotion_gate.target_cases cannot be below implemented cases")
    if not isinstance(met, bool) or met != (len(raw_cases) >= target_cases):
        raise CorpusError("promotion_gate.met does not match implemented progress")

    cases: list[BenchmarkCase] = []
    seen: set[str] = set()
    for index, value in enumerate(raw_cases):
        if not isinstance(value, dict):
            raise CorpusError(f"cases[{index}] must be an object")
        case_id = _required_text(value, "id", index)
        if case_id in seen:
            raise CorpusError(f"Duplicate benchmark case id: {case_id}")
        seen.add(case_id)

        framework = _required_text(value, "framework", index)
        if framework not in {"pytest", "vitest"}:
            raise CorpusError(f"{case_id}: framework must be pytest or vitest")

        fixture_root = _contained(manifest_path.parent, _required_text(value, "fixture", index))
        probe_path = _contained(fixture_root, _required_text(value, "probe", index))
        probe_target = _relative_path(_required_text(value, "probe_target", index))
        command = _argv(value.get("command"), f"{case_id}.command")
        availability = _argv(value.get("availability"), f"{case_id}.availability")
        expected = value.get("expected")
        if not isinstance(expected, dict) or set(expected) != set(REVISIONS):
            raise CorpusError(f"{case_id}: expected must define base, buggy, and fixed")
        normalized_expected = {name: str(expected[name]) for name in REVISIONS}
        if any(outcome not in OUTCOMES for outcome in normalized_expected.values()):
            raise CorpusError(f"{case_id}: expected outcomes must be pass or fail")
        if normalized_expected != {"base": "pass", "buggy": "fail", "fixed": "pass"}:
            raise CorpusError(
                f"{case_id}: corpus cases must establish base-pass/buggy-fail/fixed-pass"
            )

        repetitions = value.get("repetitions", 3)
        if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions != 3:
            raise CorpusError(f"{case_id}: repetitions must be exactly 3")

        if not fixture_root.is_dir():
            raise CorpusError(f"{case_id}: fixture directory does not exist: {fixture_root}")
        if not probe_path.is_file() or not probe_path.read_text(encoding="utf-8").strip():
            raise CorpusError(f"{case_id}: probe must be a nonempty regular file")
        for revision in REVISIONS:
            revision_root = fixture_root / revision
            if not revision_root.is_dir():
                raise CorpusError(f"{case_id}: missing {revision} fixture directory")

        cases.append(
            BenchmarkCase(
                case_id=case_id,
                framework=framework,
                summary=_required_text(value, "summary", index),
                fixture_root=fixture_root,
                probe_path=probe_path,
                probe_target=probe_target,
                command=command,
                availability=availability,
                expected=normalized_expected,
                repetitions=repetitions,
            )
        )
    return tuple(cases)


def execute_case(case: BenchmarkCase, *, strict_tools: bool = False) -> dict[str, object]:
    """Execute one trusted causal fixture and classify base versus buggy evidence."""
    from evalyn.engine.classifier import DifferentialClassifier
    from evalyn.engine.models import ClassificationKind

    if not _tool_available(case):
        message = f"{case.case_id}: required {case.framework} tool is unavailable"
        if strict_tools:
            raise CorpusError(message)
        return {
            "id": case.case_id,
            "framework": case.framework,
            "status": "skipped",
            "reason": message,
        }

    outcomes = {revision: _execute_revision(case, revision) for revision in REVISIONS}
    mismatches = [
        revision
        for revision, outcome in outcomes.items()
        if ("pass" if outcome.passed else "fail") != case.expected[revision]
    ]

    classification = DifferentialClassifier().classify(
        outcomes["base"].results,
        outcomes["buggy"].results,
    )
    causal = (
        not mismatches
        and classification.kind == ClassificationKind.STABLE_CHANGE
        and outcomes["fixed"].passed
    )
    return {
        "id": case.case_id,
        "framework": case.framework,
        "status": "passed" if causal else "failed",
        "summary": case.summary,
        "outcomes": {
            revision: {
                "passed": outcome.passed,
                "runs": [
                    {
                        "exit_code": result.exit_code,
                        "status": result.status.value,
                    }
                    for result in outcome.results
                ],
            }
            for revision, outcome in outcomes.items()
        },
        "classification": classification.kind.value,
        "classification_reason": classification.reason,
        "failure_signature": classification.failure_signature,
        "causal": causal,
        "mismatches": mismatches,
    }


def _execute_revision(case: BenchmarkCase, revision: str) -> RevisionOutcome:
    from evalyn.engine.models import ExecutionResult
    from evalyn.execution import CommandRunner

    results: list[ExecutionResult] = []
    with tempfile.TemporaryDirectory(prefix=f"evalyn-bench-{case.case_id}-{revision}-") as raw:
        workspace = Path(raw) / "workspace"
        shutil.copytree(case.fixture_root / revision, workspace)
        target = (workspace / case.probe_target).resolve()
        target.relative_to(workspace.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(case.probe_path, target)

        with CommandRunner(
            workspace,
            max_output_bytes=_MAX_CAPTURE_BYTES,
            default_timeout_seconds=_COMMAND_TIMEOUT_SECONDS,
        ) as runner:
            for _ in range(case.repetitions):
                completed = runner.run(_local_tool_command(case, case.command))
                if completed.timed_out or completed.error is not None:
                    detail = completed.error or "command did not complete"
                    raise CorpusError(f"{case.case_id}:{revision}: {detail}")
                assert completed.exit_code is not None
                output = (completed.stderr or completed.stdout)[-_MAX_CAPTURE_BYTES:]
                if completed.exit_code == 0:
                    result = ExecutionResult.passed(
                        stdout=completed.stdout[-_MAX_CAPTURE_BYTES:],
                        duration_seconds=completed.duration_seconds,
                    )
                else:
                    result = ExecutionResult.failed(
                        exit_code=completed.exit_code,
                        stderr=output,
                        duration_seconds=completed.duration_seconds,
                        failure_signature=(f"{case.case_id}:{revision}:exit-{completed.exit_code}"),
                    )
                results.append(result)
    return RevisionOutcome(revision=revision, results=tuple(results))


def _tool_available(case: BenchmarkCase) -> bool:
    from evalyn.execution import CommandRunner

    command = _local_tool_command(case, case.command)
    if shutil.which(command[0]) is None and not Path(command[0]).is_file():
        return False
    if not case.availability:
        return True
    with CommandRunner(
        case.fixture_root,
        max_output_bytes=1_024,
        default_timeout_seconds=_AVAILABILITY_TIMEOUT_SECONDS,
    ) as runner:
        completed = runner.run(
            _local_tool_command(case, case.availability),
            cwd=case.fixture_root / "base",
        )
    return completed.succeeded


def _local_tool_command(
    case: BenchmarkCase,
    command: tuple[str, ...],
) -> tuple[str, ...]:
    """Use an already-installed Vitest binary without allowing npx downloads."""
    if case.framework == "vitest" and command[:3] == ("npx", "--no-install", "vitest"):
        executable = shutil.which("vitest")
        if executable is not None:
            return (executable, *command[3:])
    return command


def _required_text(value: dict[str, object], key: str, index: int) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise CorpusError(f"cases[{index}].{key} must be a nonempty string")
    return result.strip()


def _argv(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(part, str) for part in value):
        raise CorpusError(f"{label} must be a nonempty string array")
    if any(not part for part in value):
        raise CorpusError(f"{label} cannot contain empty arguments")
    return tuple(value)


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or value.strip() != value:
        raise CorpusError(f"Unsafe relative fixture path: {value!r}")
    return path


def _contained(root: Path, value: str) -> Path:
    relative = _relative_path(value)
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise CorpusError(f"Fixture path escapes corpus root: {value!r}") from exc
    return candidate


def _render_result(result: dict[str, object]) -> None:
    if result["status"] == "skipped":
        print(f"{result['id']}\n  skipped    {result['reason']}")
        return
    outcomes = result["outcomes"]
    assert isinstance(outcomes, dict)
    print(result["id"])
    for revision in REVISIONS:
        detail = outcomes[revision]
        assert isinstance(detail, dict)
        runs = detail["runs"]
        assert isinstance(runs, list)
        label = "PASS" if detail["passed"] else "FAIL"
        print(f"  {revision:<10} {label} {len(runs)}/{len(runs)}")
    print(f"  evidence   {result['classification']}")
    causal = "yes (base pass -> buggy fail -> fixed pass)" if result["causal"] else "no"
    print(f"  causal     {causal}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--validate", action="store_true", help="validate corpus files only")
    parser.add_argument("--execute", action="store_true", help="execute available causal fixtures")
    parser.add_argument(
        "--strict-tools",
        action="store_true",
        help="fail instead of skipping missing tools",
    )
    parser.add_argument("--case", action="append", dest="case_ids", default=[])
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)

    try:
        cases = load_manifest(args.manifest)
        if args.case_ids:
            selected = tuple(case for case in cases if case.case_id in set(args.case_ids))
            missing = sorted(set(args.case_ids) - {case.case_id for case in selected})
            if missing:
                raise CorpusError(f"Unknown benchmark case(s): {', '.join(missing)}")
        else:
            selected = cases

        if not args.execute:
            print(f"Validated {len(selected)} benchmark case(s) from {args.manifest}")
            return 0

        results = [execute_case(case, strict_tools=args.strict_tools) for case in selected]
        if args.json_output:
            print(json.dumps({"schema_version": 1, "results": results}, indent=2, sort_keys=True))
        else:
            for result in results:
                _render_result(result)
        return 1 if any(result["status"] == "failed" for result in results) else 0
    except (CorpusError, OSError) as exc:
        print(f"benchmark error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
