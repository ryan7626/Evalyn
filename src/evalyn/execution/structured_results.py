"""Bounded, framework-aware parsing for machine-readable probe outcomes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

PYTEST_RESULT_PREFIX = "EVALYN_PYTEST_RESULT_V1:"


@dataclass(frozen=True)
class StructuredTestResult:
    """Small normalized result used as execution evidence.

    Console text remains available for diagnostics, but classification can use a
    deterministic signature derived from this normalized representation.
    """

    framework: str
    total: int
    passed: int
    failed: int
    skipped: int
    collection_errors: int = 0
    failure_signature: str = ""

    def __post_init__(self) -> None:
        values = (
            self.total,
            self.passed,
            self.failed,
            self.skipped,
            self.collection_errors,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values
        ):
            raise ValueError("structured test counts must be non-negative integers")
        if self.passed + self.failed + self.skipped != self.total:
            raise ValueError("structured test counts must exactly equal total")

    def metadata(self) -> dict[str, str]:
        return {
            "result_protocol": "evalyn.test-result.v1",
            "result_framework": self.framework,
            "test_total": str(self.total),
            "test_passed": str(self.passed),
            "test_failed": str(self.failed),
            "test_skipped": str(self.skipped),
            "collection_errors": str(self.collection_errors),
        }


def parse_structured_result(adapter: str, stdout: str) -> StructuredTestResult | None:
    """Parse the adapter's bounded stdout, returning ``None`` on malformed data."""

    try:
        if adapter == "pytest":
            return _parse_pytest(stdout)
        if adapter == "vitest":
            return _parse_vitest(stdout)
    except (TypeError, ValueError):
        return None
    return None


def _parse_pytest(stdout: str) -> StructuredTestResult | None:
    payload: object | None = None
    marker_count = 0
    for line in stdout.splitlines():
        if line.startswith(PYTEST_RESULT_PREFIX):
            marker_count += 1
            try:
                payload = json.loads(line[len(PYTEST_RESULT_PREFIX) :])
            except (json.JSONDecodeError, UnicodeError):
                return None
    if marker_count != 1 or not isinstance(payload, dict) or payload.get("version") != 1:
        return None
    if payload.get("framework") != "pytest":
        return None
    tests = payload.get("tests")
    collection_errors = payload.get("collection_errors")
    if not isinstance(tests, list) or not isinstance(collection_errors, list):
        return None
    outcomes: list[tuple[str, str, str]] = []
    counts = {"passed": 0, "failed": 0, "skipped": 0}
    for value in tests:
        if not isinstance(value, dict):
            return None
        nodeid = value.get("nodeid")
        outcome = value.get("outcome")
        detail = value.get("detail", "")
        if (
            not isinstance(nodeid, str)
            or not nodeid
            or outcome not in counts
            or not isinstance(detail, str)
        ):
            return None
        counts[outcome] += 1
        if outcome == "failed":
            outcomes.append((nodeid, outcome, detail))
    errors: list[str] = []
    for value in collection_errors:
        if not isinstance(value, str):
            return None
        errors.append(value)
    signature = _signature({"failures": outcomes, "collection_errors": errors})
    return StructuredTestResult(
        framework="pytest",
        total=len(tests),
        passed=counts["passed"],
        failed=counts["failed"],
        skipped=counts["skipped"],
        collection_errors=len(errors),
        failure_signature=signature if outcomes or errors else "",
    )


def _parse_vitest(stdout: str) -> StructuredTestResult | None:
    payload = _last_json_object(stdout)
    if payload is None:
        return None
    keys = (
        "numTotalTests",
        "numPassedTests",
        "numFailedTests",
        "numPendingTests",
    )
    values = [_plain_non_negative_int(payload.get(key)) for key in keys]
    if any(value is None for value in values):
        return None
    total, passed, failed, skipped = (int(value) for value in values)
    messages: list[str] = []
    suite_errors: list[str] = []
    results = payload.get("testResults", [])
    if not isinstance(results, list):
        return None
    for suite in results:
        if not isinstance(suite, dict):
            return None
        assertion_results = suite.get("assertionResults", [])
        if not isinstance(assertion_results, list):
            return None
        suite_status = suite.get("status", "")
        suite_message = suite.get("message", suite.get("failureMessage", ""))
        if suite_status == "failed" and not assertion_results:
            if not isinstance(suite_message, str):
                return None
            suite_errors.append(suite_message[:16_384] or "suite failed before tests ran")
        for assertion in assertion_results:
            if not isinstance(assertion, dict) or assertion.get("status") != "failed":
                continue
            name = assertion.get("fullName", assertion.get("title", "unknown"))
            failure_messages = assertion.get("failureMessages", [])
            if not isinstance(name, str) or not isinstance(failure_messages, list):
                return None
            detail = "\n".join(item for item in failure_messages if isinstance(item, str))
            messages.append(f"{name}\n{detail[:16_384]}")
    return StructuredTestResult(
        framework="vitest",
        total=total,
        passed=passed,
        failed=failed,
        skipped=skipped,
        collection_errors=len(suite_errors),
        failure_signature=(
            _signature({"failures": messages, "suite_errors": suite_errors})
            if failed or suite_errors
            else ""
        ),
    )


def _last_json_object(value: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    result: dict[str, Any] | None = None
    index = 0
    while index < len(value):
        start = value.find("{", index)
        if start < 0:
            break
        try:
            candidate, consumed = decoder.raw_decode(value[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(candidate, dict):
            result = candidate
        index = start + max(consumed, 1)
    return result


def _plain_non_negative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _signature(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(canonical.encode("utf-8")).hexdigest()


# This source is materialized by the trusted executor and loaded as a pytest
# plugin. It emits one bounded JSON line after the session, independent of
# pytest's human-oriented terminal formatting.
PYTEST_REPORTER_SOURCE = f'''"""Evalyn-owned pytest result reporter."""
import json

_tests = {{}}
_collection_errors = []


def _bounded(value):
    return str(value)[:16384]


def pytest_collectreport(report):
    if report.failed:
        _collection_errors.append(_bounded(report.longrepr))


def pytest_runtest_logreport(report):
    current = _tests.get(report.nodeid)
    if report.when == "call":
        _tests[report.nodeid] = {{
            "nodeid": report.nodeid,
            "outcome": report.outcome,
            "detail": _bounded(report.longrepr) if report.failed else "",
        }}
    elif report.when in {{"setup", "teardown"}} and report.failed:
        _tests[report.nodeid] = {{
            "nodeid": report.nodeid,
            "outcome": "failed",
            "detail": _bounded(report.longrepr),
        }}
    elif report.when == "setup" and report.skipped and current is None:
        _tests[report.nodeid] = {{
            "nodeid": report.nodeid,
            "outcome": "skipped",
            "detail": "",
        }}


def pytest_sessionfinish(session, exitstatus):
    payload = {{
        "version": 1,
        "framework": "pytest",
        "exitstatus": int(exitstatus),
        "tests": [_tests[key] for key in sorted(_tests)],
        "collection_errors": sorted(_collection_errors),
    }}
    print("{PYTEST_RESULT_PREFIX}" + json.dumps(payload, sort_keys=True, separators=(",", ":")))
'''
