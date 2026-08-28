"""Deterministic, credential-free Evalyn demonstration."""

from __future__ import annotations

import os
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from evalyn.execution import CommandRunner
from evalyn.run_store import RunStore, default_state_root


class DemoError(RuntimeError):
    pass


@dataclass(frozen=True)
class DemoResult:
    run_path: Path
    report_path: Path


_PROBE = """\
from pricing import discounted_price

def test_discount_rounds_only_the_final_price():
    assert discounted_price(10.05, 10) == 9.05

if __name__ == "__main__":
    test_discount_rounds_only_the_final_price()
"""

_REVISIONS = {
    "base": """\
from decimal import Decimal, ROUND_HALF_UP

def discounted_price(price, percent):
    value = Decimal(str(price)) * (Decimal(100 - percent) / Decimal(100))
    return float(value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
""",
    "buggy": """\
from decimal import Decimal, ROUND_HALF_UP

def discounted_price(price, percent):
    rounded_price = Decimal(str(price)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
    value = rounded_price * (Decimal(100 - percent) / Decimal(100))
    return float(value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
""",
    "fixed": """\
from decimal import Decimal, ROUND_HALF_UP

def discounted_price(price, percent):
    value = Decimal(str(price)) * (Decimal(100 - percent) / Decimal(100))
    return float(value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
""",
}

_MAX_DEMO_STREAM_BYTES = 32 * 1024
_MAX_DEMO_LOG_BYTES = 64 * 1024
_DEMO_TIMEOUT_SECONDS = 10


def run_demo(*, state_root: Path | None = None) -> DemoResult:
    """Run a trusted causal fixture using a clearly recorded probe."""
    requested_root = (state_root or default_state_root()).expanduser()
    root = Path(os.path.abspath(os.fspath(requested_root))) / "demo"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_id = f"{timestamp}-{uuid.uuid4().hex[:12]}"
    run_path = root / run_id
    RunStore.write_text(run_path / "probe.py", _PROBE)
    outcomes: dict[str, list[int]] = {}
    with tempfile.TemporaryDirectory(prefix="evalyn-demo-") as raw:
        workspace = Path(raw)
        with CommandRunner(
            workspace,
            max_output_bytes=_MAX_DEMO_STREAM_BYTES,
            default_timeout_seconds=_DEMO_TIMEOUT_SECONDS,
        ) as runner:
            for revision, source in _REVISIONS.items():
                revision_path = workspace / revision
                revision_path.mkdir()
                (revision_path / "pricing.py").write_text(source, encoding="utf-8")
                (revision_path / "test_discount.py").write_text(_PROBE, encoding="utf-8")
                outcomes[revision] = []
                for repetition in range(1, 4):
                    completed = runner.run(
                        [sys.executable, "test_discount.py"],
                        cwd=revision_path,
                    )
                    if completed.timed_out or completed.error is not None:
                        detail = completed.error or "command did not complete"
                        raise DemoError(f"{revision} fixture execution failed: {detail}")
                    assert completed.exit_code is not None
                    outcomes[revision].append(completed.exit_code)
                    RunStore.write_bytes(
                        run_path / f"{revision}-{repetition}.log",
                        _render_log(completed.stdout, completed.stderr),
                    )
    causal = (
        outcomes["base"] == [0, 0, 0]
        and all(code != 0 for code in outcomes["buggy"])
        and outcomes["fixed"] == [0, 0, 0]
    )
    if not causal:
        raise DemoError(f"Bundled causal fixture produced unexpected outcomes: {outcomes}")
    report = "\n".join(
        (
            "# Evalyn no-key demo",
            "",
            "> Recorded model probe; live deterministic fixture execution.",
            "",
            "## LIKELY REGRESSION — EVL-DEMO0001",
            "",
            "Discount calculation rounds the input before applying the percentage.",
            "",
            "- Oracle: prices are rounded only after the discount is applied",
            "- Provenance: `pre_change_behavior`",
            "- Base: PASS 3/3",
            "- Candidate: FAIL 3/3 — `AssertionError`",
            "- Fixed control: PASS 3/3",
            "- Causal proof: base-pass → buggy-fail → fixed-pass",
            "",
            "The demo makes no model request and uses no API key.",
            "",
        )
    )
    report_path = run_path / "report.md"
    RunStore.write_text(report_path, report)
    RunStore.write_json(
        run_path / "manifest.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "recorded_inference": True,
            "live_fixture_execution": True,
            "outcomes": outcomes,
            "causal": causal,
        },
    )
    return DemoResult(run_path, report_path)


def _render_log(stdout: str, stderr: str) -> bytes:
    """Render a combined demo log with an independent hard byte bound."""
    payload = (stdout + stderr).encode("utf-8")
    if len(payload) <= _MAX_DEMO_LOG_BYTES:
        return payload
    marker = b"\n... demo log truncated ...\n"
    retained = _MAX_DEMO_LOG_BYTES - len(marker)
    head_size = (retained + 1) // 2
    tail_size = retained // 2
    return payload[:head_size] + marker + payload[-tail_size:]
