"""Run one causal Evalyn benchmark without an API key."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    print("Evalyn 0.2 no-key differential demo", flush=True)
    print("Same trusted probe, three runs per revision.\n", flush=True)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "bench.runner",
            "--execute",
            "--strict-tools",
            "--case",
            "pytest-discount-rounding",
        ],
        cwd=ROOT,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
