"""Convenience entry point for the deterministic no-key demo."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    return subprocess.run(
        [sys.executable, "examples/no-key-demo/run_demo.py"],
        cwd=ROOT,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
