import json
from pathlib import Path

import pytest

import evalyn.demo as demo_module
from evalyn.demo import run_demo


def test_demo_is_keyless_causal_and_persists_evidence(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = run_demo(state_root=tmp_path)

    manifest = json.loads((result.run_path / "manifest.json").read_text())
    assert manifest["causal"] is True
    assert manifest["recorded_inference"] is True
    assert "Base: PASS 3/3" in result.report_path.read_text()
    assert (result.run_path / "probe.py").is_file()


def test_demo_bounds_logs_and_drops_ambient_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-secret")
    noisy_probe = (
        "import os\n"
        "print(os.environ.get('OPENAI_API_KEY', 'unset'))\n"
        "print('x' * 100_000)\n" + demo_module._PROBE
    )
    monkeypatch.setattr(demo_module, "_PROBE", noisy_probe)

    result = run_demo(state_root=tmp_path)

    logs = sorted(result.run_path.glob("*.log"))
    assert len(logs) == 9
    for log in logs:
        payload = log.read_text(encoding="utf-8")
        assert log.stat().st_size <= demo_module._MAX_DEMO_LOG_BYTES
        assert "output truncated" in payload
        assert "ambient-secret" not in payload
        assert "unset" in payload


def test_demo_refuses_a_symlinked_state_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked-state"
    try:
        linked.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")

    with pytest.raises(ValueError, match="unsafe"):
        run_demo(state_root=linked)

    assert list(outside.iterdir()) == []


def test_demo_combined_log_has_an_independent_hard_bound() -> None:
    payload = demo_module._render_log(
        "stdout-head-" + ("x" * demo_module._MAX_DEMO_LOG_BYTES),
        ("y" * demo_module._MAX_DEMO_LOG_BYTES) + "-stderr-tail",
    )

    assert len(payload) == demo_module._MAX_DEMO_LOG_BYTES
    assert payload.startswith(b"stdout-head-")
    assert payload.endswith(b"-stderr-tail")
    assert b"demo log truncated" in payload
