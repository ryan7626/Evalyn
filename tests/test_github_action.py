from __future__ import annotations

import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ACTION_PATH = ROOT / "action.yml"
DOCS_PATH = ROOT / "docs" / "github-actions.md"


def _action() -> dict:
    payload = yaml.safe_load(ACTION_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _check_step() -> dict:
    steps = _action()["runs"]["steps"]
    return next(step for step in steps if step.get("id") == "check")


def test_composite_action_wires_evidence_outputs_to_the_check_step() -> None:
    outputs = _action()["outputs"]
    expected = {
        "run-id",
        "verdict",
        "run-status",
        "report",
        "manifest",
        "sarif",
        "finding-count",
        "gating-finding-count",
        "exit-code",
    }

    assert set(outputs) == expected
    for name in expected:
        assert outputs[name]["value"] == f"${{{{ steps.check.outputs.{name} }}}}"
        assert outputs[name]["description"]


def test_check_script_is_valid_bash_and_keeps_inputs_out_of_source_code() -> None:
    step = _check_step()
    script = step["run"]

    parsed = subprocess.run(
        ["bash", "-n"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert parsed.returncode == 0, parsed.stderr
    python_blocks = re.findall(r"<<'PY'\n(.*?)\nPY", script, flags=re.DOTALL)
    assert len(python_blocks) == 3
    for block in python_blocks:
        compile(block, str(ACTION_PATH), "exec")
    assert "${{ inputs." not in script
    assert '"${args[@]}"' in script
    assert step["env"] == {
        "EVALYN_ACTION_FAIL_ON": "${{ inputs.fail-on }}",
        "EVALYN_ACTION_INTENT": "${{ inputs.intent }}",
        "EVALYN_ACTION_MODEL": "${{ inputs.model }}",
        "EVALYN_ACTION_OUTPUT": "${{ inputs.output }}",
        "EVALYN_ACTION_PROVIDER": "${{ inputs.provider }}",
        "EVALYN_ACTION_RANGE": "${{ inputs.range }}",
        "EVALYN_ACTION_RUNTIME": "${{ inputs.runtime }}",
        "EVALYN_ACTION_STATE_ROOT": "${{ inputs.state-root }}",
    }


def test_action_preserves_cli_status_and_emits_metadata_on_nonzero_paths() -> None:
    script = _check_step()["run"]

    assert 'set +e\n"${args[@]}"\nstatus=$?' in script
    assert 'args+=(-- "$EVALYN_ACTION_RANGE")' in script
    assert 'check "$EVALYN_ACTION_RANGE"' not in script
    assert script.rstrip().endswith('exit "$status"')
    assert "exit-code=2" in script
    assert "printf 'exit-code=%s\\n' \"$status\"" in script
    assert '"exit-code": str(status)' in script
    assert "GITHUB_OUTPUT" in script
    assert "GITHUB_STEP_SUMMARY" in script
    assert "This result is not evidence that the candidate is regression-free." in script
    caveat_verdicts = script.split("if verdict in {", 1)[1].split("}:", 1)[0]
    assert '"no_differential_observed"' in caveat_verdicts


def test_action_attributes_runs_and_artifacts_fail_closed() -> None:
    script = _check_step()["run"]

    assert "from evalyn.run_store import RunStore" in script
    assert 'snapshot = {"ok": False, "run_ids": []}' in script
    assert "if snapshot_ok" in script
    assert "run.run_id not in previous" in script
    assert "run = new_runs[0] if len(new_runs) == 1 else None" in script
    assert "follow_symlinks=False" in script
    assert "stat.S_ISREG" in script
    assert "resolved.relative_to(beneath.resolve(strict=True))" in script
    assert "if digest(published_sarif_path) == digest(internal_sarif)" in script
    assert "sarif = internal_sarif" in script
    assert 'echo "sarif=$output_path"' not in script


def test_dogfood_documentation_is_explicitly_private_and_parses_as_yaml() -> None:
    text = DOCS_PATH.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    match = re.search(r"~~~yaml\n(.*?)\n~~~", text, flags=re.DOTALL)
    assert match is not None
    example = yaml.safe_load(match.group(1))

    assert example["jobs"]["evalyn"]["permissions"] == {"contents": "read"}
    assert example["jobs"]["evalyn"]["if"] == "${{ github.ref_protected }}"
    assert example["jobs"]["evalyn"]["environment"] == "evalyn-private-alpha"
    assert "source-only private-alpha preview" in text
    assert "not licensed or published for external reuse" in normalized
    assert "untrusted pull request" in text
    assert "workflow_dispatch" in text
    assert "github.event.pull_request" not in text
    assert "no_differential_observed" in text
    assert "change_observed" in text
    assert "not proof that the candidate is regression-free" in normalized
