"""Tests for the trusted framework-adapter boundary."""

from dataclasses import FrozenInstanceError, fields
from pathlib import PurePosixPath

import pytest

from evalyn.adapters.base import AdapterError, ProbeArtifact, TestAdapter
from evalyn.engine.models import ProbeSource


class DummyAdapter(TestAdapter):
    name = "dummy"
    media_type = "text/plain"

    def detect(self, repo_path):
        return True

    def probe_path(self, probe_id):
        return PurePosixPath(".evalyn", "probes", f"{probe_id}.txt")

    def probe_command(self, relative_path):
        return ("dummy-runner", relative_path.as_posix())


def test_adapter_content_addresses_and_normalizes_source():
    adapter = DummyAdapter()

    first = adapter.build_artifact("probe()")
    second = adapter.build_artifact("probe()\n\n")

    assert first == second
    assert first.source == "probe()\n"
    assert len(first.probe_id) == 16
    assert first.relative_path.as_posix().endswith(f"{first.probe_id}.txt")


def test_model_authorable_probe_has_no_command_or_path_fields():
    assert {field.name for field in fields(ProbeSource)} == {
        "source",
        "title",
        "oracle",
        "oracle_provenance",
        "metadata",
    }
    with pytest.raises(TypeError):
        ProbeSource(source="probe()", command=("sh", "-c", "bad"))


def test_artifact_is_immutable():
    artifact = DummyAdapter().build_artifact("probe()")

    with pytest.raises(FrozenInstanceError):
        artifact.command = ("other",)


def test_artifact_rejects_parent_traversal():
    source = "probe()\n"
    safe = DummyAdapter().build_artifact(source)

    with pytest.raises(AdapterError, match="traverse"):
        ProbeArtifact(
            probe_id=safe.probe_id,
            adapter="dummy",
            source=source,
            relative_path=PurePosixPath("..", "stolen.txt"),
            command=("dummy-runner", "../stolen.txt"),
            source_sha256=safe.source_sha256,
            media_type="text/plain",
        )


def test_empty_source_is_rejected():
    with pytest.raises(AdapterError, match="empty"):
        DummyAdapter().build_artifact("   \n")
