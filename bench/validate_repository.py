"""Validate repository-owned YAML, JSON, Markdown links, and benchmark assets."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

from bench.runner import CorpusError, load_manifest

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def _yaml_files() -> tuple[Path, ...]:
    root_action = (ROOT / "action.yml",) if (ROOT / "action.yml").is_file() else ()
    return (
        root_action
        + tuple(sorted((ROOT / ".github").rglob("*.yml")))
        + tuple(sorted((ROOT / ".github").rglob("*.yaml")))
    )


def validate_yaml(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return [f"{path.relative_to(ROOT)}: invalid YAML: {exc}"]
    if not isinstance(value, dict):
        return [f"{path.relative_to(ROOT)}: top-level YAML value must be a mapping"]

    relative = path.relative_to(ROOT).as_posix()
    if "/workflows/" in f"/{relative}":
        trigger = value.get("on", value.get(True))
        if not value.get("name") or trigger is None or not isinstance(value.get("jobs"), dict):
            errors.append(f"{relative}: workflow requires name, on, and jobs")
    if (relative == "action.yml" or "/actions/" in f"/{relative}") and path.name in {
        "action.yml",
        "action.yaml",
    }:
        runs = value.get("runs")
        if (
            not value.get("name")
            or not value.get("description")
            or not isinstance(runs, dict)
            or runs.get("using") != "composite"
            or not isinstance(runs.get("steps"), list)
        ):
            errors.append(f"{relative}: composite action metadata is incomplete")
    return errors


def validate_json(path: Path) -> list[str]:
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{path.relative_to(ROOT)}: invalid JSON: {exc}"]
    return []


def validate_markdown(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    for raw_target in MARKDOWN_LINK.findall(text):
        target = raw_target.strip()
        if (
            not target
            or target.startswith(("#", "http://", "https://", "mailto:"))
            or "://" in target
        ):
            continue
        target = target.split("#", 1)[0].replace("%20", " ")
        candidate = (path.parent / target).resolve()
        try:
            candidate.relative_to(ROOT)
        except ValueError:
            errors.append(f"{path.relative_to(ROOT)}: link escapes repository: {raw_target}")
            continue
        if not candidate.exists():
            errors.append(f"{path.relative_to(ROOT)}: broken link: {raw_target}")
    return errors


def main() -> int:
    errors: list[str] = []
    for path in _yaml_files():
        errors.extend(validate_yaml(path))
    for path in sorted(ROOT.rglob("*.json")):
        if any(part in {".git", ".venv", ".pytest_cache"} for part in path.parts):
            continue
        errors.extend(validate_json(path))
    for path in sorted(ROOT.rglob("*.md")):
        if any(part in {".git", ".venv", ".pytest_cache"} for part in path.parts):
            continue
        errors.extend(validate_markdown(path))
    try:
        load_manifest()
    except CorpusError as exc:
        errors.append(f"bench/manifest.json: {exc}")

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("Repository assets are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
