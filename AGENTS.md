# Repository Guidelines

## Project Structure & Module Organization

Evalyn uses a Python `src/` layout. Core package code lives in `src/evalyn/`, with the CLI entry point in `src/evalyn/cli.py`. Planning logic is under `src/evalyn/planner/`, execution and repair flow under `src/evalyn/runner/`, reporting under `src/evalyn/reporter/`, LLM providers under `src/evalyn/llm/`, and shared helpers under `src/evalyn/utils/`. Tests live in `tests/` and mirror feature modules with files such as `tests/test_config.py`. `demo.py` is a lightweight demonstration script; package metadata and pytest configuration are in `pyproject.toml`.

## Build, Test, and Development Commands

- `python3 -m venv .venv && source .venv/bin/activate`: create and activate a virtual environment.
- `pip install -e '.[dev]'`: install Evalyn in editable mode with pytest tooling.
- `pip install -e '.[browser]' && playwright install chromium`: add optional browser testing support.
- `pytest`: run the full test suite configured for `tests/test_*.py`.
- `pytest --cov=evalyn`: run tests with coverage reporting.
- `evalyn --help`: verify the installed CLI and inspect commands.
- `evalyn init --repo /path/to/project`: scaffold target-repo agent files for manual testing.

## Coding Style & Naming Conventions

Use standard Python style with 4-space indentation, clear type hints where useful, and `pathlib.Path` for filesystem paths. Keep modules focused by domain, following the existing `planner`, `runner`, `reporter`, `llm`, and `utils` boundaries. Name tests `test_*.py`, test functions `test_*`, and group related assertions in `Test*` classes when useful. No formatter or linter is currently configured, so keep edits consistent with surrounding code.

## Testing Guidelines

The project uses pytest, with settings in `pyproject.toml`. Add or update tests for behavior changes, especially around git operations, config loading, plan parsing, rollback, and repair flow. Prefer `tmp_path` fixtures for filesystem scenarios and keep tests isolated from real repositories or credentials. Use coverage locally for broader changes.

## Commit & Pull Request Guidelines

Recent history uses short, imperative summaries with occasional prefixes, for example `chore: ...`, `feat- ...`, and concise update messages. Prefer `type: summary` when a clear type applies, such as `fix: handle missing plan file`. Pull requests should describe the change, list test commands run, link related issues, and include terminal output when CLI behavior changes.

## Security & Configuration Tips

Do not commit `.env`, `.agent/`, temporary repos, coverage output, or virtual environments; these are ignored already. LLM credentials should come from environment variables such as `ANTHROPIC_API_KEY` or `OPENROUTER_API_KEY`. Evalyn leaves target-repo fixes unstaged for review, so preserve that safety model when changing runner behavior.
