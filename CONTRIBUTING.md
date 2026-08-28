# Contributing to Evalyn

Evalyn is building an evidence-first regression detector for Git diffs. The
project is currently a private alpha and cannot accept external contributions
until a license is selected.

## Development setup

~~~bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
~~~

Run the same checks used by CI before opening a change:

~~~bash
ruff check src tests bench examples demo.py
ruff format --check src tests bench examples demo.py
pytest --cov=evalyn --cov-fail-under=75
python -m bench.validate_repository
python -m bench.runner --validate
python demo.py
python -m build
~~~

CI gates the entire Python source, test, benchmark, and example tree. The total
coverage floor is 75%; safety-critical execution changes should add focused
failure-path tests rather than relying on aggregate coverage alone.

## Engineering rules

- Never mutate, reset, clean, or switch branches in the repository being
  evaluated. Work belongs in Evalyn-owned disposable worktrees.
- Never execute model-authored paths or shell commands. Framework adapters own
  generated paths and argument arrays.
- Generated probes run without network or developer credentials.
- Add isolated tests for all behavior changes, especially cleanup, Git state,
  path containment, process termination, classification, and report schemas.
- Findings describe evidence. Automated output must not be called a verified
  regression without human confirmation or benchmark-causal proof.
- Benchmark promotion requires 60 distinct reviewed causal fixtures. There are
  currently two seed fixtures; cosmetic duplicates do not count.

Commits should use a short imperative summary such as
<code>fix: preserve dirty working trees</code>.
