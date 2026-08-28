# Evalyn

Diff-aware regression evidence for Python and TypeScript repositories.

> **Private alpha — 0.2.0a1 is unreleased.** This repository intentionally has
> no license. It is therefore not open source, external use and redistribution
> are not granted, and public package or Action distribution is blocked until
> the owner selects a license. The benchmark promotion gate is also unmet:
> **2 of 60 required causal cases exist.** Do not use this alpha on production
> repositories.

Evalyn generates a focused test probe from a Git diff, gives the probe to a
framework adapter, and runs the same content-addressed artifact against isolated
base and candidate worktrees. A candidate-only failure must repeat three times
with one normalized signature before Evalyn records a stable behavior change.

That is evidence, not an automatic verdict. A stable change becomes a regression
gate only after semantic assessment or explicit maintainer triage.

## Why this exists

Existing tests protect expectations someone already wrote down. Evalyn explores
the smaller gap between a specific diff and behavior that may not have a test
yet:

~~~text
Git range + optional intent
          |
          v
  bounded diff/context ---> model proposes probe source
                                    |
                           adapter owns path + argv
                                    |
                  +-----------------+-----------------+
                  v                                   v
        detached base worktree             detached candidate worktree
                  |                                   |
                  +------ same probe, 3 runs each ----+
                                      |
                                      v
                    stable evidence + reproduction data
~~~

Evalyn is not a replacement for a normal test suite, a general coding agent, or
a proof that containerized generated code is harmless.

## See it without an API key

The checked-in demo uses a seeded causal fixture: the base passes, the buggy
candidate fails with one stable signature, and the fixed revision passes again.
It does not call a model, use the network, or modify a Git checkout.

~~~bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python demo.py
~~~

Expected result:

~~~text
pytest-discount-rounding
  base       PASS 3/3
  buggy      FAIL 3/3
  fixed      PASS 3/3
  evidence   stable_change
  causal     yes (base pass -> buggy fail -> fixed pass)
~~~

The complete fixture and runner are in
[examples/no-key-demo](examples/no-key-demo/README.md) and
[bench](bench/README.md).

## Alpha workflow

The 0.2 CLI is source-only while the release is blocked:

~~~bash
# Global options belong before the command.
evalyn --repo . doctor

# Compare an explicit Git range and optionally describe intended behavior.
evalyn --repo . check origin/main...HEAD --intent "Preserve integer-cent rounding"

# Inspect evidence saved by a run.
evalyn --repo . show RUN_ID

# Print exact reproduction details for one finding.
# Replay defaults to the image, setup command, resolved runtime,
# setup-environment names, repeat count, and budget saved by the check.
evalyn --repo . reproduce FINDING_ID

# Record a maintainer disposition.
evalyn --repo . triage FINDING_ID --confirm --reason "Breaks the pricing contract"

# Experimental: only likely or confirmed findings are eligible. Evalyn first
# reproduces the saved failure 3+ times with its original manifest settings.
# A runtime override is accepted only when it exactly matches the saved engine.
# The output path must not exist, and the patch is never applied to source.
evalyn --repo . repair FINDING_ID --emit-patch evalyn-fix.patch
~~~

Use <code>--intent</code> as an invariant when you want it to support gating,
for example <code>--intent "Preserve integer-cent rounding"</code>. Free-form
topical overlap is deliberately inconclusive. An explicit opposite change such
as <code>change: allow negative retries instead of rejecting them</code> can
mark a category-grounded repository oracle as intended, but cannot suppress an
unverified model oracle.

A source-only composite Action preview is checked in at
[action.yml](action.yml) for private dogfooding. It preserves the same exit
codes and emits SARIF; see [docs/github-actions.md](docs/github-actions.md).
It is not licensed or published for external reuse.

Use <code>evalyn demo</code> for the no-key fixture and
<code>evalyn clean</code> to remove only Evalyn-owned run artifacts. Run
<code>evalyn --help</code> and each subcommand’s <code>--help</code> before use;
the command surface may still change during the alpha.

## Requirements

- Python 3.10 through 3.14.
- Git.
- Docker or Podman for generated-probe execution.
- A repository detected by the pytest or Vitest adapter.
- An Anthropic or OpenAI-compatible provider for probe generation.

The deterministic demo and corpus validation require no model credential or
container runtime.

## Configuration

Repository configuration is optional and lives in <code>.evalyn.yml</code>.
Commands are argument arrays owned by configuration and adapters; model output
never supplies an executable, shell expression, or working directory. Probe
generation cannot choose its output path; experimental repair can propose only
a file from a deterministic source allowlist, and Evalyn revalidates it.

~~~yaml
version: 1
provider:
  kind: anthropic
  model: claude-sonnet-4-20250514
sandbox:
  runtime: auto
  setup_env: []
adapters:
  - name: pytest
    setup_argv: []
    suite_argv: []
limits:
  max_tests: 12
  repeat: 3
  budget_seconds: 600
~~~

OpenAI-compatible endpoints use <code>kind: openai-compatible</code> with an
explicit <code>base_url</code>. Provider credentials come from environment
variables; do not commit them. See
[docs/configuration.md](docs/configuration.md) for the schema and trust model.

## Evidence semantics

| Evidence | Meaning |
|---|---|
| <code>unchanged</code> | Probe passed all base and candidate repetitions. |
| <code>stable_change</code> | Base passed 3/3; candidate failed 3/3 with one signature. |
| <code>unstable</code> | Outcomes or signatures varied. |
| <code>baseline_failure</code> | The probe did not establish passing base behavior. |
| <code>improvement</code> | Base failed 3/3 and candidate passed 3/3. |
| <code>invalid</code> | Probe validation or execution infrastructure failed. |

Stable differential evidence alone does not prove the candidate is wrong.
Automatic gating requires either a verbatim user-supplied preservation invariant
or an oracle quoted exactly from a category-matched existing test, documentation
file, API schema, or declaration file. Model-claimed provenance, implementation
text, diff text, context headers, and topical intent overlap do not establish
that trust. Intent, oracle grounding, semantic assessment, and human triage stay
separate fields so reports do not launder uncertainty into confidence.

## Safety and privacy

The 0.2 engine includes:

- detached base and candidate worktrees outside the developer checkout;
- symlink-resistant, repository-bound state with atomic writes, bounded JSON,
  ownership manifests, and permanent per-repository lock files;
- structured adapter-owned argument arrays instead of shell strings;
- canonical path and symlink containment checks;
- sanitized host Git reads that disable hooks, filesystem monitors, recursive
  submodules, configured content filters, external diff drivers, and textconv;
- a separate trusted dependency-setup phase with network and a writable
  disposable workspace, followed by generated-probe runs with no network and
  both the container root and workspace mounted read-only;
- bounded captured output and full logs, timeout, process-group termination,
  and container cleanup;
- streaming byte ceilings for host Git metadata and diffs plus bounded source,
  context, overlay, manifest, and replay reads;
- fixed/validated provider destinations, refused compatible-endpoint redirects,
  bounded provider payloads, and one request deadline without nested SDK retries;
- one-time resolution of <code>auto</code> to a concrete Docker or Podman engine,
  with reproduction and repair bound to the relevant saved execution settings;
- patch-only experimental repair gated to likely or maintainer-confirmed
  regressions, with pre-repair failure revalidation, manifest-locked execution
  inputs, protected test/config/hook targets, and exclusive no-overwrite output
  creation.

These controls reduce risk; they do not make hostile generated code safe in an
absolute sense. Dependency installation, compromised images, container-runtime
vulnerabilities, provider retention, and host failure remain material risks.
Read the [threat model](docs/threat-model.md), [security policy](SECURITY.md),
and [privacy notes](docs/privacy.md) before evaluating a private repository.

## Repository status

| Area | 0.2 alpha state |
|---|---|
| Differential domain engine | Implemented with immutable evidence models and three-repeat classification. |
| Git ranges and dirty overlay | Implemented; dirty inclusion is explicit and candidate-only. |
| Worktree and process safety | Implemented with isolated tests; integration remains alpha. |
| Adapters | pytest and Vitest. |
| Providers | Anthropic and OpenAI-compatible endpoints. |
| Reports and CI integration | Alpha command and local workflow surfaces; not published or supported externally. |
| Benchmark | 2 implemented seed cases out of the 60-case public-promotion gate; gate not met. |
| Repair | Experimental; likely/confirmed only, original failure and settings revalidated, protected non-source targets, patch output only, never overwrite. |
| Public release | **Blocked: no license and only 2/60 benchmark cases.** |

See [ROADMAP.md](ROADMAP.md) for scope and [CHANGELOG.md](CHANGELOG.md) for
unreleased changes.

## Development

~~~bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
ruff check src tests bench examples demo.py
pytest --cov=evalyn
python -m bench.runner --validate
# With vitest@2.1.9 already available on PATH:
python -m bench.runner --execute --strict-tools
python -m build
~~~

Architecture and extension boundaries are documented in
[docs/architecture.md](docs/architecture.md) and
[docs/extensions.md](docs/extensions.md). Contribution rules are in
[CONTRIBUTING.md](CONTRIBUTING.md), but external contributions cannot be
accepted until licensing is resolved.

## License

No license has been selected. All rights are reserved by default. Public launch,
PyPI publication, reusable Action distribution, and outside contributions remain
blocked. Independently, the benchmark promotion threshold remains 2/60.
