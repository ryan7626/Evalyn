# Evalyn

Diff-aware regression evidence for Python and TypeScript repositories.

> **Private alpha — 0.2.0a1 is unreleased.** This repository intentionally has
> no license. It is therefore not open source, external use and redistribution
> are not granted, and public package or Action distribution is blocked until
> the owner selects a license. The benchmark promotion gate is also unmet:
> **0 of 60 detector-scoring cases are registered.** The two checked-in causal
> fixtures are execution smoke tests and do not count toward detector accuracy.
> Do not use this alpha on production repositories.

Evalyn builds a deterministic change graph from a Git diff, including changed
Python/TypeScript symbols, side-aware ranked context, and repository contract
excerpts. A model plans a small portfolio of graph-bound hypotheses and generates
each probe independently. Evalyn then gives each accepted probe to a framework
adapter and runs the same content-addressed artifact against isolated base and
candidate worktrees.

Normal checks use adaptive execution: one base/candidate discovery pair for
every probe, followed by interleaved confirmation runs only for a candidate-only
failure (three total per side by default). A confirmed transition needs passing
base runs and candidate failures with one normalized signature before Evalyn
records a stable behavior change.

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
 deterministic change graph
  targets + ranked context
  + verified oracle IDs
          |
          v
 model plans hypotheses ---> one independently generated probe each
                                      |
                       adapter validation + base collection
                       (one bounded correction if rejected)
                                      |
                  +-------------------+-------------------+
                  v                                       v
        detached base worktree                 detached candidate worktree
                  |                                       |
                  +-- discovery pair; confirm failures ---+
                                      |
                                      v
                 structured evidence + verdict + reproduction data
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

# Re-run one saved finding. Exit 0 means reproduced, 1 means not reproduced,
# and 3 means incomplete. The command also prints that explicit verdict.
evalyn --repo . reproduce FINDING_ID

# Record a maintainer disposition.
evalyn --repo . triage FINDING_ID --confirm --reason "Breaks the pricing contract"

# Export deterministic, integrity-manifested evidence without overwriting.
evalyn --repo . bundle FINDING_ID --output artifacts/FINDING_ID.zip

# After confirmation, copy the content-verified probe into a new normal test.
evalyn --repo . promote FINDING_ID --output tests/test_evalyn_regression.py

# Experimental: only likely or confirmed findings are eligible. Evalyn first
# reproduces the saved failure 3+ times with its original manifest settings.
# The saved provider kind/model is used unless an external reviewed --config is
# supplied; a current worktree config cannot silently change repair behavior.
# A runtime override is accepted only when it exactly matches the saved engine.
# It validates the patch again in a fresh candidate with the probe and suite.
# The output path must not exist, and the patch is never applied to source.
evalyn --repo . repair FINDING_ID --emit-patch evalyn-fix.patch
~~~

Reproduction launches the saved declared image and requires it to resolve to the
saved immutable identity and endpoint scope before setup. It also reuses the
saved setup command, concrete runtime, setup-environment names, repeat count,
budget, and remote-endpoint choice. Selecting only <code>--side base</code> or
<code>--side candidate</code> tests that side's expected half of the transition.
Invalid/infrastructure results are <code>incomplete</code> (exit 3); conclusive
results that no longer match are <code>not_reproduced</code> (exit 1).

Remote <code>tcp://</code> and <code>ssh://</code> container endpoints are
rejected unless the operator supplies the global
<code>--allow-remote-container</code> flag before the subcommand. Reproduction
and repair require that opt-in to match the saved run; the flag expands the
trust boundary and does not make a remote daemon trustworthy.

Use <code>--intent</code> to record an operator invariant for generation and
maintainer review, for example <code>--intent "Preserve integer-cent rounding"</code>.
It does not by itself authorize an automatic gate. Free-form topical overlap is
deliberately inconclusive. An explicit opposite change can mark a grounded
repository oracle as intended, but cannot suppress an unverified model oracle.

A source-only composite Action preview is checked in at
[action.yml](action.yml) for private dogfooding. It preserves the same exit
codes and exposes verified run/report/SARIF metadata; see
[docs/github-actions.md](docs/github-actions.md).
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
It is executable trust input: images, dependency-setup argument arrays, suite
argument arrays, and provider settings must be reviewed. A repository-resident
config is accepted only when unchanged across base, candidate, and selected
working-tree state; a candidate config edit fails closed. Intentional config
changes require a reviewed external file selected with the global
<code>--config /path/to/trusted.yml</code> option. Non-empty
<code>provider.base_url</code> and <code>sandbox.setup_env</code> are available
only through that explicit external boundary. Model output never supplies an
executable, shell expression, working directory, container image, or probe path.

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

OpenAI-compatible endpoints use <code>kind: openai-compatible</code> with either
<code>OPENAI_BASE_URL</code> from the operator environment or an explicit
external configuration containing <code>base_url</code>. Provider credentials
come from environment variables; do not commit them. Evalyn does not implicitly
load a repository <code>.env</code>. The optional global
<code>--env-file</code> must name a non-symlink file outside the target
repository and does not override variables already present in the process
environment. See
[docs/configuration.md](docs/configuration.md) for the schema and trust model.

A real container check first verifies that the container can read a random
content-addressed workspace attestation, then performs bounded image inspection.
Evalyn records the engine-reported immutable image ID and a
fingerprint of the relevant OCI policy, including only whether the selected
endpoint is local or remote—not its URI. Meaningful dependency setup,
generated-probe execution, and saved replay then use that image ID. If identity
cannot be established, the check is incomplete rather than silently hashing a
mutable tag.

## Evidence semantics

| Evidence | Meaning |
|---|---|
| <code>unchanged</code> | The base and candidate both passed the discovery execution; no differential was observed. |
| <code>stable_change</code> | Base passed and candidate failed/timeout with one signature for the configured confirmation count (3/3 by default). |
| <code>unstable</code> | Confirmation outcomes or failure signatures did not stay stable. |
| <code>baseline_failure</code> | Discovery did not establish passing base behavior. |
| <code>improvement</code> | Base failed/timeout and candidate passed during discovery. |
| <code>invalid</code> | Probe validation, structured-result parsing, or execution infrastructure was invalid/incomplete. |

Pytest emits an Evalyn-owned bounded result record and Vitest runs with its JSON
reporter. Counts, collection errors, and normalized failing-test details drive
classification and failure signatures; human console text remains diagnostic.
A missing or malformed structured result is infrastructure evidence, not a test
failure.

Reports also carry a run verdict:

| Verdict | Meaning |
|---|---|
| <code>incomplete</code> | At least one probe or infrastructure path was invalid/incomplete; this takes precedence over other observations. |
| <code>regression_found</code> | With no incomplete evidence, at least one stable finding passed the automatic or maintainer gate. |
| <code>change_observed</code> | With no incomplete/gating finding, a stable differential was observed but remains advisory. |
| <code>no_differential_observed</code> | Every executed probe passed its base and candidate discovery pair. This is not proof of absence. |
| <code>insufficient_evidence</code> | No invalid path occurred, but observations such as baseline failure, instability, or improvement do not support a clean or stable-change verdict. |

For <code>check</code>, incomplete or insufficient evidence exits 3. A gating
regression exits 1 under the default <code>--fail-on likely</code>; advisory
<code>change_observed</code> and <code>no_differential_observed</code> exit 0.
<code>--fail-on never</code> leaves findings in the report but does not return 1.
CLI usage and settings-schema errors use exit 2; inspection or execution that
cannot complete uses exit 3.

Stable differential evidence alone does not prove the candidate is wrong.
Maintainer confirmation gates; dismissal or uncertain triage does not. Automatic
gating additionally requires environment-verified three-repeat evidence, a
verified executable graph-target reference, an exact single assertion copied
from an unchanged existing test, and three passing executions after only the
claimed target file is reverted to its base bytes in the disposable candidate.

Prose, schemas, types, operator invariants, candidate-authored or changed
contracts, model-claimed provenance, metamorphic/model opinion, unsupported
target interventions, and topical intent overlap remain advisory. Intent,
oracle grounding, binding, intervention evidence, semantic assessment, and
human triage stay separate so reports do not launder uncertainty into
confidence.

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
- deterministic, side-aware change targets and oracle identifiers; model output
  cannot invent a trusted location or provenance, and each rejected portfolio
  probe receives at most one correction using bounded sanitized feedback;
- adapter validation plus no-network, read-only base collection/import preflight
  before a generated probe enters differential execution;
- framework-structured pytest/Vitest outcomes, adaptive discovery, and
  interleaved confirmation of candidate-only failures;
- fixed/validated provider destinations, refused compatible-endpoint redirects,
  bounded provider payloads, and one request deadline without nested SDK retries;
- one-time resolution of <code>auto</code> to a concrete Docker or Podman engine,
  workspace-mount attestation, engine image-ID inspection and OCI policy
  fingerprints, local container endpoints by default, and reproduction/repair
  bound to the saved image identity, endpoint scope, and execution settings;
- deterministic integrity-manifested evidence bundles and exclusive promotion
  of content-verified probes only after maintainer confirmation;
- patch-only experimental repair gated to likely or maintainer-confirmed
  regressions, with pre-repair failure revalidation, manifest-locked execution
  inputs, protected test/config/hook targets, fresh-candidate probe and suite
  validation, tracked/untracked mutation checks, and exclusive no-overwrite
  output creation.

These controls reduce risk; they do not make hostile generated code safe in an
absolute sense. Dependency installation, compromised images, container-runtime
vulnerabilities, explicitly opted-in remote daemons, provider retention, and
host failure remain material risks.
Read the [threat model](docs/threat-model.md), [security policy](SECURITY.md),
and [privacy notes](docs/privacy.md) before evaluating a private repository.

## Repository status

| Area | 0.2 alpha state |
|---|---|
| Analysis and generation | Deterministic Python/TypeScript change graph, ranked side-aware context, graph-bound portfolio planning, per-probe salvage, and one bounded correction. |
| Differential domain engine | Immutable evidence models, base collection preflight, structured framework results, adaptive discovery, and confirmation of candidate-only failures. |
| Git ranges and dirty overlay | Implemented; dirty inclusion is explicit and candidate-only. |
| Worktree and process safety | Implemented with isolated tests; integration remains alpha. |
| Adapters | pytest and Vitest. |
| Providers | Anthropic and OpenAI-compatible endpoints. |
| Evidence artifacts | Markdown/JSON/JUnit/SARIF, deterministic bundles, confirmed-probe promotion, and explicit reproduction verdicts. |
| CI integration | Source-only repository-local Action preview; not published or supported externally. |
| Benchmark | 2 deterministic smoke cases and 0/60 registered detector-scoring cases; gate not met. |
| Repair | Experimental; likely/confirmed only, fresh-candidate probe/suite validation, patch output only, never overwrite. |
| Public release | **Blocked: no license and 0/60 scored benchmark cases.** |

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
blocked. Independently, the detector-scoring promotion threshold remains 0/60.
