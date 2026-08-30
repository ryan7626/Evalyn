# Architecture

Evalyn is an evidence-first differential test engine. A model helps propose test
source, while application-owned code derives locations and provenance, validates
artifacts, executes both revisions, classifies results, and writes reports.

```text
Git range + optional operator intent
                 │
                 ▼
       detached base/candidate worktrees
                 │
                 ▼
 deterministic change graph
 targets • both-side definitions • ranked context • oracle IDs
                 │
                 ▼
 graph-bound hypothesis portfolio (maximum 8 today)
                 │
        one source candidate per hypothesis
                 │
     adapter validation + base collect/import preflight
                 │
       one bounded correction for that probe only
                 │
                 ▼
 immutable adapter-owned artifacts
                 │
       base/candidate discovery pair
                 │
 candidate-only failure ──► interleaved confirmation runs
                 │
                 ▼
 structured pytest/Vitest outcomes + environment identity
                 │
                 ▼
 classification ─► evidence vector ─► semantic gate ─► run verdict
                 │
                 ▼
 reports • replay • bundle • confirmed-test promotion
```

## Analysis and portfolio generation

The change graph is constructed without model-authored metadata. It records
changed files, additions/deletions/renames, diff line ranges, and detected
Python or JavaScript/TypeScript declarations. Context is selected from tracked
base/candidate inventories plus explicitly included dirty changed content,
under file/count/character bounds, and ranked using changed-path,
symbol-reference, import/call, path-proximity, contract, and manifest
relationships. Every context item records its path, side
(<code>base</code>, <code>candidate</code>, or <code>both</code>), category,
content hash, target IDs, and ranking reasons.

Repository oracle identifiers come only from path-classified existing tests,
API schemas, type declarations, and documentation that is visible on the base
side. Content equal on both sides is marked <code>both</code>; changed or
candidate-only content is not upgraded to an unchanged contract. Operator intent
is represented separately as an operator-sourced oracle. The graph and portfolio
audit trail are persisted under the run's <code>analysis/</code> directory.

Generation is a two-stage protocol. The first response plans bounded hypotheses
using only trusted target/oracle IDs. Unknown IDs, unrelated oracle bindings,
duplicate hypotheses, and malformed individual entries are discarded without
discarding valid siblings. The engine then requests one source-only candidate
for each accepted hypothesis. Model-authored paths, commands, environment,
revision names, and provenance are never accepted.

Every candidate passes adapter validation. The normal container executor also
materializes it on the prepared base revision and runs framework collection or
import without executing the test body. A rejection receives at most one
correction request containing a bounded source and stable sanitized feedback;
failure of one hypothesis does not poison other portfolio members. If no probe
survives, the check is incomplete.

## Execution and environment identity

Dependency setup is an explicit trusted phase. It can use network access and a
writable disposable worktree, with only configured package-registry environment
names forwarded. Generated-probe collection and execution use no network, a
read-only container root, a read-only workspace mount, a non-root user where
supported, and fixed resource limits.

Before meaningful dependency setup, Evalyn writes a random bounded attestation
inside each disposable workspace and requires the container to read and hash
those exact bytes. This detects a remote daemon interpreting a host bind path as
some unrelated remote filesystem. It then performs bounded image inspection
through the selected endpoint. Only one canonical engine-reported SHA-256 image
ID is accepted and pinned for setup/probes. The run also records a secret-free
environment fingerprint covering the engine label,
local/remote endpoint scope (never its URI), declared/resolved image,
network/read-only/pull policies, and security limits. Failed, timed-out,
truncated, or malformed warmup/inspection leaves identity incomplete and
prevents environment-verified check evidence.

Docker/Podman client configuration is sanitized. Local
<code>unix://</code>/<code>npipe://</code> endpoints are accepted by default;
<code>tcp://</code> and <code>ssh://</code> endpoints require the invocation-wide
<code>--allow-remote-container</code> operator flag. That choice is saved;
reproduction and repair must match both the saved endpoint scope and resolved
image identity before meaningful setup.

Both revisions receive the exact same content-addressed adapter artifact. Pytest
uses an Evalyn-owned bounded result reporter; Vitest uses its JSON reporter.
Normalized counts, collection errors, and failed-test details drive status and
failure signatures. Missing or contradictory machine-readable results are
invalid/infrastructure evidence rather than inferred from console wording.

Execution is adaptive by default. Each probe starts with one base and one
candidate discovery run. A base-pass/candidate-fail-or-timeout transition is
confirmed on both sides up to the configured repetition count, interleaving the
sides to reduce temporal drift. Pass/pass, baseline failure, and discovery
improvement normally stop after the discovery pair. This means
<code>unchanged</code> is “no differential observed by the discovery pair,” not
“exhaustively unchanged.”

## Evidence, gates, and verdicts

Classification is observation-only:
<code>unchanged</code>, <code>stable_change</code>, <code>unstable</code>,
<code>baseline_failure</code>, <code>improvement</code>, or <code>invalid</code>.
A stable finding carries separate evidence dimensions for causal transition,
base/candidate repetition counts, repeatability, oracle kind/source, static
target and oracle binding, target-revert intervention, corroboration count, and
environment integrity. Evalyn does not replace these fields with a
model-authored confidence score.

Human triage has explicit precedence: confirmed findings gate; dismissed or
uncertain findings do not. Unreviewed automatic gating is deliberately narrower:
it requires environment-verified three-repeat causal evidence, an exact
single-assertion transplant from an unchanged existing test, static executable
binding to the graph target, and three additional passing runs after that target
file alone is reverted to its base bytes in the disposable candidate. Prose,
schemas, types, operator invariants, changed/candidate-authored contracts,
model-only expectations, and unsupported add/delete interventions remain useful
advisory evidence until a maintainer decides. This keeps “stable transition”
separate from “likely regression.”

The aggregate verdict preserves that distinction, with incomplete evidence
taking precedence over otherwise interesting observations:

- <code>incomplete</code>: invalid probe or infrastructure evidence is present;
- <code>regression_found</code>: absent incomplete evidence, at least one finding
  passes the gate;
- <code>change_observed</code>: absent incomplete/gating evidence, a stable
  differential exists but remains advisory;
- <code>no_differential_observed</code>: all executed probes were unchanged at
  discovery, without claiming absence of bugs;
- <code>insufficient_evidence</code>: results such as baseline failure,
  instability, or improvement do not support the other verdicts.

## Stored operations

Run state is external to the target repository and repository/run-identity
bound. JSON is the lossless record of classifications, semantic state, evidence
vectors, verdict, and environment fingerprints. Markdown presents the same
evidence for maintainers. JUnit maps probe outcomes into test/failure/error/skip
states and carries the run verdict plus environment digests; SARIF carries
finding-level semantic/evidence data plus run and environment properties. They
are integration views, not substitutes for the JSON evidence record.

Reproduction reconstructs the saved revisions/dirty overlay, launches the saved
declared image, verifies that it resolves to the saved immutable image ID and
endpoint scope before setup, and reuses setup argv, concrete runtime,
setup-environment names, repeat count, budget, and remote-endpoint choice. Its explicit verdict is
<code>reproduced</code>, <code>not_reproduced</code>, or
<code>incomplete</code>, mapped by the CLI to exit 0, 1, or 3 respectively.

<code>bundle</code> exports a deterministic ZIP containing an internally
cross-checked probe, oracle, run snapshot, reproduction metadata, and a manifest
of every
other payload's digest. <code>promote</code> copies a content-verified probe only
from a maintainer-confirmed finding into a new adapter-appropriate test path.
Both use exclusive no-overwrite publication.

Repair remains experimental and separate from detection. It accepts only likely
or confirmed findings, revalidates the original stable candidate failure before
generation, and proceeds only when its deterministic allowlist contains exactly
one changed tracked non-test source file. Provider creation defaults to the
saved kind/model; only explicit configuration outside the repository can
override operator provider policy. It validates the probe and
saved/default suite, then repeats those checks in a freshly reconstructed
candidate while detecting tracked/untracked validation mutations. It emits a
new patch file and does not apply it to the developer checkout.

## Trust and extension boundaries

- The selected configuration is trusted executable input because it controls
  images, setup argv, suite argv, and other runtime choices. A
  repository-resident file must be byte-identical across base, candidate, and
  selected working-tree state. Intentional edits require an operator-owned
  external <code>--config</code>; only that explicit external boundary may set
  <code>provider.base_url</code> or a non-empty
  <code>sandbox.setup_env</code> credential-name list.
- Credentials come from the existing process environment or an explicit
  non-symlink <code>--env-file</code> outside the repository. Repository
  <code>.env</code> files are not loaded implicitly.
- Repository text, diffs, and generated probes are untrusted data. Provider
  responses cannot select host/container control values.
- Framework adapters detect repositories, validate source, select a safe probe
  path, construct argument arrays, and normalize outcomes. Model providers
  implement structured completion.
- Execution backends are not a plugin boundary in this alpha: only Docker and
  Podman are supported.
