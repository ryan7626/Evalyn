# Threat model

Evalyn treats model-authored tests as hostile code and repository contents as
potentially sensitive.

## Protected assets

- The developer’s checkout, index, branches, commits, and untracked work.
- API keys, Git credentials, SSH agents, cloud credentials, and local services.
- Source outside the selected repository and selected context boundary.
- CI integrity and the distinction between a finding and an incomplete run.

## Principal threats

- A generated probe attempts command injection, network exfiltration, fork
  bombs, resource exhaustion, filesystem traversal, or credential discovery.
- A malicious repository uses symlinks, submodules, configuration, or test hooks
  to escape its workspace.
- Repository text attempts prompt injection or asks the model to invent a trusted
  target, contract source, command, or provenance category.
- Candidate-controlled configuration chooses a hostile image, setup command, or
  provider endpoint; ambient daemon settings choose a remote endpoint; or a
  repository <code>.env</code> attempts to replace operator credentials.
- Cancellation or crashes leave process trees, services, Git worktrees, or
  misleading successful state behind.
- A generated assertion encodes an intended change or invented expectation and
  is reported as a regression.
- Human-oriented framework output is missing, malformed, or manipulated in a way
  that could produce an unstable failure signature or a false clean result.
- Pull requests from forks attempt to obtain a model-provider secret.

## Controls

- Structured adapter-owned commands; no shell execution.
- Strict, size-bounded configuration with no-follow final paths. Configuration
  is explicitly treated as trusted executable input. A repository-resident file
  must be unchanged across base, candidate, and working-tree state; otherwise
  the check fails closed and requires an operator-owned external
  <code>--config</code>. Only an explicit external config may set
  <code>provider.base_url</code> or non-empty
  <code>sandbox.setup_env</code>. Repository <code>.env</code> files are not
  loaded implicitly; an explicit
  <code>--env-file</code> must be a non-symlink outside the target repository and
  cannot override existing process values.
- Non-root generated-probe execution with resource limits, a minimal
  environment, no network, a read-only container root filesystem, and a
  read-only disposable-worktree mount. Trusted dependency setup is a separate,
  explicitly writable and network-enabled phase.
- A deterministic side-aware change graph derives targets, source categories,
  hashes, ranked context, and base-visible oracle IDs. Hypothesis plans can use
  only those IDs. Invalid model entries are isolated; each generated candidate
  gets adapter validation, base collection/import preflight, and at most one
  bounded correction using sanitized feedback.
- Adaptive execution starts with one run per side and confirms only an observed
  base-pass/candidate-fail-or-timeout transition. Confirmation sides are
  interleaved, and pytest/Vitest machine-readable results—not terminal phrasing—
  supply counts, collection errors, and normalized failure signatures. Missing
  or contradictory structured data is incomplete/invalid evidence.
- Host Git inspection uses a minimal environment and neutralizes hooks,
  filesystem monitors, recursive submodules, configured clean/smudge/process
  filters, external diff drivers, and textconv.
- Canonical path containment and no-follow/symlink checks on managed reads and
  writes.
- Detached base/candidate worktrees and exact-path cleanup guarded by ownership
  manifests and a per-repository lock. Managed state is opened through anchored
  directory descriptors without following symlinks; JSON is size-bounded and
  written atomically, and cleanup skips active or unfinished runs.
- Captured process output and persisted full logs are bounded. Final log paths
  reject symlinks rather than following them.
- Schema, repository, run-identity, size, and symlink checks on saved evidence
  before reproduction or repair.
- Repeated differential execution, stable failure signatures, graph targets,
  static target/oracle binding, target-revert intervention, and
  environment-integrity evidence. Automatic semantic gating requires an exact
  single assertion from an unchanged existing test and three passing runs after
  reverting only the claimed target file to its base bytes. Prose, schemas,
  types, operator invariants, changed/candidate-only contracts, model
  provenance, metamorphic opinion, and unsupported interventions remain
  advisory. Maintainer confirmation, dismissal, and uncertainty are stored
  separately from automatic assessment.
- Reports distinguish a gating <code>regression_found</code> from advisory
  <code>change_observed</code>, and incomplete evidence takes precedence in the
  aggregate verdict. <code>no_differential_observed</code> is a bounded
  observation rather than a regression-free claim.
- Host Git output is drained with hard byte ceilings; diff rendering may return
  an explicit bounded truncation marker, while oversized identity or metadata
  output fails closed. Context, overlays, manifests, and replay inputs are
  bounded before decoding or copying.
- Provider bodies are bounded and requests have a total deadline without nested
  SDK retries. Anthropic uses its fixed official endpoint; compatible endpoints
  are validated and redirects are rejected before a credential can cross an
  origin boundary.
- Repair eligibility requires a likely regression or explicit maintainer
  confirmation. The untouched saved candidate must reproduce the exact stable
  status/signature at least three times before any model request, and repaired
  validation also runs at least three times.
- Real checks resolve <code>auto</code> once, require a random workspace-mount
  attestation, and perform bounded engine inspection before meaningful setup.
  Only a canonical SHA-256 image ID produces a complete secret-free environment
  fingerprint; setup, probes, and replay use the resolved ID. The fingerprint
  stores only local/remote endpoint scope, not the URI. Local socket endpoints
  are the default. Remote TCP/SSH daemon endpoints require
  <code>--allow-remote-container</code>, and saved reproduction/repair must
  repeat the same opt-in and match the saved endpoint scope.
- Reproduction and repair launch the saved declared image, verify its resolved
  ID against the saved environment, and load setup argv, engine, setup
  environment names, repetition count, budget, and endpoint choice from the
  saved run; repair also reuses the saved suite argv. Both refuse a runtime or
  identity mismatch. Test harness, package/test configuration, setup, and CI-hook paths
  are excluded from repair. Final repair validation applies the patch to a
  newly reconstructed candidate, reruns the probe and suite, and rejects
  tracked/untracked validation mutations. Patch creation is exclusive and
  never overwrites a path.
- Evidence bundles rebuild adapter metadata, cross-check the run-owned source
  and reproduction copy, and manifest every ZIP payload. Promotion applies the
  same checks, requires maintainer confirmation, stays within the repository,
  and creates a new non-symlink test path without overwrite.
- No `pull_request_target` workflow and no provider secrets for forked PRs.

## Residual risks

Containers are not virtual machines. Container-runtime vulnerabilities,
malicious dependency installers, compromised base images, provider retention,
and repository-specific build hooks remain possible. An engine-resolved image
ID is local daemon evidence, not registry provenance; a digest-pinned declared
reference remains stronger supply-chain input than a mutable tag. Explicit
remote daemon opt-in moves worktrees and secrets into a larger operator-managed
trust boundary and does not authenticate or encrypt a plain TCP endpoint by
itself.
Private dependency setup may receive explicitly allowlisted package-registry
credentials through a short-lived owner-only env file, so it is a separate
trusted phase. Output logs are bounded but are not content-redacted, so trusted
installers must not print those values. Base collection imports repository code
inside the restricted container and therefore still relies on the container
boundary. Existing tests, schemas, declarations, and documentation can be wrong;
conservative binding/intervention gates reduce but cannot eliminate semantic
false positives. Reverting a target file can also revert neighboring hunks, so
the intervention establishes file-level causality rather than proving that the
named symbol alone caused the differential. Bundle digests and duplicate checks provide integrity and
corruption detection, not signer authenticity against an actor able to rewrite
all local run state. Promoted probes and portable bundles still require human
review before use in another trust domain. SIGKILL and host failure may require
ownership-manifest recovery.
