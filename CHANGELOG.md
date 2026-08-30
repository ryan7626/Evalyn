# Changelog

All notable changes will be recorded here. Evalyn has no public release yet.

## 0.2.0a1 — Unreleased

- Pivoted from overnight autonomous repair to diff-aware regression detection.
- Added disposable base/candidate workspaces and container-only probe execution.
- Added deterministic Python/TypeScript change graphs with changed symbols,
  side-aware ranked context, content hashes, and base-visible repository oracle
  identifiers.
- Replaced monolithic probe generation in the check workflow with graph-bound
  hypothesis portfolios, independently salvageable candidates, one bounded
  correction per rejected probe, sanitized feedback, and collection/import
  preflight on the prepared base revision.
- Added pytest and Vitest adapters with machine-readable normalized outcomes.
  Checks now use a base/candidate discovery pair and confirm only interesting
  candidate-only failures by default, interleaving both sides.
- Added evidence vectors and explicit run verdicts that distinguish gating
  <code>regression_found</code>, advisory <code>change_observed</code>,
  <code>no_differential_observed</code>, insufficient evidence, and incomplete
  execution.
- Added reproducible findings, Markdown/JSON/JUnit/SARIF artifacts, a no-key
  demo, and repository-local GitHub Actions.
- Added two deterministic benchmark smoke cases and a rescorable detector
  protocol. There are currently 0/60 registered scoring cases, so the promotion
  gate is not met; smoke fixtures do not count as detector accuracy.
- Made repair experimental and patch-only, gated it to likely or confirmed
  regressions, added exact pre-repair failure revalidation, locked repair
  execution to the saved manifest, excluded test/config/hook targets, and added
  exclusive no-overwrite output creation. Final verification now applies the
  patch to another freshly reconstructed candidate, repeats the saved probe,
  runs the saved/default suite, and rejects tracked or unexpected untracked
  validation mutations.
- Added deterministic integrity-manifested finding bundles and exclusive
  promotion of content-verified probes to new adapter-appropriate test paths
  only after maintainer confirmation.
- Bound saved evidence to its repository and run identity before reproduction
  or repair.
- Made the generated-probe workspace read-only after the separate trusted,
  writable dependency-setup phase.
- Sanitized host Git inspection to disable hooks, filesystem monitors,
  recursive submodules, configured content filters, external diffs, and
  textconv.
- Resolved automatic runtime selection once per real check, persisted the
  concrete Docker or Podman engine, and made reproduction and repair replay the
  saved execution snapshot while rejecting runtime mismatches.
- Required random workspace-mount attestation plus bounded inspection to obtain
  a canonical engine SHA-256 image ID before meaningful setup. Setup and probes
  pin that ID; replay launches the saved declaration and verifies the same ID
  and endpoint scope before setup. Reports/manifests include a secret-free OCI
  policy fingerprint, never the endpoint URI.
- Rejected remote TCP/SSH container endpoints by default, added explicit global
  <code>--allow-remote-container</code> opt-in, and required reproduction/repair
  to match the saved remote-endpoint choice.
- Stopped implicitly loading repository <code>.env</code> files. Explicit
  <code>--env-file</code> inputs must be non-symlinks outside the repository and
  cannot override the process environment.
- Required repository-resident configuration to remain byte-identical across
  base, candidate, and working-tree state. Candidate config edits fail closed;
  custom <code>provider.base_url</code> and non-empty
  <code>sandbox.setup_env</code> now require explicit external configuration.
- Hardened run locks, state reads/writes, and cleanup against symlink,
  hard-link, identity-swap, and oversized-artifact attacks; full logs now have a
  hard size bound and refuse symlink targets.
- Required automatic regression gating to use environment-verified three-repeat
  evidence, an exact assertion from an unchanged existing test, correlated
  executable target/oracle binding, and a further three passing runs after only
  the claimed target file is reverted to base bytes. Prose, schemas, types,
  operator invariants, changed/candidate-only contracts, model self-attestation,
  and unsupported interventions remain advisory until maintainer triage.
- Disabled Git replacement objects and compare repository configuration as
  bounded exact blobs/working bytes, catching ignored, untracked, and
  skip-worktree policy drift.
- Added hard streaming ceilings for host Git output and bounded reads for
  context, overlays, saved state, repair inputs, reports, and logs.
- Disabled pytest and Vitest result caches so generated probes can run with the
  disposable workspace mounted read-only.
- Pinned Anthropic credentials to the official endpoint, refused redirects from
  OpenAI-compatible endpoints, bounded provider bodies, and removed nested SDK
  retries.

Publication is blocked until the repository owner selects a license.
