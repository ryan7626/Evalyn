# Roadmap

## 0.2 private alpha — implemented, unreleased

- Safe differential execution for Git refs and explicitly included dirty edits.
- Deterministic Python/TypeScript change targets, ranked side-aware context, and
  base-visible repository oracle identifiers.
- Graph-bound portfolio planning (currently at most eight probes), independent
  candidate salvage, one bounded correction per rejected probe, and collection
  or import preflight on the prepared base revision.
- Pytest and Vitest adapters with machine-readable normalized outcomes, adaptive
  discovery, and interleaved confirmation of candidate-only failures.
- Separate evidence vectors, semantic/triage state, and run verdicts, including
  advisory <code>change_observed</code> versus gating
  <code>regression_found</code>.
- Conservative automatic gating only for exact unchanged-test assertions with
  verified target/oracle binding and a three-run target-revert intervention;
  other contracts and operator invariants remain maintainer evidence.
- Byte-exact trusted configuration/env boundaries, workspace-mount attestation,
  pre-setup engine resolution to immutable image IDs, OCI policy fingerprints
  with non-secret endpoint scope, and explicit opt-in for remote endpoints.
- Reproducible findings; Markdown, JSON, JUnit, and SARIF output; deterministic
  portable bundles; confirmed-probe promotion; and a source-only local Action
  preview.
- Experimental patch-only repair with saved-failure revalidation and fresh-
  candidate probe/suite verification.

The checked-in benchmark has two deterministic execution smoke cases and zero
registered detector-scoring cases. Those smoke cases do not count toward the
60-case promotion gate.

## Before release promotion

- Register and review 30 positive and 30 negative end-to-end detector cases,
  with all required trials/fingerprints and the configured validity, discovery,
  false-gate, precision, reproduction, and latency thresholds met.
- Continue dogfooding portfolio validity, context relevance, automatic-gate
  precision, clean-environment reproduction, and incomplete-run behavior.
- Select a license and complete the separate packaging/Action publication
  decision. Until then there is no supported external distribution.
- Keep repair explicitly experimental until detector and repair-specific
  precision/safety evidence justifies a narrower promotion decision.

## After the alpha

- Go and Rust adapter contracts.
- Additional local and hosted model-provider integrations.
- Probe minimization, richer language-aware dependency edges, and stronger
  deterministic contract assessors.
- Registry-verifiable image identities and dependency-environment locking beyond
  the current engine-local image ID and OCI policy fingerprint.
- Reviewed integrations for evidence bundles and patch export to external coding
  systems.

Scheduling, in-place autonomous repair, and broad browser orchestration are not
planned until the core detector meets its precision and safety gates.
