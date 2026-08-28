# Changelog

All notable changes will be recorded here. Evalyn has no public release yet.

## 0.2.0a1 — Unreleased

- Pivoted from overnight autonomous repair to diff-aware regression detection.
- Added disposable base/candidate workspaces and container-only probe execution.
- Added pytest and Vitest adapters, evidence-based findings, reproducible
  artifacts, a no-key demo, and repository-local GitHub Actions.
- Added two causal benchmark seed cases. The 60-case public-promotion gate is
  not met.
- Made repair experimental and patch-only, gated it to likely or confirmed
  regressions, added exact pre-repair failure revalidation, locked repair
  execution to the saved manifest, excluded test/config/hook targets, and added
  exclusive no-overwrite output creation.
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
- Hardened run locks, state reads/writes, and cleanup against symlink,
  hard-link, identity-swap, and oversized-artifact attacks; full logs now have a
  hard size bound and refuse symlink targets.
- Required automatic regression gating to use a verbatim user preservation
  invariant or a category-matched repository oracle. Model self-attestation,
  implementation/diff/header text, and topical intent overlap remain uncertain.
- Added hard streaming ceilings for host Git output and bounded reads for
  context, overlays, saved state, repair inputs, reports, and logs.
- Disabled pytest and Vitest result caches so generated probes can run with the
  disposable workspace mounted read-only.
- Pinned Anthropic credentials to the official endpoint, refused redirects from
  OpenAI-compatible endpoints, bounded provider bodies, and removed nested SDK
  retries.

Publication is blocked until the repository owner selects a license.
