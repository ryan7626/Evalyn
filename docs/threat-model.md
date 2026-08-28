# Threat model

Evalyn treats model-authored tests as hostile code and repository contents as
potentially sensitive.

## Protected assets

- The developer’s checkout, index, branches, commits, and untracked work.
- API keys, Git credentials, SSH agents, cloud credentials, and local services.
- Source outside the selected repository and context allowlist.
- CI integrity and the distinction between a finding and an incomplete run.

## Principal threats

- A generated probe attempts command injection, network exfiltration, fork
  bombs, resource exhaustion, filesystem traversal, or credential discovery.
- A malicious repository uses symlinks, submodules, configuration, or test hooks
  to escape its workspace.
- Cancellation or crashes leave process trees, services, Git worktrees, or
  misleading successful state behind.
- A generated assertion encodes an intended change or invented expectation and
  is reported as a regression.
- Pull requests from forks attempt to obtain a model-provider secret.

## Controls

- Structured adapter-owned commands; no shell execution.
- Non-root generated-probe execution with resource limits, a minimal
  environment, no network, a read-only container root filesystem, and a
  read-only disposable-worktree mount. Trusted dependency setup is a separate,
  explicitly writable and network-enabled phase.
- Host Git inspection uses a minimal environment and neutralizes hooks,
  filesystem monitors, recursive submodules, configured clean/smudge/process
  filters, external diff drivers, and textconv.
- Canonical path containment and symlink checks before every controlled read or
  write.
- Detached base/candidate worktrees and exact-path cleanup guarded by ownership
  manifests and a per-repository lock. Managed state is opened through anchored
  directory descriptors without following symlinks; JSON is size-bounded and
  written atomically, and cleanup skips active or unfinished runs.
- Captured process output and persisted full logs are bounded. Final log paths
  reject symlinks rather than following them.
- Schema, repository, run-identity, size, and symlink checks on saved evidence
  before reproduction or repair.
- Repeated differential execution and stable failure signatures. Automatic
  semantic gating additionally requires a verbatim user-supplied preservation
  invariant or an oracle found verbatim in a category-matched existing test,
  documentation file, API schema, or declaration file. Model provenance,
  implementation/diff/header text, and topical intent overlap are not trusted.
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
- Real checks resolve <code>auto</code> once and save the concrete Docker or
  Podman engine. Reproduction and repair load the relevant image reference,
  setup argv, engine, setup environment names, repetition count, and budget
  from the saved run; repair also reuses the saved suite argv. Both refuse a
  runtime mismatch. Test harness, package/test configuration, setup, and
  CI-hook paths are excluded from repair. Patch creation is exclusive and never
  overwrites a path.
- No `pull_request_target` workflow and no provider secrets for forked PRs.

## Residual risks

Containers are not virtual machines. Container-runtime vulnerabilities,
malicious dependency installers, compromised base images, provider retention,
mutable image tags, and repository-specific build hooks remain possible.
Private dependency setup may receive explicitly allowlisted package-registry
credentials through a short-lived owner-only env file, so it is a separate
trusted phase. Output logs are bounded but are not content-redacted, so trusted
installers must not print those values. SIGKILL and host failure may require
ownership-manifest recovery.
