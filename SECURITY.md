# Security policy

Evalyn executes generated test code, so its isolation boundary is part of the
product—not an optional hardening layer.

## Supported versions

Only the latest `0.2` alpha receives security fixes. No version is currently
published or supported for production use.

## Reporting a vulnerability

Do not open a public issue for credential exposure, sandbox escape, arbitrary
host execution, path traversal, destructive Git behavior, or dependency-chain
vulnerabilities. Once the repository is public, use GitHub private vulnerability
reporting. Until then, report privately to the repository owner.

Include the Evalyn version, operating system, container runtime, reproduction,
and whether the original checkout or credentials were affected. Do not include
real secrets.

## Security boundaries

- Model prompts receive repository content and are sent to the configured model
  provider. Prompt and response bodies are not persisted by default.
- Anthropic credentials are sent only to the fixed official API endpoint.
  OpenAI-compatible endpoints require validated HTTP(S) URLs, refuse redirects,
  and bound request and response bodies so credentials cannot be forwarded by a
  redirect.
- Generated probes execute only in Docker or Podman containers, without network
  access or inherited developer credentials. Both the container root and the
  prepared disposable workspace are read-only during probe execution.
- Dependency setup is a separate trusted, network-enabled phase with a writable
  disposable workspace. Only environment names explicitly allowlisted by the
  user may be passed to it.
- Host-side Git inspection neutralizes hooks, filesystem monitors, recursive
  submodules, configured filters, external diffs, and textconv. Managed run
  state rejects linked or identity-mismatched paths, uses atomic bounded JSON,
  and bounds persisted logs.
- Real checks record the concrete Docker or Podman engine selected for the run.
  Reproduction and repair replay the saved execution configuration and reject a
  different runtime.
- A generated oracle cannot gate on its provenance claim alone. It must be a
  verbatim user-supplied preservation invariant or appear verbatim in a
  category-matched existing test, documentation file, API schema, or
  declaration file. Implementation/diff/header text and topical overlap do not
  count as grounding.
- Repair is experimental, accepts only likely or maintainer-confirmed findings,
  and emits a reviewable patch to a new path; it never applies changes to the
  original checkout or overwrites an existing artifact.
- A container boundary materially reduces risk but is not a proof of complete
  isolation. Treat generated code as hostile.
