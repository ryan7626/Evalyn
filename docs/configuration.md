# Configuration

Evalyn 0.2 reads an optional <code>.evalyn.yml</code> from the target repository.
The schema is strict: unknown keys are rejected instead of being silently
ignored.

The selected file is trusted executable input, not ordinary repository context.
It can choose container images, dependency setup argv, suite argv, limits, and
(within the boundary below) provider or package-registry settings. A real check
accepts a repository-resident configuration only when its bytes are unchanged
across the base revision, candidate revision, and selected working-tree state.
A candidate or dirty edit to that file fails closed instead of becoming trusted
check configuration.

When a configuration change is intentional—or when policy requires a runner-
owned trust root—select an operator-reviewed file outside the repository:

~~~bash
evalyn --repo /path/to/repo \
  --config /path/to/operator-owned/evalyn.yml \
  check BASE...CANDIDATE
~~~

The selected configuration file cannot itself be a symlink, and reads are
size-bounded. An explicit path may live outside the target repository, but the
operator remains responsible for its ownership and permissions. Non-empty
<code>provider.base_url</code> and <code>sandbox.setup_env</code> are accepted
only from such an explicit external configuration. The invariant repository
configuration remains useful for adapter/image/setup/suite selection and limits
without granting it a credential-forwarding or custom-provider-destination
boundary.

## Minimal example

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
    image: python:3.12-slim
    setup_argv: []
    suite_argv: []
limits:
  max_tests: 12
  repeat: 3
  budget_seconds: 600
~~~

## Provider

<code>provider.kind</code> accepts:

- <code>anthropic</code>, using <code>ANTHROPIC_API_KEY</code>;
- <code>openai-compatible</code>, using <code>OPENAI_API_KEY</code> and either
  <code>provider.base_url</code> or <code>OPENAI_BASE_URL</code>.

The <code>provider.base_url</code> field is allowed only in an explicit
configuration outside the target repository. <code>OPENAI_BASE_URL</code>, when
used instead, belongs to the operator-controlled process environment or explicit
external <code>--env-file</code>; it is not read from a repository
<code>.env</code>.

An OpenAI-compatible localhost endpoint may omit a key. Non-local endpoints
require one and HTTPS. Plain HTTP is accepted only for loopback endpoints. URLs
containing user-info credentials, queries, or fragments are rejected, and
provider redirects are refused so bearer credentials cannot cross origins.
Anthropic traffic is pinned to the official API endpoint; ambient
<code>ANTHROPIC_BASE_URL</code> overrides are intentionally ignored.

Credentials belong in the process environment, never in
<code>.evalyn.yml</code>. Repository diff and selected context are sent to the
configured provider; read [privacy.md](privacy.md) before using private source.

Evalyn does not implicitly load <code>.env</code> from the target repository or
current directory. To load a credential file deliberately, use the global
<code>--env-file PATH</code> option before the subcommand. The file must exist,
must not be a symlink, and must resolve outside the target repository. Values
already present in the process environment win; the file does not override
them. This prevents a candidate-controlled <code>.env</code> from silently
changing provider credentials or trusted endpoints.

## Sandbox

<code>sandbox.runtime</code> accepts <code>auto</code>, <code>docker</code>, or
<code>podman</code>. A real check resolves <code>auto</code> once, before
execution, and records the concrete <code>docker</code> or <code>podman</code>
engine in the completed run. Local <code>unix://</code> and
<code>npipe://</code> client endpoints are accepted by default. A
<code>tcp://</code> or <code>ssh://</code> Docker/Podman endpoint is rejected
unless the operator opts in for that invocation:

~~~bash
evalyn --allow-remote-container --repo . check BASE...CANDIDATE
~~~

The opt-in is recorded. Reproduction and repair require the flag to match the
saved run, preventing a replay from silently moving between local and remote
daemons. The flag permits use of the endpoint; it does not validate the remote
host, secure an unencrypted TCP transport, or make a remote workspace boundary
equivalent to a local one.

Generated-probe execution uses a structured OCI invocation with a read-only
container root filesystem, a read-only disposable-worktree mount, no network,
a non-root user where supported, resource bounds, and a small explicit
environment.

<code>sandbox.setup_env</code> is a list of environment-variable names that may
be forwarded to a separate trusted dependency-setup phase. Values are resolved
from the parent process; values do not belong in configuration. Only supported
package-registry names (including <code>NPM_TOKEN</code>,
<code>NODE_AUTH_TOKEN</code>, pip/uv index URLs, and documented Poetry, Cargo,
Bundler, or NuGet prefixes) are accepted. Model-provider credentials are
rejected. Values are passed in an owner-only ephemeral env file, never in the
OCI argument list, and the file is removed after setup. A declared but unset
variable makes setup incomplete. Keep this list empty unless a private
dependency source is required.

A non-empty <code>sandbox.setup_env</code> is accepted only from an explicit
configuration outside the target repository. An unchanged repository config may
define trusted setup argv, but it cannot request that ambient credential values
be forwarded into setup.

Dependency setup is intentionally a different trust boundary from generated
probe execution: its container has network access and the disposable worktree
is writable so dependencies can be installed. Setup commands and images are
trusted configuration and must be reviewed before a check.

On macOS, custom <code>--state-root</code> directories must be shared with the
Docker Desktop or Colima VM. The default under the user home directory is
daemon-visible; Evalyn places disposable worktrees there rather than in the
macOS system temporary directory.

## Adapters

The alpha accepts <code>pytest</code> and <code>vitest</code>. Each adapter may
define:

- <code>image</code>: a trusted OCI image reference; registry paths and digest
  pins are accepted. Evalyn first launches a harmless no-network warmup from the
  declared reference, then requires bounded engine inspection to return exactly
  one canonical SHA-256 image ID. Meaningful dependency setup and generated
  probes use that immutable ID. The manifest stores declared/resolved identity,
  local/remote endpoint scope without its URI, and a fingerprint of the OCI
  execution policy. If the image ID cannot be resolved, the check becomes
  incomplete;
- <code>setup_argv</code>: a trusted string array for dependency preparation;
- <code>suite_argv</code>: a trusted string array for the repository’s existing
  suite.

Arrays are executed as arguments, not through a shell. They remain trusted user
configuration and may execute arbitrary programs, so review configuration from
untrusted branches before running it.

Probe-generation responses cannot override adapter name, probe path,
executable, arguments, working directory, environment, or revision. An
experimental repair response may name only a source file from the deterministic
repair allowlist; Evalyn rejects any other target.

The pytest adapter disables its cache plugin and the Vitest adapter disables
its result cache during probe runs, so framework caching does not need to write
into the read-only workspace.

Reproduction and repair do not adopt later edits to these execution settings.
Both replay the resolved image ID, setup array, concrete runtime, setup
environment names, repeat count, and execution budget recorded in the finding's
completed run manifest; repair validation also reuses the saved suite array.
Reproduction defaults to the saved repeat count. A <code>--runtime</code> value
may only restate the recorded engine; a mismatch is refused. This keeps a later
<code>.evalyn.yml</code> edit or a change in engine availability from silently
changing the execution configuration being replayed.

## Limits

- <code>max_tests</code> defaults to 12 requested probes. The current
  graph-guided portfolio accepts at most 8 hypotheses per adapter, and the CLI
  and generation boundary reject requests above 50.
- <code>repeat</code> defaults to 3 and accepts values from 2 through 10.
- <code>budget_seconds</code> defaults to 600 for a check.

All limits must be actual positive YAML integers; booleans, strings, and floats
are not coerced. The settings parser also applies defensive maxima. Repetitions
are used to confirm candidate-only failures; adaptive pass/pass and baseline
discovery paths normally stop after one execution per side.

## Generated default

<code>evalyn init --repo PATH</code> detects supported adapters without running
project code and writes a reviewable version-one configuration. It does not
write credentials or install dependencies.
