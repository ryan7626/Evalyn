# Privacy

Evalyn sends selected diff and repository context to the configured model
provider. It does not upload the entire repository automatically.

Context selection is restricted to tracked regular files inside the repository,
plus explicitly included dirty content. Environment files, Git metadata, Evalyn
state, binary files, oversized files, and paths outside the repository are
excluded. Users remain responsible for reviewing provider terms and repository
policy before sending proprietary source.

Run artifacts include probes, normalized evidence, and command output. Prompts
and raw model responses are not retained by default. Captured output and
persisted full logs have hard size bounds, but their content is not automatically
redacted. A trusted dependency installer that prints an allowlisted registry
credential can therefore place it in a setup log. Managed state records are
size-bounded and accessed without following symlinks. Sensitive values must not
be printed by setup commands or tests.

Provider request and response bodies are also size-bounded. Anthropic requests
use the fixed official endpoint. Compatible endpoints are explicit, and HTTP
redirects are refused rather than forwarding an authorization header.
