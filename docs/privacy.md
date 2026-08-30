# Privacy

Evalyn sends the selected diff, operator intent, deterministic change targets,
and ranked repository context to the configured model provider. A planning call
receives the bounded graph context; each candidate call receives context related
to its selected target/oracle (or a small top-ranked fallback). It does not
upload every repository file automatically, but graph-guided selection is wider
than only the changed file.

Candidate context is drawn from bounded base/candidate inventories of
repository-relative regular text files, plus explicitly included dirty changed
content. Ranking can include changed implementation, related tests and imports,
documentation, schemas, declarations, and project manifests. Items retain their
source side and content hash. Symlink files, binary content, oversized content,
Git metadata, external Evalyn state, and paths outside the repository are not
eligible. However, a tracked sensitive/vendor file can still become graph
context when it is changed or related; graph ranking is not a secret scanner or
data-classification system. Users must inspect the diff and repository policy
before sending proprietary source.

Provider request and response bodies are size-bounded. Provider exception text
is not fed back into correction prompts because it could contain a remote body,
path, or secret. Raw provider transport payloads are not stored wholesale, but
the run's <code>analysis/</code> artifacts deliberately retain the selected
change graph (including bounded context content), parsed hypotheses, accepted
and rejected generated source, attempt status, and sanitized correction
feedback. This audit trail can contain proprietary code and should be protected
like the source repository.

Run artifacts also include generated probes, normalized evidence, structured
test metadata, reports, and bounded command logs. Console output and persisted
full logs are not content-redacted. A trusted dependency installer or test that
prints an allowlisted registry credential can therefore place it in a setup or
execution log. Managed state is external to the checkout, size-bounded, and
accessed with repository/run identity and symlink checks, but filesystem access
to that state should still be restricted.

<code>bundle</code> intentionally creates a portable archive containing the
probe source, oracle, finding, run manifest, reproduction metadata, and payload
digests. It does not include all run logs, but its contents may still reveal
source and behavior. <code>promote</code> copies a confirmed generated probe into
the repository, where it becomes subject to normal version-control and CI data
handling. Review both outputs before moving them to another system.

Evalyn does not implicitly load a repository <code>.env</code>. An explicit
<code>--env-file</code> must be outside the repository, cannot be a symlink, and
does not override existing environment values. Setup receives only values named
by the package-registry allowlist, and a non-empty
<code>sandbox.setup_env</code> name list is accepted only from explicit external
configuration. Model-provider credentials are not forwarded to probe containers.
Environment fingerprints record normalized engine/image policy, not daemon
endpoint URIs or credential values; they retain only whether the endpoint scope
was local or remote.

Anthropic requests use the fixed official endpoint. OpenAI-compatible endpoints
are explicit and validated, and HTTP redirects are refused rather than
forwarding an authorization header. A repository-resident config cannot set
<code>provider.base_url</code>; that field requires operator-selected external
configuration (while <code>OPENAI_BASE_URL</code> remains an operator environment
input). Provider retention and training policy are outside Evalyn's control. An
explicitly allowed remote container daemon is also a separate privacy boundary:
mounted worktree paths/content and forwarded setup credentials may become
visible to infrastructure controlled by that endpoint's operator.
