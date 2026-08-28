# GitHub Actions

The checked-in Actions are repository-local development automation. They are not
a published reusable Evalyn Action.

- <code>.github/actions/setup-python-project</code> installs the selected Python
  version and editable development dependencies.
- <code>.github/actions/validate-assets</code> validates repository YAML, JSON,
  Markdown links, benchmark fixtures, and the deterministic no-key demo.
- <code>action.yml</code> is a source-only preview of the eventual check Action.
  It uses array-safe inputs, emits SARIF, and preserves Evalyn’s 0/1/2/3 exit
  contract.
- <code>.github/workflows/ci.yml</code> runs tests, static checks, asset
  validation, a trusted live container-boundary smoke test, and package builds
  without provider credentials.

The workflows deliberately:

- use <code>pull_request</code>, never <code>pull_request_target</code>;
- declare read-only repository permissions;
- do not expose model-provider credentials;
- use only a checked-in trusted probe for the live container smoke test;
- do not publish packages, releases, images, or Actions.

External <code>uses: owner/Evalyn@ref</code> examples are intentionally omitted.
The repository has no license and publication is blocked.

For private dogfooding inside this repository, a workflow may use the checked-out
Action directly:

~~~yaml
- uses: actions/checkout@v4
  with:
    fetch-depth: 0
- uses: ./
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  with:
    range: ${{ github.event.pull_request.base.sha }}...${{ github.sha }}
    intent: Preserve documented compatibility
~~~

Do not use this on untrusted forks with provider credentials. The Action source
exists for integration testing; it is not licensed or published for external
reuse.
