# GitHub Actions

The checked-in Actions are repository-local development automation. They are not
a published reusable Evalyn Action. In particular, <code>action.yml</code> is a
source-only private-alpha preview; external
<code>uses: owner/Evalyn@ref</code> examples are intentionally omitted.

## Private dogfooding

On a manually selected protected ref, the root Action installs that trusted
checkout, runs one differential check, and returns Evalyn's exact process status.
It also writes a concise job summary and exposes paths to validated evidence that
is still available on the runner. Configure the named GitHub environment with
required reviewers before storing a provider credential in it.

~~~yaml
name: Evalyn private dogfood

on:
  workflow_dispatch:
    inputs:
      range:
        description: Reachable same-repository BASE...CANDIDATE comparison
        required: true
        type: string

jobs:
  evalyn:
    if: ${{ github.ref_protected }}
    runs-on: ubuntu-latest
    environment: evalyn-private-alpha
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.sha }}
          fetch-depth: 0
          persist-credentials: false

      - name: Run Evalyn
        id: evalyn
        uses: ./
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        with:
          range: ${{ inputs.range }}
          intent: Preserve documented compatibility
          output: artifacts/evalyn.sarif

      - name: Preserve Evalyn evidence
        if: ${{ always() && steps.evalyn.outputs.manifest != '' }}
        uses: actions/upload-artifact@v4
        with:
          name: evalyn-${{ steps.evalyn.outputs.run-id }}
          if-no-files-found: error
          path: |
            ${{ steps.evalyn.outputs.manifest }}
            ${{ steps.evalyn.outputs.report }}
            ${{ steps.evalyn.outputs.sarif }}
~~~

The manual input must name objects already reachable from the fetched
same-repository history. The checkout remains at the protected workflow commit;
Evalyn creates disposable worktrees for the requested revisions, so candidate
content does not replace the Action or installation source that receives the
provider credential.

The evidence step uses <code>always()</code> because exit 1, 2, and 3 make the
Evalyn step fail normally. Do not set <code>continue-on-error</code> when Evalyn
should gate the job. If a workflow does set it so later custom policy can run,
use the <code>exit-code</code> output rather than GitHub's normalized step
conclusion.

The output destination must not already exist and must not be a tracked file.
Those constraints come from Evalyn's no-overwrite output safety model. The
Action's default state directory is beneath <code>RUNNER_TEMP</code>, outside the
target repository; upload evidence during the same job if it must outlive the
runner.

## Outputs and evidence semantics

| Output | Meaning |
| --- | --- |
| <code>run-id</code> | The uniquely attributed new run, or empty if no run could be safely identified. |
| <code>verdict</code> | The manifest evidence verdict. <code>not_run</code> and <code>unknown</code> are Action-level sentinels, not detector verdicts. |
| <code>run-status</code> | <code>complete</code> or <code>incomplete</code> from the validated run manifest. |
| <code>manifest</code> | Absolute path to the validated, regular run manifest. |
| <code>report</code> | Absolute path to the regular Markdown evidence report. |
| <code>sarif</code> | Absolute path to the requested SARIF copy when its digest matches the run-owned SARIF, otherwise the run-owned path; empty when unavailable. |
| <code>finding-count</code> | Stable findings recorded by the run, when available. |
| <code>gating-finding-count</code> | Findings admitted by Evalyn's evidence policy, when available. |
| <code>exit-code</code> | The exact Evalyn status returned by the Action step. |

Run attribution is deliberately fail-closed. The Action snapshots Evalyn's
validated external run store before the check and emits run-owned paths only when
exactly one new run appears. It does not guess when attribution is ambiguous, and
it does not advertise a SARIF path merely because one was requested.

The four CLI statuses remain distinct:

- <code>0</code>: the check completed without triggering the selected
  <code>fail-on</code> policy. With <code>fail-on: never</code>, findings may still
  be present.
- <code>1</code>: the check completed and the selected policy gate triggered.
- <code>2</code>: invocation, range, or settings-schema input was invalid; the
  check was not run.
- <code>3</code>: repository/configuration trust inspection, execution, evidence,
  or output was incomplete. Partial reports may still be available.

A verdict of <code>no_differential_observed</code> means only that the executed
portfolio observed no differential. It is not proof that the candidate is
regression-free, and the job summary repeats that caveat.
<code>change_observed</code> means a stable differential was found but did not
satisfy the conservative automatic-gate policy.
<code>insufficient_evidence</code>, <code>incomplete</code>, <code>not_run</code>,
and <code>unknown</code> are likewise never clear results; the job summary says
so explicitly.

## Trust boundary

Do not run <code>uses: ./</code> from a checkout an untrusted pull request can
modify while also providing provider credentials or other secrets. The Action
metadata and installation source are part of that checkout, so a modified local
Action executes before Evalyn's container boundary applies. This preview is for
trusted private dogfooding only. It is not licensed or published for external
reuse.

The repository's own workflows use <code>pull_request</code>, never
<code>pull_request_target</code>; declare read-only repository permissions; do
not expose model-provider credentials; and do not publish packages, releases,
images, or Actions. Other checked-in composite Actions support Python setup and
asset validation, while <code>.github/workflows/ci.yml</code> runs tests, static
checks, benchmark validation, a trusted live container-boundary smoke test, and
package builds without provider credentials.
