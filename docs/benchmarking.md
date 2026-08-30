# Benchmarking

Evalyn separates execution smoke tests from detector scoring. This matters
because a hand-written probe can prove that the adapter and classifier work,
but it says nothing about whether Evalyn can generate a useful probe from a
diff.

The version-two manifest at <code>bench/manifest.json</code> currently contains:

- two deterministic, non-scoring smoke cases;
- zero reviewed detector-scoring cases;
- a promotion target of 30 positive and 30 negative scoring cases, balanced as
  30 pytest and 30 Vitest cases.

> **Promotion gate: 0/60 scoring cases; not met.** The two smoke seeds do not
> advance this count and must not be reported as detector accuracy.

Validate the schema and fixture boundaries without executing framework code:

~~~bash
python -m bench.runner --validate
~~~

Run the existing deterministic pytest smoke:

~~~bash
python -m bench.runner \
  --execute \
  --strict-tools \
  --case pytest-discount-rounding
~~~

## Scoring case contract

Each reviewed scoring case is labeled <code>positive</code> or
<code>negative</code>. Negative cases also identify the control kind, such as
an intended change, behavior-preserving refactor, unchanged behavior, flaky
baseline, or adversarial repository instructions.

A case provides base and candidate snapshots, plus a fixed snapshot for every
positive. Its gold probe lives outside all revision snapshots. The gold probe
validates the fixture, but it is never copied into the detector repository,
included in context, or supplied to the provider. Positive gold evidence must
be base-pass, candidate-fail, fixed-pass.
Negative gold may explicitly expect an unstable side for a reviewed flaky-
baseline control; instability is never allowed for a positive case.

Run gold validation for every registered scoring case without invoking a
provider:

~~~bash
python -m bench.runner --execute-gold --strict-tools
~~~

With the checked-in 0/60 corpus this reports that no reviewed scoring cases are
registered; it does not treat that absence as a passing detector score.

The checked-in protocol runs three independently seeded generation trials per
case, scores the generated portfolio at <code>k=8</code>, and requires three
consistent executions per side for a causal finding. Every result bundle must
identify:

- provider and model;
- the exact generation prompt SHA-256;
- a secret-free environment SHA-256 covering the runtime, image digest,
  dependency inputs, architecture, and resource policy used by the evaluator.

Each trial repeats the canonical evaluation-fingerprint digest and its effective
<code>portfolio_k</code> and repetition policy. The scorer rejects a trial whose
fingerprint or protocol differs from the enclosing bundle and manifest, which
prevents cherry-picking observations from another model, environment, or search
budget.

Recorded providers and content-addressed recorded executions are available in
<code>bench.scoring</code> for no-key end-to-end harness tests. Providers can
replay the exact planning/candidate/correction reply sequence or use a scripted
fixture reply. The effective seed is passed to each scripted reply and bound
into both the request and ordered-response transcript digests. That path invokes
the public <code>run_check</code> workflow, including context collection, portfolio
generation, adapters, differential classification, semantic gating, and report
creation. Intervention-aware recordings separately pin the three candidate
executions made after the workflow temporarily reverts the claimed target file;
missing intervention outcomes are infrastructure errors rather than advisory
successes. Recorded outcomes test the harness; by themselves they are not
publishable detector evidence.

## Rescorable results and metrics

Raw observations use result schema version two and can be rescored without
provider or container access. Version one bundles are rejected instead of being
silently reinterpreted. Each trial carries a case-revision digest over the
canonical case contract, all base/candidate/fixed snapshot contents, and the
hidden-gold probe, so it cannot be rescored against a changed fixture with the
same case ID:

~~~bash
python -m bench.runner \
  --score-observations /path/to/observations.json \
  --json
~~~

The scorer reports:

- generated-probe validity;
- positive discovery at portfolio <code>pass@8</code>;
- false gates on negative controls and automatic-gate precision;
- invalid trials/probes and infrastructure failures;
- P50 and P95 end-to-end latency;
- clean-environment reproduction success.

Validity uses every evaluable initial candidate and bounded correction as its
denominator; rejected attempts and runtime-invalid probes count as invalid. Generation
rejections are reported separately and do not make a trial incomplete when the
workflow ultimately produces a complete verdict. Runtime-invalid probes,
verification-invalid probes, infrastructure failures, and insufficient workflow
evidence remain distinct trial outcomes. Automatic gates are copied from the
workflow verdict rather than inferred from the presence of a finding. A positive
gate enters the precision numerator only when that exact credited finding was
also fixed-verified; unrelated positive gates remain in the denominator.

Provider and collection transport failures are generation infrastructure, not
invalid model probes. They make the trial incomplete, are reported separately,
and are excluded from the validity denominator even if bounded correction later
produces an executable probe.

A positive discovery is credited only when the same generated probe also passes
the positive case's fixed revision three times. Its rank is the planned
hypothesis position of that first fixed-verified stable differential, so a
stable but unrelated probe cannot earn <code>pass@8</code>. Every credited
discovery then requires its own clean replay: three base passes and three
candidate failures with the original stable failure signature. Missing fixed or
reproduction recordings are infrastructure failures; an executed replay that
does not reproduce is complete evidence with a failed reproduction result.

The output distinguishes corpus composition, trial completeness, and metric
thresholds. Promotion passes only when all 60 reviewed cases are registered,
all three trials are present for every case, and every configured threshold is
met. Proportion metrics include 95% Wilson confidence intervals. Missing
denominators produce <code>null</code> metrics and never pass a threshold.

Do not manufacture cosmetic variants to fill the corpus. Cases must represent
distinct, reviewed failure modes or negative controls, pin all evaluation
inputs, and keep generated tests and fixed source hidden from the provider.
