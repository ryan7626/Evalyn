# Benchmark corpus

The version-two corpus has two deliberately separate tiers:

- <code>smoke_cases</code> execute trusted, prewritten probes against base,
  buggy, and fixed snapshots to verify adapter and classifier plumbing;
- <code>scoring_cases</code> run Evalyn’s detector end to end against reviewed
  positive and negative cases, with hidden gold probes used only to validate
  corpus truth.

> **Current progress: 2 smoke cases and 0/60 scoring cases.** The promotion
> target is 30 positives, 30 negatives, 30 pytest cases, and 30 Vitest cases.
> Smoke cases never count toward it.

Validate paths, commands, expectations, scoring policy, and exact promotion
counts:

~~~bash
python -m bench.runner --validate
~~~

Execute the deterministic pytest smoke:

~~~bash
python -m bench.runner --execute --strict-tools --case pytest-discount-rounding
~~~

Execute every locally available smoke and emit JSON:

~~~bash
python -m bench.runner --execute --json
~~~

The Vitest smoke is skipped unless Vitest is already installed. The runner
uses <code>npx --no-install</code>, or an installed <code>vitest</code> binary on
<code>PATH</code>; it never downloads benchmark dependencies.

## Adding a smoke case

1. Add <code>base</code>, <code>buggy</code>, <code>fixed</code>, and
   <code>probe</code> directories under <code>bench/corpus/CASE_ID</code>.
2. Register it under <code>smoke_cases</code> with adapter-owned probe path and
   argv.
3. Require base-pass, buggy-fail, fixed-pass for exactly three repetitions.
4. Keep it deterministic, credential-free, and network-free.

## Adding a scoring case

1. Create reviewed base and candidate snapshots; positives also require a
   fixed snapshot.
2. Place the hidden gold probe outside every revision snapshot. Positive gold
   must produce base-pass, candidate-fail, fixed-pass.
3. Label the case positive or negative. Give negative controls an explicit
   kind: intended change, refactor, unchanged behavior, flaky baseline, or
   adversarial instructions.
4. Increment only the matching scoring counts in <code>promotion_gate</code>.
   The schema rejects counts that include smoke cases.
5. Run <code>python -m bench.runner --execute-gold --strict-tools</code> to verify
   the hidden oracle independently of generation.
6. Run three independently seeded detector trials and retain the effective seed,
   request and ordered-response transcript digests, plus provider, model, prompt,
   environment, canonical evaluation, and canonical case-revision fingerprints.
7. Credit a positive discovery only after the generated probe passes the fixed
   revision three times, then record a separate three-by-three base/candidate
   reproduction replay for that discovery.
8. When the workflow performs a target-file-revert intervention, retain its
   three candidate outcomes separately from ordinary candidate execution.

No aggregate score is published until all required cases and trials exist.
Use <code>bench.scoring</code> for the versioned observation schema, metric
aggregation, and no-key recorded workflow harness. See
<code>docs/benchmarking.md</code> for the full scoring contract.
