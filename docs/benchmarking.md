# Benchmarking

Evalyn’s public quality claim should be grounded in causal fixtures rather than
the number of unit tests or model-generated anecdotes.

The version-one manifest at <code>bench/manifest.json</code> registers a pytest
case and a Vitest case. Each case has:

1. a passing base snapshot;
2. a candidate snapshot containing one seeded defect;
3. a fixed snapshot;
4. one identical probe;
5. adapter-owned argument arrays;
6. explicit base-pass, buggy-fail, fixed-pass expectations.

> **Promotion gate: 2/60 cases implemented; not met.** These are seed fixtures
> for the corpus contract, not evidence that Evalyn is broadly effective. The
> machine-readable manifest records the target, implemented count, and gate
> status. Public promotion remains blocked until 60 reviewed causal cases exist.

Static validation is dependency-free after Evalyn’s normal installation:

~~~bash
python -m bench.runner --validate
~~~

The pytest fixture is the deterministic no-key smoke test used by CI:

~~~bash
python -m bench.runner \
  --execute \
  --strict-tools \
  --case pytest-discount-rounding
~~~

Vitest execution is optional locally. The runner uses
<code>npx --no-install</code> and skips the case when Vitest is unavailable; it
never downloads benchmark dependencies.

## Metrics planned for the alpha

- causal detection rate;
- false-positive rate on unchanged and intended-change controls;
- unstable/invalid rate;
- median probes and executions per diff;
- wall-clock time and provider tokens;
- reproduction success from a clean checkout.

No aggregate score is published yet. Two seeded cases establish corpus and
runner contracts, not meaningful product performance. Cases must represent
distinct real failure modes; duplicating or mechanically varying a seed does not
advance the 60-case gate.
