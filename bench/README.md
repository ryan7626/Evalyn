# Benchmark corpus

The corpus is a deterministic quality gate for Evalyn’s evidence pipeline. Each
case contains three small repository snapshots:

- <code>base</code> establishes passing behavior;
- <code>buggy</code> introduces exactly one intended defect;
- <code>fixed</code> removes that defect.

One identical probe is copied into every snapshot. A valid causal case must
produce base-pass, buggy-fail, fixed-pass across three repetitions and classify
base versus buggy as <code>stable_change</code>.

> **Current progress: 2 implemented seed cases / 60 required promotion cases.**
> The promotion gate is not met. Do not interpret the seed corpus as a benchmark
> score, and do not create cosmetic duplicates to increase the count.

Validate paths, commands, expectations, and fixture shape without executing
framework code:

~~~bash
python -m bench.runner --validate
~~~

Execute the pytest case:

~~~bash
python -m bench.runner --execute --strict-tools --case pytest-discount-rounding
~~~

Execute every locally available framework and emit JSON:

~~~bash
python -m bench.runner --execute --json
~~~

CI installs the manifest-pinned <code>vitest@2.1.9</code> with lifecycle
scripts disabled, adds only its executable directory to <code>PATH</code>, and
executes both seed cases as causal base/buggy/fixed checks.

The Vitest fixture is skipped unless Vitest is already installed. The runner
uses the manifest's structured <code>npx --no-install</code> command, or an
already-installed <code>vitest</code> executable on <code>PATH</code>; corpus
execution never downloads dependencies.

## Adding a case

1. Add <code>base</code>, <code>buggy</code>, <code>fixed</code>, and
   <code>probe</code directories under <code>bench/corpus/CASE_ID</code>.
2. Keep the changed behavior minimal and independently understandable.
3. Register adapter-owned probe path and argv in <code>manifest.json</code>.
4. Require base-pass, buggy-fail, fixed-pass and exactly three repetitions.
5. Run validation and the framework-specific case locally.

Do not add network calls, credentials, nondeterministic time behavior, or a
fixture whose expected result depends on model judgment.
