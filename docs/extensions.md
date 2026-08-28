# Writing a test adapter

The alpha supports pytest and Vitest. An adapter is deliberately narrower than
a general plugin: it owns how one generated probe is named, validated, executed,
and normalized for a specific framework.

An adapter must:

1. Detect its framework from repository files without executing project code.
2. Validate that generated source is nonempty, collectable, and contains a
   meaningful assertion.
3. Choose a content-addressed relative path inside the disposable worktree.
4. Return an argument array that executes only that probe.
5. Normalize nondeterministic output before repeated signatures are compared.

Adapters must never accept a model-authored path or command. Add contract tests
covering detection, path containment, command construction, invalid probes, and
stable signature normalization. New framework support should be proposed as a
small adapter plus a seeded benchmark fixture, not as conditionals in the core
engine.
