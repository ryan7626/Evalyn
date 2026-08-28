## What changed

Describe the observable behavior and why the change belongs in Evalyn’s current
diff-regression scope.

## Evidence

List the exact commands run and summarize their result.

~~~text
python -m pytest
python -m bench.validate_repository
python -m bench.runner --validate
~~~

## Safety review

- [ ] The original target checkout, index, branches, and untracked files remain untouched.
- [ ] Model output does not control paths, executables, arguments, environment, or working directory.
- [ ] New generated-code execution is containerized, network-disabled, bounded, and cleaned up.
- [ ] Reports distinguish evidence from semantic or human judgment.
- [ ] Tests cover cleanup, cancellation, containment, and failure behavior where relevant.
- [ ] Documentation and benchmark fixtures are updated when public behavior changes.
- [ ] No secrets, private source, generated run artifacts, or dependency directories are included.

## Compatibility

Call out schema, CLI, report, adapter, provider, Python, operating-system, or
container-runtime compatibility changes.
