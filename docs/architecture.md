# Architecture

Evalyn is a differential test engine. It asks a model for test source, but keeps
all paths, commands, execution, classification, and reporting deterministic.

```text
Git range + optional intent
          │
          ▼
  immutable diff/context ──► model generates probe source
          │                            │
          │                   framework adapter validates
          │                            │
          ▼                            ▼
 detached base worktree       identical probe artifact
 detached candidate worktree           │
          │                             │
          └──────── container runs ◄────┘
                        │
                        ▼
        repeated outcomes + normalized signatures
                        │
                        ▼
         classification, artifacts, reproduction
```

## Trust boundaries

- Git revision resolution, generated paths, command arguments, and finding IDs
  are deterministic application logic.
- Probe-generation output is untrusted test source. It never selects a filename,
  working directory, executable, or shell expression. Experimental repair may
  propose only a file in a deterministic source allowlist, and the path is
  revalidated before use.
- Both revisions run the exact same content-addressed probe.
- Trusted dependency setup may use network access and a writable disposable
  worktree. The generated-probe phase then mounts that prepared worktree and
  the container root read-only, disables network access, and uses a small
  explicit environment.
- Host Git inspection runs with hooks, filesystem monitors, recursive
  submodules, configured filters, external diffs, and textconv neutralized.
- The original checkout is read-only from Evalyn’s perspective. Temporary
  worktrees and run state have explicit ownership manifests.
- A stable differential is not automatically a regression. Deterministic
  gating requires either a verbatim user-supplied preservation invariant or an
  oracle quoted exactly from a path-derived, category-matched existing test,
  documentation file, API schema, or declaration file. Model provenance,
  implementation/diff/header text, and topical intent overlap are insufficient.
- Repair is a second operation over a likely-regression or maintainer-confirmed
  finding. Before consulting a model, it reconstructs the untouched candidate
  and requires the saved status and failure signature for at least three runs.
  Reproduction and repair reuse the original manifest's relevant image
  reference, setup argv, concrete runtime, setup environment names, repeat
  count, and budget; repair validation also reuses the saved suite argv. Neither
  operation may substitute another runtime. Test harnesses, package/test
  configuration, setup files, and CI hooks are ineligible targets. Repair can
  only emit a new patch file and refuses existing output paths.
- Saved run and finding records are schema- and repository-bound; symlinked or
  identity-mismatched records are not accepted for later operations. Managed
  state uses no-follow, directory-anchored access, atomic JSON writes, bounded
  record sizes, and bounded full logs.
- Host Git streams, rendered diffs, context files, overlays, manifests, and
  replay/repair inputs have byte ceilings before decoding or copying.

## Extension boundaries

Framework adapters detect repositories, validate generated source, select a
safe probe path, construct argument arrays, and normalize test failures. Model
providers implement structured text completion. Execution backends are not a
plugin boundary in the alpha: only Docker and Podman are supported.
