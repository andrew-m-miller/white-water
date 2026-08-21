# White Water

An OpenFX plugin bringing RAFT/RIFE optical flow tracking into Autodesk Flame.
Rocky Linux 9.5+ and arm64 macOS. MTI Film internal.
Linux artifacts build on EL8 for a glibc 2.28 floor; see the build rule below.

## Read first

- `docs/plan.md` — the approved plan. Architecture, parameters, phasing, verification.
- `docs/host-notes.md` — what Flame actually does. **The authority on every host claim.**
- `docs/context.md` — why decisions went the way they did, and known weaknesses.

Documents own facts; this file owns process. Do not restate a measured fact here.

## Where the project is

**Scaffolding, Phase 0A closed.** The build produces the two probes and nothing else. The
plugin does not exist.

Phase 0A answered all five questions in Flame on 2026-08-20, including the two that could
have changed the architecture — `clipGetImage` works during render, and in-process inference
works. **Two gates remain before inference code:** 0B (a real model through the CUDA EP, plus
its payload closure) and 0C (Flame's ST convention, and instance/process lifetime). Phase 1
and the host-free parts of Phase 2 are unblocked now.

The plan was amended on 2026-08-20 after an architecture review. Read `docs/plan.md` and
`docs/context.md` before writing code against anything you remember about it.

## Branching and review

**Phase 0 works on `main`.** Planning, documents, probes and their measurement runs commit
straight to `main` — including 0B and 0C, which are still probe work however much code they
take. Throwaway probes reviewed as pull requests would be ceremony around something whose
whole value is being run on the box the same day.

**From Phase 1, nothing lands on `main` directly.** Once the plugin itself is being built:

- Branch from an up-to-date `main`, one topic per branch.
- Commit there, push, and open a PR with `gh pr create`. The PR body says what was measured
  or tested, not just what changed.
- **Do not merge it.** The PR exists to be reviewed by a human; that review is the point.
- Once it is merged, delete the branch, local and remote.
- Never commit or push to `main` from Phase 1 onward, and never force-push a branch under
  review.

If you find yourself already on `main` with Phase 1+ changes in the working tree, move them
to a branch before committing rather than committing and fixing it up afterwards.

CI is `workflow_dispatch` only and will not run itself on a PR — see `.github/workflows/`.
A PR that needs an artifact should say which `purpose` to dispatch it with, and which host
test the artifact is for.

## Rules that came from real failures

These are warp-drive's, paid for in lost days. They apply here unchanged.

- **Never throw out of `describe` or `describeInContext`.** Flame responds by showing no
  plugin at all and logging nothing. Set risky properties through the property set with the
  non-throwing flag, not through a setter that throws. Do not call `setEnabled()`.
- **Measure, do not assume.** Autodesk's OFX documentation is wrong in both directions.
  Every host claim traces to a probe report that names the host build and the date.
- **But read the specification when the question is the specification.** The OFX headers and
  reference docs are vendored in `third_party/openfx/`. Host *behaviour* is measured; protocol
  *semantics* are read, and citing a header beats a probe cycle. See `docs/context.md`,
  correction 6, which cost nothing only because it was caught in review.
- **The probe stays dependency-free.** Raw OFX C headers only. Anything in between could
  itself be why a measurement came out the way it did.
- **Export only the three OFX entry points.** `cmake/ofx.map` plus `--no-undefined`. This
  matters more here than in warp-drive: ONNX Runtime drags in protobuf, abseil and a CUDA
  runtime, all of which Flame may already have loaded at other versions.
- **`src/core` stays host-free.** No OFX, no ONNX Runtime, no I/O. Enforced by
  `ctest -R core::dependency_boundary`, not by convention. `src/infer` must stay OFX-free
  too — that half is **not yet gated**, and the plan as first written broke it. Give the
  script its second invocation before relying on the rule.
- **Build Linux artifacts in an `almalinux:8` container** against a hard-coded glibc 2.28
  baseline, and gate with `scripts/check-glibc-baseline.sh`. Never derive that floor from the
  build machine. A local modern-distro build is not a Flame artifact — a wrong-glibc plugin is
  simply absent from the menu, with no error anywhere.
- **Short parameter labels.** Flame's panel is a flat list and truncates at ~12 characters.
  Order is the only layout tool available.
- **Choice option order is API.** A saved setup stores the index. Options are appended,
  never inserted.
- **Flame's log is `/opt/Autodesk/log/`.** It captures plugin stderr, and on a machine
  nobody can attach a debugger to that is the only diagnostic channel there is. Print what
  matters at `load()`.

## Vendored code

Much of `src/core` and `src/ofx` is copied from the sibling `warp-drive` repository at
`887a123`, which is the version Flame has actually loaded. Each file names its provenance in
its first line. Copied rather than submoduled — see `docs/context.md`.

When fixing something in a vendored file, check whether warp-drive has the same bug, and
say so in the commit message. Nothing propagates automatically.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build --output-on-failure
```
