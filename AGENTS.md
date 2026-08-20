# White Water

An OpenFX plugin bringing RAFT/RIFE optical flow tracking into Autodesk Flame.
Rocky Linux 9.5+ and arm64 macOS. MTI Film internal.

## Read first

- `docs/plan.md` — the approved plan. Architecture, parameters, phasing, verification.
- `docs/host-notes.md` — what Flame actually does. **The authority on every host claim.**
- `docs/context.md` — why decisions went the way they did, and known weaknesses.

Documents own facts; this file owns process. Do not restate a measured fact here.

## Where the project is

**Scaffolding, pre-Phase 0.** The build produces the host probe and nothing else. The
plugin does not exist.

**Do not write Phase 1+ code before Phase 0 has been run on the box.** Two of the five
open questions change the architecture rather than the schedule: whether `clipGetImage`
works at arbitrary times *during* render, and whether ONNX Runtime initialises inside
Flame's process at all. Building on either assumption before it is measured is how this
project wastes a month.

## Rules that came from real failures

These are warp-drive's, paid for in lost days. They apply here unchanged.

- **Never throw out of `describe` or `describeInContext`.** Flame responds by showing no
  plugin at all and logging nothing. Set risky properties through the property set with the
  non-throwing flag, not through a setter that throws. Do not call `setEnabled()`.
- **Measure, do not assume.** Autodesk's OFX documentation is wrong in both directions.
  Every host claim traces to a probe report that names the host build and the date.
- **The probe stays dependency-free.** Raw OFX C headers only. Anything in between could
  itself be why a measurement came out the way it did.
- **Export only the three OFX entry points.** `cmake/ofx.map` plus `--no-undefined`. This
  matters more here than in warp-drive: ONNX Runtime drags in protobuf, abseil and a CUDA
  runtime, all of which Flame may already have loaded at other versions.
- **`src/core` stays host-free.** No OFX, no ONNX Runtime, no I/O. Enforced by
  `ctest -R core::dependency_boundary`, not by convention.
- **Build Linux artifacts in a `rockylinux:9` container** and gate with
  `scripts/check-glibc-baseline.sh`. A local modern-distro build is not a Flame artifact —
  a wrong-glibc plugin is simply absent from the menu, with no error anywhere.
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
