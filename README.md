# White Water

Machine-learning optical flow tracking as an OpenFX plugin, for Autodesk Flame.

Takes a source plate, solves motion with learned optical flow, and either carries a second layer
along on those vectors from a reference frame or hands the compositor an ST map to do the
warp downstream. It is Flame's motion vector tracking with a modern solver behind it.

Targets **Rocky Linux 9.5+** and **arm64 macOS**.

## Status

**Phases 0 and 1 are closed.** The repository now builds the permanent
`WhiteWater.ofx.bundle` alongside its diagnostic probes. The product bundle contains the
Track/Insert and float-only ST Map descriptors; Phase 1 intentionally renders deterministic
fallbacks while the host-free flow pipeline begins in Phase 2.

All five Phase 0 questions were measured in Flame 2026.2 on 2026-08-20, and two came back
better than budgeted: `clipGetImage` works at arbitrary times *during* render, and a private
ONNX Runtime coexists with Flame's own in the same process. So the on-demand chain design
holds, and inference runs **in-process** — no IPC boundary to build.

The pinned SEA-RAFT M export now passes identity and direction on both CPU and CUDA through
the private runtime inside Flame. Warmed 480p–1080p timing/VRAM, lifecycle, cancellation,
provider-init fallback, duplicated-node behavior and the bounded CUDA arena-limit/CPU-recovery
gate are also measured. The actual Flame loader-path report closes CUDA payload ownership and
size accounting with no unresolved dependencies. The GPU-only qualification then measured
UHD, DCI 4K and Alexa 35 open gate under a 16 GiB ORT arena ceiling: all three produced
controlled bounded-allocation stops before completing a warm inference. That negative result
closes **0B** without imposing a product resolution cap. **0C closed** the ST convention,
render behavior and process lifetime, and settled v1 on an instance-lifetime RAM cache with no
persistence. **Phase 1 closed on 2026-08-22:** Flame 2026.2 verified the two descriptors,
parameter and reference-time contracts, Source/Insert fallbacks, matte propagation, partial
renders, native ST round-trip and load diagnostics. Phase 2 is the next implementation phase.
See [docs/host-notes.md](docs/host-notes.md), [docs/context.md](docs/context.md), and *Phasing*
in [docs/plan.md](docs/plan.md).

## Documents

| | |
|---|---|
| [docs/plan.md](docs/plan.md) | The approved plan: architecture, parameters, phasing, verification |
| [docs/phase2-implementation-plan.md](docs/phase2-implementation-plan.md) | The scoped Phase 2 work packages and explicit later-phase exclusions |
| [docs/phase2.5-implementation-plan.md](docs/phase2.5-implementation-plan.md) | The exact-artifact bake-off, airgapped production evaluation, and selection gates |
| [docs/host-notes.md](docs/host-notes.md) | What Flame actually does — measured, inherited, and open |
| [docs/context.md](docs/context.md) | Why decisions went the way they did; known weaknesses |
| [models/MODELS.md](models/MODELS.md) | Model provenance, licences, export procedure |

## Build — on a development machine

These steps need network for the submodule checkout, and a toolchain. They are for a Mac
or a Linux dev box, **not** for the Flame box. See the next section for that.

```bash
git submodule update --init --recursive
```

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```

```bash
ctest --test-dir build --output-on-failure
```

Bundles land in `build/bundles/`. Install by copying into the host's scan directory
(`/usr/OFX/Plugins` on Linux, `/Library/OFX/Plugins` on macOS) or by pointing
`OFX_PLUGIN_PATH` at `build/bundles`.

## Build — for the Flame box

**Do not build on the Flame box.** It is airgapped and it is a production machine. Build a
Linux artifact in CI and carry it over, the way warp-drive does:

- The submodule checkout needs network.
- A stock Flame install has no C++ toolchain, and installing one needs `dnf`.
- Linux artifacts must be built in an `almalinux:8` container against a **hard-coded glibc
  2.28** baseline, gated with `scripts/check-glibc-baseline.sh`. This was originally
  `rockylinux:9`/2.34 "matching the stated target", which produced a plugin Flame refused to
  load — see [docs/context.md](docs/context.md), correction 3. This is the check that matters most, because its
  failure mode has no symptom: a plugin needing a newer glibc than the host provides is
  simply **absent from the menu, with no error in any log**. A build done on some other
  distribution is not a Flame artifact, however cleanly it compiles.
- Phase 0B stages an ONNX Runtime CUDA redistributable and a pinned real model for the probe.
  Phase 3 turns the qualified runtime into the production inference payload. Neither is ever
  fetched on the airgapped machine.

Run the **build** workflow from the Actions tab. It is `workflow_dispatch` only and takes a
required `purpose` — name the human test the build is for, so a run in the history says why
it exists. It produces two artifacts, `whitewater-linux-el8` and `whitewater-macos`, each
a tarball with a SHA256 alongside and an `INSTALL.txt` inside.

On the box, verify the checksum before unpacking:

```bash
sha256sum -c whitewater-linux-*.tar.gz.sha256
```

Then copy the bundle into `/usr/OFX/Plugins` and restart Flame.

Beyond compiling, the workflow gates each artifact on the three things that fail silently:
the glibc baseline, the architecture directory inside the bundle, and that the binary
exports **exactly** the three OFX entry points and nothing else. The last one matters more
here than in most plugins — from Phase 3 this privately loads ONNX Runtime, whose dependency
closure includes protobuf, abseil and a CUDA runtime that Flame may already have loaded at
other versions. The module itself must retain no `DT_NEEDED` entry for ONNX Runtime.

## Layout

```
src/core/     host-free: the flow algebra and the resampler. No OFX, no ONNX Runtime, no I/O.
src/infer/    Pairwise estimator contract and test double; ONNX Runtime arrives in Phase 3. No OFX.
src/ofx/      the host boundary: image adapters, pixel formats, frame capture, host quirks
tools/        hostprobe and ortprobe (Phase 0), ww-flow (Phase 2, offline CLI)
cmake/        bundle layout, rpath, version script
scripts/      the layering and glibc gates, run as tests
```

The `src/core` and `src/infer` boundaries are enforced by `ctest`, not by convention: core
rejects OFX, ONNX Runtime and I/O headers; infer rejects OFX while allowing inference
dependencies. Both policies have negative fixtures that prove the forbidden include is rejected.
That is what keeps the flow pipeline testable on a laptop with no GPU and no model weights.

Much of `src/core` and `src/ofx` is vendored from the sibling **warp-drive** repository,
which is the version Flame has actually loaded. Each such file names its provenance in its
first line; see [docs/context.md](docs/context.md) for why it was copied rather than shared.
