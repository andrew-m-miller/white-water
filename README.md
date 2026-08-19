# White Water

Machine-learning optical flow tracking as an OpenFX plugin, for Autodesk Flame.

Takes a source plate, solves motion with RAFT or RIFE, and either carries a second layer
along on those vectors from a reference frame or hands the compositor an ST map to do the
warp downstream. It is Flame's motion vector tracking with a modern solver behind it.

Targets **Rocky Linux 9.5+** and **arm64 macOS**. MTI Film internal.

## Status

**Scaffolding.** The build configures and produces the host probe; the plugin itself does
not exist yet. Phase 0 is a measurement phase that has not been run — see the Open section
of [docs/host-notes.md](docs/host-notes.md). Nothing past Phase 0 should be written until
those answers are in, because two of them change the architecture rather than the schedule.

## Documents

| | |
|---|---|
| [docs/plan.md](docs/plan.md) | The approved plan: architecture, parameters, phasing, verification |
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
- Linux artifacts must be built in a `rockylinux:9` container (glibc 2.34) and gated with
  `scripts/check-glibc-baseline.sh`. This is the check that matters most, because its
  failure mode has no symptom: a plugin needing a newer glibc than the host provides is
  simply **absent from the menu, with no error in any log**. A build done on some other
  distribution is not a Flame artifact, however cleanly it compiles.
- From Phase 3 the build pulls ONNX Runtime and CUDA redistributables — hundreds of
  megabytes that are never going to be fetched on an airgapped machine.

Run the **build** workflow from the Actions tab. It is `workflow_dispatch` only and takes a
required `purpose` — name the human test the build is for, so a run in the history says why
it exists. It produces two artifacts, `whitewater-linux-rocky9` and `whitewater-macos`, each
a tarball with a SHA256 alongside and an `INSTALL.txt` inside.

On the box, verify the checksum before unpacking:

```bash
sha256sum -c whitewater-linux-*.tar.gz.sha256
```

Then copy the bundle into `/usr/OFX/Plugins` and restart Flame.

Beyond compiling, the workflow gates each artifact on the three things that fail silently:
the glibc baseline, the architecture directory inside the bundle, and that the binary
exports **exactly** the three OFX entry points and nothing else. The last one matters more
here than in most plugins — from Phase 3 this links ONNX Runtime, which carries protobuf,
abseil and a CUDA runtime, all of which Flame may already have loaded at other versions.

## Layout

```
src/core/     host-free: the flow algebra and the resampler. No OFX, no ONNX Runtime, no I/O.
src/ofx/      the host boundary: image adapters, pixel formats, frame capture, host quirks
tools/        hostprobe (Phase 0), ww-flow (Phase 2, offline CLI)
cmake/        bundle layout, rpath, version script
scripts/      the layering and glibc gates, run as tests
```

The `src/core` boundary is enforced by `ctest`, not by convention: a core source that
includes an OFX or ONNX Runtime header fails the build. That is what keeps the whole flow
pipeline testable on a laptop with no GPU and no model weights.

Much of `src/core` and `src/ofx` is vendored from the sibling **warp-drive** repository,
which is the version Flame has actually loaded. Each such file names its provenance in its
first line; see [docs/context.md](docs/context.md) for why it was copied rather than shared.
