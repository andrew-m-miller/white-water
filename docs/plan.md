# White Water — ML optical flow tracking OFX plugin for Flame

## Context

Flame's built-in motion vector tracking is a classical solver. Shots with motion blur,
low contrast, non-rigid deformation or fine detail defeat it, and the artist falls back
to hand-tracking or a roundtrip out of Flame. `white-water` is a new OpenFX plugin that
replaces that solver with modern learned optical flow (RAFT for accuracy, RIFE for
speed), keeping the familiar Flame workflow: analyse a source plate, position an insert
at a reference frame, let the vectors carry it — or hand a compositor an ST map and let
them do the warp downstream.

Targets are Rocky Linux 9.5+ and arm64 macOS, host is Autodesk Flame.

The repository is empty (git only, no commits). Everything below is new, except where it
is deliberately vendored from `/Users/andrew/repos/warp-drive`, whose `docs/host-notes.md`
is the measured record of what Flame's OFX implementation actually does. That document is
the authority for every host claim in this plan; none of it comes from Autodesk's docs,
which are wrong in both directions.

---

## Decisions already taken

| Question | Decision |
|---|---|
| warp-drive reuse | **Vendor a copy** — independent build and release cadence |
| Inference runtime | **ONNX Runtime** — CUDA EP on Linux, CoreML/CPU on macOS, CPU everywhere as fallback |
| Model weights | **Bundled in `Contents/Resources/models/`, with env var + param override** |
| Analysis trigger | **On-demand caching by default, plus an explicit `Analyze` button with progress** |
| RAFT vs RIFE | **Selectable, one `Model` param, RAFT default**, both behind one `FlowEstimator` |
| ST map convention | **Both** — absolute normalized UV (default) or relative pixel offset; origin toggle |
| Occlusion handling | **Forward-backward consistency check, parameter-gated, off by default** (2× inference cost) |

---

## The load-bearing host facts

From `warp-drive/docs/host-notes.md` (Flame 2026.2 Linux / 2027 macOS, measured):

- **No CUDA/Metal/OpenCL OFX render suite.** Only OpenGL is advertised, and the GPU
  properties are *strings* (`"true"`/`"false"`), not ints. ML inference therefore runs
  inside the plugin process on its own device context, entirely outside OFX.
- **`clipGetImage` works outside render at every offset tested** (0, ±1, ±5, ±24). This is
  what makes on-demand chain pulls viable at all.
- `SupportsTiles = 1`, `SupportsMultiResolution = 0`. Depths byte/short/half/float,
  components RGBA/RGB/Alpha, **unpremultiplied**.
- **Flame hands OFX 0-based time**: a batch starting at frame 1001 arrives as time 0.
- **Panel is flat**, groups do not nest, **labels truncate at ~12 characters**.
- `OfxProgressSuite` v1/v2 and `OfxTimeLineSuite` available. Message suite V2 present in 2027.
- **A throw out of `describe`/`describeInContext` means the plugin is invisible with nothing
  in any log.** Set risky properties through the property set with the non-throwing flag.
- **`-fvisibility=hidden` hides the OFX entry points** unless they carry explicit
  `visibility("default")`; a version script cannot resurrect them.
- Flame Batch **splits each RGBA OFX clip into Front + Matte sockets** — two RGBA clips
  present as four sockets, which is exactly the workflow the artist expects.

---

## Phase 0 — Probe before building (blocking)

Nothing else in this plan is worth writing until these are measured on the actual Rocky 9
Flame box. Extend a vendored copy of `warp-drive/tools/hostprobe/hostprobe.cpp` (raw C API,
no support library, no dependencies) and run it in Batch:

1. **General context with two RGBA clips**, second one `setOptional(true)` — do both appear,
   how do the sockets present, what does `clipGetImage` return on the disconnected one?
2. **`clipGetImage` at arbitrary times *during* the render action** (as opposed to during an
   instance-changed action, which is all warp-drive measured). The whole flow-chain design
   depends on this. Test ±1, ±10, ±100 and out-of-range.
3. **An ONNX Runtime session created inside Flame's process** with the CUDA EP — does it
   initialise, does it get a device, does it survive alongside Flame's own CUDA context, and
   what does VRAM look like? Include a trivial matmul model, ~1 MB, in the probe bundle.
   **Do the Mocha Pro bundle inspection in `host-notes.md` first** — Mocha Pro ships an OFX
   plugin doing ML matting on this hardware in production, and `ldd` plus `nvidia-smi` on it
   costs minutes and may answer this before we build anything.
4. Whether declaring `setSupportsTiles(false)` actually yields whole-frame render windows.
5. `getFramesNeeded` behaviour: does declaring only `{N}` while pulling other frames work?

Write results to `docs/host-notes.md` in white-water. Items 2 and 3 are the project's real
risk; if either fails, the architecture changes:

- **Item 2 fails** → a mandatory Analyze pass writing a disk cache, driven from an
  instance-changed action, which warp-drive measured to work.
- **Item 3 fails** → *not* automatically CPU-only. There are two outcomes, and the Mocha
  evidence suggests the second is live: inference could move **out of process**, into a
  supervised helper with its own address space and its own CUDA context. warp-drive already
  has that machinery working for its editor (`src/ofx/EditorProcess.cpp`, fork/exec with
  environment scrubbing, plus `src/ipc/` for the frame channel), so it is a port rather than
  a design. CPU-only is the outcome only if GPU inference fails in *both* process models —
  and that is what makes RAFT at 4K unusable and promotes RIFE from fast option to only
  option.

---

## Architecture

Three layers, enforced mechanically by a vendored `scripts/check-core-dependencies.cmake`:

```
src/core/    host-free, no OFX, no ONNX Runtime, no I/O   → unit-testable
src/infer/   ONNX Runtime, no OFX                          → testable with a CLI
src/ofx/     the plugin; one TU (Plugin.cpp) in the module
```

### Vendored verbatim from warp-drive

Copy with provenance headers, renamespaced `warpdrive` → `whitewater`:

| Source | Why |
|---|---|
| `src/core/geom/Vec2.h`, `src/core/image/Image.h` | float RGBA view with row stride, bottom-left origin |
| `src/core/warp/WarpMap.h` | the `mapToSource(Vec2) → Vec2` interface a dense flow field implements directly |
| `src/core/warp/Resampler.{h,cpp}` | backward warp with correct unpremultiplied filtering, edge modes, row-range threading, bit-exact identity |
| `src/ofx/HostImage.{h,cpp}`, `PixelFormat.{h,cpp}` | zero-copy float RGBA borrow, depth/component conversion incl. hand-written half, negative `rowBytes` |
| `src/ofx/FrameCapture.{h,cpp}` | `CapturedFrame` — an owned host-free frame; exactly the input an inference call needs |
| `src/ofx/HostQuirks.{h,cpp}` | per-host quirks keyed on host name, dumped to stderr at `load()` |
| `cmake/OfxBundle.cmake`, `cmake/ofx.map`, `cmake/Info.plist.in` | bundle layout, `$ORIGIN`/`@loader_path` rpath, three-symbol version script |
| `scripts/check-core-dependencies.cmake`, `check-glibc-baseline.sh` | layering and ABI gates |
| `tests/hostharness/` | minimal OFX host that actually renders |

`Resampler`'s `WarpMap` abstraction is the key reuse: a dense flow field is a `WarpMap`
subclass and the entire correct-alpha, correct-edge, threaded backward warp comes for free.

### New — `src/core/flow/`

- **`FlowField.{h,cpp}`** — dense 2-channel field over a pixel rectangle, half-float storage,
  bilinear sample, and a `FlowWarpMap : WarpMap` adapter so `Resampler` consumes it unchanged.
- **`FlowCompose.{h,cpp}`** — `compose(a, b)(q) = a(q) + b(q + a(q))`; forward-backward
  consistency; Gaussian smoothing; scale-up from analysis resolution.
- **`FlowChain.{h,cpp}`** — the reference-frame accumulation policy (below). Pure: it decides
  *which links are needed in which direction*, and takes cached ones as input.
- **`FlowCache.{h,cpp}`** — LRU keyed on `(fromFrame, toFrame, model, analysisScale, matteMode,
  inputCurve, modelParams)`, byte-budgeted, storing fields at analysis resolution.
- **`StMap.{h,cpp}`** — field → absolute normalized UV or relative pixel offset, bottom-left
  or top-left origin.
- **`Composite.{h,cpp}`** — the `over` operator with explicit premultiplication handling.
- **`Preprocess.{h,cpp}`** — premultiply-by-matte, input curve, downscale, reflect-pad to the
  model's required multiple, crop back.

### New — `src/infer/`

- **`FlowEstimator.h`** — `estimate(const CapturedFrame &a, const CapturedFrame &b,
  const FlowRequest &) → FlowResult`. RAFT and RIFE are two implementations; so is
  `NullEstimator`, which synthesises a deterministic analytic flow so every test above this
  line runs with no weights and no GPU.
- **`ModelSpec.{h,cpp}`** — the per-model tensor contract as data: input/output names,
  normalization, pad multiple (8 for RAFT, 32 for RIFE), iteration handling, output layout.
- **`ModelRegistry.{h,cpp}`** — resolve weights: `Model Dir` param → `WHITEWATER_MODEL_DIR`
  env → `Contents/Resources/models/` in the bundle. Report which one won, to stderr.
- **`OrtEnvironment.{h,cpp}`** — one process-wide `Ort::Env`; a mutex-guarded session cache
  keyed on `(model, resolution, execution provider)` because session creation costs seconds
  and must never happen per frame; EP selection with explicit fallback; bounded intra-op
  thread count (Flame owns all 72 CPUs and an unbounded ORT pool will fight it);
  `RunOptions::SetTerminate` wired to the host's abort.
- **`RaftEstimator`, `RifeEstimator`** — thin, differing only through `ModelSpec`.

**No ONNX Runtime exception may escape an OFX action.** Every call is wrapped; failure sets
a persistent message and renders passthrough.

### New — `src/ofx/`

- `OpticalFlowPlugin.{h,cpp}` — `describe`, `describeInContext`, `createInstance`, `render`,
  `isIdentity`, `getRegionsOfInterest`, `getFramesNeeded`, `getClipPreferences`, `changedParam`.
- `FlowParameters.{h,cpp}` — parameter definitions and typed fetch.
- `FlowPreparation.{h,cpp}` — cache keying, chain construction, frame pulls, abort, progress.
  Modelled on `warp-drive/src/ofx/WarpPreparation.h`'s staged one-entry cache.
- `Plugin.cpp` — the only TU compiled into the module.

---

## The flow chain

Let `R` be the reference frame and `N` the frame being rendered, both in OFX time. The
renderer needs the **backward** map `B(N→R)`: for each output pixel `q` at frame `N`, where
in frame `R`'s space it came from.

```
N == R :  identity
N >  R :  B(N→R) = compose( flow(N→N-1), B(N-1→R) )
N <  R :  B(N→R) = compose( flow(N→N+1), B(N+1→R) )
```

Each link is one pairwise inference, in the direction *toward* `R` — so a chain of length
`k` costs `k` inferences, not `2k`. Composing needs a bilinear sample of the accumulated
field at a warped position, which is `FlowField::sample`.

**Caching.** Pairwise links are cached individually, so scrubbing backwards reuses everything.
The accumulated field for the last-rendered `N` is cached separately, so the common sequential
case is one new inference plus one compose. Fields are stored at analysis resolution as
half-float 2-channel: a 1080p half-res link is ~2 MB, so a `Cache MB` budget of a few GB holds
a long shot.

**Drift** is inherent to accumulation and is not solved in v1. It is mitigated by the
`Smooth` parameter and bounded by keeping the artist's reference frame near the working range.
A later `Max Chain` parameter can fall back to a direct `R→N` inference past a threshold —
noted, not built.

---

## Parameters

Flat list, order is the only layout tool, labels ≤ 12 characters. **Choice option order is
API** — a saved setup stores the index, so options are only ever appended. **No `setEnabled()`
anywhere**: a host that refuses `kOfxParamPropEnabled` turns that into a throw out of an
action, and in Flame that means a plugin that simply is not there.

| Script name | Type | Label | Notes |
|---|---|---|---|
| `model` | Choice | `Model` | RAFT (default), RIFE |
| `refFrame` | Int | `Ref Frame` | OFX time — 0-based within the Flame batch |
| `setRef` | Push | `Set Ref` | writes `args.time` into `refFrame`; the only honest way for an artist to set it given Flame's 0-based time |
| `output` | Choice | `Output` | Composite (default), Warped Insert, ST Map |
| `matte` | Choice | `Matte` | Premultiply (default), Full Frame |
| `inputCurve` | Choice | `Input` | Clamp 0-1 (default), None, Filmic — models are trained on 0-1 data and Flame plates are log or scene-linear |
| `analysisScale` | Choice | `Analysis` | Full, Half (default), Quarter — RAFT's all-pairs correlation volume is infeasible at 4K full |
| `iterations` | Int | `Iters` | RAFT refinement iterations |
| `smooth` | Double | `Smooth` | post-flow smoothing, 0 default |
| `fbCheck` | Bool | `FB Check` | off; 2× inference cost |
| `fbTolerance` | Double | `FB Tol` | round-trip tolerance in pixels |
| `stMode` | Choice | `ST Mode` | Absolute UV (default), Relative Pixels |
| `stOrigin` | Choice | `ST Origin` | Bottom Left (default, Flame's native), Top Left |
| `filter` | Choice | `Filter` | Nearest, Bilinear (default), Catmull-Rom |
| `edges` | Choice | `Edges` | Black (default), Clamp, Mirror |
| `device` | Choice | `Device` | Auto (default), GPU, CPU |
| `threads` | Int | `Threads` | ORT intra-op cap |
| `cacheMB` | Int | `Cache MB` | flow cache budget |
| `analyze` | Push | `Analyze` | walk the range with the progress suite, pre-warm the cache |
| `clearCache` | Push | `Clear` | drop the cache |
| `modelDir` | String | `Model Dir` | file-path string mode set via `propSetString(..., false)`, never `setStringType()` |

---

## Plugin description

```
Identifier   com.mtifilm.whitewater.opticalflow
Contexts     General (two clips); Filter registered as ST-Map-only
Clips        Source (mandatory, RGBA/RGB), Insert (optional, RGBA/RGB), Output
Depths       byte, short, half, float
Tiles        false   — flow is inherently whole-frame; still handle a partial window defensively
MultiRes     false   — Flame does not support it
Temporal     true    — on the effect and on the Source clip
ThreadSafety eRenderInstanceSafe; setHostFrameThreading(false)
OpenGL       propSetString(kOfxImageEffectPropOpenGLRenderSupported, "false", 0, false)
```

- **`getFramesNeeded`** declares only `{N}` on both clips; chain frames are pulled with
  `clipGetImage`. Declaring `[R..N]` would invite Flame to materialise hundreds of upstream
  frames. This is the assumption Phase 0 item 2 exists to verify.
- **`getRegionsOfInterest`** requests the complete connected RoD of both clips.
- **`isIdentity`** returns Source when `N == R` and the Insert is disconnected in Composite
  mode, and when inference has failed. ST Map mode is never identity.
- **`getClipPreferences`** reports unpremultiplied output unconditionally; ST Map writes
  `A = 1`. Making premultiplication depend on a parameter is legal OFX but untested in Flame.
- **Render** splits rows across `OFX::MultiThread` workers exactly as warp-drive does,
  with an abort check every 16 rows and a `FlowWarpMap` fully built before threads start.

---

## Build and packaging

Bundle layout (from `OfxBundle.cmake`), extended with two directories:

```
WhiteWater.ofx.bundle/Contents/
  Info.plist              (macOS)
  Linux-x86-64/WhiteWater.ofx
  MacOS/WhiteWater.ofx
  Resources/              PNG icons only — Flame reads nothing else
  Resources/models/       *.onnx
  Libraries/              libonnxruntime + providers
```

> **This layout is wrong for the GPU build and must be revised before Phase 3.** Measured
> 2026-08-20 on the box: Mocha Pro 2026.5's equivalent payload is **3.11 GB** — CUDA, cuDNN
> and `libonnxruntime_providers_cuda.so` (775 MB by itself). The estimate elsewhere in this
> plan of "roughly 20-80 MB" is right for model *weights* and three orders of magnitude out
> for the runtime that dominates.
>
> Boris FX's answer is worth copying: a sibling tree next to the bundle
> (`/usr/OFX/Plugins/BorisFX/MochaPro2026.5/Resources/…`) rather than inside it, so the
> bundle stays a bundle and the runtime is an installed component. Also likely: a small
> CPU-only default artifact with GPU as a separate download. And check first whether a
> current ONNX Runtime shrinks this materially — 1.22 made cuDNN and cuFFT optional at
> runtime for the CUDA EP specifically to cut this footprint, and Mocha's build may predate
> that. See `docs/host-notes.md`, *Measured — Mocha Pro's ML architecture*.

- **Linux**: build in a `rockylinux:9` container (glibc 2.34, matching the stated target).
  `-static-libstdc++ -static-libgcc`. Gate with `objdump -T`. Verify the bundled ORT libraries
  resolve under `env -u LD_LIBRARY_PATH ldd` — warp-drive lost a release to a dropped ICU that
  passed CI and failed at 127 on the host. *If EL8 Flame boxes must also be supported, the
  container drops to `almalinux:8` and the ORT build must satisfy glibc 2.28.*
- **macOS**: arm64 only, deployment target 12.0, `lipo -archs` asserted. Every `install_name_tool`
  rewrite invalidates the signature and **arm64 SIGKILLs an invalidly-signed image before
  `main()`** — re-sign every rewritten dylib ad-hoc, following `warp-drive/cmake/DeployEditor.cmake`.
  A new `cmake/DeployOnnxRuntime.cmake` does for ORT what that file does for Qt.
- Version script exports only `OfxGetNumberOfPlugins`, `OfxGetPlugin`, `OfxSetHost`;
  `-Wl,--no-undefined`; entry points carry explicit `visibility("default")`.
- `models/export_raft.py`, `models/export_rife.py` with pinned checkpoints and SHA256s.
  Weights are staged into the bundle at build time, not committed. `docs/models.md` records
  provenance and licences — **RAFT is BSD-3 (princeton-vl), RIFE and Practical-RIFE are MIT
  including the weights**; both are clean for commercial use, and this must be re-verified
  against the exact checkpoints shipped.
- CI mirrors warp-drive: `workflow_dispatch` only, with a required `purpose` input naming the
  human test. The Flame box is airgapped; artifacts are tarballs carried over with a SHA256.

---

## Verification

**Host-free, runs everywhere, no weights and no GPU** — via `NullEstimator`:

- `compose(a, b)` against an analytic composition; `compose(a, identity) == a` exactly.
- A chain of `k` identity links is exactly identity, for `k` up to a few hundred.
- Forward-backward check returns zero residual on an invertible synthetic field.
- ST map round trip: absolute-UV output fed back through `Resampler` reproduces the warped
  image bit-for-bit.
- `Preprocess`: pad-to-multiple then crop is exact; premultiply-by-matte matches a reference.
- Layering gate: `src/core` must not include OFX, ONNX Runtime, or any I/O header.

**Offline CLI** `tools/ww-flow`: two images → flow, ST map, or warped result, as PFM.
Golden test — a synthetically translated noise plate must recover the translation to
sub-pixel tolerance, with the real RAFT weights when present and skipped when not.

**Host harness** (`tests/hostharness`, extended for a General context with two clips and for
serving `clipGetImage` at arbitrary times):

- Identity across `{byte, short, half, float} × {RGBA, RGB, Alpha}`, whole-frame and tiled.
- At `N == R` with the Insert disconnected, Composite output is the Source bit-for-bit.
- At `N == R` with the Insert connected, output equals a reference `over`, bit-for-bit.
- ST Map at `N == R` is the exact normalized pixel grid, in both origin conventions.
- Multi-link chains exercised through the harness's arbitrary-time clip service.
- The plugin loads under `--host-name com.autodesk.flame` so the Flame quirk branch is walked
  by something other than Flame, and under a host that refuses the file-path string mode.

**On the box**: run the Phase 0 probe first; then the plugin in a Batch node with a real plate
and insert; check `/opt/Autodesk/log/` for the plugin's stderr, which is the only diagnostic
channel on a machine nobody can attach a debugger to.

**Performance gate**: 1080p and 4K, RAFT at half analysis scale, ms/frame recorded on the
target box before any optimisation work, with a regression threshold.

---

## Phasing

| Phase | Content | Exit |
|---|---|---|
| **0** | Extended `hostprobe`, run in Flame | `docs/host-notes.md` answers the five Phase 0 questions |
| **1** | Vendor, CMake, bundle, harness, `describe`/`describeInContext`, passthrough render | Plugin loads in Flame with two inputs; every parameter is visible and legible |
| **2** | `src/core/flow` complete, `NullEstimator`, `ww-flow`, full unit + harness coverage | All host-free tests green; ST map and comp correct with synthetic flow |
| **3** | ONNX Runtime, `ModelRegistry`, `OrtEnvironment`, RAFT, model export, library bundling | `ww-flow` recovers a synthetic translation with real RAFT weights, on GPU and CPU |
| **4** | `FlowPreparation` wired into render; on-demand pulls, abort, progress, Analyze/Clear, persistent messages | A real shot tracks in Flame from a reference frame |
| **5** | RIFE behind `FlowEstimator` | Model parameter switches cleanly; both paths covered |
| **6** | FB check, smoothing, input curves, perf gate, CI packaging | Perf threshold recorded; tarball installs on the airgapped box |

---

## Deferred, deliberately

- Third matte mode ("matte as flow confidence") — cheap to add on top of `fbCheck`'s
  confidence plumbing, but the user asked for two modes.
- Disk-backed flow cache surviving a Flame restart.
- Display-only overlay showing the reference frame and chain state. Flame does deliver pen
  events and the `OfxDrawSuite`, but **never keyboard events**, and `kOfxInteractPropPixelScale`
  reads `[1,1]` at every zoom — so any overlay must be display-only and zoom-independent.
- Drift mitigation past a chain-length threshold.
- An OpenGL render path — the only GPU option OFX exposes in Flame, and irrelevant while
  inference dominates the frame time.
