# White Water — ML optical flow tracking OFX plugin for Flame

## Context

Flame's built-in motion vector tracking is a classical solver. Shots with motion blur,
low contrast, non-rigid deformation or fine detail defeat it, and the artist falls back
to hand-tracking or a roundtrip out of Flame. `white-water` is a new OpenFX plugin that
replaces that solver with modern learned optical flow, keeping the familiar Flame workflow: analyse a source plate, position an insert
at a reference frame, let the vectors carry it — or hand a compositor an ST map and let
them do the warp downstream.

Targets are Rocky Linux 9.5+ and arm64 macOS, host is Autodesk Flame.

This plan was written against an empty repository and has been amended since — most
substantially on 2026-08-20, after Phase 0A closed and after the review in
`docs/architecture-review-2026-08-20.md`. `docs/context.md` records why each change was made.
Everything below is new, except where it is deliberately vendored from `/Users/andrew/repos/warp-drive`, whose `docs/host-notes.md`
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
| Analysis trigger | **On-demand caching by default, plus an explicit `Precache` button with progress.** **v1 is RAM-only, no persistence** (decided 2026-08-22): 0C measured final render as a separate process, but the facility renders almost everything in the foreground and uses single-node Burn rarely, so RAM covers it. A durable disk cache is a future option gated on Burn renders fanning out across a farm |
| Model | **One selected artifact behind `PairwiseFlowEstimator`; selectable only if more than one shipping model qualifies.** The default is chosen by the Phase 2.5 bake-off — *not named in advance*, because choice index is API. A one-model release is valid; a fast alternative is added only if it passes every gate. SEA-RAFT is the leading candidate |
| ST map | **Its own float-only plugin descriptor**, not an output mode of the main effect — see *Two descriptors*. Absolute normalized UV (default) or relative pixel offset; origin toggle |
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
- **`SupportsMultipleClipDepths = 0`** — measured in four separate probe transcripts. Per
  `third_party/openfx/Documentation/sources/Reference/ofxClipPreferences.rst`, a host
  reporting 0 gives every clip on the effect one common depth and **the plugin may not remap
  them**. There is no negotiating float output from a byte source. This is why the ST map is
  a separate float-only descriptor rather than a mode of the main effect.
- **`eRenderInstanceSafe` means one render per instance at a time**, and multiple *instances*
  concurrently — `ofxsImageEffect.h:94`. `eRenderFullySafe` is the level that would permit
  concurrent renders of one instance, and we do not declare it. So per-instance state needs
  no same-instance render de-duplication; **process-wide state does need to be thread-safe**,
  because two instances may render at once.
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

Nothing else in this plan was worth writing until these were measured on the actual Flame
box. Extend a vendored copy of `warp-drive/tools/hostprobe/hostprobe.cpp` (raw C API,
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

- **Item 2 — ANSWERED 2026-08-20, in our favour.** Had it failed, the fallback was a
  mandatory analysis pass writing a disk cache, driven from an instance-changed action, which
  warp-drive measured to work.
- **Item 3 — ANSWERED 2026-08-20, in our favour.** Inference runs **in-process**, with the
  bundled runtime opened by `dlopen` under plain `RTLD_LOCAL`. Measured: our ONNX Runtime
  1.29.0 loaded alongside Flame's 1.22.0, kept its own identity, and ran. `RTLD_DEEPBIND`
  also works but is deliberately *not* used — it breaks malloc interposition and exception
  unwinding, and paying that to solve an already-solved problem would be a poor trade. The
  out-of-process design (warp-drive's `EditorProcess` plus `src/ipc/`, the shape Mocha Pro
  ships) is retained as a documented fallback rather than the plan. See
  `docs/host-notes.md`. The 2026-08-21 SEA-RAFT M run also passed through the CUDA
  execution provider, and the follow-up live-loader-path report closed exact payload
  ownership and size accounting. The bounded high-resolution runs subsequently completed
  the remaining 0B qualification measurement; none of the three full-resolution targets fit
  the 16 GiB ORT arena ceiling, which is a measured configuration result rather than a product
  resolution limit.

### 0A — closed 2026-08-20

All five questions answered on Flame 2026.2 / Rocky 9.5. `docs/host-notes.md` holds the
measured record; `docs/measurements/` holds the raw transcripts.

### 0B — closed 2026-08-21

**SEA-RAFT M is the 0B probe network.** This names a representative candidate for the
measurement; it does not make SEA-RAFT the shipping default or assign it a choice index.
Before the host run, 0B adds a pinned export script and manifest recording the upstream
commit, checkpoint URL and SHA256, tensor contract, exported ONNX SHA256, and a synthetic
translation test that checks output identity and direction. The resulting real network —
not the 128-byte `Add` model — runs through the private ONNX Runtime inside Flame, on CPU
and on the CUDA EP:

- output identity and *direction* checked numerically, not just "it ran";
- which CUDA/cuDNN/cuBLAS libraries actually get selected, and by whom;
- provider init and first-run latency; peak and steady VRAM beside a live Batch;
- repeated create/run/destroy, and node duplication;
- cancellation from another thread via a per-run `RunOptions`;
- provider-init fallback, plus a bounded recovery proxy for GPU allocation failure;
- qualify one higher production resolution per Flame launch through a separate GPU-only
  path and a fresh CUDA session. The Phase 0B targets are source H×W 2160×3840 (UHD),
  2160×4096 (DCI 4K), and 3164×4608 (Alexa 35 open gate). The last is replication-padded
  on the bottom to a 3168×4608 network tensor to satisfy the export's multiple-of-eight
  contract, while reporting both source and tensor dimensions. The path skips the ordinary
  CPU/useful-size measurements, uses heuristic cuDNN search and `kSameAsRequested` arena
  growth, requires an explicit probe-only `gpu_mem_limit`, samples device-wide NVML use at
  operation boundaries, and serializes attempts inside the process. A successful inference
  and an explicit bounded-arena stop are distinct reported outcomes. These three diagnostic
  targets and the current 16 GiB arena ceiling are qualification guardrails, not product
  resolution limits;
- **Measured 2026-08-21:** exercise a controlled CUDA-arena limit by setting
  `OrtCUDAProviderOptions::gpu_mem_limit` to a fixed 64 MiB, attempting the real SEA-RAFT M
  at 480×640, requiring a recognisable allocator-limit failure at session creation or first
  run, tearing down that limited CUDA attempt, and numerically checking recovery through a
  fresh CPU session at the probe's 128×192 identity/direction size. The limit is not a
  device-wide allocation fence and this does not test an automatic production fallback.
  The probe records the failure stage/kind and device-wide NVML use before/after, skips the
  exercise unless baseline CUDA inference passed cleanly, skips it if a cancellation timeout
  retained resources, and can gate the action with
  `WHITEWATER_ORT_REQUIRE_GPU_MEM_LIMIT=1`. On the target Flame 2026.2 host it failed during
  `CreateSession` with an explicit `BFCArena` limit diagnostic, returned device-wide NVML use
  to the pre-test value, passed fresh-session numerical CPU recovery, and passed the required
  gate. This is recovery evidence, not automatic production fallback; that shipping behavior
  remains Phase 4;
- **Measured 2026-08-21:** the exact dependency closure and on-disk size of the chosen CUDA
  build. Under Flame's live loader search path all dependencies resolve, the four CUDA
  SONAMEs resolve to Flame 2026.2.1, and the diagnostic payload is 646,116,476 apparent
  bytes or 614,157,572 bytes without the probe-only second ORT copy. The unique external
  closure is 1,138,007,368 bytes and is ownership accounting rather than bundle content;
- **Measured 2026-08-21:** the GPU-only high-resolution path under a 16 GiB ORT arena limit.
  UHD 2160×3840, DCI 4K 2160×4096, and Alexa 35 open gate 3164×4608 all reached a classified
  `bounded-allocation-stop` during the warm `Run`. The Alexa input was correctly
  replication-padded on the bottom to a 3168×4608 tensor. All three required measurement-
  result gates passed, session cleanup returned the boundary-sampled whole-device NVML value
  to +2 MiB of its pre-session value, and no steady samples were produced. This closes the
  qualification measurement negatively for the current 16 GiB arena configuration; it is
  not a hard plugin resolution cap or proof about a larger future GPU budget.

**Measured 2026-08-21:** the pinned SEA-RAFT M export passed identity and both translation
directions on CPU and CUDA inside Flame 2026.2 through private ORT 1.29 under plain
`RTLD_LOCAL`. At 128×192, CPU session/first-run timings were 941.6/529.4 ms with 0.0026 px
identity EPE; CUDA timings were 934.8/1164.0 ms with 0.0027 px identity EPE. Flame's
pre-existing CUDA, cuBLAS and cuDNN libraries were mapped alongside the probe's private ORT
CUDA providers; cuDNN component libraries appeared after provider session/run. The raw report
and ORT warnings are archived in
`docs/measurements/2026-08-21-ortprobe-sea-raft-m-flame.txt` and
`docs/measurements/2026-08-21-ortprobe-cuda-warnings.txt`.

The follow-up multiresolution reports supplied on 2026-08-21 add warmed CPU and CUDA runs at
480×640, 720×1280 and 1080×1920, plus repeated lifecycle, cancellation, provider-init
fallback and duplicate-node comparisons. The host-notes entry records their measured values
and scope; the supplied reports are not archived in this repository. These results close
those checks for the reported run, but do not close production-resolution qualification
above 1080p.

The controlled arena-limit transcript is archived in
`docs/measurements/2026-08-21-ortprobe-gpu-mem-limit-flame.txt`; the result closes that
bounded recovery measurement without making a claim about physical device-wide OOM or
automatic production fallback.

The live-loader-path closure report resolves `libcublas.so.12`, `libcublasLt.so.12`,
`libcudart.so.12` and `libcurand.so.10` to Flame 2026.2.1 and reports no unresolved
dependencies. This closes the Phase 0B dependency ownership and size question for the
measured host/runtime pair. The CUDA log's nine inserted `Memcpy` nodes and CPU-assigned
shape operations are performance warnings, not correctness failures.

The three high-resolution qualification reports close the last 0B item. Under the configured
16 GiB ORT arena ceiling, UHD stopped at `FusedMatMul` with 3,095,502,080 bytes available for
a 16,796,160,000-byte request; DCI 4K stopped with 1,423,074,560 available for a
19,110,297,600-byte request; and padded Alexa 35 stopped with 4,298,024,192 available for a
13,006,946,304-byte request. These are allocator diagnostics at different failure points and
must not be treated as comparable total-memory estimates. The 704.8–866.7 ms warm-attempt
durations are time to failure, not inference performance, and the boundary-sampled NVML peaks
do not capture the rejected allocation. Phase 2.5 and the shipping performance gate still
choose model/settings and practical megapixel caps on measured hardware; they do not reject
larger source formats at the plugin boundary.

### 0C — before ST map and cache integration

Two questions, both cheap, both answerable with probe extensions:

1. **The ST convention Flame's own downstream tool expects. — MEASURED 2026-08-21.** Both
   Flame's native ST Map node and Action's UV map, via `tools/stprobe/`, are identical: pixel
   centres map as `(x + 0.5) / width` (fit residual 0.000 px), bottom-left origin, U→R/V→G,
   real-pixel normalization, backward-map semantics. Out-of-range differs — the ST Map node
   blacks, Action mirrors. RoD-vs-project-extent normalization is left open (needs an
   undersized/offset source; out of v1 scope). See `docs/host-notes.md`, *Measured — Phase 0C
   item 1*. This settles the `stOrigin` default and the `StMap.{h,cpp}` encoding.
2. **Instance and process lifetime. — MEASURED 2026-08-21.** Background/final render runs in a
   **separate process — Autodesk Burn** (`com.autodesk.backgroundreactor`, `IsBackground=1`),
   a different pid from the Flame foreground session, and it rendered the full range.
   Duplicating the node = new instance, same process; reopening Flame = new process. **So a
   RAM-only `Precache` has no production value for the final render.** See `docs/host-notes.md`,
   *Measured — Phase 0C items 2-5*. **Decided 2026-08-22 (see *Deferred*): v1 ships a RAM-only
   `Precache` and no persistence** — the facility renders almost everything in the foreground and
   uses single-node Burn rarely, so RAM covers it; a durable disk cache is a future option gated
   on Burn renders fanning out across a farm. (Items 3, 4 and 5 also closed, at PAR 1 and PAR 2.
   **All of Phase 0C, and Phase 0, is now closed.**)

*The depth half of question 1 is already answered:* Flame reports
`SupportsMultipleClipDepths = 0`, so no depth negotiation is possible and the ST descriptor
must declare float only.

---

## Architecture

Three layers, enforced mechanically by a vendored `scripts/check-core-dependencies.cmake`:

```
src/core/    host-free, no OFX, no ONNX Runtime, no I/O   → unit-testable
src/infer/   ONNX Runtime, no OFX                          → testable with a CLI
src/ofx/     the plugin; one TU (Plugin.cpp) in the module
```

**Both boundaries are now gated.** Phase 1 gave the dependency script separate `src/core`
and `src/infer` invocations with distinct allow-lists: core rejects OFX, ONNX Runtime and I/O;
infer rejects OFX while allowing inference dependencies. Each policy has a negative fixture
test that must fail for the expected diagnostic, so the gate is known to reject the dependency
it forbids rather than merely passing the current tree. The plan as first written broke this
boundary by handing `src/infer` a type from `src/ofx`; `OwnedFrame` now lives in `src/core`.

### Vendored verbatim from warp-drive

Copy with provenance headers, renamespaced `warpdrive` → `whitewater`:

| Source | Why |
|---|---|
| `src/core/geom/Vec2.h`, `src/core/image/Image.h` | float RGBA view with row stride, bottom-left origin |
| `src/core/warp/WarpMap.h` | the `mapToSource(Vec2) → Vec2` interface a dense flow field implements directly |
| `src/core/warp/Resampler.{h,cpp}` | backward warp with correct unpremultiplied filtering, edge modes, row-range threading, bit-exact identity |
| `src/ofx/HostImage.{h,cpp}`, `PixelFormat.{h,cpp}` | zero-copy float RGBA borrow, depth/component conversion incl. hand-written half, negative `rowBytes` |
| `src/ofx/FrameCapture.{h,cpp}` | the OFX image lifetime and conversion. **The owned frame value it produces moves down to `src/core/image/OwnedFrame.h`** — `FrameCapture.h` includes `ofxsImageEffect.h`, so leaving the value there drags OFX into `src/infer` |
| `src/ofx/HostQuirks.{h,cpp}` | per-host quirks keyed on host name, dumped to stderr at `load()` |
| `cmake/OfxBundle.cmake`, `cmake/ofx.map`, `cmake/Info.plist.in` | bundle layout, `$ORIGIN`/`@loader_path` rpath, three-symbol version script |
| `scripts/check-core-dependencies.cmake`, `check-glibc-baseline.sh` | layering and ABI gates |
| `tests/hostharness/` | minimal OFX host that actually renders |

`Resampler`'s `WarpMap` abstraction is the key reuse: a dense flow field is a `WarpMap`
subclass and the entire correct-alpha, correct-edge, threaded backward warp comes for free.

### New — `src/core/flow/`

- **`Field.h`** — dense `kChannels` field over a lattice, **float** storage (not half — see
  `docs/context.md`), bilinear sample. `Field<2>` and `Field<1>` are *storage*, not the public
  unit of chain algebra.
- **`FieldGeometry`** — where a field's nodes sit, as **`origin`, `spacingX`, `spacingY`**.
  A single scalar spacing cannot express an anamorphic plate, an odd extent reduced by a
  fraction, or asymmetric rounding, and this project has already lost two rebuild cycles to
  comparing quantities across coordinate spaces (`docs/context.md`, corrections 4 and 5).
  Deliberately *separable and no more*: a general affine would admit rotation and shear that
  nothing here produces, and then every consumer carries the general case forever. Checked at
  construction — zero, negative, NaN and infinite spacing all reach a division in `sample()`.
- **`FlowLink.{h,cpp}`** — the public unit: `fromTime`, `toTime`, source and destination
  geometry, the backward displacement, and a model fingerprint. `compose` **rejects**
  mismatched endpoints or geometry rather than trusting the caller. Direction errors are the
  most likely bug in this design precisely because a reversed chain still produces plausible
  motion; putting the endpoints in the type is what makes them checkable.
  *Confidence is stored separately*, as a parallel `Field<1>` under the same cache key — it is
  optional, toggleable, and recomputable without invalidating the flow it accompanies.
- **`FlowCompose.{h,cpp}`** — `compose(a, b)(q) = a(q) + b(q + a(q))`; forward-backward
  consistency; confidence propagation along the composed path; Gaussian smoothing.
- **`FlowChain.{h,cpp}`** — the reference-frame accumulation policy (below). Pure: it decides
  *which links are needed in which direction*, and takes cached ones as input.
- **`FlowCache.{h,cpp}`** — byte-budgeted LRU, **instance-lifetime** (see *Cache ownership*),
  keyed on `(fromFrame, toFrame, generation, model SHA256, model geometry, matteMode,
  inputCurve, modelParams, cache schema version)`.
- **`StMap.{h,cpp}`** — field → absolute normalized UV or relative pixel offset, bottom-left
  or top-left origin.
- **`Composite.{h,cpp}`** — the `over` operator with explicit premultiplication handling.
- **`Preprocess.{h,cpp}`** — premultiply-by-matte, input curve, downscale, reflect-pad to the
  model's required multiple, crop back.

### New — `src/infer/`

- **`PairwiseFlowEstimator.h`** — `estimate(const OwnedFrame &a, const OwnedFrame &b,
  const FlowRequest &) → FlowResult`. This is deliberately the narrow v1 contract used by
  the pairwise chain. `NullPairwiseEstimator` synthesises deterministic analytic flow so
  every test above this line runs with no weights and no GPU. A model such as AllTracker
  consumes a temporal window and returns several reference-relative fields; pretending that
  fits an `estimate(a, b)` call would discard the property that makes it useful. If that
  model class becomes viable, add a separate `ReferenceFlowEstimator` contract and let
  `FlowPreparation` select a chain or reference strategy. Keep the chain orchestration out
  of the pairwise estimator so that replacement remains local.
- **`ModelSpec.{h,cpp}`** — the per-model tensor contract as data: input/output names,
  normalization, pad multiple, iteration handling, output layout. The per-model numbers land
  at Phase 2.5, with the checkpoints they were measured against.
- **`ModelRegistry.{h,cpp}`** — resolve weights: `Model Dir` param → `WHITEWATER_MODEL_DIR`
  env → `Contents/Resources/models/` in the bundle. Report which one won, to stderr.
- **`RuntimeLoader.{h,cpp}`** — opens the bundled runtime by absolute path under
  `RTLD_LOCAL`, resolves `OrtGetApiBase` from *that handle*, and initialises the C++ wrapper
  with **`ORT_API_MANUAL_INIT`**. Without that define the wrapper's global API pointer
  initialises through normal linkage and quietly defeats the entire isolation measured in
  Phase 0 — a one-line omission with a process-wide blast radius. The handle stays alive
  until process exit; never `dlclose` while sessions, provider code, TLS or static wrapper
  objects may outlive it.
- **`OrtEnvironment.{h,cpp}`** — one process-wide `Ort::Env`; a mutex-guarded session cache
  keyed on `(model SHA256, execution provider, runtime ABI)` because session creation costs
  seconds and must never happen per frame. **Not keyed on resolution** unless the exported
  model actually has fixed shapes — with dynamic shapes it is one immutable session plus a
  small pool of per-shape I/O bindings. EP selection with explicit fallback; bounded intra-op
  thread count (Flame owns all 72 CPUs and an unbounded ORT pool will fight it); a per-run
  `RunOptions` with `SetTerminate` wired to the host's abort. **Serialize GPU inference
  behind a semaphore initially** — throughput matters less than not spiking VRAM into a
  Flame that is also holding a Batch. This layer is process-wide and therefore genuinely
  concurrent: two *instances* may render at once even though one instance may not.
- **`OnnxPairwiseFlowEstimator`** — one data-driven implementation parameterised by
  `ModelSpec`. Do not create model-named estimator classes when tensor contracts are the only
  difference; model-specific code is justified only when an export genuinely needs distinct
  execution or post-processing.

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
case is one new inference plus one compose. Fields are stored at model resolution as **float**
2-channel: a 1080p half-res link is **4.1 MB**, a 4K half-res link about 17 MB, before
allocator overhead, confidence and the inference tensors themselves. (An earlier draft said
2 MB, which was true under half storage and was not updated when `Field.h` chose float.)
Capturing two UHD RGBA float frames is another ~253 MiB, so chain links are processed as a
rolling pair and captured frames released promptly.

### Cache ownership

Two caches with different lifetimes and different correctness rules. They are not one object.

```
process lifetime          instance lifetime
  RuntimeLoader             generation counter
  Ort::Env                  pairwise link LRU + accumulated-field LRU
  session cache             (no durable namespace in v1 — decided 2026-08-22)
  GPU semaphore
```

The "optional durable namespace" this diagram once carried is **not built in v1**: 0C decided
the flow cache stays RAM-only and instance-lifetime (see *Deferred* and `docs/context.md`).

- On `changedClip`, bump the generation and invalidate every flow result. OFX offers no
  trustworthy content hash for an upstream graph, so no amount of extra key material makes a
  process-global flow cache correct — the generation bump is the honest mechanism.
- On a flow-affecting `changedParam`, prefer a generation bump over a clever partial
  invalidation that can go stale.
- **Never hold a cache mutex across `clipGetImage` or `Session::Run`.** That is the shape of
  thing that deadlocks a host.
- A result computed in an old generation must not publish into a new one.

**Drift** is inherent to accumulation and is not solved in v1. It is **not** mitigated by the
`Smooth` parameter — that was wrong in an earlier draft of this plan. Smoothing addresses
local spatial noise; drift is accumulated systematic bias along the temporal axis, and
blurring the field additionally softens motion boundaries, which makes foreground/background
leakage worse. What actually bounds drift in v1 is keeping the reference frame near the
working range. Past a chain-length threshold, a `Max Chain` parameter can fall back to a
direct `N→R` inference — noted, not built.

---

## Parameters

Flat list, order is the only layout tool, labels ≤ 12 characters. **Choice option order is
API** — a saved setup stores the index, so options are only ever appended. **No `setEnabled()`
anywhere**: a host that refuses `kOfxParamPropEnabled` turns that into a throw out of an
action, and in Flame that means a plugin that simply is not there.

| Script name | Type | Label | Notes |
|---|---|---|---|
| `model` | Choice | `Model` | Present only if more than one shipping model passes Phase 2.5. Options and order are then fixed by the bake-off. **Do not guess an index here** — a saved setup stores it |
| `refFrame` | Int | `Ref Frame` | OFX time — 0-based within the Flame batch |
| `setRef` | Push | `Set Ref` | writes `args.time` into `refFrame`; the only honest way for an artist to set it given Flame's 0-based time |
| `output` | Choice | `Output` | Composite (default), Warped Insert. **ST Map is not here** — it is a separate float-only descriptor |
| `insertTime` | Choice | `Insert At` | Current (default) — the insert advances normally while its canvas is carried by `B(N→R)`; Reference — the insert is frozen at `R`. Both are legitimate and they differ completely for an animated insert, so it is a parameter rather than a guess. Determines what `getFramesNeeded` declares for the Insert clip |
| `matte` | Choice | `Matte` | Premultiply (default), Full Frame |
| `inputCurve` | Choice | `Input` | Options fixed by the Phase 2.5 bake-off. **"Filmic" is struck** — it was a name without a defined transform, and a tensor contract has to be reproducible. Clamp-to-0..1 also erases the highlight texture needed to track specular and motion-blurred footage, so it cannot simply be the default either. Candidates in *Input conditioning* below |
| `analysisScale` | Choice | `Analysis` | A **megapixel cap**, not a fraction. A fraction of a PAR-normalized image is not PAR-independent: a 2048×1556 PAR 2 plate is 4096×1556 canonical, and "Half" of that is 2048×778 model pixels against 1024×778 for the PAR 1 equivalent — still 2×, and it presents as an unexplained OOM on anamorphic shots only. One number the artist can read is also one number VRAM can be planned against |
| `iterations` | Int | `Iters` | refinement iterations, where the selected model has them |
| `smooth` | Double | `Smooth` | post-flow spatial smoothing, 0 default. **Does not address drift** — see *The flow chain* |
| `fbCheck` | Bool | `FB Check` | off; 2× inference cost |
| `fbTolerance` | Double | `FB Tol` | round-trip tolerance in pixels |
| `filter` | Choice | `Filter` | Nearest, Bilinear (default), Catmull-Rom |
| `edges` | Choice | `Edges` | Black (default), Clamp, Mirror |
| `device` | Choice | `Device` | Auto (default), GPU, CPU |
| `threads` | Int | `Threads` | ORT intra-op cap |
| `cacheMB` | Int | `Cache MB` | flow cache budget |
| `precacheRange` | Choice | `Pre Range` | Current-to-Ref (default), Work Range, Custom. **Never the full source range** — "walk the range" was undefined and could mean thousands of frames |
| `precacheStart` | Int | `Pre Start` | first frame when `precacheRange` is Custom; always visible because `setEnabled()` is forbidden |
| `precacheEnd` | Int | `Pre End` | last frame when `precacheRange` is Custom; order is normalized before the walk |
| `precache` | Push | `Precache` | walk `precacheRange` with the progress suite, filling the **instance-lifetime RAM** cache. Named `Precache` rather than `Analyze` because a RAM-only pre-warm is honestly what it is — confirmed by the 2026-08-22 no-persistence decision, not durable |
| `clearCache` | Push | `Clear` | drop the (RAM) cache |
| `modelDir` | String | `Model Dir` | file-path string mode set via `propSetString(..., false)`, never `setStringType()` |

`output` and `insertTime` exist only on the Track/Insert descriptor. The ST Map descriptor
carries everything above except those two, plus:

| Script name | Type | Label | Notes |
|---|---|---|---|
| `stMode` | Choice | `ST Mode` | Absolute UV (default), Relative Pixels. **Relative is signed and leaves `[0, 1]`**, which is half the reason this descriptor is float-only |
| `stOrigin` | Choice | `ST Origin` | Bottom Left (default, Flame's native), Top Left |

Both conventions are unverified against Flame's own downstream ST tool until 0C. Nothing here
is settled by a round trip through our own resampler.

---

## Plugin description

### Two descriptors, sharing one implementation

The original "General with two clips; Filter registered as ST-Map-only" had no workable UI
contract: the same flat parameter list would show Composite and Warped Insert in a context
with no Insert, and **`setEnabled()` is forbidden here**, so they cannot be hidden. Flame also
reports `SupportsMultipleClipDepths = 0`, so a single effect cannot serve float ST output
from a byte source. Two descriptors resolve both at once.

```
Track/Insert                         ST Map
  com.mtifilm.whitewater.opticalflow   com.mtifilm.whitewater.stmap
  General                              General
  Source (req), Insert (opt), Output   Source (req), Output
  byte, short, half, float             float only
  Output: Composite | Warped Insert    ST Mode, ST Origin
```

Both: `Tiles false` (flow is whole-frame; still handle a partial window defensively),
`MultiRes false`, `Temporal true` on the effect and the Source clip, `eRenderInstanceSafe`
with `setHostFrameThreading(false)`, and OpenGL declined via
`propSetString(kOfxImageEffectPropOpenGLRenderSupported, "false", 0, false)`.

Declaring float-only on the ST descriptor makes Flame map its input to float too, which is
correct and costs nothing we were not already paying — inference needs float32 tensors
regardless. Declaring float across the *whole* plugin would instead force float buffers for a
byte source in Composite mode as well: 4× the memory on 4K, for a mode that does not need it.
Two descriptors confine the float cost to the output that requires the precision.

Both identifiers were made permanent in Phase 1, before the first artist build, so saved setups
never observe a temporary combined descriptor.

- **`getFramesNeeded`** declares `{N}` on Source, and `{N}` or `{R}` on Insert per
  `insertTime`; chain frames are pulled with `clipGetImage`. Declaring `[R..N]` would invite
  Flame to materialise hundreds of upstream frames. Measured working in Phase 0.
- **`getRegionsOfInterest`** requests the complete connected RoD of both clips.
- **`isIdentity` must stay cheap and deterministic.** Flame calls query actions far more often
  than render — 256 `getFramesNeeded` calls against 47 renders, measured. It answers from
  parameters, time and clip connection state only. **It must never run inference to discover
  that inference will fail**; a render-time failure renders its own documented fallback.
- **Documented failure output**, because these are artist-visible semantics rather than an
  error policy: Composite → Source at `N`; Warped Insert → the unwarped Insert at the selected
  insert time, or transparent black if disconnected; ST Map → an identity ST map. Every one
  logged as a fallback. `R` is clamped to the measured source range before a chain is built —
  Flame *clamps* rather than failing on out-of-range pulls, so an unclamped `R` yields
  plausible-looking zero-motion links rather than an error.
- **`getClipPreferences`** reports unpremultiplied output unconditionally. Making
  premultiplication depend on a parameter is legal OFX but untested in Flame.
- **Render** splits rows across `OFX::MultiThread` workers exactly as warp-drive does,
  with an abort check every 16 rows and a `FlowWarpMap` fully built before threads start.

### Anamorphic policy

**Analyse in square pixels.** Resample the plate to canonical geometry before inference, then
map vectors back to source pixel units independently on X and Y — which is what `spacingX`
and `spacingY` exist for. The networks are trained on square-pixel imagery; feeding a 2:1
squeeze distorts both the learned features and the displacement metric. Storage-pixel analysis
is cheaper and remains a legitimate measured trade, but it has to be chosen on quality
evidence rather than inherited from the buffer layout.

PAR 0.5 and 2.0, negative image origins, odd extents and non-zero bounds all belong in the
core test matrix from Phase 2.

### Input conditioning

Separate what the *model* requires from what the *artist* selects. For the bake-off, compare:

- model-native normalization after a hard 0..1 clamp;
- a fixed signed/log compression suited to scene-linear values;
- **shared pairwise percentile normalization** — one transform computed from both frames, never
  independent per-frame auto-exposure, which would inject apparent brightness change straight
  into the flow estimate;
- unmodified log input where the upstream plate is already log.

No OCIO dependency for this. The goal is stable features, not a display rendering. The exact
math goes in the cache key and the model manifest.

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

- **Linux**: build in an `almalinux:8` container against a **glibc 2.28** baseline, with
  `gcc-toolset-12`. This was originally specced as `rockylinux:9`/2.34 "matching the stated
  target"; that produced a plugin Flame refused to load, because Rocky 9.5's `libgcc.a`
  needs `_dl_find_object` (glibc 2.35) and the certified box — also nominally Rocky 9.5 —
  has an older glibc. See `docs/context.md`, correction 3. A 2.28 binary loads on EL8 and
  EL9 alike whatever point release, so this widens compatibility rather than narrowing it.
  `-static-libstdc++ -static-libgcc`. Gate with `objdump -T` against a **hard-coded**
  baseline, never one read off the build machine. Verify the bundled ORT libraries resolve
  under `env -u LD_LIBRARY_PATH ldd` — warp-drive lost a release to a dropped ICU that
  passed CI and failed at 127 on the host.
  **Phase 3 constraint:** the ONNX Runtime build must itself satisfy glibc 2.28.
- **macOS**: arm64 only, deployment target 12.0, `lipo -archs` asserted. Every `install_name_tool`
  rewrite invalidates the signature and **arm64 SIGKILLs an invalidly-signed image before
  `main()`** — re-sign every rewritten dylib ad-hoc, following `warp-drive/cmake/DeployEditor.cmake`.
  A new `cmake/DeployOnnxRuntime.cmake` does for ORT what that file does for Qt.
- Version script exports only `OfxGetNumberOfPlugins`, `OfxGetPlugin`, `OfxSetHost`;
  `-Wl,--no-undefined`; entry points carry explicit `visibility("default")`.
- One `models/export_<model>.py` per shipped model, with pinned checkpoints and SHA256s.
  Weights are staged into the bundle at build time, not committed. `models/MODELS.md` records
  provenance and licences. **Every licence claim is verified against the actual repository and
  the actual checkpoint file before it counts** — including backbone weights, which may carry
  different terms from the code that loads them, and secondhand claims from a review or a
  survey, which do not.
- Model files must be readable by Flame. The 0B staging failure that reported the ONNX model
  as absent was a present file installed mode `0600`, readable only by its owner while the
  Flame runtime user was distinct; correcting it to `0644` made the same probe pass. Packaging
  CI must assert mode `0644` (or an equivalent ACL for the Flame runtime user) so a mode bit
  cannot masquerade as a missing model again.
- CI mirrors warp-drive: `workflow_dispatch` only, with a required `purpose` input naming the
  human test. The Flame box is airgapped; artifacts are tarballs carried over with a SHA256.

---

## Verification

**Host-free, runs everywhere, no weights and no GPU** — via `NullPairwiseEstimator`:

- `compose(a, b)` against an analytic composition; `compose(a, identity) == a` exactly.
- A chain of `k` identity links is exactly identity, for `k` up to a few hundred.
- Forward-backward check returns zero residual on an invertible synthetic field.
- ST map round trip: absolute-UV output fed back through `Resampler` reproduces the warped
  image bit-for-bit.
- `Preprocess`: pad-to-multiple then crop is exact; premultiply-by-matte matches a reference.
- Layering gate: `src/core` must not include OFX, ONNX Runtime, or any I/O header, **and
  `src/infer` must not include OFX**.
- **Direction-labelled** constant translations in both temporal directions, including a test
  that deliberately swaps endpoints and must fail.
- Composition over affine and spatially varying fields, not only identity.
- **PAR 0.5 and 2**, odd dimensions, non-zero and negative bounds, asymmetric padding.
- `FieldGeometry` rejects zero, negative, NaN and infinite spacing at construction.
- Confidence propagation across a multi-link chain and at an out-of-bounds sample.
- Cache: a generation bump while an old inference is in flight — the stale result must not
  publish. Abort before a frame pull, during the precache loop, and during `Run`.
- Failure paths: provider init failure, GPU OOM, missing model, bad model hash, incompatible
  tensor contract, CPU fallback.

**Offline CLI** `tools/ww-flow`: two images → flow, ST map, or warped result, as PFM.
Golden test — a synthetically translated noise plate must recover the translation to
sub-pixel tolerance, with a pinned candidate ONNX artifact when present and skipped when not.

**Host harness** (`tests/hostharness`, extended for a General context with two clips and for
serving `clipGetImage` at arbitrary times):

- Identity across `{byte, short, half, float} × {RGBA, RGB, Alpha}`, whole-frame and tiled.
- At `N == R` with the Insert disconnected, Composite output is the Source bit-for-bit.
- At `N == R` with the Insert connected, output equals a reference `over`, bit-for-bit.
- ST Map at `N == R` is the exact normalized pixel grid, in both origin conventions — and
  separately, **consumed by Flame's own downstream tool** at float depth, because a round trip
  through our own resampler can carry the same half-pixel error on both sides and still close.
- Source frame sentinels distinct at every time, and Insert sentinels distinct at `N` and `R`,
  so the `insertTime` contract is observable rather than assumed.
- Multi-link chains exercised through the harness's arbitrary-time clip service.
- The plugin loads under `--host-name com.autodesk.flame` so the Flame quirk branch is walked
  by something other than Flame, and under a host that refuses the file-path string mode.

**On the box**: when the host build or environment changes, run the Phase 0 probe first. On the
qualified Flame 2026.2 host, install the plugin in a Batch node with a real plate and insert;
check `/opt/Autodesk/log/` for the plugin's stderr, which is the only diagnostic channel on a
machine nobody can attach a debugger to.

**Performance gate**: 1080p and 4K, each selected candidate at the shipping megapixel caps,
ms/frame and peak VRAM recorded on the target box before optimisation, with regression
thresholds tied to the exact model and runtime hashes.

---

## Phasing

| Phase | Content | Exit |
|---|---|---|
| **0A** | Extended `hostprobe`, run in Flame | **Closed 2026-08-20.** All five questions answered; the measured report is the authority |
| **0B** | Pinned SEA-RAFT M export through the private ORT on CPU and CUDA, in Flame | **Closed 2026-08-21.** Export provenance and hashes, direction/identity, exact CUDA payload closure/ownership, 480p–1080p VRAM/timing, cancellation, provider-init fallback, controlled arena-limit/CPU recovery, lifecycle and duplicate-node behavior are recorded. UHD, DCI 4K and Alexa 35 open gate each produced a valid bounded-allocation-stop measurement under the 16 GiB arena ceiling. This does not choose the shipping default or impose a product resolution cap |
| **0C** | Flame ST round trip, and instance/process lifetime | **Closed 2026-08-21.** ST convention measured (item 1); final render is a separate process — Burn (item 2); render scale, rowBytes/sub-window and anamorphic tiling closed at PAR 1 and PAR 2 (items 3–5). **`Precache` decided 2026-08-22: RAM-only, no persistence** (facility is foreground / single-node Burn); durable disk cache deferred until Burn renders fan out |
| **1** | Vendor, CMake, **two descriptors**, bundle, harness, `describe`/`describeInContext`, passthrough render | **Closed 2026-08-22.** PR #1 merged at `5fa267f`. All eleven Release CTest gates pass; exact exports and both dependency boundaries are enforced. Flame 2026.2 verified sockets, parameter separation, Set Ref, Current/Reference routing, colour/matte fallbacks, partial renders, native ST round-trip and load diagnostics. The empty-menu grouping found on-box was corrected before merge; its replacement artifact was deliberately not reinstalled because the fix is a standard property matching both probes |
| **2** | `src/core/flow` complete, `NullPairwiseEstimator`, `ww-flow`, full unit + harness coverage | **Closed 2026-08-22 through PR #4.** Separable lattice transform; typed flow links; confidence propagation; generation-safe caches; deterministic CLI; concurrency, unit and host-harness coverage are present |
| **2.5** | Model and export bake-off in `ww-flow` | **In progress. P25-0 through P25-4 are merged; P25-5 is qualified on draft PR #15 at `adfd4fb`:** protocol/schemas, artifact validation, corpus/conditioning, candidate export/evidence, offline runner, active protocol v2, and the exact evaluation-only EL8 airgap package are present. P25-6 target measurement, ranking, a shipping default, any qualifying fast alternative, persistent choice order, and the measured `analysisScale` decision remain open |
| **3** | Runtime loader, `ModelRegistry`, `OrtEnvironment`, selected estimators, library packaging | No link-time ORT dependency; `ORT_API_MANUAL_INIT`; CPU/CUDA/CoreML qualified; packaging baseline passes for every shipped library |
| **4** | `FlowPreparation` wired into render; on-demand pulls, abort, progress, Precache/Clear, persistent messages | A real shot tracks in Flame from a reference frame; cancellation, invalidation, OOM fallback and reference-boundary behaviour all verified |
| **5** | Optional second model behind `PairwiseFlowEstimator` | If Phase 2.5 selects a second shipping model, the Model parameter switches cleanly and both paths are covered; otherwise this phase is omitted and no one-option Model choice is published |
| **6** | FB check, smoothing, input curves, perf gate, CI packaging | Perf threshold recorded; tarball installs on the airgapped box |

---

## Deferred, deliberately

- Third matte mode ("matte as flow confidence") — cheap to add on top of `fbCheck`'s
  confidence plumbing, but the user asked for two modes.
- **Disk-backed flow cache surviving a Flame restart — DECIDED 2026-08-22: not in v1.** 0C
  measured final render as a separate process (Burn), which is what a persistent cache would
  bridge — but two facts remove the need here. The same run showed a single Burn process
  rendering **sequentially**, so it reuses its own RAM chain cache (one inference/frame after the
  prefix); and the facility renders **almost everything in the foreground** — same process and
  instance as the interactive session — and uses Burn rarely and **never fanned out** across a
  farm. So RAM (instance-lifetime) covers the real workflows; a disk cache would buy mainly
  *instant reopen after restart*, which is exactly where automatic persistence silently serves
  stale flow after an overnight regrade — worse than no cache. **v1 ships RAM-only.** The disk
  cache is retained as a **future option gated on one fact changing: final renders fanning out
  across multiple Burn nodes**; if built then, it must be explicitly **user-managed** (explicit
  `Clear`, a visible cache-state readout, per-setup scoping), never automatic. Per-instance RAM
  caching is correct for all of this either way (duplicate = new instance, same process, measured).
- Display-only overlay showing the reference frame and chain state. Flame does deliver pen
  events and the `OfxDrawSuite`, but **never keyboard events**, and `kOfxInteractPropPixelScale`
  reads `[1,1]` at every zoom — so any overlay must be display-only and zoom-independent.
- Drift mitigation past a chain-length threshold: a direct `N→R` candidate, or rebasing
  through an anchor. Comparing direct against composed in confident regions and blending is
  further out still.
- An OpenGL render path — the only GPU option OFX exposes in Flame, and irrelevant while
  inference dominates the frame time.
