# Architecture review — 2026-08-20

Status: advisory review, not an amendment to `docs/plan.md`.

Scope: the approved plan, host findings, model survey, and repository scaffold at
`0b81a52`. Existing files were not changed as part of this review.

## Executive view

The project is viable and the broad decomposition is good. The host probe has retired the
riskiest OFX assumptions: two inputs work, arbitrary-time pulls work during render, Flame
honours whole-frame rendering, and a private ONNX Runtime can coexist in-process with
Flame's runtime. The vendored image boundary and resampler are a strong starting point, and
the build/export/ABI gates are unusually appropriate for a Flame plugin.

I would proceed with Phase 1 and the host-free parts of Phase 2. I would not begin the
shipping inference backend until a short "Phase 0B" closes the remaining CUDA/provider and
real-network questions.

The main plan changes I recommend are:

1. Move the owned inference frame type out of `src/ofx`; otherwise `src/infer` cannot remain
   OFX-free as promised.
2. Replace scalar field spacing with an explicit model-to-image transform. Pixel aspect
   ratio, odd-sized images, padding, and asymmetric resize make one scalar insufficient.
3. Make output data type and ST-map convention a host-measured contract. Relative-pixel ST
   values require signed float, and normalized UV loses unacceptable precision at byte or
   half depth.
4. Specify cache ownership, invalidation, in-flight de-duplication, and persistence before
   implementing `FlowPreparation`. A process-wide model cache and a per-instance flow cache
   have different correctness rules and should not be one object.
5. Do a deployment bake-off before naming a default model. SEA-RAFT is the safest first
   shipping candidate; WAFT is the most promising accuracy/memory candidate but currently
   carries more export and checkpoint risk. NeuFlow v2 is a better stateless fast candidate
   than RIFE. MemFlow's temporal state is a poor fit for arbitrary-order OFX rendering.
6. Treat confidence/visibility and drift correction as first-class flow products. Spatial
   smoothing does not correct accumulated temporal drift.
7. Promote a durable analysis cache, or rename `Analyze` to `Precache`. A RAM-only Analyze
   pass is not a reliable production artifact across setup reloads, instance recreation, or
   background rendering.

## Findings that should change the architecture

### 1. The proposed layer boundary is currently impossible

The plan says `src/infer` is independent of OFX, but its proposed public interface accepts
`CapturedFrame`, which currently lives in `src/ofx/FrameCapture.h`. That header includes
`HostImage.h` and the OpenFX support library. The value itself is host-free; its location is
not.

Recommended split:

```text
src/core/image/OwnedFrame.h
    Owned float RGB/RGBA pixels, pixel bounds, PAR, alpha association

src/ofx/FrameCapture.h/.cpp
    OFX image lifetime and conversion; produces OwnedFrame

src/infer/FlowEstimator.h
    Consumes immutable image views plus ImageToModelTransform
```

This keeps dependency direction `ofx -> infer -> core`, rather than allowing `infer -> ofx`.
It also prevents storage provenance such as byte/half source depth from becoming part of the
model contract when the estimator only needs normalized RGB and geometry.

### 2. Geometry needs a transform, not `origin + scalar spacing`

`FieldGeometry` is already a good decision because flow values stay in full-resolution
source-pixel units. Its current scalar `spacing`, however, assumes the same exact scale on X
and Y. That is not enough for:

- anamorphic input, where Flame's buffer is in real pixels but the image's canonical shape
  is scaled by PAR on X;
- an odd source dimension reduced to half or quarter size;
- a resize whose rounded X and Y dimensions produce slightly different scale factors;
- model padding/cropping whose first output node is not simply `0.5 * scale` from the source
  origin;
- a future non-unit OFX render scale.

Use an explicit affine mapping from model-lattice pixel centres to source-image pixel
centres. At minimum it needs `origin`, `spacingX`, and `spacingY`; a small 2D affine type is
safer and costs nothing. Store the inverse as well, or provide checked conversion methods.
Validate that every scale is finite and positive at construction.

For anamorphic plates, make one policy explicit and test it:

- **Square-pixel analysis (recommended):** resample the plate into canonical geometry before
  inference, then transform vectors back into source pixel units independently on X and Y.
  The analysis-resolution control should apply after PAR normalization so PAR 2 does not
  accidentally double the planned VRAM budget.
- **Storage-pixel analysis:** feed the squeezed buffer to the network and accept that learned
  features and displacement metrics see distorted geometry. This is cheaper, but it should
  be a measured quality trade rather than an accidental consequence of buffer layout.

The first policy is the defensible default for a VFX tool. Include PAR 0.5 and 2.0, negative
image origins, odd extents, and non-zero bounds in the core test matrix.

### 3. A naked `FlowField` does not carry enough semantic information

The code aliases `Field<2>` to `FlowField`, while direction and meaning live in comments.
That leaves several expensive mistakes representable: treating an absolute map as a
displacement, composing `R->N` where `N->R` is required, or combining fields with different
geometry.

Wrap the samples in a semantic value such as:

```text
FlowLink {
    fromTime, toTime
    sourceGeometry, destinationGeometry
    backwardDisplacement
    optional confidence/visibility
    model fingerprint
}
```

Composition should reject incompatible endpoints and geometry rather than relying on the
caller. Keep `Field<2>` as storage, but do not use it as the public unit of chain algebra.

### 4. ST maps must be float data, and their convention must be measured in Flame

The plan advertises byte, short, half, and float output for all modes. That is safe for an
image composite, but not for an ST map:

- relative pixel offsets are signed and commonly outside `[0, 1]`; the existing integer
  writer clamps them and destroys the data;
- byte normalized UV has only 256 levels;
- half normalized UV has roughly pixel-scale quantization by 4K and gets worse outside the
  unit interval.

ST-map mode should request float output. Because Flame reports no multiple clip depths,
probe whether `getClipPreferences` can negotiate the whole effect to float from a byte or
half source. If not, make ST output a separate float-only descriptor or fail visibly; do not
silently quantize.

The normalized convention also needs an external round-trip test in Flame. A test that
generates an ST map and consumes it with White Water's own resampler can preserve the same
half-pixel error on both sides. Measure Flame's downstream ST tool with an asymmetric image
and a known translation, at PAR 1 and PAR 2, and record:

- whether pixel centres map using `x / width`, `(x + 0.5) / width`, or `x / (width - 1)`;
- whether normalization is relative to the image bounds, RoD, or project extent;
- channel layout (`R=U`, `G=V`, plus defined B/A values);
- bottom-left/top-left origin behavior;
- values outside `[0, 1]` and negative relative vectors.

Confidence should not be discarded. A practical RGBA data layout is RG = ST, B = confidence,
A = 1, if Flame's consumer ignores B. If pipeline convention requires B=0, offer a separate
confidence output mode rather than hard-coding `A=1` as the only information downstream can
receive.

### 5. Cache architecture needs separate lifetimes

The planned key omits the hardest invalidation inputs: effect instance, upstream clip
revision, actual model-file identity, image geometry/PAR, render scale, and algorithm/schema
version. OFX does not provide a trustworthy content hash for an upstream graph, so a
process-global flow cache cannot be made correct merely by adding more parameter values.

Recommended ownership:

```text
process lifetime
  RuntimeLoader / Ort::Env
  Model sessions keyed by model SHA256 + EP + runtime ABI
  GPU inference semaphore (start at one concurrent Run per device)

effect-instance lifetime
  generation counter
  pairwise link LRU and accumulated-field LRU
  in-flight table: key -> shared future/result
  optional disk-cache namespace identified by a persisted node UUID
```

On `changedClip`, increment the generation and invalidate all flow results. On a
flow-affecting `changedParam`, invalidate only the relevant stages where that distinction is
worth maintaining; otherwise prefer a generation bump over a clever but stale cache. Include
the exact model SHA256 and cache schema version in every persistent entry.

Concurrent renders of the same effect instance must be expected under
`eRenderInstanceSafe`, even if Flame often renders sequentially. Do not hold a cache mutex
while calling `clipGetImage` or `Session::Run`. Use an in-flight entry so two requests for the
same link wait for one computation. Define what happens when one waiter aborts but another
still needs the result.

The switch from half to float fields is correct, but update all budgets: a half-resolution
1080p link is about 4 MiB, not 2 MiB; a half-resolution 4K link is about 16 MiB before
allocator overhead, confidence, accumulated fields, and inference tensors. Capture two 4K
RGBA float frames also costs roughly 253 MiB of host RAM. Process chain links as a rolling
pair and release captured frames promptly.

### 6. A RAM-only `Analyze` action is not a production analysis pass

The plan calls `Analyze` an explicit analysis pass but defers disk persistence. In Flame,
artists will reasonably expect analysis to survive a setup reload and a final render. A
memory pre-warm may be lost if the host recreates the instance or renders in another
process. It also makes a successful interactive analysis irrelevant to a later farm or
background render.

Before deciding, probe instance and process lifetime across:

- save/reload of a Batch setup;
- switching away from and back to the node;
- foreground and background/final render;
- duplicating the node;
- reopening Flame.

If v1 remains memory-only, call the button `Precache` and specify its range. "Walk the
range" is currently undefined and could mean thousands of frames. Prefer Current-to-Ref,
Work Range, and Custom Range; do not default to the full source range.

For production, promote a simple append-safe disk cache. It can remain manual and
per-node—no need to solve perfect upstream hashing in v1. Store a manifest with source
geometry/range, parameter fingerprint, model hash, cache schema, completed links, and a
clean/partial status. Write each link atomically via temporary file plus rename.

### 7. ONNX Runtime loading needs to be a documented ABI boundary

The isolation result implies that the product must never acquire a normal link-time
dependency on ONNX Runtime. Make that a gate, not only a probe property:

1. Open the exact runtime by absolute path with `RTLD_LOCAL`.
2. Resolve `OrtGetApiBase` from that handle.
3. Define `ORT_API_MANUAL_INIT` and initialize the C++ wrapper with the API pointer, or use
   the C API throughout.
4. Keep the handle alive until process exit; do not `dlclose` while sessions, provider code,
   TLS, or static wrapper objects may survive.
5. Assert in CI that the shipping plugin has no `DT_NEEDED` entry for ONNX Runtime.

ONNX Runtime's own C++ header documents manual initialization for exactly this kind of
unlinked use: [onnxruntime_cxx_api.h](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_cxx_api.h).

The CUDA provider remains a distinct gate. Test a real candidate network, not the 128-byte
Add model, and capture:

- runtime and provider symbol ownership;
- CUDA/cuDNN/cuBLAS libraries actually selected;
- provider initialization and first-run latency;
- peak and steady VRAM alongside a representative Flame Batch;
- repeated create/run/destroy cycles and node duplication;
- cancellation from another thread using a per-run `RunOptions` object;
- fallback after provider initialization failure and after GPU OOM.

`RunOptions::SetTerminate` is valid cross-thread for runs using that object, according to the
[ORT API reference](https://onnxruntime.ai/docs/api/c/struct_ort_1_1_run_options.html), but
the OFX side still needs a safe abort-watcher design. A host abort cannot magically be polled
while the render thread is blocked inside `Run`.

Do not key one heavyweight session per resolution unless the exported model actually has
fixed shapes. Prefer one immutable session per model/EP and a small pool of per-shape I/O
bindings/buffers. Even though ORT permits concurrent `Run` calls, serialize GPU inference
initially: throughput is less important than avoiding a transient VRAM spike that takes down
Flame.

For macOS, treat CoreML as a separate backend qualification. The official provider docs note
that dynamic shapes can hurt performance and expose a static-shape-only option; fixed export
profiles may be the right trade for the advertised Full/Half/Quarter controls:
[CoreML Execution Provider](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html).

### 8. Drift and occlusion need different remedies

Gaussian smoothing can suppress local noise, but it does not correct accumulated temporal
bias. It can also blur motion boundaries and make foreground/background leakage worse. The
plan should not describe `Smooth` as drift mitigation.

A modest v1 strategy is:

- cache adjacent links for interactive incremental work;
- propagate confidence along the composed path (`min` or a calibrated product sampled at
  the advected location);
- at a configurable chain length, compute a direct `N->R` candidate or rebase through an
  anchor/keyframe;
- compare direct and composed fields in confident areas and select/blend or warn;
- expose invalid/confidence data to the ST-map and composite paths.

Forward-backward consistency is useful but expensive. SEA-RAFT predicts an uncertainty
distribution, which is one reason it is attractive: model confidence may supply a cheap
first-pass validity signal, with the 2x forward-backward check retained as a high-quality
option. Neither confidence source should be called an occlusion matte until calibrated on
real plates.

For composite output, specify what invalid flow does: hide the insert, keep the last valid
motion, fall back to direct flow, or leave it unwarped. These are artist-visible semantics,
not just an error policy.

## Workflow contracts to settle in Phase 1

### Insert time

Two useful behaviors exist, and the current text can be read as either:

- **Current (recommended default):** sample Insert at `N`, so a screen replacement or other
  animated insert advances normally while its canvas is carried by `B(N->R)`.
- **Reference/Hold:** sample Insert at `R`, freezing the insert content at the frame where it
  was positioned.

Make this an explicit choice if both are needed. Then `getFramesNeeded` can honestly declare
Source `{N}` plus Insert `{N}` or `{R}` while continuing to pull chain frames on demand.
Clamp `R` to the measured source range before constructing a chain; Flame's out-of-range
pulls return held images and otherwise create plausible-looking zero-motion links.

### General versus Filter context

"General with two clips; Filter registered as ST-Map-only" needs a concrete UI contract.
The same flat parameter list would show Composite and Warped Insert choices in a context
where no Insert exists and `setEnabled()` cannot safely hide them.

The simplest v1 is one General-context effect; the optional Insert may be disconnected for
ST output. If discoverability requires Filter context, consider two plugin descriptors with
shared implementation and stable identifiers: a General Track/Insert effect and a float-only
ST effect. That is easier to explain and test than context-dependent meanings for one Output
choice.

### Identity and failure

Keep `isIdentity` cheap and deterministic. It should not perform inference to discover that
inference will fail. A render-time failure must render its documented fallback itself:

- Composite -> Source at `N`;
- Warped Insert -> unwarped Insert at the selected insert time, or transparent black if
  disconnected;
- ST Map -> identity ST map, clearly logged as a fallback.

Only return an OFX identity for conditions known from parameters, time, and clip connection
state. This matters because Flame calls query actions much more often than render.

### Input conditioning

Separate model-required normalization from artist-selectable plate conditioning. "Filmic"
is not a reproducible tensor contract, and clamp-to-0..1 can erase exactly the highlight
texture needed to track motion-blurred or specular footage.

For the bake-off, compare at least:

- model-native normalization after a hard 0..1 clamp;
- a fixed signed/log compression suitable for scene-linear values;
- shared pairwise percentile normalization (one transform computed from both frames, never
  independent per-frame auto exposure);
- unmodified log-encoded input where the upstream plate is already log.

Do not add an OCIO dependency to the plugin merely for this. The goal is stable features,
not display rendering. Record the exact math in the cache key and model manifest.

## Model recommendation

Do not lock the UI's first choice index until export, licensing, and target-box measurements
are complete. Choice order becomes setup API after the first artist build.

### Recommended v1 bake-off

1. **SEA-RAFT M — conservative default candidate.** It is purpose-built pairwise optical
   flow, BSD-3 at the repository level, materially faster than original RAFT, and its
   reference inference exposes uncertainty. It has fewer checkpoint/backbone questions than
   WAFT. Source: [official SEA-RAFT repository](https://github.com/princeton-vl/SEA-RAFT).
2. **WAFT with a permissive backbone — quality/memory candidate.** Its official results are
   excellent and replacing the cost volume attacks the right VRAM problem. However, the
   official environment requires xformers and offers Twins, Depth Anything v2, and DINOv3
   backbones. That makes "simpler ONNX export" an unproven assumption, not yet a reason to
   name it the default. Start with the Twins checkpoint and audit the exact file. Source:
   [official WAFT repository](https://github.com/princeton-vl/WAFT).
3. **NeuFlow v2 — fast stateless candidate.** It is a real optical-flow model rather than a
   frame-interpolation model, the repository is Apache-2.0, and the paper targets high
   efficiency. Its official repository does not advertise ONNX support, so export remains a
   spike and checkpoint terms still need recording. Sources: [official repository](https://github.com/neufieldrobotics/NeuFlow_v2),
   [paper](https://arxiv.org/abs/2408.10161).
4. **Original RAFT — validation baseline, not necessarily shipped.** Its known behavior and
   mature exports make it valuable as an oracle for pipeline correctness even if a newer
   model wins the product slot.

Benchmark each candidate at the exact export used by the plugin, not in upstream PyTorch.
The gate should include CPU, CUDA, and CoreML operator coverage; output direction; dynamic or
fixed shapes; peak VRAM; first/subsequent latency; model and runtime payload; uncertainty;
and facility-shot quality.

### Candidates to keep out of the v1 drop-down

- **RIFE:** fast and exportable in multiple community projects, but its flow is trained as
  an internal representation for frame interpolation. Keep it only if it wins a facility
  tracking test, not because it produces a tensor named flow. The original work explicitly
  targets intermediate-frame synthesis: [RIFE paper](https://arxiv.org/abs/2011.06294).
- **MemFlow:** promising, Apache-2.0 at the repository level, and designed around temporal
  memory. That state conflicts with random-access scrubbing, repeated frames, concurrent
  renders, and restart determinism. It belongs behind an explicit sequential Analyze + disk
  cache architecture, not behind the same stateless pairwise estimator contract. Source:
  [official MemFlow repository](https://github.com/DQiaole/MemFlow).
- **AllTracker:** architecturally aligned with reference-frame tracking and returns visibility
  and confidence, but the authors report 768x1024 operation on a 40 GB GPU. Keep it as a
  research branch after the CUDA/VRAM gate, not a v1 dependency. Source:
  [official AllTracker project](https://alltracker.github.io/).
- **UFM:** its wide-baseline behavior is interesting for direct anchor correction, but the
  official pretrained checkpoints are currently CC BY-NC-SA and therefore unsuitable for
  commercial facility work. Source: [official UFM repository](https://github.com/UniFlowMatch/UFM).

### Add a VFX-relevant validation set

Sintel/KITTI/Spring rankings are useful but do not select a comp tracker. Build a small,
versioned internal evaluation set covering motion blur, defocus, low contrast, grain,
occlusion/reveal, rolling shutter, fine hair, reflections, screens, and anamorphic plates.
Score point/patch drift at artist-selected landmarks, warp residual on a reference patch,
temporal stability, invalid-region behavior, and resource use.

The public [LayeredFlow](https://github.com/princeton-vl/LayeredFlow) benchmark is unusually
relevant because it targets transparent, reflective, and multi-layer motion. Even if
MultiRAFT never ships, include its single-layer subsets in offline evaluation after checking
dataset terms. It is a better stress test for VFX plates than another driving-only ranking.

## Revised phase gates

| Gate | Work | Exit condition |
|---|---|---|
| 0A — complete | Existing host probe | Preserve the measured report as the authority. |
| 0B — before inference implementation | Real candidate ONNX model through private CPU and CUDA ORT in Flame | Correct output identity/direction; provider libraries identified; abort exercised; VRAM and latency recorded; repeated lifecycle clean. |
| 0C — before ST/cache integration | Flame ST round-trip and instance/background-render lifetime probe | Float-depth negotiation and exact ST convention recorded; Analyze persistence decision made. |
| 1 | Plugin shell and workflow contract | Decide one vs two descriptors, Insert time semantics, output-depth policy, cheap query actions, and visible fallbacks. |
| 2 | Host-free geometry/flow/cache interfaces | Affine lattice transform, typed flow links, confidence propagation, concurrency tests, serialization format if Analyze is durable. |
| 2.5 | Model/export bake-off in `ww-flow` | One default and one optional fast candidate selected by exact ONNX artifact, target performance, quality, and licence audit. |
| 3 | Runtime loader and selected estimators | No link-time ORT dependency; manual API initialization; CPU/CUDA/CoreML qualification; packaging baseline passes for every library. |
| 4 | Flame integration | Random and sequential renders deterministic; cancellation, cache invalidation, OOM/fallback, and reference-boundary behavior verified. |
| 5+ | Quality features | Direct/anchor drift strategy, confidence output, optional FB check, plate conditioning, and performance regression gates. |

## Scaffold review

What is already strong:

- Clean local Release build at `0b81a52`; `core::dependency_boundary` passes.
- Raw-C, dependency-free host probe and a separate explicit-loader ORT probe.
- Exact exported-symbol checks on both platforms and `--no-undefined` on Linux.
- EL8/glibc 2.28 build correction and a hard-coded ABI gate.
- Float field decision and full-resolution displacement units.
- Row-stride, negative-row-byte, depth/component, alpha-aware resampling infrastructure.
- Provenance headers on vendored files and an explicit port-back rule.

Items to clean up before Phase 1, without changing the architecture:

- `README.md` and `AGENTS.md` still say Phase 0 has not run and still prescribe Rocky 9;
  the workflow and governing docs now use AlmaLinux 8/glibc 2.28.
- `cmake/OfxBundle.cmake` still documents bundled runtime libraries even though the plan now
  questions that packaging layout.
- `cmake/OnnxRuntime.cmake` says CPU `RTLD_DEEPBIND` isolation implies CUDA isolation; the
  measured notes correctly say the provider is a separate, untested problem.
- `WHITEWATER_ENABLE_ONNXRUNTIME` is currently informational while setting
  `WHITEWATER_ORT_ROOT` independently builds the probe. That is reasonable for scaffolding,
  but use separate probe/product options once Phase 3 arrives.
- `WHITEWATER_BUILD_TESTS` is not consumed yet, and the repository has only the boundary
  test. This matches the stated phase, but Phase 1 should make the option real.
- `alphaModeFor()` comments say opaque pixels take the cheap premultiplied path, while the
  implementation routes opaque to unpremultiplied. The result is normally correct because
  alpha is one, but the comment and performance behavior disagree. Check warp-drive for the
  same issue before fixing, per repository policy.
- Add checked construction to `FieldGeometry`; zero, negative, NaN, or infinite spacing
  currently reaches division in `sample()`.

## Minimum additional tests

- Direction-labelled constant translations in both temporal directions, including a test
  that deliberately swaps endpoints and must fail.
- Composition over affine and spatially varying fields, not only identity.
- PAR 0.5/2, odd dimensions, non-zero/negative bounds, asymmetric padding, and any observed
  render scale.
- Source frame sentinels distinct at every time and Insert sentinels distinct at `N` and
  `R`, so the selected Insert-time contract is observable.
- ST maps consumed by Flame's own downstream tool at float depth.
- Confidence propagation across a multi-link chain and at an out-of-bounds sample.
- Concurrent same-key and different-key renders; only one same-key inference may run.
- Abort before frame pull, during analysis loop, during ORT Run, and while waiting on another
  request's in-flight result.
- Cache generation change while an old inference is still running; the old result must not
  publish into the new generation.
- GPU provider initialization failure, GPU OOM, missing model, bad model hash, incompatible
  tensor contract, and CPU fallback.
- Real exported model translation/rotation tests at every supported fixed/dynamic shape and
  EP, with direction and scale checked numerically.

## Bottom line

The plan is fundamentally sound, and the measured host work has earned the right to start
the plugin shell and host-free flow algebra. The changes above are mostly about making
implicit coordinate, lifetime, and data contracts explicit before they harden into APIs.
The two areas most likely to cost a rewrite if deferred are anamorphic/model geometry and
cache lifetime. The model name itself is less important than proving the exact exported
artifact on the exact Flame box.
