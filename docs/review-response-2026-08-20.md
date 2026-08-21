# Response to the architecture review — 2026-08-20

Status: advisory reply to `docs/architecture-review-2026-08-20.md`. Not an amendment to
`docs/plan.md`. Nothing in the repository was changed to produce this.

Scope: I read the review against the actual tree at `0b81a52` rather than accepting its
claims about the code. Every scaffold-level assertion it makes checks out — verified below.
Roughly 80% of it I would act on, and it correctly identifies the two items that get
expensive if deferred (field geometry, cache lifetime).

## Claims verified against the tree

| Review claim | Verified |
|---|---|
| `CapturedFrame` pulls OFX into `src/infer` | Yes — `src/ofx/FrameCapture.h:26` includes `ofxsImageEffect.h` |
| `FieldGeometry` has scalar `spacing` | Yes — `src/core/flow/Field.h:70`, single `double spacing` |
| `FieldGeometry` reaches division unchecked | Yes — no validation at construction; `sample()` divides by `spacing` |
| `alphaModeFor()` comment disagrees with code | Yes — `src/ofx/HostImage.cpp:59` routes `eImageOpaque` to unpremultiplied while the comment claims the premultiplied fast path |
| README/AGENTS still say pre-Phase 0 and Rocky 9 | Yes — `README.md:14`, `README.md:55`, `AGENTS.md:16`, `AGENTS.md:41` |
| `OnnxRuntime.cmake` claims CPU isolation implies CUDA | Yes — `cmake/OnnxRuntime.cmake:13-15` |
| Cache budget arithmetic is stale | Yes — `docs/plan.md:193` says ~2 MB per 1080p half-res link, which was true under half storage. `Field.h` is float. It is 4.1 MB. `docs/plan.md:129` also still says "half-float storage" |

One addition the review did not make: `scripts/check-core-dependencies.cmake` takes only
`CORE_DIR`. It has no notion of `src/infer`. So the promise that `src/infer` stays OFX-free
is currently enforced by nothing at all — the boundary violation in finding 1 would not have
been caught by the gate that exists.

## Agreed, and the plan should change

### Geometry — the highest-value finding

Scalar `spacing` hard-codes "x and y scale identically." A PAR 2 plate has already been
measured on the target box. This is the same class of error as corrections 4 and 5 in
`docs/context.md`: comparing quantities that live in two different coordinate spaces. Twice
now that has cost a rebuild-and-rerun cycle, and both times the wrong answer looked
plausible. Getting it into the type before anything consumes it is cheap; retrofitting after
`compose`, `StMap`, `Composite` and the resampler adapter all assume isotropy is not.

**One narrowing.** The review suggests "a small 2D affine is safer and costs nothing." I
would stop at separable: `origin`, `spacingX`, `spacingY`. A general affine admits rotation
and shear that nothing in this pipeline produces, and then every consumer carries the general
case forever. Separable covers PAR, odd extents, asymmetric rounding, and render scale —
which is the entire measured problem. If a rotation ever genuinely appears, it is a new type,
not a generalisation of this one.

**The sharpest point in the document** is that analysis scale must apply *after* PAR
normalisation. Without that, "Half" on a PAR 2 plate silently doubles the pixel count the
VRAM budget was computed against. It would present as an unexplained OOM on anamorphic shots
only — the worst possible failure signature.

Square-pixel analysis as the default is right. The models are trained on square-pixel
imagery; a 2:1 squeeze distorts both the learned features and the displacement metric.

### Layer boundary

Move the owned frame type to `src/core`. Beyond the move, the boundary gate needs a second
invocation with its own allow-list for `src/infer`, or the same violation recurs silently.

### ST maps must be float

The plan advertises byte/short/half/float uniformly across all three output modes, which is
wrong for ST. The review understates the half case if anything: near 1.0, half's step is
~2^-10, which at 4096 across is about **4 pixels** of quantisation, not "roughly pixel-scale."
Relative-pixel mode is signed and outside [0,1], so an integer writer destroys it outright.

Given warp-drive already measured that Flame reports a single depth, the likely outcome of
the proposed `getClipPreferences` probe is that we *cannot* negotiate up. In that case the
honest behaviour is a visible refusal, not silent quantisation.

**Declining the RGBA layout suggestion** (B = confidence). Data hidden in a channel that
downstream consumers may or may not preserve is how you get a bug report six months later.
If confidence needs to reach downstream, it is a separate explicit output mode.

### Cache lifetimes

The core insight is right and the plan had it wrong: a process-lifetime session cache and an
instance-lifetime flow cache have different invalidation rules and should not be one object.
"Do not hold a cache mutex across `clipGetImage` or `Session::Run`" is exactly the shape of
thing that deadlocks a host.

### Drift

`docs/plan.md:196` says `Smooth` mitigates drift. It does not. Spatial smoothing addresses
local noise; drift is accumulated systematic bias along the temporal axis. Worse, blurring
the field softens motion boundaries, which makes foreground/background leakage worse — the
opposite of the intent. That sentence is wrong on its own terms and should be fixed whether
or not anchor-rebasing gets built.

### Insert time

Genuinely ambiguous in the plan text, and the two readings produce completely different
results for an animated insert. Needs to be an explicit parameter, and it changes what
`getFramesNeeded` declares.

### Input conditioning

"Filmic is not a reproducible tensor contract" is a fair hit — a named curve went into a
parameter table without a definition. The shared-pairwise-percentile suggestion is good:
per-frame normalisation would inject apparent exposure change directly into the flow estimate.

### Model choice index

Do not lock it before the bake-off. This follows directly from a rule already in `CLAUDE.md`
— choice option order is API — which the plan's "RAFT default" violated.

### ORT manual initialisation

Worth calling out separately from the rest of finding 7, most of which CI already covers.
The C++ wrapper carries a global API pointer initialised by calling `OrtGetApiBase()`. If
that resolves through normal linkage it quietly defeats the entire dlopen isolation just
measured. Small check, large blast radius.

## Disagreed, or would scope differently

### Disk cache in v1

The review argues for promoting durable analysis while conceding that upstream hashing cannot
be solved correctly. Those do not reconcile. A disk cache that cannot detect that someone
regraded the plate is *worse* than no disk cache — it silently serves stale flow and the
artist has no reason to suspect it.

The rename to `Precache` with a defined range (Current-to-Ref / Work Range / Custom, never
"the full source range") is the correct cheap move now. Durable analysis is a Phase 5+
feature that needs its own invalidation story first.

### Concurrency as a premise

The review asserts concurrent same-instance renders "must be expected" under
`eRenderInstanceSafe`. True as OFX contract. But there are 47 measured renders in the probe
transcript and `setHostFrameThreading(false)` is set — whether Flame actually overlaps them
may already be answerable from data in hand. The in-flight table is cheap and worth having
regardless, but it should not drive the cache design on an assumption when a measurement
exists.

### Confidence inside `FlowLink`

Coupling them into one type makes the common no-confidence path pay for the optional field,
in the cache and in the budget. A parallel `ScalarField` under the same key is cleaner. The
rest of the `FlowLink` proposal — endpoints in the type, `compose` rejecting mismatches — is
right, and direction errors are the most likely bug in this design precisely because a
reversed chain still produces plausible-looking motion.

### External citations need independent verification

The review makes specific licensing claims — NeuFlow v2 as Apache-2.0, UFM checkpoints as
CC-BY-NC-SA — that would directly gate commercial facility use. Those get checked against the
actual repositories and the actual checkpoint files before anything lands in
`models/MODELS.md`, not accepted secondhand. Same standard as the DINOv3 backbone question
already open for WAFT.

The model reasoning itself is sound. NeuFlow v2 over RIFE is the review's best call: RIFE's
flow is an internal representation supervised for frame synthesis, never as flow. "Fast and
exportable" drove that slot in the plan, which was the wrong criterion. The MemFlow exclusion
is right for the right reason — temporal state versus random-access scrubbing is an
architecture incompatibility, not a quality question.

## Where the review under-reaches

It dismisses AllTracker on VRAM and moves on, but the architectural point deserved more
weight: a reference-frame tracker natively computes what the chain *approximates*. If one
becomes viable, `FlowChain`, drift, anchor-rebasing and the link cache all collapse into a
single inference. Not v1 — but the `FlowEstimator` interface should not be shaped so tightly
around pairwise that it forecloses a window-based model later. The review's own `FlowLink`
proposal happens to stay compatible, since a direct `N->R` is still just a pair.

It is also silent on the ~3 GB CUDA payload and bundle layout, which is already on the open
list in `docs/host-notes.md` and is a real packaging blocker for Phase 3.

## Suggested triage

| Before Phase 1 code | Probe work (0B / 0C) | Defer |
|---|---|---|
| Separable field geometry + checked construction | CUDA EP with a real model | Disk-backed cache |
| Move owned frame type to `src/core` | ST convention round-trip in Flame | Direct/composed blending |
| Extend boundary gate to cover `src/infer` | Instance lifetime across save/reload/bg render | LayeredFlow evaluation |
| Typed `FlowLink` with endpoint checking | `ORT_API_MANUAL_INIT` verification | Anchor rebasing beyond a simple threshold |
| Fix plan text: drift, budgets, half-float, Filmic | | |
| Doc cleanup: README/AGENTS Rocky->EL8, `OnnxRuntime.cmake` | | |

## Bottom line

Good review. The two items it names as rewrite-risk are the right two. The disagreements
above are about scope and enforceability rather than direction — the disk cache and the
concurrency premise are both cases of designing against an assumption when a cheaper honest
answer is available now.
