# Phase 2 implementation plan

Phase 2 completes the host-free tracking pipeline and its deterministic test double. It
stops before the Phase 2.5 model/export bake-off and preserves the later boundaries in
`docs/plan.md`: no ONNX Runtime loading, no model-specific tensor contract, no permanent
`model` or `inputCurve` choice order, and no OFX `FlowPreparation` integration.

## 2.1 — Geometry, typed links, and chain algebra

Replace `FieldGeometry`'s scalar spacing with checked `spacingX` and `spacingY`, keeping
field values in full-resolution real-pixel units. Add explicit lattice/image conversion
helpers and reject zero, negative, NaN, and infinite spacing before sampling can divide by
it.

Add `FlowLink` as the public algebra value. It owns direction-labelled endpoints, from/to
geometry, a backward-displacement field, and an opaque model fingerprint. Construction and
composition reject inconsistent field geometry, endpoint order, intermediate geometry, or
fingerprint. Confidence remains a parallel `ScalarField`, never mandatory storage inside a
link.

Add:

- composition using `a(q) + b(q + a(q))`;
- forward/backward residual and confidence generation;
- confidence composition sampled along the same warped path;
- Gaussian smoothing as a spatial-noise primitive, without claiming it fixes drift;
- a `WarpMap` adapter for a completed backward flow;
- pure chain planning for `N -> R`, including exact identity and direction-labelled link
  requests on both sides of the reference.

Tests cover exact identity over hundreds of links, constant translations in both temporal
directions, a deliberately reversed endpoint rejection, affine and spatially varying
fields, mismatched geometry/fingerprint, confidence at field bounds, non-zero/negative
origins, odd dimensions, and anisotropic spacing representative of PAR 0.5 and 2.

## 2.2 — Instance flow caches

Add byte-budgeted, thread-safe LRU storage for pairwise links and accumulated fields. The
two stores may share reusable policy code, but remain distinct caches because they have
different key/value semantics. Keys carry exact endpoints, generation, model fingerprint,
source/destination model geometry, matte/input-conditioning tokens, model-parameter fingerprint, and a cache
schema version without inventing Phase 2.5 choice indices.

Cache use is split into lookup, out-of-lock computation, and conditional publication. A
generation token captured before work prevents an old inference from publishing after
`changedClip` or a flow-affecting parameter invalidates the instance. No cache lock may be
held while caller work runs. Zero budget is a valid disabled cache; oversize single entries
are not retained; eviction and byte accounting include confidence when present.

Tests cover recency, replacement, exact byte budgets, oversize/disabled behavior, separate
generations, clear/invalidate, and a deterministic concurrency case where an old in-flight
result loses the race to a generation bump and is refused publication.

## 2.3 — Host-free image preparation and output transforms

Add model-independent primitives only:

- preprocessing from `OwnedFrame`: optional premultiply-by-matte, square-pixel analysis
  geometry, megapixel-cap sizing, bilinear downscale, reflect padding to a caller-supplied
  multiple, and exact crop metadata;
- field-to-ST conversion for absolute normalized UV and relative real-pixel displacement,
  with bottom-left (Flame-native default) and top-left origins;
- explicit straight/premultiplied `over` compositing;
- image/field adapters needed by the resampler and CLI.

Input transforms are represented by stable mathematical configuration or an opaque token;
this phase does not name or order the bake-off candidates. Tests cover pad/crop exactness,
asymmetric padding, matte reference values, PAR 0.5/2 square-pixel geometry, odd sizes,
negative bounds, ST identity/translation in both origins and modes, and alpha edge cases.

## 2.4 — Deterministic inference, offline CLI, and gates

Create `src/infer` now for the ORT-free portion of the architecture:

- `PairwiseFlowEstimator`, `FlowRequest`, and `FlowResult` use only `src/core` types and no
  OFX types;
- `NullPairwiseEstimator` returns deterministic analytic fields selected by test/tool
  request data, including identity, translation, affine, and spatial variation;
- the real `src/infer` directory replaces the Phase 1 boundary fixture as the positive
  inference dependency-boundary target; the expected-failure OFX fixture remains.

Add `tools/ww-flow` with dependency-free PFM read/write and modes for flow, ST output, and
warped output. Its null-estimator path is always tested. A real-model golden hook may skip
cleanly when no explicitly supplied candidate artifact/backend exists; Phase 2 must not
select or silently download an artifact.

Extend the raw host harness only with Phase 2 facilities that do not wire production
inference: arbitrary-time Source/Insert serving, distinct time sentinels, and assertions
that the permanent Phase 1 fallbacks and frame-needs contract remain unchanged. Multi-link
product rendering, abort/progress, cache invalidation from OFX actions, and persistent
messages stay with `FlowPreparation` in Phase 4.

Integrate the new targets and tests into CMake, run a clean Release build and full `ctest`,
and re-run both dependency-boundary negative fixtures and the exact OFX export gate.

## Exit and exclusions

Phase 2 exits when all host-free tests are green, cache publication is generation-safe,
the null estimator drives the CLI without weights or a GPU, the host harness still proves
the Phase 1 contract, and `src/core`/`src/infer` boundaries remain mechanically enforced.

The following are explicitly out of scope until later phases:

- model/default selection, input-curve names or choice indices, quality/licence comparison,
  or any other Phase 2.5 bake-off result;
- `ModelSpec`, `ModelRegistry`, `RuntimeLoader`, `OrtEnvironment`, ONNX sessions/providers,
  and runtime packaging (Phase 3);
- OFX frame pulls, `FlowPreparation`, render integration, progress/abort wiring, Precache,
  and persistent artist messages (Phase 4);
- durable flow storage, direct/anchor drift mitigation, and model-named estimator classes.
