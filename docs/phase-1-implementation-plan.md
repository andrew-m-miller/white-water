# Phase 1 implementation plan

Phase 1 turns the closed host measurements into the smallest real White Water plugin. Its
purpose is to freeze and verify the OFX-facing workflow contract before flow algebra or ONNX
Runtime enters the module.

**Status: closed 2026-08-22.** The implementation lived on `codex/phase-1`, passed all eleven
local tests and the Flame 2026.2 product smoke test, and merged through PR #1 at `5fa267f`.
The topic branch was deleted. This document remains the historical work breakdown and exclusion
record; `docs/plan.md` and `docs/context.md` own the current phase status.

## Exit contract

The phase is complete when one `WhiteWater.ofx.bundle`:

- exports exactly `OfxGetNumberOfPlugins`, `OfxGetPlugin`, and `OfxSetHost`;
- enumerates the permanent Track/Insert and ST Map identifiers;
- describes both effects without throwing, including under Flame's measured property quirks;
- exposes the settled flat parameter contract with labels no longer than 12 characters;
- renders the documented inference-failure outputs at every advertised layout;
- declares only the current Source frame and selected Insert frame through
  `getFramesNeeded`, while ROI asks for complete connected RoDs;
- passes the dependency-boundary, unit, bundle-export, and host-harness tests.

The two choice controls whose ordering is explicitly owned by Phase 2.5 (`model` and
`inputCurve`) are not defined in Phase 1. `analysisScale` is likewise omitted until measured
shipping caps exist. A zero-option numeric Choice has no valid OFX value, while a temporary
option would become saved-setup API. Their script names remain centralized for later use,
but Phase 1 publishes neither invalid values nor a fake ordering.

## Work packages

### P1-A — OFX contract and fallback renderer

Owner: `phase1_ofx_contract`

Files: `src/ofx/FlowParameters.*`, `src/ofx/OpticalFlowPlugin.*`, `src/ofx/Plugin.cpp`, and
`src/ofx/CMakeLists.txt`.

Deliverables:

1. Register `com.mtifilm.whitewater.opticalflow` and `com.mtifilm.whitewater.stmap` from the
   module's single translation unit.
2. Describe the Track/Insert clip/depth/component contract and the float-only ST contract.
3. Define stable parameters in plan order without `setEnabled()` or throwing risky property
   setters.
4. Implement cheap `isIdentity`, `getFramesNeeded`, full-RoD ROI, unpremultiplied clip
   preferences, and `Set Ref`.
5. Implement the Phase 1 fallback renderer: Source copy for Composite, selected unwarped
   Insert or transparent black for Warped Insert, and an exact identity ST field.
6. Handle partial render windows and all advertised Track pixel layouts without inference.

### P1-B — executable host contract

Owner: `phase1_host_harness`

Files: `tests/**`.

Deliverables:

1. Port the raw minimal host from warp-drive commit `887a123` with provenance.
2. Add General-context Source/Insert clips and arbitrary-time `clipGetImage` service.
3. Use distinct time sentinels so Current-versus-Reference behavior is observable.
4. Verify both descriptors, stable parameter order/labels, advertised depths/components,
   frame-needs declarations, query-action purity, and unpremultiplied output.
5. Render whole and partial windows across the Track layout matrix and verify both ST origin
   conventions plus relative-zero identity.

### P1-C — build, bundle, and layer enforcement

Owner: `phase1_build_boundaries`

Files: top-level `CMakeLists.txt`, `cmake/**`, `scripts/**`,
`src/core/image/OwnedFrame.h`, and `src/ofx/FrameCapture.*`.

Deliverables:

1. Move the owned host-free frame value below the OFX boundary.
2. Preserve the core dependency check and add a mechanically exercised infer policy that
   rejects OFX while allowing inference dependencies.
3. Wire `tests/**` behind `WHITEWATER_BUILD_TESTS`.
4. Add a platform-aware test for the exact three-symbol module export set.
5. Keep the Phase 1 bundle free of ONNX Runtime, models, and any runtime payload staging.

## Integration order

1. Land P1-C's owned-frame seam and boundary checks first; it has no dependency on the new
   plugin classes.
2. Integrate P1-A and make the production bundle compile and enumerate both factories.
3. Integrate P1-B against the real bundle rather than a duplicated descriptor model.
4. Resolve only interface mismatches found by compilation; do not weaken the harness to fit
   an implementation mistake.
5. Run a clean Release configure, build, and complete `ctest --output-on-failure`.
6. Inspect the built module's dynamic dependencies, rpath/install name, architecture, and
   exported symbols.
7. Review the diff against `docs/plan.md`, `docs/host-notes.md`, and this phase's explicit
   exclusions.

## Explicit exclusions

- No flow fields, chain composition, cache, precache execution, smoothing, FB check, or
  inference orchestration; those begin in Phases 2–4.
- No ONNX Runtime linkage or packaging, session cache, execution-provider selection, or
  model registry.
- No model, input-curve, or megapixel-cap choice ordering before the Phase 2.5 measurements.
- No persistent cache; Phase 0C closed v1 as instance-lifetime RAM only.
- No new host claims. Phase 1 consumes the measured record and the vendored OFX protocol.

## Verification and handoff

The local gate is:

```bash
cmake -S . -B build-phase1 -DCMAKE_BUILD_TYPE=Release
cmake --build build-phase1 -j
ctest --test-dir build-phase1 --output-on-failure
```

The PR recorded the local matrix and dispatched an EL8 artifact for the Flame Phase 1 contract
test. On-box verification confirmed both nodes and sockets, legible and descriptor-specific
controls, scalar Set Ref behavior, Source/Insert routing, connected and disconnected mattes,
partial renders, the native ST round-trip, and clean load diagnostics. The test exposed a missing
menu-grouping property; the standard `White Water` grouping and a harness assertion were added
before PR #1 merged. The replacement artifact was not reinstalled before merge by explicit human
choice because the change matches both already-working probes and is low-risk to correct later.
