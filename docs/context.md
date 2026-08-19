# Context

The live ledger: why decisions went the way they did, what is measured versus assumed, and
what is known to be weak. `docs/plan.md` says what we are building; this says why, and
records the things a future session would otherwise re-derive.

The discipline is warp-drive's: agent instructions own process, documents own facts. If a
fact appears here it should not also be restated in a role file or a prompt.

---

## Session 1 — 2026-08-19 — scaffolding

Greenfield. The repository had a remote and no commits.

### What exists now

- Build configures and builds clean on macOS arm64: the vendored core, the vendored OFX
  boundary, and the host probe as a real `.ofx.bundle` exporting exactly the three OFX
  entry points (verified with `nm -gU`).
- `ctest` runs one test, `core::dependency_boundary`, which is the mechanical guard on the
  layering rather than a comment about it.
- `docs/plan.md` is the approved plan, copied verbatim into the repo.

### What is deliberately absent

Everything past scaffolding. Phase 0 is a measurement phase and its answers change the
architecture, so writing Phase 2+ code first would be writing code against assumptions.
See the Open section of `docs/host-notes.md`.

`src/infer`, `tools/ww-flow` and `tests/` are named in the top-level CMakeLists as
comments, in dependency order, rather than existing as empty directories — an empty
directory with a stub CMakeLists reads as "someone started this", which is worse than
nothing.

---

## Decisions

### Vendored from warp-drive rather than submoduled

`887a123`. The two plugins ship on independent cadences, and warp-drive's CMake is not
factored to export a narrow set of targets — coupling the builds would mean either
refactoring warp-drive to serve a consumer it does not have, or dragging Qt, Eigen and the
editor into a build that wants none of them. A fix on either side gets ported by hand,
which is a real ongoing cost and the reason each vendored file names its provenance in its
first line.

What came over: `Vec2`, `Image`, `WarpMap`, `Resampler`, `HostImage`, `PixelFormat`,
`FrameCapture`, `HostQuirks`, the bundle/rpath/version-script CMake, the two check scripts,
and the host probe.

The single most valuable piece is `Resampler` + `WarpMap`. `WarpMap` is a one-method
interface — "for this destination point, where does it come from?" — which a dense flow
field satisfies directly. That means the correct-alpha, correct-edge, row-threaded,
bit-exact-identity backward warp is already written and already shipped. Reimplementing it
would have been the single likeliest place to introduce a matte fringe that survives review
and gets found in a grade.

### Float fields, not half

`src/core/flow/Field.h` stores float even though half would halve the cache footprint. Half
carries an 11-bit significand: at a 1000-pixel displacement, which an accumulated chain
across a fast camera move reaches easily, the representable step is about half a pixel —
visible judder on a locked-off insert. Under ~64 pixels half is fine, which is exactly what
makes it dangerous: a field type whose precision silently depends on how far the shot has
travelled is a bug waiting for a specific shot.

This is a change from the approved plan, which said half-float storage. The cache budget is
the knob for memory; the number format is not.

### Analysis scale lives in the field's geometry, not in an upscale pass

A field carries `FieldGeometry{origin, spacing}` and its *values are always in
full-resolution pixels*. A half-resolution RAFT result becomes a field with `spacing = 2`
whose vectors were scaled by 2 on arrival. From that point every consumer — compose, ST
map, resampler — works in one coordinate space and never learns that an analysis scale
exists.

The alternative, resampling each field up to full resolution on arrival, costs a copy per
chain link and filters the same data twice: once going up, once when it is sampled.

### The reference frame gets a button, not just a number

Flame hands OFX 0-based time: a batch starting at frame 1001 arrives as time 0. An artist
reading a frame number off the Flame UI and typing it into a `Ref Frame` field would be
wrong by the batch start every single time, and the failure is silent — the track just
starts from the wrong place. `changedParam` receives `args.time`, so a `Set Ref` push
button can write the actual current frame. The Int parameter stays, because a setup has to
store something and an artist has to be able to nudge it.

Note that the 0-based claim is **reported, not instrumented** — see `docs/host-notes.md`.
It came from facility experience with these batches, including the same behaviour in Mocha.
The button is the right design whether or not the offset is zero, which is part of why it
is the right design.

---

## Known weaknesses

- **Chain drift is not solved.** Composing `k` pairwise flows accumulates `k` interpolation
  errors. Mitigated by the `Smooth` parameter and by keeping the reference frame near the
  working range; a fall-back to a direct `R→N` inference past a chain-length threshold is
  noted in the plan and not built.
- **Occlusion is opt-in.** The forward-backward check doubles inference cost, so it is off
  by default. Until an artist turns it on, a warped insert smears through an occlusion.
- **Nothing has been measured on the box.** Every host claim in this repo is inherited from
  a different plugin asking different questions.
- **Model licences are read, not audited.** RAFT is BSD-3 (princeton-vl); Practical-RIFE
  states its weights carry the same MIT licence as its code. Both need re-verifying against
  the exact checkpoints shipped, before anything goes to a client. See `models/MODELS.md`.
- **`cmake/ofx.map` protects the host from us, not us from the host.** It stops our symbols
  being visible in Flame's process. It does nothing about *our* references binding to
  Flame's already-loaded copy of something ONNX Runtime also carries — protobuf and the
  CUDA runtime being the obvious candidates. That is Phase 0 probe item 3.

---

## Mistakes and corrections

None yet. When one happens it goes here, with what the symptom looked like — the symptom is
the part that saves the next person time.
