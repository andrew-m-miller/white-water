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

## Session 2 — 2026-08-20 — Phase 0A, and a review

Two things happened. The probe ran in Flame and closed all five Phase 0 questions, and an
advisory review of the plan and scaffold came back (`docs/architecture-review-2026-08-20.md`,
with the reply and a second round of corrections in `docs/review-response-2026-08-20.md`).

**Phase 0A closed, with two answers better than budgeted.** `clipGetImage` works during
render, so the on-demand chain design survives intact. In-process inference works under plain
`RTLD_LOCAL`, so there is no IPC boundary, no supervised helper and no frame transport to
build — a substantial simplification against what the plan had provisioned for. Details in
`docs/host-notes.md`; raw transcripts in `docs/measurements/`.

**The review changed the plan in seven places**, all recorded as decisions below or in the
weaknesses list: separable field geometry, the owned frame type moving to `src/core`, typed
flow links, the ST map becoming its own float-only descriptor, split cache lifetimes,
`Analyze` becoming `Precache`, and the removal of the claim that `Smooth` mitigates drift. It
also produced correction 6.

Worth recording about the process rather than the content: the review was read *against the
tree*, not accepted. Every claim it made about this repository was checked to a file and a
line before being acted on, and the one claim it made about the OFX specification turned out
to be wrong — as did my first reply to it. Both are cited to vendored headers now. A review
is a lead, and so is a reply to one.

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

A field carries a `FieldGeometry` and its *values are always in full-resolution pixels*. A
half-resolution result becomes a field whose vectors were scaled on arrival. From that point
every consumer — compose, ST map, resampler — works in one coordinate space and never learns
that an analysis scale exists.

The alternative, resampling each field up to full resolution on arrival, costs a copy per
chain link and filters the same data twice: once going up, once when it is sampled.

**Amended 2026-08-20: the geometry is separable, not scalar.** `FieldGeometry` was first
written as `{origin, spacing}` with one `double`. That silently assumes X and Y scale
identically, which is false for an anamorphic plate — already measured on this box at PAR 2 —
and for odd extents reduced by a fraction, and for asymmetric rounding. It is now
`{origin, spacingX, spacingY}`, checked at construction, because zero, negative, NaN and
infinite spacing all reach a division in `sample()`.

Deliberately separable and no further. A general 2D affine was proposed and declined: it
admits rotation and shear that nothing in this pipeline produces, and every consumer would
then carry the general case forever. If a rotation ever genuinely appears it is a new type,
not a generalisation of this one.

The related trap is that **an analysis "scale" cannot be a fraction.** PAR normalization
expands canonical width, so any fixed divisor applied afterwards inherits the expansion: a
2048×1556 PAR 2 plate is 4096×1556 canonical, and half of that is 2048×778 model pixels
against 1024×778 for the PAR 1 equivalent — still 2×, presenting as an unexplained OOM on
anamorphic shots only. The control is a megapixel cap. An area-preserving pitch of
`a·√PAR` in canonical space is correct arithmetic and was considered, but at PAR 2 it
supersamples X by 1.41 and undersamples Y by the same factor, discarding about 30% of the
vertical resolution present in the buffer at a setting the artist reads as "Full".

### A field is storage; a flow link is the unit of chain algebra

`Field<2>` aliased to `FlowField` left direction and meaning in comments, which makes several
expensive mistakes representable: treating an absolute map as a displacement, composing
`R→N` where `N→R` was needed, combining fields with different geometry. `FlowLink` carries
`fromTime`, `toTime`, both geometries and a model fingerprint, and `compose` rejects
mismatches rather than trusting the caller.

Direction errors are the likeliest bug in this design *because a reversed chain still produces
plausible motion* — the same property that made correction 5 dangerous. Putting the endpoints
in the type is what makes them checkable at all.

Confidence stays a separate parallel `Field<1>` rather than an optional member. It is
toggleable and recomputable without invalidating the flow it accompanies, and the common
no-confidence path should not pay for it in the cache or the budget.

### The ST map is its own descriptor, not an output mode

Flame reports `SupportsMultipleClipDepths = 0`. Per `ofxClipPreferences.rst`, that means every
clip shares one depth and the plugin may not remap them — so a single effect cannot serve
float ST output from a byte source. And ST data genuinely needs float: half's significand
gives roughly 4-pixel quantization near 1.0 at 4K, byte gives 256 levels, and relative-pixel
mode is signed and outside `[0, 1]` entirely, so an integer writer destroys it outright.

Three honest designs existed. Float across the whole plugin would force float buffers for a
byte source in Composite mode too — 4× the memory on 4K for a mode that does not need it. A
visible refusal at non-float depth is honest but leaves the artist stuck. A separate
float-only descriptor confines the cost to the output that requires the precision, and
incidentally resolves the Filter-versus-General UI problem: the original "Filter registered as
ST-Map-only" would have shown Composite and Warped Insert choices in a context with no Insert,
which `setEnabled()` is forbidden here to fix.

Both identifiers are permanent from the first artist build, which is why this had to be
settled before Phase 1 rather than after.

### `Analyze` became `Precache`, and its scope is gated on a measurement

The button was specced to "walk the range", which is undefined and could mean thousands of
frames. It now walks a chosen range — Current-to-Ref, Work Range, or Custom — and never the
full source range by default.

The name changed because a RAM-only pre-warm is what it honestly is. Whether it can be more
depends on 0C: if foreground and final render keep the same instance and process, a memory
precache is genuinely useful; if final render is another process, it has no production value
and the choice is between dropping the button and an explicitly user-managed disk cache.
Automatic persistence is the one option ruled out in advance — a cache that cannot detect an
upstream regrade silently serves stale flow, which is worse than no cache.

### CI gates on symbol exports, which warp-drive does not

`.github/workflows/ci.yml` asserts that each built `.ofx` exports **exactly**
`OfxGetNumberOfPlugins`, `OfxGetPlugin`, `OfxSetHost` and nothing else — on both platforms,
because they reach that result by different mechanisms (a version script on Linux, hidden
visibility plus explicit default visibility on the entry points on macOS), so one passing
is not evidence about the other.

warp-drive passes the version script and trusts it. That is reasonable there. Here it is
not: from Phase 3 the plugin links ONNX Runtime, which carries protobuf, abseil and a CUDA
runtime, all of which Flame may already have loaded at other versions. Two visible copies
of protobuf in one process is a crash with no useful backtrace and no obvious cause. A
linker flag that silently stopped being applied — a refactor of `OfxBundle.cmake`, a
platform branch not taken — would otherwise go unnoticed until a host crashed on somebody's
machine.

The gate was verified to fail on both a leaked third-party symbol and a missing entry
point before being committed. A gate that has never been observed to fail is not known to
be a gate.

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
  errors. Bounded only by keeping the reference frame near the working range; a fall-back to a
  direct `R→N` inference past a chain-length threshold is noted in the plan and not built.
  **The `Smooth` parameter does not mitigate this**, contrary to what the plan said until
  2026-08-20. Smoothing addresses local spatial noise; drift is accumulated systematic bias
  along the temporal axis, and blurring the field additionally softens motion boundaries,
  which makes foreground/background leakage worse. Claiming a knob fixes something it cannot
  is worse than admitting the gap, because it stops anyone looking for the real remedy.
- **Occlusion is opt-in.** The forward-backward check doubles inference cost, so it is off
  by default. Until an artist turns it on, a warped insert smears through an occlusion.
- **In-process inference is measured working for the CPU runtime only.** The CUDA execution
  provider is a separate library, and Flame exposes CUDA, cuDNN, cuBLAS and TensorRT
  globally as well; nothing has tested that combination. It is the last thing gating the
  inference design.
- **The isolation result rests on a 128-byte model.** Partial symbol capture could still
  produce correct arithmetic on one `Add` node. Worth re-running against a real network the
  moment there is one.
- **`RTLD_LOCAL` sufficing depends on ONNX Runtime's hidden visibility**, which is a
  property of their build rather than a guarantee. Re-check whenever the bundled version
  changes.
- **Model licences are read, not audited.** Every claim needs verifying against the actual
  repository and the actual checkpoint file before anything goes to a client — including
  backbone weights, which may carry different terms from the code that loads them. Secondhand
  licence claims, including those in the 2026-08-20 architecture review, are leads rather than
  findings. See `models/MODELS.md`.
- **No model default is chosen, deliberately.** Choice option order is API — a saved setup
  stores the index — so `model` and `inputCurve` get their options at Phase 2.5, from the
  bake-off, measured on the exact exported artifact rather than on upstream PyTorch.
- **The chain is a workaround for pairwise models.** A reference-frame tracker computes
  directly what `FlowChain` approximates. If one becomes viable at production resolutions,
  the chain, the drift work and the link cache all collapse into a single inference — so
  `FlowEstimator` must not be shaped so tightly around pairs that it forecloses one.
- **`cmake/ofx.map` protects the host from us, not us from the host.** It stops our symbols
  being visible in Flame's process; it does nothing about our references binding to Flame's
  copy of something we also carry. Measured harmless for the CPU ONNX Runtime (see
  `docs/host-notes.md`), and still unmeasured for the CUDA stack.

---

## Mistakes and corrections

### 1. `-static-libstdc++` needs a package that Rocky 9 does not install with the compiler

**Symptom:** the first CI dispatch failed at link with `/usr/bin/ld: cannot find -lstdc++`,
which reads like a broken or missing compiler. The compiler was fine.

`-static-libstdc++` needs `libstdc++.a`. On EL9 that archive is not in `gcc-c++` — it is in
`libstdc++-static`, which lives in the **CRB** repository and is disabled by default. So
the workflow now enables CRB and installs it explicitly, then asserts the archive exists so
a future repository reshuffle fails at the install step with a reason rather than at link
time with a misleading message.

warp-drive never hits this because it installs `gcc-toolset-12`, which carries its own
static libstdc++. Dropping the toolset — correct here, since Rocky 9's default gcc 11.5 is
already C++17-complete — quietly removed the thing that was satisfying the link.

The macOS job passed on the same commit, which is the giveaway: `-static-libstdc++` is
guarded by `UNIX AND NOT APPLE`, so only one platform could ever have shown this.

### 2. The glibc release number is not the glibc symbol-version number

**Symptom:** with the build finally linking, the gate rejected the artifact —
`needs glibc 2.35 but the baseline is 2.34` — on a binary built inside a Rocky 9
container. Both numbers were correct and the conclusion was wrong.

Measured in CI on Rocky 9.5: `ldd --version` reports **2.34**, and the same machine's
`libc.so.6` **defines symbol versions up to 2.35**. RHEL 9 pins the glibc *release* for the
life of the series but backports interfaces from later releases, and those arrive carrying
their upstream `GLIBC_2.3x` version tags. So an EL9-built binary can require a tag above
2.34 and load on every EL9 host.

**This conclusion was wrong, and correction 3 below is what it cost.** The reasoning
survives here because the shape of the error is worth keeping: I noticed the fix made the
gate "nearly tautological", wrote that down, tried to patch around it by pinning the
container image, and shipped it anyway. A check that cannot fail should have been the end
of the argument, not a caveat inside it.

**Also confirmed by the same run, both previously uncertain:**

- The Linux `.ofx` exports **exactly** the three OFX entry points. No `__bss_start`,
  `_edata` or `_end` leaked through — `local: *` in `cmake/ofx.map` covers linker-generated
  symbols, which was an open question when the gate was written.
- Its only runtime dependencies are `libm` and `libc`. No `libstdc++`, no `libgcc_s`, so
  `-static-libstdc++ -static-libgcc` is doing what it is there for.

### 3. Deriving the glibc baseline from the builder made the gate unable to fail

**Symptom:** CI green on every gate; Flame then refused to load the plugin with
`GLIBC_2.35 not found`. The artist sees nothing useful, and the gate that exists precisely
to prevent this had passed.

The offending symbol is **`_dl_find_object`**, pulled in by `-static-libgcc` — it is the C++
unwinder's fast path for FDE lookup, and Rocky 9.5's `libgcc.a` is built against a glibc
that has it. The certified Flame box, *also nominally Rocky 9.5*, has an older glibc that
does not. Autodesk certifies a point release and nobody runs `dnf update` on a Flame box,
so **a distro version is not a glibc version**.

The gate could never have caught that, because correction 2 had changed its baseline to be
read from the build container's own libc. That asserts "this binary needs nothing the
machine that built it provides" — true by construction, and therefore not a check. Pinning
the image to `:9.5` looked like it addressed the gap; it did not, because the container
tagged 9.5 is rebuilt with updates while the box is frozen at whatever shipped.

**Fixed by going back to what warp-drive already does**: build on `almalinux:8`, glibc 2.28,
hard-coded baseline. A binary built against 2.28 loads on every EL8 and EL9 host whatever
point release it sits at, so the entire class of problem disappears rather than being
tracked. gcc-toolset-12 also carries its own static libstdc++, which retires correction 1's
CRB dance as a side effect.

Three lessons, in descending order of how much they would have saved:

1. **A check that cannot fail is not a check.** I wrote "nearly tautological" in the commit
   message and shipped it anyway.
2. **Never derive a compatibility floor from the machine doing the building.** If the number
   is ever raised it must come from a measured target.
3. **Departing from a sibling project's proven configuration needs a stronger reason than
   "our target is newer".** warp-drive's EL8 choice looked like over-caution for a Rocky 9.5
   target. It is the thing that makes the artifact load.

### 4. Comparing against a display string skipped a whole Phase 0 question

**Symptom:** the probe report reads `Context is not General; second input clip not defined`
inside a section headed `Describe in context "OfxImageEffectContextGeneral"`. The header and
the verdict contradict each other on the same line of output, which is what gave it away.

The probe has a `readProp` helper that renders any property for a human, and it quotes
strings to do that. I fed its result straight into a comparison:

```cpp
const std::string context = readProp(inArgs, kOfxImageEffectPropContext, kStr);
gSecondClipDefined = (context == kOfxImageEffectContextGeneral);   // never true
```

The value under test was `"OfxImageEffectContextGeneral"` **with quote characters in it**, so
the branch could not fire in any context. The optional second clip was never defined, and a
whole Phase 0 item silently reported itself as untested rather than failing.

Fixed by reading the raw value with `propGetString` for the comparison and keeping `readProp`
for display, with a comment at the site saying why there are two reads of one property.

**The lesson is about the report, not the bug.** The probe printed both facts next to each
other and the contradiction was visible on inspection — a diagnostic that states what it
observed *and* what it concluded catches its own errors, where one printing only the
conclusion would have read as a clean host finding. That is worth preserving deliberately as
more probes get written.

Also worth noting: the user said immediately that multi-input OFX must work, because Mocha
and Silhouette both do it. That was right, and it is the reason this got looked at rather
than written down as a Flame limitation.

### 5. Comparing pixel coordinates against canonical coordinates invented a host limitation

**Symptom:** an anamorphic probe run reported `NOT HONOURED -- 21 of 21 renders were
partial. The render path needs tile assembly.` Every render was in fact a full frame.

```
render 1: window [0 0 4608 3164] rod [0 0 9216 3164] PARTIAL -- host tiled us
```

`kOfxImageEffectPropRenderWindow` is in **pixels**. `clipGetRegionOfDefinition` returns
**canonical** (square-pixel) coordinates. The clip is PAR 2, so 4608 pixels wide is 9216
canonical wide — the same rectangle, stated twice in different units, and the check compared
them directly. Fixed by converting the RoD to pixels
(`x_pixel = x_canonical * renderScale.x / par`) with a one-pixel slack before comparing.

**This is correction 4 again in a different costume.** Both are the same mistake: taking a
value out of one system and comparing it against a value from another without converting.
There the systems were "display string" and "property value"; here they are "pixel space"
and "canonical space". The second one is worse, because the wrong answer was *plausible* —
"Flame ignores SupportsTiles on anamorphic" is exactly the sort of quirk this host produces,
and it would have driven the render path to grow tile-assembly logic it does not need.

Two things that should have caught it earlier and now do:

1. **PAR 1 makes the two spaces identical**, so the first 47 renders passed by coincidence.
   Anything comparing rectangles in this codebase must be tested at PAR != 1, and the
   anamorphic clip has now earned a permanent place in the probe procedure.
2. **The probe printed only the conclusion**, not the conversion. It now prints the RoD in
   both canonical and pixel terms with the PAR it used, so the arithmetic is visible in the
   report and a wrong verdict can be checked rather than believed.

The warp-drive note that image bounds are in real pixels while project size is square-pixel
was already in `docs/host-notes.md`, inherited, and I had read it. Knowing a fact is not the
same as applying it at the one site where it matters.

### 6. Reaching for a measurement when the specification was already in the tree

**Symptom:** none yet — caught in review, before any code depended on it.

An advisory review asserted that concurrent same-instance renders "must be expected" under
`eRenderInstanceSafe`. I hedged it on empirical grounds: 47 measured renders in the probe
transcript, `setHostFrameThreading(false)` declared, so *let us see what Flame actually does*.

The answer did not need Flame. It is stated in a header vendored into this repository:

```
eRenderInstanceSafe  /**< can call a single render on an instance,
                          but can render multiple instances simultaneously */
```

`third_party/openfx/Support/include/ofxsImageEffect.h:94`, and `ofxImageEffect.h:768` says the
same. `eRenderFullySafe` is the level that permits several renders on one instance, and we do
not declare it. Both the review and my reply to it were wrong.

The consequence is a redistribution rather than a deletion, and it lands where the real
hazard is: per-instance state needs no same-instance render de-duplication, while
**process-wide state — `Ort::Env`, the session cache, the GPU semaphore — genuinely does need
to be thread-safe**, because separate instances may render at once. Designing the mutex around
the wrong one of those is how a plugin either deadlocks a host or corrupts a shared session.

**The lesson is the inverse of corrections 4 and 5, and worth stating precisely because it
runs against this project's grain.** "Measure, do not assume" exists because Autodesk's
documentation is wrong in both directions about *host behaviour*. It does not license
deferring to measurement when the **OFX specification** is unambiguous and sitting in
`third_party/`. Host behaviour is measured; protocol semantics are read. Confusing the two
costs a probe cycle at best, and at worst produces a "measured" conclusion that is really an
observation of one host on one day, generalised into a contract it never was.

