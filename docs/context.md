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

**The review changed the plan in nine places**, all recorded as decisions below, in the model
ledger, or in the weaknesses list: separable field geometry, the owned frame type moving to
`src/core`, typed flow links, the ST map becoming its own float-only descriptor, split cache
lifetimes, `Analyze` becoming `Precache`, removal of the claim that `Smooth` mitigates drift,
model selection moving to an artifact bake-off, and an honest split between pairwise and
future reference/window estimator contracts. It also produced correction 6.

Worth recording about the process rather than the content: the review was read *against the
tree*, not accepted. Every claim it made about this repository was checked to a file and a
line before being acted on, and the one claim it made about the OFX specification turned out
to be wrong — as did my first reply to it. Both are cited to vendored headers now. A review
is a lead, and so is a reply to one.

---

## Session 3 — 2026-08-21 — Phase 0B real-network run

The Flame run replaced the 128-byte `Add` model with the pinned, manifest-verified SEA-RAFT
M export. Through private ORT 1.29 opened under plain `RTLD_LOCAL` inside Flame 2026.2, the
real network passed identity and both translation directions on CPU and CUDA at 128×192.
CPU session/first-run timing was 941.6/529.4 ms with 0.0026 px identity EPE; CUDA was
934.8/1164.0 ms with 0.0027 px identity EPE. The CUDA provider itself came from the probe's
private tree. Flame's CUDA, cuBLAS and cuDNN libraries were already mapped before provider
session creation; cuDNN component libraries appeared after provider session/run. TensorRT
was also already mapped, but this run does not establish that the ORT CUDA EP selected or
used it. The raw transcripts are archived in `docs/measurements/`.

This closes the basic real-network CPU/CUDA inference question, not the whole 0B gate. The
timings are tiny-input first-run measurements, so warmed production-resolution performance
is still open. The CUDA log's nine inserted `Memcpy` nodes and CPU-assigned shape operations
are performance warnings, not correctness failures.

The initial shell closure found a 646,116,341-byte diagnostic payload, including a
31,958,904-byte probe-only duplicate of ORT; without that duplicate the payload was about
614.2 MB, with 103,095,040 bytes resolved externally. Its incomplete shell search path left
`libcublas.so.12`, `libcublasLt.so.12`, `libcudart.so.12` and `libcurand.so.10` unresolved,
and the probe's library filter omitted `libcurand`, so ownership was still open at this point.
The later live-loader-path report in Session 5 closes it. VRAM, repeated lifecycle/node
duplication, cross-thread cancellation, and provider-init/GPU-OOM fallback were also still
unmeasured at this point.

The apparent missing-model result was a permissions defect: the ONNX file was present but
mode 0600, readable only by its owner while the Flame runtime user was distinct. Staging it
mode 0644 fixed the run. This is now a packaging invariant and needs a CI guard (mode 0644,
or an equivalent ACL for the Flame runtime user), not a troubleshooting note.

---

## Session 4 — 2026-08-21 — multiresolution and bounded CUDA allocation failure

Follow-up Flame reports measured warmed CPU and CUDA inference at 480×640, 720×1280 and
1080×1920 on both the original and a duplicated probe node. Direction/identity, repeated
create/run/destroy, cross-thread cancellation and provider-init failure followed by CPU
fallback all passed. The duplicate used a distinct OFX instance handle and produced
equivalent results. These runs supplied the first useful-resolution timing and device-wide
NVML record, but intentionally stopped at 1080p on the current GPU.

The next run bounded ORT's CUDA arena to 64 MiB. The real SEA-RAFT session failed during
`CreateSession` inside `BFCArena::AllocateRawInternal`, with 1,931,264 bytes available for a
2,359,296-byte request. Device-wide NVML use was 2,395.9 MiB both before the attempt and
after cleanup; a fresh CPU session then passed numerical identity/direction, and the required
arena-limit gate passed. The raw transcript is archived in `docs/measurements/`.

This is the safe Phase 0B evidence we wanted: an explicit bounded allocator failure does not
poison later inference in the host process. It is not a device-wide OOM test and does not
exercise automatic production fallback or an artist-visible error path; those are Phase 4
shipping behavior. At this point 0B remained open for CUDA payload closure and qualification
above 1080p. Session 5 closes the former; UHD, DCI 4K and the 3164×4608 H×W Alexa 35
open-gate case remain. Those qualification targets are measurements, not product hard caps.

---

## Session 5 — 2026-08-21 — actual Flame loader-path CUDA closure

The first two closure reports used only the payload plus generic system library paths, which
correctly described those shell environments but not Flame's loader. Re-running the reporter
with the live Flame process's actual loader search path resolved all four previously missing
SONAMEs from `/opt/Autodesk/lib64/2026.2.1/`: cuBLAS 12.8.3.14, cuBLASLt 12.8.3.14,
CUDA runtime 12.8.57 and cuRAND 10.3.9.55. The authoritative report ends with no unresolved
dependencies.

The installed diagnostic payload is 646,116,476 apparent bytes and 646,123,520 allocated
bytes. Its second 31,958,904-byte ORT is probe-only, leaving 614,157,572 bytes without it.
The unique external transitive closure is 1,138,007,368 bytes: 1,034,912,328 bytes of
Flame-owned CUDA libraries and 103,095,040 bytes of driver/system libraries. Those external
libraries are ownership accounting, not payload to copy into the bundle. This closes the 0B
dependency ownership and size question for this Flame 2026.2.1 / ORT 1.29 CUDA 12 pairing;
at the end of this session, only higher production-resolution qualification remained in 0B.

---

## Session 6 — 2026-08-21 — bounded full-resolution qualification

A clean no-environment launch first re-ran the ordinary extended path and passed the existing
CPU/CUDA correctness, lifecycle, cancellation, provider-fallback and 64 MiB arena-recovery
gates. Three fresh GPU-only launches then attempted UHD 2160×3840, DCI 4K 2160×4096 and
Alexa 35 open gate 3164×4608 with a 16 GiB ORT arena ceiling. The Alexa source was correctly
replication-padded by four bottom rows to the network's 3168×4608 tensor contract.

All three warm runs stopped inside `BFCArena` with classified bounded-allocation outcomes.
UHD requested 16,796,160,000 bytes with 3,095,502,080 available; DCI 4K requested
19,110,297,600 with 1,423,074,560 available; and Alexa requested 13,006,946,304 with
4,298,024,192 available. The failures occurred at different fused matrix multiplications, so
the individual rejected allocations are evidence of why each run stopped, not estimates that
can be compared as total model memory. No run produced steady samples. The reported
704.8–866.7 ms durations are time to failure, and boundary-sampled NVML does not capture the
rejected allocation. Each session-cleanup sample was +2 MiB from its pre-session value; that
is clean bounded teardown evidence, not a device-wide leak proof because Flame and the shared
ORT environment remained live.

The probe's required measurement-result gate intentionally accepts either a completed
inference or a classified bounded stop, and it passed in all three valid runs. This therefore
closes the last 0B measurement with a negative fit result for the current 16 GiB arena
configuration. It does not cap source formats in the product or say what a future GPU and
larger configurable budget can run. The Alexa transcript also retains an initial invalid
configuration session before the corrected valid session because the report file appends.

## Session 7 — 2026-08-21/22 — Phase 0C closed, and the precache decision

**0C item 1 (ST convention)** was measured with a purpose-built harness (`tools/stprobe/`)
that avoids a self-resampler round trip: a coordinate-encoded float plate warped through
Flame's *own* consumers — the native ST Map node and Action's UV map — and read back as float
EXR, so the output decodes exactly to the source pixel Flame fetched. Both consumers are
identical inside `[0, 1]`: `(x+0.5)/W` half-pixel centres (fit residual 0.000 px), bottom-left
origin, U→R/V→G, real-pixel normalization, backward-map semantics. They differ only outside the
range — the ST Map node blacks, Action mirrors. First run mistakenly fed the human-readable
landmark plate instead of the coordinate plate; that still fixed origin/channel/direction, and
the re-run against the coordinate plate pinned the rest.

**0C items 2–5** were measured by an extended `hostprobe` built by two Sonnet subagents in
parallel worktrees (instance/process lifetime; and the render-scale/rowBytes/anamorphic
observations) and integrated here. Item 2 is the load-bearing result: **background/final render
runs in a separate process — Autodesk Burn (`com.autodesk.backgroundreactor`, `IsBackground=1`)**
— which rendered the full range while the foreground Flame process held the interactive
instances. Duplicating a node makes a new instance in the same process; reopening Flame is a new
process. Items 3 and 4 closed negative across 122 renders in both hosts (render scale always
`[1,1]`, no negative rowBytes, no sub-window images), and item 5 — the anamorphic tile re-check
— closed on a follow-up PAR 2 run (all 51 frames full-frame), retiring the old "PAR-2 looked
tiled" scare as the coordinate-system bug it always was. With that, **all of Phase 0 is closed.**

The precache decision followed from item 2 plus a workflow fact from the facility — see
*`Analyze` became `Precache`* under Decisions. Short version: **no persistence in v1.** The
separate-process final render would have justified a persistent cache, but the facility renders
almost everything in the foreground (same process, cache already warm) and uses single-node Burn
rarely and never fanned out, so RAM covers it and a disk cache would be pure staleness risk.

## Session 8 — 2026-08-22 — Phase 1 implemented, reviewed, measured and merged

Phase 1 created the first product bundle. One `WhiteWater.ofx.bundle` now enumerates two
permanent descriptors: Track/Insert at `com.mtifilm.whitewater.opticalflow`, and a separate
float-only ST Map at `com.mtifilm.whitewater.stmap`. The real inference pipeline is still absent
by design; Phase 1 freezes the host-facing contract with deterministic outputs: Composite copies
Source, Warped Insert copies the selected Current or Reference Insert (or black when disconnected),
and ST Map emits an exact identity field.

Three Luna-max work packages built the descriptor/fallback contract, raw OFX host harness, and
bundle/boundary gates. Integration moved `OwnedFrame` below the OFX boundary, added independent
core and infer dependency policies with expected-failure fixtures, and asserted the exact three
module exports on both platforms. A clean Release build passed all eleven tests, including the
Flame-name and refused-string-mode harness variants.

Review caught four changes worth keeping: Set Ref now replaces a non-animating scalar instead of
planting keys; the ST expected-U test matches production double-then-float arithmetic; dead host
harness ternaries were removed; and both row renderers share one worker partition/abort/write-back
base. A proposed Output-dependent ROI/frame-needs optimization was deliberately rejected: both
Source N and Insert N/R feed the future flow chain even though the Phase 1 fallback copies only
one final input.

The EL8/glibc-2.28 artifact was then exercised in Flame 2026.2. Both nodes and their sockets
appeared; labels and descriptor-specific controls were legible; deferred bake-off choices stayed
absent; Model Dir was safe; repeated Set Ref presses replaced one Batch-relative value; Current
and Reference Insert routing were distinct; disconnected colour and matte were black; connected
Insert matte propagated correctly; partial renders were seamless; and the default absolute-UV,
bottom-left ST output reproduced Source through Flame's native ST Map node. Load diagnostics
recognized Flame and reported no recovered describe action.

That run exposed one small descriptor omission: the product nodes appeared in an unnamed OFX
submenu because they lacked `kOfxImageEffectPluginPropGrouping`, while both probes set it to
`White Water`. The property and a harness assertion were added. The fix was accepted without a
second airgapped artifact cycle because it is the same standard property already proven by both
probes and is easy to correct later if Flame still presents it differently.

PR #1 merged Phase 1 to `main` at `5fa267f`; its topic branch was deleted. At that point Phase 2
— host-free flow algebra, `NullPairwiseEstimator`, the offline CLI and their expanded tests — was
the next implementation phase; it subsequently merged through PR #4.

## Session 9 — 2026-08-22 — Phase 2.5 P25-0 protocol package under review

P25-0 is implemented on the topic branch for PR #6 and remains unmerged pending human review.
The package freezes the Phase 2.5 decision record in `docs/phase2.5-protocol-v1.md`, the
executable protocol in `bakeoff/protocol-v1.json`, versioned corpus/report schemas, and the
standard-library-only validator/CLI in `tools/bakeoff/`. Positive and negative fixtures cover
schema and semantic failures, and the `bakeoff::protocol_schema` CTest registration keeps the
package in the ordinary repository gate.

This is a measurement contract, not a candidate result: no model/default or fast alternative,
artifact manifest/export, airgapped target measurement, ranking decision, or persistent OFX choice
order is selected. P25-1 and the later bake-off packages remain open, and production inference is
still not wired into the OFX product.

## Session 10 — 2026-08-23 — pre-target admission amendment

The first protocol overloaded shipping/license eligibility with technical measurement admission:
an artifact with valid export evidence but unresolved checkpoint or backbone terms disappeared from
the matrix. Because that distinction changes report semantics, v2 is additive and versioned rather
than silently mutating v1. `status` remains the fail-closed shipping gate; v2 adds required typed
`measurement_status`, and matrix planning admits measurable excluded candidates while rejecting
unavailable artifacts. Unavailable candidates carry a separate typed measurement reason, while
excluded-but-measurable entries retain complete legal verdict and redistribution-review surfaces;
the shipping exclusion reason never doubles as a technical admission signal. Shipping eligibility
implies measurable, and the validation baseline remains evaluation-only. The unchanged corpus stays
v1 and is bound by hash. NeuFlow's fixed 432x768 export now has an explicit v2-only constrained
comparison lattice: `mp0_331776` computes to exactly 768x432 at canonical 16:9, and candidate
constraints reject every other cap/provider/source geometry before rows are generated. The shared
cap is available to other measurable models for a fair comparison, while the unchanged final
shipping `mp2` coverage remains separate. Candidate report entries also carry provider-specific
technical measurement evidence; the checked-in NeuFlow evidence is CPU only, so no CUDA pass is
implied.

---

## Session 11 — 2026-08-23/24 — P25-1 through P25-4 and evaluation-only candidates

P25-0 merged through PR #6, P25-1 through PR #8, and P25-2 through PR #7. The repository now has
the frozen v1 measurement contract, candidate-neutral artifact validation, and deterministic
corpus/conditioning inputs. P25-3 merged through PRs #11, #10 and #12; P25-4 and active protocol
v2 merged through PR #13 on 2026-08-24. P25-5 local/CI qualification is the next implementation
package; no target measurement or model selection has occurred.

The candidate policy now distinguishes technical measurement from shipping admission. Original
RAFT is a formal validation baseline, while NeuFlow and WAFT remain useful comparison candidates.
Unknown or restrictive checkpoint terms can exclude any of them from packaging and selection
without erasing valid numerical evidence. The common bake-off should therefore compare technically
qualified SEA-RAFT, RAFT, NeuFlow and WAFT artifacts where practical, even if SEA-RAFT is the only
model that ultimately ships. A fast alternative is optional: if none passes every gate, the plugin
may ship one model and omit the `model` choice rather than publish a meaningless one-option API.

---

## Session 12 — 2026-08-24 — P25-5 portable runtime boundary

P25-5 carries its Python/ONNX Runtime user-space environment as an opaque, hash-bound
`conda-pack` archive built in the EL8 CI lane. The airgapped wrapper extracts it only into a
writable operator-selected directory, runs `conda-unpack` once, sanitizes Python/conda/pip and
proxy state, and invokes the carried evaluator without installing or downloading anything. The
target still owns the NVIDIA kernel driver and `libcuda`; CUDA, cuDNN and other user-space
dependencies must either resolve inside the packed environment or remain an explicit failed
closure gate.

This is evaluation infrastructure, not a shipping decision. The deterministic outer package
records source, staged, archive and extracted identities and keeps measurement admission separate
from licence eligibility. The CI lane intentionally cannot emit the final tarball until its
explicit conda lock, package specification, run instructions, and complete candidate/runtime
licence-and-notice set have been reviewed. In particular, a passing local CPU/provider check does
not close the CUDA target qualification and does not select an OFX model or choice index.

---

## Session 13 — 2026-08-24 — P25-5 qualification exit

Andrew Miller approved both the pinned SEA-RAFT legal surfaces and the generated runtime legal
inventory. Workflow-dispatch run `32780658875` then passed the ordinary suite, exact export and
admission checks, conda-pack relocation, native ONNX Runtime 1.29.0 CPU session, runtime licence
inventory, glibc 2.28 ELF floor, dependency-closure audit, package construction and artifact
upload. The reviewed runtime inventory SHA256 is
`49ef6e85b032f64b8bfa62b939274aeee9d59c55a0f83f6fddb03d7d9aadecee`.

The uploaded artifact `whitewater-p25-5-el8-adfd4fb85ce319bfc76468a9d097f514901405c9`
contains the 638,391,000-byte evaluation tarball whose SHA256 is
`53b4e7192496a5ee8be1e0af6085980b59e287afc03599f953e2bb9a65eb8850`. An independent download
revalidated its checksum and package inventory and verified all 31 carried files after extraction.
This satisfies the P25-5 exit. It makes no target CUDA, performance, shipping-default, or
persistent OFX choice claim; those remain P25-6 and later decisions.

PR #15 merged to `main` at `946c042`. Its final follow-up review commit `0246b92` corrected the
CI lock containment guard, bounded component hashing memory without changing the digest contract,
made runtime-review timestamps timezone-qualified, and fixed the airgap-wrapper zero-match test.
None of the package specification's 30 carried sources or the native ORT bridge changed after the
qualified `adfd4fb` artifact, and the approved runtime inventory/review hashes still validate, so
no replacement airgap artifact was generated.

The carried `RUN-P25-5.txt` is a P25-5 qualification handoff: checksum/extraction, required CPU
provider/session verification, and an optional single PFM-pair smoke command. It is not the full
P25-6 production operator procedure. P25-6 begins by supplying the resumable `smoke`, `screen`,
and `final` entrypoints and exact onsite instructions before target measurements are run.

**Returned airgapped verification — 2026-08-24:** Andrew ran the required `verify --provider cpu`
command from the P25-5 evaluator handoff on the airgapped target. The returned JSON is preserved
unchanged at `docs/measurements/2026-08-24-p25-5-airgap-verify-cpu.json`; its independently returned
and locally recomputed SHA256 is
`b8e7bc88f9672d447bc1b9189e47c3faa7808907d78d60187b9ddfaaa3408888`. It binds the qualified
Linux SEA-RAFT manifest and 78,840,944-byte artifact hashes, reports ONNX Runtime 1.29.0, selects
only `CPUExecutionProvider`, and exposes the expected two-input/one-forward-flow tensor contract.
This is successful target transfer, runtime relocation and CPU session/metadata evidence. It runs
no image pair and makes no CUDA, inference-quality, latency, memory, production-footage or P25-6
completion claim.

---

## Session 14 — 2026-09-01 — P25-6 final attempt and publication correction

The latest operator attempt's evaluator and runtime hashes match workflow run `33448282919` from
source commit `b5b4cbf` (qualified outer SHA256 `ffe36cf5...`), but the operator-owned report
metadata retained `source_commit: a8e974d`, the earlier package commit. Since the report binds
measurements to source identity, the returned evidence is diagnostic only and is not formally
admissible qualification evidence. No replacement outer bundle containing the publication fix
below has yet been qualified or run.

The idle FHD and UHD cells completed at `21.967529296875` GiB and `21.7393798828125` GiB
incremental device memory respectively, against the 15 GiB resource gate. Both are completed
quality-gate overruns, not allocator failures. Beside a live Flame workload, first inference hit
an explicit `3,940,826,368`-byte ONNX Runtime `BFCArena` allocation failure, which is a runtime
allocation-exhaustion outcome.

The first publication attempt aborted because the completed idle overruns were still labeled
`pass`; publication therefore rejected them only after measurement had been committed. The
taxonomy/publication fix moves the disposition into the cell result: a completed resource
overrun is `quality_gate_failed` with `stage: resource`, while its timing, metrics and full NVML
evidence remain attached and a failed package can be published. Explicit native/ORT allocation
exhaustion is typed `out_of_memory` at the relevant runtime stage; generic runtime failures stay
`runtime_error`. This fixes the reporting contract without turning the stale-metadata diagnostic
attempt into a qualified result.

## Session 15 — 2026-09-01 — P25-6 final run completed, and the memory-gate baseline artifact

A second 2026-09-01 run, published for review in PR #31 under
[`measurements/2026-09-01-p25-6/`](measurements/2026-09-01-p25-6/), is the first to complete all
four final CUDA cells with accepted NVML evidence. Its `runner.source_commit` is `5731b2d` — main
HEAD, the #30 merge carrying the publication fix — so the runner axis no longer records the stale
`a8e974d` that made Session 14's attempt diagnostic-only. (Formal admissibility still turns on the
outer archive matching its qualifying dispatch; confirm that against the sidecar before treating
these overruns as qualification evidence rather than the runner binding alone.) CPU smoke and
screen pass 1/1 and 2/2. Both `live_flame` final cells now **pass** at ~11.4 GiB incremental
device memory and ~278 ms steady — the Session 14 live-Flame `BFCArena` allocation failure did not
recur — while both `idle` cells **fail** the 15 GiB resource gate at 21.837 (FHD) and 21.800 (UHD)
GiB incremental, typed `quality_gate_failed`/`stage: resource` with full timing, metrics and NVML
retained, exactly as the #30 taxonomy intends. Quality is excellent in every cell (endpoint error
≈ 0.001 px, `fraction_le_1px = 1.0`).

**The finding worth carrying into P25-7:** the resource gate keys on
`peak_incremental_device_memory = peak_device − baseline_device`, and that subtraction, not the
candidate, decides pass/fail here. Absolute peak device memory is ~21.6–23.4 GiB of the A5000's
24 GiB in **both** host loads. Under `idle` the baseline is ~1.08 GiB, so the increment is
~21.8 GiB and the cell fails; under `live_flame` Flame is already ~9.99 GiB resident, so the same
workload's increment is only ~11.4 GiB and the cell passes. A heavier host therefore makes the
candidate *look* lighter to the gate, and pass/fail flips on host load rather than on candidate
behaviour — the candidate genuinely wants ~22 GiB of a 24 GiB card either way, i.e. near-OOM
regardless of load. Whether P25-7 should gate on absolute peak (headroom to a device OOM) instead
of, or alongside, baseline-subtracted increment is a ranking/selection decision, not a change to
this evidence. No model, default, target result or OFX choice index is selected here.

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
frames. It now walks a chosen range — Current-to-Ref, Work Range, or explicit Custom start
and end parameters — and never the full source range by default.

The name changed because a RAM-only pre-warm is what it honestly is. Whether it can be more
was gated on 0C.

**Decided 2026-08-22: v1 ships the RAM-only precache and no persistence at all.** 0C measured
that final render *is* a separate process (Autodesk Burn), which is what would have justified a
persistent cache — but two facts remove the justification for *this facility*. First, the same
0C run showed a single Burn process rendering a shot **sequentially**, so it reuses its own
in-memory chain cache (one inference per frame after the prefix); the process boundary costs
only the one-time chain-prefix rebuild, not a full re-analysis. Second, and decisively, the
facility **rarely uses Burn at all — almost every render is foreground, in the same process and
instance the artist scrubbed with — and when Burn is used it is a single node, never fanned out
across a farm.** The one case a persistent cache clearly wins — many farm nodes each rebuilding
the prefix from the reference — does not occur here.

So a disk cache would buy, in practice, only *instant reopen after a Flame restart* — which is
exactly where automatic persistence is most dangerous (overnight the plate may be reconformed
or regraded, and the cache would silently serve stale flow), and where even a user-managed one
leans on the artist remembering to `Clear`. Paying a few seconds of re-analysis on reopen to
stay always-correct is the right trade for a correctness-first plugin. The disk cache is kept
only as a **future option gated on one fact changing: if final renders ever start fanning out
across multiple Burn nodes.** Until then, `Precache` and `Clear` operate on the instance-lifetime
RAM cache, and that is the whole story.

### CI gates on symbol exports, which warp-drive does not

`.github/workflows/ci.yml` asserts that each built `.ofx` exports **exactly**
`OfxGetNumberOfPlugins`, `OfxGetPlugin`, `OfxSetHost` and nothing else — on both platforms,
because they reach that result by different mechanisms (a version script on Linux, hidden
visibility plus explicit default visibility on the entry points on macOS), so one passing
is not evidence about the other.

warp-drive passes the version script and trusts it. That is reasonable there. Here it is
not: from Phase 3 the plugin privately loads ONNX Runtime, whose dependency closure carries
protobuf, abseil and a CUDA runtime that Flame may already have loaded at other versions.
The module itself retains no `DT_NEEDED` entry for ORT, but two visible copies of protobuf
in one process are still a crash with no useful backtrace and no obvious cause. A
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
  direct `N→R` inference past a chain-length threshold is noted in the plan and not built.
  **The `Smooth` parameter does not mitigate this**, contrary to what the plan said until
  2026-08-20. Smoothing addresses local spatial noise; drift is accumulated systematic bias
  along the temporal axis, and blurring the field additionally softens motion boundaries,
  which makes foreground/background leakage worse. Claiming a knob fixes something it cannot
  is worse than admitting the gap, because it stops anyone looking for the real remedy.
- **Occlusion is opt-in.** The forward-backward check doubles inference cost, so it is off
  by default. Until an artist turns it on, a warped insert smears through an occlusion.
- **The real SEA-RAFT M network now passes in-process on both CPU and CUDA.** Warmed
  480p–1080p timing/VRAM, repeated lifecycle and node duplication, cross-thread cancellation,
  provider-init fallback, bounded arena-limit/CPU recovery, and exact CUDA dependency
  closure/ownership are measured too. UHD, DCI 4K and Alexa 35 full-resolution attempts all
  reached controlled bounded-allocation stops under the 16 GiB arena ceiling. That closes 0B
  but leaves the practical shipping analysis cap to the model bake-off and performance gate.
- **`RTLD_LOCAL` sufficing depends on ONNX Runtime's hidden visibility**, which is a
  property of their build rather than a guarantee. Re-check whenever the bundled version
  changes.
- **Survey licence claims are leads; exact manifests own the audits.** P25-3 verifies the pinned
  repository, checkpoint and backbone surfaces separately. Unknown checkpoint terms fail closed
  for shipping without automatically blocking evaluation. Before anything goes to a client, the
  exact packaged files and required notices still need human review. See `models/MODELS.md`.
- **The resource gate is baseline-subtracted, so host load can flip a verdict.** The P25-6
  gate keys on `peak_device − baseline_device`, not absolute peak. On the RTX A5000 (24 GiB),
  SEA-RAFT M's ~22 GiB peak fails the 15 GiB gate from an idle ~1 GiB baseline but passes from a
  live-Flame ~10 GiB baseline, though the absolute footprint — and the near-OOM headroom — is the
  same. P25-7 must decide whether headroom-to-OOM (absolute peak) belongs in the gate. See
  Session 15.
- **No model default is chosen, deliberately.** Choice option order is API — a saved setup
  stores the index — so `inputCurve` and any multi-model `model` choice get their options at
  Phase 2.5, from the bake-off, measured on the exact exported artifact rather than on upstream
  PyTorch. If only one model qualifies, the Model choice is omitted.
- **The chain is a workaround for pairwise models.** A reference-frame tracker computes
  directly what `FlowChain` approximates. If one becomes viable at production resolutions,
  the chain, the drift work and the link cache all collapse into a single inference. The v1
  interface is therefore named `PairwiseFlowEstimator` honestly; a window model gets a
  separate `ReferenceFlowEstimator`, while `FlowPreparation` owns the choice of strategy.
  Forcing both through `estimate(a, b)` would erase the temporal context that distinguishes
  the second model class.
- **`cmake/ofx.map` protects the host from us, not us from the host.** It stops our symbols
  being visible in Flame's process; it does nothing about our references binding to Flame's
  copy of something we also carry. Measured harmless for the CPU ONNX Runtime (see
  `docs/host-notes.md`), and the real SEA-RAFT M CUDA run also passed on this exact Flame
  build. That is a qualification of this runtime/build pair, not a guarantee for future ORT
  provider builds.

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

### 7. An owner-only model file looked absent to Flame

**Symptom:** the first 0B run reported the model as absent even though the ONNX file had been
staged into the bundle. The file was mode `0600`: owner-readable, but not readable by the
distinct Flame runtime user. Changing it to `0644` made the same probe load the model and
pass its real-network checks.

The lesson is that an artifact can be present and still be absent from the host's point of
view. Model staging must set mode `0644` (or an equivalent ACL for the Flame runtime user),
and CI must assert that permission on the staged files. A packaging guard turns a silent host
failure into a build failure.

### 8. OpenImageIO dragged a GPL media/codec/GPU stack in for an EXR-only need

**Symptom:** none in a run — a dependency-and-licence problem, caught while assembling the P25-6
airgap runtime. The runtime uses its EXR reader for exactly one thing: decode one-file-per-frame
OpenEXR sequences into `rows`/`width`/`height`. The first cut got that decode from OpenImageIO
(`py-openimageio`), the obvious "read any image" library.

The cost is in what conda-forge's `openimageio` *transitively* pulls onto EL8: a full
media/codec/GPU stack that has nothing to do with EXR — `ffmpeg` (a **GPL** build),
`aom`/`dav1d`/`libvpx`/`libavif`, `intel-media-driver`/`intel-gmmlib`/`level-zero`/`libva`,
`libraw`, `libass`, `harfbuzz`/`freetype`/`fontconfig`. For a commercial Flame plugin that bloats
the carried runtime, explodes the runtime legal-review surface (every one of those needs a licence
verdict), and the GPL `ffmpeg` is a likely shipping blocker on its own.

**Fixed by decoding EXR directly with the official OpenEXR Python bindings** (conda-forge
`openexr-python`, the `OpenEXR` module, modern `OpenEXR.File` API). It needs only
`openexr` + `imath` + the python binding — all permissive (BSD-3-Clause) — and carries no codec or
GPU natives. `tools/bakeoff/exr.py` reads each channel as a numpy array whose dtype *is* the
on-disk storage class (`float16` → `"half"`, `float32` → `"float"`), so the existing
channel/format-classification and bottom-origin-row helpers were reused unchanged; the public
`frame_from_exr` dict shape, the file-bytes `sha256`, and the injected-decoder test design are all
preserved. The bake-off never needed to read anything but EXR, so "read any image" was always more
library than the job required.

The lesson: pick the dependency scoped to the need, not the most general one. A general media
library's transitive closure is a licence-and-size liability you inherit in full even when you call
one narrow entry point of it.

### 9. A schema-valid corpus failed only after the expensive measurement

**Symptom:** caught during the airgapped operator handoff, before a target profile ran. The first
qualified P25-6 archive (`06a72bb5…`) carried a small corpus containing the two selected synthetic
performance shots and one production smoke shot. Its input test checked the JSON schema and matrix
planning, so it passed CI. Report publication applies the stronger protocol-consistency gate: exact
coverage of all 23 frozen synthetic cases and all nine production categories. The carried
`synthetic-lattice` and `smoke-sample` categories were not protocol tokens, and fixing those names
alone would still leave the coverage incomplete.

The timing made this worse than an ordinary bad template. Matrix planning accepts an explicit
subset, while full corpus validation previously happened during report assembly. An operator could
therefore finish the costly cells and discover that no valid report could be published.

**Fixed at both boundaries:** the carried corpus now contains the complete frozen synthetic
partition and nine explicit production records; its test invokes `validate_corpus_consistency`,
not just the schema validator. The driver runs that same gate before artifact loading, resume-state
creation or inference. CI also fills the source commit, driver hash and runtime hash in the carried
report metadata, leaving the operator only truthful shot and hardware metadata. The flawed archive
is withdrawn and a replacement must be qualified; editing it in place would destroy the package
identity the report is meant to preserve.

The lesson: validating a selector is not validating the document the final report binds. Any
expensive resumable workflow must run its publication-strength input gate before its first cell.

### 10. Flame's CUDA directory also supplied an incompatible ONNX Runtime

**Symptom:** the P25-6 target CPU smoke and screen completed, but the first CUDA final invocation
could not load the carried native bridge. `ldd` selected
`/opt/Autodesk/lib64/2026.2.1/libonnxruntime.so.1` and reported that `VERS_1.29.0` was not found,
even though the package carried the required ORT 1.29 library beside the bridge.

The runbook had said the runtime and Flame CUDA directories held disjoint SONAMEs. That was false:
the Flame directory needed for `libcublas.so.12`, `libcublasLt.so.12`, `libcudart.so.12` and
`libcurand.so.10` also contains Flame's own ORT. The bridge's `$ORIGIN/onnxruntime` `RUNPATH` loses
to `LD_LIBRARY_PATH`, so exporting only the Flame directory chose the host ORT before the carried
one.

**The measured workaround is an ordered composition:** put the carried
`runtime-env/lib/whitewater/ort-cuda12/onnxruntime` first, Flame's CUDA directory second, and let
the wrapper prepend the runtime's top-level `lib`. The CUDA preflight must inspect the bridge as
well as the provider: `libonnxruntime.so.1` must resolve inside the carried native directory and
neither `ldd` closure may contain `not found`. After that ordering change the run advanced through
bridge loading to CUDA session creation, where it exposed the separate no-CPU-node-fallback defect
that blocks the qualified package's final profile.

The evaluator had turned a provider-priority requirement into a graph-partitioning prohibition:
it requested CUDA first but also set `session.disable_cpu_ep_fallback=1` in both the Python path
and native bridge. SEA-RAFT's already-measured shape/housekeeping nodes are intentionally assigned
to CPU by ORT, so session creation correctly refused that contradictory configuration. The
replacement removes only that session setting. Python ORT still enforces that a CUDA request report
`CUDAExecutionProvider` first through `session.get_providers()`; a CPU-first result is rejected.
The native bridge's `selected_providers` response instead echoes the requested provider after
session creation, so it is nominal/request-contract data rather than an observation of graph
placement. Native CUDA session creation proves provider loading and session creation, not that
graph work ran on the GPU. The target's required final NVML rows (baseline, session_create, steady,
cleanup, and process_exit) are the direct execution evidence; timing is performance evidence, not
proof by itself. This permits lower-priority per-node CPU work without treating a whole-session CPU
fallback as CUDA evidence. The rebuilt bridge changed the runtime identity and legal inventory,
so the old runtime review could not authorize it; CI generated the replacement inventory and
Andrew approved its exact `c7b36cc1…` digest before packaging. Follow-up changes to package-carried
sources still require a freshly qualified outer archive before another target run, even when they
do not change that approved runtime inventory.

The lesson: a directory selected for four known dependencies is still a loader namespace, not a
bag of only those four files. Verify ownership of the primary runtime library as well as absence
of unresolved transitive dependencies; a clean provider-only `ldd` is not enough.
