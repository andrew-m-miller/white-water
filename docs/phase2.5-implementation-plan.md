# Phase 2.5 implementation plan

Phase 2.5 selects White Water's default and fast optical-flow models by testing the exact
exported ONNX artifacts. It also fixes the artist-visible input-conditioning contract and
practical analysis megapixel caps. Selection is based on production-relevant quality first,
then target latency, memory, reproducibility and an audit of the exact code, checkpoint and
backbone terms.

This phase starts from Phase 2 at merge `4495224`: the host-free flow algebra, caches,
preprocessing, `PairwiseFlowEstimator`, `NullPairwiseEstimator`, and the ORT-free `ww-flow`
PFM CLI are present. **P25-0, P25-1 and P25-2 are merged through PRs #6, #8 and #7
respectively.** They provide the frozen protocol and schemas, candidate-neutral artifact
framework, and deterministic corpus/conditioning package. P25-3 candidate export/evidence work
merged through PRs #11, #10 and #12, and the P25-4 offline runner and active protocol v2 merged
through PR #13. P25-5 local/CI qualification merged through PR #15 at `946c042`; the exact
evaluation artifact remains the one qualified at `adfd4fb`. P25-6 airgapped target measurement is
next. No production target measurement, ranking decision, shipping model, or OFX choice order has
been selected, and broader closeout for Phases 2 and 2.5 remains deferred while the bake-off is
open.

The successful P25-5 workflow-dispatch run is `32780658875`. Its uploaded artifact is
`whitewater-p25-5-el8-adfd4fb85ce319bfc76468a9d097f514901405c9`; the exact carried evaluator
tarball is 638,391,000 bytes with SHA256
`53b4e7192496a5ee8be1e0af6085980b59e287afc03599f953e2bb9a65eb8850`. The reviewed runtime
licence inventory remains bound to SHA256
`49ef6e85b032f64b8bfa62b939274aeee9d59c55a0f83f6fddb03d7d9aadecee`. This closes P25-5's
local/CI and carried-tarball exit only; it is not a target CUDA result or shipping selection.

## Exit

Phase 2.5 exits when:

- one quality-first default has passed the exact-artifact, licence, quality and target-resource
  gates, and a genuinely faster alternative is selected only if it passes the same gates;
- the model artifacts, manifests, runtime and measurement environment are identified by hashes
  rather than model-family names;
- the input-conditioning formulas and common artist-visible option order are fixed;
- practical square-pixel analysis megapixel caps are measured on the target hardware;
- the final selection report has been reviewed by the human before any persistent OFX Choice
  order is published; and
- the selected manifests and reports form a complete, explicit handoff to Phase 3.

If no candidate satisfies the fast-alternative gate, do not invent a second option or delay a
sound one-model release indefinitely. Record the failed fast-candidate search and omit the
artist-visible `model` choice. Choice option order is saved-setup API and must not encode a result
the measurements did not produce.

## Decisions fixed before implementation

### Production footage stays on the airgapped machine

The authoritative production-quality evaluation runs on the facility's airgapped Flame box,
where representative footage already exists. Hunting public clips is not a prerequisite for
the bake-off. Repository-generated synthetic cases provide deterministic regression coverage;
public benchmark subsets remain optional supporting evidence after their terms and training-set
overlap have been recorded.

The evaluator is delivered as a self-contained EL8/glibc-2.28-compatible tarball with the exact
candidate artifacts, qualified runtime, manifests, checksums and commands. It must not depend on
network access or an unrecorded Python/runtime installation on the Flame box. A standalone
`ww-flow`-family bake-off executable may link the test runtime directly because it is an offline
tool; the shipping OFX module, the existing null-estimator CLI, and `src/infer` do not gain a
production ORT dependency in this phase.

The evaluator never uploads or embeds source frames in its reports. The operator returns only
machine-readable results, logs, measurements and optional review scores.

### Production input format

Use one-file-per-frame OpenEXR sequences:

- RGB or RGBA, half or float;
- original resolution and pixel aspect ratio;
- original scene-linear or log values, with no display LUT or viewing transform baked in;
- constant dimensions and channel layout within a sequence; and
- preferably 9-17 consecutive frames with a reference frame near the middle, so offsets through
  at least +/-8 can measure chain behaviour in both directions.

PFM remains the dependency-free format for repository-generated tests. Movie containers are not
accepted for the controlled bake-off because codec, seek and colour-conversion behaviour would
become untracked inputs.

A production corpus should contain roughly 12-20 short excerpts. One excerpt may cover several
stress categories: motion blur, defocus, low contrast, grain, occlusion/reveal, rolling shutter,
fine hair, reflections or screens, and anamorphic imagery.

### Selection policy

The default is quality-first, subject to passing reproducibility, licence, provider, target-memory
and reliability gates. Latency and lower memory break close quality results; they do not override a
material quality loss.

The fast alternative must demonstrate a material latency improvement at the selected 1080p and
4K analysis caps while remaining above a pre-registered quality floor and avoiding a material
regression in any primary production category. Exact numeric thresholds and tie-break rules are
frozen in the protocol before candidate measurements begin, then applied without post-hoc tuning.

The final `inputCurve` list must be a stable intersection supported by both selected models. If an
unsupported model/curve combination cannot be avoided, its deterministic fallback must be visible
in the contract; model-specific hidden choices are not available because Flame's panel is flat and
`setEnabled()` is forbidden. The same final review must explicitly freeze `analysisScale` or record
why it remains deferred.

Cache identity is based on the exact model artifact SHA256, conditioning formula and version, and
model parameters. It never uses a UI choice index.

## Candidate boundary

Start with:

- SEA-RAFT M, requalified from its Phase 0B diagnostic artifact under the general bake-off schema;
- WAFT with a permissively licensed backbone, beginning with Twins;
- NeuFlow v2 as the leading fast stateless candidate, if a reproducible export qualifies; and
- original RAFT as a validation baseline, not an assumed shipping candidate.

Candidate failure is a result. Unsupported operators, unavailable or unverifiable checkpoints,
wrong direction, non-reproducible export, or unacceptable resource use produce a checked-in
technical exclusion rather than a silent omission. Shipping admission is separate: unresolved or
restrictive licence terms exclude a candidate from packaging and selection but do not by
themselves erase a technically qualified evaluation result. Original RAFT is a validation
baseline, and RAFT, NeuFlow and WAFT may be compared during the bake-off even if the final plugin
ships only SEA-RAFT.

AllTracker remains a future `ReferenceFlowEstimator` investigation. MemFlow's temporal state does
not fit arbitrary OFX render order. RIFE is not admitted merely because it exports. Non-commercial
models or backbones remain excluded from commercial production use even when the plugin is used
only inside the facility. They may be used in a clearly separated non-commercial research test
when their terms permit it, but never become shipping artifacts by inference from a bake-off row.

## Work packages and Luna-max delegation

Each implementation package uses its own `codex/` topic branch and pull request. Nothing merges
without human review.

### P25-0 - Protocol and report schema - Luna A

**Status: merged through PR #6 at `035bd99`.**

Freeze before measurement:

- candidate IDs and eligibility rules;
- tensor and provider matrix;
- conditioning formulas and stable tokens;
- square-pixel megapixel cap grid;
- evaluation cases, metrics, repetitions and aggregation;
- hard rejection thresholds, quality scoring, fast-model floor and tie-breaks; and
- versioned corpus and report schemas.

Exit: a one-page decision record plus machine-validated schema fixtures. It assigns no persistent
OFX option indices.

### P25-1 - Candidate-neutral artifact framework - Luna B

**Status: merged through PR #8 at `f73b2e5`.**

Generalize the SEA-RAFT-specific workflow while preserving the Phase 0B manifest's meaning.
Every candidate manifest records:

- source repository and exact commit;
- checkpoint URL/revision, SHA256 and size;
- backbone identity and provenance where applicable;
- code, checkpoint and backbone licence audit fields;
- inputs, outputs, layout, dtype, channel order and numerical range;
- normalization inside the graph versus caller-side conditioning;
- displacement direction and units;
- padding mode/multiple, crop behaviour and dynamic/fixed shapes;
- iteration handling, confidence output and graph domains;
- export environment, opset, artifact SHA256, size and mode; and
- numerical validation results.

Validation rejects symlinks, non-regular files, mode other than `0644`, mismatched hashes/sizes,
unknown required fields and incompatible tensor contracts. Platform-specific exports receive their
own exact hashes; ONNX protobuf bytes are not assumed identical across exporters.

Exit: dependency-light schema validators, positive and negative fixtures, and a migration path for
`models/sea-raft-m.json`.

### P25-2 - Evaluation corpus and conditioning - Luna C

**Status: merged through PR #7 at `2208165`.**

Build three partitions:

1. Deterministic synthetic identity, signed translations, affine/spatial motion, borders,
   occlusion, blur, noise, HDR/log, odd sizes, asymmetric padding, PAR 0.5/2, and 1/2/4/8-link
   chain cases with analytic truth.
2. Optional public subsets whose exact frames, terms and candidate training overlap are recorded.
3. An external production manifest pointing to footage that stays on the airgapped box.

Define exact, numerically tested versions of the four approved conditioning families:

- model-native packing after a hard 0..1 clamp;
- fixed signed/log compression for scene-linear values;
- one shared pairwise percentile transform computed from both frames; and
- unmodified log input followed only by the model's declared packing.

Resolve the current contract mismatch between SEA-RAFT's caller-side replication padding and
Phase 2 preprocessing's reflect padding before any cross-candidate quality claim.

Exit: generated fixtures, conditioning tests, a production `corpus.json` template and optional
annotation schemas.

### P25-3 - Candidate export spikes - Lunas D, E and F

**Status: merged through PRs #11 (SEA-RAFT/original RAFT), #10 (WAFT) and #12 (NeuFlow) at
`5b7e059`, `2ab6121` and `366a9fc`.**

Run three independent workstreams after P25-1:

- D: SEA-RAFT requalification and original RAFT baseline;
- E: WAFT export with a permissive checkpoint/backbone; and
- F: NeuFlow v2 export and fast-candidate qualification.

Each workstream pins its own dependencies and exporter, proves PyTorch/ONNX parity where an
upstream reference exists, checks identity and both signed translation directions, validates all
advertised shapes/providers, and audits code/checkpoint/backbone terms independently.

Exit: an exact artifact and manifest or an explicit exclusion report. Candidate PRs do not touch
OFX choice parameters.

### P25-4 - Offline bake-off runner and metrics - Luna G

**Status: merged through PR #13 at `efc350f`.**

Add a self-contained offline runner in the `tools/ww-flow` family. It executes the Cartesian
product selected by the protocol:

```
candidate x case x conditioning x analysis cap x provider
```

It records preprocessing, session creation, first inference, steady inference, postprocessing and
total pair latency separately. It also records source/PAR/canonical dimensions, unpadded and padded
tensor dimensions, effective padded megapixels, spacing back to real source pixels, runtime and
provider versions, hardware/driver identity, and model/manifest/runtime hashes.

Quality metrics include:

- endpoint error and <=1 px / <=3 px fractions where dense truth exists;
- annotated landmark median/p95 error;
- visible-region warp residual;
- forward/backward residual;
- composed-chain drift at lengths 1, 2, 4 and 8;
- nonfinite or missing output fraction; and
- repeated-run stability.

Metrics aggregate by shot/category first, then macro-average categories so large plates do not
dominate. Every skip or failure has a typed reason; required cases cannot silently disappear from
an aggregate.

Exit: validated JSON and CSV reports, deterministic metric fixtures, no implicit downloads, and
resume support for interrupted airgapped runs.

#### Admission amendment (2026-08-23)

The pre-target-measurement contract is `whitewater-p25-v2`; v1 remains available for backward
compatibility. Report candidates carry shipping `status` and technical `measurement_status`
separately. Matrix planning admits only `measurement_status=measurable`, so a shipping-excluded
but technically qualified RAFT/NeuFlow/WAFT artifact can be evaluated without weakening the
fail-closed license/redistribution gate. Shipping `exclusion_reason` and technical
`measurement_exclusion_reason` are independent; excluded-but-measurable entries retain all legal
verdict and redistribution-review surfaces, including unknown/not-permitted values. Validation
baselines remain evaluation-only and cannot be shipping or P25-7 ranking winners. See
`docs/phase2.5-protocol-v2.md` for the version boundary and the fixed-shape NeuFlow follow-up.

#### Fixed-shape comparison lattice (2026-08-23)

Protocol v2 appends the shared `mp0_331776` evaluation point (0.331776 MP) to every provider's
capability list. Its frozen lattice is exactly 768x432 analysis pixels at canonical 16:9. The
`candidate_constraints` table restricts NeuFlow v2 to that cap, CPU/CUDA providers, and shots
whose computed geometry and PAR actually satisfy the lattice; matrix planning rejects unsupported
NeuFlow cells before row generation. SEA-RAFT, RAFT, and a qualified WAFT may use the same point.
The existing `mp0_5`–`mp8` grid and final CUDA `mp2` FHD/UHD gates are unchanged.

Provider capability is not provider qualification. Every measurable v2 report candidate declares
`measurement_providers`, and the planner only schedules providers listed there. The checked-in
NeuFlow evidence lists CPU only, so an operator must return a technically measurable report
candidate with CUDA explicitly listed before a NeuFlow CUDA lattice run is admitted. No CUDA pass
is implied by the protocol or by this lattice.

Corpus selection for a NeuFlow comparison should use the existing FHD/UHD PAR1 synthetic targets
or production shots that compute to 768x432 at the new cap. Non-16:9, anamorphic, undersized, and
other rounded geometries remain valid for unconstrained candidates but must be separate from a
NeuFlow matrix. The operator command must select `profile=screen`, `cap_tokens=["mp0_331776"]`,
and only those shots/providers; final shipping commands continue to select the unchanged `mp2`
target cells.

### P25-5 - Local and CI qualification - Luna H

**Status: merged through PR #15 at `946c042`; the exact artifact was qualified at `adfd4fb` by
successful workflow run `32780658875`. The evaluation tarball SHA256 is
`53b4e7192496a5ee8be1e0af6085980b59e287afc03599f953e2bb9a65eb8850`.**

Run export, schema, hash, permission, tensor-contract, direction, operator and CPU-correctness
gates locally and in CI. Build the EL8 airgap tarball containing only explicitly admitted
measurement candidates plus their manifests, licences/notices, runtime and evaluator. Evaluation
admission does not authorize any candidate for the shipping bundle. Verify the source, staged,
archived and extracted copies by hash, size, regular-file status and mode `0644`.

CPU is a correctness path at manageable sizes, not a full production-resolution performance
sweep. Phase 0B measured SEA-RAFT M at roughly 56 seconds per 1080p CPU inference; repeating that
across the whole matrix would consume time without answering the GPU shipping question.

Exit: the ordinary repository suite remains green and the exact carried tarball has a SHA256 and
human-readable run instructions.

The carried `RUN-P25-5.txt` is deliberately limited to archive verification/extraction, the
required CPU provider/session check, and one optional PFM-pair smoke command. It is not the
operator procedure for the resumable P25-6 `smoke`, `screen`, and `final` profiles. P25-6 must
publish and verify that complete operator procedure before production target measurement begins;
the inner `runtime/whitewater-p25-5-runtime.tar.gz` is only the relocatable conda environment.

### P25-6 - Airgapped target measurement - Luna I plus human operator

Before the human run, package or otherwise provide the exact resumable profile entrypoints,
production-corpus inputs, output paths, and recovery commands needed to drive the P25-4
matrix/session/reporting machinery with this qualified evaluator. The instructions must identify
the qualified outer archive by SHA256 and must not require edits to any carried file.

The human runs three resumable profiles on the production machine:

- `smoke`: verify archive/artifact hashes, runtime/provider selection, directions and a small
  production sample;
- `screen`: compare all surviving candidates, curves and practical caps; and
- `final`: apply the expensive repeated timing/VRAM protocol only to finalists.

Measure CUDA both idle and beside a live Flame Batch workload. Record baseline, session-create
peak, warm/steady peak, cleanup and process-exit memory with device-wide NVML data and per-process
data where available. Record first, warm and steady latency rather than treating time-to-failure
as performance.

Phase 0B's approximately 278 ms warmed 1080p SEA-RAFT CUDA result and the UHD/DCI/Alexa bounded
allocation stops under a 16 GiB arena are baseline evidence only. They do not set a product source
limit. Phase 2.5 selects practical square-pixel megapixel caps for each exact model/runtime pair on
the target hardware.

The operator returns only:

```
report.json
summary.txt
runner.log
nvml.csv
review.csv       # when local visual review was performed
```

No production images leave the machine.

Exit: all required target cases have a result or typed failure, and the report validator accepts
the returned files.

### P25-7 - Ranking and human selection - Luna J plus human reviewer

Apply the pre-registered gates and ranking without changing weights after results are visible.
Review anonymous local production comparisons where automated metrics are not trustworthy around
occlusions, lighting changes or reflections.

Exit: the report names the exact default artifact hash and, only if it passes every gate, the fast
artifact hash; it also records selected input-conditioning options/formulas, analysis caps, known
weak categories, licence verdicts and the rationale for the decision. Human approval is required
before the next package.

### P25-8 - API and combined Phase 2/2.5 closeout - Luna K

Only after P25-7:

- publish `inputCurve` order/defaults and, only when multiple shipping models qualified, a
  permanent `model` option order/default;
- publish or explicitly defer the measured `analysisScale` option order;
- update raw host-harness assertions for exact labels, order and defaults;
- verify all options are common to both selected models or have documented deterministic fallback;
- retain artifact/conditioning/model-parameter cache fingerprints rather than UI indices;
- record selected manifests and measurement reports;
- update `AGENTS.md`, `README.md`, `docs/plan.md`, `docs/context.md` and `models/MODELS.md` to close
  both Phase 2 and Phase 2.5; and
- run a clean Release build, full CTest suite, both dependency-boundary negative fixtures and the
  exact OFX export gate.

Exit: a reviewable closeout PR with no production runtime or render integration.

## Production corpus manifest

The external corpus manifest contains anonymous metadata and paths, not frames:

```json
{
  "schema_version": 1,
  "shots": [
    {
      "id": "blur-pan-01",
      "path_pattern": "shots/blur-pan-01/plate.%04d.exr",
      "first_frame": 1001,
      "last_frame": 1017,
      "reference_frame": 1009,
      "pixel_aspect_ratio": 1.0,
      "encoding": "scene-linear",
      "categories": ["motion-blur", "low-contrast"],
      "annotations": "annotations/blur-pan-01.csv"
    }
  ]
}
```

The runner hashes the frames it reads and writes those hashes to the local report, so a re-run can
prove it used the same bytes without exposing them.

Point annotations are optional during screening and required only where the final protocol needs
landmark evidence. They use an explicit pixel-centre/top-left coordinate convention and carry
source frame, target frame, visibility and optional weight. The runner converts them to its
canonical full-resolution real-pixel convention before scoring.

For cases without trustworthy ground truth, the runner produces anonymous candidate labels and
local comparison outputs. The human records edge adherence, occlusion/reveal behaviour, blur,
jitter and drift in `review.csv`; previews remain on the box.

## Phase boundary

Phase 2.5 owns exporters, artifact manifests and validators, the evaluation corpus contract,
offline bake-off tooling, exact reports, licence findings, and the selection/API freeze.

Phase 3 owns production `ModelSpec`, `ModelRegistry`, `RuntimeLoader`, `OrtEnvironment`, session
caching/provider fallback, selected `OnnxPairwiseFlowEstimator` integration and production runtime
packaging. Phase 4 owns OFX frame pulls, `FlowPreparation`, render wiring, abort/progress,
Precache/Clear and artist-visible failure handling. No disk cache, reference/window estimator,
model-named production estimator class, or direct/anchor drift mitigation enters Phase 2.5.
