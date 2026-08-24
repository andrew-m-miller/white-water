# Phase 2.5 corpus and conditioning (P25-2)

P25-2 is the dependency-free corpus/conditioning package.  Its executable source is
`tools/bakeoff/synthetic.py`, `tools/bakeoff/conditioning.py` and
`tools/bakeoff/padding.py`; `tools/bakeoff/corpus_conditioning_tests.py` is the numerical
gate.  The complete metadata document is produced with:

```bash
python3 -m tools.bakeoff.generate_corpus --output /tmp/corpus-v1.json
python3 tools/bakeoff/validate.py /tmp/corpus-v1.json --kind corpus
```

The validator defaults to the active `whitewater-p25-v2` protocol; the unchanged corpus remains
the v1 corpus contract. To reproduce the historical v1 gate explicitly, pass
`--protocol bakeoff/protocol-v1.json`.

The default command writes metadata only.  Add `--frames-dir DIR` to emit deterministic
RGB PFM sequences and one JSON analytic-truth sidecar per small synthetic case.  FHD and
UHD remain lazy metadata cases unless `--include-large` is explicitly requested; this keeps
the repository and ordinary test run small without weakening their required corpus coverage.
PFM row 0 is the bottom row, and the truth convention is full-resolution real-pixel centres,
x right/y up, with a pair displacement from image1 to image2.  The same y-up convention is
used by row-major buffers, visibility rectangles and later OFX/core coordinate adapters.
The generator is not an
inference oracle: model results are measured against the analytic field sidecars by the later
P25-4 runner.

The affine and spatial cases generate each frame by inverse-sampling one common plate.  Their
truth is the corresponding forward map composed as
`forward(to, inverse(from, (x,y))) - (x,y)`, so a field is evaluated in the same output/source
coordinate system as the emitted pixels rather than by subtracting two same-coordinate
displacements.  The spatial field is an exactly invertible y-dependent shear.  The
occlusion/reveal case records an analytic rectangle for every frame: the foreground moves
`(3,-1)` px per frame while its background moves `(1,.5)`, making both visibility transitions
real and measurable.

For occlusion/reveal, consumers use `analytic_pair_truth(case, from_frame, to_frame, x, y)`.
Its typed result reports the visible source/target layers and returns dense displacement only
for `status="foreground"` or `status="background"`; `status="occluded"` and
`status="revealed"` have `displacement=None` and `no_dense_truth=True`.  The result also records
fixed-coordinate `same_coordinate_transition` (`stable`, `occluded`, or `revealed`) so mask
changes remain measurable even when a moving layer has a valid correspondence elsewhere.
For a source-domain correspondence, background mapped into target foreground is `occluded`,
while foreground mapped into target background is `revealed`.  At one fixed coordinate the
labels describe the visible plate instead: foreground leaving to background is `revealed`, and
background becoming foreground is `occluded`.  This translating-rectangle construction naturally
produces source-domain occlusion; reveal is primarily fixed-coordinate/target-domain evidence
unless a later target-domain query API is added.
The compatibility `analytic_displacement` helper follows the selected layer and raises a
typed `TruthUnavailable` for the no-dense cases rather than silently returning background flow.
The occlusion truth sidecar names this API and its status contract.

The noise case records one reproducible seed (`4701`) and uses that exact seed as the generator
input; changing it changes the emitted pixels while repeated generation is identical.

## Frozen conditioning

`condition_pair` implements the four protocol tokens before model-declared tensor packing:

* `native-clamp01-v1`: `min(1,max(0,x))`;
* `signed-log-v1`: `clamp(0.5 + sign(x)*log1p(abs(x))/(2*log(17)), 0, 1)`, with `sign(0)=0`;
* `pair-percentile-v1`: one shared finite RGB sample stream from both frames, linear
  quantiles at 1% and 99% using `h=(n-1)*p`, then `clamp((x-lo)/max(hi-lo,1e-6),0,1)`;
* `native-log-v1`: the finite input values unchanged.

The pair-percentile bounds are computed once and returned in `PairConditioning.parameters`;
they are never computed independently per frame.  Any NaN or infinity is a typed
`ConditioningFailure(kind="nonfinite_input")`; an empty pair has
`ConditioningFailure(kind="empty_pair")`.  This prevents a nonfinite value from reaching a
model tensor while retaining the protocol's finite-sample quantile rule.  The numerical tests
also pin signed-log full scale (`-16 -> 0`, `0 -> .5`, `16 -> 1`), interpolation, shared bounds,
clamping, no-op log behavior and typed failures.

A constant pair has `lo == hi`, so it cannot satisfy the frozen report schema's strict
`low < high` parameter rule.  Both pair-conditioning entry points therefore return the typed
`conditioning_failure` outcome `reason="constant_pair"`; they never emit an invalid
`low == high` pass and no protocol ID is changed.

## Caller padding seam

There are two valid policies, and they are intentionally not hidden behind a candidate name:
`tools/bakeoff/padding.py` implements the canonical manifest tokens
`caller-replication-crop` (clamp to the edge) and `caller-reflection-crop` (mirror without
repeating the edge pixel, matching `src/core/flow/Preprocess.cpp`).  The short
`replication`/`reflect` spellings are compatibility aliases only.  The migrated SEA-RAFT
manifest's `caller-replication-crop` token is accepted directly.  P25-2 therefore makes the
policy an explicit required argument, tests the differing halo (`[1,2,3]` with one side on
each end becomes `[1,1,2,3,3]` versus `[2,1,2,3,2]`), and extends right/top padding so
requested dimensions are exact multiples of the declared tensor multiple.

P25-1 owns the candidate manifest and its exact padding field.  P25-2 does not modify that
schema: its adapter passes the declared string through this narrow seam.  A cross-candidate
comparison is valid only when the caller-side policy is identical in the comparison cell, exactly as frozen by
`padding_comparison_policy` in `bakeoff/protocol-v1.json`; the asymmetric synthetic case records
`caller-replication-crop` as its deterministic fixture policy for the current SEA-RAFT contract.

## Corpus partitions

The generated document always contains the exact 23 required synthetic cases (identity, signed
X/Y translations, affine/spatial fields, border, occlusion/reveal, blur, noise, HDR scene-linear,
log input, odd/asymmetric padding, PAR 0.5/PAR 2, chain lengths 1/2/4/8, and FHD/UHD PAR 1) and
the nine primary production categories via the external metadata fragment
`bakeoff/production-corpus-v1.template.json`.  That fragment contains paths and format metadata
only; operators replace the `/AIRGAP/replace/...` paths and shot metadata on the air-gapped box.
No public partition is emitted because the plan marks public subsets optional and no terms or
training-overlap record is available in this repository.
