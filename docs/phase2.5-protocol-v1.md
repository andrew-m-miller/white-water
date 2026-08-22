# Phase 2.5 protocol v1

Decision record, frozen 2026-08-22 before candidate measurements. The executable record is
`bakeoff/protocol-v1.json`; corpus and result files must validate against the version-1 schemas.
Changing a threshold, weight, token, case requirement or aggregation rule creates a new protocol
ID. It never edits v1 after measurements exist.

The machine contracts are `bakeoff/corpus-v1.schema.json` and
`bakeoff/report-v1.schema.json`, both checked with the standard-library-only validator in
`tools/bakeoff/validator.py`. A corpus has explicit `synthetic` and `production_external`
partitions; the optional `public` partition records its terms and training-set overlap. Shots
contain paths and anonymous metadata only. A public partition, when present, must record its
usage terms and training-set overlap. A report names the corpus and records the canonical SHA256
of the complete strict corpus JSON. It identifies every input frame by SHA256 and records hashes
for the exact candidate artifact, manifest, checkpoint, export environment, evaluator and runtime;
it also records one frozen environment token (`el8-x86_64` or `macos-arm64`) and the report-wide hardware identity. It never embeds source
pixels. Every `fail` or `skip` result carries a typed reason, and every `pass` records the complete
timing, geometry, metric and resource contract for its profile. Result identity includes host load;
the report's explicit, SHA256-bound `matrix` selector records the candidate, shot, conditioning,
cap and provider/host-load axes, and the validator requires the result identities to equal its
Cartesian product exactly. Paired `idle`/`live_flame` rows are required for every final CUDA cell;
complete final CUDA pass rows additionally require all five NVML stages. CPU/CoreML rows use
`not_applicable`. The summary's `required_cells` equals the result-row count, so a declared cell
cannot silently disappear.
`tools/bakeoff/schema_tests.py` runs positive and negative fixtures and checks that these schemas
remain aligned with the frozen protocol. No file in this package assigns an OFX Choice index.

## Admission and matrix

The initial IDs are `sea-raft-m`, `waft-twins`, `neuflow-v2`, and `raft-original`; the last is a
validation baseline and cannot ship from this protocol. Admission requires pinned code, checkpoint,
backbone and export identities; separate commercial-use and redistribution verdicts for all three
licence surfaces; reproducible export; PyTorch/reference parity where available; and a batch-1,
float32 RGB pair producing float32 `image1_to_image2` pixel displacement. Eligibility requires
commercial-use verdict `commercial_use_permitted`, an independent per-surface redistribution
verdict of `permitted`, and a recorded review of redistribution terms. The backbone hash is
optional when the manifest says it is not applicable. A failure produces an exclusion report, not
a missing row.

An individual report declares its measured subset in `matrix`: unique candidate and shot IDs,
conditioning and cap tokens, and a provider-to-host-load mapping. `matrix_sha256` is the lowercase
SHA256 of that selector with the hash member removed, using canonical JSON (sorted keys and compact
separators). The validator expands those axes into six-field identities
`candidate + shot + conditioning + cap + provider + host_load` and rejects both missing and extra
rows. This permits a smoke or screen report to select a small, explicit subset while preserving a
machine-checkable no-silent-disappearance gate. Excluded candidates may carry only a typed exclusion
reason (with provenance when available); eligible candidates require the complete identity,
commercial-use verdicts, permitted redistribution verdicts and reviewed redistribution terms. A final matrix must include CUDA `mp2`
cells for exact 1920x1080 PAR1 and 3840x2160 PAR1 corpus shots, and a matrix cannot select an
excluded candidate.

CUDA on EL8 is the selection provider. CPU runs correctness at 0.5 MP. CoreML on arm64 macOS is a
supporting qualification and cannot substitute for the CUDA gate. CUDA screens the square-pixel cap
tokens `mp0_5`, `mp1`, `mp2`, `mp4`, `mp6`, and `mp8` (decimal megapixels before model padding).
Every result records padded size and effective padded megapixels as well. The minimum target gate is
both 1080p and UHD source material at `mp2`; higher practical caps are the largest reliable points
that fit beside live Flame, not source-format limits.

The conditioning tokens are immutable formulas: `native-clamp01-v1`, `signed-log-v1`,
`pair-percentile-v1`, and `native-log-v1`. The signed-log transform is
`c(x)=clamp(0.5+sign(x)*log1p(abs(x))/(2*log(17)),0,1)` with `sign(0)=0`; full scale is the
stated fixed magnitude `|x|=16`. It preserves signed highlight texture through that magnitude
rather than saturating every value above 1, and nonfinite input is a typed conditioning failure.
P25-2 implements and numerically tests the
formulas in the protocol JSON. Pair percentiles are shared across both frames and RGB channels;
independent per-frame exposure is forbidden. Candidate padding remains declared per exact
artifact; a cross-candidate claim compares the same caller-side padding policy.

## Cases and measurement

The executable `required_synthetic_cases` list freezes separate signed X/Y directions, affine
and spatial fields, scene-linear HDR and log input, odd size and asymmetric padding, both PAR 0.5
and PAR 2, each 1/2/4/8-link chain, and exact FHD/UHD PAR1 performance shots. Each synthetic shot
carries one `case_id`; validation requires exact case coverage while retaining the broader category
labels for macro aggregation. Every shot records dimensions, channels, bit depth and analytic truth
for synthetic cases; production paths remain metadata-only `.exr` references.
The production corpus must cover every primary category named in the protocol; one shot may cover
several. Public data is optional and never fills a required production row.

`screen` records one first inference and five steady inferences per cell. `final` uses three fresh
sessions, each with one recorded warm-up and ten steady samples. Timing separates preprocessing,
session creation, first inference, steady inference, postprocessing, and total pair time; every
fresh session records its own creation and first-inference durations, and the report's steady
sample list is their exact concatenation. The top-level session-creation and first-inference values
are medians of fresh-session values, steady inference is the median of flattened steady samples,
and total pair time is preprocessing plus steady inference plus postprocessing. A pass binds its
source/PAR/canonical and cap-sized analysis geometry to the corpus metadata, including
`spacing_x=source_width/analysis_width` and `spacing_y=source_height/analysis_height`; it names
exactly two distinct in-range frames and agrees with any corpus frame hashes. Conditioning is checked
against shot encoding; pair-percentile passes record shared finite low/high values and epsilon
`1e-6`, while other curves carry no parameters. Final CUDA runs require paired idle and live-Flame
rows for every final CUDA cell; complete pass rows additionally record baseline/create/steady/cleanup/
process-exit NVML samples. Required rows cannot be silently skipped; every non-pass has a typed reason.

## Frozen gates and ranking

Any artifact/hash/licence/provider/direction failure, missing required result, nonfinite output,
greater than 0.05 px repeated-run p99 delta, failure at either mandatory `mp2` source target, or
more than 15.0 GiB peak incremental device memory rejects shipping eligibility. Dense synthetic
score must be at least 75 with every primary synthetic category at least 60. Final production review
must be at least 70 with every primary production category at least 60.

Dense shot score is
`100 * (0.50*fraction_le_1px + 0.30*fraction_le_3px + 0.20*max(0, 1-EPE/3))`.
Five anonymous human-review dimensions are each 0..4 and scale linearly to 0..100. Scores average
within shot, then macro-average shots within category, then macro-average categories. The final
quality score is 30% synthetic and 70% production; a final selection is invalid without both.

The default is the highest quality score. Within 3.0 points, lower geometric-mean steady latency at
the selected 1080p/UHD caps wins, then lower peak memory, artifact size, and finally candidate ID.
The fast alternative must score at least 75 and within 8.0 points of the default, lose no primary
category by more than 10.0 points, be at least 30% faster at each selected target and 35% faster by
their geometric mean, and pass every default reliability/resource gate. If none does, v1 records no
fast selection and candidate search continues. P25-7 emits a separate selection record after
consuming multiple validated report hashes and human evidence; that record is bound to those report
hashes under these frozen ranking rules. A measurement report contains no selection or
candidate-score fields, and no OFX option index is assigned here.
