# P25-7 prestage: values BLOCKED on on-box validation

This directory is a DRAFT prestaged for human review. The items below are the only values that
are deliberately not final. They cannot be filled in the repo because the data does not exist yet;
they are produced by the on-box airgapped runs. Do not invent them.

## 1. Production frame paths (all 9 shots)

`inputs/corpus.template.json` production_external partition: every shot's `path_pattern` is a
`REPLACE_WITH_ON_BOX_ABSOLUTE_PATH/.../plate.%04d.exr` placeholder. The operator replaces each with
the truthful on-box EXR sequence path. Geometry/PAR/encoding/channels/bit_depth/reference_frame are
the anonymized real-shot metadata supplied by Andrew (2026-09-02) and must not be changed. The
driver refuses to start (`corpus_invalid`) while any placeholder remains.

## 2. waft-twins and neuflow-v2 Linux ONNX identity hashes

These candidates' `checkpoint_sha256` and `source_commit` are known (from their model manifests and
carried below for reference), but their **Linux x86_64 ONNX artifact identity** only exists after
the on-box WAFT / NeuFlow validation-export runs. Until then, the following fields carry
`PENDING_ONBOX_VALIDATION_<candidate>` placeholders in `inputs/candidate-entries.json` and
`inputs/artifact-map.json`:

- `artifact_sha256`
- `export_environment_sha256`
- `manifest_sha256`
- `artifact_size_bytes`
- artifact-map `platform`

Matrix planning of these candidates already succeeds offline (the planner does not consume the
hashes), so the intended screen matrix is fully expressed and validated. But any selection naming
waft-twins or neuflow-v2 will fail on-box artifact materialization until the operator replaces the
placeholders with the exact values from the validated linux-x86_64 manifest.

Known (real) identity carried in the entries:

| candidate  | source_commit                              | checkpoint_sha256 |
|------------|--------------------------------------------|-------------------|
| waft-twins | b152ff1cad1af8c185ee7b141997c48ff3334c87   | f750cd15281fc30de477723438ff4a67fe1591deac4ab0eb9b366e27c827e070 |
| neuflow-v2 | 204b5e3744461d90303b9ff82caa7a1bb56a2ca2   | 76152c8068f247a7d073aa13e61da8cb4c3c6a798076d4dc8e20f7995fcc019f |

## 3. Structural blockers to flag (NOT operator-fillable)

- **neuflow-v2 has no `final` profile.** The frozen `_validate_final_coverage` gate requires the
  final matrix to select `mp2`, but the frozen `candidate_constraints` restrict neuflow-v2 to
  `mp0_331776`. A neuflow-only `final` selection is therefore structurally impossible and is
  rejected (`final_coverage`). NeuFlow's CUDA timing/VRAM cannot ride the `final` profile at all.
  DECISION NEEDED (Andrew): how to capture neuflow-v2 GPU latency/VRAM for ranking -- e.g. a
  `screen`-profile CUDA block at `mp0_331776` (lighter sampling than final), or accept neuflow GPU
  ranking on the constrained-cap numbers only. Not resolved in this prestage.

- **waft-twins has no CUDA measurability yet.** Its manifest exposes only CPUExecutionProvider, so
  `measurement_providers` is `[cpu]` and a CUDA (final) waft run is rejected (`provider_unavailable`).
  If waft-twins is to be a CUDA finalist, the on-box validation must establish and list cuda before
  it can join `selection-final.json` at mp2.

- **The final mandatory-cell gate is covered by sea-raft-m only right now.** `selection-final.json`
  contains the two mandatory cells (fhd/uhd @ mp2 @ cuda, idle+live_flame). sea-raft-m is the only
  candidate currently able to run them. waft-twins joins that same file at mp2 once items 2 and its
  CUDA measurability are resolved.
