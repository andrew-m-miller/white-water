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

## 2. waft-twins and neuflow-v2 Linux ONNX identity hashes — RESOLVED

**Resolved.** The on-box WAFT and NeuFlow validation-export runs have produced their validated
linux-x86_64 manifests, recorded on `main` (`models/waft-twins-artifact.json`,
`models/neuflow-v2.json`, merged in PR #36). The `artifact_sha256`, `export_environment_sha256`,
`manifest_sha256`, `artifact_size_bytes` and artifact-map `platform` fields in
`inputs/candidate-entries.json` and `inputs/artifact-map.json` are now bound to the linux-x86_64
`platform_artifacts` row of those manifests (identity fields) and to each manifest file's own
SHA256 (`manifest_sha256`), matching the `tools/p25_5/p25_6_materialize_inputs.py` linux-identity
convention. No values were invented; each traces to the validated manifest.

Bound identity:

| candidate  | source_commit                              | checkpoint_sha256                                                | artifact_sha256                                                  | artifact_size_bytes |
|------------|--------------------------------------------|------------------------------------------------------------------|------------------------------------------------------------------|---------------------|
| waft-twins | b152ff1cad1af8c185ee7b141997c48ff3334c87   | f750cd15281fc30de477723438ff4a67fe1591deac4ab0eb9b366e27c827e070 | de5206543d409f8759fcf4932543600065e66777a74ff63142c3c2426334a9e7 | 545359430           |
| neuflow-v2 | 204b5e3744461d90303b9ff82caa7a1bb56a2ca2   | 76152c8068f247a7d073aa13e61da8cb4c3c6a798076d4dc8e20f7995fcc019f | 07aa20529a93eb970bd3160ce8a2cee10b65ba410aa32f47e63676c039661548 | 66177652            |

Both candidates remain `status=excluded` (checkpoint licence terms unknown); this fill binds their
measurement identity only, not any shipping authorization. The structural blockers in section 3
(neuflow-v2 has no `final` profile; waft-twins exposes only CPUExecutionProvider) are unchanged and
still need Andrew's decisions before those candidates can join the CUDA `final` cells.

## 3. Structural blockers to flag (NOT operator-fillable)

- **neuflow-v2 has no `final` profile.** The frozen `_validate_final_coverage` gate requires the
  final matrix to select `mp2`, but the frozen `candidate_constraints` restrict neuflow-v2 to
  `mp0_331776`. A neuflow-only `final` selection is therefore structurally impossible and is
  rejected (`final_coverage`). NeuFlow's CUDA timing/VRAM cannot ride the `final` profile at all.
  DECISION NEEDED (Andrew): how to capture neuflow-v2 GPU latency/VRAM for ranking -- e.g. a
  `screen`-profile CUDA block at `mp0_331776` (lighter sampling than final), or accept neuflow GPU
  ranking on the constrained-cap numbers only. Not resolved in this prestage.

- **waft-twins is carried as CPU-only, validated-but-unmeasured evidence — RESOLVED (decision 2C).**
  Its ONNX export is a fixed 128x192 spatial shape: WAFT specializes resolution and cannot export a
  spatially-dynamic ONNX (no dynamic-shape support), and 128x192 matches no protocol cap token. A
  128x192 graph cannot accept the shared `mp0_331776` (768x432) lattice input, and its manifest
  exposes only CPUExecutionProvider (`measurement_providers` is `[cpu]`). waft-twins is therefore NOT
  scheduled in the shared `mp0_331776` lattice and has been dropped from those screen selections; it
  is retained in `candidate-entries.json` as `status=excluded` evidence. A full arena measurement
  would require a re-export at the lattice resolution plus a CUDA validation pack, which is deferred.

- **The final mandatory-cell gate is covered by sea-raft-m only right now.** `selection-final.json`
  contains the two mandatory cells (fhd/uhd @ mp2 @ cuda, idle+live_flame). sea-raft-m is the only
  candidate currently able to run them. waft-twins does not join that file: per decision 2C it is
  carried as CPU-only validated-but-unmeasured evidence and is not scheduled in the shared lattice
  (its export is fixed 128x192, resolution-specialized with no dynamic-shape support, and CPU-only).
