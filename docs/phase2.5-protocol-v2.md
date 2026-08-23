# Phase 2.5 protocol v2 — evaluation admission amendment

This is the pre-target-measurement amendment to `docs/phase2.5-protocol-v1.md`. The executable
documents are `bakeoff/protocol-v2.json`, `bakeoff/protocol-v2.schema.json`, and
`bakeoff/report-v2.schema.json`. The evaluation corpus is unchanged and remains
`bakeoff/corpus-v1.schema.json`; v2 reports bind the same corpus bytes by SHA256.

## Why a new version

The original report candidate `status` combined two decisions: whether an artifact could ship and
whether it was technically available for an offline measurement. That made a numerically
qualified RAFT, NeuFlow, or WAFT artifact disappear when its checkpoint or backbone terms were
unknown. The correction changes report admission semantics, so it gets a new protocol/report ID
(`whitewater-p25-v2`) rather than silently changing the meaning of v1. No target measurement
reports exist yet, but retaining v1 files and adding v2 is the safest compatibility boundary for
tools that already consume the frozen v1 schema.

The public validator defaults to this active v2 protocol, for example:

```bash
python3 tools/bakeoff/validate.py report.json --kind report --corpus corpus-v1.json
```

Historical v1 reports remain supported by selecting the frozen v1 bundle explicitly with
`--protocol bakeoff/protocol-v1.json`.

## Two candidate decisions

Every v2 report candidate has both fields:

- `status` is the shipping/license gate. `eligible` is fail-closed and still requires the complete
  artifact identity, all three commercial-use verdicts, all three redistribution verdicts, and
  reviewed redistribution terms. It also requires the protocol role `shipping-candidate`.
  `excluded` is never shippable, even if its artifact is measurable.
- `measurement_status` is technical measurement admission. `measurable` requires the exact source,
  checkpoint, artifact, export-environment and manifest hashes plus artifact size; `unavailable`
  means that the technical artifact is missing, unverifiable, or failed qualification and requires
  its own typed `measurement_exclusion_reason`. `measurement_exclusion_reason` is forbidden for a
  measurable candidate; it is independent of the shipping `exclusion_reason`.

The implication is one-way: shipping `eligible` implies `measurement_status=measurable`, but a
measurable candidate may remain shipping `excluded` and be evaluated. Matrix planning uses only
`measurement_status`; an `unavailable` candidate is rejected even when it is listed in the report.
This keeps the license gate exact while allowing evaluation evidence from excluded candidates.

An excluded-but-measurable candidate carries the complete `license_verdicts`,
`redistribution_permitted`, and `redistribution_terms_reviewed` surfaces even when values are
unknown or not permitted. The report therefore preserves the legal evidence that explains its
shipping exclusion. The shipping `exclusion_reason` remains required for every excluded candidate;
its type does not imply technical availability or unavailability.

`raft-original` remains a `validation-baseline`. Its measurable results are useful comparisons,
but its role prevents `status=eligible`; v2 reports contain no selection record, and the later P25-7
selection/ranking record may name only shipping-eligible candidates. Evaluation rows and scores
therefore cannot become an OFX choice or a shipping winner by accident.

NeuFlow v2 currently has a fixed `NCHW [1,3,432,768]` export. v2 therefore freezes an additional
`mp0_331776` evaluation cap/lattice: decimal area `0.331776` MP, computed analysis geometry exactly
`768x432`, and canonical square-pixel aspect ratio `16:9`. The cap is appended to the provider
lists so every measurable candidate can run the same comparison point; it does not alter the
shipping `mp0_5`–`mp8` grid or the final `mp2` gates.

The executable `candidate_constraints` entry admits `neuflow-v2` only on `cpu` or `cuda`, only at
`mp0_331776`, and only when the source/PAR metadata computes to that exact geometry and canonical
16:9. A NeuFlow matrix that requests `mp0_5`, `mp2`, CoreML, a non-16:9 source, or any other
computed dimensions is rejected before Cartesian rows are generated. SEA-RAFT, RAFT, and a
qualified WAFT may use the shared lattice under the ordinary provider/cap rules.

Provider support is potential capability, not qualification evidence. A measurable v2 report
candidate must list its `measurement_providers`; the planner requires every selected provider to
appear there.
The checked-in NeuFlow evidence remains CPU-only, so no NeuFlow CUDA row is schedulable until a
returned report candidate explicitly carries `measurement_providers: ["cuda"]` (or includes it
alongside CPU). This field must never be filled merely because the protocol allows CUDA.

Corpus/operator implication: a NeuFlow comparison run must select synthetic or air-gapped shots
whose computed analysis geometry is exactly `768x432` and whose canonical geometry is 16:9. The
existing FHD `1920x1080 PAR1` and UHD `3840x2160 PAR1` synthetic targets both satisfy this lattice;
small, non-16:9, anamorphic, or otherwise rounded shots remain useful for the other candidates but
must be omitted from a NeuFlow matrix. Use a separate `screen` invocation for the shared lattice.
The runner command should carry a selection equivalent to:

```json
{
  "profile": "screen",
  "environment": "el8-x86_64",
  "candidate_ids": ["neuflow-v2", "sea-raft-m"],
  "conditioning_tokens": ["native-clamp01-v1"],
  "cap_tokens": ["mp0_331776"],
  "providers": [{"token": "cpu", "host_loads": ["not_applicable"]}],
  "shot_ids": ["syn-fhd-1920x1080-par1", "syn-uhd-3840x2160-par1"]
}
```

The exact CLI wrapper may choose a different option spelling, but the persisted selections must
carry the same candidate/cap/provider/shot axes. Do not add `mp2` to that NeuFlow selection to
force final shipping coverage; final reports still require the unchanged CUDA `mp2` FHD/UHD cells
for candidates whose artifacts support those gates.
