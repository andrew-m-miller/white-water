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

NeuFlow v2 currently has a fixed 432x768 export. Candidate-specific shape/cap constraints must be
represented by its artifact manifest and selected as a constrained evaluation lattice (an exact
roughly-0.33 MP 16:9 point is a likely follow-up); the generic shipping `mp0_5`–`mp8` grid must not
be read as proof that this fixed-shape artifact supports those caps. This amendment admits the
artifact without pretending that its shape support is broader than measured.
