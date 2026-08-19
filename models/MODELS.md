# Models

Weights are **not committed**. They are exported from a pinned upstream checkpoint into
ONNX by the scripts here, verified against a recorded SHA256, and staged into
`Contents/Resources/models/` in the bundle at build time. `.gitignore` excludes them.

A checked-in blob is worse than a script for the one thing that actually matters here: a
licence audit has to be able to trace a shipped file back to a specific upstream release,
and a binary in git history proves nothing about where it came from.

## What ships

| Model | Role | Upstream | Licence |
|---|---|---|---|
| RAFT | Default. Pairwise optical flow, accurate, slow. | princeton-vl/RAFT | BSD-3-Clause |
| RIFE | Fast alternative. IFNet gives bidirectional intermediate flow in one pass. | hzwer/Practical-RIFE | MIT, weights stated to be under the same licence as the code |

**Both licences permit commercial use. Neither has been audited.** They were read from the
upstream repositories, not checked by anyone qualified to sign off on it. Before anything
goes to a client, re-verify against the *exact* checkpoint files shipped — some RIFE
forks and some "practical" checkpoint drops carry non-commercial terms that the parent
repository does not. Record the verdict here with a date and the checkpoint SHA256.

## Tensor contract

Each model's contract is data, not code — see `src/infer/ModelSpec` when it exists. What a
spec has to pin down:

- input and output tensor names, and their layout (NCHW, channel order, batch)
- normalization: range, mean/scale, and whether the model wants 0-1 or -1-1
- the **padding multiple**: 8 for RAFT (a network-structure constraint, not a convention),
  32 for RIFE. Inputs are reflect-padded up and the flow cropped back.
- iteration count, which for RAFT is **baked in at export time** — a traced ONNX graph has
  the GRU loop unrolled, so an `Iters` parameter means either several exported variants or
  an export that keeps the loop dynamic. Decide this at export, not at runtime.

## Export

`export_raft.py` and `export_rife.py` do not exist yet. They arrive with Phase 3. Each must:

1. Pin the upstream commit and the checkpoint URL, and assert the checkpoint's SHA256.
2. Export at the resolutions the plugin offers, or with dynamic spatial dims where the
   graph allows it.
3. Record the resulting ONNX file's SHA256 in this document.
4. Round-trip a synthetic translation and assert the recovered flow, so a bad export fails
   at export time rather than in a grading suite.

Pre-converted exports exist publicly (PINTO0309's model zoo, ibaiGorordo's ONNX-RAFT,
FuryTMP/RIFE_fp32) and are worth reading for their tensor contracts. Do not ship them: an
export whose provenance and preprocessing we did not pin is an export whose failure mode we
cannot reason about.

## Resolution at runtime

The plugin looks for weights in this order, and prints which one won to stderr — which
Flame captures in `/opt/Autodesk/log/`, the only diagnostic channel on the box:

1. the `Model Dir` parameter
2. `$WHITEWATER_MODEL_DIR`
3. `Contents/Resources/models/` inside the bundle
