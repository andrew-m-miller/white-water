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

**This table is the approved plan's choice and is now expected to change before Phase 3.**
The survey below concludes that SEA-RAFT should take the default slot and that MemFlow is a
better fast option than RIFE. Neither estimator has been written, so the change is free
today and expensive after an export and a tensor contract exist.

**Both licences permit commercial use. Neither has been audited.** They were read from the
upstream repositories, not checked by anyone qualified to sign off on it. Before anything
goes to a client, re-verify against the *exact* checkpoint files shipped — some RIFE
forks and some "practical" checkpoint drops carry non-commercial terms that the parent
repository does not. Record the verdict here with a date and the checkpoint SHA256.

## Candidates considered

Surveyed 2026-08-19. Licence column is what the upstream repository *states*; none of it
has been audited, and for a facility deliverable that audit is not optional.

| Model | Venue | Licence | Verdict |
|---|---|---|---|
| **SEA-RAFT** | ECCV 2024 (oral) | **BSD-3** | **Should replace RAFT as the default.** See below. |
| **AllTracker** | ICCV 2025 | **MIT** | Architecturally native to this problem. Investigate — it could delete the chain. |
| **MemFlow** | CVPR 2024 | **Apache-2.0** | Better "fast/temporal" option than RIFE. |
| RAFT | ECCV 2020 | BSD-3 | Superseded by SEA-RAFT from the same lab. |
| RIFE | ECCV 2022 | MIT | Works, but off-label — see below. |
| DOT | CVPR 2024 | MIT code, **CC-BY-NC front-end** | Right idea, unusable as shipped. |
| CoTracker3 | 2024 | **CC-BY-NC** | Blocked for commercial use. |
| VideoFlow | ICCV 2023 | not checked | Beaten by MemFlow on generalization with fewer params. |
| MEMFOF | 2025 | not checked | Reported SOTA on Sintel; worth a look if accuracy stalls. |
| NeuFlow v2 | 2024 | not checked | Real-time/edge focus; fallback if GPU inference is denied. |

### SEA-RAFT should be the default, not RAFT

Same lab as RAFT (princeton-vl), same BSD-3 licence, same two-frame interface — so it drops
into `FlowEstimator` with no architectural change. It reports at least **2.3× faster than
existing methods**, state of the art on Spring (3.69 EPE, 0.36 1px — a reported 22.9% and
17.8% error reduction), and **the best cross-dataset generalization on KITTI and Spring**.

That last one is the reason, not the speed. Every benchmark here is driving footage, animated
shorts and synthetic scenes; none of them are the film and commercial plates this plugin will
actually see. Cross-dataset generalization is the closest available proxy for "behaves on
material it was not trained on", which is the entire job.

The change costs nothing today because the estimator does not exist yet. Making it after
Phase 3 means redoing an export, a tensor contract and a golden test.

**Caveat:** no public ONNX export was found for SEA-RAFT, so we export from the checkpoint
ourselves. The plan already requires that for provenance reasons, so it is not extra work —
but it does mean no pre-existing export to check our tensor contract against.

### AllTracker could delete the chain entirely

`docs/plan.md` builds a chain of pairwise flows composed from the reference frame to the
current one, and accepts drift as a known weakness. That whole structure — `FlowChain`,
`FlowCache`, composition error, the `Smooth` mitigation — exists *only* because pairwise
optical flow has to be accumulated.

AllTracker estimates the flow field between a **query frame and every other frame** directly,
densely, at all pixels, in 16M parameters. That is this plugin's data model, natively. No
accumulation, therefore no drift, therefore no chain cache. DOT (CVPR 2024) does the same
thing and is explicitly framed as source-frame-to-target-frame with a visibility mask — but
its point-tracking front end is CoTracker, which is CC-BY-NC, so DOT as shipped is not
usable here. AllTracker is MIT with no CC-BY-NC dependency found.

**Why it is not the v1 plan anyway:** the paper reports tracking 768x1024 on a **40 GB GPU**.
That is below HD, on more VRAM than a Flame box is likely to spare while Flame itself is
running. Phase 0 probe item 3 — does ONNX Runtime get a CUDA device inside Flame's process,
and what does VRAM look like — is the measurement that decides whether this is reachable at
all. Until that number exists, committing the architecture to it would be guessing.

The right sequencing is unchanged: build the chain, keep `FlowEstimator` narrow, and treat
AllTracker as a candidate that could later replace the chain rather than plug into it.

### RIFE is off-label and MemFlow is the better fast option

RIFE's IFNet estimates *intermediate* flow in order to synthesise a frame between two others.
Its flow is a means to that end, optimised for producing a plausible in-between image, not for
being correct as a motion field. It will work as a tracker and it is fast, which is why it is
in the plan — but it is not what it was built for.

MemFlow (CVPR 2024, **Apache-2.0**) is purpose-built: real-time, uses memory across frames,
and reportedly beats VideoFlow on Sintel and KITTI-15 generalization with fewer parameters
and faster inference. Since this plugin renders sequentially through a shot, a model that
carries temporal state is a natural fit rather than an awkward one.

Worth reconsidering the RIFE slot against MemFlow before Phase 5. Both are permissive.

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
