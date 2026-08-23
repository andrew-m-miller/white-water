# Models

Weights are **not committed**. They are exported from a pinned upstream checkpoint into
ONNX by the scripts here, verified against a recorded SHA256, and staged into
`Contents/Resources/models/` in the bundle at build time. `.gitignore` excludes them.

A checked-in blob is worse than a script for the one thing that actually matters here: a
licence audit has to be able to trace a shipped file back to a specific upstream release,
and a binary in git history proves nothing about where it came from.

## Selection state

**Nothing is selected to ship.** P25-0's protocol/schema package is implemented and under review
in PR #6, but no P25 bake-off candidate artifact has been measured or selected. The Phase 0B
SEA-RAFT M probe remains qualification evidence, not a P25 selection. Choice option order is
persistent setup API, so the default
and fast alternative are chosen only by the Phase 2.5 bake-off of the exact exported ONNX
artifacts. SEA-RAFT is the leading conservative candidate, WAFT is the quality/memory
candidate, and NeuFlow v2 is the leading fast stateless candidate. Original RAFT remains a
useful validation baseline; RIFE and MemFlow are not leading shipping candidates.

Phase 0B deliberately uses **SEA-RAFT M** as its representative real network. That decision
exists to exercise the CUDA provider and the runtime dependency closure with a plausible
workload; it does not choose a shipping default or assign a model choice index. Before the
host run, the probe gains a pinned export script and manifest containing the upstream commit,
checkpoint URL and SHA256, tensor contract, exported ONNX SHA256, and synthetic translation
validation.

**No licence in this document has been audited.** Repository statements are leads, not
approval. Before anything goes to a client, re-verify the exact code revision, checkpoint,
and any backbone weights actually shipped, then record the verdict here with a date and the
checkpoint SHA256.

Deployment posture changes which of these obligations actually bind — see **Deployment
posture** below. The short version: not distributing the plugin relaxes a lot, but it does
**not** unlock anything CC-BY-NC for commercial work.

## Candidates considered

Surveyed 2026-08-19. Licence column is what the upstream repository *states*; none of it
has been audited, and for a facility deliverable that audit is not optional.

| Model | Venue | Licence | Verdict |
|---|---|---|---|
| **SEA-RAFT** | ECCV 2024 (oral) | **BSD-3** | Leading conservative candidate and the Phase 0B probe network. Shipping role still requires the bake-off. |
| **WAFT** | ICLR 2026 (oral) | **BSD-3 code, backbone weights vary** | Quality/memory candidate. Strong upstream results, but export and checkpoint licensing need resolving. |
| **NeuFlow v2** | 2024 | not checked | Leading fast stateless candidate; export, quality and licence all require measurement. |
| **AllTracker** | ICCV 2025 | **MIT** | Architecturally native to this problem. Investigate — it could delete the chain. |
| RAFT | ECCV 2020 | BSD-3 | Known validation baseline, not a presumed shipping model. |
| RIFE | ECCV 2022 | MIT | Fast and exportable, but off-label for motion-field accuracy. |
| MemFlow | CVPR 2024 | Apache-2.0 | Stateful temporal design conflicts with arbitrary-order OFX rendering; reconsider only with a sequential durable-analysis architecture. |
| DOT | CVPR 2024 | MIT code, **CC-BY-NC front-end** | Right idea, unusable as shipped. |
| CoTracker3 | 2024 | **CC-BY-NC** | Blocked for commercial use. |
| VideoFlow | ICCV 2023 | not checked | Beaten by MemFlow on generalization with fewer params. |
| MEMFOF | 2025 | not checked | Reported SOTA on Sintel; worth a look if accuracy stalls. |

### WAFT is the quality/memory candidate

WAFT is not the default in advance of the bake-off. It is the strongest quality/memory
candidate identified by the survey: same lab as SEA-RAFT (princeton-vl, Jia Deng), BSD-3
code, ICLR 2026 oral, but with a more complicated export and checkpoint-licensing surface.

WAFT is RAFT with the **cost volume replaced by high-resolution warping**. Reported: **1st on
Spring, Sintel and KITTI**, **best zero-shot generalization on KITTI**, 1.3-4.1x faster than
methods with similar performance, and **2x lower memory**.

Three of those matter to us specifically, in this order:

1. **2x lower memory.** The cost volume is what makes the RAFT family memory-hungry -- it is
   O((HW)^2) before pooling and is a major reason the plugin needs explicit megapixel caps.
   Removing it attacks the single hardest constraint this project has: fitting an inference
   context alongside Flame, which already owns the GPU. Phase 0B measures the available
   envelope with SEA-RAFT M; the Phase 2.5 bake-off then compares every candidate at equal
   pixel budgets rather than assuming a fractional analysis scale.
2. **Best zero-shot generalization.** Same argument as for SEA-RAFT and still the most
   important accuracy number here: the benchmarks are driving footage, animated shorts and
   synthetic scenes, and none of them are the plates this plugin will see.
3. **Export may be simpler, not harder.** The correlation lookup is the awkward part of
   exporting RAFT to ONNX. Warping is `grid_sample`, which ONNX has supported since opset 16
   and ONNX Runtime implements. This is an expectation, not a measurement -- the iterative
   refinement loop is still there, so the unroll-at-export problem in the tensor contract
   section is unchanged.

**The catch, and it is a real one: the code licence is not the checkpoint licence.**

**P25-3E provenance result (2026-08-23):** `models/waft-twins.json` is a typed exclusion
record, not an artifact manifest. The official WAFT `waftv2` source is pinned to commit
`b152ff1cad1af8c185ee7b141997c48ff3334c87` and its BSD-3-Clause code terms are audited, but
the linked A2 Drive object exposes only an aggregate `a2.zip` (3,702,705,327 bytes), not an
immutable Twins checkpoint file. WAFT also calls `timm`'s `twins_svt_large` with
`pretrained=True` without pinning the pretrained weight. Checkpoint and backbone identity,
commercial-use, and redistribution verdicts therefore remain **unknown**; no ONNX export or
tensor qualification is claimed. The candidate is excluded from the bake-off until the exact
file-level evidence listed in the record is available.

WAFT supports three backbones -- Twins, DAv2 (Depth Anything v2) and DINOv3 -- and the
README states no licence for the weights separately from the BSD-3 code, nor which checkpoint
uses which backbone. That mapping exists only inside the linked Google Drive folder. The
backbones do not share terms:

| Backbone | Licence | For us |
|---|---|---|
| Twins | Apache-2.0 | Clean. Prefer this if it performs adequately. |
| DAv2 Small | Apache-2.0 | Clean. |
| DAv2 Base / Large / Giant | **CC-BY-NC-4.0** | **Blocked.** Non-commercial. |
| DINOv3 | Meta custom licence | Permits commercial use, forbids military use, and **requires the licence to travel with any redistributed weights** -- which is exactly what shipping a checkpoint inside a .ofx.bundle is. Bespoke, so it needs an actual legal read rather than a shrug. |

So evaluating WAFT is two decisions: qualify the architecture and export, then separately
choose a checkpoint whose backbone we can ship. **Action during Phase 2.5:**
pull the model zoo, record which backbone each checkpoint uses and its SHA256 in this
document, and benchmark the Twins-backbone checkpoint against the recommended one. If the
gap is small, Twins ends the question. If it is large, DINOv3 becomes a legal question and
SEA-RAFT (unambiguously BSD-3, no foundation-model backbone) remains the conservative option.

### The general rule this keeps producing

Three times now the licence that matters has not been the one on the repository page:
CoTracker's CC-BY-NC inside DOT's MIT, Practical-RIFE's weights versus its code, and now
WAFT's backbones versus its BSD-3. Treat **code licence, checkpoint licence, and the
checkpoint's backbone licence as three separate questions**, and record all three per shipped
file. A "BSD-3" badge on a repository says nothing about the file we actually put in the
bundle.

### AllTracker could delete the chain entirely

`docs/plan.md` builds a chain of pairwise flows composed from the reference frame to the
current one, and accepts drift as a known weakness. That whole structure — `FlowChain`,
`FlowCache` and composition error — exists *only* because pairwise optical flow has to be
accumulated. `Smooth` addresses local spatial noise and is not drift mitigation.

AllTracker estimates the flow field between a **query frame and every other frame** directly,
densely, at all pixels, in 16M parameters. That is this plugin's data model, natively. No
accumulation, therefore no drift, therefore no chain cache. DOT (CVPR 2024) does the same
thing and is explicitly framed as source-frame-to-target-frame with a visibility mask — but
its point-tracking front end is CoTracker, which is CC-BY-NC, so DOT as shipped is not
usable here. AllTracker is MIT with no CC-BY-NC dependency found.

**Why it is not the v1 plan anyway:** the paper reports tracking 768x1024 on a **40 GB GPU**.
That is below HD, on more VRAM than a Flame box is likely to spare while Flame itself is
running. Phase 0B — a real network on the CUDA EP beside Flame — is the measurement that
starts to bound whether this is reachable at all. Until that number exists, committing the
architecture to it would be guessing.

The right sequencing is unchanged: build the pairwise chain behind
`PairwiseFlowEstimator`, keep chain orchestration in `FlowPreparation`, and treat AllTracker
as a candidate for a separate `ReferenceFlowEstimator` that could later replace the chain
rather than plug into it.

### RIFE is off-label; MemFlow does not fit arbitrary render order

RIFE's IFNet estimates *intermediate* flow in order to synthesise a frame between two others.
Its flow is a means to that end, optimised for producing a plausible in-between image, not for
being correct as a motion field. It may work as a tracker and it is fast, but exportability
alone is not enough to put it in the shipping pair.

MemFlow (CVPR 2024, **Apache-2.0**) is purpose-built optical flow and uses memory across
frames, but an OFX host may render frames out of order, revisit them, duplicate nodes, or
restart in another process. Reconstructing hidden temporal state makes results depend on
render history unless the plugin owns a sequential analysis pass and durable cache. That is
the wrong contract for the current on-demand design.

NeuFlow v2 is therefore the leading fast candidate: stateless pairwise inference matches the
host contract. Its exact licence, export and image-quality behavior remain bake-off work.
MemFlow can be reconsidered only if 0C leads to an explicit sequential, durable-analysis
architecture.

## Deployment posture, and what it does and does not change

Raised 2026-08-19: if the plugin is used only inside MTI Film -- on commercial work, but
never handed to anyone outside -- does the licensing picture change?

Yes, but not where it would help most, and in one place the usual intuition inverts.

**None of this is legal advice.** It is a reading of what the licences say. The
NonCommercial point below is load-bearing enough that counsel should confirm it before any
of this is on a show.

### What internal-only genuinely fixes

Most obligations in these licences are triggered by **distribution**, not by use.

- **DINOv3 becomes much easier.** Its friction was the clause requiring the licence to
  travel with redistributed weights -- which is exactly what shipping a checkpoint inside a
  `.ofx.bundle` is. Never redistribute and that clause does not fire; commercial use was
  already permitted. A DINOv3-backbone WAFT checkpoint goes from "needs a legal read" to
  "read the acceptable-use terms." Note it still forbids military use.
- **BSD-3 / MIT / Apache-2.0 attribution obligations go to roughly zero**, being
  redistribution-triggered. They were never the obstacle.

### What it does not fix

**CC-BY-NC restricts use, not only distribution.** This is the part that catches people,
because it runs opposite to the usual intuition about internal tools. NonCommercial is
defined in terms of purposes directed towards commercial advantage or monetary
compensation, and it governs exercising the licensed rights at all -- including simply
running the model. Producing shots that get billed to a client is commercial use whether or
not the software ever leaves the building.

So **CoTracker3, DOT as shipped, and DAv2 Base/Large/Giant remain blocked** for commercial
post work under an internal-only posture. Not distributing buys nothing here.

### What does not move either way

- **Training-data provenance is a separate risk from the model licence** and is unaffected
  by deployment posture. There is an open upstream issue asking exactly this about the
  *Apache-2.0* Depth Anything V2 Small variant. A permissive model licence is not a warranty
  that the training data was clean.
- **"We will never distribute it" erodes.** The day a partner facility asks, or a client
  does, or somebody wants to productise this, every obligation re-triggers -- with the model
  already baked into delivered shows and no cheap way back.

### Why the shipped default should stay permissive anyway

The plan's weight-resolution order -- `Model Dir` parameter, then `$WHITEWATER_MODEL_DIR`,
then the bundle -- already separates these two problems, and that is worth keeping
deliberately rather than by accident.

Whatever the bake-off selects, ship the bundle with permissive weights only (for example
SEA-RAFT, or a qualified Twins/DAv2-Small WAFT checkpoint). The **plugin** is then
distributable by construction, and its licensing story is one sentence long. Anyone who has
satisfied themselves about other weights points the override at them, and that decision
belongs to them and their situation rather than to the software.

This costs nothing under an internal-only posture and keeps every future option open, which
is the whole argument for doing it now rather than when it is expensive.

## Tensor contract

Each model's contract is data, not code — see `src/infer/ModelSpec` when it exists. What a
spec has to pin down:

- input and output tensor names, and their layout (NCHW, channel order, batch)
- normalization: range, mean/scale, and whether the model wants 0-1 or -1-1
- the **padding multiple**, measured for the exact export rather than copied from a model
  family name. Inputs are reflect-padded up and the flow cropped back.
- iteration handling. For iterative architectures a traced ONNX graph may bake the loop in,
  so an `Iters` parameter means either several exported variants or an export that keeps the
  loop dynamic. Decide this at export, not at runtime.

## Export

The first export script arrives in **Phase 0B**: `export_searaft.py`, producing the pinned
SEA-RAFT M probe artifact. Phase 2.5 adds one script per bake-off candidate; Phase 3 integrates
only the selected artifacts into the runtime and bundle. Each export must:

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

### Phase 0B export status

Work started 2026-08-20. `sea-raft-m.json` now pins the official SEA-RAFT source at commit
`9137517ba24e628442aec097d3afe71d03503b75` and the author's Hugging Face M checkpoint at
its file-producing revision `ea21e467a7076978b251e09d55751fcce166c2f8`. The checkpoint
was downloaded and measured as 78,778,760 bytes with SHA256
`cb8cfbf14c5e0f6734b64add383708b7ff68cc6089a0007c67165d4761346102`.

The manifest records the exact tensor contract and export settings. `export_searaft.py`
refuses a source or checkpoint mismatch, exports only the final `image1 -> image2` flow,
and checks PyTorch/ONNX parity, identity, positive-X translation, and the reverse negative-X
translation before publishing the artifact.

The first export completed on macOS arm64 with PyTorch 2.2.0 and ONNX Runtime 1.29.0 CPU.
The 78,840,944-byte opset-17 artifact has SHA256
`23cc2c850d3c116df193a24ff9ae7722d5635cd04e75dd8aeb20d7e13e4f59f1`; it contains 1,969
standard `ai.onnx` nodes and no custom or ATen domain. A second 160x256 input proved the
dynamic spatial axes. On a four-pixel synthetic translation, the exported model reported
median flow `(4.0042, 0.0087)` forward and `(-4.0097, -0.0004)` reverse; identical inputs
had median EPE 0.0039 px. The manifest holds the full parity distribution and thresholds.

ONNX protobuf bytes are not stable across the macOS arm64 and Linux x86-64 PyTorch
exporters even when the pinned weights, graph structure and numerical validation agree. The
artifact workflow therefore updates its checkout's manifest only after validation succeeds,
and stages that manifest beside the model so every test bundle records the exact hash and
measurements of the bytes it contains. The checked-in hash remains the first verified macOS
export; it is not substituted for the target build's measured hash.

The 2026-08-21 Flame 2026.2 run passed identity and both translation directions through
private ONNX Runtime 1.29 on CPU and CUDA. This qualifies the real network and CUDA provider
on that host/runtime pair; it does not choose a shipping model. Follow-up runs measured
warmed CPU/CUDA timing and device-wide VRAM at 480×640, 720×1280 and 1080×1920, repeated
lifecycle, duplicate-node equivalence, cancellation, provider-init fallback, and a bounded
64 MiB CUDA-arena failure followed by fresh-session numerical CPU recovery. Complete CUDA
dependency closure and qualification above 1080p remain in Phase 0B. Automatic production
fallback is Phase 4 behavior and is not established by the recovery probe.

The first installed probe exposed a packaging fault before that pass: the ONNX existed but
was mode `0600`, so the distinct Flame runtime user could not read it. Published model and
manifest files are now required to be regular mode-`0644` files, and CI checks the exported,
staged and extracted-package copies as well as their size and SHA256.

The artifact remains ignored by git and must be staged from a qualified build input.

```bash
python3.10 -m venv /path/to/searaft-export-env
/path/to/searaft-export-env/bin/pip install -r models/requirements-searaft-export.txt
/path/to/searaft-export-env/bin/python models/export_searaft.py \
  --upstream /path/to/SEA-RAFT-at-9137517 \
  --checkpoint /path/to/model.safetensors \
  --output models/sea-raft-m-opset17.onnx \
  --update-manifest
```

## Resolution at runtime

The plugin looks for weights in this order, and prints which one won to stderr — which
Flame captures in `/opt/Autodesk/log/`, the only diagnostic channel on the box:

1. the `Model Dir` parameter
2. `$WHITEWATER_MODEL_DIR`
3. `Contents/Resources/models/` inside the bundle
