# WAFT/Twins evaluation export

`models/waft-twins-artifact.json` is an evaluation-only record. It keeps the selected
checkpoint's commercial-use and redistribution verdicts as `unknown`, and therefore remains
excluded from shipping even if its ONNX export qualifies numerically. The ONNX and checkpoint
payloads are ignored by git.

The exporter has no network path. Before running it, an operator must provide all of these
inputs explicitly:

- a local WAFT checkout at commit
  `b152ff1cad1af8c185ee7b141997c48ff3334c87`;
- the extracted member `waftv2-ckpts/twins/zero-shot.pth`, exactly 544,230,582 bytes with
  SHA256 `f750cd15281fc30de477723438ff4a67fe1591deac4ab0eb9b366e27c827e070`; and
- the pinned config in that checkout, normally
  `config/a2/twins/chairs-things.json`, which must declare `feature_encoder=twins`,
  `iterative_module`, and `iters`; its pinned SHA256 is
  `4eb827762b132fe0e90b4d87e456088e772573b4f346d5e396e0912dad528996`.

If the checkpoint is still inside the already-verified official `a2.zip`, extraction is a
separate operator action; the exporter never opens a Drive URL or downloads a missing file:

```sh
unzip -p /path/to/a2.zip waftv2-ckpts/twins/zero-shot.pth > /path/to/zero-shot.pth
chmod 0644 /path/to/zero-shot.pth
shasum -a 256 /path/to/zero-shot.pth
```

Install the upstream/export environment explicitly (including ONNX Runtime and any required
`xformers` build) before invoking the script. A missing dependency is a typed blocker; the
script does not run `pip`, `conda`, or a model hub helper.

```sh
python3 models/export_waft.py \
  --upstream /path/to/WAFT \
  --checkpoint /path/to/zero-shot.pth \
  --config /path/to/WAFT/config/a2/twins/chairs-things.json \
  --manifest models/waft-twins-artifact.json \
  --output models/waft-twins-opset17.onnx \
  --platform macos-arm64 \
  --device cpu \
  --provider CPUExecutionProvider \
  --update-manifest

python3 models/check_waft_artifact.py
```

The exporter verifies the source/checkpoint identity, constructs Twins with pretrained
initialization disabled, loads the checkpoint with `weights_only=True` and `strict=True`,
checks ONNX domains/operators and the declared opset, then runs PyTorch-vs-ONNX parity,
identity, signed forward/reverse translation, and a second dynamic shape. It publishes the
ONNX only after all checks pass, with mode `0644`, size, SHA256, and environment hash recorded.
The source worktree must be clean, `--config` must name the exact pinned path (modified copies
are rejected), and the requested provider must be active as the first ONNX Runtime provider;
CUDA qualification uses `--device cuda --provider CUDAExecutionProvider` and refuses CPU
fallback.

With `--update-manifest`, a missing local asset, unsupported operator, strict-load error,
parity error, or direction error is recorded as a typed pending
`validation.observed.technical_blocker`; it leaves the export hash/size unset and never changes
the `checkpoint_license_terms_unknown` shipping exclusion. A previously successful artifact
record is not erased by a later failed attempt.
