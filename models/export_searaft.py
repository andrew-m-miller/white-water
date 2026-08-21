#!/usr/bin/env python3
"""Export the pinned SEA-RAFT M Phase 0B probe network to ONNX.

The source checkout and checkpoint are supplied explicitly. This script verifies both
against ``sea-raft-m.json`` before importing upstream code, exports a dynamic-spatial graph
for inputs already padded to a multiple of eight, and refuses to publish the artifact until
PyTorch/ONNX parity plus identity and bidirectional translation checks pass.

The export environment is intentionally separate from the plugin build; see
``models/requirements-searaft-export.txt``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "sea-raft-m.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_provenance(manifest: dict[str, Any], upstream: Path, checkpoint: Path) -> None:
    require((upstream / ".git").exists(), f"not a git checkout: {upstream}")
    actual_commit = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected_commit = manifest["upstream"]["commit"]
    require(
        actual_commit == expected_commit,
        f"SEA-RAFT checkout is {actual_commit}, expected {expected_commit}",
    )

    require(checkpoint.is_file(), f"checkpoint does not exist: {checkpoint}")
    expected_size = manifest["checkpoint"]["size_bytes"]
    require(
        checkpoint.stat().st_size == expected_size,
        f"checkpoint is {checkpoint.stat().st_size} bytes, expected {expected_size}",
    )
    actual_hash = sha256_file(checkpoint)
    expected_hash = manifest["checkpoint"]["sha256"]
    require(
        actual_hash == expected_hash,
        f"checkpoint SHA256 is {actual_hash}, expected {expected_hash}",
    )


def load_model(manifest: dict[str, Any], upstream: Path, checkpoint: Path, device: str):
    try:
        import torch
        import torchvision.models
        from safetensors.torch import load_model as load_safetensors_model
    except ImportError as exc:
        raise RuntimeError(
            "missing export dependency; install models/requirements-searaft-export.txt"
        ) from exc

    # raft.py inherits Hugging Face's hub mixin only to support from_pretrained(). That API
    # is not used for this pinned local checkpoint, and upstream does not pin a compatible
    # huggingface-hub version. Supply the inert inheritance hook the class declaration needs
    # instead of allowing an unrelated hub release to change export reproducibility.
    hub_stub = types.ModuleType("huggingface_hub")

    class ExportOnlyModelHubMixin:
        def __init_subclass__(cls, **_kwargs):
            super().__init_subclass__()

    hub_stub.PyTorchModelHubMixin = ExportOnlyModelHubMixin
    previous_hub_module = sys.modules.get("huggingface_hub")
    sys.modules["huggingface_hub"] = hub_stub
    sys.path.insert(0, str(upstream / "core"))
    try:
        from raft import RAFT
    finally:
        sys.path.pop(0)
        if previous_hub_module is None:
            del sys.modules["huggingface_hub"]
        else:
            sys.modules["huggingface_hub"] = previous_hub_module

    # scale is retained in the pinned upstream config, but it belongs to custom.py's caller
    # and is not read by RAFT.forward(). The ONNX contract is deliberately raw model tensor
    # resolution; the plugin owns any analysis resize and corresponding vector rescale.
    config = argparse.Namespace(**manifest["model"]["config"])
    # ResNetFPN downloads ImageNet initialization while constructing the network, immediately
    # before the complete SEA-RAFT checkpoint overwrites it. Suppress that redundant network
    # access without patching upstream source; strict checkpoint loading below proves every
    # parameter and buffer needed by the final model came from the pinned safetensors file.
    original_resnet18 = torchvision.models.resnet18
    original_resnet34 = torchvision.models.resnet34
    torchvision.models.resnet18 = lambda *args, **kwargs: original_resnet18(weights=None)
    torchvision.models.resnet34 = lambda *args, **kwargs: original_resnet34(weights=None)
    try:
        model = RAFT(config)
    finally:
        torchvision.models.resnet18 = original_resnet18
        torchvision.models.resnet34 = original_resnet34
    # Upstream registers four BatchNorm modules twice (as ``bn3`` and as
    # ``downsample.1``). Safetensors records the shared storage correctly, but its model
    # loader reports the duplicate scalar num_batches_tracked aliases as unexpected. Allow
    # only those non-learned counters; every parameter/buffer mismatch remains fatal.
    missing, unexpected = load_safetensors_model(model, str(checkpoint), strict=False)
    allowed_alias_counters = {
        f"{network}.layer{layer}.0.downsample.1.num_batches_tracked"
        for network in ("cnet", "fnet")
        for layer in (2, 3)
    }
    require(not missing, f"checkpoint is missing model keys: {missing}")
    disallowed_unexpected = sorted(set(unexpected) - allowed_alias_counters)
    require(
        not disallowed_unexpected,
        f"checkpoint has unexpected model keys: {disallowed_unexpected}",
    )
    model.eval().to(torch.device(device))
    return model


def export_onnx(model, manifest: dict[str, Any], output: Path, device: str) -> None:
    import torch

    class FinalFlow(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, image1, image2):
            return self.wrapped(image1, image2, test_mode=True)["final"]

    shape = manifest["export"]["example_shape"]
    require(shape[0] == 1 and shape[1] == 3, f"unsupported example shape: {shape}")
    require(shape[2] % 8 == 0 and shape[3] % 8 == 0, "example H and W must be multiples of 8")
    sample1 = torch.zeros(shape, dtype=torch.float32, device=device)
    sample2 = torch.zeros(shape, dtype=torch.float32, device=device)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=output.name + ".", suffix=".tmp", dir=output.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        torch.onnx.export(
            FinalFlow(model),
            (sample1, sample2),
            str(temporary),
            export_params=True,
            opset_version=manifest["export"]["opset"],
            do_constant_folding=True,
            input_names=[item["name"] for item in manifest["tensor_contract"]["inputs"]],
            output_names=[manifest["tensor_contract"]["output"]["name"]],
            dynamic_axes={
                "image1": {2: "height", 3: "width"},
                "image2": {2: "height", 3: "width"},
                "flow": {2: "height", 3: "width"},
            },
        )
        import onnx

        exported = onnx.load(str(temporary), load_external_data=False)
        onnx.checker.check_model(exported, full_check=True)
        foreign_nodes = sorted(
            {
                f"{node.domain or 'ai.onnx'}::{node.op_type}"
                for node in exported.graph.node
                if node.domain not in ("", "ai.onnx")
            }
        )
        require(
            not foreign_nodes,
            "export contains non-portable custom/ATen nodes: " + ", ".join(foreign_nodes),
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def synthetic_pair(torch, height: int, width: int, dx: int, seed: int, device: str):
    generator = torch.Generator(device=device).manual_seed(seed)
    first = torch.rand((1, 3, height, width), generator=generator, device=device) * 255.0
    first = torch.nn.functional.avg_pool2d(first, kernel_size=5, stride=1, padding=2)
    second = torch.zeros_like(first)
    second[..., dx:] = first[..., :-dx]
    return first, second


def validate_export(model, manifest: dict[str, Any], output: Path, device: str) -> dict[str, Any]:
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    validation = manifest["validation"]
    _, _, height, width = manifest["export"]["example_shape"]
    dx = validation["translation_pixels"]
    first, second = synthetic_pair(torch, height, width, dx, validation["seed"], device)

    with torch.no_grad():
        identity_pt = model(first, first, test_mode=True)["final"].cpu().numpy()
        forward_pt = model(first, second, test_mode=True)["final"].cpu().numpy()
        reverse_pt = model(second, first, test_mode=True)["final"].cpu().numpy()

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    input_names = [item["name"] for item in manifest["tensor_contract"]["inputs"]]
    output_name = manifest["tensor_contract"]["output"]["name"]

    def run_onnx(a, b):
        values = session.run(
            [output_name],
            {input_names[0]: a.cpu().numpy(), input_names[1]: b.cpu().numpy()},
        )
        return values[0]

    identity_onnx = run_onnx(first, first)
    forward_onnx = run_onnx(first, second)
    reverse_onnx = run_onnx(second, first)

    # A graph traced at one shape can advertise dynamic axes while retaining constants from
    # InputPadder, unfold or a reshape. Exercise a second multiple-of-eight shape before the
    # manifest is allowed to call this artifact dynamic.
    second_shape = validation["second_dynamic_shape"]
    require(second_shape[0] == 1 and second_shape[1] == 3,
            f"unsupported second dynamic shape: {second_shape}")
    require(second_shape[2] % 8 == 0 and second_shape[3] % 8 == 0,
            "second dynamic H and W must be multiples of 8")
    dynamic_first, dynamic_second = synthetic_pair(
        torch, second_shape[2], second_shape[3], dx, validation["seed"] + 1, device
    )
    with torch.no_grad():
        dynamic_pt = model(dynamic_first, dynamic_second, test_mode=True)["final"].cpu().numpy()
    dynamic_flow = run_onnx(dynamic_first, dynamic_second)
    require(list(dynamic_flow.shape) == [1, 2, second_shape[2], second_shape[3]],
            f"dynamic-shape run returned {list(dynamic_flow.shape)}")
    pairs = (
        ("identity", identity_pt, identity_onnx),
        ("forward", forward_pt, forward_onnx),
        ("reverse", reverse_pt, reverse_onnx),
        ("second_shape", dynamic_pt, dynamic_flow),
    )
    differences = np.concatenate([(pt - exported).reshape(-1) for _, pt, exported in pairs])
    absolute_differences = np.abs(differences)
    mean_abs = float(np.mean(absolute_differences))
    p99_abs = float(np.percentile(absolute_differences, 99.0))
    p999_abs = float(np.percentile(absolute_differences, 99.9))
    max_abs = float(np.max(absolute_differences))

    border = max(16, dx * 4)
    interior = np.s_[0, :, border:-border, border:-border]
    identity_epe = np.sqrt(np.sum(identity_onnx[interior] ** 2, axis=0))
    identity_median = float(np.median(identity_epe))
    forward_x = float(np.median(forward_onnx[0, 0, border:-border, border:-border]))
    forward_y = float(np.median(forward_onnx[0, 1, border:-border, border:-border]))
    reverse_x = float(np.median(reverse_onnx[0, 0, border:-border, border:-border]))
    reverse_y = float(np.median(reverse_onnx[0, 1, border:-border, border:-border]))

    minimum_x = dx * validation["translation_x_fraction_min"]
    graph = onnx.load(str(output), load_external_data=False)
    observed = {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "providers": session.get_providers(),
        },
        "graph_nodes": len(graph.graph.node),
        "graph_domains": sorted({node.domain or "ai.onnx" for node in graph.graph.node}),
        "onnx_pytorch_mean_abs": mean_abs,
        "onnx_pytorch_p99_abs": p99_abs,
        "onnx_pytorch_p999_abs": p999_abs,
        "onnx_pytorch_max_abs": max_abs,
        "identity_median_epe": identity_median,
        "forward_median": [forward_x, forward_y],
        "reverse_median": [reverse_x, reverse_y],
        "second_dynamic_shape": list(dynamic_flow.shape),
    }
    failures = []
    if mean_abs > validation["onnx_pytorch_mean_abs_max"]:
        failures.append("ONNX/PyTorch mean absolute error exceeds limit")
    if p99_abs > validation["onnx_pytorch_p99_abs_max"]:
        failures.append("ONNX/PyTorch p99 absolute error exceeds limit")
    if p999_abs > validation["onnx_pytorch_p999_abs_max"]:
        failures.append("ONNX/PyTorch p99.9 absolute error exceeds limit")
    if max_abs > validation["onnx_pytorch_max_abs_max"]:
        failures.append("ONNX/PyTorch max absolute error exceeds limit")
    if identity_median > validation["identity_median_epe_max"]:
        failures.append("identity median EPE exceeds limit")
    if forward_x < minimum_x:
        failures.append("image1->image2 median dx points the wrong way or is too small")
    if reverse_x > -minimum_x:
        failures.append("image2->image1 median dx points the wrong way or is too small")
    if abs(forward_y) > validation["translation_abs_y_max"]:
        failures.append("forward median dy exceeds limit")
    if abs(reverse_y) > validation["translation_abs_y_max"]:
        failures.append("reverse median dy exceeds limit")
    require(
        not failures,
        "; ".join(failures) + "; observed=" + json.dumps(observed, sort_keys=True),
    )
    return observed


def update_manifest(path: Path, manifest: dict[str, Any], output: Path, observed: dict[str, Any]) -> None:
    manifest["status"] = "host_probe_pending"
    manifest["export"]["sha256"] = sha256_file(output)
    manifest["export"]["size_bytes"] = output.stat().st_size
    manifest["validation"]["observed"] = observed
    with path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path, help="pinned SEA-RAFT checkout")
    parser.add_argument("--checkpoint", required=True, type=Path, help="pinned M safetensors file")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, help="output ONNX path; defaults beside the manifest")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--verify-provenance-only",
        action="store_true",
        help="check the source commit and checkpoint hash without importing ML dependencies",
    )
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="record the ONNX hash and validation measurements after a successful export",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = read_manifest(args.manifest)
    verify_provenance(manifest, args.upstream.resolve(), args.checkpoint.resolve())
    print("provenance: pinned source commit and checkpoint SHA256 verified")
    if args.verify_provenance_only:
        return 0

    output = args.output or (args.manifest.parent / manifest["export"]["artifact"])
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    model = load_model(manifest, args.upstream.resolve(), args.checkpoint.resolve(), args.device)
    with tempfile.NamedTemporaryFile(
        prefix=output.name + ".candidate.", suffix=".onnx", dir=output.parent, delete=False
    ) as stream:
        candidate = Path(stream.name)
    candidate.unlink()
    try:
        export_onnx(model, manifest, candidate, args.device)
        observed = validate_export(model, manifest, candidate, args.device)
        os.replace(candidate, output)
    finally:
        candidate.unlink(missing_ok=True)
    artifact_hash = sha256_file(output)
    print(f"artifact: {output}")
    print(f"sha256:   {artifact_hash}")
    print(f"validation: {json.dumps(observed, sort_keys=True)}")
    if args.update_manifest:
        update_manifest(args.manifest, manifest, output, observed)
        print(f"manifest updated: {args.manifest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"export_searaft.py: error: {exc}", file=sys.stderr)
        raise SystemExit(1)
