#!/usr/bin/env python3
"""Reproduce and locally validate the pinned NeuFlow v2 ONNX export.

The upstream implementation stores grids, correlation buffers and iteration context on the
model after ``init_bhwd``.  This exporter makes that fixed-shape state explicit and exports the
planned 432x768 graph without dynamic axes.  A graph is published only after ONNX checker,
CPU Runtime, PyTorch/reference parity, identity, and both signed translation checks pass.
Large ONNX payloads are ignored by git; the manifest records their exact hash and size.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from typing import Any, Optional

try:
    from artifact_workflow import (
        ArtifactError,
        environment_sha256,
        load_manifest,
        publish_file,
        require_regular_mode,
        sha256_file,
        update_platform_export,
        write_manifest,
    )
except ModuleNotFoundError:  # pragma: no cover - package import path
    from .artifact_workflow import (
        ArtifactError,
        environment_sha256,
        load_manifest,
        publish_file,
        require_regular_mode,
        sha256_file,
        update_platform_export,
        write_manifest,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "neuflow-v2.json"
EXPECTED_SOURCE_COMMIT = "204b5e3744461d90303b9ff82caa7a1bb56a2ca2"
EXPECTED_CHECKPOINT_SHA256 = "76152c8068f247a7d073aa13e61da8cb4c3c6a798076d4dc8e20f7995fcc019f"
EXPECTED_CHECKPOINT_SIZE = 36195519


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_provenance(manifest: dict[str, Any], upstream: Path, checkpoint: Path) -> None:
    """Refuse to import model code until source and checkpoint identity is exact."""

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
        f"NeuFlow v2 checkout is {actual_commit}, expected {expected_commit}",
    )
    status = subprocess.run(
        [
            "git",
            "-C",
            str(upstream),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(
        not status,
        "NeuFlow v2 checkout is dirty; refusing to export uncommitted source: "
        + status,
    )
    require(
        expected_commit == EXPECTED_SOURCE_COMMIT,
        "manifest changed the audited NeuFlow v2 source commit",
    )

    require_regular_mode(checkpoint, "NeuFlow checkpoint")
    expected_size = manifest["checkpoint"]["size_bytes"]
    require(
        expected_size == EXPECTED_CHECKPOINT_SIZE,
        "manifest changed the audited checkpoint size",
    )
    require(
        checkpoint.stat().st_size == expected_size,
        f"checkpoint is {checkpoint.stat().st_size} bytes, expected {expected_size}",
    )
    actual_hash = sha256_file(checkpoint)
    expected_hash = manifest["checkpoint"]["sha256"]
    require(
        expected_hash == EXPECTED_CHECKPOINT_SHA256,
        "manifest changed the audited checkpoint SHA256",
    )
    require(
        actual_hash == expected_hash,
        f"checkpoint SHA256 is {actual_hash}, expected {expected_hash}",
    )


def _import_upstream(upstream: Path):
    """Import only the pinned official class, with hub metadata made inert."""

    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "missing export dependency; install models/requirements-neuflow-export.txt"
        ) from exc

    # NeuFlow subclasses the Hugging Face mixin only to expose metadata.  Exporting from a
    # local checkpoint never calls that API, so an inert mixin avoids an unpinned hub import.
    hub_stub = types.ModuleType("huggingface_hub")

    class ExportOnlyModelHubMixin:
        def __init_subclass__(cls, **_kwargs):
            super().__init_subclass__()

    hub_stub.PyTorchModelHubMixin = ExportOnlyModelHubMixin
    previous_hub_module = sys.modules.get("huggingface_hub")
    previous_path = list(sys.path)
    sys.modules["huggingface_hub"] = hub_stub
    sys.path.insert(0, str(upstream))
    try:
        from NeuFlow.neuflow import NeuFlow
    finally:
        sys.path[:] = previous_path
        if previous_hub_module is None:
            del sys.modules["huggingface_hub"]
        else:
            sys.modules["huggingface_hub"] = previous_hub_module
    return NeuFlow


def load_model(
    manifest: dict[str, Any], upstream: Path, checkpoint: Path, device: str
):
    import torch

    NeuFlow = _import_upstream(upstream)
    model = NeuFlow()
    checkpoint_data = torch.load(str(checkpoint), map_location="cpu")
    require(
        isinstance(checkpoint_data, dict) and "model" in checkpoint_data,
        "checkpoint has no official 'model' state key",
    )
    state_dict = checkpoint_data["model"]
    require(isinstance(state_dict, dict), "checkpoint model state is not a mapping")
    incompatible = model.load_state_dict(state_dict, strict=True)
    require(
        not incompatible.missing_keys and not incompatible.unexpected_keys,
        "checkpoint state did not load strictly",
    )

    shape = manifest["export"]["example_shape"]
    require(shape[0] == 1 and shape[1] == 3, f"unsupported example shape: {shape}")
    require(
        shape[2] % 16 == 0 and shape[3] % 16 == 0,
        "NeuFlow v2 H and W must be divisible by 16",
    )
    model.eval().to(torch.device(device))
    # These are shape-dependent plain attributes in the upstream model, not registered
    # buffers. Initializing them before tracing is therefore part of the export identity.
    model.init_bhwd(1, shape[2], shape[3], device, amp=False)
    return model


def export_onnx(model, manifest: dict[str, Any], output: Path, device: str) -> None:
    import onnx
    import torch

    class FinalFlow(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, image1, image2):
            # The official defaults are iters_s16=1 and iters_s8=8. Passing constants here
            # ensures the exported graph cannot depend on a runtime iteration API.
            return self.wrapped(image1, image2, iters_s16=1, iters_s8=8)[-1]

    shape = manifest["export"]["example_shape"]
    sample1 = torch.zeros(shape, dtype=torch.float32, device=device)
    sample2 = torch.zeros(shape, dtype=torch.float32, device=device)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=output.name + ".", suffix=".tmp", dir=output.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    original_sdpa = torch.nn.functional.scaled_dot_product_attention

    def onnx_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        """Lower the official no-mask/no-dropout SDPA calls to portable ONNX operators."""

        require(attn_mask is None, "NeuFlow export encountered an attention mask")
        require(dropout_p == 0.0, "NeuFlow export encountered nonzero attention dropout")
        require(not is_causal, "NeuFlow export encountered causal attention")
        factor = scale if scale is not None else query.shape[-1] ** -0.5
        scores = torch.matmul(query, key.transpose(-2, -1)) * factor
        return torch.matmul(torch.softmax(scores, dim=-1), value)

    # The official ``infer.py`` has an optional in-place Conv/BN fusion helper.  It is not
    # needed here: the eval-mode exporter folds the fixed BatchNorm parameters into ordinary
    # ONNX Conv nodes under ``do_constant_folding=True``.  Keeping the upstream module intact
    # makes the PyTorch reference and exported graph use the same deterministic weights.
    wrapper = FinalFlow(model).eval()
    try:
        # PyTorch 2.0.1 has no ONNX symbolic for aten::scaled_dot_product_attention. The
        # pinned upstream calls it only with the defaults above, so this deterministic lowering
        # is algebraically the same operation and is scoped to export; validation restores the
        # official implementation before calculating the reference result.
        torch.nn.functional.scaled_dot_product_attention = onnx_sdpa
        # Deliberately no dynamic_axes: init_bhwd state and correlation tensors are fixed to
        # the planned shape until a separate dynamic implementation is proven.
        torch.onnx.export(
            wrapper,
            (sample1, sample2),
            str(temporary),
            export_params=True,
            opset_version=manifest["export"]["opset"],
            do_constant_folding=True,
            input_names=[item["name"] for item in manifest["tensor_contract"]["inputs"]],
            output_names=[manifest["tensor_contract"]["output"]["name"]],
        )
    finally:
        torch.nn.functional.scaled_dot_product_attention = original_sdpa
    try:
        exported = onnx.load(str(temporary), load_external_data=False)
        onnx.checker.check_model(exported, full_check=True)
        expected_input_shape = list(manifest["export"]["example_shape"])
        for value in exported.graph.input:
            actual_shape = [
                dimension.dim_value if dimension.HasField("dim_value") else None
                for dimension in value.type.tensor_type.shape.dim
            ]
            require(
                actual_shape == expected_input_shape,
                f"{value.name} is not fixed to {expected_input_shape}: {actual_shape}",
            )
        expected_output_shape = [1, 2, expected_input_shape[2], expected_input_shape[3]]
        output_shape = [
            dimension.dim_value if dimension.HasField("dim_value") else None
            for dimension in exported.graph.output[0].type.tensor_type.shape.dim
        ]
        require(
            output_shape == expected_output_shape,
            f"flow is not fixed to {expected_output_shape}: {output_shape}",
        )
        foreign_nodes = sorted(
            {
                f"{node.domain or 'ai.onnx'}::{node.op_type}"
                for node in exported.graph.node
                if node.domain not in ("", "ai.onnx")
            }
        )
        require(
            not foreign_nodes,
            "export contains non-portable custom/ATen nodes: "
            + ", ".join(foreign_nodes),
        )
        publish_file(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def synthetic_pair(torch, height: int, width: int, dx: int, seed: int, device: str):
    generator = torch.Generator(device=device).manual_seed(seed)
    first = torch.rand(
        (1, 3, height, width), generator=generator, device=device
    ) * 255.0
    first = torch.nn.functional.avg_pool2d(first, kernel_size=5, stride=1, padding=2)
    second = torch.zeros_like(first)
    second[..., dx:] = first[..., :-dx]
    return first, second


def _run_reference(model, first, second):
    """Run official PyTorch code without its in-place /255 mutation leaking between cases."""

    with __import__("torch").no_grad():
        return model(first.clone(), second.clone())[-1].cpu().numpy()


def validate_export(
    model, manifest: dict[str, Any], output: Path, device: str
) -> dict[str, Any]:
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    validation = manifest["validation"]
    _, _, height, width = manifest["export"]["example_shape"]
    dx = int(validation["translation_pixels"])
    first, second = synthetic_pair(
        torch, height, width, dx, validation["seed"], device
    )
    identity_pt = _run_reference(model, first, first)
    forward_pt = _run_reference(model, first, second)
    reverse_pt = _run_reference(model, second, first)

    try:
        session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise RuntimeError(f"ONNX Runtime CPU session creation failed: {exc}") from exc
    input_names = [item["name"] for item in manifest["tensor_contract"]["inputs"]]
    output_name = manifest["tensor_contract"]["output"]["name"]

    def run_onnx(a, b):
        try:
            return session.run(
                [output_name],
                {input_names[0]: a.cpu().numpy(), input_names[1]: b.cpu().numpy()},
            )[0]
        except Exception as exc:
            raise RuntimeError(f"ONNX Runtime CPU inference failed: {exc}") from exc

    identity_onnx = run_onnx(first, first)
    forward_onnx = run_onnx(first, second)
    reverse_onnx = run_onnx(second, first)
    require(
        list(identity_onnx.shape) == [1, 2, height, width],
        f"unexpected ONNX output shape {list(identity_onnx.shape)}",
    )
    require(identity_onnx.dtype == np.float32, f"unexpected ONNX output dtype {identity_onnx.dtype}")

    differences = np.concatenate(
        [
            (identity_pt - identity_onnx).reshape(-1),
            (forward_pt - forward_onnx).reshape(-1),
            (reverse_pt - reverse_onnx).reshape(-1),
        ]
    )
    absolute = np.abs(differences)
    mean_abs = float(np.mean(absolute))
    p99_abs = float(np.percentile(absolute, 99.0))
    p999_abs = float(np.percentile(absolute, 99.9))
    max_abs = float(np.max(absolute))

    border = max(16, dx * 4)
    interior = np.s_[0, :, border:-border, border:-border]
    identity_epe = np.sqrt(np.sum(identity_onnx[interior] ** 2, axis=0))
    identity_median = float(np.median(identity_epe))
    forward_x = float(
        np.median(forward_onnx[0, 0, border:-border, border:-border])
    )
    forward_y = float(
        np.median(forward_onnx[0, 1, border:-border, border:-border])
    )
    reverse_x = float(
        np.median(reverse_onnx[0, 0, border:-border, border:-border])
    )
    reverse_y = float(
        np.median(reverse_onnx[0, 1, border:-border, border:-border])
    )

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
    minimum_x = abs(dx) * validation["translation_x_fraction_min"]
    if forward_x < minimum_x:
        failures.append("image1->image2 median dx points the wrong way or is too small")
    if reverse_x > -minimum_x:
        failures.append("image2->image1 median dx points the wrong way or is too small")
    if abs(forward_y) > validation["translation_abs_y_max"]:
        failures.append("forward median dy exceeds limit")
    if abs(reverse_y) > validation["translation_abs_y_max"]:
        failures.append("reverse median dy exceeds limit")

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
        "output_shape": list(identity_onnx.shape),
        "output_dtype": str(identity_onnx.dtype),
        "shape_support": "fixed",
    }
    require(
        not failures,
        "; ".join(failures) + "; observed=" + json.dumps(observed, sort_keys=True),
    )
    return observed


def _platform_id(value: Optional[str]) -> str:
    if value:
        return value
    if sys.platform == "darwin":
        return "macos-arm64"
    if sys.platform.startswith("linux"):
        return "linux-x86_64"
    return f"{sys.platform}-{platform.machine().lower()}"


def _record_validation(manifest: dict[str, Any], observed: dict[str, Any]) -> None:
    forward = observed["forward_median"]
    reverse = observed["reverse_median"]
    shape = manifest["export"]["example_shape"]
    validation = manifest["validation"]
    validation["status"] = "passed"
    validation["identity"] = {
        "passed": observed["identity_median_epe"] <= validation["identity_median_epe_max"],
        "median_epe_px": observed["identity_median_epe"],
    }
    validation["directions"] = {
        "forward": {
            "median_dx_px": forward[0],
            "median_dy_px": forward[1],
            "expected_sign": "positive_x",
        },
        "reverse": {
            "median_dx_px": reverse[0],
            "median_dy_px": reverse[1],
            "expected_sign": "negative_x",
        },
    }
    validation["shapes"] = {
        "dynamic": False,
        "example": [1, 2, shape[2], shape[3]],
        "additional": [1, 2, shape[2], shape[3]],
    }
    validation["parity"] = {
        "checked": True,
        "mean_abs": observed["onnx_pytorch_mean_abs"],
        "p99_abs": observed["onnx_pytorch_p99_abs"],
        "p999_abs": observed["onnx_pytorch_p999_abs"],
        "max_abs": observed["onnx_pytorch_max_abs"],
    }
    validation["observed"] = observed


def update_manifest(
    path: Path,
    manifest: dict[str, Any],
    output: Path,
    observed: dict[str, Any],
    platform_id: str,
) -> None:
    previous_observed = manifest.get("validation", {}).get("observed")
    if (
        manifest["status"] == "excluded"
        and isinstance(previous_observed, dict)
        and previous_observed.get("numerical_status") == "passed"
        and previous_observed.get("admission_status") == "excluded"
    ):
        raise ArtifactError(
            "refusing to promote a numerically validated excluded NeuFlow manifest; "
            "copy a fresh pending manifest before updating export identity"
        )
    # Record the numerical evidence once; update_platform_export below only records bytes and
    # environment identity.
    _record_validation(manifest, observed)
    # A prior ``--record-failure`` attempt may have marked the same working manifest excluded;
    # a newly passing exact export supersedes that transient result and its diagnostic notes.
    manifest["candidate"]["role"] = "shipping-candidate"
    manifest["notes"] = [
        note
        for note in manifest["notes"]
        if not note.startswith(
            (
                "export_result=excluded",
                "failure_stage=",
                "failure_type=",
                "failure_detail=",
                "artifact_identity=",
            )
        )
    ]
    env_observed = observed["environment"]
    providers = env_observed.get("providers", ["CPUExecutionProvider"])
    environment = {
        "platform": "macos" if platform_id.startswith("macos") else "linux",
        "architecture": "arm64" if platform_id.endswith("arm64") else "x86_64",
        "python": env_observed["python"],
        "framework": f"pytorch=={env_observed['pytorch']}",
        "exporter": f"torch.onnx=={env_observed['pytorch']}; onnx=={env_observed['onnx']}",
        "runtime": f"onnxruntime=={env_observed['onnxruntime']}",
        "provider": ",".join(providers),
    }
    environment["sha256"] = environment_sha256(environment)
    update_platform_export(
        manifest, platform=platform_id, artifact=output, environment=environment
    )
    # update_platform_export prepares the common platform/hash fields and marks host_probe_pending;
    # this F2 package has only exact local export/CPU evidence, so use the narrower status.
    manifest["status"] = "export_validated"
    result_note = "export_result=exact_fixed_shape_export_and_cpu_parity_passed"
    if result_note not in manifest["notes"]:
        manifest["notes"].append(result_note)
    write_manifest(path, manifest)


def record_failure(
    path: Path, manifest: dict[str, Any], stage: str, error: BaseException
) -> None:
    """Record a typed reproducible exclusion without inventing an artifact identity."""

    observed = manifest.get("validation", {}).get("observed")
    if (
        manifest["status"] == "excluded"
        and isinstance(observed, dict)
        and observed.get("numerical_status") == "passed"
        and observed.get("admission_status") == "excluded"
    ):
        raise ArtifactError(
            "refusing to overwrite a numerically validated excluded NeuFlow manifest; "
            "copy a fresh pending manifest before recording a new export failure"
        )
    if manifest["status"] not in {
        "provenance_pinned_export_pending",
        "excluded",
    }:
        raise ArtifactError(
            "refusing to overwrite a validated NeuFlow manifest; copy a fresh pending manifest "
            "before recording a new export failure"
        )
    detail = " ".join(str(error).split())
    if len(detail) > 600:
        detail = detail[:597] + "..."
    manifest["status"] = "excluded"
    manifest["candidate"]["role"] = "excluded"
    validation = dict(manifest["validation"])
    validation["status"] = "failed"
    validation["observed"] = {
        "failure_stage": stage,
        "failure_type": type(error).__name__,
        "failure_detail": detail,
    }
    manifest["validation"] = validation
    manifest["notes"] = [
        note
        for note in manifest["notes"]
        if not note.startswith(
            (
                "export_result=excluded",
                "failure_stage=",
                "failure_type=",
                "failure_detail=",
                "artifact_identity=",
            )
        )
    ]
    manifest["notes"].extend(
        [
            "export_result=excluded",
            f"failure_stage={stage}",
            f"failure_type={type(error).__name__}",
            f"failure_detail={detail}",
            "artifact_identity=not_published",
        ]
    )
    write_manifest(path, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--platform", help="platform identity for this exact export")
    parser.add_argument("--verify-provenance-only", action="store_true")
    parser.add_argument("--update-manifest", action="store_true")
    parser.add_argument(
        "--record-failure",
        action="store_true",
        help="write a typed excluded manifest when export or validation fails",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    stage = "provenance"
    try:
        verify_provenance(manifest, args.upstream.resolve(), args.checkpoint.resolve())
        print("provenance: pinned source commit and checkpoint SHA256 verified")
        if args.verify_provenance_only:
            return 0

        output = (
            args.output
            or (args.manifest.parent / manifest["export"]["artifact"])
        ).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = "model_load"
        model = load_model(
            manifest, args.upstream.resolve(), args.checkpoint.resolve(), args.device
        )
        with tempfile.NamedTemporaryFile(
            prefix=output.name + ".candidate.",
            suffix=".onnx",
            dir=output.parent,
            delete=False,
        ) as stream:
            candidate = Path(stream.name)
        candidate.unlink()
        try:
            stage = "onnx_export"
            export_onnx(model, manifest, candidate, args.device)
            stage = "cpu_runtime_validation"
            observed = validate_export(model, manifest, candidate, args.device)
            publish_file(candidate, output)
        finally:
            candidate.unlink(missing_ok=True)
        print(f"artifact: {output}")
        print(f"sha256:   {sha256_file(output)}")
        print(f"validation: {json.dumps(observed, sort_keys=True)}")
        if args.update_manifest:
            update_manifest(
                args.manifest,
                manifest,
                output,
                observed,
                _platform_id(args.platform),
            )
            print(f"manifest updated: {args.manifest}")
        return 0
    except Exception as exc:  # record the exact stage and preserve nonzero CLI status
        if args.record_failure:
            try:
                record_failure(args.manifest, manifest, stage, exc)
                print(f"failure recorded: stage={stage}", file=sys.stderr)
            except Exception as record_error:
                print(f"could not record failure: {record_error}", file=sys.stderr)
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"export_neuflow_v2.py: error: {exc}", file=sys.stderr)
        raise SystemExit(1)
