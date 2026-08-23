#!/usr/bin/env python3
"""Export and locally qualify the pinned original RAFT Things baseline.

The official RAFT demo pads both frames with replication padding to a multiple of eight before
calling the network and crops the result back to the caller's extent.  This exporter bakes the
official ``RAFT.forward(..., iters=12, test_mode=True)`` path only: its ONNX inputs are required
to be the already-padded, batch-1 RGB tensors described by ``raft-original.json``.  The caller
owns the fixed replication-pad/crop policy, so no data-dependent padding operation is hidden in
the graph.

The source checkout and checkpoint are supplied explicitly and are verified before importing
upstream code.  The exporter publishes an ONNX file only after graph checks, CPU provider shape
checks, PyTorch/ONNX parity, identity, and both signed translation directions pass.  The
checkpoint acquisition itself is handled by ``fetch_raft_checkpoint.py`` and is never implicit.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

try:
    from artifact_workflow import (  # type: ignore  # pylint: disable=wrong-import-position
        ArtifactError,
        environment_sha256,
        load_manifest,
        publish_file,
        require_regular_mode,
        sha256_file,
        update_platform_export,
        write_manifest,
    )
except ModuleNotFoundError:  # pragma: no cover - only used as models.export_raft
    from .artifact_workflow import (  # type: ignore
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
DEFAULT_MANIFEST = SCRIPT_DIR / "raft-original.json"
EXPECTED_SOURCE_COMMIT = "2888e15a51fa41140771d3f498ed8023cff098d1"
EXPECTED_CHECKPOINT_SHA256 = "fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1"
EXPECTED_CHECKPOINT_SIZE = 21108000
EXPECTED_ARCHIVE_SHA256 = "4be6101b271f58ec49866da5cf609fd17e86e9cae2483f70630ef4a295dc66bd"
EXPECTED_MEMBER = "models/raft-things.pth"


class ExportFailure(RuntimeError):
    """A failure annotated with the last completed exporter stage."""

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        source_verified: bool = False,
        checkpoint_verified: bool = False,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.source_verified = source_verified
        self.checkpoint_verified = checkpoint_verified


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def checked_out_commit(upstream: Path) -> str:
    """Return HEAD only when the source checkout has no tracked or untracked changes."""

    revision = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(upstream), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    require(
        not status.strip(),
        "RAFT source checkout must be clean (tracked or untracked changes present): "
        + status.strip(),
    )
    return revision


def verify_provenance(manifest: dict[str, Any], upstream: Path, checkpoint: Path) -> None:
    """Verify the exact official source checkout and extracted checkpoint bytes."""

    source_verified = False
    checkpoint_verified = False
    try:
        require(manifest["candidate"]["id"] == "raft-original", "manifest is not the original RAFT candidate")
        require(
            manifest["model"]["config"] == {
                "small": False,
                "iters": 12,
                "mixed_precision": False,
                "alternate_corr": False,
                "checkpoint_name": "raft-things.pth",
            },
            "original RAFT exporter requires the pinned full model with exactly 12 iterations",
        )
        require(
            manifest["tensor_contract"]["padding"] == {
                "multiple": 8,
                "policy": "caller-replication-crop",
            },
            "original RAFT exporter requires the declared caller replication-pad/crop policy",
        )
        require((upstream / ".git").exists(), f"not a git checkout: {upstream}")
        actual_commit = checked_out_commit(upstream)
        expected_commit = manifest["upstream"]["commit"]
        require(
            actual_commit == expected_commit == EXPECTED_SOURCE_COMMIT,
            f"RAFT checkout is {actual_commit}, expected {expected_commit}",
        )
        source_verified = True

        require_regular_mode(checkpoint, "RAFT checkpoint")
        expected_size = manifest["checkpoint"]["size_bytes"]
        require(
            checkpoint.stat().st_size == expected_size == EXPECTED_CHECKPOINT_SIZE,
            f"checkpoint is {checkpoint.stat().st_size} bytes, expected {expected_size}",
        )
        actual_hash = sha256_file(checkpoint)
        expected_hash = manifest["checkpoint"]["sha256"]
        require(
            actual_hash == expected_hash == EXPECTED_CHECKPOINT_SHA256,
            f"checkpoint SHA256 is {actual_hash}, expected {expected_hash}",
        )
        checkpoint_verified = True
    except ExportFailure:
        raise
    except (ArtifactError, RuntimeError, subprocess.CalledProcessError, OSError) as exc:
        raise ExportFailure(
            str(exc),
            stage="provenance",
            source_verified=source_verified,
            checkpoint_verified=checkpoint_verified,
        ) from exc


def run_stage(
    stage: str,
    callback,
    *,
    source_verified: bool,
    checkpoint_verified: bool,
):
    """Annotate expected stage failures without inventing provenance claims."""

    try:
        return callback()
    except ExportFailure:
        raise
    except (ArtifactError, RuntimeError, subprocess.CalledProcessError, OSError) as exc:
        raise ExportFailure(
            str(exc),
            stage=stage,
            source_verified=source_verified,
            checkpoint_verified=checkpoint_verified,
        ) from exc


def load_model(manifest: dict[str, Any], upstream: Path, checkpoint: Path, device: str):
    """Construct the official full RAFT model and strictly load every checkpoint key."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised on export hosts
        raise RuntimeError(
            "missing export dependency; install models/requirements-raft-export.txt"
        ) from exc

    # The official source uses imports relative to its core/ directory.  Scope this path change
    # to construction so importing this exporter never mutates the caller's sys.path.
    sys.path.insert(0, str(upstream / "core"))
    try:
        from raft import RAFT
    finally:
        sys.path.pop(0)

    config = argparse.Namespace(**manifest["model"]["config"])
    model = RAFT(config)
    state = torch.load(str(checkpoint), map_location="cpu")
    require(isinstance(state, dict), "RAFT checkpoint is not a state-dict mapping")
    normalized: dict[str, Any] = {}
    for key, value in state.items():
        require(isinstance(key, str), "RAFT checkpoint contains a non-string state-dict key")
        require(key.startswith("module."), f"RAFT checkpoint key lacks module. prefix: {key}")
        stripped = key[len("module.") :]
        require(stripped not in normalized, f"duplicate normalized RAFT checkpoint key: {stripped}")
        normalized[stripped] = value
    missing, unexpected = model.load_state_dict(normalized, strict=False)
    require(not missing, f"checkpoint is missing model keys: {missing}")
    require(not unexpected, f"checkpoint has unexpected model keys: {unexpected}")
    model.eval().to(torch.device(device))
    return model


def make_wrapper(model, iterations: int):
    """Return the fixed-iteration, padded-input wrapper exported to ONNX."""

    import torch

    class RAFTExportWrapper(torch.nn.Module):
        def __init__(self, wrapped, baked_iterations: int):
            super().__init__()
            self.wrapped = wrapped
            self.baked_iterations = baked_iterations

        def forward(self, image1, image2):
            # Padding is deliberately caller-side and fixed by the manifest's
            # caller-replication-crop policy.  The official model's forward path is otherwise
            # unchanged, including normalization and final flow direction.
            return self.wrapped(
                image1,
                image2,
                iters=self.baked_iterations,
                test_mode=True,
            )[1]

    return RAFTExportWrapper(model, iterations)


def _check_graph(exported, manifest: dict[str, Any]) -> None:
    import onnx

    onnx.checker.check_model(exported, full_check=True)
    expected_opset = manifest["export"]["opset"]
    opsets = {item.domain or "ai.onnx": item.version for item in exported.opset_import}
    require(opsets.get("ai.onnx") == expected_opset, f"graph opset is {opsets}, expected {expected_opset}")
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
    require(
        len(exported.graph.input) == 2 and len(exported.graph.output) == 1,
        "RAFT graph must expose exactly two inputs and one flow output",
    )


def export_onnx(wrapper, manifest: dict[str, Any], output: Path, device: str) -> None:
    import onnx
    import torch

    shape = manifest["export"]["example_shape"]
    require(shape[:2] == [1, 3], f"unsupported example shape: {shape}")
    require(shape[2] % 8 == 0 and shape[3] % 8 == 0, "example H and W must be multiples of 8")
    sample1 = torch.zeros(shape, dtype=torch.float32, device=device)
    sample2 = torch.zeros(shape, dtype=torch.float32, device=device)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=output.name + ".", suffix=".candidate", dir=output.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        torch.onnx.export(
            wrapper,
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
        exported = onnx.load(str(temporary), load_external_data=False)
        _check_graph(exported, manifest)
        publish_file(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def synthetic_pair(torch, height: int, width: int, dx: int, seed: int, device: str):
    require(height % 8 == 0 and width % 8 == 0, "synthetic pair requires dimensions divisible by 8")
    generator = torch.Generator(device=device).manual_seed(seed)
    first = torch.rand((1, 3, height, width), generator=generator, device=device) * 255.0
    first = torch.nn.functional.avg_pool2d(first, kernel_size=5, stride=1, padding=2)
    second = torch.zeros_like(first)
    second[..., dx:] = first[..., :-dx]
    return first, second


def _metadata_shape(value: Any) -> list[Any]:
    return list(value.shape)


def _validate_advertised_io(session, manifest: dict[str, Any]) -> dict[str, Any]:
    """Check names, float32 types, NCHW channels, and dynamic spatial dimensions."""

    inputs = session.get_inputs()
    outputs = session.get_outputs()
    require([item.name for item in inputs] == ["image1", "image2"], "ONNX input names changed")
    require([item.name for item in outputs] == ["flow"], "ONNX output name changed")
    for item in inputs:
        require(item.type == "tensor(float)", f"{item.name} is advertised as {item.type}, expected float32")
        shape = _metadata_shape(item)
        require(
            len(shape) == 4 and type(shape[1]) is int and shape[1] == 3,
            f"{item.name} is not NCHW with three channels: {shape}",
        )
        require(isinstance(shape[2], str) and isinstance(shape[3], str), f"{item.name} spatial axes are not dynamic: {shape}")
    output = outputs[0]
    require(output.type == "tensor(float)", f"flow is advertised as {output.type}, expected float32")
    output_shape = _metadata_shape(output)
    require(
        len(output_shape) == 4 and type(output_shape[1]) is int and output_shape[1] == 2,
        f"flow is not NCHW with two channels: {output_shape}",
    )
    require(isinstance(output_shape[2], str) and isinstance(output_shape[3], str), f"flow spatial axes are not dynamic: {output_shape}")
    return {
        "inputs": [{"name": item.name, "dtype": item.type, "shape": _metadata_shape(item)} for item in inputs],
        "outputs": [{"name": output.name, "dtype": output.type, "shape": output_shape}],
    }


def validate_export(wrapper, manifest: dict[str, Any], output: Path, device: str, provider: str) -> dict[str, Any]:
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    available = ort.get_available_providers()
    require(provider in available, f"requested ONNX Runtime provider is unavailable: {provider}; available={available}")
    session = ort.InferenceSession(str(output), providers=[provider])
    advertised_io = _validate_advertised_io(session, manifest)
    selected_providers = session.get_providers()
    require(provider in selected_providers, f"ONNX Runtime did not select {provider}: {selected_providers}")

    validation = manifest["validation"]
    _, _, height, width = manifest["export"]["example_shape"]
    dx = int(validation["translation_pixels"])
    first, second = synthetic_pair(torch, height, width, dx, validation["seed"], device)
    with torch.no_grad():
        identity_pt = wrapper(first, first).cpu().numpy()
        forward_pt = wrapper(first, second).cpu().numpy()
        reverse_pt = wrapper(second, first).cpu().numpy()

    input_names = [item["name"] for item in manifest["tensor_contract"]["inputs"]]
    output_name = manifest["tensor_contract"]["output"]["name"]

    def run_onnx(a, b):
        values = session.run(
            [output_name],
            {
                input_names[0]: np.ascontiguousarray(a.detach().cpu().numpy()),
                input_names[1]: np.ascontiguousarray(b.detach().cpu().numpy()),
            },
        )
        result = values[0]
        require(result.dtype == np.float32, f"ONNX flow dtype is {result.dtype}, expected float32")
        require(
            list(result.shape) == [1, 2, int(a.shape[2]), int(a.shape[3])],
            f"ONNX flow runtime shape is {list(result.shape)}, expected [1, 2, H, W]",
        )
        require(np.all(np.isfinite(result)), "ONNX flow contains a non-finite value")
        return result

    identity_onnx = run_onnx(first, first)
    forward_onnx = run_onnx(first, second)
    reverse_onnx = run_onnx(second, first)

    second_shape = validation["second_dynamic_shape"]
    require(second_shape[:2] == [1, 3], f"unsupported second dynamic shape: {second_shape}")
    dynamic_first, dynamic_second = synthetic_pair(
        torch,
        second_shape[2],
        second_shape[3],
        dx,
        validation["seed"] + 1,
        device,
    )
    with torch.no_grad():
        dynamic_pt = wrapper(dynamic_first, dynamic_second).cpu().numpy()
    dynamic_flow = run_onnx(dynamic_first, dynamic_second)
    require(
        list(dynamic_flow.shape) == [1, 2, second_shape[2], second_shape[3]],
        f"dynamic-shape run returned {list(dynamic_flow.shape)}",
    )
    for name, value in (
        ("identity", identity_pt),
        ("forward", forward_pt),
        ("reverse", reverse_pt),
        ("dynamic", dynamic_pt),
    ):
        require(np.all(np.isfinite(value)), f"PyTorch {name} flow contains a non-finite value")

    pairs = (
        (identity_pt, identity_onnx),
        (forward_pt, forward_onnx),
        (reverse_pt, reverse_onnx),
        (dynamic_pt, dynamic_flow),
    )
    absolute_differences = np.abs(
        np.concatenate([(pt - exported).reshape(-1) for pt, exported in pairs])
    )
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
    failures: list[str] = []
    limits = (
        (mean_abs, validation["onnx_pytorch_mean_abs_max"], "ONNX/PyTorch mean absolute error"),
        (p99_abs, validation["onnx_pytorch_p99_abs_max"], "ONNX/PyTorch p99 absolute error"),
        (p999_abs, validation["onnx_pytorch_p999_abs_max"], "ONNX/PyTorch p99.9 absolute error"),
        (max_abs, validation["onnx_pytorch_max_abs_max"], "ONNX/PyTorch max absolute error"),
        (identity_median, validation["identity_median_epe_max"], "identity median EPE"),
    )
    failures.extend(f"{label} exceeds limit" for value, limit, label in limits if value > limit)
    if forward_x < minimum_x:
        failures.append("image1->image2 median dx points the wrong way or is too small")
    if reverse_x > -minimum_x:
        failures.append("image2->image1 median dx points the wrong way or is too small")
    if abs(forward_y) > validation["translation_abs_y_max"]:
        failures.append("forward median dy exceeds limit")
    if abs(reverse_y) > validation["translation_abs_y_max"]:
        failures.append("reverse median dy exceeds limit")

    graph = onnx.load(str(output), load_external_data=False)
    graph_domains = sorted({node.domain or "ai.onnx" for node in graph.graph.node})
    observed = {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "provider": provider,
            "available_providers": available,
            "selected_providers": selected_providers,
        },
        "provider_validation": {
            "requested": provider,
            "available": available,
            "selected": selected_providers,
            "passed": True,
        },
        "advertised_io": advertised_io,
        "graph_nodes": len(graph.graph.node),
        "graph_domains": graph_domains,
        "onnx_pytorch_mean_abs": mean_abs,
        "onnx_pytorch_p99_abs": p99_abs,
        "onnx_pytorch_p999_abs": p999_abs,
        "onnx_pytorch_max_abs": max_abs,
        "identity_median_epe": identity_median,
        "forward_median": [forward_x, forward_y],
        "reverse_median": [reverse_x, reverse_y],
        "example_shape": [1, 2, height, width],
        "second_dynamic_shape": list(dynamic_flow.shape),
    }
    require(not failures, "; ".join(failures) + "; observed=" + json.dumps(observed, sort_keys=True))
    return observed


def _platform_id(value: str | None) -> str:
    if value:
        return value
    if sys.platform == "darwin":
        return "macos-arm64" if platform.machine().lower() in {"arm64", "aarch64"} else "macos-x86_64"
    if sys.platform.startswith("linux"):
        return f"linux-{platform.machine().lower()}"
    return f"{sys.platform}-{platform.machine().lower()}"


def _record_generic_validation(
    manifest: dict[str, Any], observed: dict[str, Any], *, admission_pending: bool
) -> None:
    """Map numerical evidence into the shared schema without claiming admission."""

    validation = manifest["validation"]
    forward = observed["forward_median"]
    reverse = observed["reverse_median"]
    shape = manifest["export"]["example_shape"]
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
        "dynamic": True,
        "example": [1, 2, shape[2], shape[3]],
        "additional": list(observed["second_dynamic_shape"]),
    }
    validation["parity"] = {
        "checked": True,
        "mean_abs": observed["onnx_pytorch_mean_abs"],
        "p99_abs": observed["onnx_pytorch_p99_abs"],
        "p999_abs": observed["onnx_pytorch_p999_abs"],
        "max_abs": observed["onnx_pytorch_max_abs"],
    }
    # Numerical qualification and admission are separate decisions.  An artifact can pass all
    # numerical gates while remaining excluded because its checkpoint terms are unresolved.
    validation["status"] = "passed"
    validation["observed"] = {
        "numerical_gates": "passed",
        "source_commit_locally_verified": True,
        "checkpoint_locally_verified": True,
        "checkpoint_archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "checkpoint_member": EXPECTED_MEMBER,
        "checkpoint_terms": "unknown_from_official_primary_sources" if admission_pending else "unknown",
        "provider_validation": observed["provider_validation"],
        "advertised_io": observed["advertised_io"],
        "graph_nodes": observed["graph_nodes"],
        "graph_domains": observed["graph_domains"],
        "environment": observed["environment"],
        "metrics": {
            "identity_median_epe": observed["identity_median_epe"],
            "forward_median": observed["forward_median"],
            "reverse_median": observed["reverse_median"],
            "onnx_pytorch_mean_abs": observed["onnx_pytorch_mean_abs"],
            "onnx_pytorch_p99_abs": observed["onnx_pytorch_p99_abs"],
            "onnx_pytorch_p999_abs": observed["onnx_pytorch_p999_abs"],
            "onnx_pytorch_max_abs": observed["onnx_pytorch_max_abs"],
        },
    }


def update_manifest(
    path: Path,
    manifest: dict[str, Any],
    output: Path,
    observed: dict[str, Any],
    platform_id: str,
) -> None:
    # The baseline has unknown checkpoint commercial/redistribution terms. Record all numerical
    # evidence and exact bytes, but keep admission explicitly excluded.
    _record_generic_validation(manifest, observed, admission_pending=True)
    env_observed = observed["environment"]
    environment = {
        "platform": "macos" if platform_id.startswith("macos") else "linux",
        "architecture": "arm64" if platform_id.endswith("arm64") else "x86_64",
        "python": env_observed["python"],
        "framework": f"pytorch=={env_observed['pytorch']}",
        "exporter": f"torch.onnx=={env_observed['pytorch']}",
        "runtime": f"onnxruntime=={env_observed['onnxruntime']}",
        "provider": env_observed["provider"],
    }
    environment["sha256"] = environment_sha256(environment)
    update_platform_export(
        manifest,
        platform=platform_id,
        artifact=output,
        environment=environment,
    )
    manifest["status"] = "excluded"
    manifest["exclusion"] = {"reason_code": "checkpoint_license_terms_unknown"}
    manifest["notes"] = [
        note
        for note in manifest.get("notes", [])
        if not note.startswith("D1 intentionally records")
    ]
    manifest["notes"].append(
        "D2 exported the exact original-RAFT path with 12 iterations and passed local CPU graph, "
        "provider, identity, signed-direction, dynamic-shape and PyTorch/ONNX parity gates. "
        "The candidate remains excluded because official primary sources do not state checkpoint "
        "commercial-use or redistribution terms."
    )
    write_manifest(path, manifest)


def has_numerical_result(manifest: dict[str, Any]) -> bool:
    """Return true when a manifest already carries an exact, numerically checked artifact."""

    export = manifest.get("export", {})
    validation = manifest.get("validation", {})
    observed = validation.get("observed") or {}
    parity = validation.get("parity") or {}
    numerical_fields_present = all(
        key in validation for key in ("identity", "directions", "shapes", "parity")
    )
    return bool(
        export.get("sha256")
        and export.get("size_bytes")
        and (
            observed.get("numerical_gates") == "passed"
            or validation.get("status") == "passed"
            or numerical_fields_present
        )
        and parity.get("checked") is True
    )


def record_failure(
    path: Path,
    manifest: dict[str, Any],
    message: str,
    *,
    stage: str,
    source_verified: bool,
    checkpoint_verified: bool,
) -> None:
    """Record a typed exporter/operator failure without inventing verification claims."""

    if has_numerical_result(manifest):
        raise ArtifactError(
            "refusing to replace an existing numerically validated result and artifact identity "
            f"with a {stage} failure: {path}"
        )
    manifest["status"] = "excluded"
    manifest["exclusion"] = {"reason_code": "export_or_operator_failure"}
    manifest["validation"] = {
        "status": "failed",
        "observed": {
            "stage": stage,
            "failure": message,
            "source_commit_verified": source_verified,
            "checkpoint_verified": checkpoint_verified,
        },
    }
    manifest["notes"] = list(manifest.get("notes", [])) + [
        f"D2 exporter/operator failure recorded at {stage} without qualification: {message}"
    ]
    write_manifest(path, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path, help="pinned original-RAFT checkout")
    parser.add_argument("--checkpoint", required=True, type=Path, help="verified raft-things.pth member")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, help="output ONNX path; defaults beside the manifest")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--provider",
        choices=("CPUExecutionProvider", "CUDAExecutionProvider"),
        help="ONNX Runtime provider for local validation (defaults from --device)",
    )
    parser.add_argument("--platform", help="platform identity for this exact ONNX export")
    parser.add_argument(
        "--verify-provenance-only",
        action="store_true",
        help="check source commit and checkpoint hash without importing ML dependencies",
    )
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="record exact ONNX identity and measurements after successful validation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    upstream = args.upstream.resolve()
    checkpoint = args.checkpoint.absolute()
    verify_provenance(manifest, upstream, checkpoint)
    print("provenance: pinned source commit and checkpoint SHA256 verified")
    if args.verify_provenance_only:
        return 0

    provider = args.provider or (
        "CUDAExecutionProvider" if args.device == "cuda" else "CPUExecutionProvider"
    )
    output = (args.output or (args.manifest.parent / manifest["export"]["artifact"])).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    model = run_stage(
        "model_load",
        lambda: load_model(manifest, upstream, checkpoint, args.device),
        source_verified=True,
        checkpoint_verified=True,
    )
    wrapper = run_stage(
        "wrapper_setup",
        lambda: make_wrapper(model, manifest["model"]["config"]["iters"]),
        source_verified=True,
        checkpoint_verified=True,
    )
    with tempfile.NamedTemporaryFile(
        prefix=output.name + ".candidate.", suffix=".onnx", dir=output.parent, delete=False
    ) as stream:
        candidate = Path(stream.name)
    candidate.unlink()
    try:
        run_stage(
            "onnx_export",
            lambda: export_onnx(wrapper, manifest, candidate, args.device),
            source_verified=True,
            checkpoint_verified=True,
        )
        observed = run_stage(
            "operator_validation",
            lambda: validate_export(wrapper, manifest, candidate, args.device, provider),
            source_verified=True,
            checkpoint_verified=True,
        )
        run_stage(
            "artifact_publish",
            lambda: publish_file(candidate, output),
            source_verified=True,
            checkpoint_verified=True,
        )
    finally:
        candidate.unlink(missing_ok=True)
    artifact_hash = sha256_file(output)
    require_regular_mode(output, "exported RAFT artifact")
    print(f"artifact: {output}")
    print(f"sha256:   {artifact_hash}")
    print(f"validation: {json.dumps(observed, sort_keys=True)}")
    if args.update_manifest:
        update_manifest(args.manifest, manifest, output, observed, _platform_id(args.platform))
        print(f"manifest updated: {args.manifest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExportFailure as exc:
        # A failed run never publishes its candidate. With --update-manifest, retain a typed
        # exclusion record so an unsupported operator or provider is a visible result.
        try:
            parsed = parse_args()
            if parsed.update_manifest:
                failed_manifest = load_manifest(parsed.manifest)
                record_failure(
                    parsed.manifest,
                    failed_manifest,
                    str(exc),
                    stage=exc.stage,
                    source_verified=exc.source_verified,
                    checkpoint_verified=exc.checkpoint_verified,
                )
        except Exception:  # pragma: no cover - preserve the original export error
            pass
        print(f"export_raft.py: error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except (ArtifactError, RuntimeError, subprocess.CalledProcessError, OSError) as exc:
        # Unexpected failures do not carry provenance claims.  A validated manifest is still
        # protected by record_failure's identity guard if --update-manifest was requested.
        try:
            parsed = parse_args()
            if parsed.update_manifest:
                failed_manifest = load_manifest(parsed.manifest)
                record_failure(
                    parsed.manifest,
                    failed_manifest,
                    str(exc),
                    stage="unknown",
                    source_verified=False,
                    checkpoint_verified=False,
                )
        except Exception:  # pragma: no cover - preserve the original export error
            pass
        print(f"export_raft.py: error: {exc}", file=sys.stderr)
        raise SystemExit(1)
