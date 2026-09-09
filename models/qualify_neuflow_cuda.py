#!/usr/bin/env python3
"""SPIKE: torch-free CUDA provider qualification for the NeuFlow v2 ONNX.

Path A of the "CUDA qualification for NeuFlow" question. The frozen protocol
(docs/phase2.5-protocol-v2.md) forbids scheduling a NeuFlow CUDA row until a returned report
candidate carries ``measurement_providers: ["cuda"]`` -- and provider *capability* is not
qualification evidence. The checked-in NeuFlow validation was CPU-only, so there is no CUDA
qualification yet.

The original models/export_neuflow_v2.py establishes provider qualification by exporting from
PyTorch and comparing torch (reference) vs ONNX (provider). That path needs a torch+ONNX-Runtime
CUDA runtime. This qualifier instead reuses the ALREADY-signed P25-6/P25-7 measurement runtime
(ONNX Runtime CUDA-12 + numpy, no torch): the CPU numerical parity was already frozen, so a CUDA
qualification only has to prove the exact same graph runs on the CUDA execution provider,
first-selects it (no silent CPU fallback), and produces output that (a) still passes the frozen
correctness gates and (b) agrees with the CPU execution provider within a tight tolerance. The
reference is therefore the CPU ONNX session, not torch.

This is a SPIKE: it is runnable and unit-tested (with injected fake sessions) but is NOT wired into
CI, and running it for real requires a CUDA box. It does NOT flip measurement_providers or status
-- that recording step is gated on a passing on-box run (see
bakeoff/p25-7/SPIKE-neuflow-cuda-qualification.md).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

try:
    from artifact_workflow import load_manifest  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - package import path
    from .artifact_workflow import load_manifest  # type: ignore

# Reuse the EXACT acceptance helpers the CPU validation used, so the CUDA qualification applies the
# same fixed-IO and no-silent-fallback rules rather than a parallel re-implementation.
try:
    from export_neuflow_v2 import require, _validate_provider_selection, _validate_fixed_io  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    from .export_neuflow_v2 import require, _validate_provider_selection, _validate_fixed_io  # type: ignore


CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"


def synthetic_pair_numpy(height: int, width: int, dx: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A torch-free deterministic translation pair mirroring export_neuflow_v2.synthetic_pair.

    A blurred uniform-random first frame and a copy shifted +dx in x. Exact byte-equality with the
    torch RNG is not required: this qualifier compares CUDA vs the CPU execution provider on the
    SAME array and checks the flow direction, so any deterministic real translation exercises it.
    """

    rng = np.random.default_rng(seed)
    first = (rng.random((1, 3, height, width), dtype=np.float32) * 255.0).astype(np.float32)
    # 5x5 box blur with edge padding (matches the avg_pool2d(kernel=5, stride=1, padding=2) smooth).
    padded = np.pad(first, ((0, 0), (0, 0), (2, 2), (2, 2)), mode="edge")
    blurred = np.zeros_like(first)
    for oy in range(5):
        for ox in range(5):
            blurred += padded[:, :, oy:oy + height, ox:ox + width]
    first = (blurred / 25.0).astype(np.float32)
    second = np.zeros_like(first)
    second[..., dx:] = first[..., :-dx]
    return np.ascontiguousarray(first), np.ascontiguousarray(second)


def _abs_stats(delta: np.ndarray) -> dict[str, float]:
    a = np.abs(delta).reshape(-1)
    return {
        "mean_abs": float(np.mean(a)),
        "p99_abs": float(np.percentile(a, 99.0)),
        "p999_abs": float(np.percentile(a, 99.9)),
        "max_abs": float(np.max(a)),
    }


def qualify_cuda(
    manifest: Mapping[str, Any],
    onnx_path: Path,
    *,
    provider: str = CUDA,
    cuda_vs_cpu_max_abs: float = 1.0e-2,
    session_factory: Callable[[str, Sequence[str]], Any] | None = None,
    available_providers: Sequence[str] | None = None,
    onnxruntime_version: str = "unknown",
) -> dict[str, Any]:
    """Return a CUDA ``provider_validation``+evidence block, or raise on any qualification failure.

    ``session_factory(onnx_path_str, providers)`` and ``available_providers`` are injectable so the
    logic is unit-testable without a GPU; in production they default to onnxruntime.
    """

    if session_factory is None or available_providers is None:  # pragma: no cover - needs CUDA box
        import onnxruntime as ort

        available_providers = list(ort.get_available_providers())
        onnxruntime_version = ort.__version__

        def session_factory(path: str, providers: Sequence[str]) -> Any:
            return ort.InferenceSession(path, providers=list(providers))

    require(provider == CUDA, f"this qualifier establishes CUDA only, not {provider!r}")
    _validate_provider_selection(provider, list(available_providers))
    _validate_provider_selection(CPU, list(available_providers))

    validation = manifest["validation"]
    _, _, height, width = manifest["export"]["example_shape"]
    dx = int(validation["translation_pixels"])
    first, second = synthetic_pair_numpy(height, width, dx, int(validation["seed"]))

    input_names = [item["name"] for item in manifest["tensor_contract"]["inputs"]]
    output_name = manifest["tensor_contract"]["output"]["name"]

    def run(session: Any, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return session.run([output_name], {input_names[0]: a, input_names[1]: b})[0]

    # CUDA under test; assert it is actually first-selected (no silent CPU fallback).
    cuda_session = session_factory(str(onnx_path), [provider])
    selected = list(cuda_session.get_providers())
    _validate_provider_selection(provider, list(available_providers), selected)
    advertised_io = _validate_fixed_io(cuda_session, manifest)

    # CPU baseline is the reference (parity vs torch was already frozen on CPU).
    cpu_session = session_factory(str(onnx_path), [CPU])

    identity_cuda = run(cuda_session, first, first)
    forward_cuda = run(cuda_session, first, second)
    reverse_cuda = run(cuda_session, second, first)
    identity_cpu = run(cpu_session, first, first)
    forward_cpu = run(cpu_session, first, second)
    reverse_cpu = run(cpu_session, second, first)

    require(list(identity_cuda.shape) == [1, 2, height, width], f"unexpected CUDA output shape {list(identity_cuda.shape)}")
    require(identity_cuda.dtype == np.float32, f"unexpected CUDA output dtype {identity_cuda.dtype}")
    for name, arr in (("identity", identity_cuda), ("forward", forward_cuda), ("reverse", reverse_cuda)):
        require(bool(np.all(np.isfinite(arr))), f"CUDA {name} output contains a non-finite value")

    # (a) CUDA vs CPU agreement -- same graph, different EP: must be tight.
    delta = np.concatenate([
        (identity_cuda - identity_cpu).reshape(-1),
        (forward_cuda - forward_cpu).reshape(-1),
        (reverse_cuda - reverse_cpu).reshape(-1),
    ])
    parity = _abs_stats(delta)
    require(
        parity["max_abs"] <= cuda_vs_cpu_max_abs,
        f"CUDA vs CPU max abs {parity['max_abs']:.3e} exceeds tolerance {cuda_vs_cpu_max_abs:.3e}",
    )

    # (b) CUDA output must still pass the frozen correctness gates (direction + identity).
    border = max(16, dx * 4)
    interior = np.s_[0, :, border:-border, border:-border]
    identity_median = float(np.median(np.sqrt(np.sum(identity_cuda[interior] ** 2, axis=0))))
    forward_x = float(np.median(forward_cuda[0, 0, border:-border, border:-border]))
    forward_y = float(np.median(forward_cuda[0, 1, border:-border, border:-border]))
    reverse_x = float(np.median(reverse_cuda[0, 0, border:-border, border:-border]))
    reverse_y = float(np.median(reverse_cuda[0, 1, border:-border, border:-border]))
    minimum_x = abs(dx) * validation["translation_x_fraction_min"]
    require(identity_median <= validation["identity_median_epe_max"], "CUDA identity median EPE exceeds limit")
    require(forward_x >= minimum_x, "CUDA image1->image2 median dx points the wrong way or is too small")
    require(reverse_x <= -minimum_x, "CUDA image2->image1 median dx points the wrong way or is too small")
    require(abs(forward_y) <= validation["translation_abs_y_max"], "CUDA forward median dy exceeds limit")
    require(abs(reverse_y) <= validation["translation_abs_y_max"], "CUDA reverse median dy exceeds limit")

    return {
        "environment": {
            "onnxruntime": onnxruntime_version,
            "provider": provider,
            "available_providers": list(available_providers),
            "selected_providers": selected,
            "reference": "cpu-execution-provider (torch parity frozen on CPU; not re-run here)",
        },
        "provider_validation": {
            "requested": provider,
            "available": list(available_providers),
            "selected": selected,
            "passed": True,
        },
        "advertised_io": advertised_io,
        "cuda_vs_cpu_abs": parity,
        "cuda_vs_cpu_max_abs_tolerance": cuda_vs_cpu_max_abs,
        "identity_median_epe": identity_median,
        "forward_median": [forward_x, forward_y],
        "reverse_median": [reverse_x, reverse_y],
        "output_shape": list(identity_cuda.shape),
        "output_dtype": str(identity_cuda.dtype),
        "numerical_status": "passed",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("neuflow-v2.json"))
    parser.add_argument("--onnx", type=Path, required=True, help="the CPU-validated NeuFlow linux ONNX")
    parser.add_argument("--cuda-vs-cpu-max-abs", type=float, default=1.0e-2)
    parser.add_argument(
        "--record",
        type=Path,
        help="optional: write the CUDA evidence into this manifest's "
        "validation.observed.cuda_provider_qualification (does NOT flip measurement_providers/status)",
    )
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    evidence = qualify_cuda(manifest, args.onnx, cuda_vs_cpu_max_abs=args.cuda_vs_cpu_max_abs)
    print(json.dumps(evidence, indent=2))
    if args.record is not None:
        target = load_manifest(args.record)
        target["validation"].setdefault("observed", {})
        target["validation"]["observed"]["cuda_provider_qualification"] = evidence
        args.record.write_text(json.dumps(target, indent=2) + "\n", encoding="utf-8")
        print(f"recorded CUDA qualification evidence into {args.record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
