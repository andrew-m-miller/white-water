#!/usr/bin/env python3
"""SPIKE test: torch-free NeuFlow CUDA qualifier logic, exercised with injected fake sessions.

No GPU, torch, or onnxruntime required -- fakes stand in for ONNX Runtime so the acceptance logic
(CUDA-first provider selection, CUDA-vs-CPU parity ceiling, finite/shape/direction gates) is
unit-testable on any host. Run: python3 models/qualify_neuflow_cuda_tests.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from artifact_workflow import ArtifactError, load_manifest  # type: ignore
from qualify_neuflow_cuda import CPU, CUDA, qualify_cuda  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = load_manifest(ROOT / "models" / "neuflow-v2.json")
H, W = MANIFEST["export"]["example_shape"][2:]
DX = int(MANIFEST["validation"]["translation_pixels"])


class _IO:
    def __init__(self, name: str, shape: list[int]) -> None:
        self.name, self.type, self.shape = name, "tensor(float)", shape


class FakeSession:
    """Returns canned [1,2,H,W] flow outputs in call order: identity, forward, reverse."""

    def __init__(self, providers: list[str], outputs: list[np.ndarray]) -> None:
        self._providers = providers
        self._outputs = outputs
        self._i = 0

    def get_providers(self) -> list[str]:
        return self._providers

    def get_inputs(self) -> list[_IO]:
        return [_IO("image1", [1, 3, H, W]), _IO("image2", [1, 3, H, W])]

    def get_outputs(self) -> list[_IO]:
        return [_IO("flow", [1, 2, H, W])]

    def run(self, _output_names, _feeds):
        out = self._outputs[self._i]
        self._i += 1
        return [out]


def _flow(dx_value: float) -> np.ndarray:
    out = np.zeros((1, 2, H, W), dtype=np.float32)
    out[:, 0, :, :] = dx_value
    return out


def _correct_outputs() -> list[np.ndarray]:
    # identity ~0, forward +dx in x, reverse -dx in x -> passes direction + identity gates.
    return [np.zeros((1, 2, H, W), dtype=np.float32), _flow(float(DX)), _flow(-float(DX))]


def _factory(cuda_session: FakeSession, cpu_session: FakeSession):
    def make(_path: str, providers):
        return cuda_session if list(providers)[0] == CUDA else cpu_session
    return make


AVAILABLE = [CUDA, CPU]


def _expect(fn, needle: str) -> None:
    # The reused export_neuflow_v2.require() raises RuntimeError; load_manifest raises ArtifactError.
    try:
        fn()
    except (RuntimeError, ArtifactError) as exc:
        assert needle in str(exc), (needle, str(exc))
    else:
        raise AssertionError(f"expected a rejection containing {needle!r}")


def main() -> int:
    # 1. Happy path: CUDA first-selected, CUDA == CPU outputs, correct direction -> passes.
    cuda = FakeSession([CUDA, CPU], _correct_outputs())
    cpu = FakeSession([CPU], _correct_outputs())
    evidence = qualify_cuda(
        MANIFEST, Path("neuflow.onnx"),
        session_factory=_factory(cuda, cpu), available_providers=AVAILABLE,
        onnxruntime_version="1.29.0",
    )
    assert evidence["provider_validation"] == {"requested": CUDA, "available": AVAILABLE, "selected": [CUDA, CPU], "passed": True}
    assert evidence["cuda_vs_cpu_abs"]["max_abs"] == 0.0
    assert evidence["forward_median"][0] >= DX * MANIFEST["validation"]["translation_x_fraction_min"]
    assert evidence["numerical_status"] == "passed"

    # 2. Silent CPU fallback: CUDA session reports CPU first -> rejected (no qualification).
    fb = FakeSession([CPU, CUDA], _correct_outputs())
    _expect(
        lambda: qualify_cuda(MANIFEST, Path("n.onnx"), session_factory=_factory(fb, FakeSession([CPU], _correct_outputs())), available_providers=AVAILABLE),
        "did not select",
    )

    # 3. CUDA diverges from CPU beyond the tolerance -> rejected.
    diverged = _correct_outputs()
    diverged[1] = _flow(float(DX) + 5.0)  # forward differs from CPU baseline by 5px
    _expect(
        lambda: qualify_cuda(
            MANIFEST, Path("n.onnx"),
            session_factory=_factory(FakeSession([CUDA, CPU], diverged), FakeSession([CPU], _correct_outputs())),
            available_providers=AVAILABLE, cuda_vs_cpu_max_abs=1.0e-2,
        ),
        "CUDA vs CPU max abs",
    )

    # 4. Non-finite CUDA output -> rejected.
    nonfinite = _correct_outputs()
    nonfinite[0] = np.full((1, 2, H, W), np.nan, dtype=np.float32)
    _expect(
        lambda: qualify_cuda(
            MANIFEST, Path("n.onnx"),
            session_factory=_factory(FakeSession([CUDA, CPU], nonfinite), FakeSession([CPU], nonfinite)),
            available_providers=AVAILABLE,
        ),
        "non-finite",
    )

    # 5. Wrong flow direction (forward points -x) -> rejected.
    wrong = _correct_outputs()
    wrong[1] = _flow(-float(DX))
    _expect(
        lambda: qualify_cuda(
            MANIFEST, Path("n.onnx"),
            session_factory=_factory(FakeSession([CUDA, CPU], wrong), FakeSession([CPU], wrong)),
            available_providers=AVAILABLE,
        ),
        "wrong way",
    )

    # 6. CUDA provider unavailable at all -> rejected before any run.
    _expect(
        lambda: qualify_cuda(MANIFEST, Path("n.onnx"), session_factory=_factory(cuda, cpu), available_providers=[CPU]),
        "unavailable",
    )

    print("NeuFlow CUDA qualifier (spike) tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
