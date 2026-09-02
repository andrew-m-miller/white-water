#!/usr/bin/env python3
"""Focused unit tests for the dependency-injected P25-5 evaluator boundary."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import copy
import json
from pathlib import Path
import shutil
import struct
import tempfile
from typing import Any

from . import evaluator as evaluator_module
from .evaluator import (
    Evaluator,
    EvaluatorFailure,
    _dtype_token,
    _resize_bilinear,
    _run_cli,
    _write_json_result,
    condition_and_pad_pair,
    frame_from_pfm,
    validate_manifest_artifact,
    validate_session_contract,
)
from .validator import load_json


ROOT = Path(__file__).resolve().parents[2]
POSITIVE_MANIFEST = ROOT / "models/fixtures/positive/artifact-v1.json"
POSITIVE_ARTIFACT = ROOT / "models/fixtures/positive/valid.bin"
V2_PROTOCOL = ROOT / "bakeoff/protocol-v2.json"


@dataclass
class _Meta:
    name: str
    type: str
    shape: list[Any]


@dataclass
class _FakeArray:
    data: Any
    shape: tuple[int, ...]
    dtype: str = "float32"

    def tolist(self) -> Any:
        return self.data


class _Finite:
    def __init__(self, value: bool):
        self.value = value

    def all(self) -> bool:
        return self.value


class _FakeArrays:
    float32 = "float32"

    def asarray(self, value: Any, dtype: Any = None) -> _FakeArray:
        return _FakeArray(value, _shape(value))

    def ascontiguousarray(self, value: Any) -> Any:
        return value

    def isfinite(self, value: _FakeArray) -> _Finite:
        return _Finite(all(_finite(float(item)) for item in _flatten(value.data)))


def _finite(value: float) -> bool:
    return value == value and abs(value) != float("inf")


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        result: list[Any] = []
        for child in value:
            result.extend(_flatten(child))
        return result
    return [value]


def _shape(value: Any) -> tuple[int, ...]:
    shape: list[int] = []
    current = value
    while isinstance(current, (list, tuple)):
        shape.append(len(current))
        current = current[0] if current else []
    return tuple(shape)


def _zeros_flow(height: int, width: int, value: float = 0.0) -> _FakeArray:
    data = [[[[value for _ in range(width)] for _ in range(height)] for _ in range(2)]]
    return _FakeArray(data, (1, 2, height, width))


class _FakeSession:
    def __init__(self, selected: list[str], *, bad_output_name: bool = False, nonfinite: bool = False):
        self._selected = selected
        self._bad_output_name = bad_output_name
        self._nonfinite = nonfinite
        self.calls = 0

    def get_providers(self) -> list[str]:
        return list(self._selected)

    def get_inputs(self) -> list[_Meta]:
        return [
            _Meta("image1", "tensor(float)", [1, 3, "height", "width"]),
            _Meta("image2", "tensor(float)", [1, 3, "height", "width"]),
        ]

    def get_outputs(self) -> list[_Meta]:
        return [_Meta("wrong" if self._bad_output_name else "flow", "tensor(float)", [1, 2, "height", "width"])]

    def run(self, names: list[str], feeds: dict[str, Any]) -> list[_FakeArray]:
        self.calls += 1
        input_value = next(iter(feeds.values()))
        _, _, height, width = input_value.shape
        return [_zeros_flow(height, width, float("nan") if self._nonfinite else 0.0)]


_UNSET = object()


class _FakeRuntime:
    __version__ = "fake-ort-1"

    def __init__(self, selected: list[str] | None = None, *, bad_output_name: bool = False, nonfinite: bool = False):
        self.selected = selected or ["CPUExecutionProvider"]
        self.bad_output_name = bad_output_name
        self.nonfinite = nonfinite
        self.requested: list[list[str]] = []
        # One entry per InferenceSession call: the provider_options kwarg it received, or _UNSET
        # when the evaluator omitted the kwarg entirely (the historical unbounded path).
        self.requested_options: list[Any] = []

    def get_available_providers(self) -> list[str]:
        return ["CPUExecutionProvider", "CUDAExecutionProvider"]

    def InferenceSession(self, path: str, *, providers: list[str], provider_options: Any = _UNSET) -> _FakeSession:
        self.requested.append(list(providers))
        self.requested_options.append(provider_options)
        return _FakeSession(self.selected, bad_output_name=self.bad_output_name, nonfinite=self.nonfinite)


class _FailingSession(_FakeSession):
    def __init__(self, selected: list[str], error: BaseException):
        super().__init__(selected)
        self.error = error

    def run(self, names: list[str], feeds: dict[str, Any]) -> list[_FakeArray]:
        if self.calls == 0:
            self.calls += 1
            raise self.error
        return super().run(names, feeds)


class _FailingRuntime(_FakeRuntime):
    def __init__(self, *, creation_error: BaseException | None = None, inference_error: BaseException | None = None):
        super().__init__()
        self.creation_error = creation_error
        self.inference_error = inference_error

    def InferenceSession(self, path: str, *, providers: list[str]) -> _FakeSession:
        self.requested.append(list(providers))
        if self.creation_error is not None:
            raise self.creation_error
        if self.inference_error is not None:
            return _FailingSession(self.selected, self.inference_error)
        return super().InferenceSession(path, providers=providers)


class _Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.001
        return self.value


def _artifact_fixture(directory: Path) -> tuple[Path, Path]:
    manifest = directory / "manifest.json"
    artifact = directory / "valid.bin"
    shutil.copy2(POSITIVE_MANIFEST, manifest)
    shutil.copy2(POSITIVE_ARTIFACT, artifact)
    manifest.chmod(0o644)
    artifact.chmod(0o644)
    return manifest, artifact


def _frame(width: int = 2, height: int = 2, value: float = 0.25, frame: int = 0) -> dict[str, Any]:
    rows = tuple(tuple((value, value + 0.1, value + 0.2) for _ in range(width)) for _ in range(height))
    return {"width": width, "height": height, "channels": 3, "rows": rows, "pixel_aspect_ratio": 1.0, "frame": frame, "sha256": "a" * 64}


def _evaluator(directory: Path, runtime: _FakeRuntime | None = None) -> Evaluator:
    manifest, artifact = _artifact_fixture(directory)
    validated = validate_manifest_artifact(manifest, artifact, protocol_path=V2_PROTOCOL)
    return Evaluator(validated, runtime or _FakeRuntime(), _FakeArrays(), clock=_Clock())


def _expect(kind: str, function: Any) -> None:
    try:
        function()
    except EvaluatorFailure as failure:
        assert failure.kind == kind, (failure.kind, kind)
    else:
        raise AssertionError(f"expected {kind} failure")


def _test_artifact_identity() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-evaluator-artifact-") as temporary:
        manifest, artifact = _artifact_fixture(Path(temporary))
        validated = validate_manifest_artifact(manifest, artifact, protocol_path=V2_PROTOCOL)
        assert validated.artifact_size_bytes == artifact.stat().st_size
        assert validated.artifact_sha256
        artifact.write_bytes(b"tampered")
        artifact.chmod(0o644)
        _expect("artifact_hash_mismatch", lambda: validate_manifest_artifact(manifest, artifact, protocol_path=V2_PROTOCOL))


def _test_provider_and_contract() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-evaluator-provider-") as temporary:
        evaluator = _evaluator(Path(temporary))
        verified = evaluator.verify("cpu")
        assert verified["requested_provider"] == "CPUExecutionProvider"
        assert evaluator.runtime.requested == [["CPUExecutionProvider"]]

        cuda = _evaluator(Path(temporary), _FakeRuntime(["CUDAExecutionProvider", "CPUExecutionProvider"]))
        cuda_verified = cuda.verify("cuda")
        assert cuda_verified["requested_provider"] == "CUDAExecutionProvider"
        assert cuda_verified["selected_providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
        assert cuda.runtime.requested == [["CUDAExecutionProvider"]]

        cpu_first = _evaluator(Path(temporary), _FakeRuntime(["CPUExecutionProvider", "CUDAExecutionProvider"]))
        _expect("provider_unavailable", lambda: cpu_first.verify("cuda"))

        wrong_priority = _evaluator(Path(temporary), _FakeRuntime(["AzureExecutionProvider", "CPUExecutionProvider"]))
        _expect("provider_unavailable", lambda: wrong_priority.verify("cpu"))

        bad_io = _evaluator(Path(temporary), _FakeRuntime(bad_output_name=True))
        _expect("unsupported_tensor_contract", lambda: bad_io.verify("cpu"))


def _test_gpu_mem_limit_threads_to_provider_options() -> None:
    # P25-7: a configured CUDA arena ceiling reaches the runtime in onnxruntime's official
    # list-aligned shape -- providers with a trailing CPU fallback and a positionally-aligned
    # provider_options list carrying the bound in bytes plus an empty dict for the CPU fallback --
    # while the user-facing MiB selection is surfaced back on the session contract / verify record.
    with tempfile.TemporaryDirectory(prefix="whitewater-evaluator-arena-") as temporary:
        manifest, artifact = _artifact_fixture(Path(temporary))
        validated = validate_manifest_artifact(manifest, artifact, protocol_path=V2_PROTOCOL)
        runtime = _FakeRuntime(["CUDAExecutionProvider", "CPUExecutionProvider"])
        evaluator = Evaluator(
            validated, runtime, _FakeArrays(), clock=_Clock(),
            provider_options={"cuda": {"gpu_mem_limit_mib": 22000}},
        )
        verified = evaluator.verify("cuda")
        assert runtime.requested == [["CUDAExecutionProvider", "CPUExecutionProvider"]], runtime.requested
        assert runtime.requested_options == [[
            {"gpu_mem_limit": 22000 * 1024 * 1024, "arena_extend_strategy": "kSameAsRequested"},
            {},
        ]], runtime.requested_options
        assert verified["provider_options"] == {"gpu_mem_limit_mib": 22000}, verified["provider_options"]


def _test_default_open_omits_provider_options() -> None:
    # A cell that requested no ceiling must call InferenceSession exactly as before: no
    # provider_options kwarg, and an empty provider_options record.
    with tempfile.TemporaryDirectory(prefix="whitewater-evaluator-arena-default-") as temporary:
        evaluator = _evaluator(Path(temporary))
        verified = evaluator.verify("cpu")
        assert evaluator.runtime.requested_options == [_UNSET], evaluator.runtime.requested_options
        assert verified["provider_options"] == {}, verified["provider_options"]

        # A provider_options map that has no entry for the opened provider is also the default path.
        with tempfile.TemporaryDirectory(prefix="whitewater-evaluator-arena-other-") as other:
            manifest, artifact = _artifact_fixture(Path(other))
            validated = validate_manifest_artifact(manifest, artifact, protocol_path=V2_PROTOCOL)
            runtime = _FakeRuntime()
            scoped = Evaluator(
                validated, runtime, _FakeArrays(), clock=_Clock(),
                provider_options={"cuda": {"gpu_mem_limit_mib": 4096}},
            )
            scoped.verify("cpu")
            assert runtime.requested_options == [_UNSET], runtime.requested_options


def _test_provider_options_normalization_rejects_bad_selections() -> None:
    # The evaluator normalizes the user-facing selection, so a misplaced or malformed arena
    # ceiling fails as a typed evaluator failure rather than reaching the runtime.
    with tempfile.TemporaryDirectory(prefix="whitewater-evaluator-arena-bad-") as temporary:
        manifest, artifact = _artifact_fixture(Path(temporary))
        validated = validate_manifest_artifact(manifest, artifact, protocol_path=V2_PROTOCOL)

        # A ceiling on a non-CUDA provider is rejected.
        cpu_bound = Evaluator(
            validated, _FakeRuntime(), _FakeArrays(), clock=_Clock(),
            provider_options={"cpu": {"gpu_mem_limit_mib": 4096}},
        )
        _expect("provider_unavailable", lambda: cpu_bound.verify("cpu"))

        # An unknown option key is rejected.
        cuda_runtime = _FakeRuntime(["CUDAExecutionProvider", "CPUExecutionProvider"])
        unknown_option = Evaluator(
            validated, cuda_runtime, _FakeArrays(), clock=_Clock(),
            provider_options={"cuda": {"gpu_mem_limit_mib": 4096, "unexpected": 1}},
        )
        _expect("provider_unavailable", lambda: unknown_option.verify("cuda"))

        # A non-positive / non-integer ceiling is rejected.
        for bad in (0, -1, 4096.0, True):
            bad_value = Evaluator(
                validated, _FakeRuntime(["CUDAExecutionProvider", "CPUExecutionProvider"]),
                _FakeArrays(), clock=_Clock(),
                provider_options={"cuda": {"gpu_mem_limit_mib": bad}},
            )
            _expect("provider_unavailable", lambda evaluator=bad_value: evaluator.verify("cuda"))


def _test_numpy_float32_dtype_token() -> None:
    assert _dtype_token("<class 'numpy.float32'>") == "float32"


def _test_resize_geometry() -> None:
    rows = (((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),)
    resized = _resize_bilinear(rows, 4, 1)
    assert tuple(pixel[0] for pixel in resized[0]) == (0.0, 0.25, 0.75, 1.0)


def _test_verify_does_not_load_numpy() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-evaluator-verify-imports-") as temporary:
        manifest, artifact = _artifact_fixture(Path(temporary))
        runtime = _FakeRuntime()
        calls = {"onnxruntime": 0, "numpy": 0}
        original_onnxruntime = evaluator_module._onnxruntime
        original_numpy_runtime = evaluator_module._numpy_runtime

        def fake_onnxruntime() -> _FakeRuntime:
            calls["onnxruntime"] += 1
            return runtime

        def unexpected_numpy_runtime() -> Any:
            calls["numpy"] += 1
            raise AssertionError("verify must not load NumPy")

        evaluator_module._onnxruntime = fake_onnxruntime
        evaluator_module._numpy_runtime = unexpected_numpy_runtime  # type: ignore[assignment]
        try:
            result = _run_cli(argparse.Namespace(
                command="verify",
                manifest=manifest,
                artifact=artifact,
                protocol=V2_PROTOCOL,
                platform=None,
                provider="cpu",
            ))
        finally:
            evaluator_module._onnxruntime = original_onnxruntime
            evaluator_module._numpy_runtime = original_numpy_runtime
        assert result["requested_provider"] == "CPUExecutionProvider"
        assert calls == {"onnxruntime": 1, "numpy": 0}


def _test_conditioning_padding_and_execution() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-evaluator-run-") as temporary:
        directory = Path(temporary)
        evaluator = _evaluator(directory)
        manifest = copy.deepcopy(evaluator.artifact.manifest)
        manifest["tensor_contract"]["padding"] = {"multiple": 8, "policy": "caller-replication-crop"}
        first, second, geometry, metadata = condition_and_pad_pair(
            _frame(frame=0), _frame(value=0.5, frame=1),
            conditioning_token="native-clamp01-v1", cap_megapixels=0.0, manifest=manifest,
        )
        assert geometry["analysis_width"] == 2 and geometry["analysis_height"] == 2
        assert geometry["padded_width"] == 8 and geometry["padded_height"] == 8
        assert metadata["padding_policy"] == "caller-replication-crop"
        first_array = _FakeArray((1, first), (1, 3, 8, 8))
        second_array = _FakeArray((1, second), (1, 3, 8, 8))
        cell = evaluator.run_nchw_pair(
            first_array, second_array, provider="cpu",
            analysis_width=2, analysis_height=2, padded_width=8, padded_height=8,
        )
        assert cell["flow"] == (((0.0, 0.0), (0.0, 0.0)), ((0.0, 0.0), (0.0, 0.0)))
        assert len(cell["timing"]["sessions"]) == 1
        assert len(cell["timing"]["sessions"][0]["steady_samples_ms"]) == 2
        assert cell["metrics"]["nonfinite_fraction"] == 0.0
        assert cell["metrics"]["repeated_run_p99_delta_px"] == 0.0
        assert cell["timing"]["preprocessing_ms"] == 0.0

        timed = evaluator.run_nchw_pair(
            first_array, second_array, provider="cpu",
            analysis_width=2, analysis_height=2, padded_width=8, padded_height=8,
            preprocessing_ms=12.5,
        )
        assert timed["timing"]["preprocessing_ms"] == 12.5


def _test_nonfinite_output() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-evaluator-nonfinite-") as temporary:
        evaluator = _evaluator(Path(temporary), _FakeRuntime(nonfinite=True))
        first = _FakeArray([[[[0.0] * 8 for _ in range(8)] for _ in range(3)]], (1, 3, 8, 8))
        _expect("nonfinite_output", lambda: evaluator.run_nchw_pair(
            first, first, provider="cpu", analysis_width=8, analysis_height=8, padded_width=8, padded_height=8,
        ))


def _run_pair(evaluator: Evaluator) -> dict[str, Any]:
    first = _FakeArray([[[[0.0] * 8 for _ in range(8)] for _ in range(3)]], (1, 3, 8, 8))
    return evaluator.run_nchw_pair(
        first, first, provider="cpu", analysis_width=8, analysis_height=8, padded_width=8, padded_height=8,
    )


def _expect_pair_failure(kind: str, message: str, evaluator: Evaluator, *, stage: str) -> None:
    try:
        _run_pair(evaluator)
    except EvaluatorFailure as failure:
        assert failure.kind == kind, (failure.kind, kind)
        assert message in failure.message, (failure.message, message)
        assert failure.stage == stage, (failure.stage, stage)
    else:
        raise AssertionError(f"expected {kind} failure")


def _test_oom_at_session_creation_is_typed() -> None:
    errors = (
        MemoryError("GPU memory exhausted"),
        RuntimeError("BFCArena"),
        RuntimeError("gpu_mem_limit"),
        RuntimeError("arena reports available memory below request"),
        RuntimeError("BFCArena: unable to allocate requested workspace"),
        RuntimeError("failed to allocate requested GPU workspace"),
    )
    for error in errors:
        with tempfile.TemporaryDirectory(prefix="whitewater-evaluator-oom-create-") as temporary:
            evaluator = _evaluator(Path(temporary), _FailingRuntime(creation_error=error))
            _expect_pair_failure("out_of_memory", str(error), evaluator, stage="session_create")


def _test_oom_at_inference_is_typed() -> None:
    errors = (
        MemoryError("GPU memory exhausted"),
        RuntimeError("BFCArena"),
        RuntimeError("gpu_mem_limit"),
        RuntimeError("arena reports available memory below request"),
        RuntimeError("BFCArena: unable to allocate requested workspace"),
        RuntimeError("failed to allocate requested GPU workspace"),
    )
    for error in errors:
        with tempfile.TemporaryDirectory(prefix="whitewater-evaluator-oom-inference-") as temporary:
            evaluator = _evaluator(Path(temporary), _FailingRuntime(inference_error=error))
            _expect_pair_failure("out_of_memory", str(error), evaluator, stage="inference")


def _test_generic_provider_and_runtime_errors_keep_existing_kinds() -> None:
    creation_errors = (
        RuntimeError("provider initialization failed"),
        RuntimeError("arena metadata unavailable"),
        RuntimeError("available memory query"),
        RuntimeError("GPU memory is available"),
        RuntimeError("gpu_mem"),
    )
    for error in creation_errors:
        with tempfile.TemporaryDirectory(prefix="whitewater-evaluator-generic-create-") as temporary:
            evaluator = _evaluator(Path(temporary), _FailingRuntime(creation_error=error))
            _expect_pair_failure("provider_unavailable", str(error), evaluator, stage="session_create")

    inference_errors = (
        RuntimeError("inference kernel failed"),
        RuntimeError("arena metadata unavailable"),
        RuntimeError("available memory query"),
        RuntimeError("GPU memory is available"),
        RuntimeError("gpu_mem"),
    )
    for error in inference_errors:
        with tempfile.TemporaryDirectory(prefix="whitewater-evaluator-generic-inference-") as temporary:
            evaluator = _evaluator(Path(temporary), _FailingRuntime(inference_error=error))
            _expect_pair_failure("runtime_error", str(error), evaluator, stage="inference")


def _test_pfm_adapter() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-evaluator-pfm-") as temporary:
        path = Path(temporary) / "frame.pfm"
        payload = struct.pack("<6f", 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        path.write_bytes(b"PF\n2 1\n-1.0\n" + payload)
        path.chmod(0o644)
        frame = frame_from_pfm(path, frame_number=7)
        assert frame["width"] == 2 and frame["height"] == 1 and frame["frame"] == 7
        assert frame["sha256"]


def _test_typed_preparation_and_output() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-evaluator-errors-") as temporary:
        evaluator = _evaluator(Path(temporary))
        manifest = copy.deepcopy(evaluator.artifact.manifest)
        _expect("conditioning_failure", lambda: condition_and_pad_pair(
            _frame(), _frame(frame=1), conditioning_token="not-a-token", cap_megapixels=0.0, manifest=manifest,
        ))
        _expect("input_invalid", lambda: condition_and_pad_pair(
            _frame(), _frame(frame=1), conditioning_token="native-clamp01-v1", cap_megapixels=-1.0, manifest=manifest,
        ))


def _test_atomic_json_output() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-evaluator-json-") as temporary:
        path = Path(temporary) / "result.json"
        _write_json_result(path, {"ok": True})
        assert path.read_text(encoding="utf-8") == '{\n  "ok": true\n}\n'
        assert (path.stat().st_mode & 0o777) == 0o644
        _expect("runtime_error", lambda: _write_json_result(path, {"second": True}))


def main() -> int:
    _test_artifact_identity()
    _test_provider_and_contract()
    _test_gpu_mem_limit_threads_to_provider_options()
    _test_default_open_omits_provider_options()
    _test_provider_options_normalization_rejects_bad_selections()
    _test_numpy_float32_dtype_token()
    _test_resize_geometry()
    _test_verify_does_not_load_numpy()
    _test_conditioning_padding_and_execution()
    _test_nonfinite_output()
    _test_oom_at_session_creation_is_typed()
    _test_oom_at_inference_is_typed()
    _test_generic_provider_and_runtime_errors_keep_existing_kinds()
    _test_pfm_adapter()
    _test_typed_preparation_and_output()
    _test_atomic_json_output()
    print("P25-5 evaluator tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
