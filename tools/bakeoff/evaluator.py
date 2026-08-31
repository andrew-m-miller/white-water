#!/usr/bin/env python3
"""Small, dependency-injected ONNX Runtime evaluator boundary.

The P25-4 runner owns matrix/session orchestration.  This module owns the narrower runtime
seam: validate one manifest and payload, request one exact execution provider, check the
advertised tensor contract, and execute one already-preconditioned NCHW pair.  It deliberately
does not download models, decide shipping eligibility, read EXR, or collect NVML data.

The public runtime/session/array arguments are injected so all contract tests run without
NumPy, ONNX Runtime, or a GPU.  The CLI is intentionally conservative: ``verify`` only checks
the local artifact/runtime, while ``smoke`` reads two strict PFM files and executes the local
model when the optional runtime dependencies are installed.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import contextlib
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
import time
from typing import Any, Callable, ContextManager, Protocol

try:
    from .conditioning import ConditioningFailure, condition_pair
    from .geometry import GeometryFailure, analysis_dimensions
    from .measurement import PROFILES, reduce_geometry, reduce_timing
    from .padding import PaddingPolicyError, pad_rows
    from .pfm import PfmFailure, read_pfm
    from .validator import canonical_sha256, load_json
except ImportError:  # pragma: no cover - direct invocation from an air-gapped checkout
    from conditioning import ConditioningFailure, condition_pair  # type: ignore
    from geometry import GeometryFailure, analysis_dimensions  # type: ignore
    from measurement import PROFILES, reduce_geometry, reduce_timing  # type: ignore
    from padding import PaddingPolicyError, pad_rows  # type: ignore
    from pfm import PfmFailure, read_pfm  # type: ignore
    from validator import canonical_sha256, load_json  # type: ignore


ROOT = Path(__file__).resolve().parents[2]
V2_PROTOCOL = ROOT / "bakeoff" / "protocol-v2.json"
PROVIDER_EXECUTION_NAMES = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
}
REPORT_METRICS = (
    "endpoint_error_px",
    "fraction_le_1px",
    "fraction_le_3px",
    "landmark_median_error_px",
    "landmark_p95_error_px",
    "visible_warp_residual",
    "forward_backward_residual_px",
    "chain_drift_px",
)


class EvaluatorFailure(ValueError):
    """Stable, typed failure at the evaluator boundary."""

    def __init__(self, kind: str, message: str):
        self.kind = kind
        self.reason = kind
        self.failure_type = "evaluator_failure"
        self.message = message
        super().__init__(f"{kind}: {message}")


class DependencyFailure(EvaluatorFailure):
    """An optional runtime dependency is not available."""


def _fail(kind: str, message: str) -> None:
    raise EvaluatorFailure(kind, message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise EvaluatorFailure("artifact_missing", f"cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def _require_regular_file(path: Path, label: str, *, mode: int | None = 0o644) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise EvaluatorFailure("artifact_missing", f"{label} does not exist: {path}") from exc
    except OSError as exc:
        raise EvaluatorFailure("artifact_missing", f"cannot inspect {label} {path}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode):
        _fail("artifact_missing", f"{label} must not be a symlink: {path}")
    if not stat.S_ISREG(info.st_mode):
        _fail("artifact_missing", f"{label} must be a regular file: {path}")
    if mode is not None and (info.st_mode & 0o777) != mode:
        _fail("artifact_missing", f"{label} mode must be {mode:04o}: {path}")


def _load_artifact_helpers() -> Any:
    """Load models.artifact_workflow without mutating sys.path for direct CLI use."""

    try:
        from models import artifact_workflow  # type: ignore

        return artifact_workflow
    except ModuleNotFoundError:
        path = ROOT / "models" / "artifact_workflow.py"
        spec = importlib.util.spec_from_file_location("_whitewater_evaluator_artifact_workflow", path)
        if spec is None or spec.loader is None:
            raise DependencyFailure("runtime_error", f"cannot load artifact workflow: {path}")
        module = importlib.util.module_from_spec(spec)
        # Direct ``python tools/bakeoff/evaluator.py`` has tools/bakeoff, rather than the
        # checkout root, as sys.path[0].  artifact_workflow's documented direct-script fallback
        # imports exclusion_contract by its short name, so provide that one directory only while
        # loading the private helper and remove it immediately afterwards.
        model_directory = str(ROOT / "models")
        inserted = model_directory not in sys.path
        if inserted:
            sys.path.insert(0, model_directory)
        try:
            spec.loader.exec_module(module)
        finally:
            if inserted:
                sys.path.remove(model_directory)
        return module


@dataclass(frozen=True)
class ValidatedArtifact:
    """One exact manifest/payload pair; no shipping decision is inferred."""

    manifest_path: Path
    artifact_path: Path
    manifest: dict[str, Any]
    platform: str
    manifest_sha256: str
    artifact_sha256: str
    artifact_size_bytes: int

    @property
    def candidate_id(self) -> str:
        return str(self.manifest["candidate"]["id"])


def validate_manifest_artifact(
    manifest_path: Path | str,
    artifact_path: Path | str | None = None,
    *,
    platform: str | None = None,
    protocol_path: Path | str = V2_PROTOCOL,
) -> ValidatedArtifact:
    """Validate one manifest and exact payload by hash, size, mode, and contract.

    ``artifact_path`` may point into a staging directory; the manifest's declared platform row
    remains authoritative.  This function never turns a manifest's shipping status into an
    eligibility decision.
    """

    manifest_destination = Path(manifest_path)
    _require_regular_file(manifest_destination, "manifest")
    helpers = _load_artifact_helpers()
    try:
        manifest = helpers.load_manifest(
            manifest_destination,
            protocol_path=Path(protocol_path),
        )
    except Exception as exc:
        if isinstance(exc, EvaluatorFailure):
            raise
        raise EvaluatorFailure("unsupported_tensor_contract", f"manifest validation failed: {exc}") from exc
    selected_platform = platform or manifest["export"]["platform"]
    selected = next(
        (entry for entry in manifest["export"]["platform_artifacts"] if entry["platform"] == selected_platform),
        None,
    )
    if not isinstance(selected, Mapping):
        _fail("artifact_missing", f"manifest has no platform artifact row for {selected_platform!r}")
    destination = Path(artifact_path) if artifact_path is not None else manifest_destination.parent / selected["artifact"]
    try:
        helpers.validate_artifact(manifest, manifest_destination, destination, platform=selected_platform)
    except Exception as exc:
        message = str(exc)
        kind = "artifact_hash_mismatch" if "SHA256" in message or "size" in message else "artifact_missing"
        raise EvaluatorFailure(kind, message) from exc
    artifact_hash = _sha256_file(destination)
    artifact_size = destination.stat().st_size
    _validate_manifest_contract(manifest)
    return ValidatedArtifact(
        manifest_path=manifest_destination,
        artifact_path=destination,
        manifest=manifest,
        platform=selected_platform,
        manifest_sha256=_sha256_file(manifest_destination),
        artifact_sha256=artifact_hash,
        artifact_size_bytes=artifact_size,
    )


def _dtype_token(value: Any) -> str:
    token = str(value).lower().replace(" ", "")
    return {
        "tensor(float)": "float32",
        "float": "float32",
        "<class'numpy.float32'>": "float32",
        "float32": "float32",
    }.get(token, token)


def _contract(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    contract = manifest.get("tensor_contract")
    if not isinstance(contract, Mapping):
        _fail("unsupported_tensor_contract", "manifest tensor_contract is not an object")
    return contract


def _validate_manifest_contract(manifest: Mapping[str, Any]) -> None:
    contract = _contract(manifest)
    inputs = contract.get("inputs")
    output = contract.get("output")
    if not isinstance(inputs, list) or len(inputs) != 2 or not all(isinstance(item, Mapping) for item in inputs):
        _fail("unsupported_tensor_contract", "manifest must declare exactly two tensor inputs")
    if [item.get("name") for item in inputs] != ["image1", "image2"]:
        _fail("unsupported_tensor_contract", "manifest inputs must be ordered image1, image2")
    for item in inputs:
        if item.get("dtype") != "float32" or item.get("layout") != "NCHW" or item.get("channels") not in ("RGB", ["R", "G", "B"]):
            _fail("unsupported_tensor_contract", "inputs must be float32 NCHW RGB tensors")
        value_range = item.get("range")
        if not isinstance(value_range, list) or len(value_range) != 2:
            _fail("unsupported_tensor_contract", "each input must declare a two-value range")
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) for value in value_range):
            _fail("unsupported_tensor_contract", "input range must be finite")
        if float(value_range[1]) <= float(value_range[0]):
            _fail("unsupported_tensor_contract", "input range must be increasing")
    if not isinstance(output, Mapping):
        _fail("unsupported_tensor_contract", "manifest output contract is not an object")
    if (
        output.get("name") != "flow"
        or output.get("dtype") != "float32"
        or output.get("layout") != "NCHW"
        or output.get("channels") != ["dx", "dy"]
        or output.get("direction") != "image1_to_image2"
        or output.get("units") not in {"input_pixels", "unpadded_analysis_pixels"}
    ):
        _fail("unsupported_tensor_contract", "output must be float32 NCHW dx/dy image1_to_image2 flow")
    padding = contract.get("padding")
    if not isinstance(padding, Mapping) or isinstance(padding.get("multiple"), bool) or not isinstance(padding.get("multiple"), int) or padding["multiple"] <= 0:
        _fail("unsupported_tensor_contract", "manifest must declare a positive padding multiple")
    if padding.get("policy") not in {
        "caller-replication-crop",
        "caller-reflection-crop",
        "graph-internal",
        "none",
    }:
        _fail("unsupported_tensor_contract", f"unsupported padding policy: {padding.get('policy')!r}")
    if contract.get("batch") != 1:
        _fail("unsupported_tensor_contract", "only batch size one is supported")


def _metadata_shape(value: Any) -> list[Any]:
    shape = getattr(value, "shape", None)
    if shape is None and isinstance(value, Mapping):
        shape = value.get("shape")
    if shape is None:
        _fail("unsupported_tensor_contract", "runtime tensor metadata has no shape")
    try:
        return list(shape)
    except TypeError as exc:
        raise EvaluatorFailure("unsupported_tensor_contract", "runtime tensor shape is not iterable") from exc


def _metadata_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if name is None and isinstance(value, Mapping):
        name = value.get("name")
    if not isinstance(name, str) or not name:
        _fail("unsupported_tensor_contract", "runtime tensor metadata has no name")
    return name


def _metadata_type(value: Any) -> str:
    item_type = getattr(value, "type", None)
    if item_type is None and isinstance(value, Mapping):
        item_type = value.get("type", value.get("dtype"))
    if item_type is None:
        _fail("unsupported_tensor_contract", "runtime tensor metadata has no dtype")
    return _dtype_token(item_type)


def _check_shape_metadata(shape: Sequence[Any], expected_channels: int, label: str) -> None:
    if len(shape) != 4:
        _fail("unsupported_tensor_contract", f"{label} must be rank four NCHW: {shape!r}")
    if isinstance(shape[0], int) and shape[0] != 1:
        _fail("unsupported_tensor_contract", f"{label} batch must be one: {shape!r}")
    if isinstance(shape[1], int) and shape[1] != expected_channels:
        _fail("unsupported_tensor_contract", f"{label} channel count is not {expected_channels}: {shape!r}")


@dataclass(frozen=True)
class SessionContract:
    requested_provider: str
    execution_provider: str
    selected_providers: tuple[str, ...]
    inputs: tuple[dict[str, Any], ...]
    outputs: tuple[dict[str, Any], ...]


def validate_session_contract(session: Any, artifact: ValidatedArtifact, provider: str) -> SessionContract:
    """Check provider priority and exact input/output names, types, layouts, and direction."""

    if provider not in PROVIDER_EXECUTION_NAMES:
        _fail("provider_unavailable", f"unknown provider token: {provider!r}")
    try:
        selected = tuple(str(item) for item in session.get_providers())
    except Exception as exc:
        raise EvaluatorFailure("provider_unavailable", f"runtime did not expose selected providers: {exc}") from exc
    expected_provider = PROVIDER_EXECUTION_NAMES[provider]
    if not selected or selected[0] != expected_provider:
        raise EvaluatorFailure(
            "provider_unavailable",
            f"requested {expected_provider} was not selected first; runtime selected {list(selected)!r}",
        )
    # ORT can report CPU as a lower-priority implementation fallback.  The safety property here
    # is exact requested-provider priority: a requested CPU/CUDA/CoreML provider may never be
    # accepted when absent or behind another provider.
    contract = _contract(artifact.manifest)
    expected_inputs = contract["inputs"]
    expected_output = contract["output"]
    try:
        runtime_inputs = list(session.get_inputs())
        runtime_outputs = list(session.get_outputs())
    except Exception as exc:
        raise EvaluatorFailure("unsupported_tensor_contract", f"runtime did not expose tensor metadata: {exc}") from exc
    if [_metadata_name(item) for item in runtime_inputs] != [item["name"] for item in expected_inputs]:
        _fail("unsupported_tensor_contract", "runtime input names do not match image1/image2")
    if [_metadata_name(item) for item in runtime_outputs] != [expected_output["name"]]:
        _fail("unsupported_tensor_contract", "runtime outputs do not match the declared flow output")
    input_records: list[dict[str, Any]] = []
    for runtime_item, expected in zip(runtime_inputs, expected_inputs):
        if _metadata_type(runtime_item) != "float32":
            _fail("unsupported_tensor_contract", f"{_metadata_name(runtime_item)} is not tensor(float)")
        shape = _metadata_shape(runtime_item)
        _check_shape_metadata(shape, 3, _metadata_name(runtime_item))
        input_records.append({"name": _metadata_name(runtime_item), "dtype": _metadata_type(runtime_item), "shape": shape, "layout": expected["layout"]})
    runtime_output = runtime_outputs[0]
    if _metadata_type(runtime_output) != "float32":
        _fail("unsupported_tensor_contract", "flow is not tensor(float)")
    output_shape = _metadata_shape(runtime_output)
    _check_shape_metadata(output_shape, 2, "flow")
    output_records = ({"name": _metadata_name(runtime_output), "dtype": _metadata_type(runtime_output), "shape": output_shape, "layout": expected_output["layout"], "direction": expected_output["direction"]},)
    fixed = artifact.manifest.get("validation", {}).get("shapes", {}).get("dynamic") is False
    if fixed:
        example = artifact.manifest["export"]["example_shape"]
        expected_input_shape = [1, 3, example[2], example[3]]
        expected_output_shape = [1, 2, example[2], example[3]]
        if list(input_records[0]["shape"]) != expected_input_shape or list(input_records[1]["shape"]) != expected_input_shape or list(output_shape) != expected_output_shape:
            _fail("unsupported_tensor_contract", "fixed-shape runtime metadata disagrees with manifest export shape")
    return SessionContract(provider, expected_provider, selected, tuple(input_records), output_records)


class RuntimeModule(Protocol):
    __version__: str

    def get_available_providers(self) -> Sequence[str]: ...

    def InferenceSession(self, path: str, *, providers: Sequence[str], **kwargs: Any) -> Any: ...


class ArrayModule(Protocol):
    float32: Any

    def asarray(self, value: Any, dtype: Any = ...) -> Any: ...

    def ascontiguousarray(self, value: Any) -> Any: ...


@dataclass(frozen=True)
class ProviderSession:
    session: Any
    contract: SessionContract


class Evaluator:
    """Execute one candidate/cap/conditioning cell using injected runtime dependencies."""

    def __init__(
        self,
        artifact: ValidatedArtifact,
        runtime: RuntimeModule,
        arrays: ArrayModule | None,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.artifact = artifact
        self.runtime = runtime
        self.arrays = arrays
        self.clock = clock

    def open_session(self, provider: str) -> ProviderSession:
        execution_provider = PROVIDER_EXECUTION_NAMES.get(provider)
        if execution_provider is None:
            _fail("provider_unavailable", f"unknown provider token: {provider!r}")
        try:
            available = list(self.runtime.get_available_providers())
        except Exception as exc:
            raise EvaluatorFailure("provider_unavailable", f"cannot query runtime providers: {exc}") from exc
        if execution_provider not in available:
            _fail("provider_unavailable", f"requested {execution_provider} is unavailable; available={available!r}")
        try:
            session = self.runtime.InferenceSession(
                str(self.artifact.artifact_path),
                providers=[execution_provider],
            )
        except Exception as exc:
            raise EvaluatorFailure("provider_unavailable", f"{execution_provider} session creation failed: {exc}") from exc
        return ProviderSession(session, validate_session_contract(session, self.artifact, provider))

    def verify(self, provider: str) -> dict[str, Any]:
        opened = self.open_session(provider)
        return {
            "candidate_id": self.artifact.candidate_id,
            "manifest_sha256": self.artifact.manifest_sha256,
            "artifact_sha256": self.artifact.artifact_sha256,
            "artifact_size_bytes": self.artifact.artifact_size_bytes,
            "requested_provider": opened.contract.execution_provider,
            "selected_providers": list(opened.contract.selected_providers),
            "runtime_version": str(getattr(self.runtime, "__version__", "unknown")),
            "inputs": list(opened.contract.inputs),
            "outputs": list(opened.contract.outputs),
        }

    def run_nchw_pair(
        self,
        first: Any,
        second: Any,
        *,
        provider: str,
        analysis_width: int,
        analysis_height: int,
        padded_width: int,
        padded_height: int,
        profile: str = "smoke",
        geometry: Mapping[str, Any] | None = None,
        preprocessing_ms: float = 0.0,
        stage_sampler: Callable[[str], ContextManager[Any]] | None = None,
    ) -> dict[str, Any]:
        """Run an already-conditioned/padded pair and return raw seam-compatible data.

        ``stage_sampler``, when given, is called with ``"session_create"`` around each fresh
        session's :meth:`open_session` and with ``"steady"`` around that session's warm-up plus
        steady-sample loop; the returned context manager is entered/exited around the wrapped
        work.  This is a P25-6 addition so a caller can attach an NVML polling window (see
        ``tools/bakeoff/nvml.py``) without this method losing ownership of profile/timing
        semantics.  It defaults to ``None``, in which case both stages are unwrapped and behavior
        is unchanged from before this parameter existed.
        """

        def _stage(name: str) -> ContextManager[Any]:
            if stage_sampler is None:
                return contextlib.nullcontext()
            return stage_sampler(name)

        arrays = self.arrays
        if arrays is None:
            _fail("runtime_error", "array module is required to run an inference")
        if profile not in PROFILES:
            _fail("runtime_error", f"unknown timing profile: {profile!r}")
        first_shape = tuple(int(value) for value in getattr(first, "shape", ()))
        second_shape = tuple(int(value) for value in getattr(second, "shape", ()))
        expected_shape = (1, 3, padded_height, padded_width)
        if first_shape != expected_shape or second_shape != expected_shape:
            _fail("unsupported_tensor_contract", f"input arrays must have shape {expected_shape!r}")
        feeds: dict[str, Any] | None = None
        raw_sessions: list[dict[str, Any]] = []
        steady_outputs: list[Any] = []
        counts = PROFILES[profile]
        for session_index in range(counts["fresh_sessions"]):
            create_start = self.clock()
            # The "session_create" poll window spans session creation THROUGH the first
            # inference call, not just open_session(): CUDA/cuDNN algorithm-selection and
            # workspace allocation commonly happen lazily on the first run, not at session
            # construction, so a window that stopped at open_session() would miss exactly the
            # transient a P25-6 peak measurement needs to catch.
            with _stage("session_create"):
                opened = self.open_session(provider)
                creation_ms = (self.clock() - create_start) * 1000.0
                session = opened.session
                input_names = [record["name"] for record in opened.contract.inputs]
                output_name = opened.contract.outputs[0]["name"]
                feeds = {input_names[0]: first, input_names[1]: second}
                first_start = self.clock()
                try:
                    first_output = session.run([output_name], feeds)
                except Exception as exc:
                    raise EvaluatorFailure("runtime_error", f"first inference failed: {exc}") from exc
                first_ms = (self.clock() - first_start) * 1000.0
            output = _one_output(first_output)
            _validate_runtime_output(output, arrays, analysis_width, analysis_height, padded_width, padded_height)
            warmup_recorded = counts["warmups_per_session"] == 1
            session_record: dict[str, Any] = {
                "session_index": session_index,
                "warmup_recorded": warmup_recorded,
                "session_creation_ms": creation_ms,
                "first_inference_ms": first_ms,
                "steady_samples_ms": [],
            }
            with _stage("steady"):
                if warmup_recorded:
                    warm_start = self.clock()
                    try:
                        warm_output = _one_output(session.run([output_name], feeds))
                    except Exception as exc:
                        raise EvaluatorFailure("runtime_error", f"warm-up inference failed: {exc}") from exc
                    _validate_runtime_output(warm_output, arrays, analysis_width, analysis_height, padded_width, padded_height)
                    session_record["warmup_ms"] = (self.clock() - warm_start) * 1000.0
                for _ in range(counts["steady_samples_per_session"]):
                    steady_start = self.clock()
                    try:
                        steady_output = _one_output(session.run([output_name], feeds))
                    except Exception as exc:
                        raise EvaluatorFailure("runtime_error", f"steady inference failed: {exc}") from exc
                    duration_ms = (self.clock() - steady_start) * 1000.0
                    _validate_runtime_output(steady_output, arrays, analysis_width, analysis_height, padded_width, padded_height)
                    session_record["steady_samples_ms"].append(duration_ms)
                    steady_outputs.append(steady_output)
            raw_sessions.append(session_record)
        post_start = self.clock()
        flows = [_crop_flow(output, arrays, analysis_width, analysis_height, padded_width, padded_height) for output in steady_outputs]
        postprocessing_ms = (self.clock() - post_start) * 1000.0
        if not flows:
            _fail("runtime_error", "runtime returned no steady output")
        repeated_p99 = _repeated_p99(flows)
        if not math.isfinite(float(preprocessing_ms)) or preprocessing_ms < 0.0:
            _fail("runtime_error", "preprocessing_ms must be finite and non-negative")
        raw_timing = {"preprocessing_ms": float(preprocessing_ms), "postprocessing_ms": postprocessing_ms, "sessions": raw_sessions}
        timing = reduce_timing(profile, raw_timing)
        geometry_record = dict(geometry) if geometry is not None else reduce_geometry(
            analysis_width,
            analysis_height,
            1.0,
            0.0,
            padded_width,
            padded_height,
        )
        return {
            "flow": flows[-1],
            "geometry": geometry_record,
            "timing": timing,
            "raw_timing": raw_timing,
            "metrics": {
                "nonfinite_fraction": 0.0,
                "repeated_run_p99_delta_px": repeated_p99,
                "not_applicable": list(REPORT_METRICS),
            },
            "environment": {
                "runtime_version": str(getattr(self.runtime, "__version__", "unknown")),
                "provider_version": opened.contract.execution_provider,
                "model_manifest_sha256": self.artifact.manifest_sha256,
            },
            "provider": opened.contract.execution_provider,
        }


def _one_output(value: Any) -> Any:
    if not isinstance(value, (list, tuple)) or len(value) != 1:
        _fail("runtime_error", "session.run must return exactly one flow output")
    return value[0]


def _array_shape(arrays: ArrayModule, value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        _fail("runtime_error", "runtime output has no shape")
    try:
        return tuple(int(item) for item in shape)
    except (TypeError, ValueError) as exc:
        raise EvaluatorFailure("runtime_error", f"runtime output shape is invalid: {shape!r}") from exc


def _array_finite(arrays: ArrayModule, value: Any) -> bool:
    try:
        result = arrays.isfinite(value)
        return bool(result.all()) if hasattr(result, "all") else all(bool(item) for item in result)
    except Exception:
        nested = value.tolist() if hasattr(value, "tolist") else value
        return all(math.isfinite(float(item)) for item in _flatten(nested))


def _array_dtype(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return "float32"
    return _dtype_token(dtype)


def _validate_runtime_output(value: Any, arrays: ArrayModule, analysis_width: int, analysis_height: int, padded_width: int, padded_height: int) -> None:
    shape = _array_shape(arrays, value)
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 2:
        _fail("unsupported_tensor_contract", f"flow output must be [1,2,H,W], got {shape!r}")
    if (shape[2], shape[3]) not in {(analysis_height, analysis_width), (padded_height, padded_width)}:
        _fail("unsupported_tensor_contract", f"flow output spatial shape {shape[2:]} does not match analysis/padded dimensions")
    if _array_dtype(value) != "float32":
        _fail("unsupported_tensor_contract", f"flow output dtype must be float32, got {_array_dtype(value)!r}")
    if not _array_finite(arrays, value):
        _fail("nonfinite_output", "flow output contains a non-finite value")


def _nested(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value


def _crop_flow(value: Any, arrays: ArrayModule, analysis_width: int, analysis_height: int, padded_width: int, padded_height: int) -> tuple[tuple[tuple[float, float], ...], ...]:
    _validate_runtime_output(value, arrays, analysis_width, analysis_height, padded_width, padded_height)
    data = _nested(value)
    return tuple(
        tuple((float(data[0][0][y][x]), float(data[0][1][y][x])) for x in range(analysis_width))
        for y in range(analysis_height)
    )


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        result: list[Any] = []
        for child in value:
            result.extend(_flatten(child))
        return result
    return [value]


def _repeated_p99(flows: Sequence[Any]) -> float:
    if len(flows) < 2:
        return 0.0
    reference = list(_flatten(flows[0]))
    deltas: list[float] = []
    for flow in flows[1:]:
        values = list(_flatten(flow))
        if len(values) != len(reference):
            _fail("runtime_error", "repeated flow outputs have different sizes")
        deltas.extend(abs(float(a) - float(b)) for a, b in zip(reference, values))
    if not deltas:
        return 0.0
    ordered = sorted(deltas)
    index = 0.99 * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (index - lower) * (ordered[upper] - ordered[lower])


def frame_from_pfm(path: Path | str, *, frame_number: int = 0, pixel_aspect_ratio: float = 1.0) -> dict[str, Any]:
    """Read one strict RGB PFM into the simple frame mapping accepted by ``condition_and_pad``."""

    try:
        image = read_pfm(Path(path))
    except PfmFailure:
        raise
    if image.channels != 3:
        _fail("input_invalid", "evaluator PFM smoke requires RGB PF files")
    if not math.isfinite(pixel_aspect_ratio) or pixel_aspect_ratio <= 0.0:
        _fail("input_invalid", "pixel aspect ratio must be positive and finite")
    return {
        "width": image.width,
        "height": image.height,
        "channels": image.channels,
        "rows": image.rows,
        "pixel_aspect_ratio": float(pixel_aspect_ratio),
        "frame": frame_number,
        "sha256": _sha256_file(Path(path)),
        "source": str(path),
    }


def _resize_bilinear(rows: Sequence[Sequence[Sequence[float]]], out_width: int, out_height: int) -> tuple[tuple[tuple[float, ...], ...], ...]:
    source_height = len(rows)
    source_width = len(rows[0])
    scale_x = out_width / source_width
    scale_y = out_height / source_height

    def sample(x: float, y: float) -> tuple[float, ...]:
        tx = min(source_width - 1, max(0.0, x))
        ty = min(source_height - 1, max(0.0, y))
        x0, y0 = math.floor(tx), math.floor(ty)
        x1, y1 = min(source_width - 1, x0 + 1), min(source_height - 1, y0 + 1)
        fx, fy = tx - x0, ty - y0
        return tuple(
            (1.0 - fy) * ((1.0 - fx) * rows[y0][x0][channel] + fx * rows[y0][x1][channel])
            + fy * ((1.0 - fx) * rows[y1][x0][channel] + fx * rows[y1][x1][channel])
            for channel in range(3)
        )

    return tuple(
        tuple(sample((x + 0.5) / scale_x - 0.5, (y + 0.5) / scale_y - 0.5) for x in range(out_width))
        for y in range(out_height)
    )


def _condition_and_pad_pair_impl(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    conditioning_token: str,
    cap_megapixels: float,
    manifest: Mapping[str, Any],
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Apply frozen conditioning and manifest-declared caller padding, returning NCHW arrays."""

    if first.get("width") != second.get("width") or first.get("height") != second.get("height"):
        _fail("input_invalid", "input pair dimensions differ")
    if first.get("pixel_aspect_ratio") != second.get("pixel_aspect_ratio"):
        _fail("input_invalid", "input pair pixel aspect ratios differ")
    width, height = int(first["width"]), int(first["height"])
    par = float(first.get("pixel_aspect_ratio", 1.0))
    try:
        analysis_width, analysis_height = analysis_dimensions(width, height, par, cap_megapixels)
    except (GeometryFailure, ValueError, TypeError) as exc:
        raise EvaluatorFailure("input_invalid", str(exc)) from exc
    try:
        conditioned = condition_pair(first["rows"], second["rows"], conditioning_token, channels=int(first["channels"]))
    except (ConditioningFailure, ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
        raise EvaluatorFailure("conditioning_failure", str(exc)) from exc
    contract = _contract(manifest)
    input_range = contract["inputs"][0]["range"]
    low, high = float(input_range[0]), float(input_range[1])
    location = contract["normalization_location"]
    if location not in {"graph", "caller", "none"}:
        _fail("unsupported_tensor_contract", f"unsupported normalization location: {location!r}")

    def pack(rows: Any) -> tuple[tuple[tuple[float, ...], ...], ...]:
        resized = _resize_bilinear(rows, analysis_width, analysis_height)
        packed = tuple(
            tuple(
                tuple(
                    (low + float(value) * (high - low)) if location == "graph" else float(value)
                    for value in pixel[:3]
                )
                for pixel in row
            )
            for row in resized
        )
        return packed

    first_rows, second_rows = pack(conditioned.first), pack(conditioned.second)
    padding = contract["padding"]
    policy, multiple = padding["policy"], int(padding["multiple"])

    def padded(rows: Any) -> Any:
        if policy in {"caller-replication-crop", "caller-reflection-crop"}:
            try:
                return pad_rows(rows, policy=policy, multiple=multiple)
            except (PaddingPolicyError, ValueError, TypeError) as exc:
                raise EvaluatorFailure("input_invalid", str(exc)) from exc
        if policy in {"none", "graph-internal"}:
            class NoPad:
                pass
            result = NoPad()
            result.width = analysis_width
            result.height = analysis_height
            result.pad_left = result.pad_right = result.pad_bottom = result.pad_top = 0
            result.rows = rows
            return result
        _fail("unsupported_tensor_contract", f"unsupported padding policy: {policy!r}")

    first_pad, second_pad = padded(first_rows), padded(second_rows)
    if (first_pad.width, first_pad.height) != (second_pad.width, second_pad.height):
        _fail("input_invalid", "padding produced different tensor dimensions for the pair")

    def nchw(padded_image: Any) -> Any:
        return tuple(
            tuple(tuple(float(padded_image.rows[y][x][channel]) for x in range(padded_image.width)) for y in range(padded_image.height))
            for channel in range(3)
        )

    try:
        geometry = reduce_geometry(width, height, par, cap_megapixels, first_pad.width, first_pad.height)
    except (ValueError, TypeError) as exc:
        raise EvaluatorFailure("input_invalid", str(exc)) from exc
    metadata = {
        "conditioning_parameters": conditioned.parameters,
        "padding_policy": policy,
        "padding_multiple": multiple,
        "analysis_width": analysis_width,
        "analysis_height": analysis_height,
    }
    return nchw(first_pad), nchw(second_pad), geometry, metadata


def condition_and_pad_pair(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    conditioning_token: str,
    cap_megapixels: float,
    manifest: Mapping[str, Any],
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Typed public wrapper around conditioning, resize, and declared padding."""

    try:
        return _condition_and_pad_pair_impl(
            first,
            second,
            conditioning_token=conditioning_token,
            cap_megapixels=cap_megapixels,
            manifest=manifest,
        )
    except EvaluatorFailure:
        raise
    except (ConditioningFailure, ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
        # This catches malformed frame rows, resize inputs, and padding-library failures that
        # occur after the narrower formula/geometry checks above.
        raise EvaluatorFailure("input_invalid", str(exc)) from exc


def _onnxruntime() -> Any:
    try:
        import onnxruntime as ort  # type: ignore
        return ort
    except ImportError:
        # P25-5 deliberately carries Microsoft's official ORT 1.29.0 CUDA-12 C/C++ archive,
        # rather than the PyPI 1.29 wheel (which is built for CUDA 13).  The native adapter is
        # loaded only when the ordinary Python module is absent; a present module remains the
        # explicit runtime selected by the caller.
        native_module: Any
        try:
            from . import native_ort as native_module
        except ImportError:
            try:  # direct ``python tools/bakeoff/evaluator.py`` invocation
                import native_ort as native_module  # type: ignore
            except ImportError as exc:
                raise DependencyFailure(
                    "runtime_error",
                    "onnxruntime Python module is absent and the native CUDA-12 bridge module could not be imported",
                ) from exc
        try:
            return native_module.load_runtime()
        except (native_module.NativeRuntimeUnavailable, OSError) as exc:
            raise DependencyFailure(
                "runtime_error",
                "onnxruntime Python module is absent and the native CUDA-12 bridge could not be loaded",
            ) from exc


def _numpy_runtime() -> tuple[Any, Any]:
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise DependencyFailure("runtime_error", "NumPy is required for PFM smoke") from exc
    return np, _onnxruntime()


def _to_numpy(np: Any, value: Any) -> Any:
    return np.ascontiguousarray(np.asarray(value, dtype=np.float32))


def _flow_digest(flow: Any) -> str:
    digest = hashlib.sha256()
    for value in _flatten(flow):
        digest.update(struct.pack("<f", float(value)))
    return digest.hexdigest()


def _cap_value(protocol: Mapping[str, Any], token: str) -> float:
    for cap in protocol.get("analysis_caps", []):
        if isinstance(cap, Mapping) and cap.get("token") == token:
            return float(cap["decimal_megapixels"])
    _fail("input_invalid", f"unknown analysis cap token: {token!r}")


def _run_cli(args: argparse.Namespace) -> dict[str, Any]:
    protocol = load_json(Path(args.protocol))
    artifact = validate_manifest_artifact(args.manifest, args.artifact, platform=args.platform, protocol_path=Path(args.protocol))
    if args.command == "smoke":
        np, ort = _numpy_runtime()
    else:
        np, ort = None, _onnxruntime()
    evaluator = Evaluator(artifact, ort, np)
    result = evaluator.verify(args.provider)
    if args.command == "smoke":
        first = frame_from_pfm(args.first_pfm, frame_number=0, pixel_aspect_ratio=args.pixel_aspect_ratio)
        second = frame_from_pfm(args.second_pfm, frame_number=1, pixel_aspect_ratio=args.pixel_aspect_ratio)
        preprocessing_start = time.perf_counter()
        first_nchw, second_nchw, geometry, metadata = condition_and_pad_pair(
            first, second,
            conditioning_token=args.conditioning,
            cap_megapixels=_cap_value(protocol, args.cap),
            manifest=artifact.manifest,
        )
        first_array, second_array = _to_numpy(np, (first_nchw,)), _to_numpy(np, (second_nchw,))
        preprocessing_ms = (time.perf_counter() - preprocessing_start) * 1000.0
        cell = evaluator.run_nchw_pair(
            first_array,
            second_array,
            provider=args.provider,
            analysis_width=geometry["analysis_width"],
            analysis_height=geometry["analysis_height"],
            padded_width=geometry["padded_width"],
            padded_height=geometry["padded_height"],
            profile=args.profile,
            geometry=geometry,
            preprocessing_ms=preprocessing_ms,
        )
        result["cell"] = {
            "geometry": geometry,
            "conditioning_parameters": metadata["conditioning_parameters"],
            "padding_policy": metadata["padding_policy"],
            "flow_shape": [1, 2, geometry["analysis_height"], geometry["analysis_width"]],
            "flow_sha256": _flow_digest(cell["flow"]),
            "timing": cell["timing"],
            "metrics": cell["metrics"],
            "input_frames": [
                {"frame": first["frame"], "sha256": first["sha256"]},
                {"frame": second["frame"], "sha256": second["sha256"]},
            ],
        }
    return result


def _write_json_result(path: Path, result: Mapping[str, Any]) -> None:
    """Publish one small, mode-0644 JSON result without clobbering an existing file."""

    if not path.parent.is_dir():
        _fail("runtime_error", f"JSON output parent is not a directory: {path.parent}")
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    descriptor = -1
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(str(path), flags, 0o644)
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise EvaluatorFailure("runtime_error", f"JSON output already exists: {path}") from exc
    except OSError as exc:
        raise EvaluatorFailure("runtime_error", f"cannot write JSON output {path}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--manifest", required=True, type=Path)
    common.add_argument("--artifact", type=Path)
    common.add_argument("--protocol", type=Path, default=V2_PROTOCOL)
    common.add_argument("--platform")
    common.add_argument("--provider", choices=sorted(PROVIDER_EXECUTION_NAMES), default="cpu")
    verify = subparsers.add_parser("verify", parents=[common], help="validate one local artifact and runtime/provider contract")
    verify.set_defaults(command="verify")
    smoke = subparsers.add_parser("smoke", parents=[common], help="run one deterministic pair of local PFM files")
    smoke.add_argument("--first-pfm", required=True, type=Path)
    smoke.add_argument("--second-pfm", required=True, type=Path)
    smoke.add_argument("--conditioning", default="native-clamp01-v1")
    smoke.add_argument("--cap", default="mp0_5")
    smoke.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    smoke.add_argument("--pixel-aspect-ratio", type=float, default=1.0)
    smoke.add_argument("--output", type=Path, help="optional new 0644 JSON result path")
    smoke.set_defaults(command="smoke")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = _run_cli(args)
        if getattr(args, "output", None) is not None:
            _write_json_result(args.output, result)
    except (EvaluatorFailure, PfmFailure) as exc:
        print(json.dumps({"error": {"kind": getattr(exc, "kind", "runtime_error"), "message": str(exc)}}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


__all__ = [
    "DependencyFailure",
    "Evaluator",
    "EvaluatorFailure",
    "PROVIDER_EXECUTION_NAMES",
    "ValidatedArtifact",
    "V2_PROTOCOL",
    "condition_and_pad_pair",
    "frame_from_pfm",
    "main",
    "validate_manifest_artifact",
    "validate_session_contract",
]


if __name__ == "__main__":
    raise SystemExit(main())
