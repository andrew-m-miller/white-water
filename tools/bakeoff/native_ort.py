"""ctypes adapter for the P25-5 native ONNX Runtime CUDA-12 bridge.

The official ONNX Runtime 1.29.0 Python wheel is built for CUDA 13.  P25-5 instead carries
Microsoft's official 1.29.0 CUDA-12 C/C++ archive and the tiny shared bridge built from
``ort_native_bridge.cpp``.  This module deliberately implements only the subset of the Python
runtime protocol consumed by :mod:`tools.bakeoff.evaluator`: provider discovery, session
metadata, CPU/CUDA session creation, and one float32 NCHW run.

The bridge and its ORT libraries live below ``$CONDA_PREFIX/lib/whitewater/ort-cuda12`` after
CI copies them into the conda-pack environment.  Resolution is relative to ``sys.prefix`` so
``conda-unpack`` can relocate the environment without rewriting this module.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


class NativeRuntimeUnavailable(RuntimeError):
    """The checked-in/native P25-5 bridge cannot be loaded."""


@dataclass(frozen=True)
class TensorMeta:
    name: str
    type: str
    shape: tuple[Any, ...]


def _decode(value: bytes | None) -> str:
    return "" if value is None else value.decode("utf-8", errors="replace")


def _root() -> Path:
    explicit = os.environ.get("WHITEWATER_NATIVE_ORT_ROOT")
    if explicit:
        return Path(explicit)
    return Path(sys.prefix) / "lib" / "whitewater" / "ort-cuda12"


class _Bridge:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / "libwhitewater_ort_bridge.so"
        if not self.path.is_file() or self.path.is_symlink():
            raise NativeRuntimeUnavailable(f"native ORT bridge is missing: {self.path}")
        try:
            mode = getattr(ctypes, "RTLD_LOCAL", 0) | getattr(ctypes, "RTLD_NOW", 0)
            self.library = ctypes.CDLL(str(self.path), mode=mode)
        except OSError as exc:
            raise NativeRuntimeUnavailable(f"cannot load native ORT bridge {self.path}: {exc}") from exc

        self.library.ww_ort_last_error.argtypes = []
        self.library.ww_ort_last_error.restype = ctypes.c_char_p
        self.library.ww_ort_version.argtypes = []
        self.library.ww_ort_version.restype = ctypes.c_char_p
        self.library.ww_ort_available_providers.argtypes = []
        self.library.ww_ort_available_providers.restype = ctypes.c_char_p
        self.library.ww_ort_session_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.library.ww_ort_session_create.restype = ctypes.c_void_p
        self.library.ww_ort_session_release.argtypes = [ctypes.c_void_p]
        self.library.ww_ort_session_release.restype = None
        self.library.ww_ort_session_providers.argtypes = [ctypes.c_void_p]
        self.library.ww_ort_session_providers.restype = ctypes.c_char_p
        self.library.ww_ort_session_metadata.argtypes = [ctypes.c_void_p]
        self.library.ww_ort_session_metadata.restype = ctypes.c_char_p
        self.library.ww_ort_session_run.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.library.ww_ort_session_run.restype = ctypes.c_int
        self.library.ww_ort_free.argtypes = [ctypes.c_void_p]
        self.library.ww_ort_free.restype = None

    def error(self, fallback: str) -> str:
        value = _decode(self.library.ww_ort_last_error())
        return value or fallback

    def json_call(self, function: Any, fallback: str) -> Any:
        try:
            return json.loads(_decode(function()))
        except (TypeError, UnicodeError, ValueError) as exc:
            raise NativeRuntimeUnavailable(f"native ORT bridge returned invalid {fallback}: {exc}") from exc


class NativeSession:
    def __init__(self, bridge: _Bridge, handle: int, provider_name: str) -> None:
        self._bridge = bridge
        self._handle = ctypes.c_void_p(handle)
        self._provider_name = provider_name
        self._metadata: dict[str, list[TensorMeta]] | None = None

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown ordering is undefined
        handle = getattr(self, "_handle", None)
        bridge = getattr(self, "_bridge", None)
        if handle and bridge:
            try:
                bridge.library.ww_ort_session_release(handle)
            except Exception:
                pass

    def get_providers(self) -> list[str]:
        value = self._bridge.json_call(
            lambda: self._bridge.library.ww_ort_session_providers(self._handle),
            "session providers",
        )
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise NativeRuntimeUnavailable("native ORT bridge returned invalid session providers")
        return value

    def _load_metadata(self) -> dict[str, list[TensorMeta]]:
        if self._metadata is not None:
            return self._metadata
        value = self._bridge.json_call(
            lambda: self._bridge.library.ww_ort_session_metadata(self._handle),
            "session metadata",
        )
        if not isinstance(value, Mapping):
            raise NativeRuntimeUnavailable("native ORT bridge returned non-object metadata")
        result: dict[str, list[TensorMeta]] = {}
        for key in ("inputs", "outputs"):
            raw = value.get(key)
            if not isinstance(raw, list):
                raise NativeRuntimeUnavailable(f"native ORT bridge metadata has no {key} list")
            records: list[TensorMeta] = []
            for item in raw:
                if not isinstance(item, Mapping):
                    raise NativeRuntimeUnavailable(f"native ORT bridge {key} metadata item is not an object")
                name, type_name, shape = item.get("name"), item.get("type"), item.get("shape")
                if not isinstance(name, str) or not isinstance(type_name, str) or not isinstance(shape, list):
                    raise NativeRuntimeUnavailable(f"native ORT bridge {key} metadata item is malformed")
                records.append(TensorMeta(name, type_name, tuple(shape)))
            result[key] = records
        self._metadata = result
        return result

    def get_inputs(self) -> list[TensorMeta]:
        return list(self._load_metadata()["inputs"])

    def get_outputs(self) -> list[TensorMeta]:
        return list(self._load_metadata()["outputs"])

    def run(self, names: Sequence[str], feeds: Mapping[str, Any]) -> list[Any]:
        if len(names) != 1 or names[0] != "flow":
            raise RuntimeError("native ORT bridge supports exactly the declared flow output")
        inputs = self.get_inputs()
        if tuple(item.name for item in inputs) != ("image1", "image2"):
            raise RuntimeError("native ORT bridge only accepts the SEA-RAFT image1/image2 inputs")
        outputs = self.get_outputs()
        if len(outputs) != 1 or outputs[0].name != "flow" or outputs[0].type != "tensor(float)":
            raise RuntimeError("native ORT bridge only accepts the SEA-RAFT float32 flow output")
        if inputs[0].name not in feeds or inputs[1].name not in feeds:
            raise RuntimeError("native ORT bridge requires image1 and image2 feeds")
        try:
            import numpy as np  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime environment gate owns this
            raise NativeRuntimeUnavailable("NumPy is required by the native ORT adapter") from exc
        first = np.ascontiguousarray(np.asarray(feeds[inputs[0].name], dtype=np.float32))
        second = np.ascontiguousarray(np.asarray(feeds[inputs[1].name], dtype=np.float32))
        if first.shape != second.shape or first.ndim == 0 or first.ndim > 8:
            raise RuntimeError("native ORT bridge input arrays must have equal rank 1..8 shapes")
        if any(int(value) <= 0 for value in first.shape):
            raise RuntimeError("native ORT bridge input arrays must have positive dimensions")
        shape = (ctypes.c_int64 * first.ndim)(*(int(value) for value in first.shape))
        output_pointer = ctypes.POINTER(ctypes.c_float)()
        output_count = ctypes.c_size_t()
        output_shape = (ctypes.c_int64 * 8)()
        output_rank = ctypes.c_size_t()
        ok = self._bridge.library.ww_ort_session_run(
            self._handle,
            first.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            second.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            shape,
            first.ndim,
            ctypes.byref(output_pointer),
            ctypes.byref(output_count),
            output_shape,
            8,
            ctypes.byref(output_rank),
        )
        if not ok:
            raise RuntimeError(self._bridge.error("native ORT inference failed"))
        try:
            if output_rank.value > 8:
                raise RuntimeError("native ORT bridge returned an invalid output rank")
            dimensions = tuple(int(output_shape[index]) for index in range(output_rank.value))
            if any(value < 0 for value in dimensions):
                raise RuntimeError("native ORT bridge returned a negative output dimension")
            expected_count = 1
            for dimension in dimensions:
                expected_count *= dimension
            if expected_count != output_count.value:
                raise RuntimeError(
                    "native ORT bridge output shape/count mismatch: "
                    f"shape={dimensions!r} count={output_count.value}"
                )
            if output_count.value and not bool(output_pointer):
                raise RuntimeError("native ORT bridge returned a null output buffer")
            result = np.ctypeslib.as_array(output_pointer, shape=(output_count.value,)).copy()
            return [result.reshape(dimensions)]
        finally:
            self._bridge.library.ww_ort_free(output_pointer)


class NativeRuntime:
    """RuntimeModule-compatible wrapper around the native C API bridge."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else _root()
        self._bridge = _Bridge(self.root)
        self.__version__ = _decode(self._bridge.library.ww_ort_version())
        if self.__version__ != "1.29.0":
            raise NativeRuntimeUnavailable(
                f"native ORT bridge version is not exactly 1.29.0: {self.__version__!r}"
            )

    def get_available_providers(self) -> list[str]:
        value = self._bridge.json_call(
            self._bridge.library.ww_ort_available_providers,
            "available providers",
        )
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise NativeRuntimeUnavailable("native ORT bridge returned invalid available providers")
        return value

    def InferenceSession(
        self,
        path: str,
        *,
        providers: Sequence[str],
        **kwargs: Any,
    ) -> NativeSession:
        unknown = sorted(kwargs)
        if unknown:
            raise RuntimeError(f"native ORT bridge does not support session options: {unknown!r}")
        if len(providers) != 1 or providers[0] not in {"CPUExecutionProvider", "CUDAExecutionProvider"}:
            raise RuntimeError(f"native ORT bridge only accepts one CPU/CUDA provider: {providers!r}")
        provider_name = "cpu" if providers[0] == "CPUExecutionProvider" else "cuda"
        handle = self._bridge.library.ww_ort_session_create(
            os.fsencode(path), os.fsencode(provider_name)
        )
        if not handle:
            raise RuntimeError(self._bridge.error("native ORT session creation failed"))
        return NativeSession(self._bridge, int(handle), providers[0])


def load_runtime(root: Path | str | None = None) -> NativeRuntime:
    """Load the relocated native runtime, raising a typed dependency error on failure."""

    return NativeRuntime(root)


__all__ = [
    "NativeRuntime",
    "NativeRuntimeUnavailable",
    "NativeSession",
    "SessionOptions",
    "TensorMeta",
    "load_runtime",
]
