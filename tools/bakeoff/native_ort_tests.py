"""Focused contract tests for the P25-5 native ORT adapter (no ORT/GPU required)."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import native_ort


class _Function:
    def __init__(self, callback):
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.callback(*args)


class _FakeLibrary:
    def __init__(self):
        self._output = (ctypes.c_float * 4)(1.0, 2.0, 3.0, 4.0)
        self.requested_provider = b"cpu"
        self.requested_gpu_mem_limit_bytes = 0
        self.ww_ort_last_error = _Function(lambda: b"")
        self.ww_ort_version = _Function(lambda: b"1.29.0")
        self.ww_ort_available_providers = _Function(
            lambda: b'["CUDAExecutionProvider","CPUExecutionProvider"]'
        )
        def create_session(path, provider, gpu_mem_limit_bytes):
            self.requested_provider = provider
            self.requested_gpu_mem_limit_bytes = int(gpu_mem_limit_bytes)
            return 17

        self.ww_ort_session_create = _Function(create_session)
        self.ww_ort_session_release = _Function(lambda handle: None)
        self.ww_ort_session_providers = _Function(
            lambda handle: (
                b'["CUDAExecutionProvider"]'
                if self.requested_provider == b"cuda"
                else b'["CPUExecutionProvider"]'
            )
        )
        self.ww_ort_session_metadata = _Function(
            lambda handle: (
                b'{"inputs":[{"name":"image1","type":"tensor(float)","shape":[1,3,"height","width"]},'
                b'{"name":"image2","type":"tensor(float)","shape":[1,3,"height","width"]}],'
                b'"outputs":[{"name":"flow","type":"tensor(float)","shape":[1,2,"height","width"]}]}'
            )
        )

        def run(*args):
            ctypes.cast(args[5], ctypes.POINTER(ctypes.POINTER(ctypes.c_float)))[0] = ctypes.cast(
                self._output, ctypes.POINTER(ctypes.c_float)
            )
            ctypes.cast(args[6], ctypes.POINTER(ctypes.c_size_t))[0] = 4
            output_shape = ctypes.cast(args[7], ctypes.POINTER(ctypes.c_int64))
            output_shape[0], output_shape[1], output_shape[2] = 1, 2, 2
            ctypes.cast(args[9], ctypes.POINTER(ctypes.c_size_t))[0] = 3
            return 1

        self.ww_ort_session_run = _Function(run)
        self.ww_ort_free = _Function(lambda pointer: None)


class _FakeBridge:
    def __init__(self):
        self.library = _FakeLibrary()

    def json_call(self, function, fallback):
        return json.loads(function().decode("utf-8"))

    def error(self, fallback):
        return fallback


class NativeOrtTests(unittest.TestCase):
    def test_public_exports_are_defined(self) -> None:
        self.assertTrue(all(hasattr(native_ort, name) for name in native_ort.__all__))

    def test_missing_bridge_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(native_ort.NativeRuntimeUnavailable):
                native_ort.NativeRuntime(Path(temporary))

    def test_native_session_preserves_metadata_and_runs_float32_pair(self) -> None:
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("NumPy-backed execution is required by the pinned P25-5 runtime gate")

        session = native_ort.NativeSession(_FakeBridge(), 17, "CPUExecutionProvider")
        self.assertEqual(session.get_providers(), ["CPUExecutionProvider"])
        self.assertEqual([item.name for item in session.get_inputs()], ["image1", "image2"])
        self.assertEqual(session.get_outputs()[0].shape, (1, 2, "height", "width"))
        output = session.run(
            ["flow"],
            {
                "image1": np.zeros((1, 3, 1, 1), dtype=np.float32),
                "image2": np.ones((1, 3, 1, 1), dtype=np.float32),
            },
        )
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].shape, (1, 2, 2))
        np.testing.assert_array_equal(output[0], np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32))

    def test_runtime_requires_ort_129(self) -> None:
        class WrongVersionBridge(_FakeBridge):
            def __init__(self, root):
                super().__init__()
                self.library.ww_ort_version = _Function(lambda: b"1.28.0")

        with patch.object(native_ort, "_Bridge", WrongVersionBridge):
            with self.assertRaisesRegex(native_ort.NativeRuntimeUnavailable, "not exactly 1.29.0"):
                native_ort.NativeRuntime("/unused")

    def test_cuda_session_preserves_requested_provider_without_disabling_cpu_nodes(self) -> None:
        bridge = _FakeBridge()
        runtime = object.__new__(native_ort.NativeRuntime)
        runtime._bridge = bridge
        session = runtime.InferenceSession(
            "model.onnx", providers=["CUDAExecutionProvider"]
        )
        self.assertEqual(bridge.library.requested_provider, b"cuda")
        self.assertEqual(session.get_providers(), ["CUDAExecutionProvider"])
        with self.assertRaisesRegex(RuntimeError, "does not support session options"):
            runtime.InferenceSession(
                "model.onnx", providers=["CUDAExecutionProvider"], sess_options=object()
            )

    def test_cuda_gpu_mem_limit_is_forwarded_as_bytes(self) -> None:
        bridge = _FakeBridge()
        runtime = object.__new__(native_ort.NativeRuntime)
        runtime._bridge = bridge
        # The evaluator hands the adapter onnxruntime's official contract: a providers list with a
        # trailing CPU fallback and a positionally-aligned provider_options list -- the CUDA arena
        # bound in bytes plus an empty dict for the CPU fallback provider.
        runtime.InferenceSession(
            "model.onnx",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            provider_options=[
                {"gpu_mem_limit": 22000 * 1024 * 1024, "arena_extend_strategy": "kSameAsRequested"},
                {},
            ],
        )
        # The bytes value is exactly what OrtCUDAProviderOptions::gpu_mem_limit wants, and the
        # bridge session runs the primary CUDA provider.
        self.assertEqual(bridge.library.requested_gpu_mem_limit_bytes, 22000 * 1024 * 1024)
        self.assertEqual(bridge.library.requested_provider, b"cuda")

    def test_absent_provider_options_leaves_arena_unbounded(self) -> None:
        bridge = _FakeBridge()
        runtime = object.__new__(native_ort.NativeRuntime)
        runtime._bridge = bridge
        runtime.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])
        # A cell that requested no ceiling passes 0 (unbounded), preserving historical behaviour.
        self.assertEqual(bridge.library.requested_gpu_mem_limit_bytes, 0)
        # An aligned but all-empty options list is likewise unbounded.
        runtime.InferenceSession(
            "model.onnx",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            provider_options=[{}, {}],
        )
        self.assertEqual(bridge.library.requested_gpu_mem_limit_bytes, 0)

    def test_gpu_mem_limit_rejected_for_cpu_provider(self) -> None:
        bridge = _FakeBridge()
        runtime = object.__new__(native_ort.NativeRuntime)
        runtime._bridge = bridge
        with self.assertRaisesRegex(RuntimeError, "only accepts CUDA provider options"):
            runtime.InferenceSession(
                "model.onnx",
                providers=["CPUExecutionProvider"],
                provider_options=[
                    {"gpu_mem_limit": 4096 * 1024 * 1024, "arena_extend_strategy": "kSameAsRequested"},
                ],
            )

    def test_unknown_provider_option_is_rejected(self) -> None:
        bridge = _FakeBridge()
        runtime = object.__new__(native_ort.NativeRuntime)
        runtime._bridge = bridge
        with self.assertRaisesRegex(RuntimeError, "does not support provider options"):
            runtime.InferenceSession(
                "model.onnx",
                providers=["CUDAExecutionProvider"],
                provider_options=[
                    {
                        "gpu_mem_limit": 4096 * 1024 * 1024,
                        "arena_extend_strategy": "kSameAsRequested",
                        "unexpected": 1,
                    },
                ],
            )

    def test_arena_extend_strategy_must_match_probe(self) -> None:
        bridge = _FakeBridge()
        runtime = object.__new__(native_ort.NativeRuntime)
        runtime._bridge = bridge
        with self.assertRaisesRegex(RuntimeError, "arena_extend_strategy"):
            runtime.InferenceSession(
                "model.onnx",
                providers=["CUDAExecutionProvider"],
                provider_options=[
                    {"gpu_mem_limit": 4096 * 1024 * 1024, "arena_extend_strategy": "kNextPowerOfTwo"},
                ],
            )

    def test_provider_options_must_align_with_providers(self) -> None:
        bridge = _FakeBridge()
        runtime = object.__new__(native_ort.NativeRuntime)
        runtime._bridge = bridge
        with self.assertRaisesRegex(RuntimeError, "align 1:1 with providers"):
            runtime.InferenceSession(
                "model.onnx",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                provider_options=[
                    {"gpu_mem_limit": 4096 * 1024 * 1024, "arena_extend_strategy": "kSameAsRequested"},
                ],
            )
        with self.assertRaisesRegex(RuntimeError, "must be a list aligned with providers"):
            runtime.InferenceSession(
                "model.onnx",
                providers=["CUDAExecutionProvider"],
                provider_options={"gpu_mem_limit": 4096 * 1024 * 1024},
            )

    def test_trailing_non_cpu_provider_is_rejected(self) -> None:
        bridge = _FakeBridge()
        runtime = object.__new__(native_ort.NativeRuntime)
        runtime._bridge = bridge
        with self.assertRaisesRegex(RuntimeError, "trailing CPU fallback"):
            runtime.InferenceSession(
                "model.onnx",
                providers=["CUDAExecutionProvider", "CoreMLExecutionProvider"],
            )

    def test_non_positive_gpu_mem_limit_is_rejected(self) -> None:
        bridge = _FakeBridge()
        runtime = object.__new__(native_ort.NativeRuntime)
        runtime._bridge = bridge
        for bad in (0, -1, 4096.0, True):
            with self.assertRaisesRegex(RuntimeError, "positive integer number of bytes"):
                runtime.InferenceSession(
                    "model.onnx",
                    providers=["CUDAExecutionProvider"],
                    provider_options=[{"gpu_mem_limit": bad, "arena_extend_strategy": "kSameAsRequested"}],
                )


if __name__ == "__main__":
    unittest.main()
