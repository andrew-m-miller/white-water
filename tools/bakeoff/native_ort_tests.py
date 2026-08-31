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
        self.ww_ort_last_error = _Function(lambda: b"")
        self.ww_ort_version = _Function(lambda: b"1.29.0")
        self.ww_ort_available_providers = _Function(
            lambda: b'["CUDAExecutionProvider","CPUExecutionProvider"]'
        )
        def create_session(path, provider):
            self.requested_provider = provider
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


if __name__ == "__main__":
    unittest.main()
