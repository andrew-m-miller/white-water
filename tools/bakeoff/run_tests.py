#!/usr/bin/env python3
"""Tests for the P25-6 end-to-end resumable offline profile driver (``tools.bakeoff.run``).

Runs entirely without numpy, onnxruntime, the OpenEXR bindings, pynvml, or a GPU: every dependency is
injected (a fake array/runtime module pair, a fake NVML backend, a fake EXR decoder, and --
new in this revision -- a fake host-load checkpoint and a fake CUDA measurement runner).
``validate_manifest_artifact`` still needs a real manifest/artifact pair and a real protocol
file for its own contract checks, so this reuses the same checked-in fixture manifest
``models/fixtures/positive/artifact-v1.json`` that ``evaluator_tests.py`` uses, together with
the real ``bakeoff/protocol-v2.json``. Matrix planning and report validation, however, use a
small hand-built protocol/corpus (mirroring the style of ``matrix_tests.py`` and
``reporting_tests.py``) so the tests do not depend on the full, frozen candidate/provider matrix.
"""

from __future__ import annotations

import copy
import csv
import errno
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import types
from pathlib import Path
from typing import Any, Mapping

from . import run as run_module
from . import synthetic as synthetic_module
from . import validator as validator_module
from .exr import ExrFailure
from .matrix import build_matrix
from .nvml import NVML_CSV_HEADER, NvmlFailure, NvmlSampler
from .resume import ResumeFailure, mark_in_progress
from .run import CudaMeasurementResult, DriverFailure, RunConfig, run_bakeoff
from .run_spec import RunSpec
from .validator import ValidationError, canonical_sha256, load_json, validate_report_consistency

ROOT = Path(__file__).resolve().parents[2]
POSITIVE_MANIFEST = ROOT / "models" / "fixtures" / "positive" / "artifact-v1.json"
POSITIVE_ARTIFACT = ROOT / "models" / "fixtures" / "positive" / "valid.bin"
V2_PROTOCOL_PATH = ROOT / "bakeoff" / "protocol-v2.json"
REPORT_SCHEMA = load_json(ROOT / "bakeoff" / "report-v2.schema.json")
CORPUS_SCHEMA = load_json(ROOT / "bakeoff" / "corpus-v1.schema.json")

CANDIDATE_ID = "fixture-candidate"


# --------------------------------------------------------------------------------------------
# Fake array/runtime/NVML modules -- no numpy, onnxruntime, or pynvml anywhere in this file.
# --------------------------------------------------------------------------------------------


def _shape(value: Any) -> tuple[int, ...]:
    shape: list[int] = []
    current = value
    while isinstance(current, (list, tuple)):
        shape.append(len(current))
        current = current[0] if current else []
    return tuple(shape)


class _FakeArray:
    def __init__(self, data: Any):
        self.data = data
        self.shape = _shape(data)
        self.dtype = "float32"

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
        return _FakeArray(value)

    def ascontiguousarray(self, value: Any) -> Any:
        return value

    def isfinite(self, value: _FakeArray) -> _Finite:
        return _Finite(True)


class _Meta:
    def __init__(self, name: str, shape: list[Any]):
        self.name = name
        self.type = "tensor(float)"
        self.shape = shape


def _constant_flow(height: int, width: int, dx: float, dy: float) -> _FakeArray:
    return _FakeArray([[
        [[dx for _ in range(width)] for _ in range(height)],
        [[dy for _ in range(width)] for _ in range(height)],
    ]])


class _FakeSession:
    def __init__(self, selected: list[str] | None = None, *, flow_dx: float = 0.0, flow_dy: float = 0.0):
        self.calls = 0
        self._selected = selected or ["CPUExecutionProvider"]
        self._flow_dx = flow_dx
        self._flow_dy = flow_dy

    def get_providers(self) -> list[str]:
        return list(self._selected)

    def get_inputs(self) -> list[_Meta]:
        return [_Meta("image1", [1, 3, "height", "width"]), _Meta("image2", [1, 3, "height", "width"])]

    def get_outputs(self) -> list[_Meta]:
        return [_Meta("flow", [1, 2, "height", "width"])]

    def run(self, names: list[str], feeds: dict[str, Any]) -> list[_FakeArray]:
        self.calls += 1
        first = next(iter(feeds.values()))
        _, _, height, width = first.shape
        return [_constant_flow(height, width, self._flow_dx, self._flow_dy)]


class _FakeRuntime:
    __version__ = "fake-ort-run-tests"

    def __init__(self, providers: list[str] | None = None, *, path_dependent: bool = False):
        self.providers = providers or ["CPUExecutionProvider"]
        self.sessions_created = 0
        self._path_dependent = path_dependent
        # Assigns each distinct artifact path the Nth-seen (dx, dy) pair, by simple sequential
        # counting -- not a hash of the path, so there is no possibility of two distinct paths
        # coincidentally colliding on the same flow value (a real, observed flakiness this
        # replaced: a small hash-mod-N range gave two random temp-dir paths a non-negligible
        # chance of landing on the same bucket across different test runs).
        self._path_flow_values: dict[str, tuple[float, float]] = {}
        # P25-7: the provider_options each InferenceSession call received, so a driver test can
        # confirm a configured CUDA arena ceiling was threaded all the way to the runtime.
        self.provider_options_seen: list[Mapping[str, Any] | None] = []

    def get_available_providers(self) -> list[str]:
        return ["CPUExecutionProvider", "CUDAExecutionProvider"]

    def InferenceSession(
        self, path: str, *, providers: list[str], provider_options: Mapping[str, Any] | None = None,
    ) -> _FakeSession:
        self.sessions_created += 1
        self.provider_options_seen.append(provider_options)
        if self._path_dependent:
            if path not in self._path_flow_values:
                index = len(self._path_flow_values) + 1
                self._path_flow_values[path] = (float(index) * 3.0, float(index) * 5.0)
            dx, dy = self._path_flow_values[path]
        else:
            dx, dy = 0.0, 0.0
        return _FakeSession(list(providers), flow_dx=dx, flow_dy=dy)


class _FakeNvmlBackend:
    """A trivially growing device-memory fake, no pynvml/GPU involved."""

    def __init__(self):
        self._used = 1000.0

    def device_handle(self, device_index: int) -> Any:
        return device_index

    def device_used_mib(self, handle: Any) -> float:
        self._used += 1.0
        return self._used

    def process_used_mib(self, handle: Any, pid: int) -> float | None:
        return None


class _ScriptedBackend:
    """A fixed, deterministic reading sequence, shared across every ``nvml_backend_factory()``
    call (the SAME instance is always returned) so a "post-work exit reading" call continues the
    same script rather than restarting it. Used by the Fix D test to prove a specific transient
    spike is captured and that the process_exit reading is a genuinely distinct later value."""

    def __init__(self, script: list[tuple[float, float | None]]):
        self._script = list(script)
        self._index = -1

    def device_handle(self, device_index: int) -> Any:
        return device_index

    def device_used_mib(self, handle: Any) -> float:
        self._index = min(self._index + 1, len(self._script) - 1)
        return self._script[self._index][0]

    def process_used_mib(self, handle: Any, pid: int) -> float | None:
        return self._script[self._index][1]


def _fake_cuda_measurement_runner(work, nvml_backend_factory, device_index, poll_interval_s):
    """Default in-process fake :data:`run_module.CudaMeasurementRunner` for tests.

    Mirrors exactly what the real subprocess runner does (baseline -> work under a staged
    poll -> cleanup -> a fresh post-work "exit" reading) without an actual ``multiprocessing``
    child, per the WP4 follow-up review's instruction to keep the real subprocess path
    injectable/testable via an in-process fake.
    """

    backend = nvml_backend_factory()
    sampler = NvmlSampler(backend, device_index, poll_interval_s=poll_interval_s)
    sampler.sample("baseline")
    payload = work(lambda name: sampler.poll(name))
    sampler.sample("cleanup")
    exit_backend = nvml_backend_factory()
    handle = exit_backend.device_handle(device_index)
    used_mib = exit_backend.device_used_mib(handle)
    process_used_mib = exit_backend.process_used_mib(handle, os.getpid())
    return CudaMeasurementResult(payload, sampler.samples, used_mib, process_used_mib)


class _RecordingHostLoadCheckpoint:
    """Auto-confirms every host-load boundary (no interactive prompt) and records the sequence
    of ``host_load`` values it was called with, in order, for test assertions."""

    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, host_load: str) -> None:
        self.calls.append(host_load)


def _fake_exr_decoder(width: int, height: int) -> Any:
    def decoder(path: str, *, frame_number: int, pixel_aspect_ratio: float) -> dict[str, Any]:
        # Spatially varying (not a flat constant) so two different predicted flows sample
        # genuinely different pixel values -- needed by the blinded-preview test, which proves
        # two candidates' warped previews differ.
        rows = tuple(
            tuple((x / max(width - 1, 1), y / max(height - 1, 1), 0.5) for x in range(width))
            for y in range(height)
        )
        return {
            "width": width, "height": height, "channels": 3, "rows": rows,
            "pixel_aspect_ratio": pixel_aspect_ratio, "frame": frame_number,
            "sha256": f"{frame_number:064x}"[-64:], "source": path,
            "source_channels": "RGB", "source_format": "half",
        }
    return decoder


# --------------------------------------------------------------------------------------------
# Minimal hand-built v2 protocol/corpus (matrix planning + report validation only).
# --------------------------------------------------------------------------------------------


def _protocol() -> dict[str, Any]:
    providers = [
        {"token": "cpu", "environment": "el8-x86_64", "purpose": "correctness", "cap_tokens": ["mp0_5"]},
        {"token": "cuda", "environment": "el8-x86_64", "purpose": "selection", "cap_tokens": ["mp0_5"]},
    ]
    return {
        "schema_version": 2,
        "protocol_id": "whitewater-p25-v2",
        "candidate_ids": [{"id": CANDIDATE_ID, "role": "shipping-candidate"}],
        "providers": providers,
        "analysis_caps": [{"token": "mp0_5", "decimal_megapixels": 0.5}],
        "conditioning": [{"token": "native-clamp01-v1", "accepted_encoding": "scene-linear-or-log"}],
        "cap_accounting": {"unit_pixels": 1000000},
        "hard_gates": {
            "nonfinite_fraction_max": 0.0,
            "repeated_run_p99_delta_px_max": 0.05,
            "peak_incremental_device_memory_gib_max": 15.0,
        },
        "metrics": [
            "endpoint_error_px", "fraction_le_1px", "fraction_le_3px", "landmark_median_error_px",
            "landmark_p95_error_px", "visible_warp_residual", "forward_backward_residual_px",
            "chain_drift_px", "nonfinite_fraction", "repeated_run_p99_delta_px",
        ],
        "profiles": {"smoke": {"fresh_sessions": 1, "warmups_per_session": 0, "steady_samples_per_session": 2}},
        "synthetic_categories": ["identity", "chain"],
        "primary_production_categories": ["motion-blur"],
        "required_synthetic_cases": ["identity", "chain-1"],
    }


def _identity_case() -> Any:
    return synthetic_module.case_map()["identity"]


def _chain1_case() -> Any:
    return synthetic_module.case_map()["chain-1"]


def _synthetic_identity_shot() -> dict[str, Any]:
    case = _identity_case()
    return {
        "id": "syn-identity", "case_id": "identity",
        "path_pattern": f"generated://{case.path_token}/frame.%04d.pfm",
        "first_frame": case.first_frame, "last_frame": case.last_frame,
        "reference_frame": case.reference_frame,
        "width": case.width, "height": case.height, "pixel_aspect_ratio": 1.0,
        "encoding": "scene-linear", "channels": "RGB", "bit_depth": "float",
        "categories": ["identity"],
        "truth": {"kind": "analytic", "definition": "identity case has zero motion"},
    }


def _synthetic_chain1_shot() -> dict[str, Any]:
    case = _chain1_case()
    return {
        "id": "syn-chain1", "case_id": "chain-1",
        "path_pattern": f"generated://{case.path_token}/frame.%04d.pfm",
        "first_frame": case.first_frame, "last_frame": case.last_frame,
        "reference_frame": case.reference_frame,
        "width": case.width, "height": case.height, "pixel_aspect_ratio": 1.0,
        "encoding": "scene-linear", "channels": "RGB", "bit_depth": "float",
        "categories": ["chain"], "chain_length": 1,
        "truth": {"kind": "analytic", "definition": "chain-1 case has a small x shift per link"},
    }


def _production_shot() -> dict[str, Any]:
    return {
        "id": "prod-sample",
        "path_pattern": "/nonexistent/plate.%04d.exr",
        "first_frame": 1001, "last_frame": 1005, "reference_frame": 1003,
        "width": 8, "height": 6, "pixel_aspect_ratio": 1.0,
        "encoding": "scene-linear", "channels": "RGB", "bit_depth": "half",
        "categories": ["motion-blur"],
    }


def _corpus() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_id": "whitewater-p25-v1",
        "corpus_id": "run-tests-corpus",
        "partitions": [
            {
                "id": "synthetic", "kind": "synthetic",
                "shots": [_synthetic_identity_shot(), _synthetic_chain1_shot()],
            },
            {
                "id": "production_external", "kind": "production_external",
                "shots": [_production_shot()],
            },
        ],
    }


def _candidate_entry(candidate_id: str = CANDIDATE_ID) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "status": "excluded",
        "measurement_status": "measurable",
        "source_commit": "1" * 40,
        "checkpoint_sha256": "2" * 64,
        "artifact_sha256": "3" * 64,
        "export_environment_sha256": "4" * 64,
        "manifest_sha256": "5" * 64,
        "artifact_size_bytes": 1000,
        "measurement_providers": ["cpu", "cuda"],
        "exclusion_reason": {"type": "license_unknown", "message": "test fixture, not a shipping claim"},
        "license_verdicts": {"code": "unknown", "checkpoint": "unknown", "backbone": "unknown"},
        "redistribution_permitted": {"code": "unknown", "checkpoint": "unknown", "backbone": "unknown"},
        "redistribution_terms_reviewed": {"code": True, "checkpoint": True, "backbone": True},
    }


def _candidate_entries() -> list[dict[str, Any]]:
    return [_candidate_entry()]


def _selection(*, shot_ids: list[str], provider: str = "cpu", host_loads: list[str] | None = None) -> dict[str, Any]:
    return {
        "profile": "smoke",
        "environment": "el8-x86_64",
        "candidate_ids": [CANDIDATE_ID],
        "conditioning_tokens": ["native-clamp01-v1"],
        "cap_tokens": ["mp0_5"],
        "providers": [{"token": provider, "host_loads": host_loads or ["not_applicable"]}],
        "shot_ids": shot_ids,
    }


def _report_metadata(*, extra_hardware: dict[str, str] | None = None) -> dict[str, Any]:
    hardware = {"platform": "linux", "architecture": "x86_64", "os_release": "fixture-el8", "cpu": "fixture-cpu"}
    if extra_hardware:
        hardware.update(extra_hardware)
    return {
        "report_id": "run-tests-report",
        "runner": {
            "name": "ww-bakeoff-tests", "version": "0.0.0-test", "source_commit": "6" * 40,
            "evaluator_sha256": "7" * 64, "runtime": "fake-runtime", "runtime_sha256": "8" * 64,
        },
        "hardware": hardware,
    }


def _artifact_map(directory: Path) -> dict[str, dict[str, str]]:
    manifest = directory / "manifest.json"
    artifact = directory / "valid.bin"
    shutil.copy2(POSITIVE_MANIFEST, manifest)
    shutil.copy2(POSITIVE_ARTIFACT, artifact)
    manifest.chmod(0o644)
    artifact.chmod(0o644)
    return {CANDIDATE_ID: {"manifest": str(manifest), "artifact": str(artifact)}}


def _artifact_map_entry_for(directory: Path, candidate_id: str, suffix: str) -> dict[str, str]:
    """A manifest/artifact pair for a distinct candidate id, at a distinct file path (so a
    path-dependent fake runtime can tell two candidates' sessions apart)."""

    manifest_data = json.loads(POSITIVE_MANIFEST.read_text(encoding="utf-8"))
    manifest_data["candidate"]["id"] = candidate_id
    manifest_path = directory / f"manifest-{suffix}.json"
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
    manifest_path.chmod(0o644)
    artifact_path = directory / f"valid-{suffix}.bin"
    shutil.copy2(POSITIVE_ARTIFACT, artifact_path)
    artifact_path.chmod(0o644)
    return {"manifest": str(manifest_path), "artifact": str(artifact_path)}


def _config(
    tmp: Path,
    *,
    shot_ids: list[str],
    provider: str = "cpu",
    host_loads: list[str] | None = None,
    chain_offsets: tuple[int, ...] | list[int] = (1, 2, 4, 8),
    nvml_backend_factory=None,
    exr_decoder=None,
    hardware: dict[str, str] | None = None,
    runtime_module: Any = None,
    cuda_measurement_runner=None,
    host_load_checkpoint=None,
) -> RunConfig:
    output_dir = tmp / "output"
    return RunConfig(
        protocol=_protocol(),
        corpus=_corpus(),
        candidate_entries=_candidate_entries(),
        selection=_selection(shot_ids=shot_ids, provider=provider, host_loads=host_loads),
        artifact_map=_artifact_map(tmp),
        report_schema=REPORT_SCHEMA,
        corpus_schema=CORPUS_SCHEMA,
        output_dir=output_dir,
        state_path=output_dir / "state.json",
        device_index=0,
        poll_interval_s=0.01,
        chain_offsets=chain_offsets,
        report_metadata=_report_metadata(extra_hardware=hardware),
        protocol_path=V2_PROTOCOL_PATH,
        replace=False,
        runtime_module=runtime_module or _FakeRuntime(),
        array_module=_FakeArrays(),
        nvml_backend_factory=nvml_backend_factory,
        exr_decoder=exr_decoder or _fake_exr_decoder(8, 6),
        host_load_checkpoint=host_load_checkpoint or _RecordingHostLoadCheckpoint(),
        cuda_measurement_runner=cuda_measurement_runner or _fake_cuda_measurement_runner,
    )


# --------------------------------------------------------------------------------------------
# Small unit tests.
# --------------------------------------------------------------------------------------------


def test_cap_megapixels_looks_up_token_and_rejects_unknown() -> None:
    protocol = _protocol()
    assert run_module._cap_megapixels(protocol, "mp0_5") == 0.5
    try:
        run_module._cap_megapixels(protocol, "mp99")
    except run_module.DriverFailure as failure:
        assert failure.kind == "unknown_cap"
    else:
        raise AssertionError("expected DriverFailure for an unknown cap token")


def test_review_label_is_deterministic_and_does_not_embed_candidate_id() -> None:
    label_a = run_module._review_label("shot-1", "sea-raft-m")
    label_b = run_module._review_label("shot-1", "sea-raft-m")
    label_c = run_module._review_label("shot-1", "waft-twins")
    assert label_a == label_b
    assert label_a != label_c
    assert "sea-raft-m" not in label_a
    assert label_a.startswith("candidate-")


def test_exr_failure_maps_known_kinds_to_permitted_result_failure_types() -> None:
    missing = run_module._exr_failure(ExrFailure("missing_file", "x"), stage="load_input")
    assert missing["type"] == "missing_input"
    channels = run_module._exr_failure(ExrFailure("unsupported_channels", "x"), stage="load_input")
    assert channels["type"] == "unsupported_tensor_contract"
    unknown = run_module._exr_failure(ExrFailure("something_new", "x"), stage="load_input")
    assert unknown["type"] == "input_invalid"
    assert missing["stage"] == "load_input"


def test_unpadded_grid_crops_bottom_left_region() -> None:
    # [channel][y][x], 2x3 padded to 4x5 by appending on the right/top only.
    nchw = [
        [[1, 2, 3, 9] for _ in range(2)] + [[8, 8, 8, 8] for _ in range(3)],
        [[4, 5, 6, 9] for _ in range(2)] + [[8, 8, 8, 8] for _ in range(3)],
    ]
    grid = run_module._unpadded_grid(nchw, analysis_width=3, analysis_height=2)
    assert len(grid) == 2 and len(grid[0]) == 3
    assert grid[0][0] == (1, 4)
    assert grid[1][2] == (3, 6)


def test_dense_truth_and_mask_identity_case_is_zero_everywhere() -> None:
    case = _identity_case()
    grid, mask = run_module._dense_truth_and_mask(
        case, case.reference_frame, case.reference_frame + 1, case.width, case.height, 1.0, 1.0,
    )
    assert all(all(value for value in row) for row in mask)
    assert all(cell == (0.0, 0.0) for row in grid for cell in row)


def test_write_csv_file_is_single_write_no_clobber_unless_replace_and_skips_empty() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-csv-") as tmp:
        path = Path(tmp) / "review.csv"
        run_module._write_csv_file(path, ("a", "b"), [], replace=False)
        assert not path.exists()

        run_module._write_csv_file(path, ("a", "b"), [["1", "2"], ["3", "4"]], replace=False)
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        assert rows == [["a", "b"], ["1", "2"], ["3", "4"]]

        try:
            run_module._write_csv_file(path, ("a", "b"), [["9", "9"]], replace=False)
        except run_module.DriverFailure as failure:
            assert failure.kind == "output_exists"
        else:
            raise AssertionError("expected DriverFailure(output_exists)")

        run_module._write_csv_file(path, ("a", "b"), [["9", "9"]], replace=True)
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        assert rows == [["a", "b"], ["9", "9"]]


def test_verify_repair_rejects_a_symlink_even_when_target_is_canonical() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-csv-symlink-") as tmp:
        directory = Path(tmp)
        target = directory / "target.csv"
        path = directory / "review.csv"
        payload_rows = [["1", "2"]]
        run_module._write_csv_file(target, ("a", "b"), payload_rows, replace=False)
        original_target = target.read_bytes()
        path.symlink_to(target)

        try:
            run_module._write_csv_file(
                path, ("a", "b"), payload_rows, replace=False, verify_and_repair=True,
            )
        except DriverFailure as failure:
            assert failure.kind == "output_path"
        else:
            raise AssertionError("verify/repair must reject a symlinked CSV path")
        assert path.is_symlink()
        assert path.resolve() == target.resolve()
        assert target.read_bytes() == original_target


def test_runner_log_appends_timestamped_lines_and_survives_reopen() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-log-") as tmp:
        path = Path(tmp) / "runner.log"
        log = run_module.RunnerLog(path)
        log.write("first line")
        log.close()
        log2 = run_module.RunnerLog(path)
        log2.write("second line")
        log2.close()
        content = path.read_text(encoding="utf-8").splitlines()
        assert len(content) == 2
        assert content[0].endswith("first line")
        assert content[1].endswith("second line")


def test_driver_direct_and_module_help_use_the_same_package_imports() -> None:
    commands = (
        [sys.executable, str(ROOT / "tools" / "bakeoff" / "run.py"), "--help"],
        [sys.executable, "-m", "tools.bakeoff.run", "--help"],
    )
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.startswith("usage: run.py")
        assert "ImportError" not in completed.stderr


def test_runner_log_fsync_failure_is_typed() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-log-fsync-") as tmp:
        path = Path(tmp) / "runner.log"
        log = run_module.RunnerLog(path)
        original_fsync = os.fsync

        def failing_fsync(_fd: int) -> None:
            raise OSError("simulated log durability failure")

        os.fsync = failing_fsync
        try:
            try:
                log.write("must be durable")
            except DriverFailure as failure:
                assert failure.kind == "log_write"
            else:
                raise AssertionError("expected DriverFailure(log_write)")
        finally:
            os.fsync = original_fsync
            log.close()


def test_report_semantic_unicode_failure_is_typed() -> None:
    """A lone surrogate cannot be UTF-8 encoded and must not escape the reuse gate."""

    with tempfile.TemporaryDirectory(prefix="whitewater-run-report-unicode-") as tmp:
        try:
            run_module._compare_reusable_report_semantics(
                {"unexpected": "\ud800"},
                {"unexpected": "valid"},
                json_path=Path(tmp) / "report.json",
            )
        except DriverFailure as failure:
            assert failure.kind == "report_identity_mismatch"
        else:
            raise AssertionError("expected DriverFailure(report_identity_mismatch)")


def test_replay_resource_and_rows_matches_a_live_sampler() -> None:
    backend = _ScriptedBackend([(1000.0, None), (5000.0, None), (1200.0, None)])
    live = NvmlSampler(backend, 0, poll_interval_s=0.0)
    live.sample("baseline")
    live.sample("session_create")
    live.sample("cleanup")
    identity_fields = {
        "candidate_id": "c", "shot_id": "s", "conditioning_token": "t", "cap_token": "cap",
        "provider": "cuda", "host_load": "idle",
    }
    resource, rows = run_module._replay_resource_and_rows(identity_fields, live.samples, 0)
    assert resource["peak_device_memory_mib"] == 5000.0
    assert resource["baseline_device_memory_mib"] == 1000.0
    assert tuple(rows[1][:6]) == ("c", "s", "t", "cap", "cuda", "idle")


def test_nvml_evidence_is_cell_bound_at_commit_and_exact_ref_regeneration() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-nvml-binding-") as tmp:
        directory = Path(tmp).resolve()
        cell = run_module.CellKey("candidate", "shot", "conditioning", "cap", "cuda", "idle")
        result = {**run_module._base_result_fields(cell), "status": "pass"}
        valid_row = [
            "candidate", "shot", "conditioning", "cap", "cuda", "idle",
            "baseline", "0", "1.0", "100.0", "",
        ]
        wrong_identity_row = [
            "other-candidate", "other-shot", "conditioning", "cap", "cpu", "idle",
            "baseline", "0", "1.0", "100.0", "",
        ]
        bad_stage_row = list(valid_row)
        bad_stage_row[6] = "not-a-stage"
        identity = {"test": "nvml-binding"}

        with run_module.ArtifactStore(directory / "artifacts", identity) as store:
            for rows, expected_kind in (
                ((tuple(wrong_identity_row),), "nvml_identity"),
                ((tuple(bad_stage_row),), "nvml_stage"),
            ):
                try:
                    run_module._commit_cell_bundle(
                        store,
                        cell,
                        run_module.CellBundle(result=result, nvml_rows=rows),
                        nvml_enabled=True,
                    )
                except run_module.ArtifactStoreFailure as failure:
                    assert failure.kind == expected_kind
                else:
                    raise AssertionError(f"expected ArtifactStoreFailure({expected_kind})")

        # A store-level publisher can create a structurally valid but semantically wrong evidence
        # payload; the exact-ref regeneration boundary must bind it to the paired CellKey again.
        config = _config(
            directory,
            shot_ids=["syn-identity"],
            provider="cuda",
            host_loads=["idle"],
            nvml_backend_factory=lambda: _FakeNvmlBackend(),
        )
        selection_axes = {
            key: config.selection[key]
            for key in ("candidate_ids", "shot_ids", "conditioning_tokens", "cap_tokens", "providers")
        }
        plan = build_matrix(
            config.protocol, config.corpus, config.candidate_entries, selection_axes,
            config.selection["profile"], config.selection["environment"],
        )
        exact_cell = plan.cells[0]
        exact_result = {**run_module._base_result_fields(exact_cell), "status": "pass"}
        wrong_payload = run_module._canonical_nvml_rows_bytes((tuple(wrong_identity_row),))
        with run_module.ArtifactStore(directory / "exact-artifacts", {"test": "nvml-exact-ref"}) as store:
            attempt = store.begin(run_module._artifact_cell_id(exact_cell))
            attempt.stage_bytes("result.json", run_module._canonical_result_bytes(exact_result))
            attempt.stage_bytes("evidence/nvml_rows.json", wrong_payload)
            artifact_ref = attempt.commit()
            try:
                run_module._regenerate_public_evidence_outputs(
                    store,
                    config.output_dir,
                    config.corpus,
                    plan,
                    [{"result": exact_result, "artifact_ref": artifact_ref}],
                    replace=True,
                    nvml_enabled=True,
                )
            except run_module.ArtifactStoreFailure as failure:
                assert failure.kind == "nvml_identity"
            else:
                raise AssertionError("exact-ref regeneration must reject cross-cell NVML evidence")


# --------------------------------------------------------------------------------------------
# Integration tests.
# --------------------------------------------------------------------------------------------


def test_smoke_profile_synthetic_identity_produces_a_valid_report_and_no_nvml_csv() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-smoke-") as tmp:
        config = _config(Path(tmp), shot_ids=["syn-identity"])
        result = run_bakeoff(config)
        assert not result.incomplete
        report = result.report
        assert report is not None
        validate_report_consistency(report, config.protocol, config.report_schema, config.corpus, config.corpus_schema)
        assert report["summary"]["passed_cells"] == 1
        cell_result = report["results"][0]
        assert cell_result["status"] == "pass"
        assert cell_result["metrics"]["endpoint_error_px"] == 0.0
        assert cell_result["metrics"]["fraction_le_1px"] == 1.0
        assert "chain_drift_px" in cell_result["metrics"]["not_applicable"]
        assert cell_result["resource"] == {"peak_incremental_device_memory_gib": 0.0}

        for name in ("report.json", "report.csv", "summary.txt", "runner.log"):
            assert (config.output_dir / name).is_file(), name
        log_text = (config.output_dir / "runner.log").read_text(encoding="utf-8")
        assert "cell artifact committed status=pass" in log_text
        assert "cell pass" not in log_text
        assert not (config.output_dir / "nvml.csv").exists()
        assert not (config.output_dir / "review.csv").exists()

        on_disk = json.loads((config.output_dir / "report.json").read_text(encoding="utf-8"))
        assert on_disk == report


def test_rerun_with_same_state_is_idempotent_and_does_not_recompute() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-resume-") as tmp:
        config = _config(Path(tmp), shot_ids=["syn-identity"])
        first = run_bakeoff(config)
        sessions_after_first = config.runtime_module.sessions_created
        assert sessions_after_first >= 1

        # A second driver invocation against the SAME state/output points at a fresh runtime
        # module; if the driver recomputed the cell it would call InferenceSession again.
        config2 = copy.copy(config)
        config2.runtime_module = _FakeRuntime()
        second = run_bakeoff(config2)
        assert not second.incomplete
        assert config2.runtime_module.sessions_created == 0
        assert second.report == first.report


def test_production_partition_uses_injected_exr_decoder_and_emits_review_row() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-production-") as tmp:
        config = _config(Path(tmp), shot_ids=["prod-sample"], exr_decoder=_fake_exr_decoder(8, 6))
        result = run_bakeoff(config)
        assert not result.incomplete
        validate_report_consistency(result.report, config.protocol, config.report_schema, config.corpus, config.corpus_schema)
        cell_result = result.report["results"][0]
        assert cell_result["status"] == "pass"
        not_applicable = set(cell_result["metrics"]["not_applicable"])
        assert {"endpoint_error_px", "fraction_le_1px", "fraction_le_3px", "chain_drift_px"} <= not_applicable
        assert "visible_warp_residual" in cell_result["metrics"]
        assert "forward_backward_residual_px" in cell_result["metrics"]

        review_path = config.output_dir / "review.csv"
        assert review_path.is_file()
        with review_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        assert tuple(rows[0]) == run_module.REVIEW_CSV_HEADER
        assert len(rows) == 2
        assert rows[1][1] == "prod-sample"
        # The candidate label must not leak the real candidate id into the anonymous review row.
        assert CANDIDATE_ID not in rows[1][0]

        # Fix C: the preview directory holds a blinded warp + flow-visualization PFM pair, not
        # a raw dump of the (identical for every candidate) input frames.
        preview_dir = Path(rows[1][7])
        assert (preview_dir / "offset_1_warped.pfm").is_file()
        assert (preview_dir / "offset_1_flow.pfm").is_file()
        # An offset within the shot's frame range (1003 reference, 1001..1005) also gets a
        # preview; one well outside the range (8) is silently skipped, not a cell failure.
        assert (preview_dir / "offset_2_warped.pfm").is_file()
        assert not (preview_dir / "offset_8_warped.pfm").exists()


def test_chain_shot_computes_chain_drift_px() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-chain-") as tmp:
        config = _config(Path(tmp), shot_ids=["syn-chain1"])
        result = run_bakeoff(config)
        assert not result.incomplete
        validate_report_consistency(result.report, config.protocol, config.report_schema, config.corpus, config.corpus_schema)
        cell_result = result.report["results"][0]
        assert cell_result["status"] == "pass"
        assert "chain_drift_px" in cell_result["metrics"]
        assert "chain_drift_px" not in cell_result["metrics"]["not_applicable"]


def test_cuda_cell_writes_nvml_csv_with_required_stages_and_resource() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-cuda-") as tmp:
        config = _config(
            Path(tmp), shot_ids=["syn-identity"], provider="cuda", host_loads=["idle"],
            nvml_backend_factory=lambda: _FakeNvmlBackend(),
            hardware={"gpu": "fixture-gpu", "driver": "fixture-driver"},
        )
        result = run_bakeoff(config)
        assert not result.incomplete
        validate_report_consistency(result.report, config.protocol, config.report_schema, config.corpus, config.corpus_schema)
        cell_result = result.report["results"][0]
        assert cell_result["provider"] == "cuda"
        assert cell_result["host_load"] == "idle"
        resource = cell_result["resource"]
        assert resource["peak_incremental_device_memory_gib"] >= 0.0
        stages = {sample["stage"] for sample in resource.get("nvml_samples", [])}
        assert {"baseline", "session_create", "steady", "cleanup", "process_exit"} <= stages

        nvml_path = config.output_dir / "nvml.csv"
        assert nvml_path.is_file()
        with nvml_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        assert tuple(rows[0]) == NVML_CSV_HEADER
        assert len(rows) > 1

        # Fix B: the CUDA host_load boundary was confirmed before the cell ran.
        assert config.host_load_checkpoint.calls == ["idle"]


def test_cuda_gpu_mem_limit_is_recorded_in_resource_evidence() -> None:
    # P25-7: a selection that bounds the CUDA arena must (a) reach OrtCUDAProviderOptions via the
    # evaluator's provider_options (asserted by native_ort/evaluator unit tests) and (b) record
    # the ceiling as resource evidence in report.json, next to the peak/baseline numbers, without
    # disturbing the frozen incremental-memory gate.
    with tempfile.TemporaryDirectory(prefix="whitewater-run-cuda-arena-") as tmp:
        runtime = _FakeRuntime(["CUDAExecutionProvider", "CPUExecutionProvider"])
        config = _config(
            Path(tmp), shot_ids=["syn-identity"], provider="cuda", host_loads=["idle"],
            nvml_backend_factory=lambda: _FakeNvmlBackend(), runtime_module=runtime,
            hardware={"gpu": "fixture-gpu", "driver": "fixture-driver"},
        )
        config.selection = {
            **config.selection,
            "providers": [{"token": "cuda", "host_loads": ["idle"], "gpu_mem_limit_mib": 22000}],
        }
        result = run_bakeoff(config)
        # The ceiling reached the runtime in onnxruntime's official list-aligned shape on every
        # session this cell opened: the CUDA arena bound in bytes plus an empty dict for the CPU
        # fallback provider (see native_ort/evaluator unit tests).
        expected_options = [
            {"gpu_mem_limit": 22000 * 1024 * 1024, "arena_extend_strategy": "kSameAsRequested"},
            {},
        ]
        assert runtime.provider_options_seen, runtime.provider_options_seen
        assert all(
            options == expected_options for options in runtime.provider_options_seen
        ), runtime.provider_options_seen
        assert not result.incomplete
        # The schema validator accepts the added optional resource/matrix field.
        validate_report_consistency(
            result.report, config.protocol, config.report_schema, config.corpus, config.corpus_schema,
        )
        cell_result = result.report["results"][0]
        resource = cell_result["resource"]
        assert resource["gpu_mem_limit_mib"] == 22000, resource
        # The ceiling is descriptive provenance only: the frozen gate metric is still present and
        # is not the ceiling value.
        assert "peak_incremental_device_memory_gib" in resource
        assert resource["peak_incremental_device_memory_gib"] != 22000
        # The matrix selector in the report also records which provider was bounded and to what.
        assert result.report["matrix"]["providers"] == [
            {"token": "cuda", "host_loads": ["idle"], "gpu_mem_limit_mib": 22000}
        ]


def test_cuda_cell_without_arena_limit_records_no_ceiling() -> None:
    # Back-compat: a CUDA selection that omits gpu_mem_limit_mib produces a resource block with no
    # gpu_mem_limit_mib field, exactly as before P25-7.
    with tempfile.TemporaryDirectory(prefix="whitewater-run-cuda-noarena-") as tmp:
        config = _config(
            Path(tmp), shot_ids=["syn-identity"], provider="cuda", host_loads=["idle"],
            nvml_backend_factory=lambda: _FakeNvmlBackend(),
            hardware={"gpu": "fixture-gpu", "driver": "fixture-driver"},
        )
        result = run_bakeoff(config)
        assert not result.incomplete
        resource = result.report["results"][0]["resource"]
        assert "gpu_mem_limit_mib" not in resource, resource
        assert "gpu_mem_limit_mib" not in result.report["matrix"]["providers"][0]


def test_validator_rejects_each_pass_only_hard_gate() -> None:
    """Each report-side per-cell hard gate must reject a still-passing result.

    These checks intentionally mutate an otherwise valid published report.  The runner's
    execution-side classification is covered below; this test pins the validator contract so a
    future change cannot accidentally make an over-limit pass row publishable again.
    """

    with tempfile.TemporaryDirectory(prefix="whitewater-run-hard-gates-validator-") as tmp:
        config = _config(Path(tmp), shot_ids=["syn-identity"])
        result = run_bakeoff(config)
        assert result.report is not None
        report = result.report

        violations = (
            ("metrics", "nonfinite_fraction", 0.000001),
            ("metrics", "repeated_run_p99_delta_px", 0.050001),
            ("resource", "peak_incremental_device_memory_gib", 15.000001),
        )
        for section, field, value in violations:
            with_value = copy.deepcopy(report)
            with_value["results"][0][section][field] = value
            expected_path = f"$.results[0].{section}.{field}"
            try:
                validate_report_consistency(
                    with_value,
                    config.protocol,
                    config.report_schema,
                    config.corpus,
                    config.corpus_schema,
                )
            except ValidationError as failure:
                assert failure.path == expected_path
                assert "pass result exceeds" in failure.message
            else:
                raise AssertionError(f"expected validator rejection for {expected_path}")


def test_execution_classifies_metric_hard_gate_overruns() -> None:
    protocol = _protocol()
    resource = {"peak_incremental_device_memory_gib": 0.0}
    for field, value in (
        ("nonfinite_fraction", 0.000001),
        ("repeated_run_p99_delta_px", 0.050001),
    ):
        metrics = {"nonfinite_fraction": 0.0, "repeated_run_p99_delta_px": 0.0}
        metrics[field] = value
        failure = run_module._hard_gate_failure(protocol, metrics, resource)
        assert failure is not None
        assert failure["type"] == "quality_gate_failed"
        assert failure["stage"] == "metrics"
        assert field in failure["message"]


def test_cuda_memory_gate_becomes_failed_cell_with_evidence_and_publishes_package() -> None:
    """A measured peak-memory overrun is a typed cell failure, not a report-publication abort.

    The peak is deliberately transient (the session-create sample), matching the target failure
    shape that motivated the regression.  The failed result must retain timing/resource evidence,
    commit the exact NVML rows, and regenerate the public CSV from that committed generation.
    """

    with tempfile.TemporaryDirectory(prefix="whitewater-run-hard-gate-memory-") as tmp:
        shared_backend = _ScriptedBackend([
            (1000.0, None),   # baseline
            (18000.0, None),  # session_create transient: 16.60 GiB incremental
            (12000.0, None),  # steady
            (1100.0, None),   # cleanup
            (900.0, None),    # process_exit
        ])
        config = _config(
            Path(tmp),
            shot_ids=["syn-identity"],
            provider="cuda",
            host_loads=["idle"],
            nvml_backend_factory=lambda: shared_backend,
            hardware={"gpu": "fixture-gpu", "driver": "fixture-driver"},
        )

        result = run_bakeoff(config)
        assert not result.incomplete
        assert result.report is not None
        report = result.report
        validate_report_consistency(
            report,
            config.protocol,
            config.report_schema,
            config.corpus,
            config.corpus_schema,
        )

        cell_result = report["results"][0]
        assert cell_result["status"] == "fail"
        failure = cell_result["failure"]
        assert failure["type"] == "quality_gate_failed"
        assert failure["stage"] == "resource"
        assert "peak_incremental_device_memory_gib" in failure["message"]

        # A gate failure after a completed measurement keeps the full timing/resource record.
        assert cell_result["timing"]["steady_samples_ms"]
        resource = cell_result["resource"]
        assert resource["peak_incremental_device_memory_gib"] > config.protocol["hard_gates"][
            "peak_incremental_device_memory_gib_max"
        ]
        assert resource["peak_device_memory_mib"] == max(
            sample["used_mib"] for sample in resource["nvml_samples"]
        )
        assert {sample["stage"] for sample in resource["nvml_samples"]} >= {
            "baseline", "session_create", "steady", "cleanup", "process_exit",
        }

        # The exact failed-cell generation must carry the same NVML evidence used to compute the
        # resource gate, rather than silently dropping it because the result is not ``pass``.
        state = json.loads(config.state_path.read_text(encoding="utf-8"))
        artifact_ref = state["entries"][0]["artifact_ref"]
        with run_module.ArtifactStore(config.output_dir.resolve() / ".artifacts", state["identity"]) as store:
            manifest = store.load_ref(artifact_ref)
            assert "evidence/nvml_rows.json" in {entry["path"] for entry in manifest["artifacts"]}
            committed_rows = run_module._decode_nvml_rows(
                store.read_artifact(artifact_ref, "evidence/nvml_rows.json"),
                expected_identity={
                    "candidate_id": cell_result["candidate_id"],
                    "shot_id": cell_result["shot_id"],
                    "conditioning_token": cell_result["conditioning_token"],
                    "cap_token": cell_result["cap_token"],
                    "provider": cell_result["provider"],
                    "host_load": cell_result["host_load"],
                },
            )
        assert len(committed_rows) == len(resource["nvml_samples"])
        assert [row[6] for row in committed_rows] == [sample["stage"] for sample in resource["nvml_samples"]]
        assert [float(row[9]) for row in committed_rows] == [sample["used_mib"] for sample in resource["nvml_samples"]]

        # Report publication completes all required public outputs, including the regenerated
        # device log.  A report-side validation exception would leave report.json absent.
        for name in ("report.json", "report.csv", "summary.txt", "runner.log", "nvml.csv"):
            assert (config.output_dir / name).is_file(), name
        with (config.output_dir / "nvml.csv").open(newline="", encoding="utf-8") as stream:
            public_rows = list(csv.reader(stream))
        assert tuple(public_rows[0]) == NVML_CSV_HEADER
        assert public_rows[1:] == committed_rows
        with (config.output_dir / "report.csv").open(newline="", encoding="utf-8") as stream:
            report_rows = list(csv.DictReader(stream))
        assert len(report_rows) == 1
        assert report_rows[0]["status"] == "fail"
        assert report_rows[0]["failure_type"] == "quality_gate_failed"
        assert json.loads((config.output_dir / "report.json").read_text(encoding="utf-8")) == report


def test_cuda_memory_gate_failure_resume_reuses_generation_and_repairs_nvml() -> None:
    """A published CUDA gate failure resumes without inference or a new artifact generation."""

    with tempfile.TemporaryDirectory(prefix="whitewater-run-hard-gate-resume-") as tmp:
        shared_backend = _ScriptedBackend([
            (1000.0, None),   # baseline
            (18000.0, None),  # session_create transient: 16.60 GiB incremental
            (12000.0, None),  # steady
            (1100.0, None),   # cleanup
            (900.0, None),    # process_exit
        ])
        config = _config(
            Path(tmp),
            shot_ids=["syn-identity"],
            provider="cuda",
            host_loads=["idle"],
            nvml_backend_factory=lambda: shared_backend,
            hardware={"gpu": "fixture-gpu", "driver": "fixture-driver"},
        )

        first = run_bakeoff(config)
        assert not first.incomplete
        assert first.report is not None
        assert first.report["results"][0]["status"] == "fail"
        assert config.runtime_module.sessions_created >= 1

        state_before = json.loads(config.state_path.read_text(encoding="utf-8"))
        artifact_ref_before = state_before["entries"][0]["artifact_ref"]
        nvml_path = config.output_dir / "nvml.csv"
        nvml_before = nvml_path.read_bytes()
        assert nvml_before

        # Force the report-reuse branch to repair a damaged derivative.  The exact immutable
        # cell generation remains untouched while the canonical CSV is regenerated from it.
        nvml_path.write_bytes(nvml_before[:-1] + b"x")

        config2 = copy.copy(config)
        config2.runtime_module = _FakeRuntime()
        config2.host_load_checkpoint = _RecordingHostLoadCheckpoint()

        def unexpected_measurement(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("a complete failed cell must not be measured again on resume")

        config2.cuda_measurement_runner = unexpected_measurement
        second = run_bakeoff(config2)
        assert not second.incomplete
        assert second.report == first.report
        assert config2.runtime_module.sessions_created == 0
        assert config2.host_load_checkpoint.calls == []

        state_after = json.loads(config.state_path.read_text(encoding="utf-8"))
        assert state_after["entries"][0]["artifact_ref"] == artifact_ref_before
        assert state_after["entries"][0]["artifact_ref"]["attempt_id"] == artifact_ref_before["attempt_id"]
        assert nvml_path.read_bytes() == nvml_before


def _assert_cuda_scoring_failure_published_with_evidence(
    config: RunConfig,
    result: Any,
    expected_message: str,
) -> None:
    assert not result.incomplete
    assert result.report is not None
    validate_report_consistency(
        result.report, config.protocol, config.report_schema, config.corpus, config.corpus_schema,
    )
    cell_result = result.report["results"][0]
    assert cell_result["status"] == "fail"
    assert cell_result["failure"] == {
        "type": "quality_gate_failed",
        "message": expected_message,
        "stage": "metrics",
    }
    # Scoring did not produce a complete metrics disposition.  The measured fields remain useful
    # evidence, but an incomplete metrics mapping must not be emitted as if it were a score.
    assert "metrics" not in cell_result
    for field in (
        "input_frames", "geometry", "timing", "resource", "environment", "conditioning_parameters",
    ):
        assert field in cell_result
    resource = cell_result["resource"]
    assert resource["nvml_samples"]

    state = json.loads(config.state_path.read_text(encoding="utf-8"))
    artifact_ref = state["entries"][0]["artifact_ref"]
    with run_module.ArtifactStore(config.output_dir.resolve() / ".artifacts", state["identity"]) as store:
        manifest = store.load_ref(artifact_ref)
        assert "evidence/nvml_rows.json" in {entry["path"] for entry in manifest["artifacts"]}
        committed_rows = run_module._decode_nvml_rows(
            store.read_artifact(artifact_ref, "evidence/nvml_rows.json"),
            expected_identity={
                "candidate_id": cell_result["candidate_id"],
                "shot_id": cell_result["shot_id"],
                "conditioning_token": cell_result["conditioning_token"],
                "cap_token": cell_result["cap_token"],
                "provider": cell_result["provider"],
                "host_load": cell_result["host_load"],
            },
        )
    assert committed_rows
    assert len(committed_rows) == len(resource["nvml_samples"])
    with (config.output_dir / "nvml.csv").open(newline="", encoding="utf-8") as stream:
        public_rows = list(csv.reader(stream))
    assert tuple(public_rows[0]) == NVML_CSV_HEADER
    assert public_rows[1:] == committed_rows
    assert json.loads((config.output_dir / "report.json").read_text(encoding="utf-8")) == result.report


def test_cuda_dense_scoring_failure_preserves_measurement_evidence_and_publishes_nvml() -> None:
    """A dense scoring failure after CUDA measurement retains exact NVML evidence."""

    with tempfile.TemporaryDirectory(prefix="whitewater-run-dense-score-failure-") as tmp:
        shared_backend = _ScriptedBackend([
            (1000.0, None), (1200.0, None), (1300.0, None), (1100.0, None), (900.0, None),
        ])
        config = _config(
            Path(tmp), shot_ids=["syn-identity"], provider="cuda", host_loads=["idle"],
            nvml_backend_factory=lambda: shared_backend,
            hardware={"gpu": "fixture-gpu", "driver": "fixture-driver"},
        )
        original_dense = run_module.metrics_module.dense_metrics
        original_build_executor = run_module.build_executor
        contexts: list[Any] = []

        def fail_dense(*_args: Any, **_kwargs: Any) -> Any:
            raise run_module.MetricFailure("test_dense", "injected dense scoring failure")

        def capture_build_executor(*args: Any, **kwargs: Any) -> Any:
            executor, context = original_build_executor(*args, **kwargs)
            contexts.append(context)
            return executor, context

        run_module.metrics_module.dense_metrics = fail_dense
        run_module.build_executor = capture_build_executor
        try:
            result = run_bakeoff(config)
        finally:
            run_module.metrics_module.dense_metrics = original_dense
            run_module.build_executor = original_build_executor

        _assert_cuda_scoring_failure_published_with_evidence(
            config, result, "dense metrics could not be computed: test_dense: injected dense scoring failure",
        )
        assert contexts and contexts[0].measured_cell is None
        log_text = (config.output_dir / "runner.log").read_text(encoding="utf-8")
        assert log_text.count("cell start ") == 1


def test_cuda_chain_scoring_failure_preserves_measurement_evidence_and_publishes_nvml() -> None:
    """A chain-drift scoring failure after CUDA measurement retains exact NVML evidence."""

    with tempfile.TemporaryDirectory(prefix="whitewater-run-chain-score-failure-") as tmp:
        shared_backend = _ScriptedBackend([
            (1000.0, None), (1200.0, None), (1300.0, None), (1100.0, None), (900.0, None),
        ])
        config = _config(
            Path(tmp), shot_ids=["syn-chain1"], provider="cuda", host_loads=["idle"],
            nvml_backend_factory=lambda: shared_backend,
            hardware={"gpu": "fixture-gpu", "driver": "fixture-driver"},
        )
        original_chain = run_module.metrics_module.chain_drift_px
        original_build_executor = run_module.build_executor
        contexts: list[Any] = []

        def fail_chain(*_args: Any, **_kwargs: Any) -> Any:
            raise run_module.MetricFailure("test_chain", "injected chain scoring failure")

        def capture_build_executor(*args: Any, **kwargs: Any) -> Any:
            executor, context = original_build_executor(*args, **kwargs)
            contexts.append(context)
            return executor, context

        run_module.metrics_module.chain_drift_px = fail_chain
        run_module.build_executor = capture_build_executor
        try:
            result = run_bakeoff(config)
        finally:
            run_module.metrics_module.chain_drift_px = original_chain
            run_module.build_executor = original_build_executor

        _assert_cuda_scoring_failure_published_with_evidence(
            config, result, "chain drift could not be computed: test_chain: injected chain scoring failure",
        )
        assert contexts and contexts[0].measured_cell is None
        log_text = (config.output_dir / "runner.log").read_text(encoding="utf-8")
        assert log_text.count("cell start ") == 1


def test_cuda_chain_error_after_measurement_preserves_evidence_and_publishes_nvml() -> None:
    """A returned chain-link error retains completed base/NVML measurement evidence."""

    with tempfile.TemporaryDirectory(prefix="whitewater-run-chain-error-") as tmp:
        shared_backend = _ScriptedBackend([
            (1000.0, None), (1200.0, None), (1300.0, None), (1100.0, None), (900.0, None),
        ])
        config = _config(
            Path(tmp), shot_ids=["syn-chain1"], provider="cuda", host_loads=["idle"],
            nvml_backend_factory=lambda: shared_backend,
            hardware={"gpu": "fixture-gpu", "driver": "fixture-driver"},
        )
        original_inference = run_module._perform_cell_inference
        original_build_executor = run_module.build_executor
        contexts: list[Any] = []

        def return_chain_error(*args: Any, **kwargs: Any) -> Any:
            payload = original_inference(*args, **kwargs)
            payload["chain_error"] = "injected chain-link inference failure"
            return payload

        def capture_build_executor(*args: Any, **kwargs: Any) -> Any:
            executor, context = original_build_executor(*args, **kwargs)
            contexts.append(context)
            return executor, context

        run_module._perform_cell_inference = return_chain_error
        run_module.build_executor = capture_build_executor
        try:
            result = run_bakeoff(config)
        finally:
            run_module._perform_cell_inference = original_inference
            run_module.build_executor = original_build_executor

        _assert_cuda_scoring_failure_published_with_evidence(
            config, result, "chain drift could not be computed: injected chain-link inference failure",
        )
        assert contexts and contexts[0].measured_cell is None
        log_text = (config.output_dir / "runner.log").read_text(encoding="utf-8")
        assert log_text.count("cell start ") == 1


def test_missing_artifact_map_entry_raises_typed_driver_failure() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-missing-artifact-") as tmp:
        config = _config(Path(tmp), shot_ids=["syn-identity"])
        config.artifact_map = {}
        try:
            run_bakeoff(config)
        except DriverFailure as failure:
            assert failure.kind == "artifact_map_missing"
        else:
            raise AssertionError("expected DriverFailure(artifact_map_missing)")
        # No half-published report should exist after a startup failure.
        assert not (config.output_dir / "report.json").exists()


def test_manifest_candidate_id_mismatch_is_a_typed_driver_failure() -> None:
    """A selected candidate id whose manifest declares a *different* candidate id must fail
    startup rather than silently binding the wrong artifact's hashes into the resume identity.
    """

    with tempfile.TemporaryDirectory(prefix="whitewater-run-bad-manifest-") as tmp:
        directory = Path(tmp)
        protocol = _protocol()
        # Register a second protocol candidate id that points at the SAME fixture manifest,
        # whose own "candidate.id" field is still "fixture-candidate" -- a genuine mismatch.
        mismatched_id = "other-candidate"
        protocol["candidate_ids"].append({"id": mismatched_id, "role": "shipping-candidate"})
        candidates = _candidate_entries()
        mismatched_entry = copy.deepcopy(candidates[0])
        mismatched_entry["candidate_id"] = mismatched_id
        candidates.append(mismatched_entry)

        output_dir = directory / "output"
        config = RunConfig(
            protocol=protocol,
            corpus=_corpus(),
            candidate_entries=candidates,
            selection=_selection(shot_ids=["syn-identity"]),
            artifact_map={},
            report_schema=REPORT_SCHEMA,
            corpus_schema=CORPUS_SCHEMA,
            output_dir=output_dir,
            state_path=output_dir / "state.json",
            device_index=0,
            poll_interval_s=0.01,
            chain_offsets=(1, 2, 4, 8),
            report_metadata=_report_metadata(),
            protocol_path=V2_PROTOCOL_PATH,
            replace=False,
            runtime_module=_FakeRuntime(),
            array_module=_FakeArrays(),
            nvml_backend_factory=None,
            exr_decoder=_fake_exr_decoder(8, 6),
        )
        config.selection = {**config.selection, "candidate_ids": [mismatched_id]}
        artifact_map = _artifact_map(directory)
        config.artifact_map = {mismatched_id: artifact_map[CANDIDATE_ID]}
        try:
            run_bakeoff(config)
        except DriverFailure as failure:
            assert failure.kind == "candidate_artifact_invalid"
        else:
            raise AssertionError("expected DriverFailure(candidate_artifact_invalid)")


# --------------------------------------------------------------------------------------------
# Fix A: profile + evaluator bound into the resume identity; stale report.json is rejected.
# --------------------------------------------------------------------------------------------


def test_identity_differs_between_profiles_for_an_otherwise_identical_selection() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-identity-") as tmp:
        directory = Path(tmp)
        protocol = _protocol()
        corpus = _corpus()
        selection_axes = {
            "candidate_ids": [CANDIDATE_ID], "shot_ids": ["syn-identity"],
            "conditioning_tokens": ["native-clamp01-v1"], "cap_tokens": ["mp0_5"],
            "providers": [{"token": "cpu", "host_loads": ["not_applicable"]}],
        }
        plan = build_matrix(protocol, corpus, _candidate_entries(), selection_axes, "smoke", "el8-x86_64")
        artifact_map = _artifact_map(directory)
        artifacts = run_module._validate_selected_artifacts(plan, artifact_map, V2_PROTOCOL_PATH)
        runner_section = _report_metadata()["runner"]
        hardware = {"platform": "linux", "architecture": "x86_64"}

        identity_smoke = run_module._compute_identity(
            protocol, corpus, plan, "el8-x86_64", "smoke", artifacts, runner_section, hardware, (1, 2, 4, 8),
            candidate_entries=_candidate_entries(), report_schema=REPORT_SCHEMA, corpus_schema=CORPUS_SCHEMA,
            device_index=0, poll_interval_s=0.05, nvml_enabled=False,
        )
        identity_screen = run_module._compute_identity(
            protocol, corpus, plan, "el8-x86_64", "screen", artifacts, runner_section, hardware, (1, 2, 4, 8),
            candidate_entries=_candidate_entries(), report_schema=REPORT_SCHEMA, corpus_schema=CORPUS_SCHEMA,
            device_index=0, poll_interval_s=0.05, nvml_enabled=False,
        )
        # matrix_sha256 itself does not encode profile -- this is exactly the gap Fix A closes.
        assert identity_smoke["matrix_sha256"] == identity_screen["matrix_sha256"]
        assert canonical_sha256(identity_smoke) != canonical_sha256(identity_screen)

        identity_diff_evaluator = run_module._compute_identity(
            protocol, corpus, plan, "el8-x86_64", "smoke", artifacts,
            {**runner_section, "evaluator_sha256": "f" * 64}, hardware, (1, 2, 4, 8),
            candidate_entries=_candidate_entries(), report_schema=REPORT_SCHEMA, corpus_schema=CORPUS_SCHEMA,
            device_index=0, poll_interval_s=0.05, nvml_enabled=False,
        )
        assert canonical_sha256(identity_smoke) != canonical_sha256(identity_diff_evaluator)


def test_run_spec_builder_returns_compact_complete_identity() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-spec-driver-") as tmp:
        directory = Path(tmp)
        protocol = _protocol()
        corpus = _corpus()
        selection_axes = {
            "candidate_ids": [CANDIDATE_ID], "shot_ids": ["syn-identity"],
            "conditioning_tokens": ["native-clamp01-v1"], "cap_tokens": ["mp0_5"],
            "providers": [{"token": "cpu", "host_loads": ["not_applicable"]}],
        }
        plan = build_matrix(protocol, corpus, _candidate_entries(), selection_axes, "smoke", "el8-x86_64")
        artifacts = run_module._validate_selected_artifacts(plan, _artifact_map(directory), V2_PROTOCOL_PATH)
        runner = dict(_report_metadata()["runner"])
        runner.pop("name")
        runner.pop("version")
        runner.pop("source_commit")
        spec = run_module._build_run_spec(
            protocol, corpus, plan, "el8-x86_64", "smoke", artifacts,
            runner, _report_metadata()["hardware"], (1, 2, 4, 8),
            candidate_entries=_candidate_entries(), report_schema=REPORT_SCHEMA,
            corpus_schema=CORPUS_SCHEMA, device_index=0, poll_interval_s=0.05,
            nvml_enabled=False,
            report_inputs={"warnings": ["operator warning"], "summary": {"final_quality_score": 80.0}},
        )
        assert isinstance(spec, RunSpec)
        stable = spec.stable_inputs
        assert stable["protocol"]["protocol_id"] == protocol["protocol_id"]
        assert stable["protocol"]["sha256"] == canonical_sha256(protocol)
        assert "analysis_caps" not in stable["protocol"]
        assert stable["corpus"]["sha256"] == canonical_sha256(corpus)
        assert stable["matrix"] == plan.selector
        assert stable["artifacts"][CANDIDATE_ID]["artifact_sha256"]
        assert stable["runner"]["name"] == "ww-bakeoff"
        assert stable["runner"]["version"] == "0.1.0"
        assert "command" not in stable["runner"]
        assert stable["report_inputs"]["warnings"] == ["operator warning"]
        assert stable["report_inputs"]["summary"]["final_quality_score"] == 80.0


def test_chain_offsets_are_sorted_unique_before_execution_and_identity() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-chain-offsets-") as tmp:
        config = _config(Path(tmp), shot_ids=["syn-identity"], chain_offsets=[8, 4, 4, 1])
        assert config.chain_offsets == (1, 4, 8)

        protocol = _protocol()
        corpus = _corpus()
        selection_axes = {
            "candidate_ids": [CANDIDATE_ID], "shot_ids": ["syn-identity"],
            "conditioning_tokens": ["native-clamp01-v1"], "cap_tokens": ["mp0_5"],
            "providers": [{"token": "cpu", "host_loads": ["not_applicable"]}],
        }
        plan = build_matrix(protocol, corpus, _candidate_entries(), selection_axes, "smoke", "el8-x86_64")
        artifacts = run_module._validate_selected_artifacts(plan, _artifact_map(Path(tmp)), V2_PROTOCOL_PATH)
        runner = _report_metadata()["runner"]
        hardware = _report_metadata()["hardware"]
        first = run_module._compute_identity(
            protocol, corpus, plan, "el8-x86_64", "smoke", artifacts, runner, hardware, (8, 4, 4, 1),
            candidate_entries=_candidate_entries(), report_schema=REPORT_SCHEMA, corpus_schema=CORPUS_SCHEMA,
            device_index=0, poll_interval_s=0.05, nvml_enabled=False,
        )
        second = run_module._compute_identity(
            protocol, corpus, plan, "el8-x86_64", "smoke", artifacts, runner, hardware, (1, 4, 8),
            candidate_entries=_candidate_entries(), report_schema=REPORT_SCHEMA, corpus_schema=CORPUS_SCHEMA,
            device_index=0, poll_interval_s=0.05, nvml_enabled=False,
        )
        assert first["chain_offsets"] == [1, 4, 8]
        assert first == second
        assert canonical_sha256(first) == canonical_sha256(second)


def test_runner_and_hardware_metadata_preflight_is_strict_and_normalized() -> None:
    metadata = _report_metadata()
    uppercase_runner = copy.deepcopy(metadata["runner"])
    uppercase_runner["source_commit"] = uppercase_runner["source_commit"].upper()
    uppercase_runner["evaluator_sha256"] = uppercase_runner["evaluator_sha256"].upper()
    uppercase_runner["runtime_sha256"] = uppercase_runner["runtime_sha256"].upper()
    normalized_runner = run_module._normalise_runner_metadata(uppercase_runner)
    assert normalized_runner["source_commit"] == "6" * 40
    assert normalized_runner["evaluator_sha256"] == "7" * 64
    assert normalized_runner["runtime_sha256"] == "8" * 64

    base_hardware = run_module._normalise_hardware(metadata["hardware"])
    absent_optional = run_module._normalise_hardware({**metadata["hardware"], "gpu": None, "driver": ""})
    assert absent_optional == base_hardware
    assert "gpu" not in absent_optional and "driver" not in absent_optional

    for section, unknown_key in (("runner", "unexpected_runner"), ("hardware", "unexpected_hardware")):
        with tempfile.TemporaryDirectory(prefix=f"whitewater-run-metadata-{section}-") as tmp:
            config = _config(Path(tmp), shot_ids=["syn-identity"])
            changed = copy.deepcopy(config.report_metadata)
            changed[section][unknown_key] = "must be rejected"
            config.report_metadata = changed
            runtime = config.runtime_module
            try:
                run_bakeoff(config)
            except DriverFailure as exc:
                assert exc.kind == "report_metadata_unknown"
            else:
                raise AssertionError(f"unknown {section} metadata must fail during preflight")
            assert runtime.sessions_created == 0, "metadata failure must happen before any cell executes"

    for key, value in (("evaluator_sha256", "not-a-sha"), ("runtime_sha256", ""), ("source_commit", "xyz")):
        with tempfile.TemporaryDirectory(prefix=f"whitewater-run-metadata-invalid-{key}-") as tmp:
            config = _config(Path(tmp), shot_ids=["syn-identity"])
            changed = copy.deepcopy(config.report_metadata)
            changed["runner"][key] = value
            config.report_metadata = changed
            try:
                run_bakeoff(config)
            except DriverFailure as exc:
                assert exc.kind == "report_metadata_value"
            else:
                raise AssertionError(f"invalid runner {key} must fail during preflight")


def test_protocol_incomplete_corpus_fails_before_any_cell_executes() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-corpus-preflight-") as tmp:
        config = _config(Path(tmp), shot_ids=["syn-identity"])
        runtime = config.runtime_module
        # Schema-valid, and sufficient for the selected matrix cell, but missing the protocol's
        # chain-1 case. This is the exact late-publication defect the P25-6 handoff shipped.
        config.corpus = copy.deepcopy(config.corpus)
        config.corpus["partitions"][0]["shots"] = [
            config.corpus["partitions"][0]["shots"][0]
        ]
        try:
            run_bakeoff(config)
        except DriverFailure as exc:
            assert exc.kind == "corpus_invalid"
            assert "synthetic category 'chain' has no shot" in str(exc)
        else:
            raise AssertionError("protocol-incomplete corpus must fail during preflight")
        assert runtime.sessions_created == 0
        assert not config.state_path.exists()


def test_unmaterialized_production_path_fails_before_any_cell_executes() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-corpus-placeholder-") as tmp:
        # The placeholder is deliberately on an unselected production record. Reports bind the
        # complete corpus, so even a synthetic-only invocation must reject an unmaterialized
        # operator template before creating a session or resume state.
        config = _config(Path(tmp), shot_ids=["syn-identity"])
        runtime = config.runtime_module
        config.corpus = copy.deepcopy(config.corpus)
        config.corpus["partitions"][1]["shots"][0]["path_pattern"] = (
            "/REPLACE_WITH_ON_BOX_ABSOLUTE_PATH/plate.%04d.exr"
        )
        try:
            run_bakeoff(config)
        except DriverFailure as exc:
            assert exc.kind == "corpus_invalid"
            assert "still contains the operator placeholder 'REPLACE_WITH'" in str(exc)
        else:
            raise AssertionError("unmaterialized production path must fail during preflight")
        assert runtime.sessions_created == 0
        assert not config.state_path.exists()


def test_driver_validates_the_immutable_corpus_only_once() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-corpus-once-") as tmp:
        config = _config(Path(tmp), shot_ids=["syn-identity"])
        original = validator_module.validate_corpus_consistency
        calls = 0

        def counted(corpus, protocol, corpus_schema) -> None:
            nonlocal calls
            calls += 1
            original(corpus, protocol, corpus_schema)

        # run.py holds the preflight binding; report validation resolves the validator module's
        # binding. Patch both to catch any accidental publication-time repeat.
        run_original = run_module.validate_corpus_consistency
        run_module.validate_corpus_consistency = counted
        validator_module.validate_corpus_consistency = counted
        try:
            run_bakeoff(config)
        finally:
            run_module.validate_corpus_consistency = run_original
            validator_module.validate_corpus_consistency = original
        assert calls == 1


def test_legacy_v1_resume_state_is_rejected_with_fresh_replace_diagnostic() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-legacy-state-") as tmp:
        config = _config(Path(tmp), shot_ids=["syn-identity"])
        config.output_dir.mkdir(parents=True)
        legacy_state = {
            "schema_version": 1,
            "identity": {},
            "identity_sha256": canonical_sha256({}),
            "entries": [],
        }
        config.state_path.write_text(json.dumps(legacy_state), encoding="utf-8")
        runtime = config.runtime_module
        try:
            run_bakeoff(config)
        except DriverFailure as exc:
            assert exc.kind == "legacy_resume_state"
            diagnostic = str(exc).lower()
            assert "fresh" in diagnostic and "replace" in diagnostic
        else:
            raise AssertionError("schema-v1 state must be rejected without migration")
        assert runtime.sessions_created == 0, "legacy state rejection must happen before any cell executes"


def test_rerun_with_different_profile_same_state_path_is_rejected_not_reused() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-smoke-then-screen-") as tmp:
        config = _config(Path(tmp), shot_ids=["syn-identity"])
        first = run_bakeoff(config)
        assert not first.incomplete
        original_report_bytes = (config.output_dir / "report.json").read_bytes()

        config2 = copy.copy(config)
        config2.selection = {**config.selection, "profile": "screen"}
        config2.runtime_module = _FakeRuntime()
        try:
            run_bakeoff(config2)
        except Exception as exc:  # noqa: BLE001 - asserting rejection, not one exact exception type
            assert "identity" in str(exc).lower()
        else:
            raise AssertionError("expected the screen rerun to be rejected, not silently reused")

        # The original smoke report.json must be untouched -- never silently overwritten with,
        # or presented as, a screen-profile result.
        assert (config.output_dir / "report.json").read_bytes() == original_report_bytes


def test_changed_report_warnings_and_summary_are_rejected_on_reuse() -> None:
    for field, changed_value in (
        ("warnings", ["changed operator warning"]),
        ("summary", {"final_quality_score": 82.0}),
    ):
        with tempfile.TemporaryDirectory(prefix=f"whitewater-run-report-input-{field}-") as tmp:
            config = _config(Path(tmp), shot_ids=["syn-identity"])
            baseline_metadata = copy.deepcopy(config.report_metadata)
            baseline_metadata["warnings"] = ["baseline operator warning"]
            baseline_metadata["summary"] = {"final_quality_score": 81.0}
            config.report_metadata = baseline_metadata
            first = run_bakeoff(config)
            assert not first.incomplete

            config2 = copy.copy(config)
            config2.runtime_module = _FakeRuntime()
            changed_metadata = copy.deepcopy(baseline_metadata)
            changed_metadata[field] = changed_value
            config2.report_metadata = changed_metadata
            try:
                run_bakeoff(config2)
            except DriverFailure as exc:
                assert "identity" in str(exc).lower()
            else:
                raise AssertionError(f"changed report {field} must not reuse completed output")


def test_stale_report_json_under_a_different_identity_is_explicitly_rejected() -> None:
    """A different --state path in the SAME --output-dir bypasses resume.load_state's own
    identity check (a narrower, state-file-scoped guard); Fix A's explicit report.json check
    must still catch the stale report rather than silently returning it."""

    with tempfile.TemporaryDirectory(prefix="whitewater-run-stale-report-") as tmp:
        config = _config(Path(tmp), shot_ids=["syn-identity"])
        first = run_bakeoff(config)
        assert not first.incomplete

        config2 = copy.copy(config)
        config2.selection = {**config.selection, "profile": "screen"}
        config2.state_path = config.output_dir / "state-screen.json"
        config2.runtime_module = _FakeRuntime()
        try:
            run_bakeoff(config2)
        except DriverFailure as failure:
            assert failure.kind == "report_identity_mismatch"
        else:
            raise AssertionError("expected DriverFailure(report_identity_mismatch)")


# --------------------------------------------------------------------------------------------
# Fix B: supervised CUDA host-load boundary.
# --------------------------------------------------------------------------------------------


def test_host_load_checkpoint_called_once_per_group_in_order() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-hostload-") as tmp:
        checkpoint = _RecordingHostLoadCheckpoint()
        config = _config(
            Path(tmp), shot_ids=["syn-identity"], provider="cuda", host_loads=["idle", "live_flame"],
            nvml_backend_factory=lambda: _FakeNvmlBackend(),
            hardware={"gpu": "fixture-gpu", "driver": "fixture-driver"},
            host_load_checkpoint=checkpoint,
        )
        result = run_bakeoff(config)
        assert not result.incomplete
        assert len(result.report["results"]) == 2
        assert checkpoint.calls == ["idle", "live_flame"]


def test_host_load_checkpoint_does_not_refire_for_a_repeated_host_load() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-hostload-repeat-") as tmp:
        checkpoint = _RecordingHostLoadCheckpoint()
        config = _config(
            Path(tmp), shot_ids=["syn-identity", "syn-chain1"], provider="cuda", host_loads=["idle"],
            nvml_backend_factory=lambda: _FakeNvmlBackend(),
            hardware={"gpu": "fixture-gpu", "driver": "fixture-driver"},
            host_load_checkpoint=checkpoint,
        )
        result = run_bakeoff(config)
        assert not result.incomplete
        assert len(result.report["results"]) == 2
        assert checkpoint.calls == ["idle"]


def test_cpu_cells_never_trigger_a_host_load_checkpoint() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-hostload-cpu-") as tmp:
        checkpoint = _RecordingHostLoadCheckpoint()
        config = _config(Path(tmp), shot_ids=["syn-identity"], host_load_checkpoint=checkpoint)
        result = run_bakeoff(config)
        assert not result.incomplete
        assert checkpoint.calls == []


# --------------------------------------------------------------------------------------------
# Fix C: blinded, per-candidate preview content (not just per-candidate paths).
# --------------------------------------------------------------------------------------------


def test_two_candidates_blinded_previews_differ_and_review_csv_references_them() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-blind-") as tmp:
        directory = Path(tmp)
        candidate_a, candidate_b = "candidate-a", "candidate-b"
        entry_a = _artifact_map_entry_for(directory, candidate_a, "a")
        entry_b = _artifact_map_entry_for(directory, candidate_b, "b")
        protocol = _protocol()
        protocol["candidate_ids"] = [
            {"id": candidate_a, "role": "shipping-candidate"},
            {"id": candidate_b, "role": "shipping-candidate"},
        ]
        candidates = [_candidate_entry(candidate_a), _candidate_entry(candidate_b)]
        output_dir = directory / "output"
        config = RunConfig(
            protocol=protocol,
            corpus=_corpus(),
            candidate_entries=candidates,
            selection={
                "profile": "smoke", "environment": "el8-x86_64",
                "candidate_ids": [candidate_a, candidate_b],
                "conditioning_tokens": ["native-clamp01-v1"], "cap_tokens": ["mp0_5"],
                "providers": [{"token": "cpu", "host_loads": ["not_applicable"]}],
                "shot_ids": ["prod-sample"],
            },
            artifact_map={candidate_a: entry_a, candidate_b: entry_b},
            report_schema=REPORT_SCHEMA, corpus_schema=CORPUS_SCHEMA,
            output_dir=output_dir, state_path=output_dir / "state.json",
            device_index=0, poll_interval_s=0.01, chain_offsets=(1, 2, 4, 8),
            report_metadata=_report_metadata(), protocol_path=V2_PROTOCOL_PATH, replace=False,
            runtime_module=_FakeRuntime(path_dependent=True), array_module=_FakeArrays(),
            nvml_backend_factory=None, exr_decoder=_fake_exr_decoder(8, 6),
        )
        result = run_bakeoff(config)
        assert not result.incomplete
        assert {r["candidate_id"] for r in result.report["results"]} == {candidate_a, candidate_b}

        review_path = config.output_dir / "review.csv"
        with review_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        assert len(rows) == 3  # header + one row per candidate
        labels = [row[0] for row in rows[1:]]
        assert len(set(labels)) == 2  # distinct blinded labels
        assert candidate_a not in labels and candidate_b not in labels
        for label in labels:
            assert candidate_a not in label and candidate_b not in label

        preview_paths = {row[7] for row in rows[1:]}
        assert len(preview_paths) == 2
        warped_bytes = []
        for path in preview_paths:
            warped_file = Path(path) / "offset_1_warped.pfm"
            assert warped_file.is_file()
            warped_bytes.append(warped_file.read_bytes())
        assert warped_bytes[0] != warped_bytes[1]


# --------------------------------------------------------------------------------------------
# Fix D: a first-run transient inside the peak window, and a genuine post-exit reading.
# --------------------------------------------------------------------------------------------


def test_cuda_peak_captures_a_first_run_transient_a_boundary_only_design_would_miss() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-transient-") as tmp:
        # baseline, session_create(+first inference), steady, cleanup, [exit reading]. The
        # 9000 spike appears ONLY at the session_create poll entry -- a design that samples
        # only baseline/cleanup boundaries (1000.0 / 1100.0) would never observe it.
        shared_backend = _ScriptedBackend([
            (1000.0, None), (9000.0, None), (1200.0, None), (1100.0, None), (900.0, None),
        ])
        config = _config(
            Path(tmp), shot_ids=["syn-identity"], provider="cuda", host_loads=["idle"],
            nvml_backend_factory=lambda: shared_backend,
            hardware={"gpu": "fixture-gpu", "driver": "fixture-driver"},
        )
        result = run_bakeoff(config)
        assert not result.incomplete
        resource = result.report["results"][0]["resource"]
        assert resource["peak_device_memory_mib"] == 9000.0
        assert resource["baseline_device_memory_mib"] == 1000.0
        # process_exit is a genuinely distinct, later reading -- not the "cleanup" boundary
        # value (1100.0) and not the in-process peak (9000.0) reused as a stand-in.
        exit_samples = [s for s in resource["nvml_samples"] if s["stage"] == "process_exit"]
        assert len(exit_samples) == 1
        assert exit_samples[0]["used_mib"] == 900.0
        cleanup_samples = [s for s in resource["nvml_samples"] if s["stage"] == "cleanup"]
        assert cleanup_samples[0]["used_mib"] == 1100.0


def test_real_subprocess_cuda_measurement_runner_isolates_work_and_reads_post_exit() -> None:
    """Exercises the REAL, fork-based default runner directly (not through the full driver),
    proving the actual subprocess plumbing (queue hand-off, join-before-reading, a fresh
    post-exit device query) works end to end. Every argument is plain/picklable so nothing here
    depends on numpy/onnxruntime/pynvml/a GPU."""

    shared_backend = _ScriptedBackend([(1000.0, None), (2000.0, None), (1500.0, None), (700.0, None)])

    def work(stage_sampler) -> dict[str, Any]:
        with stage_sampler("session_create"):
            pass
        return {"ok": True, "value": 42}

    result = run_module.run_cuda_measurement_in_subprocess(
        work, lambda: shared_backend, 0, 0.01,
    )
    assert result.payload == {"ok": True, "value": 42}
    stages = [sample["stage"] for sample in result.samples]
    assert stages == ["baseline", "session_create", "cleanup"]
    assert result.process_exit_used_mib is not None


# --------------------------------------------------------------------------------------------
# Fix E: nvml.csv/review.csv are regenerated from durable state, never duplicated on resume.
# --------------------------------------------------------------------------------------------


def test_interrupted_cell_resume_does_not_duplicate_bundle_evidence_rows() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-interrupt-") as tmp:
        config = _config(
            Path(tmp), shot_ids=["syn-identity"], provider="cuda", host_loads=["idle"],
            nvml_backend_factory=lambda: _FakeNvmlBackend(),
            hardware={"gpu": "fixture-gpu", "driver": "fixture-driver"},
        )
        selection_axes = {
            key: config.selection[key]
            for key in ("candidate_ids", "shot_ids", "conditioning_tokens", "cap_tokens", "providers")
        }
        plan = build_matrix(
            config.protocol, config.corpus, config.candidate_entries, selection_axes,
            config.selection["profile"], config.selection["environment"],
        )
        artifacts = run_module._validate_selected_artifacts(plan, config.artifact_map, config.protocol_path)
        hardware = run_module._default_hardware()
        hardware.update(config.report_metadata.get("hardware", {}))
        runner_section = dict(config.report_metadata["runner"])
        identity = run_module._compute_identity(
            config.protocol, config.corpus, plan, config.selection["environment"], config.selection["profile"],
            artifacts, runner_section, hardware, config.chain_offsets,
            candidate_entries=config.candidate_entries, report_schema=config.report_schema, corpus_schema=config.corpus_schema,
            device_index=config.device_index, poll_interval_s=config.poll_interval_s,
            nvml_enabled=config.nvml_backend_factory is not None,
        )
        from .resume import create_state
        run_module._ensure_output_dir(config.output_dir)
        create_state(config.state_path, identity, plan)
        cell = plan.cells[0]
        mark_in_progress(config.state_path, identity, plan, cell)

        executor, _ctx = run_module.build_executor(
            protocol=config.protocol, corpus=config.corpus, profile=config.selection["profile"],
            artifacts=artifacts, runtime_module=config.runtime_module, array_module=config.array_module,
            nvml_backend_factory=config.nvml_backend_factory, device_index=config.device_index,
            poll_interval_s=config.poll_interval_s, chain_offsets=config.chain_offsets,
            exr_decoder=config.exr_decoder, review_enabled=True,
            host_load_checkpoint=lambda host_load: None,
            cuda_measurement_runner=config.cuda_measurement_runner,
        )
        # The executor is pure: running it creates no public evidence files. Simulate
        # an interrupted state transition by committing its exact bundle, but leaving state.json
        # with this cell still in_progress before RunCoordinator can mark it complete.
        bundle = executor(cell)
        # Deliberately include both a duplicate-looking sample and a unique row from this
        # abandoned generation. Regeneration must preserve duplicates in the selected exact
        # generation while excluding every row that belongs only to this interrupted attempt.
        assert bundle.nvml_rows
        abandoned_row = list(bundle.nvml_rows[0])
        abandoned_row[8] = "999999.0"
        stale_bundle = run_module.CellBundle(
            result=bundle.public_result(),
            nvml_rows=(*bundle.nvml_rows, bundle.nvml_rows[0], tuple(abandoned_row)),
            previews=bundle.previews,
            log_messages=bundle.log_messages,
        )
        before = {
            path.relative_to(config.output_dir)
            for path in config.output_dir.rglob("*")
            if ".artifacts" not in path.relative_to(config.output_dir).parts
        }
        assert not (config.output_dir / "review-previews").exists()
        with run_module.ArtifactStore(
            config.output_dir.resolve() / ".artifacts", identity,
        ) as store:
            stale_execution = run_module._commit_cell_bundle(
                store, cell, stale_bundle, nvml_enabled=True,
            )
            stale_rows = run_module._decode_nvml_rows(
                store.read_artifact(stale_execution.artifact_ref, "evidence/nvml_rows.json")
            )
        assert len(stale_rows) > len({tuple(row) for row in stale_rows})
        assert any(row[8] == "999999.0" for row in stale_rows)
        after = {
            path.relative_to(config.output_dir)
            for path in config.output_dir.rglob("*")
            if ".artifacts" not in path.relative_to(config.output_dir).parts
        }
        assert after == before
        assert stale_rows, "the interrupted attempt should have committed exact NVML evidence"

        # A fresh run_bakeoff call now resumes: resume.load_state recovers the in_progress cell
        # to pending (documented behavior) and RunCoordinator re-executes it, which must
        # Regeneration follows only the completed state's exact ref and must not append the
        # abandoned generation's rows to the retried cell's current generation.
        result = run_bakeoff(config)
        assert not result.incomplete
        nvml_path = config.output_dir / "nvml.csv"
        with nvml_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        data_rows = rows[1:]
        state_after = json.loads(config.state_path.read_text(encoding="utf-8"))
        current_ref = state_after["entries"][0]["artifact_ref"]
        with run_module.ArtifactStore(
            config.output_dir.resolve() / ".artifacts", identity,
        ) as store:
            current_rows = run_module._decode_nvml_rows(
                store.read_artifact(current_ref, "evidence/nvml_rows.json")
            )
        assert data_rows == current_rows, "regeneration must use only the completed exact generation"
        assert any(tuple(row) == tuple(current_rows[0]) for row in data_rows)
        assert not any(row[8] == "999999.0" for row in data_rows)


def test_regeneration_reads_old_exact_ref_after_current_pointer_advances() -> None:
    """Public evidence follows the state-held generation, never a newer current pointer."""

    with tempfile.TemporaryDirectory(prefix="whitewater-run-exact-ref-publication-") as tmp:
        config = _config(
            Path(tmp), shot_ids=["prod-sample"], provider="cuda", host_loads=["idle"],
            nvml_backend_factory=lambda: _FakeNvmlBackend(),
            hardware={"gpu": "fixture-gpu", "driver": "fixture-driver"},
        )
        first = run_bakeoff(config)
        assert not first.incomplete
        state = json.loads(config.state_path.read_text(encoding="utf-8"))
        identity = state["identity"]
        entry_a = state["entries"][0]
        ref_a = entry_a["artifact_ref"]
        result_a = entry_a["result"]
        with (config.output_dir / "review.csv").open(newline="", encoding="utf-8") as stream:
            review_a = list(csv.reader(stream))[1:]
        assert review_a and review_a[0][7]
        preview_path_a = review_a[0][7]

        selection_axes = {
            key: config.selection[key]
            for key in ("candidate_ids", "shot_ids", "conditioning_tokens", "cap_tokens", "providers")
        }
        plan = build_matrix(
            config.protocol, config.corpus, config.candidate_entries, selection_axes,
            config.selection["profile"], config.selection["environment"],
        )
        cell = plan.cells[0]
        with run_module.ArtifactStore(config.output_dir.resolve() / ".artifacts", identity) as store:
            rows_a = run_module._decode_nvml_rows(
                store.read_artifact(ref_a, "evidence/nvml_rows.json")
            )
            rows_b = [list(row) for row in rows_a]
            rows_b[0][8] = "999998.0"
            execution_b = run_module._commit_cell_bundle(
                store,
                cell,
                run_module.CellBundle(
                    result=result_a,
                    nvml_rows=tuple(tuple(row) for row in rows_b),
                    previews=(
                        run_module.PreviewPayload("previews/offset_1_warped.pfm", b"current-warped"),
                        run_module.PreviewPayload("previews/offset_1_flow.pfm", b"current-flow"),
                    ),
                ),
                nvml_enabled=True,
            )
            assert execution_b.artifact_ref["attempt_id"] != ref_a["attempt_id"]
            assert store.artifact_path(execution_b.artifact_ref, "previews/offset_1_warped.pfm").parent != Path(preview_path_a)

            run_module._regenerate_public_evidence_outputs(
                store,
                config.output_dir,
                config.corpus,
                plan,
                [{"result": result_a, "artifact_ref": ref_a}],
                replace=True,
                nvml_enabled=True,
                review_enabled=True,
            )

        with (config.output_dir / "nvml.csv").open(newline="", encoding="utf-8") as stream:
            public_nvml_rows = list(csv.reader(stream))[1:]
        assert public_nvml_rows == rows_a
        with (config.output_dir / "review.csv").open(newline="", encoding="utf-8") as stream:
            public_review_rows = list(csv.reader(stream))[1:]
        assert public_review_rows[0][7] == preview_path_a
        assert "999998.0" not in {row[8] for row in public_nvml_rows}


def test_regeneration_rejects_short_or_misordered_completed_records_before_writing() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-evidence-records-") as tmp:
        directory = Path(tmp)
        config = _config(directory, shot_ids=["syn-identity", "syn-chain1"], provider="cpu")
        first = run_bakeoff(config)
        assert not first.incomplete
        state = json.loads(config.state_path.read_text(encoding="utf-8"))
        completed = [
            {"result": entry["result"], "artifact_ref": entry["artifact_ref"]}
            for entry in state["entries"]
        ]
        selection_axes = {
            key: config.selection[key]
            for key in ("candidate_ids", "shot_ids", "conditioning_tokens", "cap_tokens", "providers")
        }
        plan = build_matrix(
            config.protocol, config.corpus, config.candidate_entries, selection_axes,
            config.selection["profile"], config.selection["environment"],
        )
        sentinel = config.output_dir / "nvml.csv"
        sentinel.write_bytes(b"must-not-be-touched\n")
        with run_module.ArtifactStore(config.output_dir.resolve() / ".artifacts", state["identity"]) as store:
            try:
                run_module._regenerate_public_evidence_outputs(
                    store, config.output_dir, config.corpus, plan, completed[:-1],
                    replace=True, nvml_enabled=False,
                )
            except run_module.ArtifactStoreFailure as failure:
                assert failure.kind == "completed_count"
            else:
                raise AssertionError("short completed records must be rejected")

            try:
                run_module._regenerate_public_evidence_outputs(
                    store, config.output_dir, config.corpus, plan, list(reversed(completed)),
                    replace=True, nvml_enabled=False,
                )
            except run_module.ArtifactStoreFailure as failure:
                assert failure.kind == "cell_mismatch"
            else:
                raise AssertionError("misordered completed records must be rejected")
        assert sentinel.read_bytes() == b"must-not-be-touched\n"


# --------------------------------------------------------------------------------------------
# Fix F: nvml.csv/review.csv are repaired (not just left missing) when report.json is reused.
# --------------------------------------------------------------------------------------------


def test_regenerates_missing_sidecar_outputs_when_report_already_published() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-repair-") as tmp:
        config = _config(
            Path(tmp), shot_ids=["syn-identity"], provider="cuda", host_loads=["idle"],
            nvml_backend_factory=lambda: _FakeNvmlBackend(),
            hardware={"gpu": "fixture-gpu", "driver": "fixture-driver"},
        )
        first = run_bakeoff(config)
        assert not first.incomplete
        nvml_path = config.output_dir / "nvml.csv"
        review_path = config.output_dir / "review.csv"
        assert nvml_path.is_file()
        original_nvml_bytes = nvml_path.read_bytes()

        # Simulate a crash (or an operator deleting the CSVs) that left report.json present
        # but its sidecar-derived outputs missing.
        nvml_path.unlink()

        config2 = copy.copy(config)
        config2.runtime_module = _FakeRuntime()
        second = run_bakeoff(config2)
        assert not second.incomplete
        assert second.report == first.report  # report.json itself was still reused, unchanged
        assert nvml_path.is_file(), "nvml.csv must be repaired, not left missing"
        assert nvml_path.read_bytes() == original_nvml_bytes
        # review.csv never existed for this synthetic (analytic-truth) shot in the first place;
        # regeneration must not fabricate one.
        assert not review_path.exists()


# --------------------------------------------------------------------------------------------
# Fix K: sidecar writes are crash-atomic and self-healing (content-verified, not existence-only).
# --------------------------------------------------------------------------------------------


def test_atomic_publish_never_leaves_a_partial_file_on_interruption() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-atomic-") as tmp:
        directory = Path(tmp)
        destination = directory / "output.csv"
        destination.write_bytes(b"original,content\n")
        original_bytes = destination.read_bytes()

        real_fsync = os.fsync

        def failing_fsync(fd):
            raise OSError("simulated interruption during atomic publish")

        os.fsync = failing_fsync
        try:
            try:
                run_module._atomic_publish(destination, b"new,content,that,must,never,land\n", replace_existing=True)
            except DriverFailure as failure:
                assert failure.kind == "atomic_write"
            else:
                raise AssertionError("expected DriverFailure(atomic_write)")
        finally:
            os.fsync = real_fsync

        # The destination must be exactly what it was before the interrupted attempt -- never
        # truncated, never partially overwritten by the staged (but never fsynced/renamed) bytes.
        assert destination.read_bytes() == original_bytes
        # No leftover temp file should remain in the directory either.
        leftovers = [p for p in directory.iterdir() if p.name.startswith(".output.csv.")]
        assert leftovers == [], f"leftover temp files: {leftovers}"


def test_truncated_sidecar_outputs_are_repaired_to_canonical_bytes_and_left_alone_once_correct() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-repair-truncated-") as tmp:
        config = _config(
            Path(tmp), shot_ids=["prod-sample"], provider="cuda", host_loads=["idle"],
            nvml_backend_factory=lambda: _FakeNvmlBackend(),
            hardware={"gpu": "fixture-gpu", "driver": "fixture-driver"},
        )
        first = run_bakeoff(config)
        assert not first.incomplete
        nvml_path = config.output_dir / "nvml.csv"
        review_path = config.output_dir / "review.csv"
        original_nvml_bytes = nvml_path.read_bytes()
        original_review_bytes = review_path.read_bytes()
        assert original_nvml_bytes and original_review_bytes

        # Simulate a partial/interrupted write left behind by an earlier crash -- a truncated
        # nvml.csv (matching codex's repro) and an emptied review.csv.
        nvml_path.write_bytes(b"candidate_id,shot")
        review_path.write_bytes(b"")

        config2 = copy.copy(config)
        config2.runtime_module = _FakeRuntime()
        second = run_bakeoff(config2)
        assert not second.incomplete
        assert nvml_path.read_bytes() == original_nvml_bytes
        assert review_path.read_bytes() == original_review_bytes

        # Idempotence: once both files are already correct, a further resumed run must not even
        # rewrite them -- same inode, same mtime, not merely byte-identical after a rewrite.
        nvml_stat_after_repair = nvml_path.stat()
        review_stat_after_repair = review_path.stat()
        third = run_bakeoff(config2)
        assert not third.incomplete
        nvml_stat_after_idempotent_rerun = nvml_path.stat()
        review_stat_after_idempotent_rerun = review_path.stat()
        assert nvml_stat_after_idempotent_rerun.st_ino == nvml_stat_after_repair.st_ino
        assert nvml_stat_after_idempotent_rerun.st_mtime_ns == nvml_stat_after_repair.st_mtime_ns
        assert review_stat_after_idempotent_rerun.st_ino == review_stat_after_repair.st_ino
        assert review_stat_after_idempotent_rerun.st_mtime_ns == review_stat_after_repair.st_mtime_ns


# --------------------------------------------------------------------------------------------
# Fix G: EVERY inference a CUDA cell needs runs inside the isolated child, never the parent.
# --------------------------------------------------------------------------------------------


def test_cuda_cell_runs_all_inference_in_the_child_not_the_parent() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-child-isolation-") as tmp:
        runtime = _FakeRuntime()
        config = _config(
            Path(tmp), shot_ids=["prod-sample"], provider="cuda", host_loads=["idle"],
            nvml_backend_factory=lambda: _FakeNvmlBackend(),
            hardware={"gpu": "fixture-gpu", "driver": "fixture-driver"},
            runtime_module=runtime,
            # The REAL fork-based runner, not the in-process test fake: this is the whole
            # point of the assertion below.
            cuda_measurement_runner=run_module.run_cuda_measurement_in_subprocess,
        )
        result = run_bakeoff(config)
        assert not result.incomplete
        # This cell needs several inferences (base pair, forward/backward reverse pair, and at
        # least one review-offset pair). Every one of them must have happened inside the forked
        # child: under fork the child gets its own copy-on-write copy of `runtime`, so mutating
        # its sessions_created counter there is invisible back here in the parent. If ANY
        # inference had run in-process (the Fix G bug this guards against), this would be > 0.
        assert runtime.sessions_created == 0

        cell_result = result.report["results"][0]
        assert cell_result["status"] == "pass"
        assert cell_result["provider"] == "cuda"
        # The same metrics/previews a CPU cell would produce are still produced.
        assert "visible_warp_residual" in cell_result["metrics"]
        assert "forward_backward_residual_px" in cell_result["metrics"]
        review_path = config.output_dir / "review.csv"
        assert review_path.is_file()
        with review_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        assert len(rows) == 2
        preview_dir = Path(rows[1][7])
        assert (preview_dir / "offset_1_warped.pfm").is_file()
        assert (preview_dir / "offset_2_warped.pfm").is_file()


# --------------------------------------------------------------------------------------------
# Fix H: a child that exits without queueing a result must not hang the parent.
# --------------------------------------------------------------------------------------------


def _call_with_timeout(fn, timeout_s: float):
    """Run ``fn`` on a background daemon thread and fail loudly instead of hanging the whole
    test suite if it does not return within ``timeout_s``."""

    outcome: dict[str, Any] = {}

    def _runner() -> None:
        try:
            outcome["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the calling thread below
            outcome["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise AssertionError(f"operation did not complete within {timeout_s}s (suspected hang)")
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


def test_child_exit_without_queueing_yields_typed_failure_not_a_hang() -> None:
    def work(stage_sampler):
        # Simulates an OOM-kill/segfault: the child terminates immediately, bypassing Python's
        # exception machinery entirely, so it never reaches queue.put(...).
        os._exit(9)

    def attempt():
        return run_module.run_cuda_measurement_in_subprocess(work, lambda: _FakeNvmlBackend(), 0, 0.01)

    try:
        _call_with_timeout(attempt, 15.0)
    except DriverFailure as failure:
        assert failure.kind in {"out_of_memory", "runtime_error"}
    else:
        raise AssertionError("expected DriverFailure for a child that exited without queueing")


def test_cuda_child_preserves_typed_failure_stage() -> None:
    class _StagedFailure(RuntimeError):
        kind = "out_of_memory"
        stage = "session_create"

    def work(stage_sampler):
        raise _StagedFailure("BFCArena failed to allocate memory")

    try:
        _call_with_timeout(
            lambda: run_module.run_cuda_measurement_in_subprocess(
                work, lambda: _FakeNvmlBackend(), 0, 0.01,
            ),
            15.0,
        )
    except DriverFailure as failure:
        assert failure.kind == "out_of_memory"
        assert failure.stage == "session_create"
    else:
        raise AssertionError("expected typed child failure")


def test_final_cuda_without_nvml_is_rejected_before_state_or_cells() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-final-no-nvml-") as tmp:
        config = _config(
            Path(tmp), shot_ids=["syn-identity"], provider="cuda", host_loads=["idle"],
            nvml_backend_factory=None,
        )
        # Use a smoke-shaped fixture plan only to reach the driver's final/NVML preflight; the
        # real matrix planner separately enforces final coverage before this same guard runs.
        selection_axes = {
            key: config.selection[key]
            for key in ("candidate_ids", "shot_ids", "conditioning_tokens", "cap_tokens", "providers")
        }
        plan = build_matrix(
            config.protocol, config.corpus, config.candidate_entries, selection_axes,
            config.selection["profile"], config.selection["environment"],
        )
        config.selection = {**config.selection, "profile": "final"}
        original_build_matrix = run_module.build_matrix
        run_module.build_matrix = lambda *args, **kwargs: plan
        try:
            try:
                run_bakeoff(config)
            except DriverFailure as failure:
                assert failure.kind == "nvml_required"
            else:
                raise AssertionError("final CUDA without NVML must be rejected before execution")
        finally:
            run_module.build_matrix = original_build_matrix
        assert not config.state_path.exists()
        assert not (config.output_dir / "report.json").exists()
        assert config.host_load_checkpoint.calls == []


def test_final_cuda_missing_nvml_stage_becomes_a_typed_cell_failure() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-final-missing-stage-") as tmp:
        directory = Path(tmp)
        config = _config(
            directory, shot_ids=["syn-identity"], provider="cuda", host_loads=["idle"],
            nvml_backend_factory=lambda: _FakeNvmlBackend(),
        )
        selection_axes = {
            key: config.selection[key]
            for key in ("candidate_ids", "shot_ids", "conditioning_tokens", "cap_tokens", "providers")
        }
        plan = build_matrix(
            config.protocol, config.corpus, config.candidate_entries, selection_axes,
            config.selection["profile"], config.selection["environment"],
        )
        artifacts = run_module._validate_selected_artifacts(plan, config.artifact_map, config.protocol_path)

        def incomplete_measurement(work, nvml_backend_factory, device_index, poll_interval_s):
            complete = _fake_cuda_measurement_runner(
                work, nvml_backend_factory, device_index, poll_interval_s,
            )
            return CudaMeasurementResult(complete.payload, [complete.samples[0]], None, None)

        executor, _context = run_module.build_executor(
            protocol=config.protocol,
            corpus=config.corpus,
            profile="final",
            artifacts=artifacts,
            runtime_module=config.runtime_module,
            array_module=config.array_module,
            nvml_backend_factory=config.nvml_backend_factory,
            device_index=config.device_index,
            poll_interval_s=config.poll_interval_s,
            chain_offsets=config.chain_offsets,
            exr_decoder=config.exr_decoder,
            review_enabled=True,
            host_load_checkpoint=lambda _host_load: None,
            cuda_measurement_runner=incomplete_measurement,
        )
        bundle = executor(plan.cells[0])
        assert bundle.result["status"] == "fail"
        assert bundle.result["failure"]["type"] == "runtime_error"
        assert bundle.result["failure"]["stage"] == "resource"

        identity = {"test": "final-missing-stage"}
        with run_module.ArtifactStore(directory.resolve() / "artifacts", identity) as store:
            execution = run_module._commit_cell_bundle(
                store,
                plan.cells[0],
                bundle,
                nvml_enabled=True,
                require_nvml_stages=True,
            )
            assert execution.result["status"] == "fail"
            assert {entry["path"] for entry in store.load_ref(execution.artifact_ref)["artifacts"]} == {"result.json"}


# --------------------------------------------------------------------------------------------
# Fix I: device/NVML/hardware measurement config is bound into the resume identity.
# --------------------------------------------------------------------------------------------


def test_identity_differs_for_measurement_config_changes() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-identity-measure-") as tmp:
        directory = Path(tmp)
        protocol = _protocol()
        corpus = _corpus()
        selection_axes = {
            "candidate_ids": [CANDIDATE_ID], "shot_ids": ["syn-identity"],
            "conditioning_tokens": ["native-clamp01-v1"], "cap_tokens": ["mp0_5"],
            "providers": [{"token": "cuda", "host_loads": ["idle"]}],
        }
        plan = build_matrix(protocol, corpus, _candidate_entries(), selection_axes, "smoke", "el8-x86_64")
        artifacts = run_module._validate_selected_artifacts(plan, _artifact_map(directory), V2_PROTOCOL_PATH)
        runner_section = _report_metadata()["runner"]
        base_hardware = {"platform": "linux", "architecture": "x86_64", "gpu": "gpu-a", "driver": "driver-1"}

        def identity(hardware=base_hardware, **kwargs):
            resolved = {
                "candidate_entries": _candidate_entries(), "report_schema": REPORT_SCHEMA,
                "corpus_schema": CORPUS_SCHEMA, "device_index": 0, "poll_interval_s": 0.05,
                "nvml_enabled": True, **kwargs,
            }
            return run_module._compute_identity(
                protocol, corpus, plan, "el8-x86_64", "smoke", artifacts, runner_section, hardware, (1, 2, 4, 8),
                **resolved,
            )

        baseline = canonical_sha256(identity())
        assert baseline != canonical_sha256(identity(nvml_enabled=False))
        assert baseline != canonical_sha256(identity(device_index=1))
        assert baseline != canonical_sha256(identity(poll_interval_s=0.1))
        assert baseline != canonical_sha256(identity(hardware={**base_hardware, "gpu": "gpu-b"}))
        assert baseline != canonical_sha256(identity(hardware={**base_hardware, "driver": "driver-2"}))


def test_identity_binds_protocol_content_not_just_protocol_id() -> None:
    # The matrix selector carries only cap/conditioning TOKENS; the definitions behind them live
    # in the protocol and are result-affecting. Editing a cap's decimal_megapixels (which drives
    # analysis geometry, hence every metric) while keeping protocol_id must change the identity,
    # so it can never be silently reused/resumed against the same --state/--output-dir.
    with tempfile.TemporaryDirectory(prefix="whitewater-run-identity-protocol-") as tmp:
        directory = Path(tmp)
        protocol = _protocol()
        corpus = _corpus()
        selection_axes = {
            "candidate_ids": [CANDIDATE_ID], "shot_ids": ["syn-identity"],
            "conditioning_tokens": ["native-clamp01-v1"], "cap_tokens": ["mp0_5"],
            "providers": [{"token": "cuda", "host_loads": ["idle"]}],
        }
        plan = build_matrix(protocol, corpus, _candidate_entries(), selection_axes, "smoke", "el8-x86_64")
        artifacts = run_module._validate_selected_artifacts(plan, _artifact_map(directory), V2_PROTOCOL_PATH)
        runner_section = _report_metadata()["runner"]
        hardware = {"platform": "linux", "architecture": "x86_64", "gpu": "gpu-a", "driver": "driver-1"}

        def identity(proto):
            return run_module._compute_identity(
                proto, corpus, plan, "el8-x86_64", "smoke", artifacts, runner_section, hardware, (1, 2, 4, 8),
                candidate_entries=_candidate_entries(), report_schema=REPORT_SCHEMA, corpus_schema=CORPUS_SCHEMA,
                device_index=0, poll_interval_s=0.05, nvml_enabled=True,
            )

        baseline = canonical_sha256(identity(protocol))

        # Same protocol_id, same selector (mp0_5 is still the token), but the cap now resolves to a
        # different megapixel budget -- a genuinely different measurement.
        edited = copy.deepcopy(protocol)
        edited["analysis_caps"][0]["decimal_megapixels"] = 0.6
        assert edited["protocol_id"] == protocol["protocol_id"]
        assert plan.matrix_sha256 == build_matrix(
            edited, corpus, _candidate_entries(), selection_axes, "smoke", "el8-x86_64",
        ).matrix_sha256, "the selector token set is unchanged; only the protocol definition changed"
        assert canonical_sha256(identity(edited)) != baseline, "protocol content must be bound into the identity"


def test_rerun_with_nvml_disabled_after_nvml_enabled_is_rejected() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-nvml-toggle-") as tmp:
        config = _config(
            Path(tmp), shot_ids=["syn-identity"], provider="cuda", host_loads=["idle"],
            nvml_backend_factory=lambda: _FakeNvmlBackend(),
            hardware={"gpu": "fixture-gpu", "driver": "fixture-driver"},
        )
        first = run_bakeoff(config)
        assert not first.incomplete

        # Same --state/--output-dir, but NVML now disabled (--no-nvml): a real resource
        # measurement must never be silently swapped out for -- or reused to stand in for -- a
        # flat zero placeholder, in either direction.
        config2 = copy.copy(config)
        config2.nvml_backend_factory = None
        config2.runtime_module = _FakeRuntime()
        try:
            run_bakeoff(config2)
        except Exception as exc:  # noqa: BLE001 - asserting rejection, not one exact exception type
            assert "identity" in str(exc).lower()
        else:
            raise AssertionError("expected the nvml-disabled rerun to be rejected, not silently reused")


# --------------------------------------------------------------------------------------------
# Fix J: the FULL persisted identity, not just report-derived fields, gates report reuse.
# --------------------------------------------------------------------------------------------


def test_fresh_state_path_with_different_nvml_config_reuses_output_dir_but_is_rejected() -> None:
    """Codex's exact repro: resume.load_state's identity check only guards the SAME --state
    path. A FRESH --state file pointed at the SAME --output-dir bypasses it entirely, and
    device_index/poll_interval_s/nvml_enabled have no home in the report-v2 schema at all, so
    the report-derived cross-check alone cannot catch this either -- only the persisted
    .run-identity.json hash (Fix J) can.
    """

    with tempfile.TemporaryDirectory(prefix="whitewater-run-nvml-fresh-state-") as tmp:
        config = _config(
            Path(tmp), shot_ids=["syn-identity"], provider="cuda", host_loads=["idle"],
            nvml_backend_factory=lambda: _FakeNvmlBackend(),
            hardware={"gpu": "fixture-gpu", "driver": "fixture-driver"},
        )
        first = run_bakeoff(config)
        assert not first.incomplete
        original_report_bytes = (config.output_dir / "report.json").read_bytes()
        identity_path = config.output_dir / ".run-identity.json"
        assert identity_path.is_file()

        config2 = copy.copy(config)
        config2.nvml_backend_factory = None  # --no-nvml
        config2.state_path = config.output_dir / "state-no-nvml.json"  # FRESH state path
        config2.runtime_module = _FakeRuntime()
        try:
            run_bakeoff(config2)
        except DriverFailure as failure:
            assert failure.kind == "report_identity_mismatch"
        else:
            raise AssertionError("expected DriverFailure(report_identity_mismatch)")
        # The original NVML-measured report must be completely untouched.
        assert (config.output_dir / "report.json").read_bytes() == original_report_bytes


def test_fresh_state_path_with_identical_measurement_config_cannot_reuse_old_report() -> None:
    """A fresh state re-executes cells, so its nonvolatile timing/results are not the old report.

    Full report-semantic reuse intentionally rejects that case.  Operators can use the original
    state for an idempotent publication repair, or explicitly choose ``--replace`` when they want
    to publish a newly measured report.
    """

    with tempfile.TemporaryDirectory(prefix="whitewater-run-fresh-state-identical-") as tmp:
        config = _config(Path(tmp), shot_ids=["syn-identity"])
        first = run_bakeoff(config)
        assert not first.incomplete

        config2 = copy.copy(config)
        config2.state_path = config.output_dir / "state-again.json"
        config2.runtime_module = _FakeRuntime()
        original_report_bytes = (config.output_dir / "report.json").read_bytes()
        try:
            run_bakeoff(config2)
        except DriverFailure as failure:
            assert failure.kind == "report_identity_mismatch"
        else:
            raise AssertionError("freshly measured nonvolatile results must not reuse the old report")
        assert config2.runtime_module.sessions_created >= 1
        assert (config.output_dir / "report.json").read_bytes() == original_report_bytes


# --------------------------------------------------------------------------------------------
# Fix L: the .run-identity.json mismatch guard runs BEFORE any cell executes, not only at
# finalize time -- a rejected run must never have mutated a single CellKey-keyed sidecar or
# review preview belonging to the original, already-published run it collided with.
# --------------------------------------------------------------------------------------------


def _snapshot_tree(directory: Path) -> dict[Path, bytes]:
    return {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*") if path.is_file()}


def test_fresh_state_path_reusing_output_dir_is_rejected_before_any_cell_executes() -> None:
    """A fresh state path with a different identity is rejected before any store mutation."""

    with tempfile.TemporaryDirectory(prefix="whitewater-run-fixl-") as tmp:
        config = _config(
            Path(tmp), shot_ids=["prod-sample"], provider="cuda", host_loads=["idle"],
            nvml_backend_factory=lambda: _FakeNvmlBackend(),
            hardware={"gpu": "fixture-gpu", "driver": "fixture-driver"},
        )
        first = run_bakeoff(config)
        assert not first.incomplete

        artifacts_dir = config.output_dir / ".artifacts"
        original_nvml_bytes = (config.output_dir / "nvml.csv").read_bytes()
        original_review_bytes = (config.output_dir / "review.csv").read_bytes()
        original_identity_bytes = (config.output_dir / ".run-identity.json").read_bytes()
        original_runner_bytes = (config.output_dir / "runner.log").read_bytes()
        original_artifacts = _snapshot_tree(artifacts_dir)
        assert original_artifacts, "expected at least one durable committed bundle"

        config2 = copy.copy(config)
        config2.poll_interval_s = config.poll_interval_s * 5  # different measurement config
        fresh_state_path = config.output_dir / "state-fresh.json"
        config2.state_path = fresh_state_path
        config2.runtime_module = _FakeRuntime()
        try:
            run_bakeoff(config2)
        except DriverFailure as failure:
            assert failure.kind == "report_identity_mismatch"
        else:
            raise AssertionError("expected DriverFailure(report_identity_mismatch)")

        # The rejected run must never have gotten far enough to create its own resume state.
        assert not fresh_state_path.exists()

        # Every durable output the original run produced is byte-for-byte untouched -- not just
        # report.json and the immutable artifact generations must all remain untouched.
        assert (config.output_dir / "nvml.csv").read_bytes() == original_nvml_bytes
        assert (config.output_dir / "review.csv").read_bytes() == original_review_bytes
        assert (config.output_dir / ".run-identity.json").read_bytes() == original_identity_bytes
        assert (config.output_dir / "runner.log").read_bytes() == original_runner_bytes
        assert _snapshot_tree(artifacts_dir) == original_artifacts


def test_replace_validates_existing_state_before_overwriting_identity_or_log() -> None:
    """An incompatible --replace state must fail closed before sidecar/log mutation."""

    with tempfile.TemporaryDirectory(prefix="whitewater-run-replace-state-guard-") as tmp:
        config = _config(Path(tmp), shot_ids=["syn-identity"])
        first = run_bakeoff(config)
        assert not first.incomplete
        identity_path = config.output_dir / ".run-identity.json"
        runner_path = config.output_dir / "runner.log"
        original_identity = identity_path.read_bytes()
        original_runner = runner_path.read_bytes()
        original_state = config.state_path.read_bytes()
        original_report = (config.output_dir / "report.json").read_bytes()

        incompatible = copy.copy(config)
        incompatible.poll_interval_s = config.poll_interval_s * 5
        incompatible.replace = True
        incompatible.runtime_module = _FakeRuntime()
        try:
            run_bakeoff(incompatible)
        except ResumeFailure as failure:
            assert failure.kind == "identity_mismatch"
        else:
            raise AssertionError("--replace must reject an incompatible existing state")

        assert identity_path.read_bytes() == original_identity
        assert runner_path.read_bytes() == original_runner
        assert config.state_path.read_bytes() == original_state
        assert (config.output_dir / "report.json").read_bytes() == original_report
        assert incompatible.runtime_module.sessions_created == 0


# --------------------------------------------------------------------------------------------
# Fix M: .run-identity.json is established BEFORE report.json is published (not after), so a
# crash/write-failure in that window can no longer happen going forward; and a report.json that
# already exists without one (the artifact such a crash -- or a pre-fix build -- would leave) is
# a recoverable, finish-publication state, not a hard refuse demanding --replace.
# --------------------------------------------------------------------------------------------


def test_report_json_without_identity_sidecar_recovers_without_replace() -> None:
    """Simulates exactly the artifact an interruption between report.json publication and the
    identity write would leave behind (report.json present, .run-identity.json absent) -- which,
    pre-fix, ``_write_run_identity`` being the LAST write meant any such interruption always left
    behind, and post-fix would require an actual mid-write crash to reach, since the identity is
    now written first. Either way, the next invocation must resume/finish cleanly WITHOUT
    --replace, and must leave the previously published report untouched while doing so.
    """

    with tempfile.TemporaryDirectory(prefix="whitewater-run-fixm-recover-") as tmp:
        config = _config(Path(tmp), shot_ids=["syn-identity"])
        first = run_bakeoff(config)
        assert not first.incomplete
        identity_path = config.output_dir / ".run-identity.json"
        assert identity_path.is_file()
        original_report_bytes = (config.output_dir / "report.json").read_bytes()

        identity_path.unlink()

        config2 = copy.copy(config)
        config2.runtime_module = _FakeRuntime()
        second = run_bakeoff(config2)  # must NOT require --replace
        assert not second.incomplete
        assert second.report == first.report
        assert (config.output_dir / "report.json").read_bytes() == original_report_bytes
        assert identity_path.is_file(), "the identity sidecar must be re-established, not left missing"

        # The re-established identity must be genuinely correct, not merely present: a further
        # ordinary invocation must still recognize it as a matching no-op rather than a fresh
        # mismatch (which would happen if the recovered identity were wrong).
        config3 = copy.copy(config)
        config3.runtime_module = _FakeRuntime()
        third = run_bakeoff(config3)
        assert not third.incomplete
        assert third.report == first.report
        assert (config.output_dir / "report.json").read_bytes() == original_report_bytes


def test_missing_identity_sidecar_does_not_adopt_tampered_exact_generation() -> None:
    """A missing identity sidecar is repaired only after exact refs validate successfully."""

    with tempfile.TemporaryDirectory(prefix="whitewater-run-fixm-tampered-") as tmp:
        config = _config(Path(tmp), shot_ids=["syn-identity"])
        first = run_bakeoff(config)
        assert not first.incomplete
        identity_path = config.output_dir / ".run-identity.json"
        identity_path.unlink()

        state = json.loads(config.state_path.read_text(encoding="utf-8"))
        ref = state["entries"][0]["artifact_ref"]
        manifest_path = (
            config.output_dir.resolve() / ".artifacts" / state["identity_sha256"] / "cells"
            / ref["cell_sha256"] / "manifests" / f"{ref['attempt_id']}.json"
        )
        manifest_path.unlink()

        config2 = copy.copy(config)
        config2.runtime_module = _FakeRuntime()
        try:
            run_bakeoff(config2)
        except Exception as exc:  # noqa: BLE001 - backend-specific exact-ref failure
            assert "artifact" in str(exc).lower() or "manifest" in str(exc).lower()
        else:
            raise AssertionError("tampered exact generation must refuse reuse")
        assert not identity_path.exists(), "failed exact-ref validation must not create identity sidecar"


def test_report_json_without_identity_sidecar_but_mismatched_content_is_still_refused() -> None:
    """The Fix M recovery path is not a blanket amnesty: a report.json missing its identity
    sidecar whose own content genuinely disagrees with this invocation (a different, unrelated
    report reusing the --output-dir, e.g. after a manual .run-identity.json deletion) must still
    be refused, not silently adopted as recoverable."""

    with tempfile.TemporaryDirectory(prefix="whitewater-run-fixm-mismatch-") as tmp:
        config = _config(Path(tmp), shot_ids=["syn-identity"])
        first = run_bakeoff(config)
        assert not first.incomplete
        (config.output_dir / ".run-identity.json").unlink()
        original_report_bytes = (config.output_dir / "report.json").read_bytes()

        config2 = copy.copy(config)
        config2.selection = {**config.selection, "profile": "screen"}
        config2.runtime_module = _FakeRuntime()
        try:
            run_bakeoff(config2)
        except DriverFailure as failure:
            assert failure.kind == "report_identity_mismatch"
        else:
            raise AssertionError("expected DriverFailure(report_identity_mismatch)")
        assert (config.output_dir / "report.json").read_bytes() == original_report_bytes


# --------------------------------------------------------------------------------------------
# Fix N: review.csv repair preserves the human-edited columns an operator already filled in,
# while still repairing corrupted driver-owned columns and re-adding missing rows; nvml.csv (no
# human-edited columns) keeps its exact-byte repair, unaffected by the merge logic.
# --------------------------------------------------------------------------------------------


def test_review_csv_repair_preserves_human_edits_and_still_fixes_driver_owned_corruption() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-fixn-") as tmp:
        config = _config(
            Path(tmp), shot_ids=["prod-sample"], provider="cuda", host_loads=["idle", "live_flame"],
            nvml_backend_factory=lambda: _FakeNvmlBackend(),
            hardware={"gpu": "fixture-gpu", "driver": "fixture-driver"},
        )
        first = run_bakeoff(config)
        assert not first.incomplete
        review_path = config.output_dir / "review.csv"
        nvml_path = config.output_dir / "nvml.csv"

        with review_path.open(newline="", encoding="utf-8") as stream:
            all_rows = list(csv.reader(stream))
        header, original_rows = all_rows[0], all_rows[1:]
        assert header == list(run_module.REVIEW_CSV_HEADER)
        # Same shot and candidate -> same candidate_label/shot_id for both rows; they differ only
        # by host_load. Proves the merge key must include host_load, not just candidate_label+
        # shot_id, or these two rows would be conflated.
        assert len(original_rows) == 2
        assert {row[6] for row in original_rows} == {"idle", "live_flame"}
        assert len({(row[0], row[1]) for row in original_rows}) == 1

        idle_row = next(row for row in original_rows if row[6] == "idle")
        live_flame_row = next(row for row in original_rows if row[6] == "live_flame")

        # A human fills in their review for the "idle" row, and its category column also gets
        # corrupted somehow (driver-owned, but not part of the row's identity).
        human_values = ["0.9", "0.7", "0.1", "0.05", "0.2", "clean edges, slight jitter on reveal"]
        edited_idle_row = list(idle_row)
        edited_idle_row[8:14] = human_values
        original_category = edited_idle_row[2]
        edited_idle_row[2] = original_category + "-CORRUPTED"

        # The "live_flame" row is entirely missing from the file (e.g. an operator's editor
        # mangled it, or a manual edit dropped it) -- it must be re-added, blank.
        with review_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, delimiter=",", quotechar='"', lineterminator="\n")
            writer.writerow(header)
            writer.writerow(edited_idle_row)

        # nvml.csv (no human columns) is also corrupted, to confirm its repair stays exact-byte,
        # unaffected by review.csv's new merge logic.
        original_nvml_bytes = nvml_path.read_bytes()
        nvml_path.write_bytes(b"corrupted,nvml,header\n")

        config2 = copy.copy(config)
        config2.runtime_module = _FakeRuntime()
        second = run_bakeoff(config2)
        assert not second.incomplete
        assert second.report == first.report  # report.json itself untouched/reused

        with review_path.open(newline="", encoding="utf-8") as stream:
            repaired_all_rows = list(csv.reader(stream))
        repaired_header, repaired_rows = repaired_all_rows[0], repaired_all_rows[1:]
        assert repaired_header == list(run_module.REVIEW_CSV_HEADER)
        assert len(repaired_rows) == 2

        repaired_idle_row = next(row for row in repaired_rows if row[6] == "idle")
        repaired_live_flame_row = next(row for row in repaired_rows if row[6] == "live_flame")

        # The corrupted driver-owned column is repaired back to canonical, every other
        # driver-owned column still matches the canonical row exactly, and the human columns the
        # operator filled in survive untouched.
        assert repaired_idle_row[2] == original_category
        assert repaired_idle_row[:8] == idle_row[:8]
        assert repaired_idle_row[8:14] == human_values

        # The missing row is re-added, matching the canonical row exactly, with blank human
        # columns -- same contract as a row that had never been reviewed.
        assert repaired_live_flame_row == live_flame_row
        assert repaired_live_flame_row[8:14] == ["", "", "", "", "", ""]

        # nvml.csv keeps its unaffected exact-byte repair.
        assert nvml_path.read_bytes() == original_nvml_bytes


# --------------------------------------------------------------------------------------------
# Fix O: a missing identity sidecar is recovered ONLY on full-identity evidence -- a matching,
# fully-complete resume state file for the SAME --state path -- never from the partial,
# report-derived cross-check alone (report-v2 has no home for device_index/poll_interval_s/
# nvml_enabled/candidate artifact hashes, so passing it proves nothing about those axes).
# --------------------------------------------------------------------------------------------


def test_nvml_disabled_fresh_state_with_deleted_sidecar_reusing_output_dir_is_refused() -> None:
    """Codex's Fix O repro: an NVML-enabled report.json, .run-identity.json deleted, then
    re-invoked with --no-nvml and a FRESH --state path against the SAME --output-dir. Before this
    fix, _report_matches_current_run alone gated recovery -- and it cannot see device_index/
    poll_interval_s/nvml_enabled at all, so it passed, silently adopting the old NVML-measured
    resource data as though it belonged to a run that never measured it. A fresh --state path has
    no matching, complete resume state to offer as full-identity evidence, so this must now be
    refused, not adopted -- and refused with zero side effects, before a single cell executes."""

    with tempfile.TemporaryDirectory(prefix="whitewater-run-fixo-attack-") as tmp:
        config = _config(
            Path(tmp), shot_ids=["prod-sample"], provider="cuda", host_loads=["idle"],
            nvml_backend_factory=lambda: _FakeNvmlBackend(),
            hardware={"gpu": "fixture-gpu", "driver": "fixture-driver"},
        )
        first = run_bakeoff(config)
        assert not first.incomplete
        (config.output_dir / ".run-identity.json").unlink()
        original_report_bytes = (config.output_dir / "report.json").read_bytes()
        original_nvml_bytes = (config.output_dir / "nvml.csv").read_bytes()
        original_review_bytes = (config.output_dir / "review.csv").read_bytes()

        config2 = copy.copy(config)
        config2.nvml_backend_factory = None  # --no-nvml
        fresh_state_path = config.output_dir / "state-no-nvml.json"
        config2.state_path = fresh_state_path
        config2.runtime_module = _FakeRuntime()
        try:
            run_bakeoff(config2)
        except DriverFailure as failure:
            assert failure.kind == "report_identity_mismatch"
        else:
            raise AssertionError("expected DriverFailure(report_identity_mismatch)")

        # Refused before a single cell executed -- no fresh resume state was even created.
        assert not fresh_state_path.exists()
        # The original NVML-measured evidence must be completely untouched.
        assert (config.output_dir / "report.json").read_bytes() == original_report_bytes
        assert (config.output_dir / "nvml.csv").read_bytes() == original_nvml_bytes
        assert (config.output_dir / "review.csv").read_bytes() == original_review_bytes


def test_same_state_path_with_deleted_sidecar_and_matching_complete_state_still_recovers() -> None:
    """The Fix O gate must not break the legitimate case: a genuine crash-recovery resume, using
    the SAME --state path (whose state file therefore matches the current identity and every
    cell is complete), must still recover the missing identity sidecar and finish publication
    without --replace."""

    with tempfile.TemporaryDirectory(prefix="whitewater-run-fixo-recover-") as tmp:
        config = _config(
            Path(tmp), shot_ids=["prod-sample"], provider="cuda", host_loads=["idle"],
            nvml_backend_factory=lambda: _FakeNvmlBackend(),
            hardware={"gpu": "fixture-gpu", "driver": "fixture-driver"},
        )
        first = run_bakeoff(config)
        assert not first.incomplete
        identity_path = config.output_dir / ".run-identity.json"
        identity_path.unlink()
        original_report_bytes = (config.output_dir / "report.json").read_bytes()

        config2 = copy.copy(config)  # SAME state_path -- its state file matches and is complete
        config2.runtime_module = _FakeRuntime()
        second = run_bakeoff(config2)
        assert not second.incomplete
        assert second.report == first.report
        assert (config.output_dir / "report.json").read_bytes() == original_report_bytes
        assert identity_path.is_file(), "the identity sidecar must be re-established, not left missing"


# --------------------------------------------------------------------------------------------
# Exact-generation previews are distinct for each provider/host_load cell because each committed
# bundle lives under its own immutable attempt directory.
# --------------------------------------------------------------------------------------------


def test_idle_and_live_flame_cells_get_distinct_preview_dirs_and_review_paths() -> None:
    """Each review row resolves to the exact immutable generation that produced its previews."""

    with tempfile.TemporaryDirectory(prefix="whitewater-run-fixp-") as tmp:
        config = _config(
            Path(tmp), shot_ids=["prod-sample"], provider="cuda", host_loads=["idle", "live_flame"],
            nvml_backend_factory=lambda: _FakeNvmlBackend(),
            hardware={"gpu": "fixture-gpu", "driver": "fixture-driver"},
        )
        result = run_bakeoff(config)
        assert not result.incomplete

        with (config.output_dir / "review.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))[1:]
        assert len(rows) == 2
        rows_by_host_load = {row[6]: row for row in rows}
        assert set(rows_by_host_load) == {"idle", "live_flame"}
        # Same shot/candidate/cap/conditioning -> same blinded label for both rows, proving any
        # path difference below comes from provider/host_load, not a different cell identity.
        assert rows_by_host_load["idle"][0] == rows_by_host_load["live_flame"][0]

        preview_paths = {host_load: Path(row[7]) for host_load, row in rows_by_host_load.items()}
        assert preview_paths["idle"] != preview_paths["live_flame"], (
            "idle and live_flame cells must not share a preview directory"
        )
        # Each cell's own preview evidence exists independently -- neither was overwritten by the
        # other cell's later write.
        for preview_dir in preview_paths.values():
            warped_file = preview_dir / "offset_1_warped.pfm"
            assert warped_file.is_file(), f"missing per-cell preview evidence in {preview_dir}"
        # Both paths are immutable artifact-store preview directories, not a mutable public
        # review tree; neither leaks the real candidate id.
        assert preview_paths["idle"].name == "previews"
        assert preview_paths["live_flame"].name == "previews"
        assert "attempts" in preview_paths["idle"].parts
        assert "attempts" in preview_paths["live_flame"].parts
        assert CANDIDATE_ID not in str(preview_paths["idle"])


# --------------------------------------------------------------------------------------------
# Fix Q: an empty canonical row set removes a stale prior file under replace/verify_and_repair
# instead of leaving it behind, and output_paths only ever advertises a CSV that actually exists.
# --------------------------------------------------------------------------------------------


def test_replace_with_empty_optional_rows_removes_stale_csvs_and_stops_advertising_them() -> None:
    """Codex's Fix Q repro: create nvml.csv/review.csv via a production CUDA run, then --replace
    with a synthetic CPU-only selection whose canonical row sets are both empty. The prior run's
    CSVs must not survive underneath the new report -- stale evidence misattributed to the
    current identity -- and output_paths must not point at them once they are gone."""

    with tempfile.TemporaryDirectory(prefix="whitewater-run-fixq-") as tmp:
        config = _config(
            Path(tmp), shot_ids=["prod-sample"], provider="cuda", host_loads=["idle"],
            nvml_backend_factory=lambda: _FakeNvmlBackend(),
            hardware={"gpu": "fixture-gpu", "driver": "fixture-driver"},
        )
        first = run_bakeoff(config)
        assert not first.incomplete
        nvml_path = config.output_dir / "nvml.csv"
        review_path = config.output_dir / "review.csv"
        assert nvml_path.is_file() and review_path.is_file()
        # A run with real rows must advertise both -- the positive side of the same contract.
        assert "nvml.csv" in first.output_paths
        assert "review.csv" in first.output_paths

        # --replace with a synthetic (analytic-truth), CPU-only selection: no CUDA cells means no
        # nvml rows, and analytic-truth shots are never review-eligible, so review rows are also
        # empty.
        config2 = copy.copy(config)
        config2.selection = _selection(shot_ids=["syn-identity"], provider="cpu")
        config2.state_path = config.output_dir / "state-replace.json"
        config2.replace = True
        config2.nvml_backend_factory = None
        config2.runtime_module = _FakeRuntime()
        second = run_bakeoff(config2)
        assert not second.incomplete

        # The stale CUDA-run CSVs are gone, not left behind to be misread as belonging to the new
        # (CPU-only) report.
        assert not nvml_path.exists()
        assert not review_path.exists()
        assert "nvml.csv" not in second.output_paths
        assert "review.csv" not in second.output_paths


def test_fresh_empty_optional_rows_remove_stale_csvs_after_identity_is_established() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-fresh-empty-sidecars-") as tmp:
        config = _config(Path(tmp), shot_ids=["syn-identity"], provider="cpu")
        config.output_dir.mkdir(parents=True)
        (config.output_dir / "nvml.csv").write_text("stale,nvml\n", encoding="utf-8")
        (config.output_dir / "review.csv").write_text("stale,review\n", encoding="utf-8")

        result = run_bakeoff(config)
        assert not result.incomplete
        assert not (config.output_dir / "nvml.csv").exists()
        assert not (config.output_dir / "review.csv").exists()


def test_truncated_summary_and_report_csv_repaired_on_reuse_and_left_alone_once_correct() -> None:
    # summary.txt and report.csv are driver-owned outputs with no human-edited columns. A crash
    # that truncates either (the old summary write was not even crash-atomic), or a stale copy an
    # interrupted earlier attempt left behind, must be repaired to canonical bytes on the next
    # resumed invocation -- and an already-correct file must be left byte-identical, same inode.
    with tempfile.TemporaryDirectory(prefix="whitewater-run-summary-repair-") as tmp:
        config = _config(Path(tmp), shot_ids=["syn-identity"])
        first = run_bakeoff(config)
        assert not first.incomplete
        summary_path = config.output_dir / "summary.txt"
        csv_path = config.output_dir / "report.csv"
        original_summary_bytes = summary_path.read_bytes()
        original_csv_bytes = csv_path.read_bytes()
        assert original_summary_bytes and original_csv_bytes

        # Simulate an interrupted/partial write left behind by an earlier crash: a truncated
        # summary.txt and a truncated report.csv (header row only, no data rows).
        summary_path.write_bytes(original_summary_bytes[: len(original_summary_bytes) // 2])
        csv_path.write_bytes(b"candidate_id,shot_id\n")
        assert summary_path.read_bytes() != original_summary_bytes
        assert csv_path.read_bytes() != original_csv_bytes

        # Reuse branch: report.json already published under this identity, every cell complete.
        config2 = copy.copy(config)
        config2.runtime_module = _FakeRuntime()
        second = run_bakeoff(config2)
        assert not second.incomplete
        assert second.report == first.report  # report.json itself reused, unchanged
        assert summary_path.read_bytes() == original_summary_bytes, "summary.txt must be repaired"
        assert csv_path.read_bytes() == original_csv_bytes, "report.csv must be repaired"
        assert oct(summary_path.stat().st_mode & 0o777) == "0o644"
        assert oct(csv_path.stat().st_mode & 0o777) == "0o644"

        # Idempotence: once both files are already correct, a further resumed run must not even
        # rewrite them -- same inode, same mtime, not merely byte-identical after a rewrite.
        summary_stat_after_repair = summary_path.stat()
        csv_stat_after_repair = csv_path.stat()
        third = run_bakeoff(copy.copy(config2))
        assert not third.incomplete
        assert summary_path.stat().st_ino == summary_stat_after_repair.st_ino
        assert summary_path.stat().st_mtime_ns == summary_stat_after_repair.st_mtime_ns
        assert csv_path.stat().st_ino == csv_stat_after_repair.st_ino
        assert csv_path.stat().st_mtime_ns == csv_stat_after_repair.st_mtime_ns


def test_interrupted_canonical_publish_never_leaves_a_partial_summary_file() -> None:
    # An interrupted summary/report.csv publish (the driver-owned, no-human-column outputs now
    # routed through _publish_canonical_bytes) must never leave a truncated file at the final
    # path: the prior content survives intact and no staging temp file is left behind.
    with tempfile.TemporaryDirectory(prefix="whitewater-run-summary-atomic-") as tmp:
        directory = Path(tmp)
        destination = directory / "summary.txt"
        destination.write_bytes(b"prior complete summary\n")
        original_bytes = destination.read_bytes()

        real_fsync = os.fsync

        def failing_fsync(fd):
            raise OSError("simulated interruption during canonical publish")

        os.fsync = failing_fsync
        try:
            try:
                run_module._publish_canonical_bytes(destination, b"a much longer canonical summary that must never land\n")
            except DriverFailure as failure:
                assert failure.kind == "atomic_write"
            else:
                raise AssertionError("expected DriverFailure(atomic_write)")
        finally:
            os.fsync = real_fsync

        assert destination.read_bytes() == original_bytes
        leftovers = [p for p in directory.iterdir() if p.name.startswith(".summary.txt.")]
        assert leftovers == [], f"leftover temp files: {leftovers}"


class _StartFailingContext:
    """Wraps a real multiprocessing context; the first ``fail_count`` ``Process.start()`` calls
    raise ``OSError(errno_value)`` (simulating a failed fork under memory pressure) with no child
    ever created. Later ``Process`` objects start for real, so a matrix keeps running."""

    def __init__(self, real: Any, errno_value: int, fail_count: int):
        self._real = real
        self._errno = errno_value
        self._remaining = fail_count

    def Queue(self) -> Any:
        return self._real.Queue()

    def Process(self, *args: Any, **kwargs: Any) -> Any:
        process = self._real.Process(*args, **kwargs)
        if self._remaining > 0:
            self._remaining -= 1
            captured_errno = self._errno

            def _failing_start() -> None:
                raise OSError(captured_errno, "simulated fork failure under memory pressure")

            process.start = _failing_start  # type: ignore[method-assign]
        return process


def test_cuda_fork_start_failure_maps_errno_to_typed_driver_failure() -> None:
    # A failed fork()/start() must become a TYPED DriverFailure, not an uncaught OSError.
    # Resource-exhaustion errnos (ENOMEM/EAGAIN) map to out_of_memory; anything else to
    # runtime_error. Guarded with a timeout because it drives the real subprocess entry point.
    real_fork = run_module._mp.get_context("fork")

    def attempt(errno_value: int):
        wrapper = _StartFailingContext(real_fork, errno_value, fail_count=1)
        original_mp = run_module._mp
        run_module._mp = types.SimpleNamespace(get_context=lambda method: wrapper)
        try:
            return run_module.run_cuda_measurement_in_subprocess(
                lambda stage_sampler: {"base": {}}, lambda: _FakeNvmlBackend(), 0, 0.01,
            )
        finally:
            run_module._mp = original_mp

    for errno_value, expected_kind in (
        (errno.ENOMEM, "out_of_memory"),
        (errno.EAGAIN, "out_of_memory"),
        (errno.EPERM, "runtime_error"),
    ):
        try:
            _call_with_timeout(lambda ev=errno_value: attempt(ev), 15.0)
        except DriverFailure as failure:
            assert failure.kind == expected_kind, (errno_value, failure.kind)
        else:
            raise AssertionError(f"expected a typed DriverFailure for errno={errno_value}")


def test_cuda_fork_start_failure_is_a_typed_cell_failure_not_a_whole_run_abort() -> None:
    # Two CUDA cells (same shot, idle + live_flame). The first cell's fork fails; it must record a
    # TYPED out_of_memory cell failure while the SECOND cell still runs to completion -- the whole
    # matrix must not abort, and completed cells stay durable.
    with tempfile.TemporaryDirectory(prefix="whitewater-run-fork-fail-") as tmp:
        config = _config(
            Path(tmp), shot_ids=["syn-identity"], provider="cuda", host_loads=["idle", "live_flame"],
            nvml_backend_factory=lambda: _FakeNvmlBackend(),
            hardware={"gpu": "fixture-gpu", "driver": "fixture-driver"},
            cuda_measurement_runner=run_module.run_cuda_measurement_in_subprocess,
        )
        real_fork = run_module._mp.get_context("fork")
        wrapper = _StartFailingContext(real_fork, errno.ENOMEM, fail_count=1)
        original_mp = run_module._mp
        run_module._mp = types.SimpleNamespace(get_context=lambda method: wrapper)
        try:
            result = _call_with_timeout(lambda: run_bakeoff(config), 90.0)
        finally:
            run_module._mp = original_mp

        assert not result.incomplete, "the run must complete, not abort, when one cell's fork fails"
        results = result.report["results"]
        assert len(results) == 2
        fails = [r for r in results if r["status"] == "fail"]
        passes = [r for r in results if r["status"] == "pass"]
        assert len(fails) == 1 and len(passes) == 1
        assert fails[0]["failure"]["type"] == "out_of_memory", fails[0]["failure"]
        assert fails[0]["host_load"] == "idle"
        assert passes[0]["host_load"] == "live_flame"


def test_nonfinite_derived_metric_degrades_to_not_applicable_not_a_whole_run_abort() -> None:
    # A derived OPTIONAL metric that computes nonfinite from finite flow must never reach the
    # coordinator (which rejects any nonfinite number as a hard, whole-run-aborting failure). It
    # degrades to a logged not_applicable; the cell still passes and the run completes.
    with tempfile.TemporaryDirectory(prefix="whitewater-run-nonfinite-") as tmp:
        config = _config(Path(tmp), shot_ids=["prod-sample"], exr_decoder=_fake_exr_decoder(8, 6))
        original = run_module.metrics_module.visible_warp_residual
        run_module.metrics_module.visible_warp_residual = lambda *a, **k: float("inf")
        try:
            result = run_bakeoff(config)
        finally:
            run_module.metrics_module.visible_warp_residual = original

        assert not result.incomplete
        # The coordinator accepted every result, so it never saw a nonfinite value.
        validate_report_consistency(result.report, config.protocol, config.report_schema, config.corpus, config.corpus_schema)
        cell_result = result.report["results"][0]
        assert cell_result["status"] == "pass"
        assert "visible_warp_residual" not in cell_result["metrics"]
        assert "visible_warp_residual" in cell_result["metrics"]["not_applicable"]
        # The OTHER optional derived metric was unaffected.
        assert "forward_backward_residual_px" in cell_result["metrics"]
        log_text = (config.output_dir / "runner.log").read_text(encoding="utf-8")
        assert "visible_warp_residual nonfinite" in log_text


def test_review_preview_write_failure_degrades_to_logged_skip_not_a_cell_failure() -> None:
    # A pure preview encoding failure must degrade to a logged skip: the measured cell still
    # passes, while the exact bundle contains no preview pair and review.csv advertises no path.
    with tempfile.TemporaryDirectory(prefix="whitewater-run-preview-fail-") as tmp:
        config = _config(Path(tmp), shot_ids=["prod-sample"], exr_decoder=_fake_exr_decoder(8, 6))
        original = run_module.encode_pfm

        def _failing_encode(*args: Any, **kwargs: Any) -> bytes:
            raise ValueError("simulated preview encoding failure")

        run_module.encode_pfm = _failing_encode
        try:
            result = run_bakeoff(config)
        finally:
            run_module.encode_pfm = original

        assert not result.incomplete
        validate_report_consistency(result.report, config.protocol, config.report_schema, config.corpus, config.corpus_schema)
        cell_result = result.report["results"][0]
        assert cell_result["status"] == "pass", "a preview write failure must not fail the measured cell"
        assert "visible_warp_residual" in cell_result["metrics"]
        log_text = (config.output_dir / "runner.log").read_text(encoding="utf-8")
        assert "review preview offset 1 skipped" in log_text
        # No preview artifacts were committed, but review.csv still records the blinded row with
        # a blank preview path.
        review_path = config.output_dir / "review.csv"
        assert review_path.is_file()
        with review_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        assert len(rows) == 2
        assert rows[1][7] == ""


def test_identity_binds_candidate_entries_report_schema_and_corpus_schema() -> None:
    # Surface A: candidate admission/legal evidence and the two validation-contract schemas each
    # change the report's content or acceptance, so each must change the identity.
    with tempfile.TemporaryDirectory(prefix="whitewater-run-identity-entries-") as tmp:
        directory = Path(tmp)
        protocol = _protocol()
        corpus = _corpus()
        selection_axes = {
            "candidate_ids": [CANDIDATE_ID], "shot_ids": ["syn-identity"],
            "conditioning_tokens": ["native-clamp01-v1"], "cap_tokens": ["mp0_5"],
            "providers": [{"token": "cpu", "host_loads": ["not_applicable"]}],
        }
        plan = build_matrix(protocol, corpus, _candidate_entries(), selection_axes, "smoke", "el8-x86_64")
        artifacts = run_module._validate_selected_artifacts(plan, _artifact_map(directory), V2_PROTOCOL_PATH)
        runner_section = _report_metadata()["runner"]
        hardware = {"platform": "linux", "architecture": "x86_64"}

        def ident(candidate_entries=None, report_schema=REPORT_SCHEMA, corpus_schema=CORPUS_SCHEMA):
            return run_module._compute_identity(
                protocol, corpus, plan, "el8-x86_64", "smoke", artifacts, runner_section, hardware, (1, 2, 4, 8),
                candidate_entries=candidate_entries if candidate_entries is not None else _candidate_entries(),
                report_schema=report_schema, corpus_schema=corpus_schema,
                device_index=0, poll_interval_s=0.05, nvml_enabled=False,
            )

        baseline = canonical_sha256(ident())
        entries_changed = copy.deepcopy(_candidate_entries())
        entries_changed[0]["exclusion_reason"]["message"] = "an updated legal-review verdict"
        assert baseline != canonical_sha256(ident(candidate_entries=entries_changed))
        assert baseline != canonical_sha256(ident(report_schema={**REPORT_SCHEMA, "x-extra-rule": True}))
        assert baseline != canonical_sha256(ident(corpus_schema={**CORPUS_SCHEMA, "x-extra-rule": True}))


def test_changed_excluded_candidate_legal_evidence_is_refused_on_reuse() -> None:
    # Codex's exact repro: editing only an excluded-but-measurable candidate's legal-review
    # message and rerunning the same completed state must NOT hand back report.json with the OLD
    # legal evidence -- report-v2 requires the report to carry each excluded candidate's verdict.
    with tempfile.TemporaryDirectory(prefix="whitewater-run-legal-evidence-") as tmp:
        config = _config(Path(tmp), shot_ids=["syn-identity"])
        first = run_bakeoff(config)
        assert not first.incomplete
        candidate = first.report["candidates"][0]
        assert candidate["exclusion_reason"]["message"] == "test fixture, not a shipping claim"

        config2 = copy.copy(config)
        config2.runtime_module = _FakeRuntime()
        changed_entries = copy.deepcopy(config.candidate_entries)
        changed_entries[0]["exclusion_reason"]["message"] = "UPDATED verdict: redistribution now cleared"
        config2.candidate_entries = changed_entries
        try:
            run_bakeoff(config2)
        except DriverFailure as exc:
            assert "identity" in str(exc).lower()
        else:
            raise AssertionError("a changed candidate legal surface must be refused on reuse, not silently reused")


def _assert_report_reuse_tamper_is_rejected_without_derivative_repair(
    config: RunConfig,
    baseline_report: Mapping[str, Any],
    mutate: Any,
) -> None:
    """Run one report mutation through the complete reuse gate and snapshot every derivative."""

    output_dir = config.output_dir
    derivative_names = ("summary.txt", "report.csv", "nvml.csv", "review.csv")
    original_derivatives = {
        name: (output_dir / name).read_bytes()
        for name in derivative_names
        if (output_dir / name).exists()
    }
    original_identity = (output_dir / ".run-identity.json").read_bytes()
    mutated_report = copy.deepcopy(baseline_report)
    mutate(mutated_report)
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(mutated_report), encoding="utf-8")
    config2 = copy.copy(config)
    config2.runtime_module = _FakeRuntime()
    try:
        run_bakeoff(config2)
    except DriverFailure as failure:
        assert failure.kind == "report_identity_mismatch", failure
    else:
        raise AssertionError("tampered report must be rejected before derivative repair")
    assert report_path.read_text(encoding="utf-8") == json.dumps(mutated_report)
    assert (output_dir / ".run-identity.json").read_bytes() == original_identity
    for name, expected in original_derivatives.items():
        assert (output_dir / name).read_bytes() == expected, name


def test_report_semantic_tampering_is_rejected_before_any_derivative_repair() -> None:
    """Every nonvolatile report surface is binding, even when its JSON shape remains valid."""

    with tempfile.TemporaryDirectory(prefix="whitewater-run-report-semantic-tamper-") as tmp:
        config = _config(
            Path(tmp), shot_ids=["prod-sample"], provider="cuda", host_loads=["idle"],
            nvml_backend_factory=lambda: _FakeNvmlBackend(),
            hardware={"gpu": "fixture-gpu", "driver": "fixture-driver"},
        )
        config.report_metadata = {
            **config.report_metadata,
            "warnings": ["baseline operator warning"],
            "summary": {"final_quality_score": 81.0},
        }
        first = run_bakeoff(config)
        assert not first.incomplete
        assert first.report is not None
        baseline = copy.deepcopy(first.report)

        mutations = (
            ("result metric", lambda report: report["results"][0]["metrics"].__setitem__(
                "repeated_run_p99_delta_px", 0.01,
            )),
            # identity is a schema-valid id but is not a declared category for this shot.
            ("result category", lambda report: report["results"][0].__setitem__("category", "identity")),
            (
                "candidate legal content",
                lambda report: report["candidates"][0]["exclusion_reason"].__setitem__(
                    "message", "edited legal-review message",
                ),
            ),
            ("corpus id", lambda report: report.__setitem__("corpus_id", "other-corpus")),
            ("corpus hash", lambda report: report.__setitem__("corpus_sha256", "0" * 64)),
            ("warnings added", lambda report: report["warnings"].append("new warning")),
            ("warnings removed", lambda report: report.pop("warnings")),
            ("warnings changed", lambda report: report["warnings"].__setitem__(0, "changed warning")),
            (
                "summary added",
                lambda report: report["summary"].__setitem__("category_scores", {"motion-blur": 81.0}),
            ),
            ("summary removed", lambda report: report["summary"].pop("final_quality_score")),
            ("summary changed", lambda report: report["summary"].__setitem__("final_quality_score", 82.0)),
            ("unknown property", lambda report: report.__setitem__("unexpected_property", True)),
        )
        for name, mutate in mutations:
            _assert_report_reuse_tamper_is_rejected_without_derivative_repair(
                config, baseline, mutate,
            )
            # Each mutation helper writes its own complete copy, so the next case starts from the
            # original valid report rather than from the previous rejected mutation.


def test_report_reuse_ignores_only_explicit_volatile_publication_metadata() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-report-volatile-") as tmp:
        config = _config(
            Path(tmp), shot_ids=["prod-sample"], provider="cuda", host_loads=["idle"],
            nvml_backend_factory=lambda: _FakeNvmlBackend(),
            hardware={"gpu": "fixture-gpu", "driver": "fixture-driver"},
        )
        first = run_bakeoff(config)
        assert not first.incomplete
        assert first.report is not None
        mutated = copy.deepcopy(first.report)
        mutated["report_id"] = "reused-report"
        mutated["started_utc"] = "2020-01-01T00:00:00+00:00"
        mutated["completed_utc"] = "2020-01-01T00:00:01+00:00"
        mutated["runner"]["command"] = "operator-repair-command"
        (config.output_dir / "report.json").write_text(json.dumps(mutated), encoding="utf-8")

        config2 = copy.copy(config)
        config2.runtime_module = _FakeRuntime()
        second = run_bakeoff(config2)
        assert not second.incomplete
        assert second.report == mutated


def test_missing_identity_sidecar_plus_semantic_report_tamper_stays_unrepaired() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-report-sidecar-tamper-") as tmp:
        config = _config(
            Path(tmp), shot_ids=["prod-sample"], provider="cuda", host_loads=["idle"],
            nvml_backend_factory=lambda: _FakeNvmlBackend(),
            hardware={"gpu": "fixture-gpu", "driver": "fixture-driver"},
        )
        first = run_bakeoff(config)
        assert not first.incomplete
        assert first.report is not None
        identity_path = config.output_dir / ".run-identity.json"
        identity_path.unlink()
        tampered_report = copy.deepcopy(first.report)
        tampered_report["results"][0]["metrics"]["repeated_run_p99_delta_px"] = 0.01
        report_path = config.output_dir / "report.json"
        report_path.write_text(json.dumps(tampered_report), encoding="utf-8")
        derivative_names = ("summary.txt", "report.csv", "nvml.csv", "review.csv")
        original_derivatives = {
            name: (config.output_dir / name).read_bytes()
            for name in derivative_names
        }

        config2 = copy.copy(config)
        config2.runtime_module = _FakeRuntime()
        try:
            run_bakeoff(config2)
        except DriverFailure as failure:
            assert failure.kind == "report_identity_mismatch"
        else:
            raise AssertionError("missing-sidecar semantic tamper must be rejected")
        assert not identity_path.exists(), "semantic validation must precede sidecar recreation"
        assert report_path.read_text(encoding="utf-8") == json.dumps(tampered_report)
        for name, expected in original_derivatives.items():
            assert (config.output_dir / name).read_bytes() == expected, name


def test_candidate_order_tampering_is_rejected_by_the_ordered_projection() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-candidate-order-") as tmp:
        directory = Path(tmp)
        candidate_a, candidate_b = "candidate-a", "candidate-b"
        protocol = _protocol()
        protocol["candidate_ids"] = [
            {"id": candidate_a, "role": "shipping-candidate"},
            {"id": candidate_b, "role": "shipping-candidate"},
        ]
        config = RunConfig(
            protocol=protocol,
            corpus=_corpus(),
            candidate_entries=[_candidate_entry(candidate_a), _candidate_entry(candidate_b)],
            selection={
                "profile": "smoke", "environment": "el8-x86_64",
                "candidate_ids": [candidate_a, candidate_b],
                "conditioning_tokens": ["native-clamp01-v1"], "cap_tokens": ["mp0_5"],
                "providers": [{"token": "cpu", "host_loads": ["not_applicable"]}],
                "shot_ids": ["prod-sample"],
            },
            artifact_map={
                candidate_a: _artifact_map_entry_for(directory, candidate_a, "a"),
                candidate_b: _artifact_map_entry_for(directory, candidate_b, "b"),
            },
            report_schema=REPORT_SCHEMA, corpus_schema=CORPUS_SCHEMA,
            output_dir=directory / "output", state_path=directory / "output" / "state.json",
            device_index=0, poll_interval_s=0.01, chain_offsets=(1, 2, 4, 8),
            report_metadata=_report_metadata(), protocol_path=V2_PROTOCOL_PATH, replace=False,
            runtime_module=_FakeRuntime(path_dependent=True), array_module=_FakeArrays(),
            nvml_backend_factory=None, exr_decoder=_fake_exr_decoder(8, 6),
        )
        first = run_bakeoff(config)
        assert not first.incomplete
        assert first.report is not None
        baseline = copy.deepcopy(first.report)
        derivative_names = ("summary.txt", "report.csv", "review.csv")
        original_derivatives = {
            name: (config.output_dir / name).read_bytes()
            for name in derivative_names
        }
        tampered = copy.deepcopy(baseline)
        tampered["candidates"].reverse()
        (config.output_dir / "report.json").write_text(json.dumps(tampered), encoding="utf-8")
        config2 = copy.copy(config)
        config2.runtime_module = _FakeRuntime(path_dependent=True)
        try:
            run_bakeoff(config2)
        except DriverFailure as failure:
            assert failure.kind == "report_identity_mismatch"
        else:
            raise AssertionError("candidate order tampering must be rejected")
        for name, expected in original_derivatives.items():
            assert (config.output_dir / name).read_bytes() == expected, name


def _pid_gated_failing_backend_factory():
    """Factory whose backend succeeds inside the forked child (a different pid) but raises
    NvmlFailure at the PARENT's post-exit device query (the original pid)."""

    parent_pid = os.getpid()

    class _PidGatedBackend:
        def device_handle(self, device_index: int) -> Any:
            return device_index

        def device_used_mib(self, handle: Any) -> float:
            if os.getpid() == parent_pid:
                raise NvmlFailure("query_failed", "simulated parent post-exit device query failure")
            return 1234.0

        def process_used_mib(self, handle: Any, pid: int):
            return None

    return lambda: _PidGatedBackend()


def test_post_exit_nvml_query_failure_is_a_typed_cell_failure_not_a_whole_run_abort() -> None:
    # Surface B / finding 2: the PARENT's post-join NVML reading (outside the fork-start mapping)
    # must never escape. A device-query failure there becomes a typed runtime_error cell failure;
    # the matrix still completes. Guarded with a timeout (drives the real subprocess entry point).
    with tempfile.TemporaryDirectory(prefix="whitewater-run-postexit-nvml-") as tmp:
        config = _config(
            Path(tmp), shot_ids=["syn-identity"], provider="cuda", host_loads=["idle", "live_flame"],
            nvml_backend_factory=_pid_gated_failing_backend_factory(),
            hardware={"gpu": "fixture-gpu", "driver": "fixture-driver"},
            cuda_measurement_runner=run_module.run_cuda_measurement_in_subprocess,
        )
        result = _call_with_timeout(lambda: run_bakeoff(config), 90.0)
        assert not result.incomplete, "a post-exit NVML failure must not abort the whole matrix"
        results = result.report["results"]
        assert len(results) == 2
        assert all(r["status"] == "fail" for r in results), results
        assert all(r["failure"]["type"] == "runtime_error" for r in results), results


def test_required_bundle_failure_leaves_cell_recoverably_in_progress() -> None:
    # Surface B: a CUDA cell's REQUIRED evidence bundle write failing mid-run (e.g. disk full)
    # must remain recoverably in progress, never silently producing a missing NVML row.
    with tempfile.TemporaryDirectory(prefix="whitewater-run-bundle-fail-") as tmp:
        config = _config(
            Path(tmp), shot_ids=["syn-identity"], provider="cuda", host_loads=["idle"],
            nvml_backend_factory=lambda: _FakeNvmlBackend(),
            hardware={"gpu": "fixture-gpu", "driver": "fixture-driver"},
        )
        original = run_module._commit_cell_bundle

        def _failing_bundle(*args: Any, **kwargs: Any) -> Any:
            raise run_module.ArtifactStoreFailure("write", "simulated required bundle failure")

        run_module._commit_cell_bundle = _failing_bundle
        try:
            try:
                run_bakeoff(config)
            except run_module.ArtifactStoreFailure as exc:
                assert exc.kind == "write"
            else:
                raise AssertionError("required bundle failure must escape without completing state")
        finally:
            run_module._commit_cell_bundle = original

        state = json.loads(config.state_path.read_text(encoding="utf-8"))
        assert state["entries"][0]["state"] == "in_progress"
        assert not (config.output_dir / "report.json").exists()
        log_text = (config.output_dir / "runner.log").read_text(encoding="utf-8")
        assert "cell start" in log_text
        assert "cell persistence failed" in log_text
        assert "cell pass" not in log_text


def test_optional_preview_store_failure_retries_without_previews() -> None:
    # A failure while staging optional preview bytes poisons that attempt. A fresh attempt must
    # re-stage the required result and commit a valid pass with no preview pair.
    with tempfile.TemporaryDirectory(prefix="whitewater-run-partial-preview-") as tmp:
        cell = run_module.CellKey("candidate", "shot", "conditioning", "cap", "cpu", "idle")
        result = {
            "candidate_id": "candidate", "shot_id": "shot", "conditioning_token": "conditioning",
            "cap_token": "cap", "provider": "cpu", "host_load": "idle", "status": "pass",
        }
        bundle = run_module.CellBundle(
            result=result,
            previews=(
                run_module.PreviewPayload("previews/offset_1_warped.pfm", b"warped"),
                run_module.PreviewPayload("previews/offset_1_flow.pfm", b"flow"),
            ),
        )
        failed = False

        def fault_hook(operation: str, path: Path) -> None:
            nonlocal failed
            if operation == "write" and "offset_1_warped.pfm" in path.name and not failed:
                failed = True
                raise run_module.ArtifactStoreFailure("write", "simulated preview staging failure")

        identity = {"test": "optional-preview-retry"}
        with run_module.ArtifactStore(Path(tmp).resolve() / "artifacts", identity, fault_hook=fault_hook) as store:
            execution = run_module._commit_cell_bundle(store, cell, bundle, nvml_enabled=False)
            manifest = store.load_ref(execution.artifact_ref)
            paths = {entry["path"] for entry in manifest["artifacts"]}
            assert paths == {"result.json"}
            assert store.read_artifact(execution.artifact_ref, "result.json") == run_module._canonical_result_bytes(result)
            attempts = list((store.run_root / "cells" / execution.artifact_ref["cell_sha256"] / "attempts").iterdir())
            assert len(attempts) >= 2, "preview failure must abandon the first attempt"
        assert failed


def main() -> int:
    test_cap_megapixels_looks_up_token_and_rejects_unknown()
    test_review_label_is_deterministic_and_does_not_embed_candidate_id()
    test_exr_failure_maps_known_kinds_to_permitted_result_failure_types()
    test_unpadded_grid_crops_bottom_left_region()
    test_dense_truth_and_mask_identity_case_is_zero_everywhere()
    test_write_csv_file_is_single_write_no_clobber_unless_replace_and_skips_empty()
    test_verify_repair_rejects_a_symlink_even_when_target_is_canonical()
    test_runner_log_appends_timestamped_lines_and_survives_reopen()
    test_driver_direct_and_module_help_use_the_same_package_imports()
    test_runner_log_fsync_failure_is_typed()
    test_report_semantic_unicode_failure_is_typed()
    test_replay_resource_and_rows_matches_a_live_sampler()
    test_nvml_evidence_is_cell_bound_at_commit_and_exact_ref_regeneration()
    test_smoke_profile_synthetic_identity_produces_a_valid_report_and_no_nvml_csv()
    test_rerun_with_same_state_is_idempotent_and_does_not_recompute()
    test_production_partition_uses_injected_exr_decoder_and_emits_review_row()
    test_chain_shot_computes_chain_drift_px()
    test_cuda_cell_writes_nvml_csv_with_required_stages_and_resource()
    test_cuda_gpu_mem_limit_is_recorded_in_resource_evidence()
    test_cuda_cell_without_arena_limit_records_no_ceiling()
    test_validator_rejects_each_pass_only_hard_gate()
    test_execution_classifies_metric_hard_gate_overruns()
    test_cuda_memory_gate_becomes_failed_cell_with_evidence_and_publishes_package()
    test_cuda_memory_gate_failure_resume_reuses_generation_and_repairs_nvml()
    test_cuda_dense_scoring_failure_preserves_measurement_evidence_and_publishes_nvml()
    test_cuda_chain_scoring_failure_preserves_measurement_evidence_and_publishes_nvml()
    test_cuda_chain_error_after_measurement_preserves_evidence_and_publishes_nvml()
    test_missing_artifact_map_entry_raises_typed_driver_failure()
    test_manifest_candidate_id_mismatch_is_a_typed_driver_failure()
    test_identity_differs_between_profiles_for_an_otherwise_identical_selection()
    test_run_spec_builder_returns_compact_complete_identity()
    test_chain_offsets_are_sorted_unique_before_execution_and_identity()
    test_runner_and_hardware_metadata_preflight_is_strict_and_normalized()
    test_protocol_incomplete_corpus_fails_before_any_cell_executes()
    test_unmaterialized_production_path_fails_before_any_cell_executes()
    test_driver_validates_the_immutable_corpus_only_once()
    test_legacy_v1_resume_state_is_rejected_with_fresh_replace_diagnostic()
    test_rerun_with_different_profile_same_state_path_is_rejected_not_reused()
    test_changed_report_warnings_and_summary_are_rejected_on_reuse()
    test_stale_report_json_under_a_different_identity_is_explicitly_rejected()
    test_host_load_checkpoint_called_once_per_group_in_order()
    test_host_load_checkpoint_does_not_refire_for_a_repeated_host_load()
    test_cpu_cells_never_trigger_a_host_load_checkpoint()
    test_two_candidates_blinded_previews_differ_and_review_csv_references_them()
    test_cuda_peak_captures_a_first_run_transient_a_boundary_only_design_would_miss()
    test_real_subprocess_cuda_measurement_runner_isolates_work_and_reads_post_exit()
    test_final_cuda_without_nvml_is_rejected_before_state_or_cells()
    test_final_cuda_missing_nvml_stage_becomes_a_typed_cell_failure()
    test_interrupted_cell_resume_does_not_duplicate_bundle_evidence_rows()
    test_regeneration_reads_old_exact_ref_after_current_pointer_advances()
    test_regeneration_rejects_short_or_misordered_completed_records_before_writing()
    test_regenerates_missing_sidecar_outputs_when_report_already_published()
    test_atomic_publish_never_leaves_a_partial_file_on_interruption()
    test_truncated_sidecar_outputs_are_repaired_to_canonical_bytes_and_left_alone_once_correct()
    test_cuda_cell_runs_all_inference_in_the_child_not_the_parent()
    test_child_exit_without_queueing_yields_typed_failure_not_a_hang()
    test_cuda_child_preserves_typed_failure_stage()
    test_identity_differs_for_measurement_config_changes()
    test_identity_binds_protocol_content_not_just_protocol_id()
    test_rerun_with_nvml_disabled_after_nvml_enabled_is_rejected()
    test_fresh_state_path_with_different_nvml_config_reuses_output_dir_but_is_rejected()
    test_fresh_state_path_with_identical_measurement_config_cannot_reuse_old_report()
    test_fresh_state_path_reusing_output_dir_is_rejected_before_any_cell_executes()
    test_replace_validates_existing_state_before_overwriting_identity_or_log()
    test_report_json_without_identity_sidecar_recovers_without_replace()
    test_missing_identity_sidecar_does_not_adopt_tampered_exact_generation()
    test_report_json_without_identity_sidecar_but_mismatched_content_is_still_refused()
    test_report_semantic_tampering_is_rejected_before_any_derivative_repair()
    test_report_reuse_ignores_only_explicit_volatile_publication_metadata()
    test_missing_identity_sidecar_plus_semantic_report_tamper_stays_unrepaired()
    test_candidate_order_tampering_is_rejected_by_the_ordered_projection()
    test_review_csv_repair_preserves_human_edits_and_still_fixes_driver_owned_corruption()
    test_nvml_disabled_fresh_state_with_deleted_sidecar_reusing_output_dir_is_refused()
    test_same_state_path_with_deleted_sidecar_and_matching_complete_state_still_recovers()
    test_idle_and_live_flame_cells_get_distinct_preview_dirs_and_review_paths()
    test_replace_with_empty_optional_rows_removes_stale_csvs_and_stops_advertising_them()
    test_fresh_empty_optional_rows_remove_stale_csvs_after_identity_is_established()
    test_truncated_summary_and_report_csv_repaired_on_reuse_and_left_alone_once_correct()
    test_interrupted_canonical_publish_never_leaves_a_partial_summary_file()
    test_cuda_fork_start_failure_maps_errno_to_typed_driver_failure()
    test_cuda_fork_start_failure_is_a_typed_cell_failure_not_a_whole_run_abort()
    test_nonfinite_derived_metric_degrades_to_not_applicable_not_a_whole_run_abort()
    test_review_preview_write_failure_degrades_to_logged_skip_not_a_cell_failure()
    test_identity_binds_candidate_entries_report_schema_and_corpus_schema()
    test_changed_excluded_candidate_legal_evidence_is_refused_on_reuse()
    test_post_exit_nvml_query_failure_is_a_typed_cell_failure_not_a_whole_run_abort()
    test_required_bundle_failure_leaves_cell_recoverably_in_progress()
    test_optional_preview_store_failure_retries_without_previews()
    print("P25-6 profile driver tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
