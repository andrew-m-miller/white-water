#!/usr/bin/env python3
"""Tests for the P25-6 end-to-end resumable offline profile driver (``tools.bakeoff.run``).

Runs entirely without numpy, onnxruntime, OpenImageIO, pynvml, or a GPU: every dependency is
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
import json
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

from . import run as run_module
from . import synthetic as synthetic_module
from .exr import ExrFailure
from .matrix import build_matrix
from .nvml import NVML_CSV_HEADER, NvmlSampler
from .resume import mark_in_progress
from .run import CudaMeasurementResult, DriverFailure, RunConfig, run_bakeoff
from .validator import canonical_sha256, load_json, validate_report_consistency

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

    def get_available_providers(self) -> list[str]:
        return ["CPUExecutionProvider", "CUDAExecutionProvider"]

    class _Options:
        def add_session_config_entry(self, key: str, value: str) -> None:
            pass

    def SessionOptions(self) -> Any:
        return self._Options()

    def InferenceSession(self, path: str, *, providers: list[str], **kwargs: Any) -> _FakeSession:
        self.sessions_created += 1
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
        chain_offsets=(1, 2, 4, 8),
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
            device_index=0, poll_interval_s=0.05, nvml_enabled=False,
        )
        identity_screen = run_module._compute_identity(
            protocol, corpus, plan, "el8-x86_64", "screen", artifacts, runner_section, hardware, (1, 2, 4, 8),
            device_index=0, poll_interval_s=0.05, nvml_enabled=False,
        )
        # matrix_sha256 itself does not encode profile -- this is exactly the gap Fix A closes.
        assert identity_smoke["matrix_sha256"] == identity_screen["matrix_sha256"]
        assert canonical_sha256(identity_smoke) != canonical_sha256(identity_screen)

        identity_diff_evaluator = run_module._compute_identity(
            protocol, corpus, plan, "el8-x86_64", "smoke", artifacts,
            {**runner_section, "evaluator_sha256": "f" * 64}, hardware, (1, 2, 4, 8),
            device_index=0, poll_interval_s=0.05, nvml_enabled=False,
        )
        assert canonical_sha256(identity_smoke) != canonical_sha256(identity_diff_evaluator)


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


def test_interrupted_cell_resume_does_not_duplicate_sidecar_rows() -> None:
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
            device_index=config.device_index, poll_interval_s=config.poll_interval_s,
            nvml_enabled=config.nvml_backend_factory is not None,
        )
        from .resume import create_state
        run_module._ensure_output_dir(config.output_dir)
        create_state(config.state_path, identity, plan)
        cell = plan.cells[0]
        mark_in_progress(config.state_path, identity, plan, cell)

        review_dir = config.output_dir / "review-previews"
        executor, _ctx = run_module.build_executor(
            protocol=config.protocol, corpus=config.corpus, profile=config.selection["profile"],
            artifacts=artifacts, runtime_module=config.runtime_module, array_module=config.array_module,
            nvml_backend_factory=config.nvml_backend_factory, device_index=config.device_index,
            poll_interval_s=config.poll_interval_s, chain_offsets=config.chain_offsets,
            exr_decoder=config.exr_decoder, output_dir=config.output_dir, review_dir=review_dir,
            log=lambda message: None, host_load_checkpoint=lambda host_load: None,
            cuda_measurement_runner=config.cuda_measurement_runner,
        )
        # Simulate an "interrupted" attempt: the cell's real work runs (writing its nvml
        # sidecar) but the process dies before RunCoordinator would have marked it complete --
        # state.json is left with this cell still "in_progress".
        executor(cell)
        stale_rows = run_module._read_nvml_sidecar(config.output_dir, cell)
        assert stale_rows, "the interrupted attempt should have written a sidecar"

        # A fresh run_bakeoff call now resumes: resume.load_state recovers the in_progress cell
        # to pending (documented behavior) and RunCoordinator re-executes it, which must
        # OVERWRITE, not append to, the sidecar written above.
        result = run_bakeoff(config)
        assert not result.incomplete
        nvml_path = config.output_dir / "nvml.csv"
        with nvml_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        data_rows = rows[1:]
        assert len(data_rows) == len(stale_rows), "resume must not duplicate the retried cell's rows"


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
            resolved = {"device_index": 0, "poll_interval_s": 0.05, "nvml_enabled": True, **kwargs}
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


def main() -> int:
    test_cap_megapixels_looks_up_token_and_rejects_unknown()
    test_review_label_is_deterministic_and_does_not_embed_candidate_id()
    test_exr_failure_maps_known_kinds_to_permitted_result_failure_types()
    test_unpadded_grid_crops_bottom_left_region()
    test_dense_truth_and_mask_identity_case_is_zero_everywhere()
    test_write_csv_file_is_single_write_no_clobber_unless_replace_and_skips_empty()
    test_runner_log_appends_timestamped_lines_and_survives_reopen()
    test_replay_resource_and_rows_matches_a_live_sampler()
    test_smoke_profile_synthetic_identity_produces_a_valid_report_and_no_nvml_csv()
    test_rerun_with_same_state_is_idempotent_and_does_not_recompute()
    test_production_partition_uses_injected_exr_decoder_and_emits_review_row()
    test_chain_shot_computes_chain_drift_px()
    test_cuda_cell_writes_nvml_csv_with_required_stages_and_resource()
    test_missing_artifact_map_entry_raises_typed_driver_failure()
    test_manifest_candidate_id_mismatch_is_a_typed_driver_failure()
    test_identity_differs_between_profiles_for_an_otherwise_identical_selection()
    test_rerun_with_different_profile_same_state_path_is_rejected_not_reused()
    test_stale_report_json_under_a_different_identity_is_explicitly_rejected()
    test_host_load_checkpoint_called_once_per_group_in_order()
    test_host_load_checkpoint_does_not_refire_for_a_repeated_host_load()
    test_cpu_cells_never_trigger_a_host_load_checkpoint()
    test_two_candidates_blinded_previews_differ_and_review_csv_references_them()
    test_cuda_peak_captures_a_first_run_transient_a_boundary_only_design_would_miss()
    test_real_subprocess_cuda_measurement_runner_isolates_work_and_reads_post_exit()
    test_interrupted_cell_resume_does_not_duplicate_sidecar_rows()
    test_regenerates_missing_sidecar_outputs_when_report_already_published()
    test_cuda_cell_runs_all_inference_in_the_child_not_the_parent()
    test_child_exit_without_queueing_yields_typed_failure_not_a_hang()
    test_identity_differs_for_measurement_config_changes()
    test_rerun_with_nvml_disabled_after_nvml_enabled_is_rejected()
    print("P25-6 profile driver tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
