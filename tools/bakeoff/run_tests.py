#!/usr/bin/env python3
"""Tests for the P25-6 end-to-end resumable offline profile driver (``tools.bakeoff.run``).

Runs entirely without numpy, onnxruntime, OpenImageIO, pynvml, or a GPU: every dependency is
injected (a fake array/runtime module pair, a fake NVML backend, and a fake EXR decoder for the
production-partition test). ``validate_manifest_artifact`` still needs a real manifest/artifact
pair and a real protocol file for its own contract checks, so this reuses the same checked-in
fixture manifest ``models/fixtures/positive/artifact-v1.json`` that ``evaluator_tests.py`` uses,
together with the real ``bakeoff/protocol-v2.json``. Matrix planning and report validation,
however, use a small hand-built protocol/corpus (mirroring the style of ``matrix_tests.py`` and
``reporting_tests.py``) so the tests do not depend on the full, frozen candidate/provider matrix.
"""

from __future__ import annotations

import copy
import csv
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from . import run as run_module
from . import synthetic as synthetic_module
from .exr import ExrFailure
from .run import DriverFailure, RunConfig, run_bakeoff
from .validator import load_json, validate_report_consistency

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


def _zero_flow(height: int, width: int) -> _FakeArray:
    return _FakeArray([[[[0.0 for _ in range(width)] for _ in range(height)] for _ in range(2)]])


class _FakeSession:
    def __init__(self, selected: list[str] | None = None):
        self.calls = 0
        self._selected = selected or ["CPUExecutionProvider"]

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
        return [_zero_flow(height, width)]


class _FakeRuntime:
    __version__ = "fake-ort-run-tests"

    def __init__(self, providers: list[str] | None = None):
        self.providers = providers or ["CPUExecutionProvider"]
        self.sessions_created = 0

    def get_available_providers(self) -> list[str]:
        return ["CPUExecutionProvider", "CUDAExecutionProvider"]

    class _Options:
        def add_session_config_entry(self, key: str, value: str) -> None:
            pass

    def SessionOptions(self) -> Any:
        return self._Options()

    def InferenceSession(self, path: str, *, providers: list[str], **kwargs: Any) -> _FakeSession:
        self.sessions_created += 1
        return _FakeSession(list(providers))


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


def _fake_exr_decoder(width: int, height: int) -> Any:
    def decoder(path: str, *, frame_number: int, pixel_aspect_ratio: float) -> dict[str, Any]:
        rows = tuple(
            tuple((0.1, 0.2, 0.3) for _ in range(width)) for _ in range(height)
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


def _candidate_entries() -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": CANDIDATE_ID,
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
    ]


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


def _config(
    tmp: Path,
    *,
    shot_ids: list[str],
    provider: str = "cpu",
    host_loads: list[str] | None = None,
    nvml_backend_factory=None,
    exr_decoder=None,
    hardware: dict[str, str] | None = None,
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
        runtime_module=_FakeRuntime(),
        array_module=_FakeArrays(),
        nvml_backend_factory=nvml_backend_factory,
        exr_decoder=exr_decoder or _fake_exr_decoder(8, 6),
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


def test_append_csv_rows_writes_header_once_and_appends() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-run-csv-") as tmp:
        path = Path(tmp) / "review.csv"
        run_module._append_csv_rows(path, ("a", "b"), [["1", "2"]])
        run_module._append_csv_rows(path, ("a", "b"), [["3", "4"]])
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        assert rows == [["a", "b"], ["1", "2"], ["3", "4"]]


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
        from .nvml import NVML_CSV_HEADER
        assert tuple(rows[0]) == NVML_CSV_HEADER
        assert len(rows) > 1


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


def main() -> int:
    test_cap_megapixels_looks_up_token_and_rejects_unknown()
    test_review_label_is_deterministic_and_does_not_embed_candidate_id()
    test_exr_failure_maps_known_kinds_to_permitted_result_failure_types()
    test_unpadded_grid_crops_bottom_left_region()
    test_dense_truth_and_mask_identity_case_is_zero_everywhere()
    test_append_csv_rows_writes_header_once_and_appends()
    test_runner_log_appends_timestamped_lines_and_survives_reopen()
    test_smoke_profile_synthetic_identity_produces_a_valid_report_and_no_nvml_csv()
    test_rerun_with_same_state_is_idempotent_and_does_not_recompute()
    test_production_partition_uses_injected_exr_decoder_and_emits_review_row()
    test_chain_shot_computes_chain_drift_px()
    test_cuda_cell_writes_nvml_csv_with_required_stages_and_resource()
    test_missing_artifact_map_entry_raises_typed_driver_failure()
    test_manifest_candidate_id_mismatch_is_a_typed_driver_failure()
    print("P25-6 profile driver tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
