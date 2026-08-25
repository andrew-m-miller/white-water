#!/usr/bin/env python3
"""Focused tests for the P25-6 NVML sampler, resource reduction and nvml.csv writer.

These tests inject a scripted fake backend and never touch pynvml or a GPU, so they run
identically on the macOS development machine and on the EL8 CUDA target.
"""

from __future__ import annotations

import csv
import math
import os
import stat
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterator, Sequence

from .nvml import (
    NVML_CSV_HEADER,
    NvmlFailure,
    NvmlSampler,
    PollWindow,
    PynvmlBackend,
    STAGES,
    append_nvml_csv,
    write_nvml_csv,
    write_or_append_nvml_csv,
)
from .validator import load_json, validate

ROOT = Path(__file__).resolve().parents[2]

IDENTITY = {
    "candidate_id": "sea-raft-m",
    "shot_id": "syn-identity",
    "conditioning_token": "native-clamp01-v1",
    "cap_token": "mp2",
    "provider": "cuda",
    "host_load": "idle",
}

_SAMPLE_ALLOWED_KEYS = {"stage", "used_mib", "process_used_mib"}
_RESOURCE_ALLOWED_KEYS = {
    "peak_incremental_device_memory_gib", "baseline_device_memory_mib", "peak_device_memory_mib",
    "cleanup_device_memory_mib", "process_exit_device_memory_mib", "nvml_samples",
}


class _ScriptedBackend:
    """Fake NvmlBackend driven by a fixed sequence of scripted readings."""

    def __init__(self, device_used_mib: Sequence[float], process_used_mib: Sequence[float | None] | None = None):
        self._device_iter: Iterator[float] = iter(device_used_mib)
        self._process_iter: Iterator[float | None] | None = (
            iter(process_used_mib) if process_used_mib is not None else None
        )
        self.handle_calls: list[int] = []

    def device_handle(self, device_index: int) -> Any:
        self.handle_calls.append(device_index)
        return device_index

    def device_used_mib(self, handle: Any) -> float:
        return next(self._device_iter)

    def process_used_mib(self, handle: Any, pid: int) -> float | None:
        if self._process_iter is None:
            return None
        return next(self._process_iter)


class _LatchedBackend:
    """Scripted NvmlBackend for poll() tests.

    Readings advance through a fixed ``(device_used_mib, process_used_mib)`` list, one pair
    consumed per full read (matching one ``PollWindow._record_reading`` -- device then
    process), and hold at the last scripted pair once exhausted rather than raising or
    blocking. That makes a real background poller thread safe to keep running past the last
    value a test cares about: it just keeps re-reading the same held value, which never moves
    a tracked maximum. Index advancement is lock-guarded since a real poller thread reads
    concurrently with the immediate synchronous read ``PollWindow.__enter__`` takes.
    """

    def __init__(self, readings: Sequence[tuple[float, float | None]]):
        assert readings, "at least one scripted reading is required"
        self._readings = list(readings)
        self._index = 0
        self._lock = threading.Lock()
        self.handle_calls: list[int] = []

    def device_handle(self, device_index: int) -> Any:
        self.handle_calls.append(device_index)
        return device_index

    def device_used_mib(self, handle: Any) -> float:
        with self._lock:
            return self._readings[self._index][0]

    def process_used_mib(self, handle: Any, pid: int) -> float | None:
        with self._lock:
            value = self._readings[self._index][1]
            if self._index < len(self._readings) - 1:
                self._index += 1
            return value


def _failure(kind: str, callback) -> None:
    try:
        callback()
    except NvmlFailure as failure:
        assert failure.kind == kind, (failure.kind, kind)
        assert failure.reason == kind
        assert failure.failure_type == "nvml_failure"
    else:
        raise AssertionError(f"expected NvmlFailure({kind})")


def _clock(values: list[float]) -> Any:
    iterator = iter(values)
    return lambda: next(iterator)


def test_sample_produces_schema_shaped_records_with_and_without_process() -> None:
    backend = _ScriptedBackend([1000.0, 1500.0], [250.0, None])
    sampler = NvmlSampler(backend, device_index=0, pid=4242, clock=_clock([1.0, 2.0]))
    assert backend.handle_calls == [0]

    with_process = sampler.sample("baseline")
    assert with_process == {"stage": "baseline", "used_mib": 1000.0, "process_used_mib": 250.0}
    assert set(with_process) == {"stage", "used_mib", "process_used_mib"}

    without_process = sampler.sample("session_create")
    assert without_process == {"stage": "session_create", "used_mib": 1500.0}
    assert "process_used_mib" not in without_process

    assert sampler.samples == [with_process, without_process]
    # The returned dict and the accumulated list are independent copies.
    with_process["used_mib"] = -1.0
    assert sampler.samples[0]["used_mib"] == 1000.0


def test_resource_computes_peak_incremental_and_stage_snapshots() -> None:
    backend = _ScriptedBackend([1000.0, 1500.0, 1800.0, 1700.0, 1200.0, 1000.0])
    sampler = NvmlSampler(backend, device_index=0, clock=_clock([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
    for stage in ("baseline", "session_create", "steady", "steady", "cleanup", "process_exit"):
        sampler.sample(stage)

    resource = sampler.resource()
    assert resource["baseline_device_memory_mib"] == 1000.0
    assert resource["peak_device_memory_mib"] == 1800.0
    assert math.isclose(resource["peak_incremental_device_memory_gib"], 800.0 / 1024.0)
    assert resource["cleanup_device_memory_mib"] == 1200.0
    assert resource["process_exit_device_memory_mib"] == 1000.0
    assert len(resource["nvml_samples"]) == 6


def test_resource_peak_incremental_never_negative_when_below_baseline() -> None:
    backend = _ScriptedBackend([1000.0, 800.0])
    sampler = NvmlSampler(backend, device_index=0)
    sampler.sample("baseline")
    sampler.sample("cleanup")
    resource = sampler.resource()
    assert resource["peak_incremental_device_memory_gib"] == 0.0
    assert "process_exit_device_memory_mib" not in resource


def test_resource_uses_first_baseline_when_sampled_more_than_once() -> None:
    backend = _ScriptedBackend([1000.0, 1100.0, 1600.0])
    sampler = NvmlSampler(backend, device_index=0)
    sampler.sample("baseline")
    sampler.sample("baseline")
    sampler.sample("steady")
    resource = sampler.resource()
    assert resource["baseline_device_memory_mib"] == 1000.0
    assert math.isclose(resource["peak_incremental_device_memory_gib"], 600.0 / 1024.0)


def test_missing_baseline_raises_typed_failure() -> None:
    backend = _ScriptedBackend([1500.0])
    sampler = NvmlSampler(backend, device_index=0)
    sampler.sample("session_create")
    _failure("missing_baseline", sampler.resource)


def test_no_samples_raises_typed_failure() -> None:
    backend = _ScriptedBackend([])
    sampler = NvmlSampler(backend, device_index=0)
    _failure("no_samples", sampler.resource)


def test_unknown_stage_raises_typed_failure() -> None:
    backend = _ScriptedBackend([1000.0])
    sampler = NvmlSampler(backend, device_index=0)
    _failure("unknown_stage", lambda: sampler.sample("warmup"))
    _failure("unknown_stage", lambda: sampler.sample(""))


def test_all_declared_stages_are_accepted() -> None:
    backend = _ScriptedBackend([100.0 * (index + 1) for index in range(len(STAGES))])
    sampler = NvmlSampler(backend, device_index=0)
    for stage in STAGES:
        record = sampler.sample(stage)
        assert record["stage"] == stage


def test_negative_or_nonfinite_reading_raises_typed_failure() -> None:
    for bad_value in (-1.0, math.nan, math.inf):
        backend = _ScriptedBackend([bad_value])
        sampler = NvmlSampler(backend, device_index=0)
        _failure("invalid_measurement", lambda: sampler.sample("baseline"))


def test_resource_and_samples_have_only_schema_allowed_keys_and_finite_nonnegative_numbers() -> None:
    backend = _ScriptedBackend([1000.0, 1500.0, 1800.0], [None, 300.0, 400.0])
    sampler = NvmlSampler(backend, device_index=0)
    for stage in ("baseline", "session_create", "steady"):
        sample = sampler.sample(stage)
        assert set(sample).issubset(_SAMPLE_ALLOWED_KEYS)
        assert {"stage", "used_mib"}.issubset(sample)
        for key, value in sample.items():
            if key == "stage":
                continue
            assert isinstance(value, (int, float)) and not isinstance(value, bool)
            assert math.isfinite(value) and value >= 0.0

    resource = sampler.resource()
    assert set(resource).issubset(_RESOURCE_ALLOWED_KEYS)
    assert "peak_incremental_device_memory_gib" in resource
    for key, value in resource.items():
        if key == "nvml_samples":
            continue
        assert isinstance(value, (int, float)) and not isinstance(value, bool)
        assert math.isfinite(value) and value >= 0.0


def test_resource_and_samples_validate_against_report_v2_schema_defs() -> None:
    report_schema = load_json(ROOT / "bakeoff/report-v2.schema.json")
    resource_schema = report_schema["$defs"]["resource"]
    sample_schema = report_schema["$defs"]["nvml_sample"]

    backend = _ScriptedBackend([1000.0, 1500.0, 1800.0, 1200.0, 1050.0], [None, 300.0, 400.0, None, 100.0])
    sampler = NvmlSampler(backend, device_index=0)
    for stage in ("baseline", "session_create", "steady", "cleanup", "process_exit"):
        sampler.sample(stage)

    resource = sampler.resource()
    validate(resource, resource_schema, root=report_schema)
    for sample in sampler.samples:
        validate(sample, sample_schema, root=report_schema)


def test_pynvml_backend_missing_dependency_raises_typed_failure() -> None:
    # This development environment has neither pynvml nor a GPU; PynvmlBackend must therefore
    # fail cleanly with a typed nvml_failure rather than an unguarded ImportError/other crash.
    try:
        import pynvml  # type: ignore  # noqa: F401
    except ImportError:
        _failure("runtime_error", PynvmlBackend)
    else:  # pragma: no cover - only reachable on a host with pynvml installed
        pass


def test_poll_captures_a_transient_spike_missed_by_boundary_snapshots() -> None:
    # __enter__'s immediate synchronous read consumes readings[0]; the background thread then
    # advances through readings[1], readings[2]. A pure boundary-snapshot design -- read once
    # at entry, once at exit -- would see only readings[0] and readings[-1] and miss the
    # transient 1800.0 spike recorded in between, which is exactly the gap docs/context.md
    # (Session 6) documents: "boundary-sampled NVML does not capture the rejected allocation."
    readings = [(1000.0, None), (1800.0, None), (1050.0, None)]
    backend = _LatchedBackend(readings)
    sampler = NvmlSampler(backend, device_index=0, poll_interval_s=0.0)

    with sampler.poll("session_create") as window:
        assert isinstance(window, PollWindow)
        window.wait_for_reading_count(len(readings))

    polled_sample = sampler.samples[-1]
    assert polled_sample == {"stage": "session_create", "used_mib": 1800.0}

    boundary_only_peak = max(readings[0][0], readings[-1][0])
    assert boundary_only_peak == 1050.0
    assert boundary_only_peak < polled_sample["used_mib"]

    report_schema = load_json(ROOT / "bakeoff/report-v2.schema.json")
    validate(polled_sample, report_schema["$defs"]["nvml_sample"], root=report_schema)


def test_poll_tracks_device_and_process_maxima_independently_and_thread_safely() -> None:
    # Device peak (1600.0) and process peak (500.0) occur on different scripted readings, so
    # this also proves the two running maxima are tracked independently rather than only the
    # process value paired with the device peak's own reading.
    readings = [(1000.0, 200.0), (900.0, 500.0), (1600.0, 150.0), (1400.0, 480.0)]
    backend = _LatchedBackend(readings)
    sampler = NvmlSampler(backend, device_index=0, poll_interval_s=0.0)

    with sampler.poll("steady") as window:
        window.wait_for_reading_count(len(readings))
        assert window.reading_count >= len(readings)

    polled_sample = sampler.samples[-1]
    assert polled_sample == {"stage": "steady", "used_mib": 1600.0, "process_used_mib": 500.0}


def test_poll_window_always_yields_at_least_one_sample_even_with_no_work() -> None:
    backend = _LatchedBackend([(1234.0, None)])
    # A cadence far longer than the window: the background thread's first wait() will not
    # have elapsed before __exit__ sets the stop event, so it takes zero background readings.
    sampler = NvmlSampler(backend, device_index=0, poll_interval_s=5.0)

    with sampler.poll("session_create"):
        pass

    assert sampler.samples[-1] == {"stage": "session_create", "used_mib": 1234.0}


def test_poll_stops_and_still_records_when_the_polled_block_raises() -> None:
    class _Boom(Exception):
        pass

    readings = [(1700.0, None), (1900.0, None)]
    backend = _LatchedBackend(readings)
    sampler = NvmlSampler(backend, device_index=0, poll_interval_s=0.0)

    raised = False
    try:
        with sampler.poll("session_create") as window:
            window.wait_for_reading_count(len(readings))
            raise _Boom("simulated failure mid session-create")
    except _Boom:
        raised = True
    # Reaching this point at all proves __exit__ (which joins the background thread before
    # letting the exception propagate) already ran to completion -- the poller cannot still
    # be running here.
    assert raised, "expected _Boom to propagate out of the poll() context manager"

    polled_sample = sampler.samples[-1]
    assert polled_sample["stage"] == "session_create"
    assert polled_sample["used_mib"] == 1900.0


def test_poll_rejects_unknown_stage() -> None:
    backend = _LatchedBackend([(1000.0, None)])
    sampler = NvmlSampler(backend, device_index=0)
    _failure("unknown_stage", lambda: sampler.poll("warmup"))


def test_poll_rejects_invalid_interval() -> None:
    backend = _LatchedBackend([(1000.0, None)])
    sampler = NvmlSampler(backend, device_index=0)
    _failure("invalid_interval", lambda: sampler.poll("steady", interval_s=-1.0))
    _failure("invalid_interval", lambda: NvmlSampler(backend, device_index=0, poll_interval_s=-0.5))


def test_resource_incorporates_polled_peaks_alongside_boundary_samples() -> None:
    readings = [(800.0, None), (800.0, None), (2200.0, None), (1900.0, None)]
    backend = _LatchedBackend(readings)
    sampler = NvmlSampler(backend, device_index=0, poll_interval_s=0.0)

    sampler.sample("baseline")
    with sampler.poll("session_create") as window:
        window.wait_for_reading_count(len(readings) - 1)
    sampler.sample("cleanup")

    resource = sampler.resource()
    assert resource["baseline_device_memory_mib"] == 800.0
    assert resource["peak_device_memory_mib"] == 2200.0
    assert math.isclose(resource["peak_incremental_device_memory_gib"], (2200.0 - 800.0) / 1024.0)
    assert resource["cleanup_device_memory_mib"] == 1900.0

    report_schema = load_json(ROOT / "bakeoff/report-v2.schema.json")
    validate(resource, report_schema["$defs"]["resource"], root=report_schema)


def test_nvml_csv_writer_header_rows_and_empty_process_cells() -> None:
    backend = _ScriptedBackend([1000.0, 1500.0], [None, 275.5])
    sampler = NvmlSampler(backend, device_index=0, clock=_clock([10.0, 11.0]))
    sampler.sample("baseline")
    sampler.sample("session_create")
    rows = sampler.csv_rows(IDENTITY)
    assert len(rows) == 2

    with tempfile.TemporaryDirectory(prefix="whitewater-nvml-") as temporary:
        path = Path(temporary) / "nvml.csv"
        write_nvml_csv(path, rows)
        assert stat.S_IMODE(path.stat().st_mode) == 0o644

        with path.open(newline="", encoding="utf-8") as stream:
            read_rows = list(csv.reader(stream))
        assert tuple(read_rows[0]) == NVML_CSV_HEADER
        assert read_rows[1][NVML_CSV_HEADER.index("stage")] == "baseline"
        assert read_rows[1][NVML_CSV_HEADER.index("process_used_mib")] == ""
        assert read_rows[2][NVML_CSV_HEADER.index("stage")] == "session_create"
        assert read_rows[2][NVML_CSV_HEADER.index("process_used_mib")] == "275.5"
        assert read_rows[1][NVML_CSV_HEADER.index("candidate_id")] == IDENTITY["candidate_id"]
        assert read_rows[1][NVML_CSV_HEADER.index("sample_index")] == "0"
        assert read_rows[2][NVML_CSV_HEADER.index("sample_index")] == "1"

        _failure("output_exists", lambda: write_nvml_csv(path, rows))

        more_backend = _ScriptedBackend([1800.0], [None])
        more_sampler = NvmlSampler(more_backend, device_index=0, clock=_clock([12.0]))
        more_sampler.sample("steady")
        append_nvml_csv(path, more_sampler.csv_rows(IDENTITY))
        with path.open(newline="", encoding="utf-8") as stream:
            appended_rows = list(csv.reader(stream))
        assert len(appended_rows) == 4
        assert appended_rows[3][NVML_CSV_HEADER.index("stage")] == "steady"
        assert appended_rows[3][NVML_CSV_HEADER.index("sample_index")] == "0"
        assert stat.S_IMODE(path.stat().st_mode) == 0o644


def test_append_nvml_csv_requires_existing_regular_mode_0644_destination() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-nvml-") as temporary:
        directory = Path(temporary)
        missing = directory / "missing.csv"
        _failure("missing_output", lambda: append_nvml_csv(missing, [["x"]]))

        target = directory / "target.csv"
        target.write_text("header\n", encoding="utf-8")
        os.chmod(target, 0o644)
        link = directory / "link.csv"
        link.symlink_to(target)
        _failure("symlink_output", lambda: append_nvml_csv(link, [["x"]]))

        subdirectory = directory / "directory.csv"
        subdirectory.mkdir()
        _failure("nonregular_output", lambda: append_nvml_csv(subdirectory, [["x"]]))

        wrong_mode = directory / "mode.csv"
        wrong_mode.write_text("header\n", encoding="utf-8")
        os.chmod(wrong_mode, 0o600)
        _failure("output_mode", lambda: append_nvml_csv(wrong_mode, [["x"]]))


def test_write_or_append_creates_then_extends() -> None:
    backend = _ScriptedBackend([1000.0, 1400.0])
    sampler = NvmlSampler(backend, device_index=0, clock=_clock([0.0, 1.0]))
    sampler.sample("baseline")
    rows_a = sampler.csv_rows(IDENTITY)
    sampler.sample("cleanup")
    rows_b = [sampler.csv_rows(IDENTITY)[-1]]

    with tempfile.TemporaryDirectory(prefix="whitewater-nvml-") as temporary:
        path = Path(temporary) / "nvml.csv"
        write_or_append_nvml_csv(path, rows_a)
        write_or_append_nvml_csv(path, rows_b)
        with path.open(newline="", encoding="utf-8") as stream:
            read_rows = list(csv.reader(stream))
        assert len(read_rows) == 3
        assert tuple(read_rows[0]) == NVML_CSV_HEADER


def test_csv_writer_rejects_malformed_identity() -> None:
    backend = _ScriptedBackend([1000.0])
    sampler = NvmlSampler(backend, device_index=0)
    sampler.sample("baseline")
    bad_identity = dict(IDENTITY)
    del bad_identity["provider"]
    _failure("identity_shape", lambda: sampler.csv_rows(bad_identity))

    empty_field_identity = dict(IDENTITY)
    empty_field_identity["shot_id"] = ""
    _failure("identity_shape", lambda: sampler.csv_rows(empty_field_identity))


def main() -> int:
    test_sample_produces_schema_shaped_records_with_and_without_process()
    test_resource_computes_peak_incremental_and_stage_snapshots()
    test_resource_peak_incremental_never_negative_when_below_baseline()
    test_resource_uses_first_baseline_when_sampled_more_than_once()
    test_missing_baseline_raises_typed_failure()
    test_no_samples_raises_typed_failure()
    test_unknown_stage_raises_typed_failure()
    test_all_declared_stages_are_accepted()
    test_negative_or_nonfinite_reading_raises_typed_failure()
    test_resource_and_samples_have_only_schema_allowed_keys_and_finite_nonnegative_numbers()
    test_resource_and_samples_validate_against_report_v2_schema_defs()
    test_pynvml_backend_missing_dependency_raises_typed_failure()
    test_poll_captures_a_transient_spike_missed_by_boundary_snapshots()
    test_poll_tracks_device_and_process_maxima_independently_and_thread_safely()
    test_poll_window_always_yields_at_least_one_sample_even_with_no_work()
    test_poll_stops_and_still_records_when_the_polled_block_raises()
    test_poll_rejects_unknown_stage()
    test_poll_rejects_invalid_interval()
    test_resource_incorporates_polled_peaks_alongside_boundary_samples()
    test_nvml_csv_writer_header_rows_and_empty_process_cells()
    test_append_nvml_csv_requires_existing_regular_mode_0644_destination()
    test_write_or_append_creates_then_extends()
    test_csv_writer_rejects_malformed_identity()
    print("P25-6 nvml tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
