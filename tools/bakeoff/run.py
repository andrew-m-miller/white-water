#!/usr/bin/env python3
"""P25-6 end-to-end, resumable offline profile driver (WP4).

This module wires together the already-built P25-4/P25-5/P25-6 bake-off modules
(``matrix``, ``resume``, ``coordinator``, ``evaluator``, ``exr``, ``synthetic``, ``nvml``,
``measurement``, ``metrics``, ``reporting``, ``validator``) into one resumable command that
runs a ``smoke``/``screen``/``final`` profile and publishes exactly the five operator return
files described in ``docs/phase2.5-implementation-plan.md`` ("P25-6 - Airgapped target
measurement"):

    report.json  summary.txt  runner.log  nvml.csv  review.csv

It does not reimplement matrix planning, resume-state transitions, report assembly, EXR
decoding, NVML sampling, or metric math -- it only resolves shots/artifacts, runs inference
through :class:`tools.bakeoff.evaluator.Evaluator`, and shapes the results those modules
already expect.

Metric scope (documented here because the exact set is a driver decision, not a contract
frozen elsewhere):

* ``nonfinite_fraction`` and ``repeated_run_p99_delta_px`` are always taken from
  :meth:`Evaluator.run_nchw_pair`'s own reduction.
* ``endpoint_error_px``/``fraction_le_1px``/``fraction_le_3px`` (dense metrics) are computed
  whenever the shot's corpus truth is ``analytic`` (the synthetic partition); this is mandatory
  once a shot declares analytic truth, so a cell that cannot produce them fails rather than
  silently degrading (``docs/phase2.5-protocol-v2.md`` and the report validator both treat
  these as required whenever analytic truth exists).
* ``chain_drift_px`` is computed whenever the shot declares ``chain_length``; likewise
  mandatory once declared, so a cell that cannot complete the chain fails.
* ``visible_warp_residual`` and ``forward_backward_residual_px`` are self-consistency metrics
  that need no truth (both source frames plus the model's own forward/backward flow). They are
  computed opportunistically and degrade to ``not_applicable`` -- a typed, non-fatal skip -- when
  the needed auxiliary frame or metric computation is unavailable.
* ``landmark_median_error_px``/``landmark_p95_error_px`` are always ``not_applicable``: no
  landmark-annotation ingestion exists yet in this repository (see docs/phase2.5-implementation-
  plan.md, "Point annotations are optional during screening..."). This is a known limitation,
  called out again in the PR description, and does not affect any shot in the checked-in corpus
  since none declares ``truth.kind == "landmarks"``.

Every runtime/array/NVML/EXR-decoder dependency is injected so this module -- and
``run_tests.py`` -- import and run without numpy, onnxruntime, OpenImageIO, pynvml, or a GPU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as _mp
import os
import platform as platform_module
import queue as _queue_module
import struct
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from . import exr as exr_module
    from . import synthetic as synthetic_module
    from .coordinator import CoordinatorFailure, IncompleteFailure, RunCoordinator
    from .evaluator import (
        DependencyFailure,
        Evaluator,
        EvaluatorFailure,
        PROVIDER_EXECUTION_NAMES,
        REPORT_METRICS,
        ValidatedArtifact,
        V2_PROTOCOL,
        condition_and_pad_pair,
        validate_manifest_artifact,
    )
    from .exr import ExrFailure
    from .matrix import CellKey, MatrixFailure, MatrixPlan, build_matrix
    from . import metrics as metrics_module
    from .metrics import MetricFailure
    from .nvml import NVML_CSV_HEADER, NvmlBackend, NvmlSampler, PynvmlBackend
    from .resume import ResumeFailure, create_state, load_state
    from .reporting import ReportFailure, assemble_report, write_report_pair
    from .synthetic import SyntheticCase, write_pfm as synthetic_write_pfm
    from .validator import canonical_sha256, load_json
except ImportError:  # pragma: no cover - supports direct air-gapped invocation
    import exr as exr_module  # type: ignore
    import synthetic as synthetic_module  # type: ignore
    from coordinator import CoordinatorFailure, IncompleteFailure, RunCoordinator  # type: ignore
    from evaluator import (  # type: ignore
        DependencyFailure,
        Evaluator,
        EvaluatorFailure,
        PROVIDER_EXECUTION_NAMES,
        REPORT_METRICS,
        ValidatedArtifact,
        V2_PROTOCOL,
        condition_and_pad_pair,
        validate_manifest_artifact,
    )
    from exr import ExrFailure  # type: ignore
    from matrix import CellKey, MatrixFailure, MatrixPlan, build_matrix  # type: ignore
    import metrics as metrics_module  # type: ignore
    from metrics import MetricFailure  # type: ignore
    from nvml import NVML_CSV_HEADER, NvmlBackend, NvmlSampler, PynvmlBackend  # type: ignore
    from resume import ResumeFailure, create_state, load_state  # type: ignore
    from reporting import ReportFailure, assemble_report, write_report_pair  # type: ignore
    from synthetic import SyntheticCase, write_pfm as synthetic_write_pfm  # type: ignore
    from validator import canonical_sha256, load_json  # type: ignore


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = V2_PROTOCOL
DEFAULT_REPORT_SCHEMA = ROOT / "bakeoff" / "report-v2.schema.json"
DEFAULT_CORPUS_SCHEMA = ROOT / "bakeoff" / "corpus-v1.schema.json"
DEFAULT_CHAIN_OFFSETS: tuple[int, ...] = (1, 2, 4, 8)

# The stable, driver-owned review.csv header. Rows are anonymous: candidate_id is replaced by
# a deterministic pseudonymous label so a human review pass does not see which candidate is
# which while scoring. See ``_review_label``.
REVIEW_CSV_HEADER: tuple[str, ...] = (
    "candidate_label", "shot_id", "category", "conditioning_token", "cap_token", "provider",
    "host_load", "preview_path", "edge_adherence", "occlusion_reveal", "blur", "jitter", "drift", "notes",
)

# Fix N: which REVIEW_CSV_HEADER columns identify a row's CellKey (indices into the header
# tuple above) versus the trailing human-edited columns a repair pass must never blank. Deliberately
# NOT the full 8-column driver-owned prefix: "category" and "preview_path" are driver-owned but
# derived metadata (shot category lookup, review-preview path), not row identity, so a corrupted
# value in either of those two must still be repairable without losing that row's human columns --
# only candidate_label/shot_id/conditioning_token/cap_token/provider/host_load (which together
# mirror a CellKey exactly, once candidate_id is replaced by its pseudonymous label) are load-bearing
# for matching a row across a repair.
_REVIEW_IDENTITY_COLUMNS: tuple[int, ...] = (0, 1, 3, 4, 5, 6)
_REVIEW_HUMAN_COLUMN_COUNT = 6  # edge_adherence, occlusion_reveal, blur, jitter, drift, notes

_OUTPUT_FILE_MODE = 0o644


class DriverFailure(ValueError):
    """Stable, reportable driver-level (non-per-cell) failure."""

    def __init__(self, kind: str, message: str):
        self.kind = kind
        self.reason = kind
        self.failure_type = "driver_failure"
        self.message = message
        super().__init__(f"{kind}: {message}")


def _fail(kind: str, message: str) -> None:
    raise DriverFailure(kind, message)


# --------------------------------------------------------------------------------------------
# Small typed control-flow helpers for the per-cell executor.
# --------------------------------------------------------------------------------------------


class _CellOutcome(Exception):
    """Base for the two typed non-pass per-cell outcomes."""

    def __init__(self, failure: dict[str, Any]):
        self.failure = failure
        super().__init__(failure.get("message", failure.get("type", "cell outcome")))


class _CellFail(_CellOutcome):
    pass


class _CellSkip(_CellOutcome):
    pass


def _failure(kind: str, message: str, *, stage: str | None = None, retryable: bool | None = None) -> dict[str, Any]:
    failure: dict[str, Any] = {"type": kind, "message": message}
    if stage is not None:
        failure["stage"] = stage
    if retryable is not None:
        failure["retryable"] = retryable
    return failure


_EXR_FAILURE_TYPES = {
    "missing_file": "missing_input",
    "target_out_of_range": "missing_input",
    "empty_range": "missing_input",
    "reference_out_of_range": "missing_input",
    "corpus_shape": "missing_input",
    "malformed_frame_token": "missing_input",
    "multiple_frame_tokens": "missing_input",
    "missing_frame_token": "missing_input",
    "decode_error": "input_invalid",
    "nonfinite_sample": "input_invalid",
    "geometry_mismatch": "input_invalid",
    "layout_mismatch": "input_invalid",
    "metadata_mismatch": "input_invalid",
    "input_invalid": "input_invalid",
    "unsupported_channels": "unsupported_tensor_contract",
    "unsupported_storage": "unsupported_tensor_contract",
    "runtime_error": "runtime_error",
}


def _exr_failure(exc: ExrFailure, *, stage: str) -> dict[str, Any]:
    return _failure(_EXR_FAILURE_TYPES.get(exc.kind, "input_invalid"), str(exc), stage=stage)


# --------------------------------------------------------------------------------------------
# Cap lookup (mirrors the private evaluator._cap_value pattern without importing it).
# --------------------------------------------------------------------------------------------


def _cap_megapixels(protocol: Mapping[str, Any], token: str) -> float:
    for cap in protocol.get("analysis_caps", []):
        if isinstance(cap, Mapping) and cap.get("token") == token:
            return float(cap["decimal_megapixels"])
    _fail("unknown_cap", f"unknown analysis cap token: {token!r}")


# --------------------------------------------------------------------------------------------
# Corpus/shot indexing.
# --------------------------------------------------------------------------------------------


def _shot_index(corpus: Mapping[str, Any]) -> dict[str, tuple[str, Mapping[str, Any]]]:
    """Return ``shot_id -> (partition_kind, shot)`` across every corpus partition."""

    index: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for partition in corpus.get("partitions", []):
        kind = partition.get("kind")
        for shot in partition.get("shots", []):
            shot_id = shot.get("id")
            if isinstance(shot_id, str):
                index[shot_id] = (kind, shot)
    return index


def _frame_paths(shot: Mapping[str, Any]) -> dict[int, str]:
    return dict(exr_module.expand_shot_sequence(shot))


# --------------------------------------------------------------------------------------------
# Frame loading: synthetic (analytic) and production (EXR) partitions share one frame shape.
# --------------------------------------------------------------------------------------------


def _synthetic_frame_hash(rows: Sequence[Sequence[Sequence[float]]]) -> str:
    """Deterministic content hash for a generated frame (no source file to hash)."""

    digest = hashlib.sha256()
    for row in rows:
        for pixel in row:
            for channel in pixel:
                digest.update(struct.pack("<f", float(channel)))
    return digest.hexdigest()


def _synthetic_frame(case: SyntheticCase, frame_number: int, pixel_aspect_ratio: float) -> dict[str, Any]:
    rows = tuple(synthetic_module.frame_rows(case, frame_number))
    return {
        "width": case.width,
        "height": case.height,
        "channels": 3,
        "rows": rows,
        "pixel_aspect_ratio": float(pixel_aspect_ratio),
        "frame": frame_number,
        "sha256": _synthetic_frame_hash(rows),
        "source": f"generated://{case.path_token}/frame.{frame_number:04d}.pfm",
    }


@dataclass(frozen=True)
class _ShotContext:
    shot_id: str
    partition_kind: str
    shot: Mapping[str, Any]
    synthetic_case: SyntheticCase | None
    frame_paths: Mapping[int, str]

    @property
    def has_analytic_truth(self) -> bool:
        truth = self.shot.get("truth")
        return isinstance(truth, Mapping) and truth.get("kind") == "analytic"

    @property
    def chain_length(self) -> int | None:
        value = self.shot.get("chain_length")
        return value if isinstance(value, int) and not isinstance(value, bool) else None


def _build_shot_context(
    shot_id: str,
    partition_kind: str,
    shot: Mapping[str, Any],
    synthetic_cases: Mapping[str, SyntheticCase],
) -> _ShotContext:
    if partition_kind == "synthetic":
        case_id = shot.get("case_id")
        case = synthetic_cases.get(case_id) if isinstance(case_id, str) else None
        if case is None:
            _fail("unknown_synthetic_case", f"synthetic shot {shot_id!r} has no known case_id {case_id!r}")
        return _ShotContext(shot_id, partition_kind, shot, case, {})
    return _ShotContext(shot_id, partition_kind, shot, None, _frame_paths(shot))


def _load_frame(
    context: _ShotContext,
    frame_number: int,
    *,
    exr_decoder: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    par = float(context.shot.get("pixel_aspect_ratio", 1.0))
    if context.partition_kind == "synthetic":
        assert context.synthetic_case is not None
        first, last = context.synthetic_case.first_frame, context.synthetic_case.last_frame
        if not (first <= frame_number <= last):
            raise ExrFailure("target_out_of_range", f"frame {frame_number} outside [{first}, {last}]")
        return _synthetic_frame(context.synthetic_case, frame_number, par)
    path = context.frame_paths.get(frame_number)
    if path is None:
        raise ExrFailure("target_out_of_range", f"frame {frame_number} is outside the shot range")
    return exr_decoder(path, frame_number=frame_number, pixel_aspect_ratio=par)


def _load_pair(
    context: _ShotContext,
    frame_a: int,
    frame_b: int,
    *,
    exr_decoder: Callable[..., dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    first = _load_frame(context, frame_a, exr_decoder=exr_decoder)
    second = _load_frame(context, frame_b, exr_decoder=exr_decoder)
    exr_module.validate_pair_geometry(first, second)
    if context.partition_kind != "synthetic":
        exr_module.validate_pair_layout(first, second)
        exr_module.validate_frame_matches_shot_metadata(context.shot, first)
        exr_module.validate_frame_matches_shot_metadata(context.shot, second)
    return first, second


# --------------------------------------------------------------------------------------------
# NCHW conversion and grid reshaping.
# --------------------------------------------------------------------------------------------


def _to_array(array_module: Any, nchw: Any) -> Any:
    return array_module.ascontiguousarray(array_module.asarray((nchw,), dtype=array_module.float32))


def _unpadded_grid(
    nchw: Sequence[Sequence[Sequence[float]]], analysis_width: int, analysis_height: int
) -> list[list[tuple[float, ...]]]:
    """Crop a padded ``[channel][y][x]`` tensor to its bottom-left unpadded analysis region.

    Every padding policy this evaluator supports pads only on the right/top (the caller-side
    ``pad_rows`` helper is always invoked with ``left=bottom=0``; ``none``/``graph-internal``
    apply no padding at all), so the unpadded analysis content is always exactly the bottom-left
    ``analysis_width`` x ``analysis_height`` region -- no separate offset bookkeeping is needed.
    """

    return [
        [tuple(nchw[channel][y][x] for channel in range(len(nchw))) for x in range(analysis_width)]
        for y in range(analysis_height)
    ]


# --------------------------------------------------------------------------------------------
# Dense analytic truth for the synthetic partition.
# --------------------------------------------------------------------------------------------


def _dense_truth_and_mask(
    case: SyntheticCase,
    from_frame: int,
    to_frame: int,
    analysis_width: int,
    analysis_height: int,
    spacing_x: float,
    spacing_y: float,
) -> tuple[list[list[tuple[float, float] | None]], list[list[bool]]]:
    grid: list[list[tuple[float, float] | None]] = []
    mask: list[list[bool]] = []
    for row in range(analysis_height):
        source_y = row * spacing_y
        grid_row: list[tuple[float, float] | None] = []
        mask_row: list[bool] = []
        for column in range(analysis_width):
            source_x = column * spacing_x
            truth = synthetic_module.analytic_pair_truth(case, from_frame, to_frame, source_x, source_y)
            if truth.no_dense_truth:
                grid_row.append(None)
                mask_row.append(False)
            else:
                dx, dy = truth.displacement  # type: ignore[misc]
                grid_row.append((dx / spacing_x, dy / spacing_y))
                mask_row.append(True)
        grid.append(grid_row)
        mask.append(mask_row)
    return grid, mask


# --------------------------------------------------------------------------------------------
# One inference pair (base or auxiliary), returning run_nchw_pair's raw dict plus geometry.
# --------------------------------------------------------------------------------------------


def _infer_pair(
    evaluator_instance: Evaluator,
    artifact: ValidatedArtifact,
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    provider: str,
    conditioning_token: str,
    cap_megapixels: float,
    array_module: Any,
    profile: str,
    stage_sampler: Callable[[str], Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Any, Any]:
    """Condition, pad, run one pair, and return ``(result, geometry, metadata, first_nchw, second_nchw)``."""

    preprocessing_start = time.perf_counter()
    first_nchw, second_nchw, geometry, metadata = condition_and_pad_pair(
        first, second,
        conditioning_token=conditioning_token,
        cap_megapixels=cap_megapixels,
        manifest=artifact.manifest,
    )
    first_array = _to_array(array_module, first_nchw)
    second_array = _to_array(array_module, second_nchw)
    preprocessing_ms = (time.perf_counter() - preprocessing_start) * 1000.0
    result = evaluator_instance.run_nchw_pair(
        first_array, second_array,
        provider=provider,
        analysis_width=geometry["analysis_width"],
        analysis_height=geometry["analysis_height"],
        padded_width=geometry["padded_width"],
        padded_height=geometry["padded_height"],
        profile=profile,
        geometry=geometry,
        preprocessing_ms=preprocessing_ms,
        stage_sampler=stage_sampler,
    )
    return result, geometry, metadata, first_nchw, second_nchw  # type: ignore[return-value]


# --------------------------------------------------------------------------------------------
# Review label and blinded per-candidate previews.
# --------------------------------------------------------------------------------------------


def _review_label(shot_id: str, candidate_id: str) -> str:
    """A deterministic pseudonymous per-shot candidate label for a blind review pass."""

    digest = hashlib.sha256(f"{shot_id}:{candidate_id}".encode("utf-8")).hexdigest()
    return f"candidate-{digest[:12]}"


def _review_preview_dir(review_dir: Path, cell: CellKey) -> Path:
    """The per-cell preview directory, keyed only by the anonymized label -- never the real
    candidate id -- plus shot/cap/conditioning. Shared by the writer (during a cell's execution)
    and the end-of-run review.csv regeneration (Fix E), so both agree on the same path without
    needing any extra durable state."""

    return review_dir / cell.shot / _review_label(cell.shot, cell.candidate) / cell.cap / cell.conditioning


def _flow_visualization_grid(
    flow: Sequence[Sequence[tuple[float, float]]], width: int, height: int, *, scale: float = 8.0
) -> list[list[tuple[float, float, float]]]:
    """A small, dependency-free color encoding of a flow field for a human review preview.

    Not a metric and not reused anywhere else: R/G encode signed dx/dy (clamped to [-scale,
    scale] and mapped to [0, 1]), B encodes normalized magnitude. Deterministic and cheap; a PFM
    viewer is enough to read direction/magnitude by eye without needing real HSV wheel plumbing.
    """

    grid: list[list[tuple[float, float, float]]] = []
    for row in range(height):
        out_row: list[tuple[float, float, float]] = []
        for column in range(width):
            dx, dy = flow[row][column]
            magnitude = min(1.0, math.hypot(dx, dy) / max(scale, 1e-6))
            red = min(1.0, max(0.0, 0.5 + dx / (2.0 * scale)))
            green = min(1.0, max(0.0, 0.5 + dy / (2.0 * scale)))
            out_row.append((red, green, magnitude))
        grid.append(out_row)
    return grid


def _grid_or_zero(
    grid: Sequence[Sequence[tuple[float, ...] | None]], width: int, height: int
) -> list[list[tuple[float, float, float]]]:
    """Replace out-of-bounds (``None``) warp samples with black so the result is a writable PFM.

    A ``None`` cell already means "this destination pixel is not measurable" (advected outside
    the target frame); the metric functions drop it, but a preview still needs one value per
    pixel, so an out-of-bounds sample renders as black rather than being silently omitted.
    """

    return [
        [(0.0, 0.0, 0.0) if grid[row][column] is None else tuple(grid[row][column][:3]) for column in range(width)]  # type: ignore[index]
        for row in range(height)
    ]


def _write_offset_preview(
    review_dir: Path, cell: CellKey, offset: int, image2: Any, flow: Any, width: int, height: int,
) -> None:
    """Write one offset's blinded per-candidate warp + flow-visualization PFM pair.

    Reuses :func:`metrics.warp_forward_samples` (the same warp step
    :func:`metrics.visible_warp_residual` scores) rather than duplicating the bilinear-advection
    math here. Filenames carry only the offset number, never the candidate id -- the directory
    itself (``_review_preview_dir``) is already keyed by the anonymized label.
    """

    destination = _review_preview_dir(review_dir, cell)
    destination.mkdir(parents=True, exist_ok=True)
    all_visible = [[True] * width for _ in range(height)]
    warped = metrics_module.warp_forward_samples(image2, flow, all_visible)
    synthetic_write_pfm(destination / f"offset_{offset}_warped.pfm", _grid_or_zero(warped, width, height), width, height)
    synthetic_write_pfm(destination / f"offset_{offset}_flow.pfm", _flow_visualization_grid(flow, width, height), width, height)


# --------------------------------------------------------------------------------------------
# Fix B: supervised CUDA host-load boundary confirmation.
# --------------------------------------------------------------------------------------------

HostLoadCheckpoint = Callable[[str], None]

_HOST_LOAD_PROMPTS = {
    "idle": "Confirm the CUDA device is IDLE: no live Flame Batch workload is running.",
    "live_flame": "Confirm a live Flame Batch workload IS running on the CUDA device.",
}


def interactive_host_load_checkpoint(host_load: str) -> None:
    """Default :data:`HostLoadCheckpoint`: blocks on an interactive stdin confirmation.

    Used for every group of consecutive cells sharing one CUDA ``host_load`` value, before the
    first cell of that group runs, so a ``final`` matrix's alternating idle/live_flame CUDA rows
    are never silently measured under whatever Flame state happens to already exist.
    """

    prompt = _HOST_LOAD_PROMPTS.get(host_load, f"Confirm host_load={host_load!r} before continuing.")
    input(f"{prompt} Press Enter to confirm, or Ctrl-C to abort: ")


def auto_confirm_host_load_checkpoint(host_load: str) -> None:
    """Non-interactive :data:`HostLoadCheckpoint` for ``--assume-host-load-ready`` unattended
    reruns. Still logged by the caller; simply skips the interactive block."""


# --------------------------------------------------------------------------------------------
# Fix D: CUDA measurement isolation (subprocess) and the true post-exit reading.
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CudaMeasurementResult:
    """What one CUDA cell's isolated measurement run hands back to the driver."""

    payload: dict[str, Any]
    samples: list[dict[str, Any]]
    process_exit_used_mib: float | None
    process_exit_process_used_mib: float | None


# work(stage_sampler) performs the cell's condition+infer call and returns a plain-JSON payload;
# a CudaMeasurementRunner is responsible for calling it under whatever isolation it provides and
# returning the NVML evidence collected around it, including a stage_exit reading taken only
# after that isolation has genuinely ended.
CudaMeasurementRunner = Callable[
    [Callable[[Callable[[str], Any]], dict[str, Any]], Callable[[], NvmlBackend], int, float],
    CudaMeasurementResult,
]


def _measure_cuda_work_once(
    work: Callable[[Callable[[str], Any]], dict[str, Any]],
    nvml_backend_factory: Callable[[], NvmlBackend],
    device_index: int,
    poll_interval_s: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Baseline -> (session_create+first_inference, steady polled) -> cleanup, once.

    Shared by the real subprocess runner (executed inside the child) and, indirectly, by
    anything that wants the identical staged-sampling sequence without duplicating it.
    """

    backend = nvml_backend_factory()
    sampler = NvmlSampler(backend, device_index, poll_interval_s=poll_interval_s)
    sampler.sample("baseline")
    payload = work(lambda name: sampler.poll(name))
    sampler.sample("cleanup")
    return payload, sampler.samples


def _map_child_failure_kind(kind: str | None) -> str:
    """Map a child-reported exception ``.kind`` onto a permitted coordinator failure type.

    An ``ExrFailure`` kind (e.g. ``"missing_file"``) goes through the same
    ``_EXR_FAILURE_TYPES`` table the in-process/CPU path already uses, so a frame-loading
    failure that happened inside the isolated child is reported exactly as it would have been
    if it had happened in-process. Anything else already in ``_RESULT_FAILURE_TYPES`` (an
    ``EvaluatorFailure``/``DependencyFailure`` kind) passes through unchanged; anything
    unrecognized (including no ``kind`` at all) falls back to ``"runtime_error"``.
    """

    if kind in _EXR_FAILURE_TYPES:
        return _EXR_FAILURE_TYPES[kind]
    if kind in _RESULT_FAILURE_TYPES:
        return kind
    return "runtime_error"


def _drain_child_outcome(process: Any, result_queue: Any, *, poll_interval_s: float = 0.05) -> tuple | None:
    """Wait for the child's queued outcome without ever blocking indefinitely (Fix H).

    A plain ``queue.get()`` blocks forever if the child dies without queueing anything (OOM-
    killed, segfaulted, or an explicit ``os._exit()`` before ``queue.put``) -- ``queue.get()``
    has no way to observe that the writer is gone. This polls with a short timeout while the
    child is alive; the decision to give up is keyed on ``process.is_alive()`` becoming False,
    not on a fixed wall-clock budget, so a legitimately long-running ``final``-profile cell is
    never aborted early. Once the process is no longer alive, one final non-blocking drain
    covers the narrow race where the child queued its result and then exited between two polls.
    Returns ``None`` if the child exited without ever producing an outcome.
    """

    while True:
        try:
            return result_queue.get(timeout=poll_interval_s)
        except _queue_module.Empty:
            if not process.is_alive():
                try:
                    return result_queue.get_nowait()
                except _queue_module.Empty:
                    return None


def run_cuda_measurement_in_subprocess(
    work: Callable[[Callable[[str], Any]], dict[str, Any]],
    nvml_backend_factory: Callable[[], NvmlBackend],
    device_index: int,
    poll_interval_s: float,
) -> CudaMeasurementResult:
    """Default :data:`CudaMeasurementRunner`: isolates the measured work in a forked child.

    Three problems this fixes relative to running everything in-process: (1) an in-process
    "process_exit" sample is meaningless -- the driver's own process, runtime module, and later
    cells are all still alive, so it can never reflect the measured work's process actually
    tearing down; (2) CUDA context/driver teardown only happens when the process that created
    it actually exits, so the PARENT must never itself touch CUDA (see ``work``'s contract:
    every inference the cell needs, not just the base pair, must happen inside this child --
    enforced by ``_perform_cell_inference`` being the only thing ever passed as ``work`` for a
    CUDA cell); and (3) a child that dies without queueing a result must not hang the parent
    forever (:func:`_drain_child_outcome`).

    Forking BEFORE any CUDA/NVML initialization (the child initializes its own NVML handle
    fresh, after the fork, never before) is the standard-safe pattern; the parent takes its own
    fresh device-wide reading only after ``process.join()`` has confirmed the child has actually
    exited and been reaped.

    Only the queued return value crosses the process boundary (a plain dict/list of picklable
    values); ``work`` itself runs as an inherited closure under fork, so no pickling of the
    runtime/array modules or the frame data is required.
    """

    ctx = _mp.get_context("fork")
    result_queue: Any = ctx.Queue()

    def _child_main() -> None:
        try:
            payload, samples = _measure_cuda_work_once(work, nvml_backend_factory, device_index, poll_interval_s)
        except BaseException as exc:  # noqa: BLE001 - reported to the parent, not raised here
            kind = getattr(exc, "kind", None)
            stage = "load_input" if isinstance(exc, ExrFailure) else "inference"
            result_queue.put(("error", kind, stage, str(exc)))
            return
        result_queue.put(("ok", payload, samples))

    process = ctx.Process(target=_child_main)
    process.start()
    outcome = _drain_child_outcome(process, result_queue)
    process.join()

    if outcome is None:
        kind = "out_of_memory" if process.exitcode == -9 else "runtime_error"
        raise DriverFailure(
            kind,
            f"CUDA measurement subprocess exited without reporting a result (exitcode={process.exitcode!r})",
        )
    if outcome[0] == "error":
        _, kind, stage, message = outcome
        raise DriverFailure(_map_child_failure_kind(kind), f"[{stage}] CUDA measurement subprocess failed: {message}")
    _, payload, samples = outcome

    exit_backend = nvml_backend_factory()
    handle = exit_backend.device_handle(device_index)
    used_mib = exit_backend.device_used_mib(handle)
    try:
        process_used_mib = exit_backend.process_used_mib(handle, process.pid)
    except Exception:  # noqa: BLE001 - per-process accounting for an exited pid is best-effort
        process_used_mib = None
    return CudaMeasurementResult(payload, samples, used_mib, process_used_mib)


# --------------------------------------------------------------------------------------------
# Fix E: durable per-cell NVML CSV rows, regenerated (never appended) at end-of-run.
# --------------------------------------------------------------------------------------------


def _cell_sidecar_key(cell: CellKey) -> str:
    return hashlib.sha256(json.dumps(cell.as_dict(), sort_keys=True).encode("utf-8")).hexdigest()


def _nvml_sidecar_path(output_dir: Path, cell: CellKey) -> Path:
    return output_dir / ".sidecars" / f"{_cell_sidecar_key(cell)}.nvml.json"


def _write_nvml_sidecar(output_dir: Path, cell: CellKey, rows: Sequence[Sequence[str]]) -> None:
    """Durably record one cell's rendered nvml.csv rows, keyed by cell identity.

    Written (fully overwriting any earlier attempt for the same cell) BEFORE the coordinator
    marks the cell complete. If the process dies before completion, resume resets the cell to
    pending and it re-executes from scratch -- this sidecar is simply overwritten again by the
    new attempt, so end-of-run regeneration (which reads exactly one sidecar per plan cell) can
    never emit duplicate rows for a cell, regardless of how many partial attempts occurred. This
    is what makes nvml.csv a deterministic function of durable completed state rather than an
    incremental append that could double-record a retried cell.
    """

    path = _nvml_sidecar_path(output_dir, cell)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"rows": [list(row) for row in rows]}, sort_keys=True).encode("utf-8")
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _OUTPUT_FILE_MODE)
    try:
        os.fchmod(descriptor, _OUTPUT_FILE_MODE)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1  # ownership transferred to the file object; only its close() now
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise


def _read_nvml_sidecar(output_dir: Path, cell: CellKey) -> list[list[str]]:
    path = _nvml_sidecar_path(output_dir, cell)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    return [list(row) for row in data.get("rows", [])]


# --------------------------------------------------------------------------------------------
# Per-cell executor.
# --------------------------------------------------------------------------------------------


@dataclass
class _RunnerContext:
    protocol: Mapping[str, Any]
    profile: str
    shot_contexts: Mapping[str, _ShotContext]
    artifacts: Mapping[str, ValidatedArtifact]
    runtime_module: Any
    array_module: Any
    nvml_backend_factory: Callable[[], NvmlBackend] | None
    device_index: int
    poll_interval_s: float
    chain_offsets: Sequence[int]
    exr_decoder: Callable[..., dict[str, Any]]
    output_dir: Path
    review_dir: Path | None
    log: Callable[[str], None]
    host_load_checkpoint: HostLoadCheckpoint
    cuda_measurement_runner: CudaMeasurementRunner
    confirmed_host_load: str | None = field(default=None, init=False)


def _base_result_fields(cell: CellKey) -> dict[str, Any]:
    """Also matches the identity key names ``nvml.csv_rows`` expects (candidate_id, shot_id,
    conditioning_token, cap_token, provider, host_load) -- ``CellKey.as_dict()`` uses the
    shorter matrix-internal field names instead, so this is the shared translation."""

    return {
        "candidate_id": cell.candidate,
        "shot_id": cell.shot,
        "conditioning_token": cell.conditioning,
        "cap_token": cell.cap,
        "provider": cell.provider,
        "host_load": cell.host_load,
    }


class _ReplayNvmlBackend:
    """Feeds already-recorded ``nvml_sample`` dicts back through a fresh :class:`NvmlSampler`.

    ``.resource()``/``.csv_rows()`` reduction logic is owned by ``nvml.py``; this lets a caller
    that already has a plain list of recorded samples (e.g. a CUDA measurement result that
    crossed a subprocess boundary, where the original live sampler no longer exists) reuse that
    logic exactly rather than duplicating the peak/baseline arithmetic here.
    """

    def __init__(self, samples: Sequence[Mapping[str, Any]]):
        self._samples = list(samples)
        self.index = 0

    def device_handle(self, device_index: int) -> Any:
        return device_index

    def device_used_mib(self, handle: Any) -> float:
        return self._samples[self.index]["used_mib"]

    def process_used_mib(self, handle: Any, pid: int) -> float | None:
        return self._samples[self.index].get("process_used_mib")


def _replay_resource_and_rows(
    identity_fields: Mapping[str, str], samples: Sequence[Mapping[str, Any]], device_index: int,
) -> tuple[dict[str, Any], list[list[str]]]:
    backend = _ReplayNvmlBackend(samples)
    sampler = NvmlSampler(backend, device_index, poll_interval_s=0.0)
    for index, sample in enumerate(samples):
        backend.index = index
        sampler.sample(sample["stage"])
    return sampler.resource(), sampler.csv_rows(identity_fields)


def _perform_cell_inference(
    cell: CellKey,
    ctx: _RunnerContext,
    artifact: ValidatedArtifact,
    shot_ctx: _ShotContext,
    stage_sampler: Callable[[str], Any] | None,
) -> dict[str, Any]:
    """Load every frame pair and run every inference this cell needs, and nothing else.

    Fix G: for a CUDA cell this ENTIRE function is what runs as ``work`` inside
    :func:`run_cuda_measurement_in_subprocess`'s isolated child -- the base pair, the reverse
    pair (forward/backward residual), every chain link, and every review-offset pair. Nothing
    outside this function may touch CUDA for that cell: the parent (see ``_run_cell`) only
    scores metrics and renders previews from the plain data this function returns, and the
    ``process_exit`` NVML reading is taken only after the child running this function has
    actually exited. Splitting it out this way is also what makes a CPU/CoreML cell able to run
    it in-process, unchanged, with no isolation at all.

    Returns one picklable payload (safe to cross the subprocess boundary via a
    ``multiprocessing.Queue``): the base pair's flow/geometry/timing/environment/metrics/
    conditioning/NCHW arrays, its input frame hashes, an optional reverse-pair flow, an optional
    list of chain-link flows (or a typed ``chain_error`` message if a *mandatory* chain could
    not be completed), an optional per-offset review preview bundle, and any skip messages the
    parent should log (kept out of this function so it never needs to touch the parent's
    already-open log file handle across the fork boundary).
    """

    cap_megapixels = _cap_megapixels(ctx.protocol, cell.cap)
    evaluator_instance = Evaluator(artifact, ctx.runtime_module, ctx.array_module)
    reference_frame = int(shot_ctx.shot["reference_frame"])
    log_messages: list[str] = []

    first, second = _load_pair(shot_ctx, reference_frame, reference_frame + 1, exr_decoder=ctx.exr_decoder)
    base_result, geometry, cond_meta, first_nchw, second_nchw = _infer_pair(
        evaluator_instance, artifact, first, second,
        provider=cell.provider, conditioning_token=cell.conditioning, cap_megapixels=cap_megapixels,
        array_module=ctx.array_module, profile=ctx.profile, stage_sampler=stage_sampler,
    )
    payload: dict[str, Any] = {
        "base": {
            "flow": base_result["flow"], "geometry": geometry, "timing": base_result["timing"],
            "environment": base_result["environment"], "metrics": base_result["metrics"],
            "conditioning_parameters": cond_meta.get("conditioning_parameters") or {},
            "first_nchw": first_nchw, "second_nchw": second_nchw,
        },
        "input_frames": [
            {"frame": first["frame"], "sha256": first["sha256"]},
            {"frame": second["frame"], "sha256": second["sha256"]},
        ],
    }

    def _aux_infer(frame_a: int, frame_b: int) -> tuple[Any, dict[str, Any], Any]:
        aux_first, aux_second = _load_pair(shot_ctx, frame_a, frame_b, exr_decoder=ctx.exr_decoder)
        result, geom, _, _, second_n = _infer_pair(
            evaluator_instance, artifact, aux_first, aux_second,
            provider=cell.provider, conditioning_token=cell.conditioning, cap_megapixels=cap_megapixels,
            array_module=ctx.array_module, profile="smoke", stage_sampler=None,
        )
        return result["flow"], geom, second_n

    # Chain drift links -- mandatory once the shot declares chain_length; link 0 is the base
    # flow already computed above, so only links 1..chain_length-1 need a fresh pair.
    chain_length = shot_ctx.chain_length
    if chain_length is not None:
        try:
            flows = [payload["base"]["flow"]]
            for link in range(1, chain_length):
                link_flow, _, _ = _aux_infer(reference_frame + link, reference_frame + link + 1)
                flows.append(link_flow)
            payload["chain_flows"] = flows
        except (ExrFailure, EvaluatorFailure, DependencyFailure) as exc:
            payload["chain_error"] = str(exc)

    # Forward/backward residual -- optional, needs the reverse-direction pair.
    try:
        reverse_flow, _, _ = _aux_infer(reference_frame + 1, reference_frame)
        payload["reverse_flow"] = reverse_flow
    except (ExrFailure, EvaluatorFailure, DependencyFailure) as exc:
        log_messages.append(f"forward_backward_residual_px skipped for {cell.as_dict()}: {exc}")

    # Blinded per-candidate review previews -- optional, per offset, only for shots without
    # trustworthy automated truth. Only the flow/image DATA is returned; file writing (pure
    # I/O, no CUDA) happens back in the parent.
    if not shot_ctx.has_analytic_truth and ctx.review_dir is not None:
        review_offsets: dict[str, dict[str, Any]] = {}
        for offset in sorted(set(ctx.chain_offsets) - {1}):
            try:
                offset_flow, offset_geometry, offset_second_nchw = _aux_infer(reference_frame, reference_frame + offset)
                offset_width = offset_geometry["analysis_width"]
                offset_height = offset_geometry["analysis_height"]
                review_offsets[str(offset)] = {
                    "flow": offset_flow,
                    "image2": _unpadded_grid(offset_second_nchw, offset_width, offset_height),
                    "width": offset_width, "height": offset_height,
                }
            except (ExrFailure, EvaluatorFailure, DependencyFailure) as exc:
                log_messages.append(f"review preview offset {offset} skipped for {cell.as_dict()}: {exc}")
        payload["review_offsets"] = review_offsets

    payload["log_messages"] = log_messages
    return payload


def _run_cell(cell: CellKey, ctx: _RunnerContext) -> dict[str, Any]:
    artifact = ctx.artifacts.get(cell.candidate)
    if artifact is None:
        raise _CellFail(_failure("artifact_missing", f"candidate {cell.candidate!r} was not validated at startup"))
    shot_ctx = ctx.shot_contexts.get(cell.shot)
    if shot_ctx is None:
        raise _CellFail(_failure("missing_input", f"shot {cell.shot!r} is absent from the corpus"))
    if cell.provider not in PROVIDER_EXECUTION_NAMES:
        raise _CellFail(_failure("provider_unavailable", f"unknown provider token: {cell.provider!r}"))

    # Cheap, inference-free validation up front, before any host-load prompt or subprocess.
    chain_length = shot_ctx.chain_length
    if chain_length is not None and chain_length not in ctx.chain_offsets and chain_length not in (1, 2, 4, 8):
        raise _CellFail(_failure("input_invalid", f"unsupported chain_length {chain_length}", stage="metrics"))

    # Fix B: a supervised host-load boundary, confirmed once per contiguous run of cells that
    # share one CUDA host_load value, before any cell in that group executes. Re-checked every
    # invocation (ctx.confirmed_host_load always starts None on a fresh process), so a resumed
    # run always reconfirms rather than trusting stale state from before an interruption.
    if cell.provider == "cuda" and ctx.confirmed_host_load != cell.host_load:
        ctx.host_load_checkpoint(cell.host_load)
        ctx.confirmed_host_load = cell.host_load
        ctx.log(f"host load boundary confirmed: host_load={cell.host_load}")

    # Fix G: for a CUDA cell, EVERY inference this cell needs (base + reverse + chain links +
    # review offsets) runs inside _perform_cell_inference, and that whole function is what gets
    # isolated in the measurement child -- the parent below this point does no inference at all
    # for a CUDA cell, so it never itself initializes a CUDA context (the unsafe
    # fork-after-CUDA-init condition this subprocess design exists to avoid). CPU/CoreML cells
    # run the identical function directly, in-process, with no isolation.
    def _inference_work(stage_sampler: Callable[[str], Any] | None) -> dict[str, Any]:
        return _perform_cell_inference(cell, ctx, artifact, shot_ctx, stage_sampler)

    use_cuda_measurement = cell.provider == "cuda" and ctx.nvml_backend_factory is not None
    if use_cuda_measurement:
        try:
            measurement = ctx.cuda_measurement_runner(
                _inference_work, ctx.nvml_backend_factory, ctx.device_index, ctx.poll_interval_s,
            )
        except DriverFailure as exc:
            # run_cuda_measurement_in_subprocess already maps the child's reported failure kind
            # (including an ExrFailure's kind, via _map_child_failure_kind) onto a permitted
            # coordinator failure type before raising, so exc.kind is used as-is here.
            raise _CellFail(_failure(exc.kind, str(exc), stage="inference")) from exc
        except (DependencyFailure, EvaluatorFailure, ExrFailure) as exc:
            # An injected test CudaMeasurementRunner may raise these directly, in-process,
            # rather than going through the subprocess boundary's own error reporting.
            kind = getattr(exc, "kind", None)
            stage = "load_input" if isinstance(exc, ExrFailure) else "inference"
            raise _CellFail(_failure(_map_child_failure_kind(kind), str(exc), stage=stage)) from exc
        payload = measurement.payload
        nvml_samples = list(measurement.samples)
        if measurement.process_exit_used_mib is not None:
            exit_sample: dict[str, Any] = {"stage": "process_exit", "used_mib": measurement.process_exit_used_mib}
            if measurement.process_exit_process_used_mib is not None:
                exit_sample["process_used_mib"] = measurement.process_exit_process_used_mib
            nvml_samples.append(exit_sample)
        resource, nvml_rows = _replay_resource_and_rows(_base_result_fields(cell), nvml_samples, ctx.device_index)
        # Durable, not incremental: overwrites this cell's own prior attempt (if any); the
        # operator-facing nvml.csv itself is assembled once, from every cell's sidecar, only
        # after every cell in the plan is complete (see _regenerate_sidecar_outputs). See Fix E.
        _write_nvml_sidecar(ctx.output_dir, cell, nvml_rows)
    else:
        try:
            payload = _inference_work(None)
        except ExrFailure as exc:
            raise _CellFail(_exr_failure(exc, stage="load_input")) from exc
        except DependencyFailure as exc:
            raise _CellFail(_failure("runtime_error", str(exc), stage="inference", retryable=True)) from exc
        except EvaluatorFailure as exc:
            raise _CellFail(_failure(exc.kind if exc.kind in _RESULT_FAILURE_TYPES else "runtime_error", str(exc), stage="inference")) from exc
        resource = {"peak_incremental_device_memory_gib": 0.0}

    # From here on: PURE scoring and preview rendering from `payload`'s already-returned data.
    # No inference call of any kind happens below this line, for either provider.
    for message in payload.get("log_messages", []):
        ctx.log(message)

    base = payload["base"]
    base_result = {"metrics": base["metrics"], "timing": base["timing"], "environment": base["environment"]}
    predicted_flow = base["flow"]
    geometry = base["geometry"]
    conditioning_parameters = base["conditioning_parameters"]
    first_nchw = base["first_nchw"]
    second_nchw = base["second_nchw"]
    analysis_width = geometry["analysis_width"]
    analysis_height = geometry["analysis_height"]
    spacing_x = geometry["spacing_x_source_pixels"]
    spacing_y = geometry["spacing_y_source_pixels"]
    reference_frame = int(shot_ctx.shot["reference_frame"])

    not_applicable = set(REPORT_METRICS)
    result_metrics: dict[str, Any] = {
        "nonfinite_fraction": base_result["metrics"]["nonfinite_fraction"],
        "repeated_run_p99_delta_px": base_result["metrics"]["repeated_run_p99_delta_px"],
    }

    # Dense metrics: mandatory once analytic truth is declared.
    if shot_ctx.has_analytic_truth and shot_ctx.synthetic_case is not None:
        truth_grid, valid_mask = _dense_truth_and_mask(
            shot_ctx.synthetic_case, reference_frame, reference_frame + 1,
            analysis_width, analysis_height, spacing_x, spacing_y,
        )
        try:
            dense = metrics_module.dense_metrics(predicted_flow, truth_grid, valid_mask)
        except MetricFailure as exc:
            raise _CellFail(_failure("quality_gate_failed", f"dense metrics could not be computed: {exc}", stage="metrics")) from exc
        result_metrics["endpoint_error_px"] = dense["endpoint_error_px"]
        result_metrics["fraction_le_1px"] = dense["fraction_le_1px"]
        result_metrics["fraction_le_3px"] = dense["fraction_le_3px"]
        not_applicable -= {"endpoint_error_px", "fraction_le_1px", "fraction_le_3px"}

    # Chain drift: mandatory once the shot declares a chain_length. The inference itself (and
    # any failure computing it) already happened in _perform_cell_inference; only truth scoring
    # happens here.
    if chain_length is not None:
        if "chain_error" in payload:
            raise _CellFail(_failure(
                "quality_gate_failed", f"chain drift could not be computed: {payload['chain_error']}", stage="metrics",
            ))
        flows = payload.get("chain_flows")
        if not flows:
            raise _CellFail(_failure("quality_gate_failed", "chain drift could not be computed: no chain flows were returned", stage="metrics"))
        if shot_ctx.synthetic_case is not None:
            truth_grid, valid_mask = _dense_truth_and_mask(
                shot_ctx.synthetic_case, reference_frame, reference_frame + chain_length,
                analysis_width, analysis_height, spacing_x, spacing_y,
            )
            try:
                chain_value = metrics_module.chain_drift_px(flows, truth_grid, valid_mask, chain_length)
                result_metrics["chain_drift_px"] = chain_value
                not_applicable.discard("chain_drift_px")
            except MetricFailure as exc:
                raise _CellFail(_failure("quality_gate_failed", f"chain drift could not be computed: {exc}", stage="metrics")) from exc

    # Unpadded analysis-resolution RGB, needed by the optional visible-warp metric and by the
    # local review preview below. Computed unconditionally (a cheap reshape/crop, not expected
    # to fail) so a metric failure below cannot leave it undefined for the preview.
    image1 = _unpadded_grid(first_nchw, analysis_width, analysis_height)
    image2 = _unpadded_grid(second_nchw, analysis_width, analysis_height)

    # Visible warp residual: optional self-consistency metric, no truth required.
    try:
        visible_mask = [[True] * analysis_width for _ in range(analysis_height)]
        visible = metrics_module.visible_warp_residual(image1, image2, predicted_flow, visible_mask)
        result_metrics["visible_warp_residual"] = visible
        not_applicable.discard("visible_warp_residual")
    except MetricFailure as exc:
        ctx.log(f"visible_warp_residual skipped for {cell.as_dict()}: {exc}")

    # Forward/backward residual: optional; the reverse-direction inference already happened (or
    # was skipped, with a log message already emitted above) in _perform_cell_inference.
    reverse_flow = payload.get("reverse_flow")
    if reverse_flow is not None:
        try:
            residual_mask = [[True] * analysis_width for _ in range(analysis_height)]
            residual = metrics_module.forward_backward_residual_px(predicted_flow, reverse_flow, residual_mask)
            result_metrics["forward_backward_residual_px"] = residual
            not_applicable.discard("forward_backward_residual_px")
        except MetricFailure as exc:
            ctx.log(f"forward_backward_residual_px skipped for {cell.as_dict()}: {exc}")

    result_metrics["not_applicable"] = sorted(not_applicable)

    # Fix C: blinded per-candidate previews of THIS candidate's own predicted output -- a
    # target-warped-by-predicted-flow image plus a flow visualization -- across every offset in
    # ctx.chain_offsets that was actually inferrable (Fix G: inference already happened in
    # _perform_cell_inference; this only renders/writes the already-returned flow/image data).
    if not shot_ctx.has_analytic_truth and ctx.review_dir is not None:
        _write_offset_preview(ctx.review_dir, cell, 1, image2, predicted_flow, analysis_width, analysis_height)
        for offset_key, offset_data in (payload.get("review_offsets") or {}).items():
            _write_offset_preview(
                ctx.review_dir, cell, int(offset_key), offset_data["image2"], offset_data["flow"],
                offset_data["width"], offset_data["height"],
            )

    category = (shot_ctx.shot.get("categories") or [None])[0]
    passing: dict[str, Any] = {
        "input_frames": payload["input_frames"],
        "geometry": geometry,
        "timing": base_result["timing"],
        "metrics": result_metrics,
        "resource": resource,
        "environment": base_result["environment"],
        "conditioning_parameters": conditioning_parameters,
    }
    if category is not None:
        passing["category"] = category
    return passing


_RESULT_FAILURE_TYPES = {
    "artifact_missing", "artifact_hash_mismatch", "license_not_permitted", "license_unknown",
    "provider_unavailable", "unsupported_tensor_contract", "wrong_direction", "export_not_reproducible",
    "missing_input", "input_invalid", "conditioning_failure", "cap_unavailable", "out_of_memory",
    "runtime_error", "nonfinite_output", "repeated_run_instability", "quality_gate_failed",
    "operator_cancelled", "not_attempted", "other",
}


def build_executor(
    *,
    protocol: Mapping[str, Any],
    corpus: Mapping[str, Any],
    profile: str,
    artifacts: Mapping[str, ValidatedArtifact],
    runtime_module: Any,
    array_module: Any,
    nvml_backend_factory: Callable[[], NvmlBackend] | None,
    device_index: int,
    poll_interval_s: float,
    chain_offsets: Sequence[int],
    exr_decoder: Callable[..., dict[str, Any]],
    output_dir: Path,
    review_dir: Path | None,
    log: Callable[[str], None],
    host_load_checkpoint: HostLoadCheckpoint,
    cuda_measurement_runner: CudaMeasurementRunner,
) -> tuple[Callable[[CellKey], dict[str, Any]], _RunnerContext]:
    """Build the executor callback plus the mutable context RunCoordinator will drive."""

    synthetic_cases = synthetic_module.case_map()
    shot_contexts: dict[str, _ShotContext] = {}
    for shot_id, (partition_kind, shot) in _shot_index(corpus).items():
        shot_contexts[shot_id] = _build_shot_context(shot_id, partition_kind, shot, synthetic_cases)

    ctx = _RunnerContext(
        protocol=protocol,
        profile=profile,
        shot_contexts=shot_contexts,
        artifacts=artifacts,
        runtime_module=runtime_module,
        array_module=array_module,
        nvml_backend_factory=nvml_backend_factory,
        device_index=device_index,
        poll_interval_s=poll_interval_s,
        chain_offsets=tuple(chain_offsets),
        exr_decoder=exr_decoder,
        output_dir=output_dir,
        review_dir=review_dir,
        log=log,
        host_load_checkpoint=host_load_checkpoint,
        cuda_measurement_runner=cuda_measurement_runner,
    )

    def executor(cell: CellKey) -> dict[str, Any]:
        base = _base_result_fields(cell)
        log(f"cell start {json.dumps(cell.as_dict(), sort_keys=True)}")
        try:
            passing = _run_cell(cell, ctx)
        except _CellFail as failure:
            log(f"cell fail {json.dumps(cell.as_dict(), sort_keys=True)} reason={failure.failure['type']}")
            return {**base, "status": "fail", "failure": failure.failure}
        except _CellSkip as skip:
            log(f"cell skip {json.dumps(cell.as_dict(), sort_keys=True)} reason={skip.failure['type']}")
            return {**base, "status": "skip", "failure": skip.failure}
        log(f"cell pass {json.dumps(cell.as_dict(), sort_keys=True)}")
        return {**base, "status": "pass", **passing}

    return executor, ctx


# --------------------------------------------------------------------------------------------
# Startup: candidate artifact validation and resume identity.
# --------------------------------------------------------------------------------------------


def _validate_selected_artifacts(
    plan: MatrixPlan,
    artifact_map: Mapping[str, Mapping[str, Any]],
    protocol_path: Path,
) -> dict[str, ValidatedArtifact]:
    """Validate every selected candidate's manifest/artifact once, up front.

    This is a hard startup check: the resume identity binds each selected candidate's exact
    manifest/artifact hashes, so a candidate that cannot be validated here cannot produce a
    stable identity at all. Per-cell failures are for conditions discovered while running a
    cell, not for an artifact that was never resolvable in the first place.
    """

    resolved: dict[str, ValidatedArtifact] = {}
    for candidate_id in plan.selector["candidate_ids"]:
        entry = artifact_map.get(candidate_id)
        if entry is None:
            _fail("artifact_map_missing", f"--artifact-map has no entry for selected candidate {candidate_id!r}")
        manifest_path = entry.get("manifest")
        if not isinstance(manifest_path, str) or not manifest_path:
            _fail("artifact_map_shape", f"--artifact-map entry for {candidate_id!r} needs a non-empty manifest path")
        artifact_path = entry.get("artifact")
        platform = entry.get("platform")
        try:
            validated = validate_manifest_artifact(
                Path(manifest_path),
                Path(artifact_path) if isinstance(artifact_path, str) else None,
                platform=platform if isinstance(platform, str) else None,
                protocol_path=protocol_path,
            )
        except EvaluatorFailure as exc:
            _fail("candidate_artifact_invalid", f"candidate {candidate_id!r} failed artifact validation: {exc}")
        if validated.candidate_id != candidate_id:
            _fail(
                "candidate_artifact_invalid",
                f"manifest for {candidate_id!r} declares candidate id {validated.candidate_id!r}",
            )
        resolved[candidate_id] = validated
    return resolved


def _require_runner_fields(runner_section: Mapping[str, Any]) -> None:
    for required in ("evaluator_sha256", "runtime", "runtime_sha256"):
        if not runner_section.get(required):
            _fail("report_metadata_missing", f"report_metadata.runner.{required} is required")


def _compute_identity(
    protocol: Mapping[str, Any],
    corpus: Mapping[str, Any],
    plan: MatrixPlan,
    environment: str,
    profile: str,
    artifacts: Mapping[str, ValidatedArtifact],
    runner_metadata: Mapping[str, Any],
    hardware: Mapping[str, Any],
    chain_offsets: Sequence[int],
    *,
    device_index: int,
    poll_interval_s: float,
    nvml_enabled: bool,
) -> dict[str, Any]:
    """Bind every input that can change a cell's produced result or reported resource use.

    ``profile`` in particular is NOT part of ``plan.matrix_sha256`` -- the matrix selector
    (candidate/shot/conditioning/cap/provider/host_load axes) is identical between e.g. smoke
    and screen for the same selection, so without ``profile`` here two different profiles run
    against the same ``--state``/``--output-dir`` would collide on one resume identity and the
    second run would silently resume/reuse the first's (wrong) results. ``evaluator_sha256``
    similarly is not otherwise bound: a different evaluator build measuring the same matrix is a
    different run. ``chain_offsets`` changes which auxiliary frames/metrics a cell attempts.

    Fix I: ``device_index``/``poll_interval_s``/``nvml_enabled`` and the GPU/driver hardware
    fields are ALSO result-affecting for a CUDA matrix -- without them, a run measured with
    ``--no-nvml`` (which reports a flat zero ``resource`` for every CUDA cell) and a rerun of
    the identical selection with NVML enabled would collide on one identity, and the second
    invocation would silently resume/reuse the first's placeholder (zero) resource evidence
    instead of measuring anything.
    """

    return {
        "protocol_id": protocol["protocol_id"],
        "matrix_sha256": plan.matrix_sha256,
        "corpus_sha256": canonical_sha256(corpus),
        "environment": environment,
        "profile": profile,
        "providers": plan.selector["providers"],
        "candidates": {
            candidate_id: {
                "manifest_sha256": artifacts[candidate_id].manifest_sha256,
                "artifact_sha256": artifacts[candidate_id].artifact_sha256,
                "platform": artifacts[candidate_id].platform,
            }
            for candidate_id in plan.selector["candidate_ids"]
        },
        "evaluator_sha256": runner_metadata.get("evaluator_sha256", ""),
        "runtime": {
            "runtime": runner_metadata.get("runtime", ""),
            "runtime_sha256": runner_metadata.get("runtime_sha256", ""),
        },
        "hardware": {
            "platform": hardware.get("platform", ""),
            "architecture": hardware.get("architecture", ""),
            "gpu": hardware.get("gpu", ""),
            "driver": hardware.get("driver", ""),
        },
        "chain_offsets": sorted(int(offset) for offset in chain_offsets),
        "measurement": {
            "device_index": device_index,
            "poll_interval_s": poll_interval_s,
            "nvml_enabled": nvml_enabled,
        },
    }


def _report_matches_current_run(report: Mapping[str, Any], identity: Mapping[str, Any]) -> bool:
    """Check an already-published report.json against the CURRENT run identity.

    report.json is schema-constrained (report-v2 forbids unknown properties), so the identity
    dict itself is never stored there; instead this reconstructs the identity-relevant fields
    from the report's own already-published content and compares them. Used to refuse silently
    reusing a stale report -- e.g. one produced by an earlier smoke run -- when the output
    directory is reused for a run with a different identity but report.json happens to already
    exist there (see also: resume.load_state's own identity check, which already covers the
    narrower case of the SAME --state path reused with a different identity).
    """

    try:
        if report.get("protocol_id") != identity["protocol_id"]:
            return False
        if report.get("corpus_sha256") != identity["corpus_sha256"]:
            return False
        if report.get("profile") != identity["profile"]:
            return False
        if report.get("environment") != identity["environment"]:
            return False
        matrix = report.get("matrix")
        if not isinstance(matrix, Mapping) or matrix.get("matrix_sha256") != identity["matrix_sha256"]:
            return False
        providers = matrix.get("providers")
        if providers != identity["providers"]:
            return False
        runner = report.get("runner")
        if not isinstance(runner, Mapping):
            return False
        if runner.get("evaluator_sha256") != identity["evaluator_sha256"]:
            return False
        if runner.get("runtime_sha256") != identity["runtime"]["runtime_sha256"]:
            return False
        report_hardware = report.get("hardware")
        if not isinstance(report_hardware, Mapping):
            return False
        if report_hardware.get("gpu", "") != identity["hardware"]["gpu"]:
            return False
        if report_hardware.get("driver", "") != identity["hardware"]["driver"]:
            return False
        # device_index/poll_interval_s/nvml_enabled (Fix I) are not part of the report-v2
        # schema at all, so they cannot be cross-checked from report.json's own content here;
        # resume.load_state's identity_sha256 check (the common case) is what enforces them.
        # Candidate manifest_sha256/artifact_sha256 are deliberately NOT compared here.
        # report["candidates"] carries whatever the *operator's* --candidates input declared
        # (legal/measurement-admission evidence); identity["candidates"] carries the hashes
        # validate_manifest_artifact actually computed from the artifact bytes on disk. Nothing
        # requires those two to be textually identical (an operator's declared manifest_sha256
        # need not equal the exact re-hash of the file --artifact-map points at), so comparing
        # them here would reject a perfectly legitimate identical rerun on a false mismatch. The
        # candidate artifact hashes ARE still fully bound into the resume identity above them --
        # resume.load_state's own identity_sha256 check (the common case, since --state
        # defaults to living beside report.json) is what actually enforces that axis; this
        # report.json check is a narrower safety net for a --state path that was not reused.
        candidates_by_id = {
            entry.get("candidate_id"): entry
            for entry in report.get("candidates", [])
            if isinstance(entry, Mapping)
        }
        for candidate_id in identity["candidates"]:
            if not isinstance(candidates_by_id.get(candidate_id), Mapping):
                return False
        return True
    except (KeyError, TypeError):
        return False


# --------------------------------------------------------------------------------------------
# Output files.
# --------------------------------------------------------------------------------------------


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _atomic_publish(path: Path, payload: bytes, *, replace_existing: bool) -> None:
    """Publish ``payload`` to ``path`` so a partial write can never survive at the final path.

    Fix K: mirrors ``reporting.py``'s ``_stage``/``write_report_pair`` discipline exactly --
    staged in a same-directory temp file, ``fchmod`` to the output mode, then
    ``flush``+``fsync`` of the file itself before it is ever linked to the final name, followed
    by an ``fsync`` of the parent directory so the publish (not just the bytes) is durable. With
    ``replace_existing`` false, installation is a no-clobber hard link (refuses a destination
    that appeared concurrently); with it true, ``os.replace`` is used. Either way, the
    destination only ever shows the complete staged bytes or its own prior content -- never a
    truncated write from an interrupted attempt.
    """

    if not path.parent.is_dir():
        _fail("output_path", f"output parent is not a directory: {path.parent}")
    descriptor = -1
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        temporary = Path(name)
        os.fchmod(descriptor, _OUTPUT_FILE_MODE)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1  # ownership transferred to the file object; only its close() now
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if replace_existing:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path, follow_symlinks=False)
            except FileExistsError as exc:
                raise DriverFailure("output_exists", f"output appeared during publication: {path}") from exc
            temporary.unlink()
        temporary = None
        directory_descriptor = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as exc:
        raise DriverFailure("atomic_write", str(exc)) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _run_identity_path(output_dir: Path) -> Path:
    return output_dir / ".run-identity.json"


def _write_run_identity(output_dir: Path, identity: Mapping[str, Any]) -> None:
    """Durably persist the FULL run identity (Fix J).

    report.json is schema-constrained (report-v2 forbids unknown properties) and several
    identity fields -- device_index, poll_interval_s, nvml_enabled -- have no home in that
    schema at all, so they can never be cross-checked from the report's own content. This
    sidecar is the authoritative record of exactly which identity is (or is about to be)
    producing the report/nvml.csv/review.csv in this output_dir, checked by hash (never by
    merely existing) before any of them is ever reused. Not one of the five operator return
    files; internal bookkeeping only, like ``.sidecars/``. Always overwritten (this file's only
    job is to describe whatever is currently -- or about to be -- on disk, so there is nothing to
    preserve from an older identity).

    Fix M: called from `_establish_run_identity_up_front`, BEFORE report.json/summary.txt/
    nvml.csv/review.csv are (re)published for this identity, not after -- so this record can
    never trail a published report the way it could when it was written last, and a crash between
    publishing report.json and this write is no longer possible (the write already happened).
    """

    payload = (
        json.dumps(
            {"identity": dict(identity), "identity_sha256": canonical_sha256(identity)},
            ensure_ascii=False, sort_keys=True, indent=2,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_publish(_run_identity_path(output_dir), payload, replace_existing=True)


def _read_run_identity_sha256(output_dir: Path) -> str | None:
    """Return the persisted identity hash (Fix J), or ``None`` if absent/unreadable."""

    path = _run_identity_path(output_dir)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return None
    value = data.get("identity_sha256") if isinstance(data, Mapping) else None
    return value if isinstance(value, str) else None


def _establish_run_identity_up_front(output_dir: Path, identity: Mapping[str, Any], *, replace: bool) -> None:
    """Validate/establish ``.run-identity.json`` before a single cell executes (Fix L), and
    guarantee the identity record can never trail a published report.json (Fix M).

    Called immediately once ``identity`` is computed -- before resume state is loaded or
    created, before the executor is built, and before ``RunCoordinator.run()`` can touch a
    single per-cell sidecar or review preview. This matters because those are keyed only by
    CellKey, not by run identity: without this up-front check, a rejected run (caught by the old
    end-of-run-only Fix J guard) had already overwritten the ORIGINAL run's sidecars/previews by
    the time it was caught, corrupting evidence that a resumed original run would then silently
    fold into a report claiming it came from the original measurement.

    Four cases, in order:

    * ``--replace``: an explicit fresh start. Overwrite the identity record unconditionally --
      whatever (if anything) previously produced this output_dir is being replaced regardless,
      and Fix N's per-file no-clobber-unless-replace logic will do the same for the other outputs.
    * report.json exists and its persisted identity hash already matches this invocation's: nothing
      to do -- a legitimate no-op re-run or repair pass.
    * report.json exists but the persisted identity is ABSENT: this is the Fix M recovery case --
      a run that crashed (or hit a write failure) between publishing report.json and writing its
      identity sidecar (that write order held even pre-fix, since ``_write_run_identity`` was
      always the LAST write) must not be permanently stuck demanding ``--replace``. Recoverable
      IFF the already-published report's own content is consistent with this invocation's identity
      (``_report_matches_current_run``, the same report-derived cross-check kept below as a
      secondary safety net); if so, the identity is established now, before anything else runs.
      Otherwise this is a genuinely different, unrelated report reusing the directory -- refused,
      same as a hash mismatch.
    * report.json exists and the persisted identity hash MISMATCHES: a different run (different
      profile, matrix selection, candidate artifacts, or measurement configuration such as
      --device-index/--poll-interval-s/--no-nvml) reusing this --output-dir. Refused immediately,
      with zero side effects -- no resume state created or loaded, no executor built, no cell
      executed, no sidecar or review preview touched.
    * report.json does not exist yet (a fresh run, or an interrupted run whose report was never
      published): if no identity is persisted yet, establish one now -- covering both a genuinely
      fresh output_dir and the first invocation of a run this fix hasn't seen complete yet. If one
      IS already persisted, it must match: a fresh ``--state``/``--output-dir`` combination that
      collides with an in-progress run under a DIFFERENT identity is refused here too, before any
      cell executes -- ``resume.load_state``'s own identity check only guards the specific
      ``--state`` path reused, not a fresh one pointed at a reused ``--output-dir`` (this is
      exactly Fix L's reproduced scenario).
    """

    if replace:
        _write_run_identity(output_dir, identity)
        return

    current_sha256 = canonical_sha256(identity)
    persisted_sha256 = _read_run_identity_sha256(output_dir)
    json_path = output_dir / "report.json"

    if json_path.exists():
        if persisted_sha256 == current_sha256:
            return
        if persisted_sha256 is None:
            try:
                published_report = load_json(json_path)
            except (OSError, ValueError) as exc:
                raise DriverFailure(
                    "report_identity_mismatch",
                    f"{json_path} exists but could not be read to recover its missing "
                    f"{_run_identity_path(output_dir)}: {exc}; rerun with --replace to overwrite, "
                    "or use a different --output-dir",
                ) from exc
            if _report_matches_current_run(published_report, identity):
                _write_run_identity(output_dir, identity)
                return
        raise DriverFailure(
            "report_identity_mismatch",
            f"{json_path} already exists but its persisted run identity "
            f"({_run_identity_path(output_dir)}) does not match this invocation's identity "
            "(different profile, evaluator, matrix selection, candidate artifacts, or "
            "measurement configuration such as --device-index/--poll-interval-s/--no-nvml); "
            "rerun with --replace to overwrite, or use a different --output-dir",
        )

    if persisted_sha256 is None:
        _write_run_identity(output_dir, identity)
        return
    if persisted_sha256 != current_sha256:
        raise DriverFailure(
            "report_identity_mismatch",
            f"{output_dir} already has a {_run_identity_path(output_dir)} from a different run "
            "identity (an interrupted run under a different configuration) but no report.json "
            "yet; rerun with --replace to overwrite, or use a different --output-dir/--state",
        )


def _open_append_0644(path: Path) -> Any:
    exists = path.exists()
    if exists and path.is_symlink():
        _fail("output_path", f"{path} must not be a symlink")
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, _OUTPUT_FILE_MODE)
    try:
        os.fchmod(descriptor, _OUTPUT_FILE_MODE)
    except OSError:
        pass
    return os.fdopen(descriptor, "a", encoding="utf-8")


class RunnerLog:
    """Append-only, timestamped ``runner.log``. Survives resume by construction (append mode)."""

    def __init__(self, path: Path):
        self.path = path
        self._stream = _open_append_0644(path)

    def write(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        self._stream.write(f"{timestamp} {message}\n")
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()


def _render_csv_bytes(header: Sequence[str], rows: Sequence[Sequence[str]]) -> bytes:
    from io import StringIO
    buffer = StringIO(newline="")
    writer = csv.writer(buffer, delimiter=",", quotechar='"', lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


def _parse_csv_data_rows(payload: bytes) -> list[list[str]] | None:
    """Parse CSV bytes into data rows (the header row, if any, is dropped).

    Returns ``None`` -- never a guess -- when ``payload`` cannot be safely interpreted as CSV
    text (not UTF-8, or the csv module itself rejects it), so a caller can fall back to treating
    the file as having no recoverable rows rather than silently matching on garbage.
    """

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return None
    from io import StringIO
    try:
        rows = list(csv.reader(StringIO(text, newline=""), delimiter=",", quotechar='"'))
    except csv.Error:
        return None
    return rows[1:] if rows else []


def _write_csv_file(
    path: Path,
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    replace: bool,
    verify_and_repair: bool = False,
    merge_existing_rows: Callable[[Sequence[Sequence[str]], Sequence[Sequence[str]]], list[list[str]]] | None = None,
) -> None:
    """Publish one driver-owned CSV (nvml.csv/review.csv) fresh, exactly once, atomically.

    Fix E: both files are a deterministic function of durable completed state (nvml.csv from
    each cell's sidecar, review.csv from the completed results plus corpus metadata), assembled
    only after every plan cell is complete -- never incrementally appended to during execution,
    which could otherwise duplicate a cell's rows if it were retried after a resume. Writes
    nothing (and creates no file) when ``rows`` is empty, matching the existing contract that
    these files exist only when there is something to report (e.g. no nvml.csv for a CPU-only
    run). No-clobber unless ``replace``, mirroring report.json/report.csv. Publication itself is
    atomic (see :func:`_atomic_publish`): an interrupted write can never leave a truncated file
    at the final path.

    ``verify_and_repair`` (Fix K, replacing the earlier ``only_if_missing``): used when
    repairing a resumed run's output directory (report.json already published under the current
    identity). Mere existence is never trusted -- the canonical bytes are regenerated and
    compared against whatever is already at ``path`` byte-for-byte; a missing, truncated, or
    otherwise different file is atomically replaced with the canonical bytes, while an
    already-correct file is left untouched (not even reopened).

    ``merge_existing_rows`` (Fix N): nvml.csv has no human-edited columns, so exact-byte
    replacement is always correct for it. review.csv is different -- the plan's human review
    pass fills edge_adherence/occlusion_reveal/blur/jitter/drift/notes directly into the
    published file, and a byte-for-byte repair on the next resumed invocation would silently
    blank every one of those cells back to canonical (empty) values. When given, this callback
    receives ``(canonical_rows, existing_data_rows)`` and returns the rows to actually publish;
    it is consulted only after the fast byte-identical check fails and only when the existing
    file parses as CSV at all (see ``_parse_csv_data_rows``) -- an unparseable file still falls
    back to the unconditional canonical replacement below, same as before this parameter existed.
    """

    if not rows:
        return
    payload = _render_csv_bytes(header, rows)
    if verify_and_repair:
        existing_bytes: bytes | None = None
        if path.exists():
            try:
                existing_bytes = path.read_bytes()
            except OSError:
                existing_bytes = None  # unreadable for some other reason -- replace it below
        if existing_bytes is not None:
            if existing_bytes == payload:
                return
            if merge_existing_rows is not None:
                existing_rows = _parse_csv_data_rows(existing_bytes)
                if existing_rows is not None:
                    merged_payload = _render_csv_bytes(header, merge_existing_rows(rows, existing_rows))
                    if merged_payload == existing_bytes:
                        return
                    _atomic_publish(path, merged_payload, replace_existing=True)
                    return
        _atomic_publish(path, payload, replace_existing=True)
        return
    if path.exists() and not replace:
        _fail("output_exists", f"{path} already exists; rerun with --replace to overwrite")
    _atomic_publish(path, payload, replace_existing=replace)


def _shot_has_analytic_truth(shot: Mapping[str, Any]) -> bool:
    truth = shot.get("truth")
    return isinstance(truth, Mapping) and truth.get("kind") == "analytic"


def _review_identity_key(row: Sequence[str]) -> tuple[str, ...]:
    return tuple(row[index] for index in _REVIEW_IDENTITY_COLUMNS)


def _merge_review_rows(
    canonical_rows: Sequence[Sequence[str]], existing_rows: Sequence[Sequence[str]],
) -> list[list[str]]:
    """Repair review.csv while preserving the human-edited columns an operator already filled in.

    Fix N: review.csv's contract (unlike nvml.csv) is that a human fills edge_adherence/
    occlusion_reveal/blur/jitter/drift/notes into the published file after the run. A rerun of
    the same completed configuration must repair the eight driver-owned columns and the expected
    row set -- without ever blanking or overwriting a human column that already has content.

    Rows are matched by their CellKey-equivalent identity (``_REVIEW_IDENTITY_COLUMNS`` --
    candidate_label/shot_id/conditioning_token/cap_token/provider/host_load); a match keeps that
    row's six human columns and takes every driver-owned column, including category/preview_path,
    fresh from ``canonical_rows`` (so a corrupted driver-owned column is still repaired). A
    canonical row with no match is new and published with blank human columns, same as a fresh
    row always has been. An existing row with no canonical counterpart belongs to a cell no
    longer in the current plan and is dropped, matching what an exact-byte repair would do.
    Malformed existing rows (wrong column count) cannot be safely attributed to any cell and are
    dropped rather than risk merging human text into the wrong row.
    """

    driver_owned = len(REVIEW_CSV_HEADER) - _REVIEW_HUMAN_COLUMN_COUNT
    existing_by_key: dict[tuple[str, ...], list[str]] = {
        _review_identity_key(row): list(row)
        for row in existing_rows
        if len(row) == len(REVIEW_CSV_HEADER)
    }
    merged: list[list[str]] = []
    for row in canonical_rows:
        prior = existing_by_key.get(_review_identity_key(row))
        if prior is None:
            merged.append(list(row))
        else:
            merged.append(list(row[:driver_owned]) + prior[driver_owned:])
    return merged


def _regenerate_sidecar_outputs(
    output_dir: Path,
    review_dir: Path | None,
    corpus: Mapping[str, Any],
    plan: MatrixPlan,
    completed: Sequence[Mapping[str, Any]],
    *,
    replace: bool,
    verify_and_repair: bool = False,
) -> None:
    """Assemble nvml.csv and review.csv once, from durable completed state (Fix E).

    Called only after ``coordinator.completed_records()`` has confirmed every plan cell is
    complete. Iterates ``plan.cells`` in its fixed order (matching ``completed``'s order, per
    ``RunCoordinator``), so re-running this after a resume always produces the identical file:
    exactly one row-set per cell, regardless of how many partial attempts that cell needed.

    Fix F/K: also called (with ``verify_and_repair=True``) when report.json already existed
    under the current identity and every cell was already complete -- a crash or a manual
    deletion/truncation that left report.json present but nvml.csv/review.csv missing or
    corrupted must still be repaired on the next resumed invocation, since the per-cell sidecars
    this reads from remain durable on disk regardless of whether report.json itself needed
    regenerating. ``verify_and_repair`` never trusts mere existence -- see ``_write_csv_file``.
    """

    shots = _shot_index(corpus)
    nvml_rows: list[list[str]] = []
    review_rows: list[list[str]] = []
    for cell, record in zip(plan.cells, completed):
        nvml_rows.extend(_read_nvml_sidecar(output_dir, cell))
        result = record["result"]
        if result.get("status") != "pass" or review_dir is None:
            continue
        shot_entry = shots.get(cell.shot)
        if shot_entry is None or _shot_has_analytic_truth(shot_entry[1]):
            continue
        category = (shot_entry[1].get("categories") or [""])[0]
        review_rows.append([
            _review_label(cell.shot, cell.candidate), cell.shot, category,
            cell.conditioning, cell.cap, cell.provider, cell.host_load,
            str(_review_preview_dir(review_dir, cell)), "", "", "", "", "", "",
        ])
    _write_csv_file(output_dir / "nvml.csv", NVML_CSV_HEADER, nvml_rows, replace=replace, verify_and_repair=verify_and_repair)
    _write_csv_file(
        output_dir / "review.csv", REVIEW_CSV_HEADER, review_rows, replace=replace,
        verify_and_repair=verify_and_repair, merge_existing_rows=_merge_review_rows,
    )


def _write_summary_txt(path: Path, report: Mapping[str, Any], plan: MatrixPlan) -> None:
    lines: list[str] = []
    lines.append(f"White Water P25-6 bake-off summary")
    lines.append(f"profile: {report.get('profile')}")
    lines.append(f"environment: {report.get('environment')}")
    lines.append(f"matrix_sha256: {plan.matrix_sha256}")
    summary = report.get("summary", {})
    lines.append(
        f"cells: required={summary.get('required_cells')} passed={summary.get('passed_cells')} "
        f"failed={summary.get('failed_cells')} skipped={summary.get('skipped_cells')}"
    )
    for score_key in ("synthetic_macro_score", "production_macro_score", "final_quality_score"):
        if score_key in summary:
            lines.append(f"{score_key}: {summary[score_key]}")
    category_scores = summary.get("category_scores")
    if isinstance(category_scores, Mapping) and category_scores:
        lines.append("category_scores:")
        for category, score in sorted(category_scores.items()):
            lines.append(f"  {category}: {score}")

    lines.append("")
    lines.append("Per-candidate headline latency (steady_inference_ms) and memory (peak_device_memory_mib):")
    by_candidate: dict[str, list[Mapping[str, Any]]] = {}
    for result in report.get("results", []):
        by_candidate.setdefault(result["candidate_id"], []).append(result)
    for candidate_id in sorted(by_candidate):
        passing = [r for r in by_candidate[candidate_id] if r.get("status") == "pass"]
        if not passing:
            lines.append(f"  {candidate_id}: no passing cells")
            continue
        steady = [r["timing"]["steady_inference_ms"] for r in passing if "timing" in r]
        peak_mib = [r["resource"]["peak_device_memory_mib"] for r in passing if "resource" in r and "peak_device_memory_mib" in r["resource"]]
        first_ms = f"{sorted(steady)[len(steady) // 2]:.3f}ms" if steady else "n/a"
        peak = f"{max(peak_mib):.1f}MiB" if peak_mib else "n/a (CPU or unmeasured)"
        lines.append(f"  {candidate_id}: median steady={first_ms} peak_device_memory={peak}")

    lines.append("")
    lines.append("Typed failures and skips:")
    any_failure = False
    for result in report.get("results", []):
        if result.get("status") == "pass":
            continue
        any_failure = True
        failure = result.get("failure", {})
        lines.append(
            f"  [{result.get('status')}] {result.get('candidate_id')}/{result.get('shot_id')}/"
            f"{result.get('conditioning_token')}/{result.get('cap_token')}/{result.get('provider')}/"
            f"{result.get('host_load')}: {failure.get('type')}: {failure.get('message')}"
        )
    if not any_failure:
        lines.append("  (none)")

    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    os.chmod(path, _OUTPUT_FILE_MODE)


# --------------------------------------------------------------------------------------------
# Orchestration.
# --------------------------------------------------------------------------------------------


@dataclass
class RunConfig:
    protocol: Mapping[str, Any]
    corpus: Mapping[str, Any]
    candidate_entries: Any
    selection: Mapping[str, Any]
    artifact_map: Mapping[str, Mapping[str, Any]]
    report_schema: Mapping[str, Any]
    corpus_schema: Mapping[str, Any]
    output_dir: Path
    state_path: Path
    device_index: int
    poll_interval_s: float
    chain_offsets: Sequence[int]
    report_metadata: Mapping[str, Any]
    protocol_path: Path
    replace: bool
    runtime_module: Any
    array_module: Any
    nvml_backend_factory: Callable[[], NvmlBackend] | None
    exr_decoder: Callable[..., dict[str, Any]]
    host_load_checkpoint: HostLoadCheckpoint = interactive_host_load_checkpoint
    cuda_measurement_runner: CudaMeasurementRunner = run_cuda_measurement_in_subprocess


@dataclass
class RunResult:
    report: dict[str, Any] | None
    output_paths: dict[str, Path]
    incomplete: bool


def run_bakeoff(config: RunConfig) -> RunResult:
    """Run (or resume) one profile and publish the five operator return files."""

    _ensure_output_dir(config.output_dir)
    runner_log = RunnerLog(config.output_dir / "runner.log")
    try:
        return _run_bakeoff(config, runner_log)
    finally:
        runner_log.close()


def _selection_axes(selection: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: selection[key]
        for key in ("candidate_ids", "shot_ids", "conditioning_tokens", "cap_tokens", "providers")
    }


def _run_bakeoff(config: RunConfig, runner_log: RunnerLog) -> RunResult:
    runner_log.write("run invoked")
    protocol = config.protocol
    corpus = config.corpus
    selection = config.selection
    profile = selection["profile"]
    environment = selection["environment"]

    try:
        plan = build_matrix(
            protocol, corpus, config.candidate_entries, _selection_axes(selection), profile, environment,
        )
    except MatrixFailure as exc:
        runner_log.write(f"matrix planning failed: {exc}")
        raise

    runner_log.write(f"matrix planned: {len(plan.cells)} cells, matrix_sha256={plan.matrix_sha256}")

    artifacts = _validate_selected_artifacts(
        plan, config.artifact_map, config.protocol_path,
    )
    runner_log.write(f"validated artifacts for candidates: {sorted(artifacts)}")

    hardware = _default_hardware()
    hardware.update({k: v for k, v in config.report_metadata.get("hardware", {}).items() if v})
    runner_section = dict(config.report_metadata.get("runner", {}))
    _require_runner_fields(runner_section)

    identity = _compute_identity(
        protocol, corpus, plan, environment, profile, artifacts, runner_section, hardware, config.chain_offsets,
        device_index=config.device_index, poll_interval_s=config.poll_interval_s,
        nvml_enabled=config.nvml_backend_factory is not None,
    )

    # Fix L/M: validate/establish .run-identity.json now -- before resume state is loaded or
    # created, before the executor is built, and before a single cell can execute and overwrite
    # a CellKey-keyed sidecar or review preview that might belong to a different, already-
    # published run reusing this --output-dir. See `_establish_run_identity_up_front`.
    _establish_run_identity_up_front(config.output_dir, identity, replace=config.replace)

    if config.state_path.exists():
        state = load_state(config.state_path, identity, plan)
        runner_log.write("resumed existing state")
    else:
        state = create_state(config.state_path, identity, plan)
        runner_log.write("created new state")

    review_dir = config.output_dir / "review-previews"
    executor, exec_ctx = build_executor(
        protocol=protocol,
        corpus=corpus,
        profile=profile,
        artifacts=artifacts,
        runtime_module=config.runtime_module,
        array_module=config.array_module,
        nvml_backend_factory=config.nvml_backend_factory,
        device_index=config.device_index,
        poll_interval_s=config.poll_interval_s,
        chain_offsets=config.chain_offsets,
        exr_decoder=config.exr_decoder,
        output_dir=config.output_dir,
        review_dir=review_dir,
        log=runner_log.write,
        host_load_checkpoint=config.host_load_checkpoint,
        cuda_measurement_runner=config.cuda_measurement_runner,
    )

    coordinator = RunCoordinator(config.state_path, identity, plan, executor)
    coordinator.run()

    try:
        completed = coordinator.completed_records()
    except IncompleteFailure as exc:
        runner_log.write(f"run interrupted, not all cells complete: {exc}")
        return RunResult(report=None, output_paths={}, incomplete=True)

    json_path = config.output_dir / "report.json"
    csv_path = config.output_dir / "report.csv"
    summary_path = config.output_dir / "summary.txt"

    if json_path.exists() and not config.replace:
        # Every cell was already complete, and an earlier invocation already published the
        # report. Re-running assemble_report here would mint a fresh completed_utc/report_id
        # and then collide with the no-clobber write below; instead this resumed invocation is
        # a true no-op that returns exactly what was already published -- but ONLY if it was
        # published under this SAME run identity (Fix A). resume.load_state above already
        # refuses a state-file identity mismatch for the same --state path.
        #
        # Fix L/M: the AUTHORITATIVE identity check -- the persisted full-identity hash in
        # .run-identity.json, catching a stale report.json left in a reused --output-dir under a
        # fresh/different --state path, which load_state's check does not see -- already ran in
        # `_establish_run_identity_up_front`, before a single cell executed. Reaching this point
        # guarantees report.json (if present) is attributed to this exact run identity. The
        # report-derived _report_matches_current_run check below is kept only as a second,
        # non-authoritative safety net (e.g. report.json edited by hand beneath an untouched
        # identity sidecar); it should be unreachable in ordinary operation.
        report = load_json(json_path)
        if not _report_matches_current_run(report, identity):
            raise DriverFailure(
                "report_identity_mismatch",
                f"{json_path} already exists but was produced under a different run identity "
                "(different profile, evaluator, matrix selection, or candidate artifacts); "
                "rerun with --replace to overwrite, or use a different --output-dir",
            )
        runner_log.write("report.json already published by an earlier invocation under the same identity; not regenerating")
        if not summary_path.exists():
            _write_summary_txt(summary_path, report, plan)
        # Fix F/K: repair nvml.csv/review.csv if a crash or manual deletion/truncation left them
        # missing or incorrect -- the per-cell sidecars this reads from are durable regardless
        # of whether report.json itself needed regenerating. Byte-verified, not existence-only
        # (nvml.csv); review.csv additionally preserves human-edited columns across a repair
        # (Fix N -- see `_merge_review_rows`).
        _regenerate_sidecar_outputs(
            config.output_dir, review_dir, corpus, plan, completed, replace=config.replace, verify_and_repair=True,
        )
    else:
        report_metadata = _build_report_metadata(config.report_metadata, corpus, environment, profile, hardware, runner_section)
        report = assemble_report(
            protocol, corpus, config.report_schema, config.corpus_schema,
            report_metadata, config.candidate_entries, plan, completed,
        )
        write_report_pair(
            json_path, csv_path, report, protocol, config.report_schema, corpus, config.corpus_schema,
            replace=config.replace,
        )
        _write_summary_txt(summary_path, report, plan)
        _regenerate_sidecar_outputs(config.output_dir, review_dir, corpus, plan, completed, replace=config.replace)
        # Fix M: .run-identity.json was already established up front, before this report was
        # published (see `_establish_run_identity_up_front`) -- it is not written here again, so
        # it can never trail report.json.
        runner_log.write("run complete; report.json/summary.txt/nvml.csv/review.csv published")

    output_paths = {
        "report.json": json_path,
        "report.csv": csv_path,
        "summary.txt": summary_path,
        "runner.log": runner_log.path,
        "nvml.csv": config.output_dir / "nvml.csv",
        "review.csv": config.output_dir / "review.csv",
    }
    return RunResult(report=report, output_paths=output_paths, incomplete=False)


def _default_hardware() -> dict[str, str]:
    return {
        "platform": platform_module.system().lower(),
        "architecture": platform_module.machine().lower(),
        "os_release": platform_module.platform(),
        "cpu": platform_module.processor() or platform_module.machine(),
    }


def _default_source_commit() -> str | None:
    try:
        output = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True, timeout=10, check=False,
        )
    except OSError:
        return None
    commit = output.stdout.strip()
    if output.returncode == 0 and len(commit) == 40:
        return commit
    return None


def _build_report_metadata(
    raw_metadata: Mapping[str, Any],
    corpus: Mapping[str, Any],
    environment: str,
    profile: str,
    hardware: Mapping[str, Any],
    runner_section: Mapping[str, Any],
) -> dict[str, Any]:
    # _require_runner_fields already validated evaluator_sha256/runtime/runtime_sha256 are
    # present, earlier, as part of computing the resume identity -- not re-checked here.
    runner = dict(runner_section)
    runner.setdefault("name", "ww-bakeoff")
    runner.setdefault("version", "0.1.0")
    runner.setdefault("source_commit", _default_source_commit() or "0" * 40)
    runner.setdefault("command", " ".join(sys.argv))

    hardware_out = {key: value for key, value in hardware.items() if key != "environment" and value}

    report_id = raw_metadata.get("report_id")
    if not isinstance(report_id, str) or not report_id:
        report_id = f"p25-6-{profile}-{datetime.now(timezone.utc).strftime('%Y%m%dt%H%M%Sz')}"

    metadata: dict[str, Any] = {
        "report_id": report_id,
        "corpus_id": corpus["corpus_id"],
        "profile": profile,
        "environment": environment,
        "started_utc": raw_metadata.get("started_utc") or datetime.now(timezone.utc).isoformat(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "runner": runner,
        "hardware": hardware_out,
    }
    if "warnings" in raw_metadata:
        metadata["warnings"] = list(raw_metadata["warnings"])
    if "summary" in raw_metadata:
        metadata["summary"] = dict(raw_metadata["summary"])
    return metadata


# --------------------------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------------------------


def _load_runtime_and_array_modules() -> tuple[Any, Any]:
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise DependencyFailure("runtime_error", "NumPy is required to run the driver") from exc
    # Reuses evaluator's private runtime-loading helper (ORT, falling back to the vendored
    # native CUDA-12 bridge) rather than duplicating that fallback logic here.
    try:
        from . import evaluator as evaluator_module
    except ImportError:  # pragma: no cover - supports direct air-gapped invocation
        import evaluator as evaluator_module  # type: ignore
    return np, evaluator_module._onnxruntime()


def _load_nvml_backend_factory() -> Callable[[], NvmlBackend]:
    def factory() -> NvmlBackend:
        return PynvmlBackend()
    return factory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--artifact-map", required=True, type=Path)
    parser.add_argument("--report-metadata", required=True, type=Path)
    parser.add_argument("--report-schema", type=Path, default=DEFAULT_REPORT_SCHEMA)
    parser.add_argument("--corpus-schema", type=Path, default=DEFAULT_CORPUS_SCHEMA)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--state", type=Path, help="defaults to <output-dir>/state.json")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--poll-interval-s", type=float, default=0.05)
    parser.add_argument(
        "--chain-offset", dest="chain_offsets", type=int, action="append",
        help="repeatable; defaults to 1 2 4 8",
    )
    parser.add_argument("--no-nvml", action="store_true", help="skip NVML sampling even for CUDA cells")
    parser.add_argument("--replace", action="store_true", help="allow replacing existing report.json/report.csv")
    parser.add_argument(
        "--assume-host-load-ready", action="store_true",
        help=(
            "skip the interactive idle/live_flame confirmation prompt at each CUDA host_load "
            "boundary (for unattended reruns); the boundary is still logged to runner.log"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        protocol = load_json(args.protocol)
        corpus = load_json(args.corpus)
        candidate_entries = load_json(args.candidates)
        selection = load_json(args.selection)
        artifact_map = load_json(args.artifact_map)
        report_metadata = load_json(args.report_metadata)
        report_schema = load_json(args.report_schema)
        corpus_schema = load_json(args.corpus_schema)
    except (OSError, ValueError) as exc:
        print(f"run: cannot load input: {exc}", file=sys.stderr)
        return 2

    try:
        array_module, runtime_module = _load_runtime_and_array_modules()
    except DependencyFailure as exc:
        print(f"run: {exc}", file=sys.stderr)
        return 2

    nvml_factory = None if args.no_nvml else _load_nvml_backend_factory()

    output_dir = args.output_dir
    state_path = args.state or (output_dir / "state.json")
    config = RunConfig(
        protocol=protocol,
        corpus=corpus,
        candidate_entries=candidate_entries,
        selection=selection,
        artifact_map=artifact_map,
        report_schema=report_schema,
        corpus_schema=corpus_schema,
        output_dir=output_dir,
        state_path=state_path,
        device_index=args.device_index,
        poll_interval_s=args.poll_interval_s,
        chain_offsets=tuple(args.chain_offsets) if args.chain_offsets else DEFAULT_CHAIN_OFFSETS,
        report_metadata=report_metadata,
        protocol_path=args.protocol,
        replace=args.replace,
        runtime_module=runtime_module,
        array_module=array_module,
        nvml_backend_factory=nvml_factory,
        exr_decoder=exr_module.frame_from_exr,
        host_load_checkpoint=(
            auto_confirm_host_load_checkpoint if args.assume_host_load_ready else interactive_host_load_checkpoint
        ),
    )
    try:
        result = run_bakeoff(config)
    except (DriverFailure, MatrixFailure, ResumeFailure, CoordinatorFailure, ReportFailure) as exc:
        print(f"run: {exc}", file=sys.stderr)
        return 2

    if result.incomplete:
        print("run: interrupted before every cell completed; rerun the same command to resume", file=sys.stderr)
        return 1
    print(f"run: published {sorted(result.output_paths)} in {output_dir}")
    return 0


__all__ = [
    "CudaMeasurementResult",
    "CudaMeasurementRunner",
    "DriverFailure",
    "HostLoadCheckpoint",
    "REVIEW_CSV_HEADER",
    "RunConfig",
    "RunResult",
    "auto_confirm_host_load_checkpoint",
    "build_executor",
    "interactive_host_load_checkpoint",
    "main",
    "run_bakeoff",
    "run_cuda_measurement_in_subprocess",
]


if __name__ == "__main__":
    raise SystemExit(main())
