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
``run_tests.py`` -- import and run without numpy, onnxruntime, the OpenEXR bindings, pynvml, or a GPU.
"""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import json
import math
import multiprocessing as _mp
import os
import platform as platform_module
import queue as _queue_module
import re
import struct
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

if not __package__:
    # Keep direct python tools/bakeoff/run.py invocation on the same package-import path as
    # python -m tools.bakeoff.run. The bake-off siblings intentionally use relative imports,
    # so importing them as top-level modules would fail (or create duplicate module state).
    _REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPOSITORY_ROOT))
    __package__ = "tools.bakeoff"

from . import exr as exr_module
from . import synthetic as synthetic_module
from .artifact_store import ArtifactStore, ArtifactStoreFailure
from .coordinator import CommittedExecution, CoordinatorFailure, IncompleteFailure, RunCoordinator
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
from .nvml import NVML_CSV_HEADER, STAGES as NVML_STAGES, NvmlBackend, NvmlSampler, PynvmlBackend
from .resume import ResumeFailure, create_state, load_state
from .reporting import ReportFailure, assemble_report, render_csv, write_report_pair
from .run_spec import IDENTITY_SCHEMA_VERSION, RunSpec, RunSpecError
from .synthetic import SyntheticCase, encode_pfm
from .validator import (
    ValidationError,
    canonical_sha256,
    load_json,
    validate_corpus_consistency,
    validate_report_consistency,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = V2_PROTOCOL
DEFAULT_REPORT_SCHEMA = ROOT / "bakeoff" / "report-v2.schema.json"
DEFAULT_CORPUS_SCHEMA = ROOT / "bakeoff" / "corpus-v1.schema.json"
DEFAULT_CHAIN_OFFSETS: tuple[int, ...] = (1, 2, 4, 8)
DEFAULT_RUNNER_NAME = "ww-bakeoff"
DEFAULT_RUNNER_VERSION = "0.1.0"

# These are the only runner/hardware metadata properties accepted by report-v2.  Keep this
# boundary local to the driver so an operator typo cannot become a stable identity field while
# also being silently omitted from the published report.  ``command`` is accepted as a known
# report property but is generated at publication time and never enters the stable runner map.
_RUNNER_REPORT_KEYS = frozenset({
    "name", "version", "source_commit", "evaluator_sha256", "runtime", "runtime_sha256", "command",
})
_HARDWARE_REPORT_KEYS = frozenset({
    "platform", "architecture", "os_release", "cpu", "gpu", "driver",
})
_RUNNER_REQUIRED_KEYS = ("name", "version", "source_commit", "evaluator_sha256", "runtime", "runtime_sha256")
# The report schema only marks platform/architecture as structurally required, but every
# supported CPU/CUDA/CoreML matrix needs OS and CPU identity too.  Defaults are filled before
# this validation, so the stable/report surfaces always agree on these four strings.
_HARDWARE_REQUIRED_KEYS = ("platform", "architecture", "os_release", "cpu")
_OPTIONAL_HARDWARE_KEYS = frozenset({"gpu", "driver"})

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

_NVML_IDENTITY_FIELDS: tuple[str, ...] = (
    "candidate_id", "shot_id", "conditioning_token", "cap_token", "provider", "host_load",
)
_REQUIRED_FINAL_NVML_STAGES = frozenset({
    "baseline", "session_create", "steady", "cleanup", "process_exit",
})


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


def _canonical_chain_offsets(values: Any) -> tuple[int, ...]:
    """Return the execution/identity representation of the requested chain offsets.

    Chain offsets are a set of requested measurements, not an ordered user-visible list.  The
    driver therefore validates the actual integer type and stores one sorted, duplicate-free
    tuple before either building the executor or hashing the ``RunSpec``.  Rejecting rather than
    coercing strings/floats keeps a malformed CLI/test boundary from changing the requested
    measurement accidentally.
    """

    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        _fail("chain_offsets", "chain_offsets must be a non-empty sequence of integers")
    offsets: list[int] = []
    for index, value in enumerate(values):
        if type(value) is not int:
            _fail("chain_offsets", f"chain_offsets[{index}] must be an integer")
        offsets.append(value)
    if not offsets:
        _fail("chain_offsets", "chain_offsets must contain at least one offset")
    return tuple(sorted(set(offsets)))


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


def _freeze_bundle_value(value: Any) -> Any:
    """Detach JSON-shaped result data into immutable containers for a :class:`CellBundle`."""

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_bundle_value(child) for key, child in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_bundle_value(child) for child in value)
    return value


def _thaw_bundle_value(value: Any) -> Any:
    """Turn a frozen bundle value back into the plain JSON containers the coordinator accepts."""

    if isinstance(value, Mapping):
        return {key: _thaw_bundle_value(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw_bundle_value(child) for child in value]
    return value


@dataclass(frozen=True)
class PreviewPayload:
    """One immutable preview file staged inside a committed cell generation."""

    relative_path: str
    payload: bytes

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", bytes(self.payload))


@dataclass(frozen=True)
class CellBundle:
    """The complete immutable output of one cell execution.

    The public report result is always present.  NVML rows and previews are optional evidence
    payloads; no member of this bundle writes to the output directory.  The frozen result tree
    and tuple payloads prevent a later staging/retry operation from observing caller mutations.
    """

    result: Mapping[str, Any]
    nvml_rows: tuple[tuple[str, ...], ...] = ()
    previews: tuple[PreviewPayload, ...] = ()
    log_messages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "result", _freeze_bundle_value(dict(self.result)))
        object.__setattr__(
            self,
            "nvml_rows",
            tuple(tuple(row) for row in self.nvml_rows),
        )
        normalized_previews: list[PreviewPayload] = []
        for preview in self.previews:
            if isinstance(preview, PreviewPayload):
                normalized_previews.append(preview)
            else:
                relative_path, payload = preview  # type: ignore[misc]
                normalized_previews.append(PreviewPayload(relative_path, payload))
        object.__setattr__(self, "previews", tuple(normalized_previews))
        object.__setattr__(self, "log_messages", tuple(self.log_messages))

    def public_result(self) -> dict[str, Any]:
        """Return a detached plain result mapping suitable for coordinator validation."""

        value = _thaw_bundle_value(self.result)
        if not isinstance(value, dict):  # pragma: no cover - constructor enforces a mapping
            raise TypeError("CellBundle result must be a mapping")
        return value


def _failure(kind: str, message: str, *, stage: str | None = None, retryable: bool | None = None) -> dict[str, Any]:
    failure: dict[str, Any] = {"type": kind, "message": message}
    if stage is not None:
        failure["stage"] = stage
    if retryable is not None:
        failure["retryable"] = retryable
    return failure


def _is_finite_number(value: Any) -> bool:
    """True only for a real finite int/float (never a bool, NaN, or +/-inf).

    A computed OPTIONAL metric that comes back nonfinite would otherwise be rejected by
    ``coordinator._validate_json`` as a hard failure aborting the whole run; the driver uses this
    to drop such a value to ``not_applicable`` with a logged reason instead.
    """

    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


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


def _preview_payloads(
    offset: int, image2: Any, flow: Any, width: int, height: int,
) -> tuple[PreviewPayload, PreviewPayload]:
    """Render one immutable preview pair without touching the filesystem."""

    all_visible = [[True] * width for _ in range(height)]
    warped = metrics_module.warp_forward_samples(image2, flow, all_visible)
    return (
        PreviewPayload(
            f"previews/offset_{offset}_warped.pfm",
            encode_pfm(_grid_or_zero(warped, width, height), width, height),
        ),
        PreviewPayload(
            f"previews/offset_{offset}_flow.pfm",
            encode_pfm(_flow_visualization_grid(flow, width, height), width, height),
        ),
    )


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


def _resource_exhaustion_or_runtime_kind(exc: BaseException) -> str:
    """Map an infrastructure error (a failed fork, an NVML device-query failure) onto a permitted
    coordinator failure kind: ``out_of_memory`` for a resource-exhaustion errno (ENOMEM/EAGAIN --
    the realistic live_flame memory-pressure signature), otherwise ``runtime_error``.

    Shared by both CUDA-subprocess boundaries the parent owns -- the ``process.start()`` fork and
    the post-``join()`` device reading -- so a failure at either becomes a TYPED per-cell failure
    (via ``_run_cell``'s ``except DriverFailure``) instead of an uncaught exception that aborts the
    whole matrix. Both kinds are in ``_RESULT_FAILURE_TYPES`` and the coordinator's own permitted
    set.
    """

    errno_value = getattr(exc, "errno", None)
    return "out_of_memory" if errno_value in (errno.ENOMEM, errno.EAGAIN) else "runtime_error"


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
    try:
        process.start()
    except OSError as exc:
        # A failed fork()/clone() (no child was ever created, so there is nothing to reap) must
        # be a TYPED per-cell failure, not an uncaught OSError that aborts the whole matrix --
        # realistic on the documented live_flame final run, where measuring CUDA beside a live
        # Flame Batch puts the box under memory pressure. _run_cell's existing `except
        # DriverFailure` turns this into a _CellFail with the same coordinator-permitted kind,
        # exactly as it already does for a child that dies mid-measurement (Fix H).
        raise DriverFailure(
            _resource_exhaustion_or_runtime_kind(exc),
            f"CUDA measurement subprocess could not be started (fork failed, errno={exc.errno}): {exc}",
        ) from exc
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

    # The measured work already succeeded in the child; this is the PARENT's own post-join
    # device reading (the true "process_exit" evidence, taken only after the CUDA context has
    # actually torn down). Constructing the backend, resolving the handle, or querying device
    # memory can all raise -- an NVML query_failed/device_unavailable, a pynvml import/init
    # error, or an OSError. None of those may escape to abort the whole matrix: like the
    # fork-start boundary above, they become a TYPED per-cell failure that _run_cell maps to a
    # permitted cell failure. (per-process accounting for an already-exited pid stays best-effort
    # and degrades to None, since a recycled/absent pid legitimately has no reading.)
    try:
        exit_backend = nvml_backend_factory()
        handle = exit_backend.device_handle(device_index)
        used_mib = exit_backend.device_used_mib(handle)
    except Exception as exc:  # noqa: BLE001 - NvmlFailure/OSError/any device-query failure is contained
        raise DriverFailure(
            _resource_exhaustion_or_runtime_kind(exc),
            f"post-exit NVML device reading failed for device {device_index}: {exc}",
        ) from exc
    try:
        process_used_mib = exit_backend.process_used_mib(handle, process.pid)
    except Exception:  # noqa: BLE001 - per-process accounting for an exited pid is best-effort
        process_used_mib = None
    return CudaMeasurementResult(payload, samples, used_mib, process_used_mib)


# --------------------------------------------------------------------------------------------
# Transactional per-cell bundle publication and exact-generation evidence.
# --------------------------------------------------------------------------------------------


def _artifact_cell_id(cell: CellKey) -> str:
    """Return the stable artifact-store cell id derived from canonical CellKey JSON."""

    encoded = json.dumps(
        cell.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_result_bytes(result: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                result, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, OverflowError) as exc:
        raise ArtifactStoreFailure("result_encoding", f"cell result is not canonical UTF-8 JSON: {exc}") from exc


def _validate_nvml_float_text(value: str, field: str, row_index: int) -> None:
    """Validate one canonical non-negative finite number emitted by :mod:`nvml`."""

    try:
        parsed = float(value)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ArtifactStoreFailure(
            "nvml_rows_value", f"NVML row {row_index} {field} must be a finite number",
        ) from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ArtifactStoreFailure(
            "nvml_rows_value", f"NVML row {row_index} {field} must be a non-negative finite number",
        )
    if repr(float(parsed)) != value:
        raise ArtifactStoreFailure(
            "nvml_rows_value", f"NVML row {row_index} {field} is not canonical numeric text",
        )


def _canonical_nvml_rows_bytes(
    rows: Sequence[Sequence[str]],
    *,
    expected_identity: Mapping[str, str] | None = None,
    required_stages: set[str] | frozenset[str] | None = None,
) -> bytes:
    """Return deterministic, cell-bound NVML evidence bytes.

    The optional identity and required-stage arguments are used at the CellBundle and exact-ref
    boundaries.  Keeping the generic form available preserves the small decoder seam used by
    existing tests, while every production commit/regeneration supplies the current CellKey
    identity.
    """

    normalized: list[list[str]] = []
    expected_prefix: tuple[str, ...] | None = None
    if expected_identity is not None:
        if set(expected_identity) != set(_NVML_IDENTITY_FIELDS):
            raise ArtifactStoreFailure(
                "nvml_identity", "expected NVML identity must contain exactly the six cell fields",
            )
        expected_prefix = tuple(expected_identity[field] for field in _NVML_IDENTITY_FIELDS)
    for index, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != len(NVML_CSV_HEADER):
            raise ArtifactStoreFailure(
                "nvml_rows_shape",
                f"NVML row {index} must contain exactly {len(NVML_CSV_HEADER)} columns",
            )
        if any(type(value) is not str for value in row):
            raise ArtifactStoreFailure("nvml_rows_shape", f"NVML row {index} must contain only strings")
        if expected_prefix is not None and tuple(row[:len(_NVML_IDENTITY_FIELDS)]) != expected_prefix:
            raise ArtifactStoreFailure(
                "nvml_identity",
                f"NVML row {index} identity does not match the current cell",
            )
        stage = row[6]
        if stage not in NVML_STAGES:
            raise ArtifactStoreFailure(
                "nvml_stage", f"NVML row {index} has unsupported stage {stage!r}",
            )
        if re.fullmatch(r"(?:0|[1-9][0-9]*)", row[7]) is None:
            raise ArtifactStoreFailure(
                "nvml_sample_index",
                f"NVML row {index} sample_index must be a canonical non-negative integer",
            )
        _validate_nvml_float_text(row[8], "timestamp_unix_s", index)
        _validate_nvml_float_text(row[9], "device_used_mib", index)
        if row[10] != "":
            _validate_nvml_float_text(row[10], "process_used_mib", index)
        normalized.append(list(row))
    if required_stages is not None:
        present_stages = {row[6] for row in normalized}
        missing = sorted(set(required_stages) - present_stages)
        if missing:
            raise ArtifactStoreFailure(
                "nvml_stages", f"NVML evidence is missing required stages: {', '.join(missing)}",
            )
    try:
        return (
            json.dumps(
                {"rows": normalized},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, OverflowError) as exc:
        raise ArtifactStoreFailure("nvml_rows_encoding", f"NVML rows are not canonical UTF-8 JSON: {exc}") from exc


def _decode_nvml_rows(
    payload: bytes,
    *,
    expected_identity: Mapping[str, str] | None = None,
    required_stages: set[str] | frozenset[str] | None = None,
) -> list[list[str]]:
    """Validate one exact canonical NVML evidence payload before CSV regeneration."""

    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ArtifactStoreFailure("nvml_rows_encoding", f"NVML evidence is not UTF-8 JSON: {exc}") from exc
    if type(value) is not dict or set(value) != {"rows"} or type(value["rows"]) is not list:
        raise ArtifactStoreFailure("nvml_rows_shape", "NVML evidence must contain only a rows list")
    rows = value["rows"]
    canonical = _canonical_nvml_rows_bytes(
        rows, expected_identity=expected_identity, required_stages=required_stages,
    )
    if canonical != payload:
        raise ArtifactStoreFailure("nvml_rows_encoding", "NVML evidence is not canonical JSON")
    return [list(row) for row in rows]


def _validate_committed_ref(
    store: ArtifactStore,
    cell: CellKey,
    result: Mapping[str, Any],
    artifact_ref: Mapping[str, Any],
) -> None:
    """Prove that one exact generation belongs to this cell and this public result."""

    if not isinstance(artifact_ref, Mapping):
        raise ArtifactStoreFailure("artifact_ref_shape", "artifact_ref must be a mapping")
    expected_cell_id = _artifact_cell_id(cell)
    if artifact_ref.get("cell_id") != expected_cell_id:
        raise ArtifactStoreFailure(
            "cell_mismatch",
            f"artifact_ref cell_id does not match CellKey {cell!r}",
        )
    # load_ref validates the immutable manifest and every named artifact before the exact read.
    store.load_ref(artifact_ref)
    stored = store.read_artifact(artifact_ref, "result.json")
    expected = _canonical_result_bytes(result)
    if stored != expected:
        raise ArtifactStoreFailure(
            "result_artifact_mismatch",
            f"result.json does not match the persisted result for {cell!r}",
        )


def _commit_cell_bundle(
    store: ArtifactStore,
    cell: CellKey,
    bundle: CellBundle,
    *,
    nvml_enabled: bool,
    require_nvml_stages: bool = False,
) -> CommittedExecution:
    """Commit one complete CellBundle, retrying optional preview staging without the previews."""

    result = bundle.public_result()
    required_nvml = result.get("status") == "pass" and result.get("provider") == "cuda" and nvml_enabled
    if required_nvml and not bundle.nvml_rows:
        raise ArtifactStoreFailure(
            "nvml_missing",
            f"CUDA cell {cell!r} passed with NVML enabled but returned no NVML evidence rows",
        )
    result_payload = _canonical_result_bytes(result)
    nvml_payload = (
        _canonical_nvml_rows_bytes(
            bundle.nvml_rows,
            expected_identity=_base_result_fields(cell),
            required_stages=_REQUIRED_FINAL_NVML_STAGES if require_nvml_stages else None,
        )
        if required_nvml else None
    )

    def publish(*, include_previews: bool) -> CommittedExecution:
        attempt = store.begin(_artifact_cell_id(cell))
        try:
            attempt.stage_bytes("result.json", result_payload)
            if required_nvml:
                assert nvml_payload is not None
                attempt.stage_bytes("evidence/nvml_rows.json", nvml_payload)
            if include_previews:
                for preview in bundle.previews:
                    try:
                        attempt.stage_bytes(preview.relative_path, preview.payload)
                    except ArtifactStoreFailure as exc:
                        # This marker is consumed only by the outer optional-preview retry.  The
                        # attempt is already closed/poisoned by ArtifactAttempt on a publication
                        # failure, so no partially staged generation can later complete state.
                        setattr(exc, "optional_preview_failure", True)
                        raise
            try:
                artifact_ref = attempt.commit()
            except ArtifactStoreFailure as exc:
                artifact_ref = getattr(exc, "artifact_ref", None)
                if artifact_ref is None:
                    raise
            _validate_committed_ref(store, cell, result, artifact_ref)
            if required_nvml:
                evidence = store.read_artifact(artifact_ref, "evidence/nvml_rows.json")
                if evidence != nvml_payload:
                    raise ArtifactStoreFailure("nvml_rows_mismatch", "committed NVML evidence differs from the bundle")
            if include_previews:
                for preview in bundle.previews:
                    if store.read_artifact(artifact_ref, preview.relative_path) != preview.payload:
                        raise ArtifactStoreFailure(
                            "preview_mismatch",
                            f"committed preview differs from the bundle: {preview.relative_path}",
                        )
            return CommittedExecution(result=result, artifact_ref=dict(artifact_ref))
        except ArtifactStoreFailure:
            # ArtifactAttempt.commit/stage closes itself on every publication failure.  Explicitly
            # retaining no attempt object outside this scope prevents a failed required artifact
            # from being accidentally reused as a completion.
            raise

    try:
        return publish(include_previews=True)
    except ArtifactStoreFailure as exc:
        if bundle.previews and getattr(exc, "optional_preview_failure", False):
            # The old generation, if any, is untouched.  A fresh attempt re-stages result and
            # required evidence, then commits a valid pass with no preview artifacts.
            return publish(include_previews=False)
        raise


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
    review_enabled: bool
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
    if not shot_ctx.has_analytic_truth and ctx.review_enabled:
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


def _run_cell(cell: CellKey, ctx: _RunnerContext) -> CellBundle:
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

    log_messages: list[str] = []

    # Fix B: a supervised host-load boundary, confirmed once per contiguous run of cells that
    # share one CUDA host_load value, before any cell in that group executes. Re-checked every
    # invocation (ctx.confirmed_host_load always starts None on a fresh process), so a resumed
    # run always reconfirms rather than trusting stale state from before an interruption.
    if cell.provider == "cuda" and ctx.confirmed_host_load != cell.host_load:
        ctx.host_load_checkpoint(cell.host_load)
        ctx.confirmed_host_load = cell.host_load
        log_messages.append(f"host load boundary confirmed: host_load={cell.host_load}")

    # Fix G: for a CUDA cell, EVERY inference this cell needs (base + reverse + chain links +
    # review offsets) runs inside _perform_cell_inference, and that whole function is what gets
    # isolated in the measurement child -- the parent below this point does no inference at all
    # for a CUDA cell, so it never itself initializes a CUDA context (the unsafe
    # fork-after-CUDA-init condition this subprocess design exists to avoid). CPU/CoreML cells
    # run the identical function directly, in-process, with no isolation.
    def _inference_work(stage_sampler: Callable[[str], Any] | None) -> dict[str, Any]:
        return _perform_cell_inference(cell, ctx, artifact, shot_ctx, stage_sampler)

    nvml_rows: list[list[str]] = []
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
        try:
            resource, nvml_rows = _replay_resource_and_rows(
                _base_result_fields(cell), nvml_samples, ctx.device_index,
            )
        except Exception as exc:  # noqa: BLE001 - malformed sampler payload is a cell-local failure
            raise _CellFail(
                _failure("runtime_error", f"NVML evidence reduction failed: {exc}", stage="resource"),
            ) from exc
        if ctx.profile == "final":
            present_stages = {
                sample.get("stage")
                for sample in nvml_samples
                if isinstance(sample, Mapping)
            }
            missing_stages = sorted(_REQUIRED_FINAL_NVML_STAGES - present_stages)
            if missing_stages:
                raise _CellFail(_failure(
                    "runtime_error",
                    "required NVML stages missing: " + ", ".join(missing_stages),
                    stage="resource",
                ))
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
    log_messages.extend(payload.get("log_messages", []))

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
        # A nonfinite derived value computed from finite flow (e.g. a degenerate division) must
        # never reach the coordinator -- coordinator._validate_json rejects any nonfinite number
        # as a hard CoordinatorFailure that would abort the WHOLE run. An OPTIONAL metric that
        # comes back nonfinite degrades to a logged not_applicable, exactly like the MetricFailure
        # path below, so the cell still passes and its required metrics are unaffected.
        if not _is_finite_number(visible):
            log_messages.append(
                f"visible_warp_residual nonfinite ({visible!r}); dropped to not_applicable for {cell.as_dict()}"
            )
        else:
            result_metrics["visible_warp_residual"] = visible
            not_applicable.discard("visible_warp_residual")
    except MetricFailure as exc:
        log_messages.append(f"visible_warp_residual skipped for {cell.as_dict()}: {exc}")

    # Forward/backward residual: optional; the reverse-direction inference already happened (or
    # was skipped, with a log message already emitted above) in _perform_cell_inference.
    reverse_flow = payload.get("reverse_flow")
    if reverse_flow is not None:
        try:
            residual_mask = [[True] * analysis_width for _ in range(analysis_height)]
            residual = metrics_module.forward_backward_residual_px(predicted_flow, reverse_flow, residual_mask)
            if not _is_finite_number(residual):
                log_messages.append(
                    f"forward_backward_residual_px nonfinite ({residual!r}); dropped to not_applicable for {cell.as_dict()}"
                )
            else:
                result_metrics["forward_backward_residual_px"] = residual
                not_applicable.discard("forward_backward_residual_px")
        except MetricFailure as exc:
            log_messages.append(f"forward_backward_residual_px skipped for {cell.as_dict()}: {exc}")

    result_metrics["not_applicable"] = sorted(not_applicable)

    # Fix C: blinded per-candidate previews of THIS candidate's own predicted output -- a
    # target-warped-by-predicted-flow image plus a flow visualization -- across every offset in
    # ctx.chain_offsets that was actually inferrable (Fix G: inference already happened in
    # _perform_cell_inference; this only renders immutable payloads from the already-returned
    # flow/image data).
    preview_payloads: list[PreviewPayload] = []
    if not shot_ctx.has_analytic_truth and ctx.review_enabled:
        preview_jobs: list[tuple[int, Any, Any, int, int]] = [
            (1, image2, predicted_flow, analysis_width, analysis_height),
        ]
        for offset_key, offset_data in (payload.get("review_offsets") or {}).items():
            preview_jobs.append((
                int(offset_key), offset_data["image2"], offset_data["flow"],
                offset_data["width"], offset_data["height"],
            ))
        for offset, preview_image, preview_flow, preview_width, preview_height in preview_jobs:
            # Preview rendering is non-load-bearing review evidence. Filesystem staging occurs
            # later as part of the one transactional CellBundle, so a store failure cannot leave
            # a partial public preview behind.
            try:
                preview_payloads.extend(
                    _preview_payloads(
                        offset, preview_image, preview_flow, preview_width, preview_height,
                    )
                )
            except (MetricFailure, ValueError) as exc:
                log_messages.append(f"review preview offset {offset} skipped for {cell.as_dict()}: {exc}")

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
    return CellBundle(
        result={**_base_result_fields(cell), "status": "pass", **passing},
        nvml_rows=tuple(tuple(row) for row in nvml_rows),
        previews=tuple(preview_payloads),
        log_messages=tuple(log_messages),
    )


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
    review_enabled: bool,
    host_load_checkpoint: HostLoadCheckpoint,
    cuda_measurement_runner: CudaMeasurementRunner,
) -> tuple[Callable[[CellKey], CellBundle], _RunnerContext]:
    """Build the executor callback plus the mutable context RunCoordinator will drive."""

    canonical_offsets = _canonical_chain_offsets(chain_offsets)
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
        chain_offsets=canonical_offsets,
        exr_decoder=exr_decoder,
        review_enabled=review_enabled,
        host_load_checkpoint=host_load_checkpoint,
        cuda_measurement_runner=cuda_measurement_runner,
    )

    def executor(cell: CellKey) -> CellBundle:
        base = _base_result_fields(cell)
        start_message = f"cell start {json.dumps(cell.as_dict(), sort_keys=True)}"
        try:
            bundle = _run_cell(cell, ctx)
        except _CellFail as failure:
            return CellBundle(
                result={**base, "status": "fail", "failure": failure.failure},
                log_messages=(start_message,),
            )
        except _CellSkip as skip:
            return CellBundle(
                result={**base, "status": "skip", "failure": skip.failure},
                log_messages=(start_message,),
            )
        return CellBundle(
            result=bundle.public_result(),
            nvml_rows=bundle.nvml_rows,
            previews=bundle.previews,
            log_messages=(start_message, *bundle.log_messages),
        )

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


def _normalise_hex(value: Any, path: str, length: int) -> str:
    """Validate a hex identity field and normalize acceptable upper-case hex to lower-case."""

    if not isinstance(value, str) or not value:
        _fail("report_metadata_value", f"{path} must be a non-empty {length}-hex string")
    if re.fullmatch(rf"[0-9a-fA-F]{{{length}}}", value) is None:
        _fail("report_metadata_value", f"{path} must be a {length}-hex string")
    return value.lower()


def _require_nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        _fail("report_metadata_value", f"{path} must be a non-empty string")
    return value


def _require_runner_fields(runner_section: Mapping[str, Any]) -> None:
    """Validate the report-v2 runner fields after normalization."""

    for required in _RUNNER_REQUIRED_KEYS:
        _require_nonempty_string(runner_section.get(required), f"report_metadata.runner.{required}")
    _normalise_hex(runner_section["source_commit"], "report_metadata.runner.source_commit", 40)
    _normalise_hex(runner_section["evaluator_sha256"], "report_metadata.runner.evaluator_sha256", 64)
    _normalise_hex(runner_section["runtime_sha256"], "report_metadata.runner.runtime_sha256", 64)


def _normalise_runner_metadata(raw_runner: Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete stable runner surface with deterministic defaults.

    ``command`` is generated publication metadata, not a runner semantic.  It is deliberately
    removed before identity formation; ``_build_report_metadata`` adds the invocation command
    back to the published report after the stable defaults have been fixed.
    """

    if not isinstance(raw_runner, Mapping):
        _fail("report_metadata_shape", "report_metadata.runner must be an object")
    unknown = [key for key in raw_runner if key not in _RUNNER_REPORT_KEYS]
    if unknown:
        unknown.sort(key=repr)
        _fail("report_metadata_unknown", f"report_metadata.runner has unknown key {unknown[0]!r}")

    runner: dict[str, Any] = {}
    for key, default in (
        ("name", DEFAULT_RUNNER_NAME),
        ("version", DEFAULT_RUNNER_VERSION),
        ("source_commit", _default_source_commit() or "0" * 40),
    ):
        if key in raw_runner:
            runner[key] = _require_nonempty_string(raw_runner[key], f"report_metadata.runner.{key}")
        else:
            runner[key] = default
    runner["evaluator_sha256"] = _normalise_hex(
        raw_runner.get("evaluator_sha256"), "report_metadata.runner.evaluator_sha256", 64,
    )
    runner["runtime"] = _require_nonempty_string(
        raw_runner.get("runtime"), "report_metadata.runner.runtime",
    )
    runner["runtime_sha256"] = _normalise_hex(
        raw_runner.get("runtime_sha256"), "report_metadata.runner.runtime_sha256", 64,
    )
    runner["source_commit"] = _normalise_hex(
        runner["source_commit"], "report_metadata.runner.source_commit", 40,
    )
    _require_runner_fields(runner)
    return runner


def _normalise_hardware(raw_hardware: Mapping[str, Any]) -> dict[str, Any]:
    """Return report-schema hardware with deterministic defaults and no absent optionals."""

    if not isinstance(raw_hardware, Mapping):
        _fail("report_metadata_shape", "report_metadata.hardware must be an object")
    unknown = [key for key in raw_hardware if key not in _HARDWARE_REPORT_KEYS]
    if unknown:
        unknown.sort(key=repr)
        _fail("report_metadata_unknown", f"report_metadata.hardware has unknown key {unknown[0]!r}")

    hardware = _default_hardware()
    for key in _HARDWARE_REQUIRED_KEYS:
        _require_nonempty_string(hardware.get(key), f"default report_metadata.hardware.{key}")
    for key, value in raw_hardware.items():
        if value is not None:
            if key in _OPTIONAL_HARDWARE_KEYS and value == "":
                # Empty optional hardware is the same absence as ``None``.  Dropping it here
                # keeps stable identity and report publication byte-for-byte aligned.
                continue
            hardware[key] = _require_nonempty_string(value, f"report_metadata.hardware.{key}")
    for key, value in tuple(hardware.items()):
        if key in _OPTIONAL_HARDWARE_KEYS and (value is None or value == ""):
            hardware.pop(key, None)
        elif key not in _HARDWARE_REPORT_KEYS:
            hardware.pop(key, None)
    for key in _HARDWARE_REQUIRED_KEYS:
        _require_nonempty_string(hardware.get(key), f"report_metadata.hardware.{key}")
    return hardware


def _stable_report_inputs(raw_metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Extract report-affecting operator inputs from publication metadata.

    Report IDs, timestamps, command lines and paths are generated publication data.  Warnings
    and operator-supplied summary/score fields are semantic report inputs and must invalidate a
    completed run when changed.  Preserve presence/absence so an explicitly supplied empty value
    cannot silently reuse a report that omitted the field.
    """

    if not isinstance(raw_metadata, Mapping):
        _fail("report_metadata_shape", "report_metadata must be an object")
    return {
        key: raw_metadata[key]
        for key in ("warnings", "summary")
        if key in raw_metadata
    }


def _build_run_spec(
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
    candidate_entries: Any,
    report_schema: Mapping[str, Any],
    corpus_schema: Mapping[str, Any],
    device_index: int,
    poll_interval_s: float,
    nvml_enabled: bool,
    report_inputs: Mapping[str, Any] | None = None,
    identity_schema_version: int = IDENTITY_SCHEMA_VERSION,
) -> RunSpec:
    """Bind every input that can change a cell's produced result or reported resource use.

    The profile and environment are kept separately from the selector because the same matrix
    axes can be measured under different execution profiles/environments.  The full selector is
    nevertheless retained under ``matrix`` so provider/host-load/candidate/shot axes cannot be
    silently omitted.

    Protocol, corpus, candidate-entry and schema payloads are represented by canonical hashes;
    mutating any of those source objects therefore changes the compact identity without copying
    their large contents into resume state.

    Device index, sampler interval, NVML enablement, chain offsets and every normalized hardware
    field remain explicit because they affect measured resource evidence or execution behavior.

    The complete validated artifact hash map binds the bytes actually used for inference, while
    the candidate-entry hash binds legal/admission content copied into the report. Schema hashes
    bind the validation contracts without persisting their full JSON bodies.

    Warnings and operator summary/score inputs are stable report semantics. Generated report ids,
    timestamps, commands and output paths are deliberately not supplied to ``RunSpec``.
    """

    stable_runner = _normalise_runner_metadata(runner_metadata)
    stable_hardware = _normalise_hardware(hardware)
    canonical_offsets = _canonical_chain_offsets(chain_offsets)
    selected_artifacts = {
        candidate_id: {
            "manifest_sha256": artifacts[candidate_id].manifest_sha256,
            "artifact_sha256": artifacts[candidate_id].artifact_sha256,
            "platform": artifacts[candidate_id].platform,
        }
        for candidate_id in plan.selector["candidate_ids"]
    }
    try:
        return RunSpec.from_inputs(
            protocol={
                "protocol_id": protocol["protocol_id"],
                "sha256": canonical_sha256(protocol),
            },
            corpus={
                "corpus_id": corpus.get("corpus_id", ""),
                "sha256": canonical_sha256(corpus),
            },
            candidate_entries={"sha256": canonical_sha256(candidate_entries)},
            selection={"profile": profile, "environment": environment},
            matrix=plan.selector,
            artifacts=selected_artifacts,
            report_schema={"sha256": canonical_sha256(report_schema)},
            corpus_schema={"sha256": canonical_sha256(corpus_schema)},
            environment=environment,
            profile=profile,
            runner=stable_runner,
            hardware=stable_hardware,
            chain_offsets=list(canonical_offsets),
            measurement={
                "device_index": device_index,
                "poll_interval_s": poll_interval_s,
                "nvml_enabled": nvml_enabled,
            },
            report_inputs=dict(report_inputs or {}),
            identity_schema_version=identity_schema_version,
            # Retain this compact compatibility field for diagnostics and old report-side
            # checks; the full selector above remains the authoritative matrix surface.
            extra_stable={"matrix_sha256": plan.matrix_sha256},
        )
    except (KeyError, TypeError, ValueError, RunSpecError) as exc:
        _fail("identity_invalid", f"cannot construct run identity: {exc}")


def _assert_run_spec_identity(run_spec: RunSpec) -> None:
    """Enforce the RunSpec hash invariant before handing identity to persistence subsystems."""

    try:
        run_spec.assert_identity()
    except RunSpecError as exc:
        _fail("identity_invalid", f"RunSpec identity does not match stable inputs: {exc}")


def _compute_identity(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compatibility adapter returning the persisted mapping from ``RunSpec``.

    New driver code calls ``_build_run_spec`` directly.  Keeping this narrow adapter lets older
    test fixtures and air-gapped operators inspect the same identity mapping during migration;
    no second hashing or field-selection implementation remains here.
    """

    return _build_run_spec(*args, **kwargs).stable_inputs


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
        protocol_identity = identity["protocol"]
        if not isinstance(protocol_identity, Mapping):
            return False
        if report.get("protocol_id") != protocol_identity["protocol_id"]:
            return False
        corpus_identity = identity["corpus"]
        if not isinstance(corpus_identity, Mapping):
            return False
        if report.get("corpus_sha256") != corpus_identity["sha256"]:
            return False
        if report.get("profile") != identity["profile"]:
            return False
        if report.get("environment") != identity["environment"]:
            return False
        matrix = report.get("matrix")
        matrix_identity = identity["matrix"]
        if not isinstance(matrix, Mapping) or matrix != matrix_identity:
            return False
        runner = report.get("runner")
        if not isinstance(runner, Mapping):
            return False
        stable_runner = identity["runner"]
        if not isinstance(stable_runner, Mapping):
            return False
        for key, value in stable_runner.items():
            if runner.get(key) != value:
                return False
        report_hardware = report.get("hardware")
        if not isinstance(report_hardware, Mapping):
            return False
        stable_hardware = identity["hardware"]
        if not isinstance(stable_hardware, Mapping):
            return False
        for key, value in stable_hardware.items():
            if report_hardware.get(key, "") != value:
                return False
        report_inputs = identity.get("report_inputs", {})
        if not isinstance(report_inputs, Mapping):
            return False
        if "warnings" in report_inputs and report.get("warnings") != report_inputs["warnings"]:
            return False
        if "summary" in report_inputs:
            stable_summary = report_inputs["summary"]
            if not isinstance(stable_summary, Mapping):
                return False
            report_summary = report.get("summary")
            if not isinstance(report_summary, Mapping):
                return False
            for key, value in stable_summary.items():
                if report_summary.get(key) != value:
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
        for candidate_id in identity["artifacts"]:
            if not isinstance(candidates_by_id.get(candidate_id), Mapping):
                return False
        return True
    except (KeyError, TypeError):
        return False


_REPORT_REUSE_VOLATILE_TOP_LEVEL = frozenset({"report_id", "started_utc", "completed_utc"})


def _report_semantic_projection(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical report-semantic surface used for reuse authorization.

    Report reuse is deliberately stricter than the identity-shaped preflight above.  The report
    itself is the complete public measurement record, so every field remains binding except the
    publication metadata that is generated afresh on a later invocation.  Only the three
    volatile top-level fields and ``runner.command`` are omitted; all ordered arrays, nested
    metrics, legal evidence, warnings, summary values, runner identity, and hardware identity
    remain in the projection.

    The caller validates the report against the current protocol and schemas before asking for a
    projection.  An unknown property or malformed nested value therefore cannot be smuggled
    through by simply being copied into both sides of a comparison.
    """

    projected = {
        key: value
        for key, value in report.items()
        if key not in _REPORT_REUSE_VOLATILE_TOP_LEVEL
    }
    runner = projected.get("runner")
    if isinstance(runner, Mapping):
        stable_runner = dict(runner)
        stable_runner.pop("command", None)
        projected["runner"] = stable_runner
    return projected


def _validate_existing_report(
    report: Any,
    *,
    json_path: Path,
    protocol: Mapping[str, Any],
    report_schema: Mapping[str, Any],
    corpus: Mapping[str, Any],
    corpus_schema: Mapping[str, Any],
) -> None:
    """Validate an existing report before any expected-report assembly or derivative repair."""

    try:
        validate_report_consistency(report, protocol, report_schema, corpus, corpus_schema)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        _fail(
            "report_identity_mismatch",
            f"{json_path} failed current report validation; refusing derivative repair: {exc}",
        )


def _compare_reusable_report_semantics(
    report: Mapping[str, Any],
    expected_report: Mapping[str, Any],
    *,
    json_path: Path,
) -> None:
    """Compare the complete report semantics after both sides have passed validation."""

    try:
        actual_semantics = canonical_sha256(_report_semantic_projection(report))
        expected_semantics = canonical_sha256(_report_semantic_projection(expected_report))
    except (TypeError, ValueError, UnicodeError) as exc:
        _fail(
            "report_identity_mismatch",
            f"{json_path} has no canonical semantic projection; refusing derivative repair: {exc}",
        )
    if actual_semantics != expected_semantics:
        _fail(
            "report_identity_mismatch",
            f"{json_path} differs from the current run's complete report semantics; "
            "refusing derivative repair",
        )


# --------------------------------------------------------------------------------------------
# Output files.
# --------------------------------------------------------------------------------------------


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _artifact_store_parent(output_dir: Path) -> Path:
    """Return the lexical artifact-store parent, allowing only macOS's trusted ``/var`` alias."""

    path = Path(output_dir)
    if not path.is_absolute():
        path = Path.cwd() / path
    # TemporaryDirectory on macOS commonly returns ``/var/...`` while the real namespace is
    # ``/private/var/...``.  Canonicalize only that known system alias; preserve every caller
    # supplied component (including ``..`` and non-system symlinks) for ArtifactStore's guard.
    if (
        len(path.parts) >= 2
        and path.parts[1] == "var"
        and ".." not in path.parts
        and Path("/var").is_symlink()
        and Path("/var").resolve() == Path("/private/var")
    ):
        path = Path("/private/var", *path.parts[2:])
    return path


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
    identity record is the authoritative record of exactly which identity is (or is about to be)
    producing the report/nvml.csv/review.csv in this output_dir, checked by hash (never by
    merely existing) before any of them is ever reused. Not one of the five operator return
    files; internal bookkeeping only, alongside ``.artifacts/``. Always overwritten (this file's only
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


def _has_matching_complete_state(state_path: Path, current_identity_sha256: str) -> bool:
    """Read-only, non-mutating full-identity evidence check for Fix O's recovery gate.

    Unlike ``resume.load_state`` (which recovers ``in_progress`` entries to ``pending`` and
    raises on a mismatch), this never mutates the state file and never raises -- a malformed,
    unreadable, absent, or non-matching state file simply answers "no", leaving the caller to
    refuse for lack of evidence rather than propagating an unrelated exception type.

    Full-identity evidence, both required: the state file's own recorded ``identity_sha256``
    equals ``current_identity_sha256`` (this hash binds every identity field, including
    device_index/poll_interval_s/nvml_enabled and the candidate artifact hashes -- Fix I -- none
    of which report.json has any home for), AND every one of its entries is ``complete`` (so this
    state genuinely produced a published report, not merely an interrupted attempt that happens
    to share an identity). The recorded ``identity_sha256`` is additionally cross-checked against
    a fresh hash of the state's own ``identity`` object, mirroring ``resume``'s own defense
    against a hand-edited or corrupted state file asserting a hash it does not actually match.
    """

    try:
        data = load_json(state_path)
    except (OSError, ValueError):
        return False
    if not isinstance(data, Mapping):
        return False
    identity_sha256 = data.get("identity_sha256")
    if not isinstance(identity_sha256, str) or identity_sha256 != current_identity_sha256:
        return False
    # Recovery adoption must not treat the pre-artifact v1 result-only state as evidence.  The
    # normal loader refuses that schema, but this deliberately read-only fast path runs earlier
    # when the identity sidecar is missing and must apply the same migration boundary.
    if data.get("schema_version") != 2:
        return False
    stored_identity = data.get("identity")
    if not isinstance(stored_identity, Mapping) or canonical_sha256(stored_identity) != identity_sha256:
        return False
    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        return False
    return all(
        isinstance(entry, Mapping)
        and entry.get("state") == "complete"
        and "result" in entry
        and "artifact_ref" in entry
        for entry in entries
    )


def _path_is_present(path: Path) -> bool:
    """Return whether a path exists, including a broken symlink or unreadable entry."""

    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        # Treat an inspection failure as present so the caller fails closed instead of creating
        # a new identity/state beside an entry it could not safely inspect.
        return True
    return True


def _establish_run_identity_up_front(
    output_dir: Path, state_path: Path, identity: Mapping[str, Any], *, replace: bool,
) -> None:
    """Validate/establish ``.run-identity.json`` before a single cell executes (Fix L), and
    guarantee the identity record can never trail a published report.json (Fix M).

    Called once ``identity`` is computed -- before a non-replace resume state is loaded or
    created, and before the executor is built or ``RunCoordinator.run()`` can commit a per-cell
    artifact bundle.  A replace invocation validates its existing state immediately before this
    function so an incompatible state cannot be discovered after the sidecar has been overwritten.
    This matters because bundles are keyed by CellKey beneath the run's artifact store: without
    this up-front check, a rejected run (caught by the old end-of-run-only Fix J guard) could
    publish evidence before it was caught, corrupting evidence that a resumed original run would
    then silently fold into a report claiming it came from the original measurement.

    Four cases, in order:

    * ``--replace``: an explicit fresh start. The caller validates an existing state path against
      this identity before entering this branch; only then is the identity record overwritten,
      and Fix N's per-file no-clobber-unless-replace logic does the same for other outputs.
    * report.json exists and its persisted identity hash already matches this invocation's: nothing
      to do -- a legitimate no-op re-run or repair pass.
    * report.json exists but the persisted identity is ABSENT: this is the Fix M recovery case --
      a run that crashed (or hit a write failure) between publishing report.json and writing its
      identity sidecar (that write order held even pre-fix, since ``_write_run_identity`` was
      always the LAST write) must not be permanently stuck demanding ``--replace``. Recoverable
      ONLY on FULL-identity evidence (Fix O): ``_has_matching_complete_state`` -- the resume state
      file for THIS ``--state`` path exists, its identity_sha256 equals
      ``canonical_sha256(identity)`` (binding every field, unlike report.json's schema-limited
      content), and every one of its cells is complete. ``_report_matches_current_run`` (the
      report-derived cross-check) is checked ADDITIONALLY as a secondary sanity net, never as the
      sole basis for adoption -- report-v2 has no home for device_index/poll_interval_s/
      nvml_enabled or the computed artifact hashes, so passing it alone proves nothing about
      those axes (Codex's repro: an NVML-enabled report, sidecar deleted, re-invoked with
      --no-nvml and a FRESH --state path -- every report-derived field still matches, since none
      of the changed fields live in report.json at all). Without matching-and-complete state
      evidence, this is refused, same as a hash mismatch -- not silently adopted.
    * report.json exists and the persisted identity hash MISMATCHES: a different run (different
      profile, matrix selection, candidate artifacts, or measurement configuration such as
      --device-index/--poll-interval-s/--no-nvml) reusing this --output-dir. Refused immediately,
      with zero side effects -- no resume state created or loaded, no executor built, no cell
      executed, and no artifact bundle committed.
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
            if _has_matching_complete_state(state_path, current_sha256):
                try:
                    published_report = load_json(json_path)
                except (OSError, ValueError) as exc:
                    raise DriverFailure(
                        "report_identity_mismatch",
                        f"{json_path} exists but could not be read to recover its missing "
                        f"{_run_identity_path(output_dir)}: {exc}; rerun with --replace to "
                        "overwrite, or use a different --output-dir",
                    ) from exc
                # Secondary sanity net only (Fix O) -- the state-file evidence above is what
                # actually gates adoption; this additionally catches a report.json edited or
                # replaced beneath a state file that otherwise matches.
                if _report_matches_current_run(published_report, identity):
                    # Do not adopt/write the sidecar from this shallow preflight.  The state
                    # shape is only permission to continue; exact artifact refs and result
                    # bytes are validated by RunCoordinator after ArtifactStore opens.  The
                    # sidecar is written only after that store-backed validation succeeds.
                    return
            raise DriverFailure(
                "report_identity_mismatch",
                f"{json_path} already exists but its identity sidecar "
                f"({_run_identity_path(output_dir)}) is missing, and no resume state file at "
                f"{state_path} provides full-identity evidence that this report belongs to the "
                "current identity (a matching identity_sha256 with every cell complete); rerun "
                "with --replace to overwrite, or use a different --output-dir",
            )
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
        # Opening lazily matters for identity preflight: a rejected invocation must not even
        # create or append to the prior run's runner.log.  Accepted runs still open in append
        # mode on their first diagnostic write, preserving resume history.
        self._stream: Any | None = None

    def _ensure_stream(self) -> Any:
        if self._stream is None:
            self._stream = _open_append_0644(self.path)
        return self._stream

    def write(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        stream = self._ensure_stream()
        stream.write(f"{timestamp} {message}\n")
        stream.flush()
        try:
            os.fsync(stream.fileno())
        except OSError as exc:
            raise DriverFailure("log_write", f"cannot durably write runner.log: {exc}") from exc

    def close(self) -> None:
        if self._stream is not None:
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


def _remove_stale_optional_output(path: Path) -> None:
    """Durably, atomically delete a driver-owned optional output (nvml.csv/review.csv) whose
    canonical row set is now empty (Fix Q).

    Deletion of a single path is already atomic at the filesystem level (there is no partial-
    delete state the way there is a partial write) -- this only adds the same durability
    discipline ``_atomic_publish`` uses elsewhere in this file, an ``fsync`` of the parent
    directory once the unlink lands, and a symlink refusal matching the rest of this module's
    output-path handling.

    Only ever called from ``_write_csv_file`` when ``replace``, ``verify_and_repair`` or the
    post-identity ``remove_empty`` publication flag is set -- i.e. only when the driver is
    actively (re)publishing this output_dir's canonical state for the CURRENT identity, never
    during a plain no-clobber fresh publish. Leaving a PRIOR run's file in place here would
    misattribute its rows to the current identity: e.g. a production CUDA run's nvml.csv/review.csv
    surviving underneath a fresh synthetic CPU-only report that has no such rows at all.
    """

    try:
        if path.is_symlink():
            _fail("output_path", f"{path} must not be a symlink")
        path.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise DriverFailure("atomic_write", f"cannot remove stale {path}: {exc}") from exc
    try:
        directory_descriptor = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as exc:
        raise DriverFailure("atomic_write", f"cannot durably remove stale {path}: {exc}") from exc


def _write_csv_file(
    path: Path,
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    replace: bool,
    verify_and_repair: bool = False,
    remove_empty: bool = False,
    merge_existing_rows: Callable[[Sequence[Sequence[str]], Sequence[Sequence[str]]], list[list[str]]] | None = None,
) -> None:
    """Publish one driver-owned CSV (nvml.csv/review.csv) fresh, exactly once, atomically.

    Fix E: both files are a deterministic function of durable completed state (nvml.csv from
    each cell's exact committed evidence, review.csv from the completed results plus corpus metadata), assembled
    only after every plan cell is complete -- never incrementally appended to during execution,
    which could otherwise duplicate a cell's rows if it were retried after a resume. Writes
    nothing (and creates no file) when ``rows`` is empty, matching the existing contract that
    these files exist only when there is something to report (e.g. no nvml.csv for a CPU-only
    run). No-clobber unless ``replace``, mirroring report.json/report.csv. Publication itself is
    atomic (see :func:`_atomic_publish`): an interrupted write can never leave a truncated file
    at the final path.

    Fix Q: an empty ``rows`` under ``replace`` or ``verify_and_repair`` additionally REMOVES any
    file already at ``path`` (see ``_remove_stale_optional_output``).  The evidence regeneration
    path also sets ``remove_empty`` after identity ownership is established, including on a fresh
    non-replace publish.  Those modes mean the driver is actively republishing this output_dir's
    canonical state for the current identity, so a PRIOR run's file (e.g. a production CUDA
    run's nvml.csv/review.csv surviving underneath a synthetic CPU-only report with no rows at
    all) must not be left behind to be misread as belonging to the current report.  A direct plain
    fresh, non-replace, non-repair publish with empty rows still does nothing at all.

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

    if path.is_symlink():
        # Check before reading the existing bytes: a symlink to an already-canonical target must
        # still be rejected rather than treated as a harmless no-op during verify/repair.
        _fail("output_path", f"{path} must not be a symlink")
    if not rows:
        if replace or verify_and_repair or remove_empty:
            _remove_stale_optional_output(path)
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


def _regenerate_public_evidence_outputs(
    store: ArtifactStore,
    output_dir: Path,
    corpus: Mapping[str, Any],
    plan: MatrixPlan,
    completed: Sequence[Mapping[str, Any]],
    *,
    replace: bool,
    nvml_enabled: bool,
    review_enabled: bool = True,
    verify_and_repair: bool = False,
    require_nvml_stages: bool = False,
) -> None:
    """Assemble public evidence CSVs from the exact committed refs in complete state.

    Each record's exact immutable generation is revalidated and read; the current pointer is
    never followed on behalf of a persisted older ref.  Cell order is the MatrixPlan order, and
    every row in each exact committed evidence file is preserved verbatim; abandoned retry
    generations are absent because they are not in complete state.  Review paths point directly
    at the immutable artifact
    generation's preview directory; no mutable preview tree is copied or overwritten.
    """

    if len(completed) != len(plan.cells):
        raise ArtifactStoreFailure(
            "completed_count",
            f"completed records contain {len(completed)} cells but the plan contains {len(plan.cells)}",
        )
    for index, (cell, record) in enumerate(zip(plan.cells, completed)):
        if not isinstance(record, Mapping):
            raise ArtifactStoreFailure("completed_shape", f"completed record {index} must be a mapping")
        result = record.get("result")
        if not isinstance(result, Mapping):
            raise ArtifactStoreFailure("result_shape", f"completed record {index} has no result mapping")
        expected_identity = _base_result_fields(cell)
        for field, expected in expected_identity.items():
            if result.get(field) != expected:
                raise ArtifactStoreFailure(
                    "cell_mismatch",
                    f"completed record {index} result.{field} does not match {cell!r}",
                )

    shots = _shot_index(corpus)
    nvml_rows: list[list[str]] = []
    review_rows: list[list[str]] = []
    for cell, record in zip(plan.cells, completed):
        result = record["result"]
        artifact_ref = record.get("artifact_ref")
        if not isinstance(artifact_ref, Mapping):
            raise ArtifactStoreFailure("artifact_ref_shape", f"completed record for {cell!r} has no exact artifact_ref")
        _validate_committed_ref(store, cell, result, artifact_ref)
        manifest = store.load_ref(artifact_ref)
        named_paths = {entry["path"] for entry in manifest["artifacts"]}
        evidence_path = "evidence/nvml_rows.json"
        if evidence_path in named_paths:
            if not (nvml_enabled and result.get("status") == "pass" and result.get("provider") == "cuda"):
                raise ArtifactStoreFailure(
                    "nvml_unexpected",
                    f"completed cell {cell!r} carries NVML evidence while NVML is not required",
                )
            decoded_rows = _decode_nvml_rows(
                store.read_artifact(artifact_ref, evidence_path),
                expected_identity=_base_result_fields(cell),
                required_stages=(
                    _REQUIRED_FINAL_NVML_STAGES
                    if require_nvml_stages and result.get("status") == "pass" and result.get("provider") == "cuda"
                    else None
                ),
            )
            if not decoded_rows:
                raise ArtifactStoreFailure(
                    "nvml_missing", f"passing CUDA cell {cell!r} has empty committed NVML evidence",
                )
            nvml_rows.extend(decoded_rows)
        elif nvml_enabled and result.get("status") == "pass" and result.get("provider") == "cuda":
            raise ArtifactStoreFailure(
                "nvml_missing",
                f"passing CUDA cell {cell!r} has no committed {evidence_path}",
            )

        result = record["result"]
        if result.get("status") != "pass" or not review_enabled:
            continue
        shot_entry = shots.get(cell.shot)
        if shot_entry is None or _shot_has_analytic_truth(shot_entry[1]):
            continue
        category = (shot_entry[1].get("categories") or [""])[0]
        preview_path = ""
        warped_path = "previews/offset_1_warped.pfm"
        flow_path = "previews/offset_1_flow.pfm"
        if warped_path in named_paths and flow_path in named_paths:
            # read_artifact revalidates the exact bytes before exposing the immutable directory
            # path, so a tampered or swapped generation fails publication rather than creating a
            # review row that points at unverified evidence.
            store.read_artifact(artifact_ref, warped_path)
            store.read_artifact(artifact_ref, flow_path)
            preview_path = str(store.artifact_path(artifact_ref, warped_path).parent)
        review_rows.append([
            _review_label(cell.shot, cell.candidate), cell.shot, category,
            cell.conditioning, cell.cap, cell.provider, cell.host_load,
            preview_path, "", "", "", "", "", "",
        ])
    _write_csv_file(
        output_dir / "nvml.csv", NVML_CSV_HEADER, nvml_rows,
        replace=replace, verify_and_repair=verify_and_repair, remove_empty=True,
    )
    _write_csv_file(
        output_dir / "review.csv", REVIEW_CSV_HEADER, review_rows, replace=replace,
        verify_and_repair=verify_and_repair, remove_empty=True, merge_existing_rows=_merge_review_rows,
    )


def _publish_canonical_bytes(path: Path, payload: bytes) -> None:
    """Crash-atomically ensure ``path`` holds exactly ``payload``.

    For the two driver-owned outputs with NO human-edited columns (``summary.txt`` and
    ``report.csv``): unlike ``review.csv`` (whose human review columns forbid byte-exact
    replacement -- see ``_merge_review_rows``), these are a pure deterministic function of the
    published report, so exact-byte publication is always correct. Idempotent: if the file
    already equals the canonical bytes it is left byte-identical and its inode untouched (not
    even reopened); otherwise the canonical bytes are published via :func:`_atomic_publish`
    (staged, fsynced, then linked/replaced into place) so an interrupted write can never leave a
    truncated file at the final path -- the gap the earlier direct ``path.write_bytes`` left, and
    which the report-reuse branch never repaired at all.
    """

    if path.is_symlink():
        _fail("output_path", f"{path} must not be a symlink")
    try:
        existing: bytes | None = path.read_bytes()
    except FileNotFoundError:
        existing = None
    except OSError:
        existing = None  # unreadable for some other reason -- republish below
    if existing == payload:
        return
    _atomic_publish(path, payload, replace_existing=True)


def _render_summary_bytes(report: Mapping[str, Any], plan: MatrixPlan) -> bytes:
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

    return ("\n".join(lines) + "\n").encode("utf-8")


# --------------------------------------------------------------------------------------------
# Orchestration.
# --------------------------------------------------------------------------------------------


@dataclass
class RunConfig:
    """Validated driver inputs plus explicit dependency seams used only by tests.

    The runtime/array/NVML/EXR and callback fields below are test-only injection seams.  The
    production CLI does not expose arbitrary module or callable paths; ``main`` always wires
    the checked-in production implementations.  They are intentionally absent from RunSpec
    identity because they are execution harness dependencies, not user-selectable run inputs.
    """

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
    # Test-only injection seams; production CLI always supplies trusted implementations.
    runtime_module: Any = field(metadata={"test_only_injection": True})
    array_module: Any = field(metadata={"test_only_injection": True})
    nvml_backend_factory: Callable[[], NvmlBackend] | None = field(metadata={"test_only_injection": True})
    exr_decoder: Callable[..., dict[str, Any]] = field(metadata={"test_only_injection": True})
    host_load_checkpoint: HostLoadCheckpoint = field(
        default=interactive_host_load_checkpoint, metadata={"test_only_injection": True},
    )
    cuda_measurement_runner: CudaMeasurementRunner = field(
        default=run_cuda_measurement_in_subprocess, metadata={"test_only_injection": True},
    )

    def __post_init__(self) -> None:
        # Normalize the execution-facing config before matrix work, identity formation, or any
        # executor can observe it.  _build_run_spec repeats this at its narrower boundary for
        # direct callers and compatibility tests.
        self.chain_offsets = _canonical_chain_offsets(self.chain_offsets)


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


def _publish_run_outputs(
    *,
    artifact_store: ArtifactStore,
    config: RunConfig,
    identity: Mapping[str, Any],
    protocol: Mapping[str, Any],
    corpus: Mapping[str, Any],
    plan: MatrixPlan,
    profile: str,
    environment: str,
    hardware: Mapping[str, str],
    runner_section: Mapping[str, Any],
    runner_log: RunnerLog,
    completed: Sequence[Mapping[str, Any]],
    completed_with_refs: Sequence[Mapping[str, Any]],
) -> RunResult:
    """Publish reports and public evidence while the exact-ref store remains open."""

    json_path = config.output_dir / "report.json"
    csv_path = config.output_dir / "report.csv"
    summary_path = config.output_dir / "summary.txt"

    if json_path.exists() and not config.replace:
        report = load_json(json_path)
        # A report.json reuse is authorized only by the report itself: first validate its full
        # schema/protocol/corpus semantics, then build the report the current normalized inputs
        # and exact completed records would produce, and compare the explicit canonical semantic
        # projection.  The narrower identity-shaped check remains an up-front missing-sidecar
        # gate, but it must never authorize derivative repair.
        _validate_existing_report(
            report,
            json_path=json_path,
            protocol=protocol,
            report_schema=config.report_schema,
            corpus=corpus,
            corpus_schema=config.corpus_schema,
        )
        expected_metadata = _build_report_metadata(
            config.report_metadata, corpus, environment, profile, hardware, runner_section,
        )
        try:
            expected_report = assemble_report(
                protocol, corpus, config.report_schema, config.corpus_schema,
                expected_metadata, config.candidate_entries, plan, completed,
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            _fail(
                "report_identity_mismatch",
                f"could not assemble the current report semantics for {json_path}; "
                f"refusing derivative repair: {exc}",
            )
        _compare_reusable_report_semantics(
            report,
            expected_report,
            json_path=json_path,
        )
        runner_log.write("report.json already published by an earlier invocation under the same identity; not regenerating")
        _publish_canonical_bytes(summary_path, _render_summary_bytes(report, plan))
        _publish_canonical_bytes(csv_path, render_csv(report))
        _regenerate_public_evidence_outputs(
            artifact_store,
            config.output_dir,
            corpus,
            plan,
            completed_with_refs,
            replace=config.replace,
            nvml_enabled=config.nvml_backend_factory is not None,
            verify_and_repair=True,
            require_nvml_stages=profile == "final",
        )
    else:
        report_metadata = _build_report_metadata(
            config.report_metadata, corpus, environment, profile, hardware, runner_section,
        )
        report = assemble_report(
            protocol, corpus, config.report_schema, config.corpus_schema,
            report_metadata, config.candidate_entries, plan, completed,
        )
        write_report_pair(
            json_path, csv_path, report, protocol, config.report_schema, corpus, config.corpus_schema,
            replace=config.replace,
        )
        _publish_canonical_bytes(summary_path, _render_summary_bytes(report, plan))
        _regenerate_public_evidence_outputs(
            artifact_store,
            config.output_dir,
            corpus,
            plan,
            completed_with_refs,
            replace=config.replace,
            nvml_enabled=config.nvml_backend_factory is not None,
            require_nvml_stages=profile == "final",
        )
        runner_log.write("run complete; report.json/summary.txt/nvml.csv/review.csv published")

    output_paths = {
        "report.json": json_path,
        "report.csv": csv_path,
        "summary.txt": summary_path,
        "runner.log": runner_log.path,
    }
    nvml_path = config.output_dir / "nvml.csv"
    if nvml_path.exists():
        output_paths["nvml.csv"] = nvml_path
    review_path = config.output_dir / "review.csv"
    if review_path.exists():
        output_paths["review.csv"] = review_path
    return RunResult(report=report, output_paths=output_paths, incomplete=False)


def _run_bakeoff(config: RunConfig, runner_log: RunnerLog) -> RunResult:
    protocol = config.protocol
    corpus = config.corpus
    selection = config.selection
    profile = selection["profile"]
    environment = selection["environment"]

    # Metadata and config preflight is deliberately before matrix execution, artifact loading,
    # state creation, and executor construction.  A malformed/ambiguous report input must never
    # get as far as a cell, and the normalized values below are reused for both identity and
    # publication.
    # Corpus consistency used to run only during final report assembly. A schema-valid but
    # protocol-incomplete operator template could therefore finish every expensive cell and then
    # fail publication. Run the exact report-side corpus gate before planning or inference.
    try:
        validate_corpus_consistency(corpus, protocol, config.corpus_schema)
    except ValidationError as exc:
        _fail("corpus_invalid", str(exc))

    if not isinstance(config.report_metadata, Mapping):
        _fail("report_metadata_shape", "report_metadata must be an object")
    chain_offsets = _canonical_chain_offsets(config.chain_offsets)
    config.chain_offsets = chain_offsets
    hardware = _normalise_hardware(config.report_metadata.get("hardware", {}))
    runner_section = _normalise_runner_metadata(config.report_metadata.get("runner", {}))
    report_inputs = _stable_report_inputs(config.report_metadata)

    try:
        plan = build_matrix(
            protocol, corpus, config.candidate_entries, _selection_axes(selection), profile, environment,
        )
    except MatrixFailure:
        # No identity has been established yet, so do not create/append runner.log for this
        # rejected invocation.  Accepted runs log the planned matrix immediately after identity
        # preflight below.
        raise

    if profile == "final" and config.nvml_backend_factory is None and any(
        cell.provider == "cuda" for cell in plan.cells
    ):
        _fail(
            "nvml_required",
            "the final profile includes CUDA cells and requires NVML sampling; remove --no-nvml",
        )

    artifacts = _validate_selected_artifacts(
        plan, config.artifact_map, config.protocol_path,
    )
    run_spec = _build_run_spec(
        protocol, corpus, plan, environment, profile, artifacts, runner_section, hardware, chain_offsets,
        candidate_entries=config.candidate_entries,
        report_schema=config.report_schema, corpus_schema=config.corpus_schema,
        device_index=config.device_index, poll_interval_s=config.poll_interval_s,
        nvml_enabled=config.nvml_backend_factory is not None,
        report_inputs=report_inputs,
    )
    _assert_run_spec_identity(run_spec)
    identity = run_spec.stable_inputs

    # Fix L/M: validate/establish .run-identity.json now -- before resume state is loaded or
    # created, before the executor is built, and before a single cell can commit a CellKey-keyed
    # artifact bundle that might belong to a different, already-published run reusing this
    # --output-dir. See `_establish_run_identity_up_front`.
    # ``--replace`` is the one exception to that general ordering: an existing state path must be
    # validated against the new identity first. Otherwise the unconditional replace branch could
    # overwrite the sidecar, then discover an incompatible state and leave the output directory
    # internally claiming the rejected run identity.
    prevalidated_state: dict[str, Any] | None = None
    if config.replace and _path_is_present(config.state_path):
        try:
            prevalidated_state = load_state(config.state_path, identity, plan)
        except ResumeFailure as exc:
            if exc.kind == "schema_version":
                raise DriverFailure(
                    "legacy_resume_state",
                    f"{config.state_path} is a legacy/incompatible resume state and cannot be migrated; "
                    "use a fresh --state path or replace the old state before using --replace",
                ) from exc
            raise

    _establish_run_identity_up_front(config.output_dir, config.state_path, identity, replace=config.replace)

    # The logger is intentionally first used only after identity preflight succeeds.  Its lazy
    # open means an identity-rejected invocation cannot append even the initial `run invoked` line
    # to the prior run's log.
    runner_log.write("run invoked")
    runner_log.write(f"matrix planned: {len(plan.cells)} cells, matrix_sha256={plan.matrix_sha256}")
    runner_log.write(f"validated artifacts for candidates: {sorted(artifacts)}")

    if prevalidated_state is not None:
        state = prevalidated_state
        runner_log.write("resumed existing state")
    elif config.state_path.exists():
        try:
            state = load_state(config.state_path, identity, plan)
        except ResumeFailure as exc:
            if exc.kind == "schema_version":
                raise DriverFailure(
                    "legacy_resume_state",
                    f"{config.state_path} is a legacy/incompatible resume state and cannot be migrated; "
                    "use a fresh --state path or rerun with --replace after replacing the old state",
                ) from exc
            raise
        runner_log.write("resumed existing state")
    else:
        state = create_state(config.state_path, identity, plan)
        runner_log.write("created new state")

    # Result artifacts and optional previews are one immutable bundle per exact cell generation.
    # Keep the store open through report/CSV publication so every public evidence read is backed
    # by the same validated exact refs and the writer lock spans the complete publication window.
    artifact_parent = _artifact_store_parent(config.output_dir)
    with ArtifactStore(artifact_parent / ".artifacts", run_spec.stable_inputs) as artifact_store:
        if artifact_store.identity_sha256 != run_spec.identity_sha256:
            _fail(
                "identity_invalid",
                "ArtifactStore identity_sha256 does not match the RunSpec identity_sha256",
            )
        executor, _exec_ctx = build_executor(
            protocol=protocol,
            corpus=corpus,
            profile=profile,
            artifacts=artifacts,
            runtime_module=config.runtime_module,
            array_module=config.array_module,
            nvml_backend_factory=config.nvml_backend_factory,
            device_index=config.device_index,
            poll_interval_s=config.poll_interval_s,
            chain_offsets=chain_offsets,
            exr_decoder=config.exr_decoder,
            review_enabled=True,
            host_load_checkpoint=config.host_load_checkpoint,
            cuda_measurement_runner=config.cuda_measurement_runner,
        )

        def committed_executor(cell: CellKey) -> CommittedExecution:
            bundle = executor(cell)
            for message in bundle.log_messages:
                runner_log.write(message)
            try:
                execution = _commit_cell_bundle(
                    artifact_store,
                    cell,
                    bundle,
                    nvml_enabled=config.nvml_backend_factory is not None,
                    require_nvml_stages=profile == "final",
                )
            except Exception as exc:
                kind = getattr(exc, "kind", type(exc).__name__)
                runner_log.write(
                    f"cell persistence failed {json.dumps(cell.as_dict(), sort_keys=True)} kind={kind}"
                )
                raise
            status = bundle.result.get("status")
            if status == "pass":
                runner_log.write(
                    f"cell artifact committed status=pass {json.dumps(cell.as_dict(), sort_keys=True)}"
                )
            elif status == "fail":
                failure = bundle.result.get("failure")
                reason = failure.get("type", "unknown") if isinstance(failure, Mapping) else "unknown"
                runner_log.write(
                    f"cell artifact committed status=fail {json.dumps(cell.as_dict(), sort_keys=True)} reason={reason}"
                )
            elif status == "skip":
                failure = bundle.result.get("failure")
                reason = failure.get("type", "unknown") if isinstance(failure, Mapping) else "unknown"
                runner_log.write(
                    f"cell artifact committed status=skip {json.dumps(cell.as_dict(), sort_keys=True)} reason={reason}"
                )
            return execution

        coordinator = RunCoordinator(
            config.state_path,
            identity,
            plan,
            committed_executor,
            lambda cell, result, artifact_ref: _validate_committed_ref(
                artifact_store, cell, result, artifact_ref,
            ),
        )
        coordinator.run()

        try:
            completed_with_refs = coordinator.completed_records_with_refs()
        except IncompleteFailure as exc:
            runner_log.write(f"run interrupted, not all cells complete: {exc}")
            return RunResult(report=None, output_paths={}, incomplete=True)

        completed = [
            {key: value for key, value in record.items() if key != "artifact_ref"}
            for record in completed_with_refs
        ]

        published = _publish_run_outputs(
            artifact_store=artifact_store,
            config=config,
            identity=identity,
            protocol=protocol,
            corpus=corpus,
            plan=plan,
            profile=profile,
            environment=environment,
            hardware=hardware,
            runner_section=runner_section,
            runner_log=runner_log,
            completed=completed,
            completed_with_refs=completed_with_refs,
        )

        # A report whose identity sidecar was missing is recoverable only after the complete
        # report semantic reuse check and public exact-ref evidence publication above both
        # succeed.  A tampered report therefore fails before this write and cannot be blessed by
        # a shallow state-shape check; the derivative files likewise remain untouched.
        if _read_run_identity_sha256(config.output_dir) != canonical_sha256(identity):
            _write_run_identity(config.output_dir, identity)
        return published

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
    # These maps were normalized during preflight.  Filter again at the publication boundary so
    # a direct/internal caller cannot accidentally emit a property outside report-v2, and always
    # regenerate the volatile command rather than preserving a caller-supplied value.
    runner = {
        key: runner_section[key]
        for key in _RUNNER_REPORT_KEYS
        if key != "command" and key in runner_section
    }
    runner["command"] = " ".join(sys.argv)

    hardware_out = {
        key: hardware[key]
        for key in _HARDWARE_REPORT_KEYS
        if key in hardware and hardware[key] not in (None, "")
    }

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
    except (DriverFailure, MatrixFailure, ResumeFailure, CoordinatorFailure, ArtifactStoreFailure, ReportFailure) as exc:
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
    "CellBundle",
    "DriverFailure",
    "HostLoadCheckpoint",
    "PreviewPayload",
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
