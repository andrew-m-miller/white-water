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
import os
import platform as platform_module
import struct
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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
    from .nvml import NvmlBackend, NvmlFailure, NvmlSampler, PynvmlBackend, write_or_append_nvml_csv
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
    from nvml import NvmlBackend, NvmlFailure, NvmlSampler, PynvmlBackend, write_or_append_nvml_csv  # type: ignore
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
# Review label and preview.
# --------------------------------------------------------------------------------------------


def _review_label(shot_id: str, candidate_id: str) -> str:
    """A deterministic pseudonymous per-shot candidate label for a blind review pass."""

    digest = hashlib.sha256(f"{shot_id}:{candidate_id}".encode("utf-8")).hexdigest()
    return f"candidate-{digest[:12]}"


def _write_review_preview(
    review_dir: Path,
    cell: CellKey,
    first_grid: Sequence[Sequence[tuple[float, ...]]],
    second_grid: Sequence[Sequence[tuple[float, ...]]],
    width: int,
    height: int,
) -> str:
    """Write two small analysis-resolution PFM previews that never leave the box."""

    destination = review_dir / cell.shot / _review_label(cell.shot, cell.candidate) / cell.cap / cell.conditioning
    destination.mkdir(parents=True, exist_ok=True)
    synthetic_write_pfm(destination / "first.pfm", first_grid, width, height)
    synthetic_write_pfm(destination / "second.pfm", second_grid, width, height)
    return str(destination)


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


def _run_cell(cell: CellKey, ctx: _RunnerContext) -> dict[str, Any]:
    artifact = ctx.artifacts.get(cell.candidate)
    if artifact is None:
        raise _CellFail(_failure("artifact_missing", f"candidate {cell.candidate!r} was not validated at startup"))
    shot_ctx = ctx.shot_contexts.get(cell.shot)
    if shot_ctx is None:
        raise _CellFail(_failure("missing_input", f"shot {cell.shot!r} is absent from the corpus"))
    if cell.provider not in PROVIDER_EXECUTION_NAMES:
        raise _CellFail(_failure("provider_unavailable", f"unknown provider token: {cell.provider!r}"))

    cap_megapixels = _cap_megapixels(ctx.protocol, cell.cap)
    evaluator_instance = Evaluator(artifact, ctx.runtime_module, ctx.array_module)

    sampler: NvmlSampler | None = None
    if cell.provider == "cuda" and ctx.nvml_backend_factory is not None:
        try:
            backend = ctx.nvml_backend_factory()
            sampler = NvmlSampler(backend, ctx.device_index, poll_interval_s=ctx.poll_interval_s)
            sampler.sample("baseline")
        except NvmlFailure as exc:
            raise _CellFail(_failure("runtime_error", f"NVML setup failed: {exc}", stage="baseline")) from exc

    def stage_sampler(name: str) -> Any:
        assert sampler is not None
        return sampler.poll(name)

    reference_frame = int(shot_ctx.shot["reference_frame"])
    try:
        first, second = _load_pair(shot_ctx, reference_frame, reference_frame + 1, exr_decoder=ctx.exr_decoder)
    except ExrFailure as exc:
        raise _CellFail(_exr_failure(exc, stage="load_input")) from exc

    try:
        (
            base_result, geometry, cond_meta, first_nchw, second_nchw,
        ) = _infer_pair(
            evaluator_instance, artifact, first, second,
            provider=cell.provider,
            conditioning_token=cell.conditioning,
            cap_megapixels=cap_megapixels,
            array_module=ctx.array_module,
            profile=ctx.profile,
            stage_sampler=stage_sampler if sampler is not None else None,
        )
    except DependencyFailure as exc:
        raise _CellFail(_failure("runtime_error", str(exc), stage="inference", retryable=True)) from exc
    except EvaluatorFailure as exc:
        raise _CellFail(_failure(exc.kind if exc.kind in _RESULT_FAILURE_TYPES else "runtime_error", str(exc), stage="inference")) from exc

    if sampler is not None:
        sampler.sample("cleanup")

    predicted_flow = base_result["flow"]
    analysis_width = geometry["analysis_width"]
    analysis_height = geometry["analysis_height"]
    spacing_x = geometry["spacing_x_source_pixels"]
    spacing_y = geometry["spacing_y_source_pixels"]

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

    # Chain drift: mandatory once the shot declares a chain_length.
    chain_length = shot_ctx.chain_length
    if chain_length is not None:
        if chain_length not in ctx.chain_offsets and chain_length not in (1, 2, 4, 8):
            raise _CellFail(_failure("input_invalid", f"unsupported chain_length {chain_length}", stage="metrics"))
        flows = [predicted_flow]
        try:
            for link in range(1, chain_length):
                link_first, link_second = _load_pair(
                    shot_ctx, reference_frame + link, reference_frame + link + 1, exr_decoder=ctx.exr_decoder,
                )
                link_result, _, _, _, _ = _infer_pair(
                    evaluator_instance, artifact, link_first, link_second,
                    provider=cell.provider, conditioning_token=cell.conditioning,
                    cap_megapixels=cap_megapixels, array_module=ctx.array_module,
                    profile="smoke", stage_sampler=None,
                )
                flows.append(link_result["flow"])
            if shot_ctx.synthetic_case is not None:
                truth_grid, valid_mask = _dense_truth_and_mask(
                    shot_ctx.synthetic_case, reference_frame, reference_frame + chain_length,
                    analysis_width, analysis_height, spacing_x, spacing_y,
                )
            else:
                truth_grid, valid_mask = None, None
            if truth_grid is not None:
                chain_value = metrics_module.chain_drift_px(flows, truth_grid, valid_mask, chain_length)
                result_metrics["chain_drift_px"] = chain_value
                not_applicable.discard("chain_drift_px")
        except (ExrFailure, EvaluatorFailure, DependencyFailure, MetricFailure) as exc:
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

    # Forward/backward residual: optional, needs a reverse-direction inference.
    try:
        reverse_first, reverse_second = _load_pair(
            shot_ctx, reference_frame + 1, reference_frame, exr_decoder=ctx.exr_decoder,
        )
        reverse_result, _, _, _, _ = _infer_pair(
            evaluator_instance, artifact, reverse_first, reverse_second,
            provider=cell.provider, conditioning_token=cell.conditioning,
            cap_megapixels=cap_megapixels, array_module=ctx.array_module,
            profile="smoke", stage_sampler=None,
        )
        residual_mask = [[True] * analysis_width for _ in range(analysis_height)]
        residual = metrics_module.forward_backward_residual_px(predicted_flow, reverse_result["flow"], residual_mask)
        result_metrics["forward_backward_residual_px"] = residual
        not_applicable.discard("forward_backward_residual_px")
    except (ExrFailure, EvaluatorFailure, DependencyFailure, MetricFailure) as exc:
        ctx.log(f"forward_backward_residual_px skipped for {cell.as_dict()}: {exc}")

    result_metrics["not_applicable"] = sorted(not_applicable)

    if sampler is not None:
        sampler.sample("process_exit")
        resource = sampler.resource()
        # Written immediately, per cell, rather than batched at end-of-run: a cell already
        # durably marked complete in resume state must not lose its NVML rows if the process is
        # killed before a later batched flush would have run.
        write_or_append_nvml_csv(ctx.output_dir / "nvml.csv", sampler.csv_rows(_base_result_fields(cell)))
    else:
        resource = {"peak_incremental_device_memory_gib": 0.0}

    # Anonymous local review for shots without trustworthy automated truth. Also written
    # immediately per cell for the same resumability reason as nvml.csv above.
    if not shot_ctx.has_analytic_truth and ctx.review_dir is not None:
        preview_path = _write_review_preview(
            ctx.review_dir, cell, image1, image2, analysis_width, analysis_height,
        )
        _append_csv_rows(ctx.output_dir / "review.csv", REVIEW_CSV_HEADER, [[
            _review_label(cell.shot, cell.candidate), cell.shot,
            (shot_ctx.shot.get("categories") or [""])[0], cell.conditioning, cell.cap,
            cell.provider, cell.host_load, preview_path, "", "", "", "", "", "",
        ]])

    input_frames = [
        {"frame": first["frame"], "sha256": first["sha256"]},
        {"frame": second["frame"], "sha256": second["sha256"]},
    ]
    category = (shot_ctx.shot.get("categories") or [None])[0]
    passing: dict[str, Any] = {
        "input_frames": input_frames,
        "geometry": geometry,
        "timing": base_result["timing"],
        "metrics": result_metrics,
        "resource": resource,
        "environment": base_result["environment"],
        "conditioning_parameters": cond_meta.get("conditioning_parameters") or {},
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


def _compute_identity(
    protocol: Mapping[str, Any],
    corpus: Mapping[str, Any],
    plan: MatrixPlan,
    environment: str,
    artifacts: Mapping[str, ValidatedArtifact],
    runner_metadata: Mapping[str, Any],
    hardware: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "protocol_id": protocol["protocol_id"],
        "matrix_sha256": plan.matrix_sha256,
        "corpus_sha256": canonical_sha256(corpus),
        "environment": environment,
        "providers": plan.selector["providers"],
        "candidates": {
            candidate_id: {
                "manifest_sha256": artifacts[candidate_id].manifest_sha256,
                "artifact_sha256": artifacts[candidate_id].artifact_sha256,
                "platform": artifacts[candidate_id].platform,
            }
            for candidate_id in plan.selector["candidate_ids"]
        },
        "runtime": {
            "runtime": runner_metadata.get("runtime", ""),
            "runtime_sha256": runner_metadata.get("runtime_sha256", ""),
        },
        "hardware": {
            "platform": hardware.get("platform", ""),
            "architecture": hardware.get("architecture", ""),
        },
    }


# --------------------------------------------------------------------------------------------
# Output files.
# --------------------------------------------------------------------------------------------


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def _append_csv_rows(path: Path, header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    """Append driver-owned CSV rows (nvml.csv/review.csv), writing the header exactly once.

    Simpler than :mod:`tools.bakeoff.reporting`'s/``nvml``'s hard-linked atomic publication:
    this file is appended to incrementally, once per completed cell, by a single sequential
    driver process, so the stronger multi-writer guarantees those modules provide are not
    needed here. Mode 0644 and append-only are still honored.
    """

    if not rows:
        return
    is_new = not path.exists()
    stream = _open_append_0644(path)
    try:
        writer = csv.writer(stream, delimiter=",", quotechar='"', lineterminator="\n")
        if is_new:
            writer.writerow(header)
        for row in rows:
            writer.writerow(row)
    finally:
        stream.close()


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

    identity = _compute_identity(protocol, corpus, plan, environment, artifacts, runner_section, hardware)

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
        # a true no-op that returns exactly what was already published.
        report = load_json(json_path)
        runner_log.write("report.json already published by an earlier invocation; not regenerating")
        if not summary_path.exists():
            _write_summary_txt(summary_path, report, plan)
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
        runner_log.write("run complete; report.json/summary.txt published")

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
    runner = dict(runner_section)
    runner.setdefault("name", "ww-bakeoff")
    runner.setdefault("version", "0.1.0")
    runner.setdefault("source_commit", _default_source_commit() or "0" * 40)
    for required in ("evaluator_sha256", "runtime", "runtime_sha256"):
        if required not in runner:
            _fail("report_metadata_missing", f"report_metadata.runner.{required} is required")
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
    "DriverFailure",
    "RunConfig",
    "RunResult",
    "build_executor",
    "main",
    "run_bakeoff",
]


if __name__ == "__main__":
    raise SystemExit(main())
