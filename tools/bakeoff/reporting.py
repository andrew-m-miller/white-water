#!/usr/bin/env python3
"""Dependency-free Phase 2.5 report assembly and deterministic publication."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping, Sequence

from .matrix import CellKey, MatrixPlan
from .validator import ValidationError, canonical_sha256, validate_report_consistency


class ReportFailure(ValueError):
    """Stable, reportable report assembly/publication failure."""

    def __init__(self, kind: str, message: str):
        self.kind = kind
        self.reason = kind
        self.failure_type = "report_failure"
        self.message = message
        super().__init__(f"{kind}: {message}")


CSV_HEADER = (
    "candidate_id", "shot_id", "conditioning_token", "cap_token", "provider", "host_load", "status",
    "failure_type", "failure_message", "failure_retryable", "failure_stage",
    "category", "source_target",
    "source_width", "source_height", "source_pixel_aspect_ratio", "canonical_width", "canonical_height",
    "analysis_width", "analysis_height", "padded_width", "padded_height", "effective_padded_megapixels",
    "spacing_x_source_pixels", "spacing_y_source_pixels",
    "preprocessing_ms", "session_creation_ms", "first_inference_ms", "steady_inference_ms",
    "postprocessing_ms", "total_pair_ms",
    "endpoint_error_px", "fraction_le_1px", "fraction_le_3px", "landmark_median_error_px",
    "landmark_p95_error_px", "visible_warp_residual", "forward_backward_residual_px", "chain_drift_px",
    "nonfinite_fraction", "repeated_run_p99_delta_px",
    "peak_incremental_device_memory_gib", "baseline_device_memory_mib", "peak_device_memory_mib",
    "cleanup_device_memory_mib", "process_exit_device_memory_mib",
    "runtime_version", "provider_version", "model_manifest_sha256",
    "conditioning_parameters_json", "input_frames_json", "steady_samples_ms_json", "sessions_json",
    "metrics_not_applicable_json", "nvml_samples_json",
)

_SCORE_KEYS = {"synthetic_macro_score", "production_macro_score", "final_quality_score", "category_scores"}
_IDENTITY_FIELDS = ("candidate_id", "shot_id", "conditioning_token", "cap_token", "provider", "host_load")
_GENERATED_REPORT_KEYS = {
    "schema_version", "protocol_id", "corpus_sha256", "matrix", "candidates", "results",
}


def _fail(kind: str, message: str) -> None:
    raise ReportFailure(kind, message)


def _is_json(value: Any, path: str = "$", seen: set[int] | None = None) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail("nonfinite", f"{path} contains a nonfinite number")
        return
    if isinstance(value, Mapping):
        if not isinstance(value, dict):
            _fail("json_value", f"{path} must use plain JSON objects")
        seen = seen if seen is not None else set()
        marker = id(value)
        if marker in seen:
            _fail("json_value", f"{path} contains a cycle")
        seen.add(marker)
        for key, child in value.items():
            if not isinstance(key, str):
                _fail("json_value", f"{path} contains a non-string object key")
            _is_json(child, f"{path}.{key}", seen)
        seen.remove(marker)
        return
    if isinstance(value, list):
        seen = seen if seen is not None else set()
        marker = id(value)
        if marker in seen:
            _fail("json_value", f"{path} contains a cycle")
        seen.add(marker)
        for index, child in enumerate(value):
            _is_json(child, f"{path}[{index}]", seen)
        seen.remove(marker)
        return
    _fail("json_value", f"{path} contains a non-JSON value")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("shape", f"{path} must be an object")
    _is_json(value, path)
    return value


def _cell_from_result(result: Mapping[str, Any], path: str) -> CellKey:
    try:
        values = tuple(result[field] for field in _IDENTITY_FIELDS)
    except KeyError as exc:
        _fail("result_identity", f"{path} is missing {exc.args[0]}")
    if any(not isinstance(value, str) or not value for value in values):
        _fail("result_identity", f"{path} identity fields must be non-empty strings")
    return CellKey(*values)


def _result_records(completed_results: Any) -> list[dict[str, Any]]:
    if not isinstance(completed_results, (list, tuple)):
        _fail("result_shape", "completed_results must be a sequence")
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(completed_results):
        entry = _mapping(raw, f"completed_results[{index}]")
        if "state" in entry or "cell" in entry:
            if entry.get("state") != "complete":
                _fail("incomplete_result", f"completed_results[{index}] is not complete")
            if set(entry) != {"cell", "state", "result"}:
                _fail("result_shape", f"completed_results[{index}] resume record has extra or missing keys")
            cell_mapping = _mapping(entry["cell"], f"completed_results[{index}].cell")
            if set(cell_mapping) != {"candidate", "shot", "conditioning", "cap", "provider", "host_load"}:
                _fail("result_identity", f"completed_results[{index}].cell has the wrong fields")
            cell = CellKey(*(cell_mapping[field] for field in ("candidate", "shot", "conditioning", "cap", "provider", "host_load")))
            result = dict(_mapping(entry["result"], f"completed_results[{index}].result"))
            for field, value in zip(_IDENTITY_FIELDS, (cell.candidate, cell.shot, cell.conditioning, cell.cap, cell.provider, cell.host_load)):
                report_field = field
                if report_field in result and result[report_field] != value:
                    _fail("cell_mismatch", f"completed_results[{index}] result identity disagrees with cell")
                result[report_field] = value
            records.append(result)
        else:
            records.append(dict(entry))
    return records


def _ordered_candidates(protocol: Mapping[str, Any], candidate_entries: Any) -> list[dict[str, Any]]:
    if not isinstance(candidate_entries, (list, tuple)):
        _fail("candidate_shape", "candidate_entries must be a sequence")
    entries: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(candidate_entries):
        entry = dict(_mapping(raw, f"candidate_entries[{index}]"))
        candidate_id = entry.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            _fail("candidate_shape", f"candidate_entries[{index}] has no candidate_id")
        if candidate_id in entries:
            _fail("duplicate_candidate", f"candidate {candidate_id!r} appears more than once")
        entries[candidate_id] = entry
    protocol_candidates = protocol.get("candidate_ids")
    if not isinstance(protocol_candidates, (list, tuple)):
        _fail("protocol_shape", "protocol.candidate_ids must be a sequence")
    order = [entry.get("id") for entry in protocol_candidates if isinstance(entry, Mapping)]
    unknown = [candidate_id for candidate_id in entries if candidate_id not in order]
    if unknown:
        _fail("unknown_candidate", f"candidate {unknown[0]!r} is absent from protocol")
    return [entries[candidate_id] for candidate_id in order if candidate_id in entries]


def _summary(results: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> dict[str, Any]:
    summary_metadata = metadata.get("summary", {})
    if summary_metadata is None:
        summary_metadata = {}
    if not isinstance(summary_metadata, Mapping):
        _fail("summary_shape", "report metadata summary must be an object")
    unknown = set(summary_metadata) - _SCORE_KEYS
    if unknown:
        _fail("summary_shape", f"summary contains unsupported field {sorted(unknown)[0]!r}")
    _is_json(summary_metadata, "summary")
    counts = {"pass": 0, "fail": 0, "skip": 0}
    for result in results:
        status = result.get("status")
        if status not in counts:
            _fail("result_status", f"result status {status!r} is not pass/fail/skip")
        counts[status] += 1
    return {
        "required_cells": len(results),
        "passed_cells": counts["pass"],
        "failed_cells": counts["fail"],
        "skipped_cells": counts["skip"],
        **dict(summary_metadata),
    }


def assemble_report(
    protocol: Mapping[str, Any],
    corpus: Mapping[str, Any],
    report_schema: Mapping[str, Any],
    corpus_schema: Mapping[str, Any],
    report_metadata: Mapping[str, Any],
    candidate_entries: Any,
    plan: MatrixPlan,
    completed_results: Any,
) -> dict[str, Any]:
    """Assemble, normalize, and validate one complete deterministic protocol report."""

    metadata = dict(_mapping(report_metadata, "report_metadata"))
    _is_json(metadata, "report_metadata")
    generated = sorted(_GENERATED_REPORT_KEYS.intersection(metadata))
    if generated:
        _fail("metadata_shape", f"report_metadata contains generated field {generated[0]!r}")
    if not isinstance(corpus, Mapping) or not isinstance(protocol, Mapping):
        _fail("shape", "protocol and corpus must be objects")
    try:
        corpus_hash = canonical_sha256(corpus)
    except (TypeError, ValueError) as exc:
        raise ReportFailure("json_value", f"cannot hash corpus: {exc}") from exc
    candidates = _ordered_candidates(protocol, candidate_entries)
    records = _result_records(completed_results)
    expected_cells = tuple(plan.cells)
    if len(records) != len(expected_cells):
        _fail("result_count", "completed_results must contain exactly one result per plan cell")
    by_cell: dict[CellKey, dict[str, Any]] = {}
    for index, record in enumerate(records):
        cell = _cell_from_result(record, f"completed_results[{index}]")
        if cell in by_cell:
            _fail("duplicate_result", f"duplicate result for {cell!r}")
        if cell not in expected_cells:
            _fail("extra_result", f"result cell {cell!r} is not in MatrixPlan")
        by_cell[cell] = record
    missing = [cell for cell in expected_cells if cell not in by_cell]
    if missing:
        _fail("missing_result", f"missing result for {missing[0]!r}")
    results = [by_cell[cell] for cell in expected_cells]
    for index, (cell, result) in enumerate(zip(expected_cells, results)):
        expected = cell.as_dict()
        actual = {
            "candidate": result["candidate_id"],
            "shot": result["shot_id"],
            "conditioning": result["conditioning_token"],
            "cap": result["cap_token"],
            "provider": result["provider"],
            "host_load": result["host_load"],
        }
        if actual != expected:
            _fail("cell_mismatch", f"result {index} identity does not match MatrixPlan")
    report = dict(metadata)
    report.update({
        "schema_version": protocol.get("schema_version", 1),
        "protocol_id": protocol["protocol_id"],
        "corpus_sha256": corpus_hash,
        "matrix": plan.selector,
        "candidates": candidates,
        "results": results,
        "summary": _summary(results, metadata),
    })
    try:
        validate_report_consistency(report, protocol, report_schema, corpus, corpus_schema)
    except (ValidationError, ValueError, TypeError) as exc:
        raise ReportFailure("validation", str(exc)) from exc
    return report


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ReportFailure("json_value", f"cannot encode CSV JSON column: {exc}") from exc


def _scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            _fail("nonfinite", "CSV scalar is nonfinite")
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    if isinstance(value, str):
        return value
    _fail("csv_scalar", f"CSV scalar has unsupported value {value!r}")


def _csv_nested_mapping(result: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = result.get(field, {})
    if not isinstance(value, Mapping):
        _fail("result_shape", f"result.{field} must be an object")
    return value


def _csv_row(result: Mapping[str, Any]) -> list[str]:
    _mapping(result, "result")
    failure = _csv_nested_mapping(result, "failure")
    geometry = _csv_nested_mapping(result, "geometry")
    timing = _csv_nested_mapping(result, "timing")
    metrics = _csv_nested_mapping(result, "metrics")
    resource = _csv_nested_mapping(result, "resource")
    environment = _csv_nested_mapping(result, "environment")
    row: dict[str, Any] = {
        **{field: result.get(field) for field in _IDENTITY_FIELDS},
        "status": result.get("status"),
        "failure_type": failure.get("type"),
        "failure_message": failure.get("message"),
        "failure_retryable": failure.get("retryable"),
        "failure_stage": failure.get("stage"),
        "category": result.get("category"), "source_target": result.get("source_target"),
    }
    for field in (
        "source_width", "source_height", "source_pixel_aspect_ratio", "canonical_width", "canonical_height",
        "analysis_width", "analysis_height", "padded_width", "padded_height", "effective_padded_megapixels",
        "spacing_x_source_pixels", "spacing_y_source_pixels",
    ):
        row[field] = geometry.get(field)
    for field in (
        "preprocessing_ms", "session_creation_ms", "first_inference_ms", "steady_inference_ms",
        "postprocessing_ms", "total_pair_ms",
    ):
        row[field] = timing.get(field)
    for field in (
        "endpoint_error_px", "fraction_le_1px", "fraction_le_3px", "landmark_median_error_px",
        "landmark_p95_error_px", "visible_warp_residual", "forward_backward_residual_px", "chain_drift_px",
        "nonfinite_fraction", "repeated_run_p99_delta_px",
    ):
        row[field] = metrics.get(field)
    for field in (
        "peak_incremental_device_memory_gib", "baseline_device_memory_mib", "peak_device_memory_mib",
        "cleanup_device_memory_mib", "process_exit_device_memory_mib",
    ):
        row[field] = resource.get(field)
    row["runtime_version"] = environment.get("runtime_version")
    row["provider_version"] = environment.get("provider_version")
    row["model_manifest_sha256"] = environment.get("model_manifest_sha256")
    row["conditioning_parameters_json"] = _canonical_json(result["conditioning_parameters"]) if "conditioning_parameters" in result else ""
    row["input_frames_json"] = _canonical_json(result["input_frames"]) if "input_frames" in result else ""
    row["steady_samples_ms_json"] = _canonical_json(timing["steady_samples_ms"]) if "steady_samples_ms" in timing else ""
    row["sessions_json"] = _canonical_json(timing["sessions"]) if "sessions" in timing else ""
    row["metrics_not_applicable_json"] = _canonical_json(metrics["not_applicable"]) if "not_applicable" in metrics else ""
    row["nvml_samples_json"] = _canonical_json(resource["nvml_samples"]) if "nvml_samples" in resource else ""
    return [_scalar(row[field]) for field in CSV_HEADER]


def render_csv(report: Mapping[str, Any]) -> bytes:
    """Render report results with the frozen header and stable CSV quoting."""

    report_mapping = _mapping(report, "report")
    results = report_mapping.get("results")
    if not isinstance(results, (list, tuple)):
        _fail("result_shape", "report.results must be a sequence")
    from io import StringIO
    output = StringIO(newline="")
    writer = csv.writer(output, delimiter=",", quotechar='"', lineterminator="\n")
    writer.writerow(CSV_HEADER)
    for index, result in enumerate(results):
        if not isinstance(result, Mapping):
            _fail("result_shape", f"report.results[{index}] must be an object")
        writer.writerow(_csv_row(result))
    return output.getvalue().encode("utf-8")


def render_json(report: Mapping[str, Any]) -> bytes:
    """Render strict deterministic UTF-8 JSON."""

    _is_json(report, "report")
    try:
        return (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReportFailure("json_value", f"cannot encode report JSON: {exc}") from exc


def _check_destination(path: Path, replace: bool) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ReportFailure("output_path", str(exc)) from exc
    if stat.S_ISLNK(info.st_mode):
        _fail("symlink_output", f"output symlink is not permitted: {path}")
    if not stat.S_ISREG(info.st_mode):
        _fail("nonregular_output", f"output must be a regular file: {path}")
    if stat.S_IMODE(info.st_mode) != 0o644:
        _fail("output_mode", f"output mode must be exactly 0644: {path}")
    if not replace:
        _fail("output_exists", f"output already exists: {path}")


def _stage(path: Path, payload: bytes) -> Path:
    if not path.parent.is_dir():
        _fail("output_path", f"output parent is not a directory: {path.parent}")
    descriptor = -1
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        temporary = Path(name)
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        return temporary
    except OSError as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise ReportFailure("atomic_write", str(exc)) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _rollback_no_clobber_publication(path: Path, expected_inode: tuple[int, int]) -> None:
    """Remove a destination only while it still names our published inode.

    A later writer may replace a destination after our hard link succeeds.  Checking the
    device/inode pair before unlinking keeps rollback from deleting that writer's file.
    Rollback is best effort: the publication failure that triggered it must retain its
    original typed error even if the destination becomes inaccessible concurrently.
    """

    try:
        info = os.lstat(path)
    except OSError:
        return
    if (info.st_dev, info.st_ino) != expected_inode:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def write_report_pair(
    json_path: Path | str,
    csv_path: Path | str,
    report: Mapping[str, Any],
    protocol: Mapping[str, Any],
    report_schema: Mapping[str, Any],
    corpus: Mapping[str, Any],
    corpus_schema: Mapping[str, Any],
    *,
    replace: bool = False,
) -> None:
    """Validate and stage both outputs, then publish each destination deterministically.

    By default each staged file is installed with an atomic same-directory hard link,
    so a destination created after the initial check cannot be clobbered. With replace
    true, os.replace is used as explicitly requested. The two destinations are staged
    together, but their publication is not jointly atomic. In the default no-clobber
    mode, a JSON destination linked by this invocation is rolled back if CSV publication
    fails before the pair is complete; an inode replaced by another writer is preserved.
    With replace true, replacement is explicit and a successful JSON replacement remains
    in place if the later CSV replacement fails.
    """

    json_destination = Path(json_path)
    csv_destination = Path(csv_path)
    if json_destination == csv_destination:
        _fail("output_path", "JSON and CSV destinations must differ")
    try:
        validate_report_consistency(report, protocol, report_schema, corpus, corpus_schema)
    except (ValidationError, ValueError, TypeError) as exc:
        raise ReportFailure("validation", str(exc)) from exc
    json_payload = render_json(report)
    csv_payload = render_csv(report)
    _check_destination(json_destination, replace)
    _check_destination(csv_destination, replace)
    json_temporary: Path | None = None
    csv_temporary: Path | None = None
    json_published_inode: tuple[int, int] | None = None
    pair_published = False
    try:
        json_temporary = _stage(json_destination, json_payload)
        csv_temporary = _stage(csv_destination, csv_payload)
        if replace:
            # Explicit replacement is intentionally not rolled back: restoring a prior
            # destination would require retaining and safely restoring its old inode.
            os.replace(json_temporary, json_destination)
            json_temporary = None
            os.replace(csv_temporary, csv_destination)
            csv_temporary = None
        else:
            json_info = os.lstat(json_temporary)
            try:
                os.link(json_temporary, json_destination, follow_symlinks=False)
            except FileExistsError:
                _fail("output_exists", f"output appeared during publication: {json_destination}")
            json_published_inode = (json_info.st_dev, json_info.st_ino)
            os.unlink(json_temporary)
            json_temporary = None

            try:
                os.link(csv_temporary, csv_destination, follow_symlinks=False)
            except FileExistsError:
                _fail("output_exists", f"output appeared during publication: {csv_destination}")
            # Both destinations now exist.  If cleanup of the CSV staging name fails,
            # retain the complete pair rather than removing the JSON destination.
            pair_published = True
            os.unlink(csv_temporary)
            csv_temporary = None
        pair_published = True
        for parent in {json_destination.parent, csv_destination.parent}:
            descriptor = os.open(str(parent), os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except OSError as exc:
        raise ReportFailure("atomic_write", str(exc)) from exc
    finally:
        if not replace and not pair_published and json_published_inode is not None:
            _rollback_no_clobber_publication(json_destination, json_published_inode)
        for temporary in (json_temporary, csv_temporary):
            if temporary is not None:
                temporary.unlink(missing_ok=True)


__all__ = ["CSV_HEADER", "ReportFailure", "assemble_report", "render_csv", "render_json", "write_report_pair"]
