#!/usr/bin/env python3
"""Dependency-free frozen metrics used by the offline Phase 2.5 bake-off.

The functions in this module operate on ordinary Python lists and tuples.  They deliberately
validate their shape at the seam: a missing pixel may be represented by ``None`` (and is an
invalid expected slot), while a missing row, ragged row, wrong vector width, or mismatched pair
is a typed input failure rather than a silently truncated score.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real
from typing import Any


class MetricFailure(ValueError):
    """Stable, reportable metric failure.

    ``kind`` is machine vocabulary; callers must not parse the human message.  ``reason`` and
    ``failure_type`` mirror the typed failure conventions used by the conditioning seam.
    """

    def __init__(self, kind: str, message: str):
        self.kind = kind
        self.reason = kind
        self.failure_type = "metric_failure"
        self.message = message
        super().__init__(f"{kind}: {message}")


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _fail(kind: str, message: str) -> None:
    raise MetricFailure(kind, message)


def linear_quantile(values: Sequence[Any], percentile: Any) -> float:
    """Return the frozen linear quantile ``h=(n-1)*p``.

    Values are sorted in ascending numeric order.  Non-finite numeric values are discarded,
    matching the protocol's finite-sample rule; malformed values and booleans remain typed
    failures.  The interpolation is inclusive at both endpoints.
    """

    if isinstance(percentile, bool) or not isinstance(percentile, Real):
        _fail("invalid_percentile", "percentile must be a real number in [0, 1]")
    p = float(percentile)
    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        _fail("invalid_percentile", "percentile must be a finite number in [0, 1]")
    if not _is_sequence(values) or not values:
        _fail("empty_values", "quantile needs at least one value")
    ordered: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool):
            _fail("bool_value", f"values[{index}] must be numeric, not bool")
        if not isinstance(value, Real):
            _fail("non_numeric", f"values[{index}] must be a real number")
        number = float(value)
        if math.isfinite(number):
            ordered.append(number)
    if not ordered:
        _fail("empty_values", "quantile has no finite values")
    ordered.sort()
    height = (len(ordered) - 1) * p
    lower = math.floor(height)
    upper = math.ceil(height)
    fraction = height - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _parse_flow(flow: Any, label: str) -> tuple[list[list[tuple[float | None, float | None]]], tuple[int, int]]:
    """Parse a rectangular ``rows -> vectors -> (dx, dy)`` flow."""

    if not _is_sequence(flow) or not flow:
        _fail("empty_flow", f"{label} must contain at least one row")
    rows: list[list[tuple[float | None, float | None]]] = []
    width: int | None = None
    for row_index, row in enumerate(flow):
        if not _is_sequence(row) or not row:
            _fail("ragged_shape", f"{label}[{row_index}] must be a non-empty row")
        if width is None:
            width = len(row)
        elif len(row) != width:
            _fail("ragged_shape", f"{label} rows have different widths")
        parsed_row: list[tuple[float | None, float | None]] = []
        for column_index, vector in enumerate(row):
            path = f"{label}[{row_index}][{column_index}]"
            if vector is None:
                parsed_row.append((None, None))
                continue
            if not _is_sequence(vector) or len(vector) != 2:
                _fail("vector_shape", f"{path} must be a two-component vector or None")
            components: list[float | None] = []
            for component_index, component in enumerate(vector):
                if component is None:
                    components.append(None)
                    continue
                if isinstance(component, bool):
                    _fail("bool_value", f"{path}[{component_index}] must be numeric, not bool")
                if not isinstance(component, Real):
                    _fail("non_numeric", f"{path}[{component_index}] must be numeric or None")
                components.append(float(component))
            parsed_row.append((components[0], components[1]))
        rows.append(parsed_row)
    assert width is not None
    return rows, (len(rows), width)


def _parse_mask(mask: Any, shape: tuple[int, int]) -> list[list[bool]] | None:
    if mask is None:
        return None
    if _is_sequence(mask) and not mask:
        _fail("empty_mask", "valid mask selects no expected flow slots")
    height, width = shape
    if not _is_sequence(mask) or len(mask) != height:
        _fail("mask_shape", "valid mask height does not match flow")
    parsed: list[list[bool]] = []
    for row_index, row in enumerate(mask):
        if not _is_sequence(row) or len(row) != width:
            _fail("mask_shape", "valid mask must match the rectangular flow shape")
        parsed_row: list[bool] = []
        for column_index, value in enumerate(row):
            if not isinstance(value, bool):
                _fail("mask_value", f"valid_mask[{row_index}][{column_index}] must be bool")
            parsed_row.append(value)
        parsed.append(parsed_row)
    if not any(any(row) for row in parsed):
        _fail("empty_mask", "valid mask selects no expected flow slots")
    return parsed


def _parse_pair(
    predicted: Any,
    truth: Any,
    valid_mask: Any = None,
) -> tuple[
    list[list[tuple[float | None, float | None]]],
    list[list[tuple[float | None, float | None]]],
    list[list[bool]] | None,
    tuple[int, int],
]:
    parsed_predicted, predicted_shape = _parse_flow(predicted, "predicted")
    parsed_truth, truth_shape = _parse_flow(truth, "truth")
    if predicted_shape != truth_shape:
        _fail("shape_mismatch", "predicted and truth flow shapes differ")
    return parsed_predicted, parsed_truth, _parse_mask(valid_mask, predicted_shape), predicted_shape


def _slot_is_finite(slot: tuple[float | None, float | None]) -> bool:
    return (
        slot[0] is not None
        and slot[1] is not None
        and math.isfinite(slot[0])
        and math.isfinite(slot[1])
    )


def _slot_selected(mask: list[list[bool]] | None, row: int, column: int) -> bool:
    return mask is None or mask[row][column]


def nonfinite_fraction(predicted: Any, truth: Any, valid_mask: Any = None) -> float:
    """Return invalid predicted slots divided by selected expected slots.

    Truth is parsed here only to establish the expected rectangular shape.  Its validity is a
    separate contract: callers select unavailable truth with ``valid_mask`` rather than having
    a bad truth value reported as a model-output failure.
    """

    parsed_predicted, parsed_truth, mask, shape = _parse_pair(predicted, truth, valid_mask)
    selected = 0
    invalid = 0
    for row in range(shape[0]):
        for column in range(shape[1]):
            if not _slot_selected(mask, row, column):
                continue
            selected += 1
            if not _slot_is_finite(parsed_predicted[row][column]):
                invalid += 1
    if selected == 0:  # Defensive: _parse_mask catches an all-false mask.
        _fail("empty_mask", "no expected flow slots remain after masking")
    return invalid / selected


def dense_metrics(predicted: Any, truth: Any, valid_mask: Any = None) -> dict[str, float]:
    """Compute mean dense endpoint error, inclusive accuracy fractions, and invalid fraction."""

    parsed_predicted, parsed_truth, mask, shape = _parse_pair(predicted, truth, valid_mask)
    errors: list[float] = []
    selected = 0
    invalid = 0
    for row in range(shape[0]):
        for column in range(shape[1]):
            if not _slot_selected(mask, row, column):
                continue
            selected += 1
            predicted_slot = parsed_predicted[row][column]
            truth_slot = parsed_truth[row][column]
            if not _slot_is_finite(truth_slot):
                _fail("invalid_truth", f"truth[{row}][{column}] is not finite")
            if not _slot_is_finite(predicted_slot):
                invalid += 1
                continue
            dx = predicted_slot[0] - truth_slot[0]  # type: ignore[operator]
            dy = predicted_slot[1] - truth_slot[1]  # type: ignore[operator]
            errors.append(math.hypot(dx, dy))
    if selected == 0:
        _fail("empty_mask", "no expected flow slots remain after masking")
    if not errors:
        _fail("no_valid_slots", "dense metrics have no finite selected flow slots")
    return {
        "endpoint_error_px": sum(errors) / len(errors),
        "fraction_le_1px": sum(error <= 1.0 for error in errors) / len(errors),
        "fraction_le_3px": sum(error <= 3.0 for error in errors) / len(errors),
        "nonfinite_fraction": invalid / selected,
    }


def repeated_run_p99_delta_px(runs: Any) -> float:
    """Return p99 Euclidean delta from the first same-shaped run."""

    if not _is_sequence(runs) or not runs:
        _fail("empty_runs", "repeated-run metric needs at least one run")
    parsed_runs: list[list[list[tuple[float | None, float | None]]]] = []
    shape: tuple[int, int] | None = None
    for index, run in enumerate(runs):
        parsed, run_shape = _parse_flow(run, f"runs[{index}]")
        if shape is None:
            shape = run_shape
        elif run_shape != shape:
            _fail("shape_mismatch", "repeated runs have different flow shapes")
        parsed_runs.append(parsed)
    if len(parsed_runs) == 1:
        return 0.0
    assert shape is not None
    deltas: list[float] = []
    first = parsed_runs[0]
    for row in range(shape[0]):
        for column in range(shape[1]):
            baseline = first[row][column]
            if not _slot_is_finite(baseline):
                continue
            for run in parsed_runs[1:]:
                current = run[row][column]
                if not _slot_is_finite(current):
                    continue
                deltas.append(math.hypot(current[0] - baseline[0], current[1] - baseline[1]))  # type: ignore[operator]
    if not deltas:
        _fail("no_valid_slots", "repeated runs have no finite comparable slots")
    return linear_quantile(deltas, 0.99)


def _scalar(value: Any, path: str) -> float:
    if isinstance(value, bool):
        _fail("bool_value", f"{path} must be numeric, not bool")
    if not isinstance(value, Real):
        _fail("non_numeric", f"{path} must be a real scalar")
    number = float(value)
    if not math.isfinite(number):
        _fail("nonfinite_value", f"{path} must be finite")
    return number


def _sample_records(samples: Any) -> dict[str, dict[str, dict[str, list[float]]]]:
    """Normalize explicit rows or nested ``partition -> category -> shot -> samples``."""

    grouped: dict[str, dict[str, dict[str, list[float]]]] = {}
    if isinstance(samples, Mapping):
        if not samples:
            _fail("empty_partitions", "macro aggregate needs at least one partition")
        for partition, categories in samples.items():
            if not isinstance(partition, str) or not partition:
                _fail("invalid_id", "partition ids must be non-empty strings")
            if not isinstance(categories, Mapping) or not categories:
                _fail("empty_categories", f"partition {partition!r} has no categories")
            partition_group: dict[str, dict[str, list[float]]] = {}
            grouped[partition] = partition_group
            for category, shots in categories.items():
                if not isinstance(category, str) or not category:
                    _fail("invalid_id", "category ids must be non-empty strings")
                if not isinstance(shots, Mapping) or not shots:
                    _fail("empty_shots", f"category {category!r} has no explicit shots")
                shot_group: dict[str, list[float]] = {}
                partition_group[category] = shot_group
                for shot, values in shots.items():
                    if not isinstance(shot, str) or not shot:
                        _fail("invalid_id", "shot ids must be non-empty strings")
                    if not _is_sequence(values) or not values:
                        _fail("empty_samples", f"shot {shot!r} has no scalar samples")
                    shot_group[shot] = [_scalar(value, f"{partition}.{category}.{shot}") for value in values]
        return grouped

    if not _is_sequence(samples) or not samples:
        _fail("empty_partitions", "macro aggregate needs explicit shot samples")
    for index, record in enumerate(samples):
        if not isinstance(record, Mapping):
            _fail("record_shape", f"samples[{index}] must be a mapping")
        partition = record.get("partition")
        category = record.get("category")
        shot = record.get("shot_id", record.get("shot"))
        if not all(isinstance(value, str) and value for value in (partition, category, shot)):
            _fail("invalid_id", f"samples[{index}] needs non-empty partition/category/shot_id")
        if "samples" in record:
            values = record["samples"]
            if not _is_sequence(values) or not values:
                _fail("empty_samples", f"samples[{index}].samples is empty")
            numbers = [_scalar(value, f"samples[{index}].samples") for value in values]
        elif "value" in record:
            numbers = [_scalar(record["value"], f"samples[{index}].value")]
        else:
            _fail("record_shape", f"samples[{index}] needs value or samples")
        partition_group = grouped.setdefault(partition, {})
        category_group = partition_group.setdefault(category, {})
        if shot in category_group:
            _fail("duplicate_shot", f"duplicate shot {shot!r} in {partition}/{category}")
        category_group[shot] = numbers
    return grouped


def _mean(values: Sequence[float], kind: str, message: str) -> float:
    if not values:
        _fail(kind, message)
    return sum(values) / len(values)


def macro_aggregate(samples: Any) -> dict[str, Any]:
    """Aggregate explicit scalar samples by shot, category, and partition.

    Each shot is first reduced by the frozen median.  Shot medians are then equally weighted
    within each category, and category means are equally weighted within each partition.  The
    result preserves both the intermediate shot/category values and each partition macro.
    """

    grouped = _sample_records(samples)
    partitions: dict[str, Any] = {}
    for partition, categories in grouped.items():
        category_results: dict[str, Any] = {}
        category_macros: list[float] = []
        for category, shots in categories.items():
            if not shots:
                _fail("empty_shots", f"category {category!r} has no explicit shots")
            shot_medians = {
                shot: linear_quantile(values, 0.5) for shot, values in shots.items()
            }
            category_macro = _mean(
                list(shot_medians.values()),
                "empty_shots",
                f"category {category!r} has no shot medians",
            )
            category_results[category] = {"shots": shot_medians, "macro": category_macro}
            category_macros.append(category_macro)
        partition_macro = _mean(
            category_macros,
            "empty_categories",
            f"partition {partition!r} has no category macros",
        )
        partitions[partition] = {"categories": category_results, "macro": partition_macro}
    return {"partitions": partitions}


__all__ = [
    "MetricFailure",
    "dense_metrics",
    "linear_quantile",
    "macro_aggregate",
    "nonfinite_fraction",
    "repeated_run_p99_delta_px",
]
