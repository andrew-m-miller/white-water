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


def _parse_coordinate(value: Any, path: str) -> float:
    if isinstance(value, bool):
        _fail("bool_value", f"{path} must be numeric, not bool")
    if not isinstance(value, Real):
        _fail("non_numeric", f"{path} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        _fail("nonfinite_value", f"{path} must be finite")
    return number


def _parse_numeric_grid(
    grid: Any, label: str, channels: int | None = None
) -> tuple[list[list[float | tuple[float, ...] | None]], tuple[int, int], int]:
    """Parse a rectangular scalar, RGB, or flow grid without dropping bad cells."""

    if not _is_sequence(grid) or not grid:
        _fail("empty_grid", f"{label} must contain at least one row")
    rows: list[list[float | tuple[float, ...] | None]] = []
    width: int | None = None
    inferred_channels = channels
    for row_index, row in enumerate(grid):
        if not _is_sequence(row) or not row:
            _fail("ragged_shape", f"{label}[{row_index}] must be a non-empty row")
        if width is None:
            width = len(row)
        elif len(row) != width:
            _fail("ragged_shape", f"{label} rows have different widths")
        parsed_row: list[float | tuple[float, ...] | None] = []
        for column_index, cell in enumerate(row):
            path = f"{label}[{row_index}][{column_index}]"
            if cell is None:
                parsed_row.append(None)
                continue
            if _is_sequence(cell):
                if inferred_channels is None:
                    inferred_channels = len(cell)
                if len(cell) != inferred_channels or inferred_channels not in (2, 3):
                    _fail("grid_shape", f"{path} has the wrong channel count")
                components = []
                for component_index, component in enumerate(cell):
                    if isinstance(component, bool):
                        _fail(
                            "bool_value",
                            f"{path}[{component_index}] must be numeric, not bool",
                        )
                    if not isinstance(component, Real):
                        _fail(
                            "non_numeric",
                            f"{path}[{component_index}] must be a real number",
                        )
                    components.append(float(component))
                parsed_row.append(tuple(components))
                continue
            if inferred_channels is None:
                inferred_channels = 1
            if inferred_channels != 1:
                _fail("grid_shape", f"{path} must have {inferred_channels} components")
            if isinstance(cell, bool):
                _fail("bool_value", f"{path} must be numeric, not bool")
            if not isinstance(cell, Real):
                _fail("non_numeric", f"{path} must be a real number")
            parsed_row.append(float(cell))
        rows.append(parsed_row)
    if inferred_channels is None:
        _fail("missing_value", f"{label} contains no typed cells")
    assert width is not None
    return rows, (len(rows), width), inferred_channels


def _parse_sample_coordinate(value: Any, path: str) -> float:
    """Parse a coordinate while retaining the typed nonfinite failure at sampling time."""

    if isinstance(value, bool):
        _fail("bool_value", f"{path} must be numeric, not bool")
    if not isinstance(value, Real):
        _fail("non_numeric", f"{path} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        _fail("nonfinite_value", f"{path} must be finite")
    return number


def _sample_parsed_grid(
    rows: list[list[float | tuple[float, ...] | None]],
    shape: tuple[int, int],
    channels: int,
    x: Any,
    y: Any,
    label: str,
) -> float | tuple[float, ...]:
    """Bilinearly sample a bottom-row-origin grid at x-right/y-up coordinates."""

    sample_x = _parse_sample_coordinate(x, f"{label}.x")
    sample_y = _parse_sample_coordinate(y, f"{label}.y")
    height, width = shape
    if sample_x < 0.0 or sample_x > width - 1 or sample_y < 0.0 or sample_y > height - 1:
        _fail("out_of_bounds", f"{label} coordinate ({sample_x}, {sample_y}) is outside grid")
    left = math.floor(sample_x)
    bottom = math.floor(sample_y)
    right = min(left + 1, width - 1)
    top = min(bottom + 1, height - 1)
    x_fraction = sample_x - left
    y_fraction = sample_y - bottom

    corners = (
        (rows[bottom][left], (1.0 - x_fraction) * (1.0 - y_fraction)),
        (rows[bottom][right], x_fraction * (1.0 - y_fraction)),
        (rows[top][left], (1.0 - x_fraction) * y_fraction),
        (rows[top][right], x_fraction * y_fraction),
    )
    if channels == 1:
        result = 0.0
        for value, weight in corners:
            if weight == 0.0:
                continue
            if value is None:
                _fail("missing_value", f"{label} neighborhood contains a missing value")
            if not math.isfinite(value):  # type: ignore[arg-type]
                _fail("nonfinite_value", f"{label} neighborhood contains a nonfinite value")
            result += float(value) * weight  # type: ignore[arg-type]
        return result

    result = [0.0] * channels
    for vector, weight in corners:
        if weight == 0.0:
            continue
        if vector is None:
            _fail("missing_value", f"{label} neighborhood contains a missing value")
        if any(not math.isfinite(component) for component in vector):  # type: ignore[union-attr]
            _fail("nonfinite_value", f"{label} neighborhood contains a nonfinite value")
        for component_index in range(channels):
            result[component_index] += vector[component_index] * weight  # type: ignore[index]
    return tuple(result)


def bilinear_sample(grid: Any, x: Any, y: Any, channels: int | None = None) -> Any:
    """Sample a scalar/RGB/flow grid with bottom-row-origin bilinear interpolation."""

    rows, shape, inferred_channels = _parse_numeric_grid(grid, "grid", channels)
    return _sample_parsed_grid(rows, shape, inferred_channels, x, y, "grid")


def _parse_required_mask(mask: Any, shape: tuple[int, int], label: str) -> list[list[bool]]:
    if mask is None:
        _fail("mask_required", f"{label} is required")
    parsed = _parse_mask(mask, shape)
    assert parsed is not None
    return parsed


def _parse_points(points: Any, label: str) -> list[tuple[float | None, float | None] | None]:
    if not _is_sequence(points) or not points:
        _fail("empty_landmarks", f"{label} must contain at least one point")
    parsed: list[tuple[float | None, float | None] | None] = []
    for index, point in enumerate(points):
        path = f"{label}[{index}]"
        if point is None:
            parsed.append(None)
            continue
        if not _is_sequence(point) or len(point) != 2:
            _fail("point_shape", f"{path} must be a two-component point or None")
        components: list[float | None] = []
        for component_index, component in enumerate(point):
            if component is None:
                components.append(None)
                continue
            component_path = f"{path}[{component_index}]"
            if isinstance(component, bool):
                _fail("bool_value", f"{component_path} must be numeric, not bool")
            if not isinstance(component, Real):
                _fail("non_numeric", f"{component_path} must be a real number")
            components.append(float(component))
        parsed.append((components[0], components[1]))
    return parsed


def _parse_point_mask(mask: Any, count: int) -> list[bool]:
    if mask is None:
        _fail("mask_required", "landmark valid_mask is required")
    if not _is_sequence(mask) or len(mask) != count:
        _fail("mask_shape", "landmark valid_mask length does not match landmarks")
    parsed = []
    for index, value in enumerate(mask):
        if not isinstance(value, bool):
            _fail("mask_value", f"landmark valid_mask[{index}] must be bool")
        parsed.append(value)
    if not any(parsed):
        _fail("empty_mask", "landmark valid_mask selects no points")
    return parsed


def _point_is_finite(point: tuple[float | None, float | None] | None) -> bool:
    return (
        point is not None
        and point[0] is not None
        and point[1] is not None
        and math.isfinite(point[0])
        and math.isfinite(point[1])
    )


def landmark_metrics(
    flow: Any,
    source_landmarks: Any,
    target_landmarks: Any,
    valid_mask: Any,
) -> dict[str, float]:
    """Measure landmark target error after sampling image1->image2 flow at each source point."""

    parsed_flow, flow_shape, flow_channels = _parse_numeric_grid(flow, "flow", 2)
    assert flow_channels == 2
    sources = _parse_points(source_landmarks, "source_landmarks")
    targets = _parse_points(target_landmarks, "target_landmarks")
    if len(sources) != len(targets):
        _fail("shape_mismatch", "source and target landmark counts differ")
    mask = _parse_point_mask(valid_mask, len(sources))
    errors = []
    for index, (source, target) in enumerate(zip(sources, targets)):
        if not mask[index]:
            continue
        if not _point_is_finite(source) or not _point_is_finite(target):
            _fail("nonfinite_value", f"landmark {index} is not finite")
        assert source is not None and target is not None
        displacement = _sample_parsed_grid(
            parsed_flow, flow_shape, 2, source[0], source[1], f"flow at landmark {index}"
        )
        assert isinstance(displacement, tuple)
        predicted_x = source[0] + displacement[0]  # type: ignore[operator]
        predicted_y = source[1] + displacement[1]  # type: ignore[operator]
        errors.append(math.hypot(predicted_x - target[0], predicted_y - target[1]))  # type: ignore[operator]
    if not errors:
        _fail("empty_mask", "landmark valid_mask selects no measurable points")
    return {
        "landmark_median_error_px": linear_quantile(errors, 0.5),
        "landmark_p95_error_px": linear_quantile(errors, 0.95),
    }


def _mean_rgb_residual(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    return sum(abs(a - b) for a, b in zip(first, second)) / len(first)


def visible_warp_residual(
    image1: Any, image2: Any, forward_flow: Any, visible_mask: Any
) -> float:
    """Return mean absolute RGB residual after forward-warping image1 into image2."""

    parsed_image1, image1_shape, image1_channels = _parse_numeric_grid(image1, "image1", 3)
    parsed_image2, image2_shape, image2_channels = _parse_numeric_grid(image2, "image2", 3)
    parsed_flow, flow_shape, flow_channels = _parse_numeric_grid(forward_flow, "forward_flow", 2)
    if image1_shape != flow_shape:
        _fail("shape_mismatch", "image1 and forward_flow source shapes differ")
    assert image1_channels == image2_channels == 3 and flow_channels == 2
    mask = _parse_required_mask(visible_mask, flow_shape, "visible_mask")
    residuals = []
    for row in range(flow_shape[0]):
        for column in range(flow_shape[1]):
            if not mask[row][column]:
                continue
            source_rgb = _sample_parsed_grid(
                parsed_image1, image1_shape, 3, column, row, "image1"
            )
            displacement = _sample_parsed_grid(
                parsed_flow, flow_shape, 2, column, row, "forward_flow"
            )
            assert isinstance(source_rgb, tuple) and isinstance(displacement, tuple)
            target_rgb = _sample_parsed_grid(
                parsed_image2,
                image2_shape,
                3,
                column + displacement[0],
                row + displacement[1],
                "image2 at forward coordinate",
            )
            assert isinstance(target_rgb, tuple)
            residuals.append(_mean_rgb_residual(source_rgb, target_rgb))
    if not residuals:
        _fail("empty_mask", "visible_mask selects no pixels")
    return sum(residuals) / len(residuals)


def forward_backward_residual_px(
    forward_flow: Any, backward_flow: Any, valid_mask: Any
) -> float:
    """Return mean norm(F(x)+B(x+F(x))) over the explicit source mask."""

    parsed_forward, forward_shape, forward_channels = _parse_numeric_grid(forward_flow, "forward_flow", 2)
    parsed_backward, backward_shape, backward_channels = _parse_numeric_grid(backward_flow, "backward_flow", 2)
    assert forward_channels == backward_channels == 2
    mask = _parse_required_mask(valid_mask, forward_shape, "valid_mask")
    residuals = []
    for row in range(forward_shape[0]):
        for column in range(forward_shape[1]):
            if not mask[row][column]:
                continue
            displacement = _sample_parsed_grid(
                parsed_forward, forward_shape, 2, column, row, "forward_flow"
            )
            assert isinstance(displacement, tuple)
            reverse = _sample_parsed_grid(
                parsed_backward,
                backward_shape,
                2,
                column + displacement[0],
                row + displacement[1],
                "backward_flow at forward coordinate",
            )
            assert isinstance(reverse, tuple)
            residuals.append(
                math.hypot(displacement[0] + reverse[0], displacement[1] + reverse[1])
            )
    if not residuals:
        _fail("empty_mask", "valid_mask selects no pixels")
    return sum(residuals) / len(residuals)


def chain_drift_px(
    flows: Any, truth_flow: Any, valid_mask: Any, link_count: Any
) -> float:
    """Measure composed chain displacement against analytic truth for one exact link count."""

    if isinstance(link_count, bool) or not isinstance(link_count, int):
        _fail("link_count", "link_count must be one of 1, 2, 4, or 8")
    if link_count not in (1, 2, 4, 8):
        _fail("link_count", "link_count must be one of 1, 2, 4, or 8")
    if not _is_sequence(flows) or len(flows) != link_count:
        _fail("link_count", "flows must contain exactly link_count links")
    parsed_flows = []
    flow_shapes = []
    for index, flow in enumerate(flows):
        parsed, shape, channels = _parse_numeric_grid(flow, f"flows[{index}]", 2)
        assert channels == 2
        parsed_flows.append(parsed)
        flow_shapes.append(shape)
    parsed_truth, truth_shape, truth_channels = _parse_numeric_grid(truth_flow, "truth_flow", 2)
    assert truth_channels == 2
    if flow_shapes[0] != truth_shape:
        _fail("shape_mismatch", "first flow and truth_flow source shapes differ")
    mask = _parse_required_mask(valid_mask, truth_shape, "valid_mask")
    residuals = []
    for row in range(truth_shape[0]):
        for column in range(truth_shape[1]):
            if not mask[row][column]:
                continue
            current_x = float(column)
            current_y = float(row)
            for index, (parsed, shape) in enumerate(zip(parsed_flows, flow_shapes)):
                displacement = _sample_parsed_grid(
                    parsed, shape, 2, current_x, current_y, f"flows[{index}] at advected coordinate"
                )
                assert isinstance(displacement, tuple)
                current_x += displacement[0]
                current_y += displacement[1]
            truth = _sample_parsed_grid(
                parsed_truth, truth_shape, 2, column, row, "truth_flow"
            )
            assert isinstance(truth, tuple)
            residuals.append(
                math.hypot(
                    (current_x - column) - truth[0],
                    (current_y - row) - truth[1],
                )
            )
    if not residuals:
        _fail("empty_mask", "valid_mask selects no pixels")
    return sum(residuals) / len(residuals)


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
    "bilinear_sample",
    "chain_drift_px",
    "dense_metrics",
    "forward_backward_residual_px",
    "landmark_metrics",
    "linear_quantile",
    "macro_aggregate",
    "nonfinite_fraction",
    "repeated_run_p99_delta_px",
    "visible_warp_residual",
]
