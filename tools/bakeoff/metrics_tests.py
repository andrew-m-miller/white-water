#!/usr/bin/env python3
"""Focused standard-library tests for the frozen P25-4 metrics seam."""

from __future__ import annotations

import math

from .metrics import (
    MetricFailure,
    dense_metrics,
    linear_quantile,
    macro_aggregate,
    nonfinite_fraction,
    repeated_run_p99_delta_px,
)


def _failure(kind: str, callback) -> None:
    try:
        callback()
    except MetricFailure as failure:
        assert failure.kind == kind, (failure.kind, kind)
        assert failure.reason == kind
        assert failure.failure_type == "metric_failure"
    else:
        raise AssertionError(f"expected MetricFailure({kind})")


def test_quantile() -> None:
    assert linear_quantile([0.0, 10.0, 20.0, 30.0], 0.0) == 0.0
    assert linear_quantile([0.0, 10.0, 20.0, 30.0], 0.25) == 7.5
    assert linear_quantile([0.0, 10.0, 20.0, 30.0], 1.0) == 30.0
    assert linear_quantile([math.nan, 0.0, 10.0], 0.5) == 5.0
    _failure("invalid_percentile", lambda: linear_quantile([1.0], True))
    _failure("bool_value", lambda: linear_quantile([1.0, False], 0.5))
    _failure("empty_values", lambda: linear_quantile([], 0.5))


def test_dense_and_mask() -> None:
    predicted = [[(0.0, 0.0), (1.0, 0.0), (3.0, 0.0), (math.nan, 0.0)]]
    truth = [[(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]]
    metrics = dense_metrics(predicted, truth)
    assert metrics == {
        "endpoint_error_px": 4.0 / 3.0,
        "fraction_le_1px": 2.0 / 3.0,
        "fraction_le_3px": 1.0,
        "nonfinite_fraction": 0.25,
    }
    masked = dense_metrics(predicted, truth, [[True, True, False, False]])
    assert masked["endpoint_error_px"] == 0.5
    assert masked["fraction_le_1px"] == 1.0
    assert masked["fraction_le_3px"] == 1.0
    assert masked["nonfinite_fraction"] == 0.0
    assert nonfinite_fraction(
        [[None, (1.0, None)]], [[(0.0, 0.0), (0.0, 0.0)]]
    ) == 1.0
    assert nonfinite_fraction(
        [[(0.0, 0.0)]], [[(math.nan, 0.0)]]
    ) == 0.0
    masked_truth = dense_metrics(
        [[(0.0, 0.0), (0.0, 0.0)]],
        [[(0.0, 0.0), (math.nan, 0.0)]],
        [[True, False]],
    )
    assert masked_truth["endpoint_error_px"] == 0.0
    inclusive = dense_metrics([[(1.0, 0.0), (3.0, 0.0)]], [[(0.0, 0.0), (0.0, 0.0)]])
    assert inclusive["fraction_le_1px"] == 0.5
    assert inclusive["fraction_le_3px"] == 1.0
    discriminating = dense_metrics(
        [[(0.0, 0.0), (0.0, 0.0), (3.0, 0.0)]],
        [[(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]],
    )
    assert discriminating["endpoint_error_px"] == 1.0


def test_shape_and_empty_failures() -> None:
    _failure("ragged_shape", lambda: dense_metrics([[(0.0, 0.0)], []], [[(0.0, 0.0)], []]))
    _failure("shape_mismatch", lambda: dense_metrics([[(0.0, 0.0)]], [[(0.0, 0.0), (0.0, 0.0)]]))
    _failure("vector_shape", lambda: dense_metrics([[(0.0,)]], [[(0.0, 0.0)]]))
    _failure("bool_value", lambda: dense_metrics([[(True, 0.0)]], [[(0.0, 0.0)]]))
    _failure("mask_value", lambda: dense_metrics([[(0.0, 0.0)]], [[(0.0, 0.0)]], [[1]]))
    _failure("empty_mask", lambda: dense_metrics([[(0.0, 0.0)]], [[(0.0, 0.0)]], []))
    _failure("empty_mask", lambda: dense_metrics([[(0.0, 0.0)]], [[(0.0, 0.0)]], [[False]]))
    _failure("no_valid_slots", lambda: dense_metrics([[(None, None)]], [[(0.0, 0.0)]]))
    _failure("invalid_truth", lambda: dense_metrics([[(0.0, 0.0)]], [[(math.nan, 0.0)]]))


def test_repeated_run_p99() -> None:
    first = [[(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]]
    second = [[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]]
    assert math.isclose(repeated_run_p99_delta_px([first, second]), 2.97)
    assert repeated_run_p99_delta_px([first]) == 0.0
    _failure("shape_mismatch", lambda: repeated_run_p99_delta_px([first, [[(0.0, 0.0)]]]))
    _failure("empty_runs", lambda: repeated_run_p99_delta_px([]))


def test_macro_equal_weighting() -> None:
    records = [
        {"partition": "synthetic", "category": "identity", "shot_id": "a", "samples": [0.0, 2.0]},
        {"partition": "synthetic", "category": "identity", "shot_id": "b", "samples": [10.0]},
        {"partition": "synthetic", "category": "translation", "shot_id": "c", "value": 4.0},
        {"partition": "production_external", "category": "blur", "shot_id": "d", "value": 8.0},
    ]
    result = macro_aggregate(records)
    assert result["partitions"]["synthetic"]["categories"]["identity"]["shots"] == {"a": 1.0, "b": 10.0}
    assert result["partitions"]["synthetic"]["categories"]["identity"]["macro"] == 5.5
    assert result["partitions"]["synthetic"]["macro"] == 4.75
    assert result["partitions"]["production_external"]["macro"] == 8.0
    assert "overall" not in result

    nested = macro_aggregate(
        {"synthetic": {"identity": {"a": [0.0, 2.0], "b": [10.0]}}}
    )
    assert nested["partitions"]["synthetic"]["macro"] == 5.5
    _failure("empty_partitions", lambda: macro_aggregate({}))
    _failure("empty_categories", lambda: macro_aggregate({"synthetic": {}}))
    _failure("empty_shots", lambda: macro_aggregate({"synthetic": {"identity": {}}}))
    _failure("duplicate_shot", lambda: macro_aggregate(records + [records[0]]))
    _failure("bool_value", lambda: macro_aggregate([{"partition": "p", "category": "c", "shot_id": "s", "value": True}]))


def main() -> int:
    test_quantile()
    test_dense_and_mask()
    test_shape_and_empty_failures()
    test_repeated_run_p99()
    test_macro_equal_weighting()
    print("P25-4 metrics tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
