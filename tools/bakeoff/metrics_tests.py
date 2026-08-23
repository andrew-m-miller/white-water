#!/usr/bin/env python3
"""Focused standard-library tests for the frozen P25-4 metrics seam."""

from __future__ import annotations

import math

from .metrics import (
    MetricFailure,
    bilinear_sample,
    chain_drift_px,
    dense_macro_aggregate,
    dense_metrics,
    dense_score,
    forward_backward_residual_px,
    landmark_metrics,
    linear_quantile,
    macro_aggregate,
    nonfinite_fraction,
    repeated_run_p99_delta_px,
    visible_warp_residual,
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


def test_dense_score_formula_and_boundaries() -> None:
    assert dense_score(0.0, 1.0, 1.0) == 100.0
    assert dense_score(3.0, 0.0, 0.0) == 0.0
    assert dense_score(30.0, 0.25, 0.75) == 35.0
    assert math.isclose(dense_score(1.5, 0.25, 0.75), 45.0, rel_tol=0.0, abs_tol=1e-12)
    # Endpoint error beyond the 3 px scale floors only its contribution; the accuracy terms
    # remain intact.
    assert math.isclose(dense_score(4.0, 0.5, 1.0), 55.0, rel_tol=0.0, abs_tol=1e-12)
    _failure("bool_value", lambda: dense_score(True, 0.0, 0.0))
    _failure("bool_value", lambda: dense_score(0.0, False, 0.0))
    _failure("bool_value", lambda: dense_score(0.0, 0.0, True))
    _failure("nonfinite_value", lambda: dense_score(math.nan, 0.0, 0.0))
    _failure("nonfinite_value", lambda: dense_score(0.0, math.inf, 0.0))
    _failure("non_numeric", lambda: dense_score(0.0, 0.0, "1.0"))
    _failure("negative_endpoint_error", lambda: dense_score(-0.001, 0.0, 0.0))
    _failure("invalid_fraction", lambda: dense_score(0.0, -0.001, 0.0))
    _failure("invalid_fraction", lambda: dense_score(0.0, 0.0, 1.001))
    _failure("fraction_order", lambda: dense_score(0.0, 0.6, 0.5))


def test_dense_macro_explicit_records_and_weighting() -> None:
    records = [
        {
            "partition": "synthetic",
            "category": "identity",
            "shot_id": "perfect",
            "endpoint_error_px": 0.0,
            "fraction_le_1px": 1.0,
            "fraction_le_3px": 1.0,
        },
        {
            "partition": "synthetic",
            "category": "identity",
            "shot_id": "failed",
            "endpoint_error_px": 3.0,
            "fraction_le_1px": 0.0,
            "fraction_le_3px": 0.0,
        },
        {
            "partition": "synthetic",
            "category": "translation",
            "shot_id": "good",
            "endpoint_error_px": 0.0,
            "fraction_le_1px": 1.0,
            "fraction_le_3px": 1.0,
        },
        {
            "partition": "production_external",
            "category": "blur",
            "shot_id": "hard",
            "endpoint_error_px": 3.0,
            "fraction_le_1px": 0.0,
            "fraction_le_3px": 0.0,
        },
        {
            "partition": "production_external",
            "category": "detail",
            "shot_id": "easy",
            "endpoint_error_px": 0.0,
            "fraction_le_1px": 1.0,
            "fraction_le_3px": 1.0,
        },
    ]
    result = dense_macro_aggregate(records)
    synthetic = result["partitions"]["synthetic"]
    assert synthetic["categories"]["identity"]["shots"] == {"perfect": 100.0, "failed": 0.0}
    assert synthetic["categories"]["identity"]["macro"] == 50.0
    assert synthetic["categories"]["translation"]["macro"] == 100.0
    assert synthetic["macro"] == 75.0  # equal category weighting
    production = result["partitions"]["production_external"]
    assert production["macro"] == 50.0  # equal partition-category weighting
    assert set(production["categories"]) == {"blur", "detail"}

    # A shot ID is scoped by partition and category, as in macro_aggregate; only an explicit
    # duplicate in the same scope is rejected.
    same_shot_other_partition = [
        {**records[0], "partition": "production_external"},
    ]
    dense_macro_aggregate(same_shot_other_partition)
    _failure("duplicate_shot", lambda: dense_macro_aggregate([records[0], records[0]]))


def test_dense_macro_rejects_malformed_or_missing_records() -> None:
    valid = {
        "partition": "synthetic",
        "category": "identity",
        "shot_id": "shot",
        "endpoint_error_px": 0.0,
        "fraction_le_1px": 1.0,
        "fraction_le_3px": 1.0,
    }
    _failure("empty_partitions", lambda: dense_macro_aggregate([]))
    _failure("record_shape", lambda: dense_macro_aggregate([None]))
    _failure("invalid_id", lambda: dense_macro_aggregate([{**valid, "shot_id": None}]))
    missing_metric = dict(valid)
    del missing_metric["fraction_le_3px"]
    _failure("record_shape", lambda: dense_macro_aggregate([missing_metric]))
    shot_alias = dict(valid)
    del shot_alias["shot_id"]
    shot_alias["shot"] = "shot"
    _failure("record_shape", lambda: dense_macro_aggregate([shot_alias]))
    bool_metric = dict(valid)
    bool_metric["fraction_le_1px"] = True
    _failure("bool_value", lambda: dense_macro_aggregate([bool_metric]))
    nonfinite_metric = dict(valid)
    nonfinite_metric["endpoint_error_px"] = math.inf
    _failure("nonfinite_value", lambda: dense_macro_aggregate([nonfinite_metric]))


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


def test_bilinear_spatial_sampling() -> None:
    scalar = [[0.0, 10.0], [20.0, 30.0]]
    assert bilinear_sample(scalar, 0.0, 0.0) == 0.0
    assert bilinear_sample(scalar, 0.0, 1.0) == 20.0
    assert bilinear_sample(scalar, 0.5, 0.5) == 15.0
    assert bilinear_sample(scalar, 1.0, 0.5) == 20.0
    assert bilinear_sample([[1.0, math.nan], [2.0, 3.0]], 0.0, 0.0) == 1.0
    rgb = [[(0.0, 1.0, 2.0), (10.0, 11.0, 12.0)], [(20.0, 21.0, 22.0), (30.0, 31.0, 32.0)]]
    assert bilinear_sample(rgb, 0.5, 0.5) == (15.0, 16.0, 17.0)
    flow = [[(1.0, -2.0), (3.0, -4.0)], [(5.0, -6.0), (7.0, -8.0)]]
    assert bilinear_sample(flow, 0.0, 0.0) == (1.0, -2.0)
    _failure("out_of_bounds", lambda: bilinear_sample(scalar, 1.01, 0.0))
    _failure("nonfinite_value", lambda: bilinear_sample([[0.0, math.nan]], 0.5, 0.0))
    _failure("bool_value", lambda: bilinear_sample([[True]], 0.0, 0.0))
    _failure("grid_shape", lambda: bilinear_sample([[(0.0, 1.0), 2.0]], 0.0, 0.0))


def test_landmark_metrics() -> None:
    flow = [[(1.0, -1.0), (1.0, -1.0)], [(1.0, -1.0), (1.0, -1.0)]]
    landmarks = [(0.0, 1.0), (0.0, 0.0), (1.0, 1.0)]
    targets = [(1.0, 0.0), (1.0, -1.0), (2.0, 0.0)]
    result = landmark_metrics(flow, landmarks, targets, [True, True, True])
    assert result == {"landmark_median_error_px": 0.0, "landmark_p95_error_px": 0.0}
    p95 = landmark_metrics(
        [[(0.0, 0.0), (0.0, 0.0)]],
        [(0.0, 0.0), (1.0, 0.0), (0.0, 0.0)],
        [(0.0, 0.0), (1.0, 0.0), (3.0, 0.0)],
        [True, True, True],
    )
    assert p95["landmark_median_error_px"] == 0.0
    assert math.isclose(p95["landmark_p95_error_px"], 2.7)
    _failure("mask_required", lambda: landmark_metrics(flow, landmarks, targets, None))
    _failure("out_of_bounds", lambda: landmark_metrics(flow, [(-0.1, 0.0)], [(0.0, 0.0)], [True]))
    masked = landmark_metrics(flow, [None, landmarks[0]], [None, targets[0]], [False, True])
    assert masked["landmark_median_error_px"] == 0.0
    masked_nonfinite = landmark_metrics(
        flow, [(math.nan, 0.0), landmarks[0]], [(0.0, 0.0), targets[0]], [False, True]
    )
    assert masked_nonfinite["landmark_median_error_px"] == 0.0


def test_warp_and_forward_backward_residuals() -> None:
    image = [[(0.0, 1.0, 2.0), (1.0, 2.0, 3.0)], [(2.0, 3.0, 4.0), (3.0, 4.0, 5.0)]]
    identity = [[(0.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (0.0, 0.0)]]
    all_visible = [[True, True], [True, True]]
    assert visible_warp_residual(image, image, identity, all_visible) == 0.0
    shifted_image = [[(1.0, 2.0, 3.0), (2.0, 3.0, 4.0)], [(3.0, 4.0, 5.0), (4.0, 5.0, 6.0)]]
    assert visible_warp_residual(image, shifted_image, identity, all_visible) == 1.0
    shifted_source = [[(99.0, 0.0, 0.0), (0.0, 1.0, 2.0), (1.0, 2.0, 3.0)], [(99.0, 0.0, 0.0), (2.0, 3.0, 4.0), (3.0, 4.0, 5.0)]]
    right_flow = [[(1.0, 0.0), (1.0, 0.0)], [(1.0, 0.0), (1.0, 0.0)]]
    # A destination exactly on the last image2 pixel remains measurable.
    assert visible_warp_residual(image, shifted_source, right_flow, all_visible) == 0.0
    _failure("mask_required", lambda: visible_warp_residual(image, image, identity, None))
    partial_flow = [[(1.0, 0.0), (2.0, 0.0)], [(1.0, 0.0), (2.0, 0.0)]]
    assert visible_warp_residual(image, shifted_source, partial_flow, all_visible) == 0.0
    off_grid_flow = [[(3.0, 0.0), (3.0, 0.0)], [(3.0, 0.0), (3.0, 0.0)]]
    _failure("empty_mask", lambda: visible_warp_residual(image, shifted_source, off_grid_flow, all_visible))

    backward = [[(-1.0, 0.0), (-1.0, 0.0), (-1.0, 0.0)]]
    forward = [[(1.0, 0.0), (1.0, 0.0)]]
    # The second source lands exactly on backward's last pixel.
    assert forward_backward_residual_px(forward, backward, [[True, True]]) == 0.0
    assert forward_backward_residual_px(forward, [[(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]], [[True, True]]) == 1.0
    _failure("mask_required", lambda: forward_backward_residual_px(forward, backward, None))
    partial_forward = [[(1.0, 0.0), (2.0, 0.0)]]
    assert forward_backward_residual_px(partial_forward, backward, [[True, True]]) == 0.0
    off_grid_forward = [[(3.0, 0.0), (3.0, 0.0)]]
    _failure("empty_mask", lambda: forward_backward_residual_px(off_grid_forward, backward, [[True, True]]))


def test_chain_drift() -> None:
    width = 10
    mask = [[True, True] + [False] * (width - 2)]
    links = [[[(1.0, 0.0)] * width]]
    for link_count in (1, 2, 4, 8):
        truth = [[(float(link_count), 0.0)] * width]
        assert chain_drift_px(links * link_count, truth, mask, link_count) == 0.0
    wrong_truth = [[(1.0, 0.0)] * width]
    assert chain_drift_px(links * 2, wrong_truth, mask, 2) == 1.0
    _failure("link_count", lambda: chain_drift_px(links * 3, wrong_truth, mask, 3))
    _failure("link_count", lambda: chain_drift_px(links, wrong_truth, mask, 2))
    _failure("mask_required", lambda: chain_drift_px(links, wrong_truth, None, 1))
    edge_first = [[(1.0, 0.0), (1.0, 0.0)]]
    edge_second = [[(1.0, 0.0)] * 3]
    edge_truth = [[(2.0, 0.0), (2.0, 0.0)]]
    # The second source reaches edge_second's final pixel exactly.
    assert chain_drift_px([edge_first, edge_second], edge_truth, [[True, True]], 2) == 0.0
    partial_second = [[(1.0, 0.0), (1.0, 0.0)]]
    assert chain_drift_px([edge_first, partial_second], edge_truth, [[True, True]], 2) == 0.0
    off_grid_first = [[(2.0, 0.0), (2.0, 0.0)]]
    _failure(
        "empty_mask",
        lambda: chain_drift_px(
            [off_grid_first, partial_second], edge_truth, [[True, True]], 2
        ),
    )
    bad_link = [[(99.0, 0.0)] * width]
    _failure("empty_mask", lambda: chain_drift_px([bad_link, links[0]], [[(1.0, 0.0)] * width], mask, 2))


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
    test_dense_score_formula_and_boundaries()
    test_dense_macro_explicit_records_and_weighting()
    test_dense_macro_rejects_malformed_or_missing_records()
    test_shape_and_empty_failures()
    test_repeated_run_p99()
    test_bilinear_spatial_sampling()
    test_landmark_metrics()
    test_warp_and_forward_backward_residuals()
    test_chain_drift()
    test_macro_equal_weighting()
    print("P25-4 metrics tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
