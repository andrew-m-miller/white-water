#!/usr/bin/env python3
"""Focused tests for the dependency-free P25-4 geometry and timing reducers."""

from __future__ import annotations

import math
from pathlib import Path

from . import measurement as measurement_module
from . import metrics as metrics_module
from .measurement import MeasurementFailure, reduce_geometry, reduce_timing
from .validator import (
    _expected_analysis_dimensions,
    load_json,
    validate_report_consistency,
)


ROOT = Path(__file__).resolve().parents[2]


def _failure(kind: str, callback) -> None:
    try:
        callback()
    except MeasurementFailure as failure:
        assert failure.kind == kind, (failure.kind, kind)
        assert failure.reason == kind
        assert failure.failure_type == "measurement_failure"
    else:
        raise AssertionError(f"expected MeasurementFailure({kind})")


def _session(
    index: int,
    *,
    warmup: bool = False,
    creation: float = 0.2,
    first: float = 1.2,
    steady: list[float] | None = None,
    warmup_ms: float = 0.5,
) -> dict:
    record = {
        "session_index": index,
        "warmup_recorded": warmup,
        "session_creation_ms": creation,
        "first_inference_ms": first,
        "steady_samples_ms": [1.0, 1.0, 1.0, 1.0, 1.0] if steady is None else steady,
    }
    if warmup:
        record["warmup_ms"] = warmup_ms
    return record


def test_geometry_rounding_par_caps_and_padding() -> None:
    odd = reduce_geometry(67, 53, 1.0, 0.0, 69, 55)
    assert odd["analysis_width"] == 67
    assert odd["analysis_height"] == 53
    assert odd["canonical_width"] == 67.0
    assert odd["spacing_x_source_pixels"] == 1.0
    assert odd["spacing_y_source_pixels"] == 1.0

    half_par = reduce_geometry(80, 64, 0.5, 0.5, 40, 64)
    assert half_par["canonical_width"] == 40.0
    assert half_par["canonical_height"] == 64.0
    assert half_par["analysis_width"] == 40
    assert half_par["analysis_height"] == 64

    # Independent half-up rounding would produce 5x4 here; the area is exactly the 20-pixel
    # cap after scaling and the explicit tensor dimensions are retained in the record.
    par_cap = reduce_geometry(3, 5, 2.0, 0.000020, 5, 4)
    assert (par_cap["analysis_width"], par_cap["analysis_height"]) == (5, 4)
    assert par_cap["effective_padded_megapixels"] == 20 / 1_000_000.0
    assert par_cap["spacing_x_source_pixels"] == 3 / 5
    assert par_cap["spacing_y_source_pixels"] == 5 / 4

    # Equal-to-cap is not reduced; just below the integer boundary exercises the correction
    # that clamps the relatively larger rounded axis first.
    exact_boundary = reduce_geometry(3, 3, 1.0, 0.000009, 3, 3)
    below_boundary = reduce_geometry(3, 3, 1.0, 0.000008, 2, 3)
    assert (exact_boundary["analysis_width"], exact_boundary["analysis_height"]) == (3, 3)
    assert (below_boundary["analysis_width"], below_boundary["analysis_height"]) == (2, 3)
    assert below_boundary["analysis_width"] * below_boundary["analysis_height"] <= 8

    # The protocol cap is on unpadded analysis area.  Padding may exceed that cap and is still
    # valid as long as it contains the analysis image.
    padded = reduce_geometry(64, 48, 1.0, 0.000001, 8, 8)
    assert (padded["analysis_width"], padded["analysis_height"]) == (1, 1)
    assert padded["effective_padded_megapixels"] == 64 / 1_000_000.0


def test_geometry_branch_parity_and_tie_breaking() -> None:
    # Exercise both correction branches.  The first case has equal rounded scales, so the
    # frozen >= tie-break must clamp width first; the second must clamp height first.
    cases = [
        (3, 3, 1.0, 0.000008, 2, 3),
        (3, 2, 0.75, 0.000003, 2, 1),
        (80, 64, 0.5, 0.5, 40, 64),
    ]
    for source_width, source_height, par, cap, padded_width, padded_height in cases:
        reduced = reduce_geometry(
            source_width,
            source_height,
            par,
            cap,
            padded_width,
            padded_height,
        )
        expected = _expected_analysis_dimensions(source_width, source_height, par, cap)
        assert (reduced["analysis_width"], reduced["analysis_height"]) == expected


def test_timing_routes_medians_through_linear_quantile() -> None:
    calls: list[tuple[list[float], float]] = []
    original_linear_quantile = metrics_module.linear_quantile

    def recording_linear_quantile(values, percentile):
        calls.append((list(values), percentile))
        return original_linear_quantile(values, percentile)

    metrics_module.linear_quantile = recording_linear_quantile
    try:
        result = reduce_timing(
            {"fresh_sessions": 2, "warmups_per_session": 0, "steady_samples_per_session": 1},
            [_session(0, creation=1.0, first=2.0, steady=[4.0]),
             _session(1, creation=3.0, first=4.0, steady=[6.0])],
            0.0,
            0.0,
        )
        assert result["session_creation_ms"] == 2.0
        assert result["first_inference_ms"] == 3.0
        assert result["steady_inference_ms"] == 5.0

        protocol = load_json(ROOT / "bakeoff/protocol-v1.json")
        corpus = load_json(ROOT / "bakeoff/fixtures/positive/corpus-v1.json")
        report_schema = load_json(ROOT / "bakeoff/report-v1.schema.json")
        corpus_schema = load_json(ROOT / "bakeoff/corpus-v1.schema.json")
        report = load_json(ROOT / "bakeoff/fixtures/positive/report-v1.json")
        validate_report_consistency(report, protocol, report_schema, corpus, corpus_schema)
    finally:
        metrics_module.linear_quantile = original_linear_quantile

    assert [percentile for _, percentile in calls] == [0.5] * 6
    assert calls[0][0] == [1.0, 3.0]
    assert calls[1][0] == [2.0, 4.0]
    assert calls[2][0] == [4.0, 6.0]


def test_geometry_rejects_invalid_and_undersized_inputs() -> None:
    _failure("bool_value", lambda: reduce_geometry(True, 48, 1.0, 0.5, 64, 48))
    _failure("bool_value", lambda: reduce_geometry(64, 48, False, 0.5, 64, 48))
    _failure("nonfinite_value", lambda: reduce_geometry(64, 48, math.nan, 0.5, 64, 48))
    _failure("nonfinite_value", lambda: reduce_geometry(64, 48, 1.0, math.inf, 64, 48))
    _failure("invalid_dimension", lambda: reduce_geometry(0, 48, 1.0, 0.5, 64, 48))
    _failure("invalid_dimension", lambda: reduce_geometry(64.0, 48, 1.0, 0.5, 64, 48))
    _failure("invalid_aspect_ratio", lambda: reduce_geometry(64, 48, 0.0, 0.5, 64, 48))
    _failure("invalid_cap", lambda: reduce_geometry(64, 48, 1.0, -0.5, 64, 48))
    _failure("undersized_padding", lambda: reduce_geometry(64, 48, 1.0, 0.5, 63, 48))
    _failure("undersized_padding", lambda: reduce_geometry(64, 48, 1.0, 0.5, 64, 47))

    # The real constant cannot underflow when multiplied by 1e6 for a representable positive
    # Python float, so force that C++ guard's defensive branch to make it executable here.
    original_unit = measurement_module._CAP_UNIT_PIXELS
    measurement_module._CAP_UNIT_PIXELS = 0.0
    try:
        _failure("invalid_cap", lambda: reduce_geometry(64, 48, 1.0, 0.5, 64, 48))
    finally:
        measurement_module._CAP_UNIT_PIXELS = original_unit


def test_timing_all_profiles_and_medians() -> None:
    smoke = reduce_timing(
        "smoke",
        {"preprocessing_ms": 0.1, "postprocessing_ms": 0.2,
         "sessions": [_session(0, steady=[1.0, 3.0])]},
    )
    assert smoke["steady_samples_ms"] == [1.0, 3.0]
    assert smoke["steady_inference_ms"] == 2.0
    assert math.isclose(smoke["total_pair_ms"], 2.3, rel_tol=0.0, abs_tol=1e-12)

    screen = reduce_timing(
        "screen",
        [_session(0, steady=[1.0, 1.0, 1.1, 1.0, 1.0])],
        0.1,
        0.1,
    )
    assert screen["steady_inference_ms"] == 1.0
    assert math.isclose(screen["total_pair_ms"], 1.2, rel_tol=0.0, abs_tol=1e-12)

    final = reduce_timing(
        "final",
        {
            "preprocessing_ms": 1.0,
            "postprocessing_ms": 2.0,
            "sessions": [
                _session(index, warmup=True, creation=float(1 + index * 2),
                         first=float(2 + index * 2),
                         steady=[float(index * 10 + sample) for sample in range(10)])
                for index in range(3)
            ],
        },
    )
    assert final["session_creation_ms"] == 3.0
    assert final["first_inference_ms"] == 4.0
    assert final["steady_samples_ms"] == [float(value) for value in range(30)]
    assert final["steady_inference_ms"] == 14.5  # even-count median of 30 flattened samples
    assert final["total_pair_ms"] == 17.5
    assert all(session["warmup_recorded"] for session in final["sessions"])
    assert all("warmup_ms" in session for session in final["sessions"])

    # The production profiles have an odd session count, but the reducer also handles an
    # even-count session median deterministically for a caller-provided profile mapping.
    even_sessions = reduce_timing(
        {"fresh_sessions": 2, "warmups_per_session": 0, "steady_samples_per_session": 1},
        [_session(0, creation=1.0, first=2.0, steady=[4.0]),
         _session(1, creation=3.0, first=4.0, steady=[6.0])],
        0.0,
        0.0,
    )
    assert even_sessions["session_creation_ms"] == 2.0
    assert even_sessions["first_inference_ms"] == 3.0
    assert even_sessions["steady_inference_ms"] == 5.0


def test_timing_rejects_malformed_nonfinite_counts_and_order() -> None:
    valid_screen = [_session(0)]
    _failure("unknown_profile", lambda: reduce_timing("unknown", valid_screen, 0.0, 0.0))
    _failure("fresh_session_count", lambda: reduce_timing("final", valid_screen, 0.0, 0.0))
    wrong_warmup = _session(0, warmup=True)
    _failure("unknown_field", lambda: reduce_timing("screen", [wrong_warmup], 0.0, 0.0))
    missing_warmup = _session(0, warmup=True, steady=[1.0] * 10)
    missing_warmup.pop("warmup_ms")
    _failure("warmup_count", lambda: reduce_timing("final", [missing_warmup] * 3, 0.0, 0.0))
    _failure("steady_count", lambda: reduce_timing("screen", [_session(0, steady=[1.0])], 0.0, 0.0))
    _failure("session_index_order", lambda: reduce_timing("final", [_session(1, warmup=True), _session(0, warmup=True), _session(2, warmup=True)], 0.0, 0.0))

    nan_stage = [_session(0)]
    _failure("nonfinite_value", lambda: reduce_timing("screen", nan_stage, math.nan, 0.0))
    negative = [_session(0, steady=[-1.0, 1.0, 1.0, 1.0, 1.0])]
    _failure("negative_duration", lambda: reduce_timing("screen", negative, 0.0, 0.0))
    boolean = [_session(0, creation=True)]
    _failure("bool_value", lambda: reduce_timing("screen", boolean, 0.0, 0.0))
    nonfinite_sample = [_session(0, steady=[1.0, math.inf, 1.0, 1.0, 1.0])]
    _failure("nonfinite_value", lambda: reduce_timing("screen", nonfinite_sample, 0.0, 0.0))
    misspelled_stage = {"sessions": valid_screen, "preprocesing_ms": 0.0, "postprocessing_ms": 0.0}
    _failure("unknown_field", lambda: reduce_timing("screen", misspelled_stage))
    extra_session_field = _session(0)
    extra_session_field["steady_sample_ms"] = extra_session_field.pop("steady_samples_ms")
    _failure("unknown_field", lambda: reduce_timing("screen", [extra_session_field], 0.0, 0.0))
    _failure("missing_field", lambda: reduce_timing("screen", valid_screen))


def test_reducers_bind_to_positive_report_fixture() -> None:
    protocol = load_json(ROOT / "bakeoff/protocol-v1.json")
    corpus = load_json(ROOT / "bakeoff/fixtures/positive/corpus-v1.json")
    report_schema = load_json(ROOT / "bakeoff/report-v1.schema.json")
    corpus_schema = load_json(ROOT / "bakeoff/corpus-v1.schema.json")
    report = load_json(ROOT / "bakeoff/fixtures/positive/report-v1.json")
    result = report["results"][0]
    result["geometry"] = reduce_geometry(64, 48, 1.0, 0.5, 64, 48)
    fixture_timing = report["results"][0]["timing"]
    result["timing"] = reduce_timing(
        "screen",
        {
            "preprocessing_ms": fixture_timing["preprocessing_ms"],
            "postprocessing_ms": fixture_timing["postprocessing_ms"],
            "sessions": fixture_timing["sessions"],
        },
    )
    validate_report_consistency(report, protocol, report_schema, corpus, corpus_schema)


def main() -> int:
    test_geometry_rounding_par_caps_and_padding()
    test_geometry_branch_parity_and_tie_breaking()
    test_geometry_rejects_invalid_and_undersized_inputs()
    test_timing_all_profiles_and_medians()
    test_timing_routes_medians_through_linear_quantile()
    test_timing_rejects_malformed_nonfinite_counts_and_order()
    test_reducers_bind_to_positive_report_fixture()
    print("P25-4 measurement reducer tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
