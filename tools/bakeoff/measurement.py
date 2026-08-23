#!/usr/bin/env python3
"""Dependency-free reducers for the Phase 2.5 measurement seam.

The offline runner measures a cell in a runtime-specific representation.  This module is the
small, deterministic boundary that turns those records into the geometry and timing objects
required by ``report-v1.schema.json``.  It intentionally does not execute inference, read
frames, or publish reports.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
import statistics
from numbers import Real
from typing import Any


class MeasurementFailure(ValueError):
    """Stable, reportable measurement-reduction failure."""

    def __init__(self, kind: str, message: str):
        self.kind = kind
        self.reason = kind
        self.failure_type = "measurement_failure"
        self.message = message
        super().__init__(f"{kind}: {message}")


PROFILES: dict[str, dict[str, int]] = {
    "smoke": {"fresh_sessions": 1, "warmups_per_session": 0, "steady_samples_per_session": 2},
    "screen": {"fresh_sessions": 1, "warmups_per_session": 0, "steady_samples_per_session": 5},
    "final": {"fresh_sessions": 3, "warmups_per_session": 1, "steady_samples_per_session": 10},
}

_CAP_UNIT_PIXELS = 1_000_000
_INT_MAX = 2_147_483_647
_LLONG_MAX = 9_223_372_036_854_775_807


def _fail(kind: str, message: str) -> None:
    raise MeasurementFailure(kind, message)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool):
        _fail("bool_value", f"{path} must be numeric, not bool")
    if not isinstance(value, Real):
        _fail("non_numeric", f"{path} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        _fail("nonfinite_value", f"{path} must be finite")
    return number


def _positive_dimension(value: Any, path: str) -> int:
    if isinstance(value, bool):
        _fail("bool_value", f"{path} must be an integer, not bool")
    if not isinstance(value, int):
        _fail("invalid_dimension", f"{path} must be a positive integer")
    if value < 1 or value > _INT_MAX:
        _fail("invalid_dimension", f"{path} must be in [1, {_INT_MAX}]")
    return value


def _rounded_dimension(value: float, path: str) -> int:
    if not math.isfinite(value) or value <= 0.0:
        _fail("invalid_geometry", f"{path} is not a positive finite extent")
    rounded = math.floor(value + 0.5)
    if rounded < 1:
        return 1
    if rounded > _INT_MAX:
        _fail("invalid_dimension", f"{path} rounds beyond the supported integer range")
    return int(rounded)


def _analysis_dimensions(
    source_width: int,
    source_height: int,
    pixel_aspect_ratio: float,
    cap_megapixels: float,
) -> tuple[int, int]:
    """Mirror ``src/core/flow/Preprocess.cpp``'s analysisGeometry-v1 sizing.

    Each axis is rounded half-up after the common canonical-area scale.  When independent
    rounding exceeds the integer cap, the axis with the larger relative rounded scale is
    clamped first, exactly as the host-free C++ preprocessing path does.
    """

    canonical_width = float(source_width) * pixel_aspect_ratio
    canonical_height = float(source_height)
    canonical_area = canonical_width * canonical_height
    if not math.isfinite(canonical_width) or not math.isfinite(canonical_area) or canonical_area <= 0.0:
        _fail("invalid_geometry", "canonical analysis extent is not finite")

    cap_pixels_value = cap_megapixels * _CAP_UNIT_PIXELS
    if cap_megapixels > 0.0 and (
        not math.isfinite(cap_pixels_value) or cap_pixels_value <= 0.0
    ):
        _fail("invalid_cap", "megapixel cap is outside the supported range")

    cap_applied = cap_megapixels > 0.0 and canonical_area > cap_pixels_value
    if cap_applied and cap_pixels_value > float(_LLONG_MAX):
        _fail("invalid_cap", "megapixel cap exceeds integer pixel accounting")

    scale = 1.0
    if cap_applied:
        scale = min(math.sqrt(cap_pixels_value / canonical_area), 1.0)
    analysis_width = _rounded_dimension(canonical_width * scale, "analysis width")
    analysis_height = _rounded_dimension(canonical_height * scale, "analysis height")

    if cap_applied:
        cap_pixels = max(1, math.floor(cap_pixels_value))
        if analysis_width * analysis_height > cap_pixels:
            rounded_scale_x = analysis_width / canonical_width
            rounded_scale_y = analysis_height / canonical_height

            def clamp_width() -> None:
                nonlocal analysis_width
                limit = cap_pixels // max(1, analysis_height)
                analysis_width = max(1, min(analysis_width, limit))

            def clamp_height() -> None:
                nonlocal analysis_height
                limit = cap_pixels // max(1, analysis_width)
                analysis_height = max(1, min(analysis_height, limit))

            if rounded_scale_x >= rounded_scale_y:
                clamp_width()
                if analysis_width * analysis_height > cap_pixels:
                    clamp_height()
            else:
                clamp_height()
                if analysis_width * analysis_height > cap_pixels:
                    clamp_width()
            if analysis_width * analysis_height > cap_pixels:
                clamp_width()
                clamp_height()

        if analysis_width * analysis_height > cap_pixels:
            # This should be unreachable for the frozen algorithm.  Keep it typed rather than
            # emitting a report that would fail the cap gate if the algorithm is edited later.
            _fail("cap_violation", "rounded analysis dimensions exceed the megapixel cap")

    return analysis_width, analysis_height


def reduce_geometry(
    source_width: Any,
    source_height: Any,
    pixel_aspect_ratio: Any,
    cap_megapixels: Any,
    padded_width: Any,
    padded_height: Any,
) -> dict[str, int | float]:
    """Return the report geometry for one source/cap/padding combination.

    ``padded_width`` and ``padded_height`` are explicit runtime tensor dimensions.  The
    protocol cap applies to the unpadded square-pixel analysis area; padding may therefore
    make ``effective_padded_megapixels`` larger than the cap.  Padding that cannot contain the
    computed analysis image is rejected.
    """

    width = _positive_dimension(source_width, "source_width")
    height = _positive_dimension(source_height, "source_height")
    par = _finite_number(pixel_aspect_ratio, "source_pixel_aspect_ratio")
    if par <= 0.0:
        _fail("invalid_aspect_ratio", "source_pixel_aspect_ratio must be positive")
    cap = _finite_number(cap_megapixels, "cap_megapixels")
    if cap < 0.0:
        _fail("invalid_cap", "cap_megapixels must be non-negative")
    padded_w = _positive_dimension(padded_width, "padded_width")
    padded_h = _positive_dimension(padded_height, "padded_height")

    analysis_width, analysis_height = _analysis_dimensions(width, height, par, cap)
    if padded_w < analysis_width or padded_h < analysis_height:
        _fail("undersized_padding", "padded dimensions must contain the analysis dimensions")

    canonical_width = float(width) * par
    canonical_height = float(height)
    spacing_x = width / analysis_width
    spacing_y = height / analysis_height
    effective_padded_megapixels = (padded_w * padded_h) / _CAP_UNIT_PIXELS
    if not all(math.isfinite(value) and value > 0.0 for value in (
        canonical_width,
        canonical_height,
        spacing_x,
        spacing_y,
        effective_padded_megapixels,
    )):
        _fail("invalid_geometry", "report geometry contains a non-finite or non-positive field")
    return {
        "source_width": width,
        "source_height": height,
        "source_pixel_aspect_ratio": par,
        "canonical_width": canonical_width,
        "canonical_height": canonical_height,
        "analysis_width": analysis_width,
        "analysis_height": analysis_height,
        "padded_width": padded_w,
        "padded_height": padded_h,
        "effective_padded_megapixels": effective_padded_megapixels,
        "spacing_x_source_pixels": spacing_x,
        "spacing_y_source_pixels": spacing_y,
    }


def _profile_counts(profile: Any) -> dict[str, int]:
    if isinstance(profile, str):
        if profile not in PROFILES:
            _fail("unknown_profile", f"unsupported timing profile {profile!r}")
        return dict(PROFILES[profile])
    if not isinstance(profile, Mapping):
        _fail("invalid_profile", "profile must be a known name or count mapping")
    expected_keys = {"fresh_sessions", "warmups_per_session", "steady_samples_per_session"}
    if set(profile) != expected_keys:
        _fail("invalid_profile", "profile count mapping has the wrong fields")
    counts: dict[str, int] = {}
    for key in expected_keys:
        value = profile[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            _fail("invalid_profile", f"profile.{key} must be a non-negative integer")
        counts[key] = value
    if counts["fresh_sessions"] < 1 or counts["steady_samples_per_session"] < 1:
        _fail("invalid_profile", "profile must require at least one session and steady sample")
    if counts["warmups_per_session"] not in (0, 1):
        _fail("invalid_profile", "profile.warmups_per_session must be 0 or 1")
    return counts


def _duration(value: Any, path: str) -> float:
    number = _finite_number(value, path)
    if number < 0.0:
        _fail("negative_duration", f"{path} must be non-negative")
    return number


def _required(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        _fail("missing_field", f"{path} is missing {key}")
    return mapping[key]


def _stage_value(
    payload: Mapping[str, Any],
    key: str,
    explicit: Any,
) -> float:
    if explicit is not None and key in payload:
        embedded = _duration(payload[key], key)
        supplied = _duration(explicit, key)
        if embedded != supplied:
            _fail("conflicting_duration", f"{key} disagrees between arguments and timing record")
        return supplied
    if explicit is not None:
        return _duration(explicit, key)
    if key not in payload:
        _fail("missing_field", f"timing record is missing {key}")
    return _duration(payload[key], key)


def reduce_timing(
    profile: Any,
    raw_sessions: Any,
    preprocessing_ms: Any = None,
    postprocessing_ms: Any = None,
) -> dict[str, Any]:
    """Aggregate raw session timings into the report schema's timing object.

    ``raw_sessions`` may be a list/tuple of session records, in which case the two pair-stage
    durations are supplied as arguments, or a mapping containing ``sessions``,
    ``preprocessing_ms`` and ``postprocessing_ms``.  Session records are never sorted: their
    indices must already be contiguous and in deterministic order.
    """

    counts = _profile_counts(profile)
    payload: Mapping[str, Any] | None = None
    if isinstance(raw_sessions, Mapping):
        payload = raw_sessions
        unknown = set(payload) - {"sessions", "preprocessing_ms", "postprocessing_ms"}
        if unknown:
            _fail("unknown_field", f"timing record has unsupported field {sorted(unknown, key=str)[0]!r}")
        sessions_value = _required(payload, "sessions", "timing")
    else:
        sessions_value = raw_sessions

    preprocessing = _stage_value(payload or {}, "preprocessing_ms", preprocessing_ms)
    postprocessing = _stage_value(payload or {}, "postprocessing_ms", postprocessing_ms)
    if not _is_sequence(sessions_value) or isinstance(sessions_value, (str, bytes)):
        _fail("invalid_sessions", "sessions must be a list or tuple")
    if len(sessions_value) != counts["fresh_sessions"]:
        _fail("fresh_session_count", "session count does not match timing profile")

    normalized_sessions: list[dict[str, Any]] = []
    steady_samples: list[float] = []
    expected_warmup = counts["warmups_per_session"] == 1
    for position, raw_session in enumerate(sessions_value):
        path = f"sessions[{position}]"
        if not isinstance(raw_session, Mapping):
            _fail("invalid_session", f"{path} must be an object")
        session_fields = {
            "session_index",
            "warmup_recorded",
            "session_creation_ms",
            "first_inference_ms",
            "steady_samples_ms",
        }
        if expected_warmup:
            session_fields.add("warmup_ms")
        unknown = set(raw_session) - session_fields
        if unknown:
            _fail("unknown_field", f"{path} has unsupported field {sorted(unknown, key=str)[0]!r}")
        session_index = _required(raw_session, "session_index", path)
        if isinstance(session_index, bool) or not isinstance(session_index, int):
            _fail("session_index_order", f"{path}.session_index must be an integer")
        if session_index != position:
            _fail("session_index_order", f"{path}.session_index must equal its ordered position")

        warmup_recorded = _required(raw_session, "warmup_recorded", path)
        if not isinstance(warmup_recorded, bool):
            _fail("warmup_count", f"{path}.warmup_recorded must be boolean")
        if warmup_recorded != expected_warmup:
            _fail("warmup_count", f"{path}.warmup_recorded does not match timing profile")

        session_creation = _duration(
            _required(raw_session, "session_creation_ms", path),
            f"{path}.session_creation_ms",
        )
        first_inference = _duration(
            _required(raw_session, "first_inference_ms", path),
            f"{path}.first_inference_ms",
        )
        steady_value = _required(raw_session, "steady_samples_ms", path)
        if not _is_sequence(steady_value) or isinstance(steady_value, (str, bytes)):
            _fail("steady_count", f"{path}.steady_samples_ms must be a list or tuple")
        if len(steady_value) != counts["steady_samples_per_session"]:
            _fail("steady_count", f"{path}.steady_samples_ms count does not match timing profile")
        session_steady = [
            _duration(value, f"{path}.steady_samples_ms[{sample_index}]")
            for sample_index, value in enumerate(steady_value)
        ]

        normalized: dict[str, Any] = {
            "session_index": session_index,
            "warmup_recorded": warmup_recorded,
            "session_creation_ms": session_creation,
            "first_inference_ms": first_inference,
            "steady_samples_ms": session_steady,
        }
        if expected_warmup:
            if "warmup_ms" not in raw_session:
                _fail("warmup_count", f"{path}.warmup_ms is required for a recorded warm-up")
            normalized["warmup_ms"] = _duration(
                raw_session["warmup_ms"], f"{path}.warmup_ms"
            )
        normalized_sessions.append(normalized)
        steady_samples.extend(session_steady)

    session_creation_median = float(statistics.median(
        session["session_creation_ms"] for session in normalized_sessions
    ))
    first_inference_median = float(statistics.median(
        session["first_inference_ms"] for session in normalized_sessions
    ))
    steady_median = float(statistics.median(steady_samples))
    return {
        "preprocessing_ms": preprocessing,
        "session_creation_ms": session_creation_median,
        "first_inference_ms": first_inference_median,
        "steady_inference_ms": steady_median,
        "postprocessing_ms": postprocessing,
        "total_pair_ms": preprocessing + steady_median + postprocessing,
        "steady_samples_ms": steady_samples,
        "sessions": normalized_sessions,
    }


# These descriptive aliases keep the seam easy to discover for callers that use the report
# vocabulary while preserving one implementation and one set of failure semantics.
analysis_geometry = reduce_geometry
timing_aggregates = reduce_timing


__all__ = [
    "MeasurementFailure",
    "PROFILES",
    "analysis_geometry",
    "reduce_geometry",
    "reduce_timing",
    "timing_aggregates",
]
