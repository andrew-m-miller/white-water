"""Dependency-free frozen ``analysisGeometry-v1`` sizing helpers.

The measurement reducer and report validator both need to reproduce the host-free C++
preprocessing dimensions.  Keeping the cap algorithm here gives them one source of truth
while leaving each caller's input and error handling at its own boundary.
"""

from __future__ import annotations

import math


CAP_UNIT_PIXELS = 1_000_000
INT_MAX = 2_147_483_647
LLONG_MAX = 9_223_372_036_854_775_807


class GeometryFailure(ValueError):
    """A typed failure from the frozen geometry calculation."""

    def __init__(self, kind: str, message: str):
        self.kind = kind
        self.message = message
        super().__init__(f"{kind}: {message}")


def _fail(kind: str, message: str) -> None:
    raise GeometryFailure(kind, message)


def _rounded_dimension(value: float, axis: str) -> int:
    if not math.isfinite(value) or value <= 0.0:
        _fail("invalid_geometry", f"{axis} is not a positive finite extent")
    rounded = math.floor(value + 0.5)
    if rounded < 1:
        return 1
    if rounded > INT_MAX:
        _fail("invalid_dimension", f"{axis} rounds beyond the supported integer range")
    return int(rounded)


def analysis_dimensions(
    source_width: int,
    source_height: int,
    pixel_aspect_ratio: float,
    cap_megapixels: float,
    *,
    cap_unit_pixels: float = CAP_UNIT_PIXELS,
) -> tuple[int, int]:
    """Return the frozen analysis dimensions for one source/cap combination.

    Each axis is rounded half-up after the common canonical-area scale.  When independent
    rounding exceeds the integer cap, the axis with the larger relative rounded scale is
    clamped first; ``>=`` is intentional because the C++ path breaks ties in favour of width.
    The public callers validate their input types before invoking this numeric helper.
    """

    canonical_width = float(source_width) * float(pixel_aspect_ratio)
    canonical_height = float(source_height)
    canonical_area = canonical_width * canonical_height
    if (
        not math.isfinite(canonical_width)
        or not math.isfinite(canonical_area)
        or canonical_area <= 0.0
    ):
        _fail("invalid_geometry", "canonical analysis extent is not finite")

    cap_pixels_value = float(cap_megapixels) * cap_unit_pixels
    if cap_megapixels > 0.0 and (
        not math.isfinite(cap_pixels_value) or cap_pixels_value <= 0.0
    ):
        _fail("invalid_cap", "megapixel cap is outside the supported range")

    cap_applied = cap_megapixels > 0.0 and canonical_area > cap_pixels_value
    if cap_applied and cap_pixels_value > float(LLONG_MAX):
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
            _fail("cap_violation", "rounded analysis dimensions exceed the megapixel cap")

    return analysis_width, analysis_height


__all__ = ["CAP_UNIT_PIXELS", "GeometryFailure", "analysis_dimensions"]
