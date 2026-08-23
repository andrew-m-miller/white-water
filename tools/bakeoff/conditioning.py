#!/usr/bin/env python3
"""The frozen Phase 2.5 input-conditioning formulas.

This module deliberately has no image/runtime dependency.  The bake-off runner can
adapt its own image container to :func:`condition_pair`, while the formulas remain
testable on the air-gapped machine with the Python standard library alone.

The returned values are the values *before* an artifact's declared tensor packing.
Packing is a P25-1 manifest concern and is intentionally not represented here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Callable, Iterable


SUPPORTED_CONDITIONING = (
    "native-clamp01-v1",
    "signed-log-v1",
    "pair-percentile-v1",
    "native-log-v1",
)

PAIR_PERCENTILE_EPSILON = 1.0e-6
SIGNED_LOG_FULL_SCALE = 16.0


class ConditioningFailure(ValueError):
    """A typed, reportable conditioning failure.

    ``kind`` is stable report vocabulary.  The runner should map it to the report
    schema's ``conditioning_failure`` reason without parsing an exception string.
    """

    def __init__(self, kind: str, message: str):
        self.kind = kind
        self.reason = kind
        self.failure_type = "conditioning_failure"
        self.message = message
        super().__init__(f"{kind}: {message}")


@dataclass(frozen=True)
class PairConditioning:
    """Conditioned pair and the shared parameters used to produce it."""

    token: str
    first: Any
    second: Any
    low: float | None = None
    high: float | None = None
    epsilon: float | None = None

    @property
    def parameters(self) -> dict[str, float]:
        if self.token != "pair-percentile-v1":
            return {}
        # The report validator requires all three fields and the exact epsilon.
        assert self.low is not None and self.high is not None and self.epsilon is not None
        return {"low": self.low, "high": self.high, "epsilon": self.epsilon}


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _map_values(value: Any, function: Callable[[float], float]) -> Any:
    """Apply ``function`` recursively while retaining list/tuple structure."""

    if _is_sequence(value):
        mapped = [_map_values(child, function) for child in value]
        return tuple(mapped) if isinstance(value, tuple) else mapped
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("conditioning inputs must contain numeric values")
    return function(float(value))


def _has_nonfinite(value: Any) -> bool:
    if _is_sequence(value):
        return any(_has_nonfinite(child) for child in value)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("conditioning inputs must contain numeric values")
    return not math.isfinite(float(value))


def _rgb_samples(value: Any, channels: int) -> Iterable[float]:
    """Yield RGB values from flat or pixel-nested input.

    A runner normally passes ``rows -> pixels -> channels``.  Flat numeric input is
    accepted for small formula tests and is grouped using ``channels``.  Alpha is
    intentionally ignored when ``channels`` is four, matching the protocol's RGB
    quantile scope.
    """

    if channels < 3:
        raise ValueError("conditioning channel count must be at least three")
    if not _is_sequence(value):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("conditioning inputs must contain numeric values")
        yield float(value)
        return

    if all(not _is_sequence(child) for child in value):
        flat = list(value)
        if not flat:
            return
        if len(flat) % channels == 0 and len(flat) != 3:
            for start in range(0, len(flat), channels):
                pixel = flat[start:start + channels]
                for child in pixel[:3]:
                    if isinstance(child, bool) or not isinstance(child, Real):
                        raise TypeError("conditioning inputs must contain numeric values")
                    yield float(child)
            return
        for child in flat[:3]:
            if isinstance(child, bool) or not isinstance(child, Real):
                raise TypeError("conditioning inputs must contain numeric values")
            yield float(child)
        return

    for child in value:
        yield from _rgb_samples(child, channels)


def linear_quantile(values: Iterable[float], percentile: float) -> float:
    """The protocol's deterministic linear quantile (``h=(n-1)*p``)."""

    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile must be in [0, 1]")
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        raise ConditioningFailure("empty_pair", "pair has no finite RGB samples")
    height = (len(ordered) - 1) * percentile
    lower = math.floor(height)
    upper = math.ceil(height)
    fraction = height - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _reject_nonfinite(first: Any, second: Any) -> None:
    if _has_nonfinite(first) or _has_nonfinite(second):
        raise ConditioningFailure(
            "nonfinite_input",
            "conditioning input contains NaN or infinity",
        )


def _native_clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def signed_log(value: float) -> float:
    """Apply the frozen signed/log formula to one finite value."""

    if not math.isfinite(float(value)):
        raise ConditioningFailure("nonfinite_input", "signed-log input is not finite")
    number = float(value)
    sign = 0.0 if number == 0.0 else (1.0 if number > 0.0 else -1.0)
    compressed = 0.5 + sign * math.log1p(abs(number)) / (2.0 * math.log(17.0))
    return min(1.0, max(0.0, compressed))


def condition_pair(
    first: Any,
    second: Any,
    token: str,
    *,
    channels: int = 3,
) -> PairConditioning:
    """Condition an image pair according to one frozen protocol token.

    All four formulas reject nonfinite source values with the typed
    ``nonfinite_input`` failure.  Pair-percentile still computes its bounds from
    the finite RGB sample stream as the protocol states; the rejection prevents a
    NaN/Inf from reaching the model tensor after those bounds are found.
    """

    if token not in SUPPORTED_CONDITIONING:
        raise ValueError(f"unknown conditioning token: {token}")
    _reject_nonfinite(first, second)

    if token == "native-clamp01-v1":
        return PairConditioning(
            token, _map_values(first, _native_clamp), _map_values(second, _native_clamp)
        )

    if token == "signed-log-v1":
        return PairConditioning(
            token, _map_values(first, signed_log), _map_values(second, signed_log)
        )

    if token == "native-log-v1":
        # The model's packing is deliberately outside this module.  Returning a
        # float copy makes the no-op explicit and avoids aliasing mutable inputs.
        return PairConditioning(
            token, _map_values(first, lambda value: float(value)),
            _map_values(second, lambda value: float(value)),
        )

    samples = list(_rgb_samples(first, channels)) + list(_rgb_samples(second, channels))
    finite_samples = [value for value in samples if math.isfinite(value)]
    if not finite_samples:
        raise ConditioningFailure("empty_pair", "pair has no finite RGB samples")
    low = linear_quantile(finite_samples, 0.01)
    high = linear_quantile(finite_samples, 0.99)
    denominator = max(high - low, PAIR_PERCENTILE_EPSILON)

    def normalize(value: float) -> float:
        return min(1.0, max(0.0, (value - low) / denominator))

    return PairConditioning(
        token,
        _map_values(first, normalize),
        _map_values(second, normalize),
        low=low,
        high=high,
        epsilon=PAIR_PERCENTILE_EPSILON,
    )


def apply_conditioning(value: Any, token: str, *, low: float | None = None,
                       high: float | None = None) -> Any:
    """Apply one formula to a single image when pair bounds are already known.

    This helper is useful at the runner seam: pair-percentile bounds must be computed
    once from both frames, then applied to each frame with the same ``low``/``high``.
    """

    if token == "pair-percentile-v1":
        if low is None or high is None or not math.isfinite(low) or not math.isfinite(high):
            raise ConditioningFailure("incomplete_parameters", "pair-percentile needs finite low/high")
        if high <= low:
            raise ConditioningFailure("incomplete_parameters", "pair-percentile high must exceed low")
        denominator = max(high - low, PAIR_PERCENTILE_EPSILON)
        _reject_nonfinite(value, value)
        if not list(_rgb_samples(value, channels=3)):
            raise ConditioningFailure("empty_pair", "pair has no finite RGB samples")
        return _map_values(value, lambda sample: min(1.0, max(0.0, (sample - low) / denominator)))
    # Route the one-frame cases through the pair implementation so validation and
    # nonfinite behavior cannot drift between code paths.
    return condition_pair(value, value, token).first


__all__ = [
    "SUPPORTED_CONDITIONING",
    "PAIR_PERCENTILE_EPSILON",
    "SIGNED_LOG_FULL_SCALE",
    "ConditioningFailure",
    "PairConditioning",
    "linear_quantile",
    "signed_log",
    "condition_pair",
    "apply_conditioning",
]
