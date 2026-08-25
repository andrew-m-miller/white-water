#!/usr/bin/env python3
"""Dependency-light OpenEXR sequence reading and pairing for the offline bake-off.

Production shots enter the bake-off as one-file-per-frame OpenEXR sequences (see
``docs/phase2.5-implementation-plan.md``, "Production input format"): RGB or RGBA, half or
float, constant dimensions and channel layout within a sequence, and a reference frame near the
middle so offsets through at least +/-8 can measure chain behaviour in both directions.

This module owns two independent concerns, kept apart so the well-tested core needs no optional
dependency:

* single-frame decode via OpenImageIO, imported lazily inside :func:`frame_from_exr` so importing
  this module -- and running everything below except the explicitly optional round-trip test --
  never requires OIIO to be installed; and
* pure sequence/pairing logic -- expanding a corpus shot's ``path_pattern`` into frame paths,
  selecting a signed-offset reference/target pair, and checking paired-frame geometry -- that
  never touches OIIO and accepts an injected decoder so it is fully exercised without one.

Frames returned by :func:`frame_from_exr` use the same mapping shape as
``evaluator.frame_from_pfm``: ``width``, ``height``, ``channels`` (always 3; alpha is dropped),
``rows`` (``rows[y][x]`` is a 3-tuple of floats R, G, B), ``pixel_aspect_ratio``, ``frame``,
``sha256`` (of the raw file bytes, not the decoded pixels) and ``source``. Unlike the bottom-origin
PFM convention, ``rows[0]`` here is the EXR's native top scanline; nothing downstream (geometry
matching, pairing) depends on vertical origin, only on both frames of a pair agreeing, which a
single reader always gives.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import math
from pathlib import Path
import re
from typing import Any


_FRAME_TOKEN = re.compile(r"%[0-9]*d")
_PERCENT = re.compile(r"%")


class ExrFailure(ValueError):
    """Stable, typed EXR input/sequence failure."""

    def __init__(self, kind: str, message: str):
        self.kind = kind
        self.reason = kind
        self.failure_type = "exr_failure"
        self.message = message
        super().__init__(f"{kind}: {message}")


def _fail(kind: str, message: str) -> None:
    raise ExrFailure(kind, message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_rgb_indices(channel_names: Sequence[str]) -> tuple[int, int, int]:
    """Return the R, G, B channel indices, dropping alpha and rejecting fewer than 3 colors.

    Pure by design (no OIIO): this is the part of "RGBA drops alpha to RGB" and "reject files
    with fewer than 3 usable color channels" that can be unit-tested with a plain list of names.
    """

    names = list(channel_names)
    try:
        return (names.index("R"), names.index("G"), names.index("B"))
    except ValueError:
        _fail(
            "unsupported_channels",
            f"EXR does not have usable R, G and B channels (found {names!r})",
        )


def frame_from_exr(path: Path | str, *, frame_number: int = 0, pixel_aspect_ratio: float = 1.0) -> dict[str, Any]:
    """Read one OpenEXR file into the frame mapping accepted by ``condition_and_pad_pair``.

    Imports OpenImageIO lazily. RGBA files drop alpha to RGB (``channels`` is always 3 on
    success). Half and float pixel storage are both accepted: OIIO is asked to return float32
    regardless of on-disk storage, and whatever compression the file uses (ZIP, PIZ, DWA, ...) is
    handled transparently by OIIO -- this module never implements decompression itself.
    """

    try:
        import OpenImageIO as oiio  # type: ignore
    except ImportError as exc:
        raise ExrFailure(
            "runtime_error",
            "OpenImageIO is required to read production EXR frames and is not installed",
        ) from exc

    if not math.isfinite(pixel_aspect_ratio) or pixel_aspect_ratio <= 0.0:
        _fail("input_invalid", "pixel aspect ratio must be positive and finite")

    input_path = Path(path)
    if not input_path.is_file():
        _fail("missing_file", f"EXR file does not exist: {input_path}")

    buf = oiio.ImageBuf(str(input_path))
    if buf.has_error:
        _fail("decode_error", f"OpenImageIO could not open {input_path}: {buf.geterror()}")

    spec = buf.spec()
    width, height = int(spec.width), int(spec.height)
    if width <= 0 or height <= 0:
        _fail("decode_error", f"EXR {input_path} has non-positive dimensions {width}x{height}")

    r_index, g_index, b_index = _select_rgb_indices(list(spec.channelnames))

    pixels = buf.get_pixels(oiio.FLOAT)
    if buf.has_error:
        _fail("decode_error", f"OpenImageIO could not read pixels from {input_path}: {buf.geterror()}")

    rows: list[tuple[tuple[float, float, float], ...]] = []
    for y in range(height):
        row: list[tuple[float, float, float]] = []
        for x in range(width):
            sample = pixels[y, x]
            pixel = (float(sample[r_index]), float(sample[g_index]), float(sample[b_index]))
            if any(not math.isfinite(value) for value in pixel):
                _fail("nonfinite_sample", f"EXR {input_path} contains a nonfinite decoded sample")
            row.append(pixel)
        rows.append(tuple(row))

    return {
        "width": width,
        "height": height,
        "channels": 3,
        "rows": tuple(rows),
        "pixel_aspect_ratio": float(pixel_aspect_ratio),
        "frame": frame_number,
        "sha256": _sha256_file(input_path),
        "source": str(path),
    }


def _validate_path_pattern(pattern: str) -> None:
    tokens = _FRAME_TOKEN.findall(pattern)
    percents = _PERCENT.findall(pattern)
    if len(percents) != len(tokens):
        _fail(
            "malformed_frame_token",
            f"path pattern has a percent sign outside a single %d-style frame token: {pattern!r}",
        )
    if not tokens:
        _fail("missing_frame_token", f"path pattern has no printf integer frame token: {pattern!r}")
    if len(tokens) > 1:
        _fail(
            "multiple_frame_tokens",
            f"path pattern has more than one printf integer frame token: {pattern!r}",
        )


def _format_path(pattern: str, frame_number: int) -> str:
    try:
        return pattern % frame_number
    except (TypeError, ValueError) as exc:
        raise ExrFailure(
            "malformed_frame_token",
            f"could not format path pattern {pattern!r} with frame {frame_number}",
        ) from exc


def _int_field(shot: Mapping[str, Any], name: str) -> int:
    value = shot.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("corpus_shape", f"shot.{name} must be an integer")
    return value


def _shot_bounds(shot: Mapping[str, Any]) -> tuple[str, int, int, int]:
    """Validate and return (path_pattern, first_frame, last_frame, reference_frame)."""

    pattern = shot.get("path_pattern")
    if not isinstance(pattern, str) or not pattern:
        _fail("corpus_shape", "shot.path_pattern must be a non-empty string")
    _validate_path_pattern(pattern)

    first_frame = _int_field(shot, "first_frame")
    last_frame = _int_field(shot, "last_frame")
    reference_frame = _int_field(shot, "reference_frame")

    if first_frame > last_frame:
        _fail(
            "empty_range",
            f"shot frame range is empty: first_frame={first_frame} > last_frame={last_frame}",
        )
    if not (first_frame <= reference_frame <= last_frame):
        _fail(
            "reference_out_of_range",
            f"shot reference_frame {reference_frame} is outside [{first_frame}, {last_frame}]",
        )
    return pattern, first_frame, last_frame, reference_frame


def expand_shot_sequence(shot: Mapping[str, Any]) -> tuple[tuple[int, str], ...]:
    """Expand a corpus shot record's ``path_pattern`` into ordered ``(frame_number, path)`` pairs.

    Validates that the pattern contains exactly one printf integer frame token, that
    ``first_frame <= reference_frame <= last_frame``, and that the frame range is non-empty.
    Raises :class:`ExrFailure` on any malformed shot record.
    """

    pattern, first_frame, last_frame, _reference_frame = _shot_bounds(shot)
    return tuple((frame, _format_path(pattern, frame)) for frame in range(first_frame, last_frame + 1))


def reference_target_pair(shot: Mapping[str, Any], offset: int) -> tuple[str, str, int, int]:
    """Return ``(reference_path, target_path, reference_frame, target_frame)`` for one offset.

    ``offset`` is signed; the target frame is ``reference_frame + offset``. Raises
    :class:`ExrFailure` if the target frame falls outside ``[first_frame, last_frame]`` rather
    than letting the caller construct a path to a frame that was never captured.
    """

    if isinstance(offset, bool) or not isinstance(offset, int):
        _fail("input_invalid", "offset must be an integer")

    pattern, first_frame, last_frame, reference_frame = _shot_bounds(shot)
    target_frame = reference_frame + offset
    if not (first_frame <= target_frame <= last_frame):
        _fail(
            "target_out_of_range",
            f"target frame {target_frame} (reference {reference_frame}, offset {offset:+d}) "
            f"is outside [{first_frame}, {last_frame}]",
        )
    reference_path = _format_path(pattern, reference_frame)
    target_path = _format_path(pattern, target_frame)
    return reference_path, target_path, reference_frame, target_frame


def validate_pair_geometry(first: Mapping[str, Any], second: Mapping[str, Any]) -> None:
    """Raise a typed failure unless two decoded frames share width, height, channels and PAR.

    The runner needs constant geometry within a sequence; this is the shared check for it.
    """

    for key in ("width", "height", "channels"):
        if first.get(key) != second.get(key):
            _fail(
                "geometry_mismatch",
                f"paired EXR frames differ in {key}: {first.get(key)!r} vs {second.get(key)!r}",
            )
    if first.get("pixel_aspect_ratio") != second.get("pixel_aspect_ratio"):
        _fail(
            "geometry_mismatch",
            "paired EXR frames differ in pixel_aspect_ratio: "
            f"{first.get('pixel_aspect_ratio')!r} vs {second.get('pixel_aspect_ratio')!r}",
        )


def load_pair(
    shot: Mapping[str, Any],
    offset: int,
    *,
    decoder: Callable[..., dict[str, Any]] = frame_from_exr,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve, decode and geometry-validate the reference/target EXR pair at a signed offset.

    ``decoder`` defaults to :func:`frame_from_exr` but is injected so callers (including this
    module's tests) can exercise pairing and geometry validation without OpenImageIO. It is
    called as ``decoder(path, frame_number=..., pixel_aspect_ratio=...)`` for each of the two
    frames, in reference-then-target order.
    """

    reference_path, target_path, reference_frame, target_frame = reference_target_pair(shot, offset)
    par = shot.get("pixel_aspect_ratio", 1.0)
    first = decoder(reference_path, frame_number=reference_frame, pixel_aspect_ratio=par)
    second = decoder(target_path, frame_number=target_frame, pixel_aspect_ratio=par)
    validate_pair_geometry(first, second)
    return first, second


__all__ = [
    "ExrFailure",
    "expand_shot_sequence",
    "frame_from_exr",
    "load_pair",
    "reference_target_pair",
    "validate_pair_geometry",
]
