#!/usr/bin/env python3
"""Dependency-light OpenEXR sequence reading and pairing for the offline bake-off.

Production shots enter the bake-off as one-file-per-frame OpenEXR sequences (see
``docs/phase2.5-implementation-plan.md``, "Production input format"): RGB or RGBA, half or
float, constant dimensions and channel layout within a sequence, and a reference frame near the
middle so offsets through at least +/-8 can measure chain behaviour in both directions.

This module owns two independent concerns, kept apart so the well-tested core needs no optional
dependency:

* single-frame decode via the OpenEXR Python bindings (the official ``OpenEXR`` module, modern
  ``OpenEXR.File`` API), imported lazily inside :func:`frame_from_exr` so importing this module --
  and running everything below except the explicitly optional round-trip test -- never requires
  ``OpenEXR`` to be installed; and
* pure sequence/pairing logic -- expanding a corpus shot's ``path_pattern`` into frame paths,
  selecting a signed-offset reference/target pair, and checking paired-frame geometry -- that
  never touches ``OpenEXR`` and accepts an injected decoder so it is fully exercised without one.

Frames returned by :func:`frame_from_exr` use the same mapping shape as
``evaluator.frame_from_pfm``: ``width``, ``height``, ``channels`` (always 3; alpha is dropped),
``rows`` (``rows[y][x]`` is a 3-tuple of floats R, G, B), ``pixel_aspect_ratio``, ``frame``,
``sha256`` (of the raw file bytes, not the decoded pixels) and ``source``. ``rows[0]`` is the
image's **bottom** row: OpenEXR presents scanlines top-to-bottom (row zero is the data window's
top line), but this repository's row convention -- shared with ``pfm.read_pfm`` and
``synthetic.COORDINATE_CONVENTION`` ("row zero is the bottom row in PFM and row-major buffers") --
is bottom-origin, so :func:`frame_from_exr` reverses OpenEXR's scanline order before returning. Two
additional keys record source provenance without disturbing that shape: ``source_channels``
(``"RGB"`` or ``"RGBA"``, alpha dropped either way) and ``source_format`` (``"half"`` or
``"float"``, the on-disk pixel storage class).
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


def _classify_channels(channel_names: Sequence[str]) -> tuple[tuple[int, int, int], str]:
    """Return ``((r_index, g_index, b_index), "RGB"|"RGBA")``, rejecting anything else.

    Pure by design (no ``OpenEXR``): unit-tested directly with plain channel-name lists. Requires R, G
    and B to be present (dropping alpha to RGB is still the decode-time behavior), and rejects
    any channel beyond R/G/B/A -- a depth AOV, a cryptomatte layer, or anything else -- because
    the plan requires a constant RGB-or-RGBA layout within a sequence, not "whatever channels
    this file happened to carry".
    """

    names = list(channel_names)
    try:
        indices = (names.index("R"), names.index("G"), names.index("B"))
    except ValueError:
        _fail(
            "unsupported_channels",
            f"EXR does not have usable R, G and B channels (found {names!r})",
        )
    extra = set(names) - {"R", "G", "B", "A"}
    if extra:
        _fail("unsupported_channels", f"EXR has channels beyond RGB/RGBA: {names!r}")
    layout = "RGBA" if "A" in names else "RGB"
    return indices, layout


_STORAGE_NAMES = {"half": "half", "float": "float"}

# OpenEXR's modern File API returns each channel's pixels as a numpy array whose dtype IS the
# on-disk storage class: HALF -> float16, FLOAT -> float32 (UINT -> uint32). Map the numpy dtype
# name to the storage token _format_name_from_string accepts; any other dtype (e.g. "uint32")
# passes through unchanged so it is rejected there as unsupported_storage rather than silently
# accepted.
_NUMPY_PIXEL_STORAGE = {"float16": "half", "float32": "float"}


def _storage_name_from_dtype(dtype: Any) -> str:
    """Map a channel's numpy dtype to the ``"half"``/``"float"`` token the helpers expect."""

    name = str(getattr(dtype, "name", dtype))
    return _NUMPY_PIXEL_STORAGE.get(name, name)


def _format_name_from_string(type_name: str) -> str:
    """Normalize one raw pixel-storage name to ``"half"``/``"float"``, or fail.

    Pure by design (no ``OpenEXR``): takes a plain storage string -- ``"half"`` or ``"float"`` for
    the two accepted OpenEXR pixel types, or any other name (e.g. ``"uint32"``) for storage this
    bake-off rejects -- so integer/other storage rejection is directly unit-testable without the
    optional dependency installed. :func:`frame_from_exr` maps each channel's on-disk numpy dtype
    (``float16``/``float32``) to one of these names before calling in.
    """

    key = type_name.strip().lower()
    normalized = _STORAGE_NAMES.get(key)
    if normalized is None:
        _fail(
            "unsupported_storage",
            f"EXR channel pixel storage must be half or float, found {type_name!r}",
        )
    return normalized


def _source_format_from_names(type_names: Sequence[str]) -> str:
    """Reduce the R/G/B channel storage names to one shared ``"half"``/``"float"`` format.

    Pure by design (no ``OpenEXR``). Rejects mixed per-channel storage as well as any non-half/float
    storage, since the plan requires one storage class across a sequence's color channels.
    """

    ordered = list(type_names)
    normalized = {_format_name_from_string(name) for name in ordered}
    if len(normalized) != 1:
        _fail(
            "unsupported_storage",
            f"EXR R/G/B channels do not share one pixel storage format: {ordered!r}",
        )
    return normalized.pop()


def _bottom_origin_rows(rows_top_to_bottom: Sequence[Any]) -> tuple[Any, ...]:
    """Reverse OpenEXR's top-to-bottom scanline order into this repository's bottom-origin rows.

    Pure by design (no ``OpenEXR``): fed a synthetic top-to-bottom sequence with a distinctive top vs
    bottom row, this is directly unit-testable for the vertical-inversion bug -- a reader that
    forgot to reverse would return the wrong row as ``rows[0]``, flipping dy for every downstream
    landmark and metric.
    """

    return tuple(reversed(rows_top_to_bottom))


def frame_from_exr(path: Path | str, *, frame_number: int = 0, pixel_aspect_ratio: float = 1.0) -> dict[str, Any]:
    """Read one OpenEXR file into the frame mapping accepted by ``condition_and_pad_pair``.

    Imports the ``OpenEXR`` Python bindings (and numpy) lazily. RGBA files drop alpha to RGB
    (``channels`` is always 3 on success) but the original layout is recorded in
    ``source_channels``. Half and float pixel storage are both accepted -- OpenEXR's modern
    ``File`` API returns each channel's pixels as a numpy array whose dtype (``float16`` for HALF,
    ``float32`` for FLOAT) IS the on-disk storage class, recorded in ``source_format`` -- and every
    sample is widened to a Python float; any other storage (integer, etc.) or any channel set
    beyond RGB/RGBA is a typed failure. Whatever compression the file uses (ZIP, PIZ, DWA, ...) is
    handled transparently by OpenEXR -- this module never implements decompression itself. Returned
    ``rows`` are bottom-origin (see the module docstring).
    """

    try:
        import numpy  # type: ignore  # noqa: F401 (imported for its side-effect on OpenEXR arrays)
        import OpenEXR  # type: ignore
    except ImportError as exc:
        raise ExrFailure(
            "runtime_error",
            "the OpenEXR Python bindings are required to read production EXR frames and are not installed",
        ) from exc

    if not math.isfinite(pixel_aspect_ratio) or pixel_aspect_ratio <= 0.0:
        _fail("input_invalid", "pixel aspect ratio must be positive and finite")

    input_path = Path(path)
    if not input_path.is_file():
        _fail("missing_file", f"EXR file does not exist: {input_path}")

    try:
        # separate_channels=True skips OpenEXR's R/G/B(/A) grouping so each channel arrives as its
        # own 2D numpy array under its raw name, giving exact per-channel names and dtypes.
        with OpenEXR.File(str(input_path), separate_channels=True) as exr_file:
            channel_map = exr_file.channels()
            channel_names = list(channel_map.keys())
            _indices, source_channels = _classify_channels(channel_names)

            r_pixels = channel_map["R"].pixels
            g_pixels = channel_map["G"].pixels
            b_pixels = channel_map["B"].pixels

            source_format = _source_format_from_names(
                _storage_name_from_dtype(pixels.dtype) for pixels in (r_pixels, g_pixels, b_pixels)
            )

            if r_pixels.ndim != 2:
                _fail("decode_error", f"EXR {input_path} R channel is not a 2D pixel array")
            height, width = int(r_pixels.shape[0]), int(r_pixels.shape[1])
            if width <= 0 or height <= 0:
                _fail("decode_error", f"EXR {input_path} has non-positive dimensions {width}x{height}")
            for name, pixels in (("G", g_pixels), ("B", b_pixels)):
                if pixels.shape != r_pixels.shape:
                    _fail(
                        "decode_error",
                        f"EXR {input_path} R and {name} channels differ in shape: "
                        f"{tuple(r_pixels.shape)!r} vs {tuple(pixels.shape)!r}",
                    )

            # OpenEXR presents scanlines top-to-bottom (row 0 is the data window's top row). Build
            # rows in that native order first, then hand them to _bottom_origin_rows so rows[0]
            # ends up as the BOTTOM row, matching pfm.read_pfm and synthetic.COORDINATE_CONVENTION.
            rows_top_to_bottom: list[tuple[tuple[float, float, float], ...]] = []
            for y in range(height):
                r_row = r_pixels[y]
                g_row = g_pixels[y]
                b_row = b_pixels[y]
                row: list[tuple[float, float, float]] = []
                for x in range(width):
                    pixel = (float(r_row[x]), float(g_row[x]), float(b_row[x]))
                    if any(not math.isfinite(value) for value in pixel):
                        _fail("nonfinite_sample", f"EXR {input_path} contains a nonfinite decoded sample")
                    row.append(pixel)
                rows_top_to_bottom.append(tuple(row))

            rows = _bottom_origin_rows(rows_top_to_bottom)
    except ExrFailure:
        raise
    except Exception as exc:  # noqa: BLE001 - OpenEXR surfaces assorted native decode errors
        raise ExrFailure("decode_error", f"OpenEXR could not read {input_path}: {exc}") from exc

    return {
        "width": width,
        "height": height,
        "channels": 3,
        "rows": rows,
        "pixel_aspect_ratio": float(pixel_aspect_ratio),
        "frame": frame_number,
        "sha256": _sha256_file(input_path),
        "source": str(path),
        "source_channels": source_channels,
        "source_format": source_format,
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


def validate_pair_layout(first: Mapping[str, Any], second: Mapping[str, Any]) -> None:
    """Raise a typed failure unless two decoded frames share source channel layout and storage.

    Companion to :func:`validate_pair_geometry`: that function checks the numeric decode geometry
    (width/height/channels/PAR) needed for conditioning; this one checks source provenance
    (``"RGB"`` vs ``"RGBA"``, ``"half"`` vs ``"float"``) so a sequence that silently mixes channel
    layout or storage class mid-shot is caught before the mismatch reaches conditioning.
    """

    for key in ("source_channels", "source_format"):
        if first.get(key) != second.get(key):
            _fail(
                "layout_mismatch",
                f"paired EXR frames differ in {key}: {first.get(key)!r} vs {second.get(key)!r}",
            )


def validate_frame_matches_shot_metadata(shot: Mapping[str, Any], frame: Mapping[str, Any]) -> None:
    """Raise a typed failure if a decoded frame disagrees with the shot's declared metadata.

    Only checks the axes the shot record actually declares -- ``channels`` in ``{"RGB",
    "RGBA"}`` and ``bit_depth`` in ``{"half", "float"}``, per
    ``bakeoff/production-corpus-v1.template.json`` -- so a shot record without one of these
    fields is simply not checked on that axis.
    """

    declared_channels = shot.get("channels")
    if declared_channels is not None and frame.get("source_channels") != declared_channels:
        _fail(
            "metadata_mismatch",
            f"shot declares channels={declared_channels!r} but the decoded frame is "
            f"{frame.get('source_channels')!r}",
        )
    declared_depth = shot.get("bit_depth")
    if declared_depth is not None and frame.get("source_format") != declared_depth:
        _fail(
            "metadata_mismatch",
            f"shot declares bit_depth={declared_depth!r} but the decoded frame is "
            f"{frame.get('source_format')!r}",
        )


def load_pair(
    shot: Mapping[str, Any],
    offset: int,
    *,
    decoder: Callable[..., dict[str, Any]] = frame_from_exr,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve, decode and validate the reference/target EXR pair at a signed offset.

    ``decoder`` defaults to :func:`frame_from_exr` but is injected so callers (including this
    module's tests) can exercise pairing and validation without the ``OpenEXR`` bindings. It is called as
    ``decoder(path, frame_number=..., pixel_aspect_ratio=...)`` for each of the two frames, in
    reference-then-target order. Validates, in order: paired decode geometry
    (:func:`validate_pair_geometry`), paired source layout/storage
    (:func:`validate_pair_layout`), and -- for each frame independently -- agreement with the
    shot's declared ``channels``/``bit_depth`` metadata, when declared
    (:func:`validate_frame_matches_shot_metadata`).
    """

    reference_path, target_path, reference_frame, target_frame = reference_target_pair(shot, offset)
    par = shot.get("pixel_aspect_ratio", 1.0)
    first = decoder(reference_path, frame_number=reference_frame, pixel_aspect_ratio=par)
    second = decoder(target_path, frame_number=target_frame, pixel_aspect_ratio=par)
    validate_pair_geometry(first, second)
    validate_pair_layout(first, second)
    validate_frame_matches_shot_metadata(shot, first)
    validate_frame_matches_shot_metadata(shot, second)
    return first, second


__all__ = [
    "ExrFailure",
    "expand_shot_sequence",
    "frame_from_exr",
    "load_pair",
    "reference_target_pair",
    "validate_frame_matches_shot_metadata",
    "validate_pair_geometry",
    "validate_pair_layout",
]
