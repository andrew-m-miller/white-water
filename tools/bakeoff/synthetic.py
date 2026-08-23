#!/usr/bin/env python3
"""Deterministic Phase 2.5 synthetic corpus and analytic-truth generator.

The generator is intentionally standard-library-only.  It can be used in two modes:

* metadata-only (the default), which is cheap and includes the exact FHD/UHD cases;
* frame emission, which writes dependency-free RGB PFM sequences and a JSON analytic
  truth sidecar for each case.

The metadata paths use ``generated://`` so the corpus never pretends that generated
frames are production footage.  A runner adapter can map that URI to the emitted
directory without changing the frozen corpus schema.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

COORDINATE_CONVENTION = (
    "x right, y down; full-resolution real-pixel coordinates at pixel centres; "
    "pair displacement is image1-to-image2"
)

REQUIRED_SYNTHETIC_CASES = (
    "identity",
    "translation-x-positive",
    "translation-x-negative",
    "translation-y-positive",
    "translation-y-negative",
    "affine",
    "spatial",
    "border",
    "occlusion-reveal",
    "blur",
    "noise",
    "hdr-scene-linear",
    "log-input",
    "odd-size",
    "asymmetric-padding",
    "par-0_5",
    "par-2",
    "chain-1",
    "chain-2",
    "chain-4",
    "chain-8",
    "fhd-1920x1080-par1",
    "uhd-3840x2160-par1",
)


@dataclass(frozen=True)
class SyntheticCase:
    case_id: str
    category: str
    width: int
    height: int
    pixel_aspect_ratio: float = 1.0
    encoding: str = "scene-linear"
    channels: str = "RGB"
    bit_depth: str = "float"
    first_frame: int = 0
    last_frame: int = 8
    reference_frame: int = 4
    chain_length: int | None = None
    seed: int = 1
    parameters: Mapping[str, Any] | None = None

    @property
    def frame_numbers(self) -> range:
        return range(self.first_frame, self.last_frame + 1)

    @property
    def path_token(self) -> str:
        return self.case_id


def _case(
    case_id: str,
    category: str,
    width: int,
    height: int,
    *,
    par: float = 1.0,
    encoding: str = "scene-linear",
    channels: str = "RGB",
    bit_depth: str = "float",
    first: int = 0,
    last: int = 8,
    reference: int = 4,
    chain_length: int | None = None,
    seed: int = 1,
    parameters: Mapping[str, Any] | None = None,
) -> SyntheticCase:
    return SyntheticCase(
        case_id,
        category,
        width,
        height,
        par,
        encoding,
        channels,
        bit_depth,
        first,
        last,
        reference,
        chain_length,
        seed,
        parameters,
    )


def all_cases() -> tuple[SyntheticCase, ...]:
    """Return the exact frozen required-case set in protocol order."""

    return (
        _case("identity", "identity", 64, 48, seed=11),
        _case("translation-x-positive", "translation-x", 65, 49, seed=13,
              parameters={"shift_x_per_link": 1.25}),
        _case("translation-x-negative", "translation-x", 65, 49, seed=17,
              parameters={"shift_x_per_link": -1.25}),
        _case("translation-y-positive", "translation-y", 65, 49, seed=19,
              parameters={"shift_y_per_link": 1.5}),
        _case("translation-y-negative", "translation-y", 65, 49, seed=23,
              parameters={"shift_y_per_link": -1.5}),
        _case("affine", "affine-spatial", 80, 64, seed=29,
              parameters={"field": "affine", "scale_x_per_link": 0.02,
                          "scale_y_per_link": -0.015}),
        _case("spatial", "affine-spatial", 80, 64, seed=31,
              parameters={"field": "sinusoidal", "amplitude": 3.0}),
        _case("border", "border", 80, 64, seed=37,
              parameters={"shift_x_per_link": 6.0, "shift_y_per_link": 4.0,
                          "edge": "replicate"}),
        _case("occlusion-reveal", "occlusion-reveal", 96, 64, seed=41,
              parameters={"foreground": "moving_rectangle", "visible_mask": "analytic"}),
        _case("blur", "blur", 96, 64, seed=43,
              parameters={"kernel": "three_sample_motion", "radius": 1.0}),
        _case("noise", "noise", 96, 64, seed=47,
              parameters={"distribution": "uniform", "amplitude": 0.025,
                          "seed": 4701}),
        _case("hdr-scene-linear", "hdr-log", 96, 64, seed=53,
              parameters={"range": "0..16", "encoding": "scene-linear"}),
        _case("log-input", "hdr-log", 96, 64, encoding="log", seed=59,
              parameters={"range": "log1p(0..16)", "encoding": "log"}),
        _case("odd-size", "odd-padding", 67, 53, seed=61,
              parameters={"extent": "odd"}),
        _case("asymmetric-padding", "odd-padding", 67, 53, seed=67,
              parameters={
                  "padding": {
                      "left": 1,
                      "right": 3,
                      "bottom": 2,
                      "top": 2,
                      "policy": "replication",
                      "multiple": 8,
                  },
              }),
        _case("par-0_5", "par", 80, 64, par=0.5, channels="RGBA",
              bit_depth="half", seed=71, parameters={"par": 0.5}),
        _case("par-2", "par", 80, 64, par=2.0, channels="RGBA",
              bit_depth="half", seed=73, parameters={"par": 2.0}),
        _case("chain-1", "chain", 96, 64, first=0, last=16, reference=8,
              chain_length=1, seed=79, parameters={"links": 1, "shift_x_per_link": 1.25}),
        _case("chain-2", "chain", 96, 64, first=0, last=16, reference=8,
              chain_length=2, seed=83, parameters={"links": 2, "shift_x_per_link": 1.25}),
        _case("chain-4", "chain", 96, 64, first=0, last=16, reference=8,
              chain_length=4, seed=89, parameters={"links": 4, "shift_x_per_link": 1.25}),
        _case("chain-8", "chain", 96, 64, first=0, last=16, reference=8,
              chain_length=8, seed=97, parameters={"links": 8, "shift_x_per_link": 1.25}),
        _case("fhd-1920x1080-par1", "identity", 1920, 1080, seed=101,
              parameters={"target": "fhd"}),
        _case("uhd-3840x2160-par1", "identity", 3840, 2160, seed=103,
              parameters={"target": "uhd"}),
    )


def case_map() -> dict[str, SyntheticCase]:
    cases = all_cases()
    result = {case.case_id: case for case in cases}
    if tuple(result) != REQUIRED_SYNTHETIC_CASES:
        raise AssertionError("synthetic case order diverges from protocol-v1")
    return result


def _case_for(case_or_id: SyntheticCase | str) -> SyntheticCase:
    if isinstance(case_or_id, SyntheticCase):
        return case_or_id
    try:
        return case_map()[case_or_id]
    except KeyError as exc:
        raise KeyError(f"unknown synthetic case: {case_or_id}") from exc


def _base_pixel(case: SyntheticCase, x: float, y: float) -> tuple[float, float, float]:
    """A deterministic, high-frequency-enough RGB plate."""

    seed = float(case.seed)
    red = 0.5 + 0.23 * math.sin((x + seed * 0.31) * 0.19) + 0.11 * math.cos((y - seed) * 0.13)
    green = 0.5 + 0.21 * math.cos((x - seed * 0.17) * 0.11) + 0.13 * math.sin((y + seed) * 0.23)
    blue = 0.5 + 0.17 * math.sin((x + y + seed) * 0.07) + 0.16 * math.cos((x - y) * 0.17)
    return red, green, blue


def _clamp_coordinate(value: float, limit: int) -> float:
    return min(float(limit - 1), max(0.0, value))


def _sample_base(case: SyntheticCase, x: float, y: float) -> tuple[float, float, float]:
    x = _clamp_coordinate(x, case.width)
    y = _clamp_coordinate(y, case.height)
    x0, y0 = math.floor(x), math.floor(y)
    x1, y1 = min(case.width - 1, x0 + 1), min(case.height - 1, y0 + 1)
    fx, fy = x - x0, y - y0
    p00 = _base_pixel(case, x0, y0)
    p10 = _base_pixel(case, x1, y0)
    p01 = _base_pixel(case, x0, y1)
    p11 = _base_pixel(case, x1, y1)
    return tuple(
        (p00[channel] + (p10[channel] - p00[channel]) * fx) * (1.0 - fy)
        + (p01[channel] + (p11[channel] - p01[channel]) * fx) * fy
        for channel in range(3)
    )


def _motion(case: SyntheticCase, frame: int, x: float, y: float) -> tuple[float, float]:
    dt = frame - case.reference_frame
    center_x = (case.width - 1) * 0.5
    center_y = (case.height - 1) * 0.5
    if case.case_id == "translation-x-positive":
        return 1.25 * dt, 0.0
    if case.case_id == "translation-x-negative":
        return -1.25 * dt, 0.0
    if case.case_id == "translation-y-positive":
        return 0.0, 1.5 * dt
    if case.case_id == "translation-y-negative":
        return 0.0, -1.5 * dt
    if case.case_id == "affine":
        return (
            dt * (1.0 + 0.02 * (x - center_x) + 0.01 * (y - center_y)),
            dt * (-0.75 - 0.008 * (x - center_x) + 0.015 * (y - center_y)),
        )
    if case.case_id == "spatial":
        return (
            dt * (2.5 * math.sin(y * 0.13) + 0.5 * math.cos(x * 0.07)),
            dt * (2.0 * math.cos(x * 0.11) - 0.5 * math.sin(y * 0.09)),
        )
    if case.case_id == "border":
        return 6.0 * dt, 4.0 * dt
    if case.case_id.startswith("chain-"):
        return 1.25 * dt, 0.0
    if case.case_id in {"odd-size", "asymmetric-padding", "par-0_5", "par-2"}:
        return 0.75 * dt, -0.5 * dt
    if case.case_id == "blur":
        return 1.5 * dt, 0.25 * dt
    if case.case_id in {"occlusion-reveal", "noise", "hdr-scene-linear", "log-input"}:
        return 1.0 * dt, 0.5 * dt
    return 0.0, 0.0


def analytic_displacement(
    case_or_id: SyntheticCase | str,
    from_frame: int,
    to_frame: int,
    x: float,
    y: float,
) -> tuple[float, float]:
    """Return deterministic dense truth in source-pixel units.

    Synthetic motion is defined as a frame-indexed displacement from a common
    reference plate.  The exact pair field is therefore the difference of the two
    frame fields; this remains analytic for affine and spatial cases and makes chain
    truth explicit rather than relying on a model output.
    """

    case = _case_for(case_or_id)
    from_motion = _motion(case, from_frame, x, y)
    to_motion = _motion(case, to_frame, x, y)
    return to_motion[0] - from_motion[0], to_motion[1] - from_motion[1]


def _noise(case: SyntheticCase, frame: int, x: int, y: int) -> float:
    # A deterministic integer hash avoids dependence on Python's randomized hash().
    value = (case.seed * 1103515245 + frame * 12345 + x * 2654435761 + y * 40503) & 0xFFFFFFFF
    value ^= value >> 16
    return (value / 4294967295.0 - 0.5) * 0.05


def _foreground(case: SyntheticCase, frame: int, x: float, y: float) -> tuple[float, float, float] | None:
    dt = frame - case.reference_frame
    left = 25.0 + dt * 1.0
    right = left + 18.0
    bottom = 18.0 + dt * 0.5
    top = bottom + 20.0
    if left <= x < right and bottom <= y < top:
        return (0.92, 0.17, 0.08)
    return None


def frame_rows(case_or_id: SyntheticCase | str, frame: int) -> Iterator[tuple[tuple[float, float, float], ...]]:
    """Yield one bottom-left row of a deterministic RGB PFM frame."""

    case = _case_for(case_or_id)
    if frame not in case.frame_numbers:
        raise ValueError(f"frame {frame} is outside {case.case_id} range")
    for y in range(case.height):
        row: list[tuple[float, float, float]] = []
        for x in range(case.width):
            dx, dy = _motion(case, frame, x, y)
            if case.case_id == "blur":
                samples = [
                    _sample_base(case, x - dx + offset, y - dy)
                    for offset in (-1.0, 0.0, 1.0)
                ]
                rgb = tuple(sum(sample[channel] for sample in samples) / 3.0 for channel in range(3))
            else:
                rgb = _sample_base(case, x - dx, y - dy)
            foreground = _foreground(case, frame, x, y) if case.case_id == "occlusion-reveal" else None
            if foreground is not None:
                rgb = foreground
            if case.case_id == "noise":
                amount = _noise(case, frame, x, y)
                rgb = tuple(channel + amount for channel in rgb)
            if case.case_id == "hdr-scene-linear":
                rgb = tuple(channel * 16.0 for channel in rgb)
            elif case.case_id == "log-input":
                rgb = tuple(math.log1p(max(0.0, channel) * 16.0) / math.log(17.0) for channel in rgb)
            row.append(rgb)
        yield tuple(row)


def generate_frame(case_or_id: SyntheticCase | str, frame: int) -> tuple[tuple[tuple[float, float, float], ...], ...]:
    return tuple(frame_rows(case_or_id, frame))


def truth_document(case_or_id: SyntheticCase | str) -> dict[str, Any]:
    case = _case_for(case_or_id)
    document: dict[str, Any] = {
        "schema_version": 1,
        "case_id": case.case_id,
        "coordinate_convention": COORDINATE_CONVENTION,
        "reference_frame": case.reference_frame,
        "frame_range": {"first": case.first_frame, "last": case.last_frame},
        "pair_field": "motion(to_frame,x,y)-motion(from_frame,x,y)",
        "motion": {
            "case": case.case_id,
            "analytic": True,
            "parameters": dict(case.parameters or {}),
        },
    }
    if case.chain_length is not None:
        document["chain"] = {
            "length": case.chain_length,
            "per_link": "1.25 px right",
            "truth": "sum of link fields equals analytic_displacement",
        }
    if case.case_id == "occlusion-reveal":
        document["visibility"] = "moving rectangle mask from _foreground(case,frame,x,y)"
    if case.case_id == "asymmetric-padding":
        document["padding"] = {
            "policy": "replication",
            "comparison_seam": "padding.pad_rows(policy=manifest.declared_policy)",
        }
    return document


def synthetic_shot(case_or_id: SyntheticCase | str) -> dict[str, Any]:
    case = _case_for(case_or_id)
    shot: dict[str, Any] = {
        "id": "syn-" + case.case_id,
        "case_id": case.case_id,
        "path_pattern": f"generated://{case.path_token}/frame.%04d.pfm",
        "first_frame": case.first_frame,
        "last_frame": case.last_frame,
        "reference_frame": case.reference_frame,
        "width": case.width,
        "height": case.height,
        "pixel_aspect_ratio": case.pixel_aspect_ratio,
        "encoding": case.encoding,
        "channels": case.channels,
        "bit_depth": case.bit_depth,
        "categories": [case.category],
        "truth": {
            "kind": "analytic",
            "definition": "frame-indexed analytic displacement field; see truth sidecar",
            "path": f"generated://{case.path_token}/truth.json",
            "coordinate_convention": COORDINATE_CONVENTION,
        },
    }
    if case.chain_length is not None:
        shot["chain_length"] = case.chain_length
    if case.parameters is not None:
        shot["generator_parameters"] = dict(case.parameters)
    return shot


def synthetic_partition() -> dict[str, Any]:
    return {
        "id": "synthetic",
        "kind": "synthetic",
        "description": "Deterministic generated PFM cases with analytic displacement truth.",
        "generator": "whitewater-synthetic-v1",
        "shots": [synthetic_shot(case) for case in all_cases()],
    }


def write_pfm(path: Path, rows: Iterable[Sequence[Sequence[float]]], width: int, height: int) -> None:
    """Write an RGB little-endian PFM without retaining the whole frame."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(f"PF\n{width} {height}\n-1.0\n".encode("ascii"))
        row_count = 0
        for row in rows:
            if len(row) != width:
                raise ValueError("synthetic row has the wrong width")
            encoded = array("f")
            for pixel in row:
                if len(pixel) != 3:
                    raise ValueError("synthetic PFM output is RGB")
                encoded.extend(float(channel) for channel in pixel)
            stream.write(encoded.tobytes())
            row_count += 1
        if row_count != height:
            raise ValueError("synthetic frame has the wrong height")


def write_case_frames(case_or_id: SyntheticCase | str, output_dir: Path, *, include_large: bool = False) -> None:
    """Emit PFM frames and analytic truth for one case.

    FHD/UHD output is opt-in because a metadata-only corpus must stay cheap to
    generate and review.  The exact same lazy generator is used when those targets
    are requested.
    """

    case = _case_for(case_or_id)
    # Keep both frozen performance targets metadata-only by default.  FHD is already
    # roughly two million pixels and nine RGB PFM frames are a sizeable fixture; the
    # explicit flag is the opt-in for either target.
    if case.width * case.height >= 1_000_000 and not include_large:
        return
    case_dir = output_dir / case.path_token
    case_dir.mkdir(parents=True, exist_ok=True)
    for frame in case.frame_numbers:
        write_pfm(case_dir / f"frame.{frame:04d}.pfm", frame_rows(case, frame), case.width, case.height)
    (case_dir / "truth.json").write_text(
        json.dumps(truth_document(case), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


__all__ = [
    "COORDINATE_CONVENTION",
    "REQUIRED_SYNTHETIC_CASES",
    "SyntheticCase",
    "all_cases",
    "case_map",
    "analytic_displacement",
    "frame_rows",
    "generate_frame",
    "truth_document",
    "synthetic_shot",
    "synthetic_partition",
    "write_pfm",
    "write_case_frames",
]
