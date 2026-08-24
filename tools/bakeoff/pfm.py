#!/usr/bin/env python3
"""Strict, dependency-free PFM input for the offline bake-off.

The public rows are immutable and use the repository convention: row zero is the
bottom row and each pixel is a tuple containing one or three channel values. The
repository's PFM payload scanlines already use that same bottom-origin order.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import stat
import struct


MAX_DIMENSION = 32768
MAX_PAYLOAD_BYTES = 512 * 1024 * 1024
MAX_HEADER_BYTES = 4096
EXPECTED_MODE = 0o644
_MAGIC = {"PF": 3, "Pf": 1}
_DIMENSIONS = re.compile(r"[1-9][0-9]* [1-9][0-9]*\Z")
_SCALE = re.compile(
    r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z"
)


class PfmFailure(ValueError):
    """Stable, typed PFM input failure."""

    def __init__(self, kind: str, message: str):
        self.kind = kind
        self.reason = kind
        self.failure_type = "pfm_failure"
        self.message = message
        super().__init__(f"{kind}: {message}")


@dataclass(frozen=True)
class PfmImage:
    """An immutable PFM image in bottom-origin row-major form.

    rows is rows[y][x][channel]. Grayscale pixels have one component;
    RGB pixels have three components.
    """

    width: int
    height: int
    channels: int
    rows: tuple[tuple[tuple[float, ...], ...], ...]


def _fail(kind: str, message: str) -> None:
    raise PfmFailure(kind, message)


def _regular_fd(path: Path) -> int:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise PfmFailure("missing_file", f"PFM file does not exist: {path}") from exc
    except OSError as exc:
        raise PfmFailure("file_error", f"cannot inspect PFM file {path}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode):
        _fail("symlink_file", f"PFM input must not be a symlink: {path}")
    if not stat.S_ISREG(info.st_mode):
        _fail("nonregular_file", f"PFM input must be a regular file: {path}")
    if stat.S_IMODE(info.st_mode) != EXPECTED_MODE:
        _fail("file_mode", f"PFM input mode must be exactly 0644: {path}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        raise PfmFailure("file_error", f"cannot open PFM file {path}: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
    except OSError as exc:
        os.close(descriptor)
        raise PfmFailure("file_error", f"cannot stat opened PFM file {path}: {exc}") from exc
    if not stat.S_ISREG(opened.st_mode):
        os.close(descriptor)
        _fail("nonregular_file", f"PFM input must be a regular file: {path}")
    if stat.S_IMODE(opened.st_mode) != EXPECTED_MODE:
        os.close(descriptor)
        _fail("file_mode", f"PFM input mode must be exactly 0644: {path}")
    if opened.st_size < 0 or opened.st_size > MAX_PAYLOAD_BYTES + MAX_HEADER_BYTES:
        os.close(descriptor)
        _fail("size_bound", f"PFM input exceeds the bounded file size: {path}")
    return descriptor


def _header_line(data: bytes, offset: int, label: str) -> tuple[str, int]:
    end = data.find(b"\n", offset)
    if end < 0:
        _fail("header", f"PFM {label} line is not terminated by LF")
    raw = data[offset:end]
    if raw.endswith(b"\r"):
        raw = raw[:-1]
    if any(byte > 0x7F for byte in raw):
        _fail("ascii", f"PFM {label} line is not ASCII")
    try:
        return raw.decode("ascii"), end + 1
    except UnicodeDecodeError as exc:
        raise PfmFailure("ascii", f"PFM {label} line is not ASCII") from exc


def _parse_header(data: bytes) -> tuple[int, int, int, float, int]:
    magic, offset = _header_line(data, 0, "magic")
    if magic not in _MAGIC:
        _fail("magic", "PFM magic must be exactly PF or Pf")
    dimensions, offset = _header_line(data, offset, "dimensions")
    if _DIMENSIONS.fullmatch(dimensions) is None:
        _fail("dimensions", "PFM dimensions must be positive ASCII integers separated by one space")
    width_text, height_text = dimensions.split(" ")
    width, height = int(width_text), int(height_text)
    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        _fail("dimensions", f"PFM dimensions exceed the {MAX_DIMENSION} bound")
    scale_text, offset = _header_line(data, offset, "scale")
    if _SCALE.fullmatch(scale_text) is None:
        _fail("scale", "PFM scale must be an ASCII decimal number")
    try:
        scale = float(scale_text)
    except ValueError as exc:
        raise PfmFailure("scale", "PFM scale is not a number") from exc
    if not math.isfinite(scale) or scale == 0.0:
        _fail("scale", "PFM scale must be finite and non-zero")
    channels = _MAGIC[magic]
    payload_bytes = width * height * channels * 4
    if payload_bytes > MAX_PAYLOAD_BYTES:
        _fail("dimensions", "PFM payload exceeds the size bound")
    return width, height, channels, scale, offset


def read_pfm(path: Path | str) -> PfmImage:
    """Read one strict PF/Pf file without following links or decompressing inputs."""

    input_path = Path(path)
    descriptor = _regular_fd(input_path)
    try:
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            data = stream.read(MAX_PAYLOAD_BYTES + MAX_HEADER_BYTES + 1)
    except OSError as exc:
        raise PfmFailure("file_error", f"cannot read PFM file {input_path}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    if len(data) > MAX_PAYLOAD_BYTES + MAX_HEADER_BYTES:
        _fail("size_bound", f"PFM input exceeds the bounded file size: {input_path}")
    width, height, channels, scale, offset = _parse_header(data)
    expected_bytes = width * height * channels * 4
    payload = data[offset:]
    if len(payload) < expected_bytes:
        _fail("truncated_payload", f"PFM payload has {len(payload)} bytes; expected {expected_bytes}")
    if len(payload) > expected_bytes:
        _fail("trailing_payload", f"PFM payload has trailing bytes after {expected_bytes} expected bytes")
    endian = "<" if scale < 0.0 else ">"
    values = struct.unpack(f"{endian}{width * height * channels}f", payload)
    magnitude = abs(scale)
    decoded: list[tuple[tuple[float, ...], ...]] = []
    cursor = 0
    for _ in range(height):
        row: list[tuple[float, ...]] = []
        for _ in range(width):
            pixel = tuple(values[cursor + channel] * magnitude for channel in range(channels))
            cursor += channels
            if any(not math.isfinite(value) for value in pixel):
                _fail("nonfinite_sample", "PFM contains a nonfinite decoded or scaled sample")
            row.append(pixel)
        decoded.append(tuple(row))
    return PfmImage(width, height, channels, tuple(decoded))


__all__ = [
    "EXPECTED_MODE", "MAX_DIMENSION", "MAX_HEADER_BYTES", "MAX_PAYLOAD_BYTES",
    "PfmFailure", "PfmImage", "read_pfm",
]
