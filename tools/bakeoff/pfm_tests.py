#!/usr/bin/env python3
"""Focused strictness and orientation tests for the dependency-free PFM adapter."""

from __future__ import annotations

import math
import os
from pathlib import Path
import struct
import tempfile
import unittest

from .pfm import MAX_DIMENSION, PfmFailure, read_pfm
from .synthetic import generate_frame, write_case_frames


def _write(path: Path, magic: str, width: int, height: int, scale: str, values: list[float],
           *, endian: str | None = None) -> None:
    if endian is None:
        endian = "<" if scale.startswith("-") else ">"
    path.write_bytes(
        f"{magic}\n{width} {height}\n{scale}\n".encode("ascii")
        + struct.pack(f"{endian}{len(values)}f", *values)
    )
    os.chmod(path, 0o644)


class PfmTests(unittest.TestCase):
    def test_grayscale_rgb_endian_scale_and_bottom_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gray = root / "gray.pfm"
            _write(gray, "Pf", 2, 2, "-2.0", [0.5, 1.0, 1.5, 2.0])
            decoded = read_pfm(gray)
            self.assertEqual((decoded.width, decoded.height, decoded.channels), (2, 2, 1))
            self.assertEqual(decoded.rows, (((1.0,), (2.0,)), ((3.0,), (4.0,))))

            rgb = root / "rgb.pfm"
            _write(rgb, "PF", 1, 2, "2.0", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0], endian=">")
            decoded = read_pfm(rgb)
            self.assertEqual(decoded.channels, 3)
            self.assertEqual(decoded.rows, (((1.0, 2.0, 3.0),), ((4.0, 5.0, 6.0),)))

    def test_big_endian_and_crlf_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "big.pfm"
            path.write_bytes(b"Pf\r\n1 1\r\n1.0\r\n" + struct.pack(">f", 7.0))
            os.chmod(path, 0o644)
            self.assertEqual(read_pfm(path).rows, (((7.0,),),))

    def test_generated_synthetic_fixture_is_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_case_frames("identity", root)
            image = read_pfm(root / "identity" / "frame.0000.pfm")
            self.assertEqual((image.width, image.height, image.channels), (64, 48, 3))
            self.assertTrue(all(math.isfinite(value) for row in image.rows for pixel in row for value in pixel))
            expected = generate_frame("identity", 0)
            for x, y in ((0, 0), (63, 0), (0, 47), (63, 47)):
                for actual, wanted in zip(image.rows[y][x], expected[y][x]):
                    self.assertAlmostEqual(actual, wanted, places=6)
            self.assertNotEqual(image.rows[0][0], image.rows[-1][0])

    def test_header_and_payload_failures_are_typed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = {
                "magic": b"PX\n1 1\n-1.0\n" + struct.pack("<f", 0.0),
                "dimensions": b"Pf\n1\t1\n-1.0\n" + struct.pack("<f", 0.0),
                "scale": b"Pf\n1 1\nNaN\n" + struct.pack("<f", 0.0),
                "header": b"Pf\n1 1\n-1.0",
            }
            for kind, payload in cases.items():
                with self.subTest(kind=kind):
                    path = root / f"{kind}.pfm"
                    path.write_bytes(payload)
                    os.chmod(path, 0o644)
                    with self.assertRaises(PfmFailure) as context:
                        read_pfm(path)
                    self.assertIn(context.exception.kind, {kind, "header"})
            huge = root / "huge.pfm"
            huge.write_text(f"Pf\n{MAX_DIMENSION + 1} 1\n-1.0\n", encoding="ascii")
            os.chmod(huge, 0o644)
            with self.assertRaises(PfmFailure) as context:
                read_pfm(huge)
            self.assertEqual(context.exception.kind, "dimensions")

            truncated = root / "truncated.pfm"
            truncated.write_bytes(b"Pf\n1 1\n-1.0\n" + b"\x00\x00")
            os.chmod(truncated, 0o644)
            with self.assertRaises(PfmFailure) as context:
                read_pfm(truncated)
            self.assertEqual(context.exception.kind, "truncated_payload")

            trailing = root / "trailing.pfm"
            trailing.write_bytes(b"Pf\n1 1\n-1.0\n" + struct.pack("<f", 0.0) + b"x")
            os.chmod(trailing, 0o644)
            with self.assertRaises(PfmFailure) as context:
                read_pfm(trailing)
            self.assertEqual(context.exception.kind, "trailing_payload")

    def test_nonfinite_samples_file_types_and_no_implicit_decompression(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nonfinite = root / "nonfinite.pfm"
            _write(nonfinite, "Pf", 1, 1, "-1.0", [float("nan")])
            with self.assertRaises(PfmFailure) as context:
                read_pfm(nonfinite)
            self.assertEqual(context.exception.kind, "nonfinite_sample")

            with self.assertRaises(PfmFailure) as context:
                read_pfm(root)
            self.assertEqual(context.exception.kind, "nonregular_file")
            link = root / "link.pfm"
            link.symlink_to(nonfinite)
            with self.assertRaises(PfmFailure) as context:
                read_pfm(link)
            self.assertEqual(context.exception.kind, "symlink_file")

            compressed = root / "compressed.pfm.gz"
            compressed.write_bytes(b"not a PFM")
            os.chmod(compressed, 0o644)
            with self.assertRaises(PfmFailure) as context:
                read_pfm(compressed)
            self.assertIn(context.exception.kind, {"magic", "header"})

    def test_oversized_payload_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oversized.pfm"
            path.write_text(f"PF\n{MAX_DIMENSION} {MAX_DIMENSION}\n-1.0\n", encoding="ascii")
            os.chmod(path, 0o644)
            with self.assertRaises(PfmFailure) as context:
                read_pfm(path)
            self.assertEqual(context.exception.kind, "dimensions")

    def test_mode_is_exactly_0644(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mode.pfm"
            _write(path, "Pf", 1, 1, "-1.0", [1.0])
            os.chmod(path, 0o600)
            with self.assertRaises(PfmFailure) as context:
                read_pfm(path)
            self.assertEqual(context.exception.kind, "file_mode")


if __name__ == "__main__":
    unittest.main()
