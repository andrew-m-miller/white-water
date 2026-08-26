#!/usr/bin/env python3
"""Focused tests for pure synthetic PFM encoding and atomic publication."""

from __future__ import annotations

from array import array
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import synthetic as synthetic_module
from .synthetic import encode_pfm, write_pfm


ROWS = (
    ((0.0, 1.0, 2.0), (-1.25, 3.5, 8.0)),
    ((4.0, 5.0, 6.0), (7.0, 8.0, 9.0)),
)


def _legacy_bytes(rows, width: int, height: int) -> bytes:
    """The former write loop, retained as a byte-equivalence oracle."""

    output = bytearray(f"PF\n{width} {height}\n-1.0\n".encode("ascii"))
    for row in rows:
        encoded = array("f")
        for pixel in row:
            encoded.extend(float(channel) for channel in pixel)
        output.extend(encoded.tobytes())
    return bytes(output)


class SyntheticPfmTests(unittest.TestCase):
    def test_encoder_matches_former_format_and_write_bytes(self):
        expected = _legacy_bytes(ROWS, 2, 2)
        self.assertEqual(encode_pfm(ROWS, 2, 2), expected)
        self.assertEqual(encode_pfm(iter(ROWS), 2, 2), expected)

        with tempfile.TemporaryDirectory(prefix="whitewater-synthetic-pfm-") as directory:
            path = Path(directory) / "nested" / "frame.pfm"
            write_pfm(path, ROWS, 2, 2)
            self.assertEqual(path.read_bytes(), expected)
            self.assertEqual((path.stat().st_mode & 0o777), 0o644)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_encoder_is_pure_and_preserves_validation_errors(self):
        with tempfile.TemporaryDirectory(prefix="whitewater-synthetic-pfm-") as directory:
            path = Path(directory) / "frame.pfm"
            original = b"prior bytes"
            path.write_bytes(original)
            os.chmod(path, 0o644)

            wrong_width_rows = (((1.0, 2.0, 3.0),),)
            wrong_rgb_rows = (((1.0, 2.0),),)
            invalid = (
                (wrong_width_rows, 2, 2, "synthetic row has the wrong width"),
                (wrong_rgb_rows, 1, 1, "synthetic PFM output is RGB"),
                ((ROWS[0], ROWS[1]), 2, 1, "synthetic frame has the wrong height"),
            )
            for rows, width, height, message in invalid:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        encode_pfm(rows, width, height)
                    # Encoding has no filesystem side effects, and write_pfm's atomic staging
                    # keeps an existing destination intact when validation fails.
                    with self.assertRaisesRegex(ValueError, message):
                        write_pfm(path, rows, width, height)
                    self.assertEqual(path.read_bytes(), original)

    def test_replace_failure_leaves_prior_file_and_cleans_private_temp(self):
        with tempfile.TemporaryDirectory(prefix="whitewater-synthetic-pfm-") as directory:
            path = Path(directory) / "frame.pfm"
            original = b"prior bytes"
            path.write_bytes(original)
            os.chmod(path, 0o644)
            with patch.object(synthetic_module.os, "replace", side_effect=OSError("injected replace")):
                with self.assertRaises(OSError):
                    write_pfm(path, ROWS, 2, 2)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_file_fsync_failure_leaves_prior_file_and_cleans_private_temp(self):
        with tempfile.TemporaryDirectory(prefix="whitewater-synthetic-pfm-") as directory:
            path = Path(directory) / "frame.pfm"
            original = b"prior bytes"
            path.write_bytes(original)
            os.chmod(path, 0o644)
            with patch.object(synthetic_module.os, "fsync", side_effect=OSError("injected fsync")):
                with self.assertRaises(OSError):
                    write_pfm(path, ROWS, 2, 2)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_write_failure_leaves_prior_file_and_cleans_private_temp(self):
        with tempfile.TemporaryDirectory(prefix="whitewater-synthetic-pfm-") as directory:
            path = Path(directory) / "frame.pfm"
            original = b"prior bytes"
            path.write_bytes(original)
            os.chmod(path, 0o644)

            def failing_fdopen(descriptor, mode):
                del mode

                class FailingStream:
                    def __enter__(self):
                        return self

                    def __exit__(self, exception_type, exception, traceback):
                        os.close(descriptor)
                        return False

                    def write(self, payload):
                        del payload
                        raise OSError("injected write")

                return FailingStream()

            with patch.object(synthetic_module.os, "fdopen", side_effect=failing_fdopen):
                with self.assertRaisesRegex(OSError, "injected write"):
                    write_pfm(path, ROWS, 2, 2)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
