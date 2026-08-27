#!/usr/bin/env python3
"""Focused unit tests for the dependency-light EXR sequence/pairing module.

Every test below runs without the OpenEXR Python bindings, NumPy, or a GPU: the sequence/pairing
logic is pure, and frame decode is exercised through an injected fake decoder. The one exception is
``test_optional_openexr_round_trip``, which skips cleanly (never fails) when the ``OpenEXR`` module
(or numpy) is not importable -- this dev machine is macOS without the binding, and CI's EL8
container is the only place that dependency is expected to be present.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from .exr import (
    ExrFailure,
    _bottom_origin_rows,
    _classify_channels,
    _format_name_from_string,
    _source_format_from_names,
    expand_shot_sequence,
    frame_from_exr,
    load_pair,
    reference_target_pair,
    validate_frame_matches_shot_metadata,
    validate_pair_geometry,
    validate_pair_layout,
)


def _shot(**overrides: Any) -> dict[str, Any]:
    shot = {
        "id": "prod-motion-blur",
        "path_pattern": "/AIRGAP/replace/motion-blur/plate.%04d.exr",
        "first_frame": 1001,
        "last_frame": 1017,
        "reference_frame": 1009,
        "width": 1920,
        "height": 1080,
        "pixel_aspect_ratio": 1.0,
        "encoding": "scene-linear",
        "channels": "RGBA",
        "bit_depth": "half",
        "categories": ["motion-blur"],
        "annotations": None,
    }
    shot.update(overrides)
    return shot


def _failure_kind(callable_: Any) -> str:
    try:
        callable_()
    except ExrFailure as exc:
        return exc.kind
    raise AssertionError("expected an ExrFailure and none was raised")


def test_expand_shot_sequence_valid_order_and_paths() -> None:
    shot = _shot(first_frame=1001, last_frame=1004, reference_frame=1002)
    expanded = expand_shot_sequence(shot)
    assert expanded == (
        (1001, "/AIRGAP/replace/motion-blur/plate.1001.exr"),
        (1002, "/AIRGAP/replace/motion-blur/plate.1002.exr"),
        (1003, "/AIRGAP/replace/motion-blur/plate.1003.exr"),
        (1004, "/AIRGAP/replace/motion-blur/plate.1004.exr"),
    ), expanded

    # A single-frame range (first == last == reference) is a valid, non-empty sequence.
    single = _shot(first_frame=1005, last_frame=1005, reference_frame=1005)
    assert expand_shot_sequence(single) == ((1005, "/AIRGAP/replace/motion-blur/plate.1005.exr"),)

    # Un-padded and explicitly-width tokens both format correctly.
    unpadded = _shot(path_pattern="/shots/x/plate.%d.exr", first_frame=1, last_frame=1, reference_frame=1)
    assert expand_shot_sequence(unpadded) == ((1, "/shots/x/plate.1.exr"),)


def test_expand_shot_sequence_malformed_pattern_kinds() -> None:
    no_token = _shot(path_pattern="/shots/x/plate.exr")
    assert _failure_kind(lambda: expand_shot_sequence(no_token)) == "missing_frame_token"

    two_tokens = _shot(path_pattern="/shots/x/plate.%04d.%04d.exr")
    assert _failure_kind(lambda: expand_shot_sequence(two_tokens)) == "multiple_frame_tokens"

    stray_percent = _shot(path_pattern="/shots/100%/plate.%04d.exr")
    assert _failure_kind(lambda: expand_shot_sequence(stray_percent)) == "malformed_frame_token"

    empty_pattern = _shot(path_pattern="")
    assert _failure_kind(lambda: expand_shot_sequence(empty_pattern)) == "corpus_shape"


def test_expand_shot_sequence_empty_range_and_reference_out_of_range() -> None:
    empty_range = _shot(first_frame=1010, last_frame=1005, reference_frame=1005)
    assert _failure_kind(lambda: expand_shot_sequence(empty_range)) == "empty_range"

    reference_before = _shot(first_frame=1001, last_frame=1017, reference_frame=1000)
    assert _failure_kind(lambda: expand_shot_sequence(reference_before)) == "reference_out_of_range"

    reference_after = _shot(first_frame=1001, last_frame=1017, reference_frame=1018)
    assert _failure_kind(lambda: expand_shot_sequence(reference_after)) == "reference_out_of_range"

    non_integer = _shot(first_frame="1001")
    assert _failure_kind(lambda: expand_shot_sequence(non_integer)) == "corpus_shape"


def test_reference_target_pair_offsets_both_directions() -> None:
    shot = _shot(first_frame=1001, last_frame=1017, reference_frame=1009)
    for offset in (1, -1, 2, -2, 4, -4, 8, -8):
        reference_path, target_path, reference_frame, target_frame = reference_target_pair(shot, offset)
        assert reference_frame == 1009
        assert target_frame == 1009 + offset
        assert reference_path == "/AIRGAP/replace/motion-blur/plate.1009.exr"
        assert target_path == f"/AIRGAP/replace/motion-blur/plate.{1009 + offset:04d}.exr"


def test_reference_target_pair_out_of_range_is_typed() -> None:
    shot = _shot(first_frame=1001, last_frame=1017, reference_frame=1009)
    # +8 and -8 are exactly in range (1001 and 1017); +9/-9 are not.
    reference_target_pair(shot, 8)
    reference_target_pair(shot, -8)
    assert _failure_kind(lambda: reference_target_pair(shot, 9)) == "target_out_of_range"
    assert _failure_kind(lambda: reference_target_pair(shot, -9)) == "target_out_of_range"
    assert _failure_kind(lambda: reference_target_pair(shot, "1")) == "input_invalid"


def test_geometry_mismatch_detection() -> None:
    base = {"width": 1920, "height": 1080, "channels": 3, "pixel_aspect_ratio": 1.0}
    validate_pair_geometry(base, dict(base))  # identical geometry is fine

    for key, other in (
        ("width", 1280),
        ("height", 720),
        ("channels", 4),
        ("pixel_aspect_ratio", 2.0),
    ):
        mismatched = dict(base)
        mismatched[key] = other
        assert _failure_kind(lambda m=mismatched: validate_pair_geometry(base, m)) == "geometry_mismatch"


def test_classify_channels_layout_and_rejection() -> None:
    assert _classify_channels(["R", "G", "B"]) == ((0, 1, 2), "RGB")
    assert _classify_channels(["R", "G", "B", "A"]) == ((0, 1, 2), "RGBA")
    # Order in the file need not be R, G, B, A -- indices should follow the actual positions.
    assert _classify_channels(["A", "B", "G", "R"]) == ((3, 2, 1), "RGBA")
    assert _failure_kind(lambda: _classify_channels(["Y"])) == "unsupported_channels"
    assert _failure_kind(lambda: _classify_channels(["R", "G"])) == "unsupported_channels"
    assert _failure_kind(lambda: _classify_channels([])) == "unsupported_channels"
    # A channel beyond RGB/RGBA (e.g. a depth AOV) is rejected, not silently ignored.
    assert _failure_kind(lambda: _classify_channels(["R", "G", "B", "Z"])) == "unsupported_channels"
    assert _failure_kind(lambda: _classify_channels(["R", "G", "B", "A", "Z"])) == "unsupported_channels"


def test_source_format_rejects_integer_and_mixed_storage() -> None:
    assert _format_name_from_string("half") == "half"
    assert _format_name_from_string("float") == "float"
    assert _format_name_from_string(" Float ") == "float"  # tolerant of case/whitespace
    assert _failure_kind(lambda: _format_name_from_string("uint32")) == "unsupported_storage"
    assert _failure_kind(lambda: _format_name_from_string("uint8")) == "unsupported_storage"
    assert _failure_kind(lambda: _format_name_from_string("double")) == "unsupported_storage"

    assert _source_format_from_names(["half", "half", "half"]) == "half"
    assert _source_format_from_names(["float", "float", "float"]) == "float"
    assert _failure_kind(lambda: _source_format_from_names(["half", "float", "half"])) == "unsupported_storage"
    assert _failure_kind(lambda: _source_format_from_names(["uint32", "uint32", "uint32"])) == "unsupported_storage"


def test_bottom_origin_row_reversal_catches_vertical_inversion() -> None:
    # Row 0 in OpenEXR's top-to-bottom order carries a distinctive "top" marker; the last row (the
    # image's bottom) carries a "bottom" marker. A reader that forgot to reverse would return
    # rows[0] as the top row here, flipping dy sign for every downstream landmark/metric.
    top_to_bottom = (
        ("top-row-marker",),
        ("middle-row",),
        ("bottom-row-marker",),
    )
    bottom_origin = _bottom_origin_rows(top_to_bottom)
    assert bottom_origin[0] == ("bottom-row-marker",)
    assert bottom_origin[-1] == ("top-row-marker",)
    assert bottom_origin == tuple(reversed(top_to_bottom))


def test_load_pair_with_fake_decoder_and_geometry_propagation() -> None:
    # Default shot fixture declares channels="RGBA", bit_depth="half"; the fake decoder below
    # matches that so this exercises the happy path through all of load_pair's validation.
    shot = _shot(first_frame=1001, last_frame=1017, reference_frame=1009, pixel_aspect_ratio=2.0)
    calls: list[tuple[str, int, float]] = []

    def fake_decoder(path: str, *, frame_number: int, pixel_aspect_ratio: float) -> dict[str, Any]:
        calls.append((path, frame_number, pixel_aspect_ratio))
        # Values chosen to only be exactly representable in IEEE-754 half precision, standing in
        # for "half storage decoded to float" -- the pairing/geometry layer must pass them
        # through unchanged regardless of the original on-disk storage class.
        half_exact = 0.333251953125
        return {
            "width": 1920,
            "height": 1080,
            "channels": 3,
            "rows": ((( half_exact, half_exact, half_exact),),),
            "pixel_aspect_ratio": pixel_aspect_ratio,
            "frame": frame_number,
            "sha256": "0" * 64,
            "source": path,
            "source_channels": "RGBA",
            "source_format": "half",
        }

    first, second = load_pair(shot, 4, decoder=fake_decoder)
    assert first["frame"] == 1009
    assert second["frame"] == 1013
    assert first["rows"][0][0] == (0.333251953125, 0.333251953125, 0.333251953125)
    assert calls == [
        ("/AIRGAP/replace/motion-blur/plate.1009.exr", 1009, 2.0),
        ("/AIRGAP/replace/motion-blur/plate.1013.exr", 1013, 2.0),
    ]

    def mismatched_decoder(path: str, *, frame_number: int, pixel_aspect_ratio: float) -> dict[str, Any]:
        width = 1920 if frame_number == 1009 else 1280
        return {
            "width": width,
            "height": 1080,
            "channels": 3,
            "rows": (),
            "pixel_aspect_ratio": pixel_aspect_ratio,
            "frame": frame_number,
            "sha256": "0" * 64,
            "source": path,
            "source_channels": "RGBA",
            "source_format": "half",
        }

    assert _failure_kind(lambda: load_pair(shot, 1, decoder=mismatched_decoder)) == "geometry_mismatch"


def test_load_pair_rejects_rgba_rgb_layout_mismatch() -> None:
    shot = _shot(first_frame=1001, last_frame=1017, reference_frame=1009, channels=None, bit_depth=None)

    def decoder(path: str, *, frame_number: int, pixel_aspect_ratio: float) -> dict[str, Any]:
        layout = "RGBA" if frame_number == 1009 else "RGB"
        return {
            "width": 1920,
            "height": 1080,
            "channels": 3,
            "rows": (((0.1, 0.1, 0.1),),),
            "pixel_aspect_ratio": pixel_aspect_ratio,
            "frame": frame_number,
            "sha256": "0" * 64,
            "source": path,
            "source_channels": layout,
            "source_format": "half",
        }

    assert _failure_kind(lambda: load_pair(shot, 1, decoder=decoder)) == "layout_mismatch"


def test_load_pair_rejects_metadata_mismatch_against_shot() -> None:
    shot = _shot(first_frame=1001, last_frame=1017, reference_frame=1009, channels="RGB", bit_depth="float")

    def decoder(path: str, *, frame_number: int, pixel_aspect_ratio: float) -> dict[str, Any]:
        return {
            "width": 1920,
            "height": 1080,
            "channels": 3,
            "rows": (((0.1, 0.1, 0.1),),),
            "pixel_aspect_ratio": pixel_aspect_ratio,
            "frame": frame_number,
            "sha256": "0" * 64,
            "source": path,
            "source_channels": "RGB",
            "source_format": "half",  # shot declares float -- this must be rejected
        }

    assert _failure_kind(lambda: load_pair(shot, 1, decoder=decoder)) == "metadata_mismatch"

    def channel_mismatch_decoder(path: str, *, frame_number: int, pixel_aspect_ratio: float) -> dict[str, Any]:
        return {
            "width": 1920,
            "height": 1080,
            "channels": 3,
            "rows": (((0.1, 0.1, 0.1),),),
            "pixel_aspect_ratio": pixel_aspect_ratio,
            "frame": frame_number,
            "sha256": "0" * 64,
            "source": path,
            "source_channels": "RGBA",  # shot declares RGB -- this must be rejected
            "source_format": "float",
        }

    assert _failure_kind(lambda: load_pair(shot, 1, decoder=channel_mismatch_decoder)) == "metadata_mismatch"


def test_validate_pair_layout_detects_mismatch() -> None:
    base = {"source_channels": "RGB", "source_format": "half"}
    validate_pair_layout(base, dict(base))  # identical layout is fine

    for key, other in (("source_channels", "RGBA"), ("source_format", "float")):
        mismatched = dict(base)
        mismatched[key] = other
        assert _failure_kind(lambda m=mismatched: validate_pair_layout(base, m)) == "layout_mismatch"


def test_validate_frame_matches_shot_metadata_skips_undeclared_fields() -> None:
    frame = {"source_channels": "RGBA", "source_format": "half"}
    validate_frame_matches_shot_metadata({}, frame)  # no declared fields -> nothing to check
    validate_frame_matches_shot_metadata({"channels": "RGBA"}, frame)  # matches -> fine
    validate_frame_matches_shot_metadata({"bit_depth": "half"}, frame)  # matches -> fine
    assert _failure_kind(
        lambda: validate_frame_matches_shot_metadata({"channels": "RGB"}, frame)
    ) == "metadata_mismatch"
    assert _failure_kind(
        lambda: validate_frame_matches_shot_metadata({"bit_depth": "float"}, frame)
    ) == "metadata_mismatch"


def test_frame_from_exr_reports_typed_dependency_failure_when_binding_absent() -> None:
    if (
        importlib.util.find_spec("OpenEXR") is not None
        and importlib.util.find_spec("numpy") is not None
    ):
        return  # the binding is installed here; the dependency-gate path is untestable.
    kind = _failure_kind(lambda: frame_from_exr("/nonexistent/does-not-matter.exr"))
    assert kind == "runtime_error"


def test_optional_openexr_round_trip() -> None:
    if importlib.util.find_spec("OpenEXR") is None or importlib.util.find_spec("numpy") is None:
        print("  (skipping OpenEXR round-trip test: the OpenEXR bindings/numpy are not installed)")
        return

    import tempfile
    from pathlib import Path

    import numpy as np  # type: ignore
    import OpenEXR  # type: ignore

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "roundtrip.exr"
        width, height = 3, 4
        # numpy row 0 is the image's first (top) scanline; row height-1 is its last (bottom)
        # scanline. R varies by row so the test can assert the returned frame's rows[0] is the
        # BOTTOM row and rows[-1] is the TOP row -- this repository's bottom-origin convention
        # (see synthetic.COORDINATE_CONVENTION and pfm.py), which OpenEXR does not follow natively.
        # Half (float16) storage exercises the HALF -> "half" source_format classification.
        R = np.zeros((height, width), dtype=np.float16)
        G = np.zeros((height, width), dtype=np.float16)
        B = np.full((height, width), 0.5, dtype=np.float16)
        A = np.ones((height, width), dtype=np.float16)
        for y in range(height):
            for x in range(width):
                R[y, x] = float(y) / 10.0
                G[y, x] = x / 10.0
        header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
        channels = {"R": R, "G": G, "B": B, "A": A}
        with OpenEXR.File(header, channels) as outfile:
            outfile.write(str(path))

        frame = frame_from_exr(path, frame_number=7, pixel_aspect_ratio=1.5)
        assert frame["width"] == width
        assert frame["height"] == height
        assert frame["channels"] == 3
        assert frame["frame"] == 7
        assert frame["pixel_aspect_ratio"] == 1.5
        assert frame["source"] == str(path)
        assert frame["source_channels"] == "RGBA"
        assert frame["source_format"] == "half"
        assert len(frame["sha256"]) == 64

        top_written_r = 0.0 / 10.0
        bottom_written_r = (height - 1) / 10.0
        bottom_row_r = frame["rows"][0][0][0]
        top_row_r = frame["rows"][-1][0][0]
        assert abs(bottom_row_r - bottom_written_r) < 1e-3, "rows[0] must be the BOTTOM scanline"
        assert abs(top_row_r - top_written_r) < 1e-3, "rows[-1] must be the TOP scanline"

        for y in range(height):
            for x in range(width):
                r, g, b = frame["rows"][height - 1 - y][x]
                assert abs(r - float(y) / 10.0) < 1e-3
                assert abs(g - x / 10.0) < 1e-3
                assert abs(b - 0.5) < 1e-3


def main() -> int:
    test_expand_shot_sequence_valid_order_and_paths()
    test_expand_shot_sequence_malformed_pattern_kinds()
    test_expand_shot_sequence_empty_range_and_reference_out_of_range()
    test_reference_target_pair_offsets_both_directions()
    test_reference_target_pair_out_of_range_is_typed()
    test_geometry_mismatch_detection()
    test_classify_channels_layout_and_rejection()
    test_source_format_rejects_integer_and_mixed_storage()
    test_bottom_origin_row_reversal_catches_vertical_inversion()
    test_load_pair_with_fake_decoder_and_geometry_propagation()
    test_load_pair_rejects_rgba_rgb_layout_mismatch()
    test_load_pair_rejects_metadata_mismatch_against_shot()
    test_validate_pair_layout_detects_mismatch()
    test_validate_frame_matches_shot_metadata_skips_undeclared_fields()
    test_frame_from_exr_reports_typed_dependency_failure_when_binding_absent()
    test_optional_openexr_round_trip()
    print("P25-6 EXR reader tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
