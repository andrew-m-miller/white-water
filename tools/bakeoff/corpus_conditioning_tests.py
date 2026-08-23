#!/usr/bin/env python3
"""Numerical and metadata gates for the P25-2 corpus/conditioning slice."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile

from .conditioning import (
    ConditioningFailure,
    PAIR_PERCENTILE_EPSILON,
    apply_conditioning,
    condition_pair,
    linear_quantile,
    signed_log,
)
from .generate_corpus import build_corpus
from .padding import normalize_policy, pad_rows
from .synthetic import (
    COORDINATE_CONVENTION,
    REQUIRED_SYNTHETIC_CASES,
    all_cases,
    analytic_pair_coordinate,
    analytic_pair_truth,
    analytic_displacement,
    foreground_displacement,
    foreground_rect,
    generate_frame,
    frame_source_coordinate,
    TruthUnavailable,
    synthetic_partition,
    truth_document,
    visibility_at,
    write_case_frames,
    _sample_base,
)
from .validator import (
    load_json,
    validate_corpus_consistency,
    validate_protocol_and_schemas,
)


ROOT = Path(__file__).resolve().parents[2]


def _near(actual: float, expected: float, tolerance: float = 1.0e-9) -> None:
    assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance), (actual, expected)


def _test_conditioning() -> None:
    clamped = condition_pair([[-1.0, 0.5, 2.0]], [[0.25, 0.75, 1.5]], "native-clamp01-v1")
    assert clamped.first == [[0.0, 0.5, 1.0]]
    assert clamped.second == [[0.25, 0.75, 1.0]]

    _near(signed_log(-16.0), 0.0)
    _near(signed_log(0.0), 0.5)
    _near(signed_log(16.0), 1.0)
    signed = condition_pair([[-16.0, 0.0, 16.0]], [[-1.0, 1.0, 2.0]], "signed-log-v1")
    _near(signed.first[0][0], 0.0)
    _near(signed.first[0][1], 0.5)
    _near(signed.first[0][2], 1.0)

    _near(linear_quantile([0.0, 10.0, 20.0, 30.0], 0.01), 0.3)
    _near(linear_quantile([math.nan, 0.0, 10.0], 0.5), 5.0)
    pair = condition_pair(
        [[0.0, 1.0, 2.0], [10.0, 20.0, 30.0]],
        [[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]],
        "pair-percentile-v1",
    )
    assert pair.low is not None and pair.high is not None
    assert pair.epsilon == PAIR_PERCENTILE_EPSILON
    assert pair.parameters == {
        "low": pair.low,
        "high": pair.high,
        "epsilon": PAIR_PERCENTILE_EPSILON,
    }
    # The second frame uses the same low/high, rather than an independently normalized
    # range.  This is the regression that catches apparent exposure motion.
    assert pair.first[0][0] == 0.0
    assert pair.second[0][0] > pair.first[0][0]
    reused = apply_conditioning(
        [[0.0, 1.0, 2.0]],
        "pair-percentile-v1",
        low=pair.low,
        high=pair.high,
    )
    assert reused == pair.first[:1]

    native_log = condition_pair([[-0.25, 0.0, 2.0]], [[3.0, 4.0, 5.0]], "native-log-v1")
    assert native_log.first == [[-0.25, 0.0, 2.0]]
    assert native_log.second == [[3.0, 4.0, 5.0]]

    for token in ("native-clamp01-v1", "signed-log-v1", "pair-percentile-v1", "native-log-v1"):
        try:
            condition_pair([[math.nan, 1.0, 2.0]], [[1.0, 2.0, 3.0]], token)
        except ConditioningFailure as failure:
            assert failure.kind == "nonfinite_input"
        else:
            raise AssertionError(f"{token} accepted nonfinite input")
    try:
        condition_pair([], [], "pair-percentile-v1")
    except ConditioningFailure as failure:
        assert failure.kind == "empty_pair"
    else:
        raise AssertionError("empty pair did not produce a typed failure")

    for function in (
        lambda: condition_pair([[7.0, 7.0, 7.0]], [[7.0, 7.0, 7.0]], "pair-percentile-v1"),
        lambda: apply_conditioning([[7.0, 7.0, 7.0]], "pair-percentile-v1", low=7.0, high=7.0),
    ):
        try:
            function()
        except ConditioningFailure as failure:
            assert failure.kind == "constant_pair"
            assert failure.failure_type == "conditioning_failure"
        else:
            raise AssertionError("constant pair did not produce the typed v1 failure")


def _test_padding() -> None:
    source = ((1, 2, 3),)
    replicated = pad_rows(source, left=1, right=1, policy="caller-replication-crop")
    reflected = pad_rows(source, left=1, right=1, policy="caller-reflection-crop")
    assert replicated.rows == ((1, 1, 2, 3, 3),)
    assert reflected.rows == ((2, 1, 2, 3, 2),)
    assert replicated.crop == (1, 0, 3, 1)
    assert replicated.policy == "caller-replication-crop"
    assert reflected.policy == "caller-reflection-crop"

    # The P25-1 manifest vocabulary is passed through byte-for-byte; no candidate
    # name or local spelling is needed at this seam.
    assert normalize_policy("caller-replication-crop") == "caller-replication-crop"
    assert normalize_policy("caller-reflection-crop") == "caller-reflection-crop"

    # Generic names remain compatibility aliases, while the P25-1 migrated SEA-RAFT
    # declaration is accepted directly and is the token carried in the result.
    assert pad_rows(source, left=1, right=1, policy="replication").policy == "caller-replication-crop"
    assert pad_rows(source, left=1, right=1, policy="reflect").policy == "caller-reflection-crop"

    # Both policies preserve the source crop, while differing only in the caller-side
    # halo.  The runner passes the manifest-declared policy through this narrow seam.
    for padded in (replicated, reflected):
        assert padded.rows[0][padded.pad_left:padded.pad_left + padded.source_width] == source[0]

    asymmetric_source = tuple(tuple(0 for _ in range(67)) for _ in range(53))
    asymmetric = pad_rows(
        asymmetric_source,
        left=1,
        right=3,
        bottom=2,
        top=2,
        policy="caller-replication-crop",
        multiple=8,
    )
    assert asymmetric.width == 72
    assert asymmetric.height == 64
    assert asymmetric.width % 8 == 0 and asymmetric.height % 8 == 0
    assert (asymmetric.pad_left, asymmetric.pad_right, asymmetric.pad_bottom, asymmetric.pad_top) == (1, 4, 2, 9)
    assert asymmetric.crop == (1, 2, 67, 53)


def _test_synthetic_cases() -> None:
    cases = all_cases()
    assert tuple(case.case_id for case in cases) == REQUIRED_SYNTHETIC_CASES
    assert generate_frame("identity", 4) == generate_frame("identity", 4)
    assert analytic_displacement("identity", 4, 5, 10.0, 10.0) == (0.0, 0.0)
    assert analytic_displacement("translation-x-positive", 4, 5, 10.0, 10.0) == (1.25, 0.0)
    assert analytic_displacement("translation-x-negative", 4, 5, 10.0, 10.0) == (-1.25, 0.0)
    assert analytic_displacement("translation-y-positive", 4, 5, 10.0, 10.0) == (0.0, 1.5)
    assert analytic_displacement("translation-y-negative", 4, 5, 10.0, 10.0) == (0.0, -1.5)
    assert analytic_displacement("affine", 4, 5, 4.0, 4.0) != analytic_displacement("affine", 4, 5, 60.0, 40.0)
    assert analytic_displacement("spatial", 4, 5, 4.0, 4.0) != analytic_displacement("spatial", 4, 5, 60.0, 40.0)
    assert analytic_displacement("border", 4, 5, 0.0, 0.0) == (6.0, 4.0)
    assert analytic_displacement("chain-8", 8, 16, 10.0, 10.0) == (10.0, 0.0)

    assert "x right, y up" in COORDINATE_CONVENTION
    assert "row zero is the bottom row" in COORDINATE_CONVENTION
    positive_y_source = frame_source_coordinate("translation-y-positive", 5, 10.0, 10.0)
    assert positive_y_source == (10.0, 8.5)

    # The truth is a generated-frame correspondence: frame pixels are inverse samples
    # of the common plate, and pair truth composes the exact forward maps.  This catches
    # the old same-coordinate subtraction bug in affine/spatial fields.
    for case_id, frame, x, y in (("affine", 6, 33, 28), ("spatial", 6, 33, 28)):
        generated = generate_frame(case_id, frame)
        source_x, source_y = frame_source_coordinate(case_id, frame, x, y)
        expected = _sample_base(next(case for case in all_cases() if case.case_id == case_id), source_x, source_y)
        for actual_channel, expected_channel in zip(generated[y][x], expected):
            _near(actual_channel, expected_channel, tolerance=1.0e-12)
        pair = analytic_pair_coordinate(case_id, 5, frame, float(x), float(y))
        displacement = analytic_displacement(case_id, 5, frame, float(x), float(y))
        _near(displacement[0], pair[0] - x, tolerance=1.0e-12)
        _near(displacement[1], pair[1] - y, tolerance=1.0e-12)

    # The moving rectangle has independent foreground/background motion, and both
    # sides of the visibility transition are represented in the emitted truth.  The
    # typed API follows the visible layer and rejects a correspondence that changes
    # layers, so a foreground pixel can never silently receive background motion.
    foreground_truth = analytic_pair_truth("occlusion-reveal", 0, 8, 20.0, 30.0)
    assert foreground_truth.status == "foreground"
    assert foreground_truth.source_layer == "foreground"
    assert foreground_truth.target_layer == "foreground"
    assert foreground_truth.displacement == (24.0, -8.0)
    # The foreground leaves this fixed coordinate, so the background is revealed.
    assert foreground_truth.same_coordinate_transition == "revealed"
    assert analytic_displacement("occlusion-reveal", 0, 8, 20.0, 30.0) == (24.0, -8.0)
    background_truth = analytic_pair_truth("occlusion-reveal", 0, 8, 50.0, 40.0)
    assert background_truth.status == "background"
    assert background_truth.source_layer == "background"
    assert background_truth.target_layer == "background"
    assert background_truth.displacement == (8.0, 4.0)
    # At this fixed coordinate, background is replaced by foreground and is therefore
    # occluded.  The source-domain correspondence below exercises the same label for
    # background mapping into a target foreground pixel.
    occluded_transition = analytic_pair_truth("occlusion-reveal", 0, 8, 50.0, 30.0)
    assert occluded_transition.status == "background"
    assert occluded_transition.same_coordinate_transition == "occluded"
    occluded_truth = analytic_pair_truth("occlusion-reveal", 0, 8, 34.0, 29.0)
    assert occluded_truth.status == "occluded"
    assert occluded_truth.no_dense_truth
    assert occluded_truth.displacement is None
    try:
        analytic_displacement("occlusion-reveal", 0, 8, 34.0, 29.0)
    except TruthUnavailable as failure:
        assert failure.truth == occluded_truth
    else:
        raise AssertionError("layer-changing sample returned dense truth")
    try:
        analytic_pair_coordinate("occlusion-reveal", 0, 8, 34.0, 29.0)
    except TruthUnavailable as failure:
        assert failure.truth == occluded_truth
    else:
        raise AssertionError("layer-changing sample returned a pair coordinate")
    assert visibility_at("occlusion-reveal", 0, 28, 30)
    assert not visibility_at("occlusion-reveal", 8, 28, 30)
    assert not visibility_at("occlusion-reveal", 0, 50, 30)
    assert visibility_at("occlusion-reveal", 8, 50, 30)
    assert foreground_rect("occlusion-reveal", 8) == (37.0, 55.0, 14.0, 34.0)
    assert foreground_displacement("occlusion-reveal", 0, 8) == (24.0, -8.0)
    assert analytic_displacement("occlusion-reveal", 0, 8, 50.0, 30.0) == (8.0, 4.0)
    assert foreground_displacement("occlusion-reveal", 0, 8) != analytic_displacement("occlusion-reveal", 0, 8, 50.0, 30.0)
    occlusion_early = generate_frame("occlusion-reveal", 0)
    occlusion_late = generate_frame("occlusion-reveal", 8)
    assert occlusion_early[30][28] == (0.92, 0.17, 0.08)
    assert occlusion_late[30][50] == (0.92, 0.17, 0.08)
    assert occlusion_late[30][28] != (0.92, 0.17, 0.08)

    blur = generate_frame("blur", 5)
    sharp = generate_frame("translation-x-positive", 5)
    assert blur != sharp
    noise_a = generate_frame("noise", 5)
    noise_b = generate_frame("noise", 5)
    assert noise_a == noise_b
    noise_case = next(case for case in cases if case.case_id == "noise")
    assert noise_case.parameters is not None
    recorded_seed = noise_case.parameters["seed"]
    assert recorded_seed == noise_case.seed == 4701
    changed_parameters = dict(noise_case.parameters)
    changed_parameters["seed"] = recorded_seed + 1
    changed_noise_case = replace(
        noise_case,
        seed=recorded_seed + 1,
        parameters=changed_parameters,
    )
    assert generate_frame(changed_noise_case, 5) != noise_a
    assert truth_document("noise")["motion"]["parameters"]["seed"] == recorded_seed
    hdr = generate_frame("hdr-scene-linear", 4)
    assert max(channel for row in hdr for pixel in row for channel in pixel) > 1.0
    log = generate_frame("log-input", 4)
    assert all(0.0 <= channel <= 1.0 for row in log for pixel in row for channel in pixel)

    by_id = {case.case_id: case for case in cases}
    assert (by_id["odd-size"].width % 2 == 1 or by_id["odd-size"].height % 2 == 1)
    assert by_id["par-0_5"].pixel_aspect_ratio == 0.5
    assert by_id["par-2"].pixel_aspect_ratio == 2.0
    assert [by_id[f"chain-{length}"].chain_length for length in (1, 2, 4, 8)] == [1, 2, 4, 8]


def _test_corpus_and_emission() -> None:
    protocol = load_json(ROOT / "bakeoff/protocol-v1.json")
    protocol_schema = load_json(ROOT / "bakeoff/protocol-v1.schema.json")
    corpus_schema = load_json(ROOT / "bakeoff/corpus-v1.schema.json")
    report_schema = load_json(ROOT / "bakeoff/report-v1.schema.json")
    validate_protocol_and_schemas(protocol, protocol_schema, corpus_schema, report_schema)
    corpus = build_corpus(ROOT / "bakeoff/production-corpus-v1.template.json", corpus_id="test-generated")
    validate_corpus_consistency(corpus, protocol, corpus_schema)
    assert len(corpus["partitions"]) == 2
    assert not any(partition["id"] == "public" for partition in corpus["partitions"])

    # generate_corpus.py intentionally supports both documented invocation forms.
    # Keep the top-level synthetic/padding fallback import alive when the script is
    # launched by path, while the module form continues to use package imports.
    with tempfile.TemporaryDirectory(prefix="whitewater-p25-corpus-cli-") as temporary:
        directory = Path(temporary)
        for label, command in (
            (
                "direct",
                [sys.executable, str(ROOT / "tools/bakeoff/generate_corpus.py"),
                 "--output", str(directory / "direct.json")],
            ),
            (
                "module",
                [sys.executable, "-m", "tools.bakeoff.generate_corpus",
                 "--output", str(directory / "module.json")],
            ),
        ):
            subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
            generated = load_json(directory / f"{label}.json")
            validate_corpus_consistency(generated, protocol, corpus_schema)

    with tempfile.TemporaryDirectory(prefix="whitewater-p25-corpus-") as temporary:
        directory = Path(temporary)
        write_case_frames("identity", directory)
        assert (directory / "identity/frame.0000.pfm").is_file()
        truth = json.loads((directory / "identity/truth.json").read_text(encoding="utf-8"))
        assert truth["case_id"] == "identity"
        assert truth["coordinate_convention"]
        write_case_frames("asymmetric-padding", directory)
        padding_truth = json.loads((directory / "asymmetric-padding/truth.json").read_text(encoding="utf-8"))
        assert padding_truth["padding"]["padded_width"] % 8 == 0
        assert padding_truth["padding"]["padded_height"] % 8 == 0
        write_case_frames("occlusion-reveal", directory)
        visibility_truth = json.loads((directory / "occlusion-reveal/truth.json").read_text(encoding="utf-8"))
        assert visibility_truth["visibility"]["frames"]
        assert visibility_truth["pair_truth"]["api"].startswith("analytic_pair_truth")
        assert visibility_truth["pair_truth"]["non_dense_displacement"] is None


def main() -> int:
    _test_conditioning()
    _test_padding()
    _test_synthetic_cases()
    _test_corpus_and_emission()
    print("P25-2 corpus and conditioning tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
