#!/usr/bin/env python3
"""Numerical and metadata gates for the P25-2 corpus/conditioning slice."""

from __future__ import annotations

import json
import math
from pathlib import Path
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
from .padding import pad_rows
from .synthetic import (
    REQUIRED_SYNTHETIC_CASES,
    all_cases,
    analytic_displacement,
    generate_frame,
    synthetic_partition,
    write_case_frames,
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


def _test_padding() -> None:
    source = ((1, 2, 3),)
    replicated = pad_rows(source, left=1, right=1, policy="replication")
    reflected = pad_rows(source, left=1, right=1, policy="reflect")
    assert replicated.rows == ((1, 1, 2, 3, 3),)
    assert reflected.rows == ((2, 1, 2, 3, 2),)
    assert replicated.crop == (1, 0, 3, 1)
    assert reflected.policy == "reflect"

    # Both policies preserve the source crop, while differing only in the caller-side
    # halo.  The runner passes the manifest-declared policy through this narrow seam.
    for padded in (replicated, reflected):
        assert padded.rows[0][padded.pad_left:padded.pad_left + padded.source_width] == source[0]


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

    blur = generate_frame("blur", 5)
    sharp = generate_frame("translation-x-positive", 5)
    assert blur != sharp
    noise_a = generate_frame("noise", 5)
    noise_b = generate_frame("noise", 5)
    assert noise_a == noise_b
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

    with tempfile.TemporaryDirectory(prefix="whitewater-p25-corpus-") as temporary:
        directory = Path(temporary)
        write_case_frames("identity", directory)
        assert (directory / "identity/frame.0000.pfm").is_file()
        truth = json.loads((directory / "identity/truth.json").read_text(encoding="utf-8"))
        assert truth["case_id"] == "identity"
        assert truth["coordinate_convention"]


def main() -> int:
    _test_conditioning()
    _test_padding()
    _test_synthetic_cases()
    _test_corpus_and_emission()
    print("P25-2 corpus and conditioning tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
