#!/usr/bin/env python3
"""Focused deterministic matrix-planner tests."""

from __future__ import annotations

from .matrix import CellKey, MatrixFailure, build_matrix
from .validator import canonical_sha256


def _protocol() -> dict:
    return {
        "candidate_ids": [
            {"id": "candidate-a", "role": "shipping-candidate"},
            {"id": "candidate-b", "role": "shipping-candidate"},
            {"id": "candidate-x", "role": "shipping-candidate"},
        ],
        "conditioning": [
            {"token": "cond-a"},
            {"token": "cond-b"},
        ],
        "analysis_caps": [
            {"token": "mp0_5"},
            {"token": "mp2"},
        ],
        "providers": [
            {"token": "cpu", "environment": "el8-x86_64", "cap_tokens": ["mp0_5"]},
            {"token": "cuda", "environment": "el8-x86_64", "cap_tokens": ["mp0_5", "mp2"]},
            {"token": "coreml", "environment": "macos-arm64", "cap_tokens": ["mp0_5"]},
        ],
    }


def _corpus() -> dict:
    return {
        "partitions": [
            {
                "kind": "synthetic",
                "shots": [
                    {"id": "identity", "case_id": "identity", "width": 8, "height": 8, "pixel_aspect_ratio": 1.0},
                    {"id": "fhd", "case_id": "fhd-1920x1080-par1", "width": 1920, "height": 1080, "pixel_aspect_ratio": 1.0},
                ],
            },
            {
                "kind": "production_external",
                "shots": [
                    {"id": "uhd", "case_id": "uhd-3840x2160-par1", "width": 3840, "height": 2160, "pixel_aspect_ratio": 1.0},
                ],
            },
        ]
    }


def _candidates() -> list[dict[str, str]]:
    return [
        {"candidate_id": "candidate-a", "status": "eligible"},
        {"candidate_id": "candidate-b", "status": "eligible"},
        {"candidate_id": "candidate-x", "status": "excluded"},
    ]


def _selections(**overrides) -> dict:
    result = {
        "candidate_ids": ["candidate-b", "candidate-a"],
        "shot_ids": ["uhd", "identity", "fhd"],
        "conditioning_tokens": ["cond-b", "cond-a"],
        "cap_tokens": ["mp2"],
        "providers": [{"token": "cuda", "host_loads": ["live_flame", "idle"]}],
    }
    result.update(overrides)
    return result


def _v2_protocol() -> dict:
    protocol = _protocol()
    protocol["protocol_id"] = "whitewater-p25-v2"
    return protocol


def _v2_candidates() -> list[dict[str, object]]:
    return [
        {"candidate_id": "candidate-a", "status": "eligible", "measurement_status": "measurable"},
        # Shipping-excluded but technically qualified: valid evaluation input.
        {
            "candidate_id": "candidate-b",
            "status": "excluded",
            "measurement_status": "measurable",
            "exclusion_reason": {"type": "license_unknown", "message": "shipping-only exclusion"},
        },
        {
            "candidate_id": "candidate-x",
            "status": "excluded",
            "measurement_status": "unavailable",
            "exclusion_reason": {"type": "license_unknown", "message": "shipping-only exclusion"},
            "measurement_exclusion_reason": {"type": "artifact_missing", "message": "not measurable"},
        },
    ]


def _failure(kind: str, callback) -> None:
    try:
        callback()
    except MatrixFailure as failure:
        assert failure.kind == kind, (failure.kind, kind)
        assert failure.reason == kind
        assert failure.failure_type == "matrix_failure"
    else:
        raise AssertionError(f"expected MatrixFailure({kind})")


def test_smoke_and_final() -> None:
    smoke = build_matrix(
        _protocol(), _corpus(), _candidates(),
        _selections(
            shot_ids=["identity"],
            conditioning_tokens=["cond-a"],
            cap_tokens=["mp0_5"],
            providers=[{"token": "cpu", "host_loads": ["not_applicable"]}],
        ),
        "smoke",
        "el8-x86_64",
    )
    assert smoke.excluded_candidate_ids == ("candidate-x",)
    assert smoke.selector["matrix_sha256"] == canonical_sha256({
        key: value for key, value in smoke.selector.items() if key != "matrix_sha256"
    })
    assert smoke.cells == (CellKey("candidate-a", "identity", "cond-a", "mp0_5", "cpu", "not_applicable"), CellKey("candidate-b", "identity", "cond-a", "mp0_5", "cpu", "not_applicable"))

    final = build_matrix(
        _protocol(), _corpus(), _candidates(),
        _selections(candidate_ids=["candidate-a"], shot_ids=["fhd", "uhd"]),
        "final",
        "el8-x86_64",
    )
    assert len(final.cells) == 8
    assert {cell.host_load for cell in final.cells} == {"idle", "live_flame"}
    assert final.cells[0] == CellKey("candidate-a", "fhd", "cond-a", "mp2", "cuda", "idle")


def test_reordered_selections_normalize_to_one_plan() -> None:
    first = build_matrix(_protocol(), _corpus(), _candidates(), _selections(), "final", "el8-x86_64")
    reversed_selection = _selections(
        candidate_ids=["candidate-a", "candidate-b"],
        shot_ids=["fhd", "identity", "uhd"],
        conditioning_tokens=["cond-a", "cond-b"],
        providers=[{"token": "cuda", "host_loads": ["idle", "live_flame"]}],
    )
    second = build_matrix(_protocol(), _corpus(), _candidates(), reversed_selection, "final", "el8-x86_64")
    assert first.selector == second.selector
    assert first.cells == second.cells


def test_negative_selection_and_provider_rules() -> None:
    _failure("unknown_selection", lambda: build_matrix(_protocol(), _corpus(), _candidates(), _selections(shot_ids=["nope"]), "final", "el8-x86_64"))
    _failure("duplicate_selection", lambda: build_matrix(_protocol(), _corpus(), _candidates(), _selections(cap_tokens=["mp2", "mp2"]), "final", "el8-x86_64"))
    _failure("excluded_candidate", lambda: build_matrix(_protocol(), _corpus(), _candidates(), _selections(candidate_ids=["candidate-x"]), "final", "el8-x86_64"))
    _failure("unknown_candidate", lambda: build_matrix(_protocol(), _corpus(), _candidates(), _selections(candidate_ids=["not-a-candidate"]), "final", "el8-x86_64"))
    _failure("provider_cap", lambda: build_matrix(_protocol(), _corpus(), _candidates(), _selections(cap_tokens=["mp2"], providers=[{"token": "cpu", "host_loads": ["not_applicable"]}], shot_ids=["identity"]), "smoke", "el8-x86_64"))
    _failure("environment", lambda: build_matrix(_protocol(), _corpus(), _candidates(), _selections(providers=[{"token": "coreml", "host_loads": ["not_applicable"]}], cap_tokens=["mp0_5"], shot_ids=["identity"]), "smoke", "el8-x86_64"))
    _failure("host_load", lambda: build_matrix(_protocol(), _corpus(), _candidates(), _selections(providers=[{"token": "cpu", "host_loads": ["idle"]}], cap_tokens=["mp0_5"], shot_ids=["identity"]), "smoke", "el8-x86_64"))
    _failure("host_load", lambda: build_matrix(_protocol(), _corpus(), _candidates(), _selections(shot_ids=["fhd", "uhd"], providers=[{"token": "cuda", "host_loads": ["idle"]}]), "final", "el8-x86_64"))
    _failure("final_coverage", lambda: build_matrix(_protocol(), _corpus(), _candidates(), _selections(shot_ids=["fhd"]), "final", "el8-x86_64"))
    _failure("selection_shape", lambda: build_matrix(_protocol(), _corpus(), _candidates(), {**_selections(), "extra": True}, "final", "el8-x86_64"))


def test_duplicate_candidate_entries_and_exclusion() -> None:
    duplicate = _candidates() + [{"candidate_id": "candidate-a", "status": "eligible"}]
    _failure("duplicate_candidate", lambda: build_matrix(_protocol(), _corpus(), duplicate, _selections(), "final", "el8-x86_64"))
    excluded = _candidates()
    excluded[0] = {"candidate_id": "candidate-a", "status": "excluded"}
    _failure("excluded_candidate", lambda: build_matrix(_protocol(), _corpus(), excluded, _selections(candidate_ids=["candidate-a"]), "final", "el8-x86_64"))


def test_v2_measurement_admission_is_independent_of_shipping_status() -> None:
    evaluation = build_matrix(
        _v2_protocol(), _corpus(), _v2_candidates(), _selections(candidate_ids=["candidate-b"]),
        "final", "el8-x86_64",
    )
    assert {cell.candidate for cell in evaluation.cells} == {"candidate-b"}
    assert evaluation.excluded_candidate_ids == ("candidate-x",)
    _failure(
        "unavailable_candidate",
        lambda: build_matrix(
            _v2_protocol(), _corpus(), _v2_candidates(),
            _selections(candidate_ids=["candidate-x"]), "final", "el8-x86_64",
        ),
    )
    shipping_unavailable = _v2_candidates()
    shipping_unavailable[0]["measurement_status"] = "unavailable"
    _failure(
        "candidate_status",
        lambda: build_matrix(
            _v2_protocol(), _corpus(), shipping_unavailable,
            _selections(candidate_ids=["candidate-b"]), "final", "el8-x86_64",
        ),
    )
    baseline_protocol = _v2_protocol()
    baseline_protocol["candidate_ids"][1]["role"] = "validation-baseline"
    baseline_candidates = _v2_candidates()
    baseline_candidates[1]["status"] = "eligible"
    _failure(
        "candidate_role",
        lambda: build_matrix(
            baseline_protocol, _corpus(), baseline_candidates,
            _selections(candidate_ids=["candidate-b"]), "final", "el8-x86_64",
        ),
    )


def main() -> int:
    test_smoke_and_final()
    test_reordered_selections_normalize_to_one_plan()
    test_negative_selection_and_provider_rules()
    test_duplicate_candidate_entries_and_exclusion()
    test_v2_measurement_admission_is_independent_of_shipping_status()
    print("P25-4 matrix tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
