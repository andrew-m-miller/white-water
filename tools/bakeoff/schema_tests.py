#!/usr/bin/env python3
"""Machine test driver for the Phase 2.5 protocol, corpus and report contracts."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validator import (  # type: ignore  # pylint: disable=wrong-import-position
    ValidationError,
    canonical_sha256,
    load_json,
    _expected_analysis_dimensions,
    validate,
    validate_corpus_consistency,
    validate_protocol_and_schemas,
    validate_protocol_consistency,
    validate_report_consistency,
)


def expect_failure(label: str, callback) -> None:
    try:
        callback()
    except (ValidationError, ValueError):
        return
    raise AssertionError(f"{label}: invalid fixture was accepted")


def expect_error_path(label: str, callback, expected_path: str) -> None:
    try:
        callback()
    except ValidationError as exc:
        if exc.path != expected_path:
            raise AssertionError(f"{label}: expected path {expected_path}, got {exc.path}") from exc
        return
    raise AssertionError(f"{label}: invalid fixture was accepted")


def set_matrix(report, *, candidate_ids, shot_ids, conditioning_tokens, cap_tokens, providers) -> None:
    """Replace a report selector and bind its canonical hash for mutation tests."""

    selector = {
        "candidate_ids": list(candidate_ids),
        "shot_ids": list(shot_ids),
        "conditioning_tokens": list(conditioning_tokens),
        "cap_tokens": list(cap_tokens),
        "providers": copy.deepcopy(providers),
    }
    selector["matrix_sha256"] = canonical_sha256(selector)
    report["matrix"] = selector


def main() -> int:
    protocol = load_json(ROOT / "bakeoff/protocol-v1.json")
    protocol_schema = load_json(ROOT / "bakeoff/protocol-v1.schema.json")
    corpus_schema = load_json(ROOT / "bakeoff/corpus-v1.schema.json")
    report_schema = load_json(ROOT / "bakeoff/report-v1.schema.json")
    validate_protocol_and_schemas(protocol, protocol_schema, corpus_schema, report_schema)
    validate_protocol_consistency(protocol, protocol_schema, report_schema)

    positive_corpus = load_json(ROOT / "bakeoff/fixtures/positive/corpus-v1.json")
    validate_corpus_consistency(positive_corpus, protocol, corpus_schema)
    positive_report = load_json(ROOT / "bakeoff/fixtures/positive/report-v1.json")
    validate_report_consistency(positive_report, protocol, report_schema, positive_corpus, corpus_schema)

    # v1 remains the backward-compatible shipping-only contract.  The admission amendment is
    # exercised separately through the new v2 protocol/report IDs so an old consumer cannot
    # silently reinterpret an excluded candidate as measurable.
    protocol_v2 = load_json(ROOT / "bakeoff/protocol-v2.json")
    protocol_schema_v2 = load_json(ROOT / "bakeoff/protocol-v2.schema.json")
    report_schema_v2 = load_json(ROOT / "bakeoff/report-v2.schema.json")
    positive_report_v2 = load_json(ROOT / "bakeoff/fixtures/positive/report-v2.json")
    validate_protocol_and_schemas(protocol_v2, protocol_schema_v2, corpus_schema, report_schema_v2)
    validate_report_consistency(
        positive_report_v2, protocol_v2, report_schema_v2, positive_corpus, corpus_schema,
    )

    # The new sub-mp0.5 lattice is available to an ordinary measurable candidate.  The
    # positive row remains the small 64x48 fixture geometry; the cap is an upper bound, not a
    # forced resize for sources already below it.
    shared_lattice_report = copy.deepcopy(positive_report_v2)
    shared_lattice_report["matrix"]["cap_tokens"] = ["mp0_331776"]
    shared_lattice_report["matrix"]["matrix_sha256"] = canonical_sha256({
        key: value for key, value in shared_lattice_report["matrix"].items()
        if key != "matrix_sha256"
    })
    shared_lattice_report["results"][0]["cap_token"] = "mp0_331776"
    validate_report_consistency(
        shared_lattice_report, protocol_v2, report_schema_v2, positive_corpus, corpus_schema,
    )

    # P25-7: the CUDA arena ceiling (gpu_mem_limit_mib) is declared once on the matrix provider
    # selector and echoed by each CUDA result's resource evidence.  The two must agree exactly,
    # the option is CUDA-only, and a result may neither invent a ceiling the selector never
    # declared nor omit/disagree with the one it did.  A bounded-CUDA report is otherwise a
    # verbatim copy of the positive screen fixture (both share the frozen el8-x86_64 environment).
    pv2_matrix = positive_report_v2["matrix"]
    base_candidate_ids = list(pv2_matrix["candidate_ids"])
    base_shot_ids = list(pv2_matrix["shot_ids"])
    base_conditioning = list(pv2_matrix["conditioning_tokens"])
    base_caps = list(pv2_matrix["cap_tokens"])

    arena_base = copy.deepcopy(positive_report_v2)
    arena_base["candidates"][0]["measurement_providers"] = ["cpu", "cuda"]
    set_matrix(
        arena_base,
        candidate_ids=base_candidate_ids,
        shot_ids=base_shot_ids,
        conditioning_tokens=base_conditioning,
        cap_tokens=base_caps,
        providers=[{"token": "cuda", "host_loads": ["idle"], "gpu_mem_limit_mib": 22000}],
    )
    arena_base["results"][0].update({"provider": "cuda", "host_load": "idle"})
    arena_base["results"][0]["resource"]["gpu_mem_limit_mib"] = 22000
    validate_report_consistency(
        arena_base, protocol_v2, report_schema_v2, positive_corpus, corpus_schema,
    )

    # (a) A ceiling on a non-CUDA provider selector is rejected.
    cpu_matrix_ceiling = copy.deepcopy(positive_report_v2)
    set_matrix(
        cpu_matrix_ceiling,
        candidate_ids=base_candidate_ids,
        shot_ids=base_shot_ids,
        conditioning_tokens=base_conditioning,
        cap_tokens=base_caps,
        providers=[{"token": "cpu", "host_loads": ["not_applicable"], "gpu_mem_limit_mib": 4096}],
    )
    expect_error_path(
        "CPU matrix selector may not carry a gpu_mem_limit_mib ceiling",
        lambda: validate_report_consistency(
            cpu_matrix_ceiling, protocol_v2, report_schema_v2, positive_corpus, corpus_schema,
        ),
        "$.matrix.providers[0].gpu_mem_limit_mib",
    )

    # (a) A ceiling in a non-CUDA result's resource evidence is rejected.
    cpu_result_ceiling = copy.deepcopy(positive_report_v2)
    cpu_result_ceiling["results"][0]["resource"]["gpu_mem_limit_mib"] = 4096
    expect_error_path(
        "CPU result may not record a gpu_mem_limit_mib ceiling",
        lambda: validate_report_consistency(
            cpu_result_ceiling, protocol_v2, report_schema_v2, positive_corpus, corpus_schema,
        ),
        "$.results[0].resource.gpu_mem_limit_mib",
    )

    # (b) A CUDA result whose recorded ceiling disagrees with the selector is rejected.
    arena_mismatch = copy.deepcopy(arena_base)
    arena_mismatch["results"][0]["resource"]["gpu_mem_limit_mib"] = 4096
    expect_error_path(
        "CUDA result ceiling must equal the selector's ceiling",
        lambda: validate_report_consistency(
            arena_mismatch, protocol_v2, report_schema_v2, positive_corpus, corpus_schema,
        ),
        "$.results[0].resource.gpu_mem_limit_mib",
    )

    # (b) A CUDA result that omits the ceiling the selector declared is rejected.
    arena_missing = copy.deepcopy(arena_base)
    del arena_missing["results"][0]["resource"]["gpu_mem_limit_mib"]
    expect_error_path(
        "CUDA result must echo the selector's ceiling",
        lambda: validate_report_consistency(
            arena_missing, protocol_v2, report_schema_v2, positive_corpus, corpus_schema,
        ),
        "$.results[0].resource.gpu_mem_limit_mib",
    )

    # (c) A CUDA result carrying a ceiling the selector never declared is rejected.
    arena_stray = copy.deepcopy(arena_base)
    set_matrix(
        arena_stray,
        candidate_ids=base_candidate_ids,
        shot_ids=base_shot_ids,
        conditioning_tokens=base_conditioning,
        cap_tokens=base_caps,
        providers=[{"token": "cuda", "host_loads": ["idle"]}],
    )
    expect_error_path(
        "CUDA result may not invent a ceiling the selector omits",
        lambda: validate_report_consistency(
            arena_stray, protocol_v2, report_schema_v2, positive_corpus, corpus_schema,
        ),
        "$.results[0].resource.gpu_mem_limit_mib",
    )

    neuflow_wrong_geometry = copy.deepcopy(shared_lattice_report)
    neuflow_candidate = copy.deepcopy(neuflow_wrong_geometry["candidates"][0])
    neuflow_candidate.update({
        "candidate_id": "neuflow-v2",
        "status": "excluded",
        "measurement_status": "measurable",
        "measurement_providers": ["cpu"],
        "exclusion_reason": {"type": "license_unknown", "message": "fixed-shape fixture"},
    })
    neuflow_wrong_geometry["candidates"] = [neuflow_candidate]
    neuflow_wrong_geometry["matrix"]["candidate_ids"] = ["neuflow-v2"]
    neuflow_wrong_geometry["matrix"]["matrix_sha256"] = canonical_sha256({
        key: value for key, value in neuflow_wrong_geometry["matrix"].items()
        if key != "matrix_sha256"
    })
    neuflow_wrong_geometry["results"][0]["candidate_id"] = "neuflow-v2"
    expect_failure(
        "NeuFlow rejects non-16:9 fixed-lattice source geometry",
        lambda: validate_report_consistency(
            neuflow_wrong_geometry, protocol_v2, report_schema_v2, positive_corpus, corpus_schema,
        ),
    )

    neuflow_wrong_cap = copy.deepcopy(neuflow_wrong_geometry)
    neuflow_wrong_cap["matrix"]["cap_tokens"] = ["mp2"]
    neuflow_wrong_cap["matrix"]["providers"] = [{"token": "cuda", "host_loads": ["idle"]}]
    neuflow_wrong_cap["matrix"]["matrix_sha256"] = canonical_sha256({
        key: value for key, value in neuflow_wrong_cap["matrix"].items()
        if key != "matrix_sha256"
    })
    neuflow_wrong_cap["results"][0].update({
        "cap_token": "mp2", "provider": "cuda", "host_load": "idle",
    })
    neuflow_wrong_cap["candidates"][0]["measurement_providers"] = ["cpu", "cuda"]
    expect_failure(
        "NeuFlow rejects frozen shipping caps",
        lambda: validate_report_consistency(
            neuflow_wrong_cap, protocol_v2, report_schema_v2, positive_corpus, corpus_schema,
        ),
    )

    protocol_v2_bad_lattice = copy.deepcopy(protocol_v2)
    protocol_v2_bad_lattice["analysis_caps"][-1]["lattice"]["analysis_width"] = 769
    expect_failure(
        "v2 fixed lattice dimensions are frozen",
        lambda: validate_protocol_consistency(
            protocol_v2_bad_lattice, protocol_schema_v2, report_schema_v2,
        ),
    )

    excluded_but_measurable = copy.deepcopy(positive_report_v2)
    evaluation_candidate = copy.deepcopy(positive_report_v2["candidates"][0])
    evaluation_candidate.update({
        "candidate_id": "waft-twins",
        "status": "excluded",
        "measurement_status": "measurable",
        "exclusion_reason": {"type": "license_unknown", "message": "evaluation-only fixture"},
    })
    evaluation_candidate["license_verdicts"]["checkpoint"] = "unknown"
    evaluation_candidate["redistribution_permitted"]["checkpoint"] = "not_permitted"
    excluded_but_measurable["candidates"].append(evaluation_candidate)
    set_matrix(
        excluded_but_measurable,
        candidate_ids=["waft-twins"],
        shot_ids=["syn-identity"],
        conditioning_tokens=["native-clamp01-v1"],
        cap_tokens=["mp0_5"],
        providers=[{"token": "cpu", "host_loads": ["not_applicable"]}],
    )
    excluded_but_measurable["results"][0]["candidate_id"] = "waft-twins"
    validate_report_consistency(
        excluded_but_measurable, protocol_v2, report_schema_v2, positive_corpus, corpus_schema,
    )

    baseline_measurable = copy.deepcopy(excluded_but_measurable)
    baseline_candidate = copy.deepcopy(positive_report_v2["candidates"][0])
    baseline_candidate.update({
        "candidate_id": "raft-original",
        "status": "excluded",
        "measurement_status": "measurable",
        "exclusion_reason": {"type": "license_unknown", "message": "baseline fixture"},
    })
    baseline_measurable["candidates"].append(baseline_candidate)
    set_matrix(
        baseline_measurable,
        candidate_ids=["raft-original"],
        shot_ids=["syn-identity"],
        conditioning_tokens=["native-clamp01-v1"],
        cap_tokens=["mp0_5"],
        providers=[{"token": "cpu", "host_loads": ["not_applicable"]}],
    )
    baseline_measurable["results"][0]["candidate_id"] = "raft-original"
    validate_report_consistency(
        baseline_measurable, protocol_v2, report_schema_v2, positive_corpus, corpus_schema,
    )

    unavailable_in_matrix = copy.deepcopy(positive_report_v2)
    unavailable_candidate = copy.deepcopy(positive_report_v2["candidates"][0])
    unavailable_candidate.update({
        "candidate_id": "waft-twins",
        "status": "excluded",
        "measurement_status": "unavailable",
        "exclusion_reason": {"type": "license_unknown", "message": "shipping exclusion fixture"},
        "measurement_exclusion_reason": {
            "type": "artifact_missing",
            "message": "measurement unavailability fixture",
        },
    })
    unavailable_candidate.pop("measurement_providers")
    unavailable_in_matrix["candidates"].append(unavailable_candidate)
    unavailable_report = copy.deepcopy(unavailable_in_matrix)
    unavailable_report["matrix"]["candidate_ids"] = ["sea-raft-m"]
    unavailable_report["matrix"]["matrix_sha256"] = canonical_sha256(
        {key: value for key, value in unavailable_report["matrix"].items() if key != "matrix_sha256"}
    )
    unavailable_report["results"][0]["candidate_id"] = "sea-raft-m"
    validate_report_consistency(
        unavailable_report, protocol_v2, report_schema_v2, positive_corpus, corpus_schema,
    )
    unavailable_with_provider = copy.deepcopy(unavailable_report)
    unavailable_with_provider["candidates"][1]["measurement_providers"] = ["cpu"]
    expect_failure(
        "v2 unavailable candidate cannot carry provider evidence",
        lambda: validate_report_consistency(
            unavailable_with_provider, protocol_v2, report_schema_v2, positive_corpus, corpus_schema,
        ),
    )
    set_matrix(
        unavailable_in_matrix,
        candidate_ids=["waft-twins"],
        shot_ids=["syn-identity"],
        conditioning_tokens=["native-clamp01-v1"],
        cap_tokens=["mp0_5"],
        providers=[{"token": "cpu", "host_loads": ["not_applicable"]}],
    )
    unavailable_in_matrix["results"][0]["candidate_id"] = "waft-twins"
    expect_failure(
        "v2 unavailable candidate matrix selection",
        lambda: validate_report_consistency(
            unavailable_in_matrix, protocol_v2, report_schema_v2, positive_corpus, corpus_schema,
        ),
    )

    eligible_unavailable = copy.deepcopy(positive_report_v2)
    eligible_unavailable["candidates"][0]["measurement_status"] = "unavailable"
    expect_failure(
        "v2 shipping eligibility requires measurable artifact",
        lambda: validate_report_consistency(
            eligible_unavailable, protocol_v2, report_schema_v2, positive_corpus, corpus_schema,
        ),
    )

    v2_license_not_permitted = copy.deepcopy(positive_report_v2)
    v2_license_not_permitted["candidates"][0]["license_verdicts"]["checkpoint"] = "unknown"
    expect_failure(
        "v2 shipping eligibility retains the fail-closed license gate",
        lambda: validate_report_consistency(
            v2_license_not_permitted, protocol_v2, report_schema_v2, positive_corpus, corpus_schema,
        ),
    )

    baseline_as_shipping = copy.deepcopy(positive_report_v2)
    baseline_as_shipping["candidates"][0]["candidate_id"] = "raft-original"
    expect_failure(
        "v2 validation baseline cannot be shipping eligible",
        lambda: validate_report_consistency(
            baseline_as_shipping, protocol_v2, report_schema_v2, positive_corpus, corpus_schema,
        ),
    )

    measurable_without_artifact = copy.deepcopy(excluded_but_measurable)
    measurable_without_artifact["candidates"][1].pop("artifact_sha256")
    expect_failure(
        "v2 measurable candidate requires artifact identity",
        lambda: validate_report_consistency(
            measurable_without_artifact, protocol_v2, report_schema_v2, positive_corpus, corpus_schema,
        ),
    )

    measurable_without_legal_evidence = copy.deepcopy(excluded_but_measurable)
    measurable_without_legal_evidence["candidates"][1].pop("license_verdicts")
    expect_failure(
        "v2 excluded measurable candidate preserves license evidence",
        lambda: validate_report_consistency(
            measurable_without_legal_evidence,
            protocol_v2,
            report_schema_v2,
            positive_corpus,
            corpus_schema,
        ),
    )

    measurable_with_measurement_reason = copy.deepcopy(excluded_but_measurable)
    measurable_with_measurement_reason["candidates"][1]["measurement_exclusion_reason"] = {
        "type": "artifact_missing",
        "message": "must be unavailable instead",
    }
    expect_failure(
        "v2 measurable candidate cannot carry a measurement exclusion reason",
        lambda: validate_report_consistency(
            measurable_with_measurement_reason,
            protocol_v2,
            report_schema_v2,
            positive_corpus,
            corpus_schema,
        ),
    )

    unavailable_without_measurement_reason = copy.deepcopy(unavailable_report)
    unavailable_without_measurement_reason["candidates"][1].pop("measurement_exclusion_reason")
    expect_failure(
        "v2 unavailable candidate requires a measurement exclusion reason",
        lambda: validate_report_consistency(
            unavailable_without_measurement_reason,
            protocol_v2,
            report_schema_v2,
            positive_corpus,
            corpus_schema,
        ),
    )

    failed_with_metrics = copy.deepcopy(positive_report)
    failed_with_metrics["results"][0]["status"] = "fail"
    failed_with_metrics["results"][0]["failure"] = {
        "type": "runtime_error",
        "message": "fixture failure after metric collection",
    }
    failed_with_metrics["summary"] = {
        "required_cells": 1,
        "passed_cells": 0,
        "failed_cells": 1,
        "skipped_cells": 0,
    }
    validate_report_consistency(
        failed_with_metrics, protocol, report_schema, positive_corpus, corpus_schema,
    )
    failed_incomplete_metrics = copy.deepcopy(failed_with_metrics)
    failed_incomplete_metrics["results"][0]["metrics"]["not_applicable"].remove("chain_drift_px")
    expect_failure(
        "failed result incomplete metric disposition",
        lambda: validate_report_consistency(
            failed_incomplete_metrics, protocol, report_schema, positive_corpus, corpus_schema,
        ),
    )

    skipped_with_metrics = copy.deepcopy(positive_report)
    skipped_with_metrics["results"][0]["status"] = "skip"
    skipped_with_metrics["results"][0]["failure"] = {
        "type": "not_attempted",
        "message": "fixture skip after metric collection",
    }
    skipped_with_metrics["summary"] = {
        "required_cells": 1,
        "passed_cells": 0,
        "failed_cells": 0,
        "skipped_cells": 1,
    }
    validate_report_consistency(
        skipped_with_metrics, protocol, report_schema, positive_corpus, corpus_schema,
    )
    skipped_incomplete_metrics = copy.deepcopy(skipped_with_metrics)
    skipped_incomplete_metrics["results"][0]["metrics"]["not_applicable"].remove("chain_drift_px")
    expect_failure(
        "skipped result incomplete metric disposition",
        lambda: validate_report_consistency(
            skipped_incomplete_metrics, protocol, report_schema, positive_corpus, corpus_schema,
        ),
    )

    divergent_report_schema = copy.deepcopy(report_schema)
    divergent_report_schema["$defs"]["metrics"]["properties"]["endpoint_error"] = (
        divergent_report_schema["$defs"]["metrics"]["properties"].pop("endpoint_error_px")
    )
    expect_failure(
        "report metric schema divergence",
        lambda: validate_protocol_and_schemas(
            protocol, protocol_schema, corpus_schema, divergent_report_schema,
        ),
    )
    protocol_bad_metric = copy.deepcopy(protocol)
    protocol_bad_metric["metrics"][0] = "endpoint_error"
    expect_failure(
        "frozen report metric token",
        lambda: validate_protocol_consistency(protocol_bad_metric, protocol_schema, report_schema),
    )
    missing_dense_metric = copy.deepcopy(positive_report)
    missing_dense_metric["results"][0]["metrics"].pop("endpoint_error_px")
    expect_failure(
        "analytic dense metric requirement",
        lambda: validate_report_consistency(
            missing_dense_metric, protocol, report_schema, positive_corpus, corpus_schema,
        ),
    )

    for label, timestamp in (
        ("year zero date-time", "0000-01-01T00:00:00.123456789Z"),
        ("UTC leap second", "1990-12-31T23:59:60Z"),
        ("offset leap second", "1990-12-31T15:59:60.123456789-08:00"),
    ):
        valid_timestamp = copy.deepcopy(positive_report)
        valid_timestamp["started_utc"] = timestamp
        validate_report_consistency(
            valid_timestamp, protocol, report_schema, positive_corpus, corpus_schema,
        )

    for label, timestamp in (
        ("timezone-less date-time", "2026-08-22T10:00:00.5"),
        ("invalid calendar date", "2026-02-30T10:00:00Z"),
        ("invalid timezone offset", "2026-08-22T10:00:00+24:00"),
        ("misplaced leap second", "2026-07-01T00:00:60Z"),
    ):
        bad_timestamp = copy.deepcopy(positive_report)
        bad_timestamp["started_utc"] = timestamp
        expect_failure(
            label,
            lambda bad_timestamp=bad_timestamp: validate_report_consistency(
                bad_timestamp, protocol, report_schema, positive_corpus, corpus_schema,
            ),
        )

    unknown_metric_disposition = copy.deepcopy(positive_report)
    unknown_metric_disposition["results"][0]["metrics"]["not_applicable"].append("bogus_metric")
    expect_failure(
        "unknown metric disposition",
        lambda: validate_report_consistency(
            unknown_metric_disposition, protocol, report_schema, positive_corpus, corpus_schema,
        ),
    )
    overlapping_metric_disposition = copy.deepcopy(positive_report)
    overlapping_metric_disposition["results"][0]["metrics"]["not_applicable"].append("visible_warp_residual")
    expect_failure(
        "overlapping metric disposition",
        lambda: validate_report_consistency(
            overlapping_metric_disposition, protocol, report_schema, positive_corpus, corpus_schema,
        ),
    )
    undisposed_metric = copy.deepcopy(positive_report)
    undisposed_metric["results"][0]["metrics"]["not_applicable"].remove("chain_drift_px")
    expect_failure(
        "undisposed metric",
        lambda: validate_report_consistency(
            undisposed_metric, protocol, report_schema, positive_corpus, corpus_schema,
        ),
    )
    duplicate_metric_disposition = copy.deepcopy(positive_report)
    duplicate_metric_disposition["results"][0]["metrics"]["not_applicable"].append("chain_drift_px")
    expect_failure(
        "duplicate metric disposition",
        lambda: validate_report_consistency(
            duplicate_metric_disposition, protocol, report_schema, positive_corpus, corpus_schema,
        ),
    )

    chain_applicability_corpus = copy.deepcopy(positive_corpus)
    identity_chain_shot = chain_applicability_corpus["partitions"][0]["shots"][0]
    identity_chain_shot["chain_length"] = 1
    chain_applicability_report = copy.deepcopy(positive_report)
    chain_applicability_report["corpus_sha256"] = canonical_sha256(chain_applicability_corpus)
    expect_failure(
        "omitted chain metric",
        lambda: validate_report_consistency(
            chain_applicability_report, protocol, report_schema,
            chain_applicability_corpus, corpus_schema,
        ),
    )

    landmark_corpus = copy.deepcopy(positive_corpus)
    landmark_shot = copy.deepcopy(landmark_corpus["partitions"][0]["shots"][0])
    landmark_shot["id"] = "public-landmark"
    landmark_shot["path_pattern"] = "public://landmark"
    landmark_shot["truth"] = {"kind": "landmarks", "definition": "fixture landmark truth"}
    landmark_corpus["partitions"].append({
        "id": "public",
        "kind": "public",
        "terms": {
            "source": "fixture-public-dataset",
            "usage": "evaluation-only",
            "training_overlap": "none_found",
            "record": "fixture-terms-v1",
        },
        "shots": [landmark_shot],
    })
    landmark_report = copy.deepcopy(positive_report)
    landmark_report["corpus_sha256"] = canonical_sha256(landmark_corpus)
    set_matrix(
        landmark_report,
        candidate_ids=["sea-raft-m"],
        shot_ids=["public-landmark"],
        conditioning_tokens=["native-clamp01-v1"],
        cap_tokens=["mp0_5"],
        providers=[{"token": "cpu", "host_loads": ["not_applicable"]}],
    )
    landmark_report["results"][0]["shot_id"] = "public-landmark"
    expect_failure(
        "omitted landmark metrics",
        lambda: validate_report_consistency(
            landmark_report, protocol, report_schema, landmark_corpus, corpus_schema,
        ),
    )

    corpus_id_mismatch = copy.deepcopy(positive_report)
    corpus_id_mismatch["corpus_id"] = "other-corpus"
    expect_failure(
        "report corpus id binding",
        lambda: validate_report_consistency(corpus_id_mismatch, protocol, report_schema, positive_corpus, corpus_schema),
    )
    corpus_hash_mismatch = copy.deepcopy(positive_report)
    corpus_hash_mismatch["corpus_sha256"] = "0" * 64
    expect_failure(
        "report corpus hash binding",
        lambda: validate_report_consistency(corpus_hash_mismatch, protocol, report_schema, positive_corpus, corpus_schema),
    )
    expect_failure(
        "report requires corpus document",
        lambda: validate_report_consistency(positive_report, protocol, report_schema, None, corpus_schema),
    )

    negative_corpus = load_json(ROOT / "bakeoff/fixtures/negative/corpus-v1-unknown-category.json")
    expect_failure(
        "unknown corpus category",
        lambda: validate_corpus_consistency(negative_corpus, protocol, corpus_schema),
    )
    incomplete_corpus = copy.deepcopy(positive_corpus)
    incomplete_corpus["partitions"][0]["shots"].pop()
    expect_failure(
        "incomplete synthetic case coverage",
        lambda: validate_corpus_consistency(incomplete_corpus, protocol, corpus_schema),
    )
    public_without_terms = copy.deepcopy(positive_corpus)
    public_shot = copy.deepcopy(public_without_terms["partitions"][0]["shots"][0])
    public_shot["id"] = "public-identity"
    public_shot["path_pattern"] = "public/identity/plate.%04d.exr"
    public_without_terms["partitions"].append({
        "id": "public",
        "kind": "public",
        "shots": [public_shot],
    })
    expect_failure(
        "public corpus terms",
        lambda: validate_corpus_consistency(public_without_terms, protocol, corpus_schema),
    )
    public_with_terms = copy.deepcopy(public_without_terms)
    public_with_terms["partitions"][-1]["terms"] = {
        "source": "fixture-public-dataset",
        "usage": "evaluation-only",
        "training_overlap": "none_found",
        "record": "fixture-terms-v1",
    }
    validate_corpus_consistency(public_with_terms, protocol, corpus_schema)

    corpus_missing_metadata = copy.deepcopy(positive_corpus)
    corpus_missing_metadata["partitions"][0]["shots"][0].pop("width")
    expect_failure(
        "corpus shot metadata",
        lambda: validate_corpus_consistency(corpus_missing_metadata, protocol, corpus_schema),
    )
    malformed_corpus = copy.deepcopy(positive_corpus)
    malformed_corpus.pop("partitions")
    expect_failure(
        "malformed corpus missing partitions",
        lambda: validate_corpus_consistency(malformed_corpus, protocol, corpus_schema),
    )
    corpus_nonanalytic = copy.deepcopy(positive_corpus)
    corpus_nonanalytic["partitions"][0]["shots"][0]["truth"]["kind"] = "none"
    expect_failure(
        "synthetic analytic truth",
        lambda: validate_corpus_consistency(corpus_nonanalytic, protocol, corpus_schema),
    )
    corpus_non_exr = copy.deepcopy(positive_corpus)
    corpus_non_exr["partitions"][1]["shots"][0]["path_pattern"] = "shots/motion-blur/plate.mov"
    expect_failure(
        "production metadata-only EXR path",
        lambda: validate_corpus_consistency(corpus_non_exr, protocol, corpus_schema),
    )
    corpus_payload = copy.deepcopy(positive_corpus)
    corpus_payload["partitions"][0]["shots"][0]["generator_parameters"] = {
        "nested": {"pixel_data": [0.0, 1.0]},
    }
    expect_failure(
        "generator pixel payload",
        lambda: validate_corpus_consistency(corpus_payload, protocol, corpus_schema),
    )
    corpus_duplicate_frame_numbers = copy.deepcopy(positive_corpus)
    corpus_duplicate_frame_numbers["partitions"][0]["shots"][0]["frame_sha256"] = [
        {"frame": 3, "sha256": "a" * 64},
        {"frame": 3, "sha256": "b" * 64},
    ]
    expect_failure(
        "duplicate corpus frame numbers",
        lambda: validate_corpus_consistency(corpus_duplicate_frame_numbers, protocol, corpus_schema),
    )
    corpus_empty_truth = copy.deepcopy(positive_corpus)
    corpus_empty_truth["partitions"][0]["shots"][0]["truth"]["definition"] = ""
    expect_failure(
        "analytic truth definition",
        lambda: validate_corpus_consistency(corpus_empty_truth, protocol, corpus_schema),
    )
    corpus_short_chain = copy.deepcopy(positive_corpus)
    chain_shot = next(
        shot for shot in corpus_short_chain["partitions"][0]["shots"]
        if shot["case_id"] == "chain-8"
    )
    chain_shot["last_frame"] = chain_shot["first_frame"] + 7
    expect_failure(
        "chain frame range",
        lambda: validate_corpus_consistency(corpus_short_chain, protocol, corpus_schema),
    )
    corpus_bad_fhd = copy.deepcopy(positive_corpus)
    fhd_shot = next(
        shot for shot in corpus_bad_fhd["partitions"][0]["shots"]
        if shot["case_id"] == "fhd-1920x1080-par1"
    )
    fhd_shot["width"] = 1919
    expect_failure(
        "exact FHD performance dimensions",
        lambda: validate_corpus_consistency(corpus_bad_fhd, protocol, corpus_schema),
    )

    protocol_bad_cap = copy.deepcopy(protocol)
    protocol_bad_cap["analysis_caps"][2]["decimal_megapixels"] = 2.1
    expect_failure(
        "frozen cap numeric value",
        lambda: validate_protocol_consistency(protocol_bad_cap, protocol_schema),
    )
    protocol_bad_provider = copy.deepcopy(protocol)
    protocol_bad_provider["providers"][0]["cap_tokens"].append("mp1")
    expect_failure(
        "frozen provider mapping",
        lambda: validate_protocol_consistency(protocol_bad_provider, protocol_schema),
    )
    protocol_bad_formula = copy.deepcopy(protocol)
    protocol_bad_formula["conditioning"][1]["accepted_encoding"] = "log"
    expect_failure(
        "frozen conditioning encoding",
        lambda: validate_protocol_consistency(protocol_bad_formula, protocol_schema),
    )
    cli_bad_cap_protocol = copy.deepcopy(protocol)
    cli_bad_cap_protocol["analysis_caps"][2]["decimal_megapixels"] = 2.1
    expect_failure(
        "corpus/report protocol bundle cap gate",
        lambda: validate_protocol_and_schemas(
            cli_bad_cap_protocol, protocol_schema, corpus_schema, report_schema,
        ),
    )
    cli_bad_provider_protocol = copy.deepcopy(protocol)
    cli_bad_provider_protocol["providers"][0]["cap_tokens"].append("mp1")
    expect_failure(
        "corpus/report protocol bundle provider gate",
        lambda: validate_protocol_and_schemas(
            cli_bad_provider_protocol, protocol_schema, corpus_schema, report_schema,
        ),
    )
    for label, mutation in (
        ("required identity policy", lambda value: value["eligibility"].__setitem__("required_identity", ["source_commit"])),
        ("source target policy", lambda value: value["cap_accounting"].__setitem__("source_targets", ["fhd-1920x1080-par1"])),
        ("padding policy", lambda value: value.__setitem__("padding_comparison_policy", "caller-defined")),
        ("quantile policy", lambda value: value["quantile"].__setitem__("index", "h=n*p")),
        ("aggregation policy", lambda value: value["aggregation"].__setitem__("shot_weighting", "weighted")),
    ):
        mutated_protocol = copy.deepcopy(protocol)
        mutation(mutated_protocol)
        expect_failure(
            label,
            lambda mutated_protocol=mutated_protocol: validate_protocol_consistency(mutated_protocol, protocol_schema),
        )

    excluded_candidate = copy.deepcopy(positive_report)
    excluded_candidate["candidates"].append({
        "candidate_id": "waft-twins",
        "status": "excluded",
        "exclusion_reason": {
            "type": "license_not_permitted",
            "message": "fixture exclusion: redistribution terms are not permitted",
        },
    })
    validate_report_consistency(excluded_candidate, protocol, report_schema, positive_corpus, corpus_schema)
    negative_missing_failure = load_json(ROOT / "bakeoff/fixtures/negative/report-v1-missing-failure.json")
    expect_failure(
        "non-pass result without failure",
        lambda: validate_report_consistency(negative_missing_failure, protocol, report_schema, positive_corpus, corpus_schema),
    )
    negative_unknown_token = load_json(ROOT / "bakeoff/fixtures/negative/report-v1-unknown-token.json")
    expect_failure(
        "unknown conditioning token",
        lambda: validate_report_consistency(negative_unknown_token, protocol, report_schema, positive_corpus, corpus_schema),
    )
    negative_duplicate = load_json(ROOT / "bakeoff/fixtures/negative/report-v1-duplicate-cell.json")
    expect_failure(
        "duplicate result cell",
        lambda: validate_report_consistency(negative_duplicate, protocol, report_schema, positive_corpus, corpus_schema),
    )
    negative_runtime_hash = load_json(ROOT / "bakeoff/fixtures/negative/report-v1-missing-runtime-hash.json")
    expect_failure(
        "missing evaluator runtime hash",
        lambda: validate_report_consistency(negative_runtime_hash, protocol, report_schema, positive_corpus, corpus_schema),
    )
    negative_duplicate_candidates = load_json(ROOT / "bakeoff/fixtures/negative/report-v1-duplicate-candidates.json")
    expect_failure(
        "duplicate candidate fixture",
        lambda: validate_report_consistency(negative_duplicate_candidates, protocol, report_schema, positive_corpus, corpus_schema),
    )
    negative_missing_result = load_json(ROOT / "bakeoff/fixtures/negative/report-v1-missing-result.json")
    expect_failure(
        "missing result fixture",
        lambda: validate_report_consistency(negative_missing_result, protocol, report_schema, positive_corpus, corpus_schema),
    )
    missing_artifact_size = copy.deepcopy(positive_report)
    missing_artifact_size["candidates"][0].pop("artifact_size_bytes")
    expect_failure(
        "eligible candidate artifact size",
        lambda: validate_report_consistency(missing_artifact_size, protocol, report_schema, positive_corpus, corpus_schema),
    )
    bad_redistribution = copy.deepcopy(positive_report)
    bad_redistribution["candidates"][0]["redistribution_permitted"]["backbone"] = "unknown"
    expect_failure(
        "eligible redistribution verdict",
        lambda: validate_report_consistency(bad_redistribution, protocol, report_schema, positive_corpus, corpus_schema),
    )
    mixed_environment = copy.deepcopy(positive_report)
    set_matrix(
        mixed_environment,
        candidate_ids=["sea-raft-m"],
        shot_ids=["syn-identity"],
        conditioning_tokens=["native-clamp01-v1"],
        cap_tokens=["mp0_5"],
        providers=[
            {"token": "cpu", "host_loads": ["not_applicable"]},
            {"token": "coreml", "host_loads": ["not_applicable"]},
        ],
    )
    expect_failure(
        "mixed provider environments",
        lambda: validate_report_consistency(mixed_environment, protocol, report_schema, positive_corpus, corpus_schema),
    )
    bad_platform = copy.deepcopy(positive_report)
    bad_platform["hardware"]["platform"] = "macOS"
    expect_failure(
        "environment platform binding",
        lambda: validate_report_consistency(bad_platform, protocol, report_schema, positive_corpus, corpus_schema),
    )

    unsupported_cap = copy.deepcopy(positive_report)
    set_matrix(
        unsupported_cap,
        candidate_ids=["sea-raft-m"],
        shot_ids=["syn-identity"],
        conditioning_tokens=["native-clamp01-v1"],
        cap_tokens=["mp1"],
        providers=[{"token": "cpu", "host_loads": ["not_applicable"]}],
    )
    expect_error_path(
        "unsupported cap diagnostic path",
        lambda: validate_report_consistency(
            unsupported_cap, protocol, report_schema, positive_corpus, corpus_schema,
        ),
        "$.matrix.cap_tokens[0]",
    )

    # Focused report-identity checks.  These are kept as mutations of the complete positive
    # report so each failure reaches the intended cross-document rule rather than stopping at
    # an unrelated missing measurement field.
    duplicate_candidates = copy.deepcopy(positive_report)
    duplicate_candidates["candidates"].append(copy.deepcopy(duplicate_candidates["candidates"][0]))
    expect_failure(
        "duplicate candidate ids",
        lambda: validate_report_consistency(duplicate_candidates, protocol, report_schema, positive_corpus, corpus_schema),
    )

    summary_mismatch = copy.deepcopy(positive_report)
    summary_mismatch["summary"]["required_cells"] = 2
    expect_failure(
        "summary/result row mismatch",
        lambda: validate_report_consistency(summary_mismatch, protocol, report_schema, positive_corpus, corpus_schema),
    )

    mislabeled_cap = copy.deepcopy(positive_report)
    mislabeled_cap["results"][0]["geometry"]["analysis_width"] = 63
    expect_failure(
        "mislabeled analysis cap geometry",
        lambda: validate_report_consistency(mislabeled_cap, protocol, report_schema, positive_corpus, corpus_schema),
    )

    incompatible_conditioning = copy.deepcopy(positive_report)
    incompatible_conditioning["results"][0]["conditioning_token"] = "native-log-v1"
    set_matrix(
        incompatible_conditioning,
        candidate_ids=["sea-raft-m"],
        shot_ids=["syn-identity"],
        conditioning_tokens=["native-log-v1"],
        cap_tokens=["mp0_5"],
        providers=[{"token": "cpu", "host_loads": ["not_applicable"]}],
    )
    expect_failure(
        "conditioning and shot encoding compatibility",
        lambda: validate_report_consistency(incompatible_conditioning, protocol, report_schema, positive_corpus, corpus_schema),
    )

    percentile_report = copy.deepcopy(positive_report)
    percentile_report["results"][0]["conditioning_token"] = "pair-percentile-v1"
    percentile_report["results"][0]["conditioning_parameters"] = {
        "low": 0.1, "high": 0.9, "epsilon": 1e-6,
    }
    set_matrix(
        percentile_report,
        candidate_ids=["sea-raft-m"],
        shot_ids=["syn-identity"],
        conditioning_tokens=["pair-percentile-v1"],
        cap_tokens=["mp0_5"],
        providers=[{"token": "cpu", "host_loads": ["not_applicable"]}],
    )
    validate_report_consistency(percentile_report, protocol, report_schema, positive_corpus, corpus_schema)
    bad_percentile = copy.deepcopy(percentile_report)
    bad_percentile["results"][0]["conditioning_parameters"]["high"] = 0.1
    expect_failure(
        "invalid percentile conditioning parameters",
        lambda: validate_report_consistency(bad_percentile, protocol, report_schema, positive_corpus, corpus_schema),
    )

    cuda_hardware_missing = copy.deepcopy(positive_report)
    cuda_hardware_missing["hardware"].pop("gpu")
    cuda_hardware_missing["hardware"].pop("driver")
    set_matrix(
        cuda_hardware_missing,
        candidate_ids=["sea-raft-m"],
        shot_ids=["syn-identity"],
        conditioning_tokens=["native-clamp01-v1"],
        cap_tokens=["mp0_5"],
        providers=[{"token": "cuda", "host_loads": ["idle"]}],
    )
    expect_failure(
        "CUDA hardware identity",
        lambda: validate_report_consistency(cuda_hardware_missing, protocol, report_schema, positive_corpus, corpus_schema),
    )

    excluded_in_matrix = copy.deepcopy(positive_report)
    excluded_in_matrix["candidates"].append({
        "candidate_id": "waft-twins",
        "status": "excluded",
        "exclusion_reason": {"type": "license_not_permitted", "message": "excluded fixture"},
    })
    excluded_in_matrix["matrix"]["candidate_ids"] = ["waft-twins"]
    excluded_in_matrix["matrix"]["matrix_sha256"] = canonical_sha256({
        key: value for key, value in excluded_in_matrix["matrix"].items() if key != "matrix_sha256"
    })
    excluded_in_matrix["results"][0]["candidate_id"] = "waft-twins"
    expect_failure(
        "excluded candidate matrix selection",
        lambda: validate_report_consistency(excluded_in_matrix, protocol, report_schema, positive_corpus, corpus_schema),
    )

    bad_timing = copy.deepcopy(positive_report)
    bad_timing["results"][0]["timing"]["session_creation_ms"] = 0.3
    expect_failure(
        "timing median binding",
        lambda: validate_report_consistency(bad_timing, protocol, report_schema, positive_corpus, corpus_schema),
    )

    bad_spacing = copy.deepcopy(positive_report)
    bad_spacing["results"][0]["geometry"]["spacing_x_source_pixels"] = 2.0
    expect_failure(
        "spacing geometry binding",
        lambda: validate_report_consistency(bad_spacing, protocol, report_schema, positive_corpus, corpus_schema),
    )
    duplicate_input_frames = copy.deepcopy(positive_report)
    duplicate_input_frames["results"][0]["input_frames"][1]["frame"] = 3
    expect_failure(
        "distinct input frames",
        lambda: validate_report_consistency(duplicate_input_frames, protocol, report_schema, positive_corpus, corpus_schema),
    )
    out_of_range_input_frame = copy.deepcopy(positive_report)
    out_of_range_input_frame["results"][0]["input_frames"][0]["frame"] = 99
    expect_failure(
        "input frame range",
        lambda: validate_report_consistency(out_of_range_input_frame, protocol, report_schema, positive_corpus, corpus_schema),
    )
    corpus_with_frame_hashes = copy.deepcopy(positive_corpus)
    identity_shot = corpus_with_frame_hashes["partitions"][0]["shots"][0]
    identity_shot["frame_sha256"] = [
        {"frame": 3, "sha256": "a" * 64},
        {"frame": 4, "sha256": "b" * 64},
    ]
    report_with_frame_hashes = copy.deepcopy(positive_report)
    report_with_frame_hashes["corpus_sha256"] = canonical_sha256(corpus_with_frame_hashes)
    validate_report_consistency(report_with_frame_hashes, protocol, report_schema, corpus_with_frame_hashes, corpus_schema)
    bad_frame_hash = copy.deepcopy(report_with_frame_hashes)
    bad_frame_hash["results"][0]["input_frames"][0]["sha256"] = "c" * 64
    expect_failure(
        "input frame hash binding",
        lambda: validate_report_consistency(bad_frame_hash, protocol, report_schema, corpus_with_frame_hashes, corpus_schema),
    )

    missing_result = copy.deepcopy(positive_report)
    set_matrix(
        missing_result,
        candidate_ids=["sea-raft-m"],
        shot_ids=["syn-identity", "syn-translation-x-positive"],
        conditioning_tokens=["native-clamp01-v1"],
        cap_tokens=["mp0_5"],
        providers=[{"token": "cpu", "host_loads": ["not_applicable"]}],
    )
    expect_failure(
        "missing declared matrix result",
        lambda: validate_report_consistency(missing_result, protocol, report_schema, positive_corpus, corpus_schema),
    )

    final_cuda_unpaired = copy.deepcopy(positive_report)
    set_matrix(
        final_cuda_unpaired,
        candidate_ids=["sea-raft-m"],
        shot_ids=["syn-fhd-1920x1080-par1", "syn-uhd-3840x2160-par1"],
        conditioning_tokens=["native-clamp01-v1"],
        cap_tokens=["mp2"],
        providers=[{"token": "cuda", "host_loads": ["idle", "live_flame"]}],
    )
    final_cuda_unpaired["profile"] = "final"

    def make_final_result(shot_id, host_load):
        result = copy.deepcopy(positive_report["results"][0])
        shot = next(
            shot for partition in positive_corpus["partitions"]
            for shot in partition["shots"] if shot["id"] == shot_id
        )
        analysis_width, analysis_height = _expected_analysis_dimensions(
            shot["width"], shot["height"], shot["pixel_aspect_ratio"], 2.0,
        )
        result.update({
            "shot_id": shot_id,
            "provider": "cuda",
            "cap_token": "mp2",
            "host_load": host_load,
        })
        result["geometry"].update({
            "source_width": shot["width"],
            "source_height": shot["height"],
            "source_pixel_aspect_ratio": shot["pixel_aspect_ratio"],
            "canonical_width": shot["width"] * shot["pixel_aspect_ratio"],
            "canonical_height": shot["height"],
            "analysis_width": analysis_width,
            "analysis_height": analysis_height,
            "padded_width": analysis_width,
            "padded_height": analysis_height,
            "effective_padded_megapixels": analysis_width * analysis_height / 1_000_000.0,
            "spacing_x_source_pixels": shot["width"] / analysis_width,
            "spacing_y_source_pixels": shot["height"] / analysis_height,
        })
        result["timing"]["session_creation_ms"] = 0.2
        result["timing"]["first_inference_ms"] = 1.2
        result["timing"]["steady_inference_ms"] = 1.0
        result["timing"]["total_pair_ms"] = 1.2
        result["timing"]["steady_samples_ms"] = [1.0] * 30
        result["timing"]["sessions"] = [
            {
                "session_index": session_index,
                "warmup_recorded": True,
                "warmup_ms": 1.0,
                "session_creation_ms": 0.2,
                "first_inference_ms": 1.2,
                "steady_samples_ms": [1.0] * 10,
            }
            for session_index in range(3)
        ]
        result["resource"]["nvml_samples"] = [
            {"stage": stage, "used_mib": 100.0}
            for stage in ("baseline", "session_create", "steady", "cleanup", "process_exit")
        ]
        return result

    final_targets = ["syn-fhd-1920x1080-par1", "syn-uhd-3840x2160-par1"]
    final_cuda_unpaired["results"] = [make_final_result(shot_id, "idle") for shot_id in final_targets]
    final_cuda_unpaired["summary"] = {
        "required_cells": 2,
        "passed_cells": 2,
        "failed_cells": 0,
        "skipped_cells": 0,
    }
    expect_failure(
        "final CUDA idle/live pairing",
        lambda: validate_report_consistency(final_cuda_unpaired, protocol, report_schema, positive_corpus, corpus_schema),
    )

    final_cuda_mixed_status = copy.deepcopy(final_cuda_unpaired)
    for idle_failure in final_cuda_mixed_status["results"]:
        idle_failure["status"] = "fail"
        idle_failure["failure"] = {"type": "out_of_memory", "message": "idle fixture failure"}
    final_cuda_mixed_status["results"].extend(
        make_final_result(shot_id, "live_flame") for shot_id in final_targets
    )
    final_cuda_mixed_status["summary"] = {
        "required_cells": 4,
        "passed_cells": 2,
        "failed_cells": 2,
        "skipped_cells": 0,
    }
    validate_report_consistency(final_cuda_mixed_status, protocol, report_schema, positive_corpus, corpus_schema)

    final_cuda_missing_nvml = copy.deepcopy(final_cuda_mixed_status)
    final_cuda_missing_nvml["results"][2]["resource"]["nvml_samples"] = [
        sample
        for sample in final_cuda_missing_nvml["results"][2]["resource"]["nvml_samples"]
        if sample["stage"] != "process_exit"
    ]
    expect_failure(
        "final CUDA pass NVML stages",
        lambda: validate_report_consistency(final_cuda_missing_nvml, protocol, report_schema, positive_corpus, corpus_schema),
    )

    report_with_selection = copy.deepcopy(positive_report)
    report_with_selection["selection"] = {"default": {"candidate_id": "sea-raft-m"}}
    expect_failure(
        "selection is a separate P25-7 record",
        lambda: validate_report_consistency(report_with_selection, protocol, report_schema, positive_corpus, corpus_schema),
    )

    # Exercise the schema-level unknown-property gate and the no-OFX-index rule without
    # adding another bulky fixture whose only difference is one key.
    protocol_with_index = copy.deepcopy(protocol)
    protocol_with_index["default_index"] = 0
    expect_failure(
        "persistent OFX option index",
        lambda: validate_protocol_consistency(protocol_with_index, protocol_schema),
    )
    report_with_payload = copy.deepcopy(positive_report)
    report_with_payload["results"][0]["source_pixels"] = [0.0]
    expect_failure(
        "embedded source pixels",
        lambda: validate_report_consistency(report_with_payload, protocol, report_schema, positive_corpus, corpus_schema),
    )
    malformed_report = copy.deepcopy(positive_report)
    malformed_report.pop("matrix")
    expect_failure(
        "malformed report missing matrix",
        lambda: validate_report_consistency(
            malformed_report, protocol, report_schema, positive_corpus, corpus_schema,
        ),
    )

    print("Phase 2.5 protocol/schema fixtures passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
