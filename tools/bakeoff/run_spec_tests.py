#!/usr/bin/env python3
"""Direct tests for the immutable bake-off RunSpec identity boundary."""

from __future__ import annotations

import copy
import math
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType

try:
    from .run_spec import (
        IDENTITY_SCHEMA_VERSION,
        KNOWN_STABLE_INPUTS,
        PublicationMetadata,
        RunSpec,
        RunSpecError,
        canonical_json,
        canonical_sha256,
    )
except ImportError:  # pragma: no cover - supports direct air-gapped invocation
    from run_spec import (  # type: ignore
        IDENTITY_SCHEMA_VERSION,
        KNOWN_STABLE_INPUTS,
        PublicationMetadata,
        RunSpec,
        RunSpecError,
        canonical_json,
        canonical_sha256,
    )


def _stable_fixture() -> dict[str, object]:
    return {
        "identity_schema_version": IDENTITY_SCHEMA_VERSION,
        "protocol": {"protocol_id": "whitewater-p25-v2", "caps": {"mp1": 1.0}},
        "corpus": {"corpus_id": "fixture", "shots": ["identity", "chain-1"]},
        "candidate_entries": [{"candidate_id": "candidate-a", "artifact_sha256": "a" * 64}],
        "selection": {
            "candidate_ids": ["candidate-a"],
            "shot_ids": ["identity"],
            "conditioning_tokens": ["native-clamp01-v1"],
            "cap_tokens": ["mp1"],
            "providers": [{"token": "cpu", "host_loads": ["not_applicable"]}],
        },
        "matrix": {"matrix_sha256": "b" * 64, "cells": ["candidate-a/identity/cpu"]},
        "artifacts": {
            "candidate-a": {
                "manifest_sha256": "c" * 64,
                "artifact_sha256": "a" * 64,
                "platform": "el8-x86_64",
            }
        },
        "report_schema": {"$id": "whitewater://schema/report-v2"},
        "corpus_schema": {"$id": "whitewater://schema/corpus-v1"},
        "environment": "el8-x86_64",
        "profile": "final",
        "runner": {
            "name": "ww-bakeoff",
            "version": "0.1.0",
            "source_commit": "d" * 40,
            "evaluator_sha256": "e" * 64,
            "runtime": "onnxruntime-1.29",
            "runtime_sha256": "f" * 64,
        },
        "hardware": {
            "platform": "linux",
            "architecture": "x86_64",
            "gpu": "fixture-gpu",
            "driver": "fixture-driver",
        },
        "chain_offsets": [1, 2, 4, 8],
        "measurement": {"device_index": 0, "poll_interval_s": 0.05, "nvml_enabled": True},
        "report_inputs": {
            "warnings": ["fixture warning"],
            "summary": {"final_quality_score": 91.5},
        },
    }


def _publication_fixture() -> dict[str, object]:
    return {
        "report_id": "p25-6-final-20260825t120000z",
        "started_utc": "2026-08-25T12:00:00+00:00",
        "completed_utc": "2026-08-25T12:30:00+00:00",
        "command": "ww-bakeoff --selection final.json",
        "output_dir": "/measurements/final",
        "state_path": "/measurements/final/state.json",
        "runner_log_path": "/measurements/final/runner.log",
    }


def _expect_run_spec_error(action, kind: str | None = None) -> None:
    try:
        action()
    except RunSpecError as exc:
        if kind is not None:
            assert exc.kind == kind, (exc.kind, exc)
    else:
        raise AssertionError("expected RunSpecError")


def test_canonical_json_is_order_independent_and_hashes_utf8_bytes() -> None:
    first = {"z": ["é", 2], "a": {"b": True, "a": None}}
    second = {"a": {"a": None, "b": True}, "z": ["é", 2]}
    assert canonical_json(first) == '{"a":{"a":null,"b":true},"z":["é",2]}'
    assert canonical_json(first) == canonical_json(second)
    assert canonical_sha256(first) == canonical_sha256(second)


def test_canonicalization_rejects_non_json_and_nonfinite_values() -> None:
    _expect_run_spec_error(lambda: canonical_json((1, 2)), "json_value")
    _expect_run_spec_error(lambda: canonical_json({"values": {1, 2}}), "json_value")
    _expect_run_spec_error(lambda: canonical_json({"bytes": b"x"}), "json_value")
    _expect_run_spec_error(lambda: canonical_json({1: "non-string key"}), "json_value")
    _expect_run_spec_error(lambda: canonical_json({"nan": math.nan}), "nonfinite")
    _expect_run_spec_error(lambda: canonical_json({"inf": math.inf}), "nonfinite")
    _expect_run_spec_error(lambda: canonical_sha256({"surrogate": "\ud800"}), "json_value")
    _expect_run_spec_error(lambda: PublicationMetadata({"surrogate": "\ud800"}), "json_value")
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    _expect_run_spec_error(lambda: canonical_json(cyclic), "json_value")
    _expect_run_spec_error(lambda: PublicationMetadata({"warnings": ["semantic"]}), "publication_field")
    _expect_run_spec_error(lambda: PublicationMetadata({"summary": {"score": 1}}), "publication_field")


def test_run_spec_rejects_non_plain_top_level_objects() -> None:
    _expect_run_spec_error(lambda: RunSpec(MappingProxyType(_stable_fixture())), "json_value")
    _expect_run_spec_error(lambda: RunSpec(tuple(_stable_fixture().items())), "json_value")


def test_every_known_stable_field_is_present_in_the_fixture() -> None:
    stable = _stable_fixture()
    assert set(KNOWN_STABLE_INPUTS) == set(stable)


def test_every_spec_serializes_the_default_identity_schema_version() -> None:
    stable = _stable_fixture()
    stable.pop("identity_schema_version")
    spec = RunSpec.from_mapping(stable)
    assert spec.stable_inputs["identity_schema_version"] == IDENTITY_SCHEMA_VERSION
    assert f'"identity_schema_version":{IDENTITY_SCHEMA_VERSION}' in spec.canonical_json


def test_identity_schema_version_is_stable_and_identity_bearing() -> None:
    baseline = RunSpec.from_mapping(_stable_fixture())
    mutated = _stable_fixture()
    mutated["identity_schema_version"] = IDENTITY_SCHEMA_VERSION + 1
    assert RunSpec.from_mapping(mutated).identity_sha256 != baseline.identity_sha256
    _expect_run_spec_error(
        lambda: RunSpec.from_mapping({**_stable_fixture(), "identity_schema_version": 0}),
        "identity_schema_version",
    )


def test_identity_assertion_centrally_checks_canonical_stable_inputs() -> None:
    spec = RunSpec.from_mapping(_stable_fixture())
    spec.assert_identity()
    object.__setattr__(spec, "_identity_hash", "0" * 64)
    _expect_run_spec_error(spec.assert_identity, "identity_hash")


def test_semantic_report_warnings_and_summary_change_identity() -> None:
    baseline = RunSpec.from_mapping(_stable_fixture())
    for report_field in ("warnings", "summary"):
        mutated = copy.deepcopy(_stable_fixture())
        report_inputs = mutated["report_inputs"]
        assert isinstance(report_inputs, dict)
        if report_field == "warnings":
            report_inputs[report_field].append("operator changed warning")
        else:
            report_inputs[report_field]["final_quality_score"] = 92.0
        assert RunSpec.from_mapping(mutated).identity_sha256 != baseline.identity_sha256, report_field


def test_every_stable_field_changes_identity() -> None:
    baseline = RunSpec.from_mapping(_stable_fixture(), publication=_publication_fixture())
    baseline_hash = baseline.identity_sha256

    for field_name in KNOWN_STABLE_INPUTS:
        mutated = copy.deepcopy(_stable_fixture())
        value = mutated[field_name]
        if isinstance(value, dict):
            value["__mutation__"] = field_name
        elif isinstance(value, list):
            value.append(f"mutation:{field_name}")
        elif isinstance(value, bool):
            mutated[field_name] = not value
        elif isinstance(value, (int, float)):
            mutated[field_name] = value + 1
        else:
            mutated[field_name] = f"{value}-mutated"
        changed = RunSpec.from_mapping(mutated, publication=_publication_fixture())
        assert changed.identity_sha256 != baseline_hash, field_name


def test_every_volatile_publication_field_does_not_change_identity() -> None:
    stable = _stable_fixture()
    baseline = RunSpec.from_mapping(stable, publication=_publication_fixture())
    baseline_hash = baseline.identity_sha256

    for field_name in _publication_fixture():
        mutated_publication = copy.deepcopy(_publication_fixture())
        value = mutated_publication[field_name]
        if isinstance(value, dict):
            value["__mutation__"] = field_name
        elif isinstance(value, list):
            value.append(f"mutation:{field_name}")
        elif isinstance(value, bool):
            mutated_publication[field_name] = not value
        elif isinstance(value, (int, float)):
            mutated_publication[field_name] = value + 1
        else:
            mutated_publication[field_name] = f"{value}-mutated"
        changed = RunSpec.from_mapping(stable, publication=mutated_publication)
        assert changed.identity_sha256 == baseline_hash, field_name


def test_publication_is_separate_in_persistence_record() -> None:
    spec = RunSpec.from_mapping(_stable_fixture(), publication=_publication_fixture())
    without_publication = spec.as_record()
    with_publication = spec.as_record(include_publication=True)
    assert set(without_publication) == {"stable", "identity_sha256"}
    assert set(with_publication) == {"stable", "identity_sha256", "publication"}
    assert "report_id" not in spec.canonical_json
    assert with_publication["publication"] == _publication_fixture()
    assert with_publication["identity_sha256"] == canonical_sha256(with_publication["stable"])


def test_run_spec_is_immutable_and_defensively_copies_inputs() -> None:
    stable = _stable_fixture()
    publication = _publication_fixture()
    spec = RunSpec.from_mapping(stable, publication=publication)
    original_hash = spec.identity_sha256

    stable["profile"] = "smoke"
    publication["report_id"] = "changed-after-construction"
    assert spec.identity_sha256 == original_hash
    assert spec.stable_inputs["profile"] == "final"
    assert spec.publication.as_dict()["report_id"] == "p25-6-final-20260825t120000z"

    returned_stable = spec.stable_inputs
    returned_stable["profile"] = "changed-copy"
    returned_publication = spec.publication.as_dict()
    returned_publication["report_id"] = "changed-copy"
    assert spec.identity_sha256 == original_hash
    assert spec.stable_inputs["profile"] == "final"
    assert spec.publication.as_dict()["report_id"] == "p25-6-final-20260825t120000z"

    try:
        spec.publication = PublicationMetadata({})  # type: ignore[misc]
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("RunSpec must be immutable")


def test_from_inputs_hashes_extra_stable_fields() -> None:
    stable = _stable_fixture()
    spec = RunSpec.from_inputs(**stable, extra_stable={"future_algorithm_switch": "v2"})
    assert spec.stable_inputs["future_algorithm_switch"] == "v2"
    altered = RunSpec.from_inputs(**stable, extra_stable={"future_algorithm_switch": "v3"})
    assert altered.identity_sha256 != spec.identity_sha256
    _expect_run_spec_error(
        lambda: RunSpec.from_inputs(**stable, extra_stable={"profile": "shadowed"}),
        "stable_field",
    )


def test_with_publication_preserves_identity() -> None:
    first = RunSpec.from_mapping(_stable_fixture(), publication={"report_id": "first"})
    second = first.with_publication({"report_id": "second", "completed_utc": "later"})
    assert second.identity_sha256 == first.identity_sha256
    assert first.publication.as_dict() == {"report_id": "first"}
    assert second.publication.as_dict()["report_id"] == "second"


def test_runner_and_validator_share_per_cell_hard_gate_specification() -> None:
    """Adding a per-cell gate to validator's public spec reaches both consumers."""

    from . import run as run_module
    from . import validator as validator_module

    assert run_module.validator_module is validator_module
    original_gates = validator_module.PER_CELL_HARD_GATES
    probe = validator_module.PerCellHardGate(
        protocol_key="probe_max",
        result_section="metrics",
        result_key="probe_metric",
        validation_message="pass result exceeds probe gate",
        failure_stage="metrics",
    )
    try:
        validator_module.PER_CELL_HARD_GATES = (*original_gates, probe)
        runner_failure = run_module._hard_gate_failure(
            {"hard_gates": {"probe_max": 0.0}},
            {"probe_metric": 1.0},
            {},
        )
        assert runner_failure == {
            "type": "quality_gate_failed",
            "message": "probe_metric=1.0 exceeds hard gate 0.0",
            "stage": "metrics",
        }

        root = Path(__file__).resolve().parents[2]
        protocol = validator_module.load_json(root / "bakeoff/protocol-v2.json")
        report = validator_module.load_json(root / "bakeoff/fixtures/positive/report-v2.json")
        corpus = validator_module.load_json(root / "bakeoff/fixtures/positive/corpus-v1.json")
        protocol["hard_gates"]["probe_max"] = 0.0
        result = copy.deepcopy(report["results"][0])
        result["metrics"]["probe_metric"] = 1.0
        shot = next(
            shot
            for partition in corpus["partitions"]
            for shot in partition["shots"]
            if shot["id"] == result["shot_id"]
        )
        cap_map = {cap["token"]: cap for cap in protocol["analysis_caps"]}
        try:
            validator_module._validate_result_measurement(
                result,
                shot,
                "$.results[0]",
                protocol["profiles"][report["profile"]],
                protocol,
                cap_map,
                report["profile"],
            )
        except validator_module.ValidationError as failure:
            assert failure.path == "$.results[0].metrics.probe_metric"
            assert failure.message == "pass result exceeds probe gate"
        else:
            raise AssertionError("validator did not consume the shared per-cell gate spec")
    finally:
        validator_module.PER_CELL_HARD_GATES = original_gates


def main() -> int:
    test_canonical_json_is_order_independent_and_hashes_utf8_bytes()
    test_canonicalization_rejects_non_json_and_nonfinite_values()
    test_run_spec_rejects_non_plain_top_level_objects()
    test_every_known_stable_field_is_present_in_the_fixture()
    test_every_spec_serializes_the_default_identity_schema_version()
    test_identity_schema_version_is_stable_and_identity_bearing()
    test_identity_assertion_centrally_checks_canonical_stable_inputs()
    test_semantic_report_warnings_and_summary_change_identity()
    test_every_stable_field_changes_identity()
    test_every_volatile_publication_field_does_not_change_identity()
    test_publication_is_separate_in_persistence_record()
    test_run_spec_is_immutable_and_defensively_copies_inputs()
    test_from_inputs_hashes_extra_stable_fields()
    test_with_publication_preserves_identity()
    test_runner_and_validator_share_per_cell_hard_gate_specification()
    print("RunSpec tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
