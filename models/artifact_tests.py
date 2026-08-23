#!/usr/bin/env python3
"""Positive and negative gates for the candidate-neutral artifact workflow.

The checked-in positive manifest is intentionally small and contains a text payload.  The
negative cases are derived from that fixture in a temporary directory so symlink, mode and
non-regular-file behavior is exercised by the same path that stages a real ONNX artifact.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile

from artifact_workflow import (  # type: ignore  # pylint: disable=wrong-import-position
    ArtifactError,
    PROTOCOL_PATH,
    ValidationError,
    load_json,
    load_manifest,
    validate_artifact,
    validate_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
POSITIVE = ROOT / "models" / "fixtures" / "positive" / "artifact-v1.json"
POSITIVE_ARTIFACT = ROOT / "models" / "fixtures" / "positive" / "valid.bin"
NEGATIVE_CASES = ROOT / "models" / "fixtures" / "negative" / "artifact-v1-cases.json"
SEA_RAFT = ROOT / "models" / "sea-raft-m.json"


def expect_failure(label: str, callback) -> None:
    try:
        callback()
    except (ArtifactError, ValidationError, OSError, KeyError, TypeError, ValueError):
        return
    raise AssertionError(f"negative artifact fixture unexpectedly passed: {label}")


def main() -> int:
    negative_cases = load_json(NEGATIVE_CASES)
    expected_case_ids = {
        "unknown-top-level-field",
        "unknown-nested-field",
        "incompatible-tensor-contract",
        "input-pair-dtype-mismatch",
        "input-pair-layout-mismatch",
        "input-pair-channel-mismatch",
        "padding-policy-unknown",
        "validation-identity-false",
        "validation-identity-threshold",
        "validation-parity-unchecked",
        "validation-parity-threshold",
        "validation-forward-sign-mismatch",
        "validation-reverse-sign-mismatch",
        "validation-direction-axis-mismatch",
        "validation-primary-motion-too-small",
        "validation-cross-axis-too-large",
        "validation-threshold-partial",
        "validation-status-incoherent",
        "export-status-validation-incoherent",
        "host-status-validation-incoherent",
        "export-status-hash-incoherent",
        "artifact-sha256-mismatch",
        "artifact-size-mismatch",
        "artifact-symlink",
        "artifact-non-regular",
        "artifact-mode-not-0644",
    }
    if {case["id"] for case in negative_cases["cases"]} != expected_case_ids:
        raise AssertionError("negative artifact fixture inventory is incomplete")
    positive = load_manifest(POSITIVE, protocol_path=False)
    validate_artifact(positive, POSITIVE, POSITIVE_ARTIFACT)
    sea_raft = load_manifest(SEA_RAFT)
    if sea_raft["tensor_contract"]["padding"]["policy"] != "caller-replication-crop":
        raise AssertionError("SEA-RAFT manifest does not expose the canonical replication policy")

    unknown = copy.deepcopy(positive)
    unknown["unknown_required_field"] = True
    expect_failure("unknown top-level field", lambda: validate_manifest(unknown))

    unknown_nested = copy.deepcopy(positive)
    unknown_nested["tensor_contract"]["unknown_required_field"] = True
    expect_failure("unknown nested field", lambda: validate_manifest(unknown_nested))

    incompatible = copy.deepcopy(positive)
    incompatible["tensor_contract"]["output"]["units"] = "input_pixels"
    incompatible["tensor_contract"]["padding"]["policy"] = "none"
    protocol = load_json(PROTOCOL_PATH)
    expect_failure(
        "incompatible P25-0 tensor contract",
        lambda: validate_manifest(incompatible, protocol=protocol),
    )

    pair_mutations = {
        "input-pair-dtype-mismatch": ("dtype", "float16"),
        "input-pair-layout-mismatch": ("layout", "NHWC"),
        "input-pair-channel-mismatch": ("channels", "RGBA"),
    }
    for label, (field, value) in pair_mutations.items():
        bad_pair = copy.deepcopy(positive)
        bad_pair["tensor_contract"]["inputs"][1][field] = value
        expect_failure(label, lambda bad_pair=bad_pair: validate_manifest(bad_pair, protocol=protocol))

    bad_pair_without_protocol = copy.deepcopy(positive)
    bad_pair_without_protocol["tensor_contract"]["inputs"][1]["dtype"] = "float16"
    expect_failure(
        "input pair mismatch without protocol",
        lambda: validate_manifest(bad_pair_without_protocol),
    )

    bad_padding = copy.deepcopy(positive)
    bad_padding["tensor_contract"]["padding"]["policy"] = "caller-custom-pad"
    expect_failure("padding-policy-unknown", lambda: validate_manifest(bad_padding))

    bad_identity = copy.deepcopy(positive)
    bad_identity["validation"]["identity"]["passed"] = False
    expect_failure("validation-identity-false", lambda: validate_manifest(bad_identity))

    bad_identity_threshold = copy.deepcopy(positive)
    bad_identity_threshold["validation"]["identity"]["median_epe_px"] = 1
    bad_identity_threshold["validation"]["identity_median_epe_max"] = 0.75
    expect_failure(
        "validation identity threshold",
        lambda: validate_manifest(bad_identity_threshold),
    )

    bad_parity = copy.deepcopy(positive)
    bad_parity["validation"]["parity"]["checked"] = False
    expect_failure("validation-parity-unchecked", lambda: validate_manifest(bad_parity))

    bad_parity_threshold = copy.deepcopy(positive)
    bad_parity_threshold["validation"]["parity"]["mean_abs"] = 1
    bad_parity_threshold["validation"]["onnx_pytorch_mean_abs_max"] = 0.05
    expect_failure(
        "validation parity threshold",
        lambda: validate_manifest(bad_parity_threshold),
    )

    bad_forward = copy.deepcopy(positive)
    bad_forward["validation"]["directions"]["forward"]["median_dx_px"] = -1
    expect_failure("validation-forward-sign-mismatch", lambda: validate_manifest(bad_forward))

    bad_reverse = copy.deepcopy(positive)
    bad_reverse["validation"]["directions"]["reverse"]["median_dx_px"] = 1
    expect_failure("validation-reverse-sign-mismatch", lambda: validate_manifest(bad_reverse))

    bad_axis = copy.deepcopy(positive)
    bad_axis["validation"]["directions"]["reverse"]["expected_sign"] = "negative_y"
    bad_axis["validation"]["directions"]["reverse"]["median_dy_px"] = -1
    expect_failure("validation direction axis mismatch", lambda: validate_manifest(bad_axis))

    bad_primary = copy.deepcopy(positive)
    bad_primary["validation"].update(
        {
            "translation_pixels": 4,
            "translation_x_fraction_min": 0.5,
            "translation_abs_y_max": 2.0,
        }
    )
    bad_primary["validation"]["directions"]["forward"]["median_dx_px"] = 1
    expect_failure("validation primary motion too small", lambda: validate_manifest(bad_primary))

    bad_cross = copy.deepcopy(positive)
    bad_cross["validation"].update(
        {
            "translation_pixels": 4,
            "translation_x_fraction_min": 0.5,
            "translation_abs_y_max": 2.0,
        }
    )
    bad_cross["validation"]["directions"]["forward"].update(
        {"median_dx_px": 4, "median_dy_px": 0}
    )
    bad_cross["validation"]["directions"]["reverse"].update(
        {"median_dx_px": -4, "median_dy_px": 3}
    )
    expect_failure("validation cross-axis motion too large", lambda: validate_manifest(bad_cross))

    y_directions = copy.deepcopy(positive)
    y_directions["validation"].update(
        {
            "translation_pixels": -4,
            "translation_x_fraction_min": 0.5,
            "translation_abs_y_max": 2.0,
        }
    )
    y_directions["validation"]["directions"]["forward"].update(
        {"median_dx_px": 0, "median_dy_px": 2, "expected_sign": "positive_y"}
    )
    y_directions["validation"]["directions"]["reverse"].update(
        {"median_dx_px": 0, "median_dy_px": -2, "expected_sign": "negative_y"}
    )
    validate_manifest(y_directions)

    threshold_fields = (
        "translation_pixels",
        "translation_x_fraction_min",
        "translation_abs_y_max",
    )
    for omitted in threshold_fields:
        partial_thresholds = copy.deepcopy(positive)
        partial_thresholds["validation"].update(
            {
                "translation_pixels": 4,
                "translation_x_fraction_min": 0.5,
                "translation_abs_y_max": 2.0,
            }
        )
        del partial_thresholds["validation"][omitted]
        expect_failure(
            f"partial direction thresholds missing {omitted}",
            lambda partial_thresholds=partial_thresholds: validate_manifest(partial_thresholds),
        )

    bad_status = copy.deepcopy(positive)
    bad_status["validation"]["status"] = "pending"
    expect_failure("validation-status-incoherent", lambda: validate_manifest(bad_status))

    bad_export_validation = copy.deepcopy(positive)
    bad_export_validation["status"] = "export_validated"
    bad_export_validation["validation"]["status"] = "pending"
    expect_failure(
        "export status/validation incoherence",
        lambda: validate_manifest(bad_export_validation),
    )

    bad_host_validation = copy.deepcopy(positive)
    bad_host_validation["status"] = "host_probe_cpu_cuda_passed"
    bad_host_validation["validation"]["status"] = "pending"
    expect_failure(
        "host status/validation incoherence",
        lambda: validate_manifest(bad_host_validation),
    )

    bad_export_status = copy.deepcopy(positive)
    bad_export_status["status"] = "provenance_pinned_export_pending"
    expect_failure("export status/hash incoherence", lambda: validate_manifest(bad_export_status))

    excluded_passed_missing_reason = copy.deepcopy(positive)
    excluded_passed_missing_reason["status"] = "excluded"
    excluded_passed_missing_reason["candidate"]["role"] = "excluded"
    expect_failure(
        "excluded passed manifest missing typed reason",
        lambda: validate_manifest(excluded_passed_missing_reason),
    )

    note_is_not_reason = copy.deepcopy(excluded_passed_missing_reason)
    note_is_not_reason["notes"] = [
        "admission_status=excluded_checkpoint_license_terms_unknown",
    ]
    expect_failure(
        "free-text note is not an exclusion reason",
        lambda: validate_manifest(note_is_not_reason),
    )

    excluded_passed = copy.deepcopy(excluded_passed_missing_reason)
    excluded_passed["exclusion"] = {
        "reason_code": "checkpoint_license_terms_unknown",
    }
    validate_manifest(excluded_passed)

    excluded_failed = copy.deepcopy(excluded_passed_missing_reason)
    excluded_failed["validation"]["status"] = "failed"
    excluded_failed["validation"]["observed"] = {
        "failure_stage": "onnx_export",
        "failure_type": "RuntimeError",
    }
    excluded_failed["exclusion"] = {"reason_code": "export_or_operator_failure"}
    validate_manifest(excluded_failed)

    unknown_reason = copy.deepcopy(excluded_passed_missing_reason)
    unknown_reason["exclusion"] = {"reason_code": "checkpoint_terms_unavailable"}
    expect_failure("unknown exclusion reason code", lambda: validate_manifest(unknown_reason))

    nonexcluded_with_reason = copy.deepcopy(positive)
    nonexcluded_with_reason["exclusion"] = {
        "reason_code": "checkpoint_license_terms_unknown",
    }
    expect_failure(
        "non-excluded manifest carries exclusion",
        lambda: validate_manifest(nonexcluded_with_reason),
    )

    with tempfile.TemporaryDirectory(prefix="whitewater-artifact-tests-") as temporary:
        directory = Path(temporary)
        artifact = directory / "valid.bin"
        artifact.write_bytes(POSITIVE_ARTIFACT.read_bytes())
        artifact.chmod(0o644)
        manifest_path = directory / "manifest.json"

        def with_manifest(mutated: dict) -> Path:
            manifest_path.write_text(json.dumps(mutated), encoding="utf-8")
            manifest_path.chmod(0o644)
            return manifest_path

        bad_hash = copy.deepcopy(positive)
        bad_hash["export"]["sha256"] = "0" * 64
        bad_hash["export"]["platform_artifacts"][0]["sha256"] = "0" * 64
        bad_hash_manifest = with_manifest(bad_hash)
        bad_hash_loaded = load_manifest(bad_hash_manifest, protocol_path=False)
        expect_failure(
            "artifact SHA256 mismatch",
            lambda: validate_artifact(bad_hash_loaded, bad_hash_manifest, artifact),
        )

        bad_size = copy.deepcopy(positive)
        bad_size["export"]["size_bytes"] = 36
        bad_size["export"]["platform_artifacts"][0]["size_bytes"] = 36
        bad_size_manifest = with_manifest(bad_size)
        bad_size_loaded = load_manifest(bad_size_manifest, protocol_path=False)
        expect_failure(
            "artifact size mismatch",
            lambda: validate_artifact(bad_size_loaded, bad_size_manifest, artifact),
        )

        wrong_mode = directory / "wrong-mode.bin"
        wrong_mode.write_bytes(POSITIVE_ARTIFACT.read_bytes())
        wrong_mode.chmod(0o600)
        expect_failure(
            "non-0644 artifact",
            lambda: validate_artifact(positive, POSITIVE, wrong_mode),
        )

        non_regular = directory / "directory-artifact"
        non_regular.mkdir()
        expect_failure(
            "non-regular artifact",
            lambda: validate_artifact(positive, POSITIVE, non_regular),
        )

        symlink = directory / "symlink-artifact"
        symlink.symlink_to(artifact)
        expect_failure(
            "symlink artifact",
            lambda: validate_artifact(positive, POSITIVE, symlink),
        )

    print("artifact framework positive and negative fixtures: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, ArtifactError, ValidationError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"artifact_tests.py: error: {exc}", file=sys.stderr)
        raise SystemExit(1)
