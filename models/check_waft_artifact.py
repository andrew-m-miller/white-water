#!/usr/bin/env python3
"""Validate the WAFT/Twins evaluation artifact record and its failure-safe states."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

from artifact_workflow import (  # type: ignore  # pylint: disable=wrong-import-position
    ArtifactError,
    load_manifest,
    validate_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "models" / "waft-twins-artifact.json"
EXPECTED_COMMIT = "b152ff1cad1af8c185ee7b141997c48ff3334c87"
EXPECTED_CHECKPOINT_SHA256 = "f750cd15281fc30de477723438ff4a67fe1591deac4ab0eb9b366e27c827e070"
EXPECTED_CHECKPOINT_SIZE = 544230582
EXPECTED_CONFIG_PATH = "config/a2/twins/chairs-things.json"
EXPECTED_CONFIG_SHA256 = "4eb827762b132fe0e90b4d87e456088e772573b4f346d5e396e0912dad528996"
BLOCKER_CODES = {
    "missing_pinned_input",
    "upstream_revision_mismatch",
    "upstream_worktree_dirty",
    "checkpoint_identity_mismatch",
    "pinned_config_invalid",
    "platform_identity_mismatch",
    "missing_export_dependency",
    "strict_checkpoint_load_failure",
    "onnx_export_failure",
    "unsupported_operator_or_domain",
    "pytorch_onnx_parity_failure",
    "direction_or_identity_failure",
    "artifact_publication_failure",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactError(message)


def _check_blocker(observed: Any) -> None:
    require(isinstance(observed, Mapping), "failed WAFT validation must record observed details")
    blocker = observed.get("technical_blocker")
    require(isinstance(blocker, Mapping), "failed WAFT validation must record technical_blocker")
    require(blocker.get("schema_version") == 1, "technical blocker schema version changed")
    require(blocker.get("code") in BLOCKER_CODES, "technical blocker code is not typed")
    require(isinstance(blocker.get("stage"), str) and blocker["stage"], "technical blocker stage missing")
    require(isinstance(blocker.get("message"), str) and blocker["message"], "technical blocker message missing")
    require(isinstance(blocker.get("details"), Mapping), "technical blocker details missing")


def main() -> int:
    manifest_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MANIFEST
    manifest = load_manifest(manifest_path)
    require(manifest["candidate"]["id"] == "waft-twins", "unexpected WAFT candidate id")
    require(manifest["status"] == "excluded", "WAFT evaluation artifact must remain excluded from shipping")
    require(manifest["upstream"]["commit"] == EXPECTED_COMMIT, "WAFT source commit changed")
    checkpoint = manifest["checkpoint"]
    require(checkpoint["size_bytes"] == EXPECTED_CHECKPOINT_SIZE, "WAFT checkpoint size changed")
    require(checkpoint["sha256"] == EXPECTED_CHECKPOINT_SHA256, "WAFT checkpoint hash changed")
    require(manifest["licenses"]["checkpoint"]["license"] == "unknown", "checkpoint licence was inferred")
    require(manifest["licenses"]["checkpoint"]["commercial_use_permitted"] == "unknown", "checkpoint commercial-use verdict changed")
    require(manifest["licenses"]["checkpoint"]["redistribution_permitted"] == "unknown", "checkpoint redistribution verdict changed")
    require(manifest["exclusion"]["reason_code"] == "checkpoint_license_terms_unknown", "shipping exclusion reason changed")
    config = manifest["model"]["config"]
    require(config["feature_encoder"] == "twins", "WAFT candidate is not Twins")
    require(config["config_path"] == EXPECTED_CONFIG_PATH, "WAFT config path changed")
    require(config["config_sha256"] == EXPECTED_CONFIG_SHA256, "WAFT config hash changed")
    contract = manifest["tensor_contract"]
    require(contract["output"]["direction"] == "image1_to_image2", "WAFT direction contract changed")
    require(contract["output"]["units"] == "unpadded_analysis_pixels", "WAFT output units changed")
    require(contract["padding"]["multiple"] == 32, "WAFT padding multiple changed")
    require(contract["graph_domains"] == ["ai.onnx"], "WAFT graph-domain contract changed")

    export = manifest["export"]
    require(export["script"] == "models/export_waft.py", "WAFT exporter path changed")
    require(export["mode"] == "0644", "WAFT artifact mode contract changed")
    require(len(export["platform_artifacts"]) == 1, "WAFT evaluation record must have one exact platform row")
    row = export["platform_artifacts"][0]
    require(row["platform"] == export["platform"], "WAFT platform row disagrees with export")
    require(row["artifact"] == export["artifact"], "WAFT platform artifact disagrees with export")
    require(row["mode"] == "0644", "WAFT platform artifact mode changed")

    validation = manifest["validation"]
    require(validation["status"] in {"pending", "failed", "passed"}, "unknown WAFT validation status")
    require(
        validation["status"] != "failed",
        "technical unavailability must remain pending while checkpoint-license exclusion is unresolved",
    )
    if validation["status"] in {"pending", "failed"}:
        require(export["sha256"] is None and export["size_bytes"] is None, "blocked WAFT export published an artifact claim")
        require(row["sha256"] is None and row["size_bytes"] is None, "blocked WAFT platform row published an artifact claim")
    if validation["status"] == "pending":
        if validation["observed"] is not None:
            _check_blocker(validation["observed"])
    elif validation["status"] == "failed":
        _check_blocker(validation["observed"])
    else:
        require(export["sha256"] is not None and export["size_bytes"] is not None, "passed WAFT export has no hash/size")
        require(row["sha256"] == export["sha256"] and row["size_bytes"] == export["size_bytes"], "WAFT platform hash/size disagrees")
        validate_artifact(manifest, manifest_path, manifest_path.parent / row["artifact"], platform=row["platform"])
        observed = validation["observed"]
        require(isinstance(observed, Mapping), "passed WAFT validation has no observed result")
        require(observed.get("strict_checkpoint", {}).get("strict") is True, "strict checkpoint evidence missing")
        require(observed.get("operator_gate", {}).get("version") == "onnx-standard-domain-v1", "operator gate evidence missing")
        require(observed.get("operator_gate", {}).get("foreign_domains") in (None, []), "passed WAFT graph has foreign domains")
        require(observed.get("operator_gate", {}).get("unsupported_operators") == [], "passed WAFT graph has unsupported operators")

    print(f"WAFT/Twins evaluation artifact record: PASS ({validation['status']})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ArtifactError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"check_waft_artifact.py: error: {exc}", file=sys.stderr)
        raise SystemExit(1)
