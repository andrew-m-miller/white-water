#!/usr/bin/env python3
"""Validate the revision-bound NeuFlow v2 P25-3F provenance record."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from artifact_workflow import ArtifactError, load_manifest, validate_artifact  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - package import path
    from .artifact_workflow import ArtifactError, load_manifest, validate_artifact  # type: ignore


EXPECTED_SOURCE_COMMIT = "204b5e3744461d90303b9ff82caa7a1bb56a2ca2"
EXPECTED_CHECKPOINT_SHA256 = "76152c8068f247a7d073aa13e61da8cb4c3c6a798076d4dc8e20f7995fcc019f"
EXPECTED_CHECKPOINT_SIZE = 36195519
EXPECTED_ENVIRONMENT_SHA256 = "b0bd2d907bdcdd4cd90e9ce532b98f22627d14f9372baeebb356993e78f21fe8"
EXPECTED_ARTIFACT_SHA256 = "f12b8030f7432f044ef41a373ae7e7e5180f9cbfc692a1793758d881cea18c82"
EXPECTED_ARTIFACT_SIZE = 66177652


def main() -> int:
    manifest_path = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("neuflow-v2.json")
    )
    manifest = load_manifest(manifest_path)

    if manifest["candidate"]["id"] != "neuflow-v2":
        raise ArtifactError("NeuFlow manifest changed candidate.id")
    if manifest["upstream"]["commit"] != EXPECTED_SOURCE_COMMIT:
        raise ArtifactError("NeuFlow manifest changed the pinned upstream commit")
    if manifest["checkpoint"]["revision"] != EXPECTED_SOURCE_COMMIT:
        raise ArtifactError("NeuFlow checkpoint revision is not pinned to the source commit")
    if manifest["checkpoint"]["size_bytes"] != EXPECTED_CHECKPOINT_SIZE:
        raise ArtifactError("NeuFlow checkpoint size changed")
    if manifest["checkpoint"]["sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise ArtifactError("NeuFlow checkpoint SHA256 changed")
    if EXPECTED_SOURCE_COMMIT not in manifest["checkpoint"]["url"]:
        raise ArtifactError("NeuFlow checkpoint URL is not revision-pinned")
    if manifest["export_environment"]["sha256"] != EXPECTED_ENVIRONMENT_SHA256:
        raise ArtifactError("NeuFlow export-environment SHA256 changed")
    if manifest["export"]["export_environment_sha256"] != EXPECTED_ENVIRONMENT_SHA256:
        raise ArtifactError("NeuFlow export environment is not bound to the manifest environment")

    contract = manifest["tensor_contract"]
    if contract["output"]["direction"] != "image1_to_image2":
        raise ArtifactError("NeuFlow output direction is not image1_to_image2")
    if contract["output"]["units"] != "unpadded_analysis_pixels":
        raise ArtifactError("NeuFlow output units changed")
    if contract["padding"] != {"multiple": 16, "policy": "caller-replication-crop"}:
        raise ArtifactError("NeuFlow padding contract changed")
    if contract["iterations"] != "1_s16_and_8_s8_baked_into_graph":
        raise ArtifactError("NeuFlow iteration contract changed")
    if contract["normalization_location"] != "graph" or contract["normalization_formula"] != "image / 255":
        raise ArtifactError("NeuFlow normalization contract changed")
    if manifest["status"] not in {
        "provenance_pinned_export_pending",
        "export_validated",
        "excluded",
    }:
        raise ArtifactError("P25-3F manifest has an unexpected status")
    if (
        manifest["status"] == "provenance_pinned_export_pending"
        and manifest["validation"]["status"] != "pending"
    ):
        raise ArtifactError("pending NeuFlow export must have pending numerical validation")
    if manifest["status"] == "export_validated":
        if manifest["validation"]["status"] != "passed":
            raise ArtifactError("validated NeuFlow export must have passed numerical validation")
        if manifest["export"]["sha256"] != EXPECTED_ARTIFACT_SHA256:
            raise ArtifactError("validated NeuFlow artifact SHA256 changed")
        if manifest["export"]["size_bytes"] != EXPECTED_ARTIFACT_SIZE:
            raise ArtifactError("validated NeuFlow artifact size changed")
        if manifest["validation"]["shapes"]["dynamic"] is not False:
            raise ArtifactError("NeuFlow fixed-shape export must not claim dynamic support")
    if manifest["status"] == "excluded":
        if manifest["candidate"]["role"] != "excluded":
            raise ArtifactError("excluded NeuFlow manifest must mark candidate.role=excluded")
        admission_exclusion = "admission_status=excluded_checkpoint_license_terms_unknown" in manifest.get(
            "notes", []
        )
        if admission_exclusion and manifest["validation"]["status"] != "passed":
            raise ArtifactError("license-excluded NeuFlow manifest must preserve passed numerical validation")
        if not admission_exclusion and manifest["validation"]["status"] != "failed":
            raise ArtifactError("technical NeuFlow exclusion must carry failed validation status")
        observed = manifest["validation"].get("observed")
        if admission_exclusion:
            if manifest["licenses"]["checkpoint"]["commercial_use_permitted"] != "unknown":
                raise ArtifactError("NeuFlow checkpoint admission exclusion lost its commercial-use evidence")
            if manifest["licenses"]["checkpoint"]["redistribution_permitted"] != "unknown":
                raise ArtifactError("NeuFlow checkpoint admission exclusion lost its redistribution evidence")
            if not isinstance(observed, dict) or observed.get("numerical_status") != "passed":
                raise ArtifactError("NeuFlow admission exclusion must preserve passed numerical evidence")
            if manifest["export"]["sha256"] != EXPECTED_ARTIFACT_SHA256:
                raise ArtifactError("NeuFlow excluded export SHA256 changed")
            if manifest["export"]["size_bytes"] != EXPECTED_ARTIFACT_SIZE:
                raise ArtifactError("NeuFlow excluded export size changed")
    checkpoint_terms_unknown = (
        manifest["licenses"]["checkpoint"]["commercial_use_permitted"] == "unknown"
        or manifest["licenses"]["checkpoint"]["redistribution_permitted"] == "unknown"
    )
    if checkpoint_terms_unknown and manifest["status"] in {
        "export_validated",
        "host_probe_pending",
        "host_probe_cpu_cuda_passed",
    }:
        raise ArtifactError("unknown NeuFlow checkpoint terms require an explicit excluded admission state")
    if (
        checkpoint_terms_unknown
        and manifest["status"] == "excluded"
        and isinstance(manifest["validation"].get("observed"), dict)
        and manifest["validation"]["observed"].get("numerical_status") == "passed"
        and "admission_status=excluded_checkpoint_license_terms_unknown" not in manifest.get("notes", [])
    ):
        raise ArtifactError("numerically validated NeuFlow exclusion lacks its checkpoint-license admission reason")

    artifact_path = manifest_path.parent / manifest["export"]["artifact"]
    # The source checkout intentionally omits ignored ONNX payloads. If a local exporter has
    # staged one, validate its exact mode, size and hash instead of silently ignoring it.
    try:
        artifact_path.lstat()
    except FileNotFoundError:
        pass
    else:
        validate_artifact(manifest, manifest_path, artifact_path)

    print(
        "NeuFlow v2 provenance manifest valid: "
        f"status={manifest['status']} checkpoint_sha256={manifest['checkpoint']['sha256']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ArtifactError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"check_neuflow_manifest.py: error: {exc}", file=sys.stderr)
        raise SystemExit(1)
