#!/usr/bin/env python3
"""Validate the revision-bound NeuFlow v2 P25-3F provenance record."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from artifact_workflow import ArtifactError, load_manifest, validate_artifact  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - package import path
    from .artifact_workflow import ArtifactError, load_manifest, validate_artifact  # type: ignore

try:
    from .exclusion_contract import ExclusionReason
except ImportError:  # Direct script imports keep the dependency-light checker runnable.
    from exclusion_contract import ExclusionReason


EXPECTED_SOURCE_COMMIT = "204b5e3744461d90303b9ff82caa7a1bb56a2ca2"
EXPECTED_CHECKPOINT_SHA256 = "76152c8068f247a7d073aa13e61da8cb4c3c6a798076d4dc8e20f7995fcc019f"
EXPECTED_CHECKPOINT_SIZE = 36195519
EXPECTED_ENVIRONMENT_SHA256 = "b0bd2d907bdcdd4cd90e9ce532b98f22627d14f9372baeebb356993e78f21fe8"
EXPECTED_ARTIFACT_SHA256 = "f12b8030f7432f044ef41a373ae7e7e5180f9cbfc692a1793758d881cea18c82"
EXPECTED_ARTIFACT_SIZE = 66177652
CHECKPOINT_EXCLUSION_REASON = ExclusionReason.CHECKPOINT_LICENSE_TERMS_UNKNOWN.value
EXPORT_FAILURE_EXCLUSION_REASON = ExclusionReason.EXPORT_OR_OPERATOR_FAILURE.value
MACOS_PLATFORM = "macos-arm64"
LINUX_PLATFORM = "linux-x86_64"


def _validate_provider_evidence(platform: str, observed: object) -> None:
    if not isinstance(observed, dict):
        raise ArtifactError("NeuFlow numerical pass is missing observed provider evidence")
    provider_validation = observed.get("provider_validation")
    if not isinstance(provider_validation, dict):
        raise ArtifactError("NeuFlow numerical pass is missing provider qualification evidence")
    if provider_validation.get("passed") is not True:
        raise ArtifactError("NeuFlow provider qualification did not pass")
    requested = provider_validation.get("requested")
    selected = provider_validation.get("selected")
    if not isinstance(selected, list) or not selected:
        raise ArtifactError("NeuFlow provider evidence has no selected providers")
    expected_provider = (
        "CPUExecutionProvider" if platform == MACOS_PLATFORM else "CUDAExecutionProvider"
    )
    if requested != expected_provider or selected[0] != expected_provider:
        raise ArtifactError(
            f"NeuFlow {platform} evidence must request and first-select {expected_provider}"
        )
    if platform == MACOS_PLATFORM and "CUDAExecutionProvider" in selected:
        raise ArtifactError("macOS NeuFlow evidence must not claim CUDA selection")
    environment = observed.get("environment")
    if isinstance(environment, dict) and environment.get("provider") not in (None, requested):
        raise ArtifactError("NeuFlow provider evidence disagrees with its environment record")


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
    platform = manifest["export"]["platform"]
    if platform not in {MACOS_PLATFORM, LINUX_PLATFORM}:
        raise ArtifactError(f"NeuFlow platform must be {MACOS_PLATFORM} or {LINUX_PLATFORM}")
    environment = manifest["export_environment"]
    if platform == MACOS_PLATFORM:
        if environment["platform"] != "macos" or environment["architecture"] != "arm64":
            raise ArtifactError("macOS NeuFlow artifact has a non-macOS export environment")
        if environment["sha256"] != EXPECTED_ENVIRONMENT_SHA256:
            raise ArtifactError("NeuFlow macOS export-environment SHA256 changed")
        if manifest["export"]["export_environment_sha256"] != EXPECTED_ENVIRONMENT_SHA256:
            raise ArtifactError("NeuFlow macOS export environment is not bound to its evidence")
    else:
        if environment["platform"] != "linux" or environment["architecture"] != "x86_64":
            raise ArtifactError("Linux NeuFlow artifact must carry a Linux x86_64 export environment")
        if manifest["export"]["export_environment_sha256"] != environment["sha256"]:
            raise ArtifactError("Linux NeuFlow export environment is not bound to its evidence")

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
    if contract["spatial_dimensions"] != (
        "fixed_shape_only; exactly 432x768; no dynamic or other-shape support"
    ):
        raise ArtifactError("NeuFlow fixed evaluation-shape contract changed")
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
        if platform == MACOS_PLATFORM:
            if manifest["export"]["sha256"] != EXPECTED_ARTIFACT_SHA256:
                raise ArtifactError("validated NeuFlow macOS artifact SHA256 changed")
            if manifest["export"]["size_bytes"] != EXPECTED_ARTIFACT_SIZE:
                raise ArtifactError("validated NeuFlow macOS artifact size changed")
        elif manifest["export"]["sha256"] is None or manifest["export"]["size_bytes"] is None:
            raise ArtifactError("validated Linux NeuFlow artifact must record exact identity")
        if manifest["validation"]["shapes"]["dynamic"] is not False:
            raise ArtifactError("NeuFlow fixed-shape export must not claim dynamic support")
        if manifest["validation"]["shapes"] != {
            "dynamic": False,
            "example": [1, 2, 432, 768],
            "additional": [1, 2, 432, 768],
        }:
            raise ArtifactError("NeuFlow fixed evaluation-shape evidence changed")
    if manifest["status"] == "excluded":
        if manifest["candidate"]["role"] != "excluded":
            raise ArtifactError("excluded NeuFlow manifest must mark candidate.role=excluded")
        reason_code = manifest["exclusion"]["reason_code"]
        if reason_code == CHECKPOINT_EXCLUSION_REASON:
            if manifest["validation"]["status"] != "passed":
                raise ArtifactError("license-excluded NeuFlow manifest must preserve passed numerical validation")
            if manifest["licenses"]["checkpoint"]["commercial_use_permitted"] != "unknown":
                raise ArtifactError("NeuFlow checkpoint admission exclusion lost its commercial-use evidence")
            if manifest["licenses"]["checkpoint"]["redistribution_permitted"] != "unknown":
                raise ArtifactError("NeuFlow checkpoint admission exclusion lost its redistribution evidence")
            if platform == MACOS_PLATFORM:
                if manifest["export"]["sha256"] != EXPECTED_ARTIFACT_SHA256:
                    raise ArtifactError("NeuFlow excluded macOS export SHA256 changed")
                if manifest["export"]["size_bytes"] != EXPECTED_ARTIFACT_SIZE:
                    raise ArtifactError("NeuFlow excluded macOS export size changed")
            elif manifest["export"]["sha256"] is None or manifest["export"]["size_bytes"] is None:
                raise ArtifactError("excluded Linux NeuFlow artifact must retain exact identity")
        elif reason_code == EXPORT_FAILURE_EXCLUSION_REASON:
            if manifest["validation"]["status"] != "failed":
                raise ArtifactError("technical NeuFlow exclusion must carry failed validation status")
        else:  # Shared contract validation should make this unreachable.
            raise ArtifactError(f"unsupported NeuFlow exclusion reason: {reason_code}")
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
        and manifest["validation"]["status"] == "passed"
        and manifest["exclusion"]["reason_code"] != CHECKPOINT_EXCLUSION_REASON
    ):
        raise ArtifactError("numerically validated NeuFlow exclusion has the wrong checkpoint-license reason")

    observed = manifest["validation"].get("observed")
    if manifest["validation"]["status"] == "passed":
        _validate_provider_evidence(platform, observed)
        if not isinstance(observed, dict):  # for type narrowing below
            raise ArtifactError("NeuFlow numerical pass is missing observed evidence")
        advertised_io = observed.get("advertised_io")
        if not isinstance(advertised_io, dict):
            raise ArtifactError("NeuFlow numerical pass is missing fixed IO evidence")
        inputs = advertised_io.get("inputs")
        outputs = advertised_io.get("outputs")
        if not isinstance(inputs, list) or len(inputs) != 2 or any(
            item.get("shape") != [1, 3, 432, 768] for item in inputs
        ):
            raise ArtifactError("NeuFlow advertised input shape is not the fixed evaluation lattice")
        if not isinstance(outputs, list) or len(outputs) != 1 or outputs[0].get("shape") != [1, 2, 432, 768]:
            raise ArtifactError("NeuFlow advertised output shape is not the fixed evaluation lattice")

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
