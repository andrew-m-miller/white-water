#!/usr/bin/env python3
"""Check the pinned original-RAFT baseline manifest without requiring ML dependencies."""

from __future__ import annotations

import sys
from pathlib import Path

from artifact_workflow import (  # type: ignore  # pylint: disable=wrong-import-position
    ArtifactError,
    load_manifest,
    validate_artifact,
)


EXPECTED_SOURCE_COMMIT = "2888e15a51fa41140771d3f498ed8023cff098d1"
EXPECTED_CHECKPOINT_SHA256 = "fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1"
EXPECTED_CHECKPOINT_SIZE = 21108000
EXPECTED_CHECKPOINT_URL = "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip"
EXPECTED_ARCHIVE_SHA256 = "4be6101b271f58ec49866da5cf609fd17e86e9cae2483f70630ef4a295dc66bd"
EXPECTED_ARTIFACT_SHA256 = "d9b8aa7d07c3e56303b336c5e1da101c5ebd09c3d71cdcf0c8a649de1044b6d2"
EXPECTED_ARTIFACT_SIZE = 21419753
EXPECTED_EXPORT_ENVIRONMENT_SHA256 = "b294d7fd5749f7342db289ad7a42e8bf654dc256450e41dcbc68f91c775c56ca"
MACOS_PLATFORM = "macos-arm64"
LINUX_CUDA_PLATFORM = "linux-x86_64"


def _platform_entry(manifest: dict, platform_id: str) -> dict:
    entries = [
        entry
        for entry in manifest["export"]["platform_artifacts"]
        if entry["platform"] == platform_id
    ]
    if len(entries) != 1:
        raise ArtifactError(f"original RAFT manifest must contain one {platform_id} artifact row")
    return entries[0]


def _check_recorded_platforms(manifest: dict) -> None:
    """Check the known CPU record and permit a separately qualified Linux CUDA row.

    The checked-in manifest has only the macOS row.  A Linux operator may work on a copy of
    that manifest; the exporter then appends a hash-bound Linux row while retaining the known
    CPU row.  The Linux hash and measurements are intentionally not hard-coded here because
    they do not exist yet.
    """

    macos = _platform_entry(manifest, MACOS_PLATFORM)
    if macos["sha256"] != EXPECTED_ARTIFACT_SHA256 or macos["size_bytes"] != EXPECTED_ARTIFACT_SIZE:
        raise ArtifactError("D2 original RAFT macOS artifact identity changed")
    if macos["export_environment_sha256"] != EXPECTED_EXPORT_ENVIRONMENT_SHA256:
        raise ArtifactError("D2 original RAFT macOS export environment hash changed")
    if macos["export_environment"]["provider"] != "CPUExecutionProvider":
        raise ArtifactError("D2 original RAFT macOS evidence must remain CPU-qualified")

    export = manifest["export"]
    if export["platform"] == MACOS_PLATFORM:
        return
    if export["platform"] != LINUX_CUDA_PLATFORM:
        raise ArtifactError(
            "original RAFT export platform must be macos-arm64 or linux-x86_64"
        )
    linux = _platform_entry(manifest, LINUX_CUDA_PLATFORM)
    if linux["sha256"] is None or linux["size_bytes"] is None:
        raise ArtifactError("Linux CUDA requalification row must record artifact hash and size")
    if linux["mode"] != "0644":
        raise ArtifactError("Linux CUDA requalification artifact must be mode 0644")
    environment = linux.get("export_environment")
    if not isinstance(environment, dict):
        raise ArtifactError("Linux CUDA requalification row must record its environment")
    if environment["platform"] != "linux" or environment["architecture"] != "x86_64":
        raise ArtifactError("Linux CUDA requalification environment is not EL8 x86-64")
    if environment["provider"] != "CUDAExecutionProvider":
        raise ArtifactError("Linux requalification row must use CUDAExecutionProvider")


def main() -> int:
    manifest_path = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("raft-original.json")
    )
    manifest = load_manifest(manifest_path)

    if manifest["candidate"]["id"] != "raft-original":
        raise ArtifactError("original RAFT manifest changed candidate.id")
    if manifest["candidate"]["role"] != "validation-baseline":
        raise ArtifactError("original RAFT manifest must remain a validation baseline")
    if manifest["upstream"]["commit"] != EXPECTED_SOURCE_COMMIT:
        raise ArtifactError("original RAFT source commit is not the pinned official revision")
    checkpoint = manifest["checkpoint"]
    if checkpoint["url"] != EXPECTED_CHECKPOINT_URL:
        raise ArtifactError("original RAFT checkpoint URL changed from the official archive")
    if checkpoint["size_bytes"] != EXPECTED_CHECKPOINT_SIZE:
        raise ArtifactError("original RAFT checkpoint size changed")
    if checkpoint["sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise ArtifactError("original RAFT checkpoint SHA256 changed")
    if checkpoint["license"] != "unknown":
        raise ArtifactError("original RAFT checkpoint terms must remain explicitly unknown")
    if manifest["licenses"]["checkpoint"]["commercial_use_permitted"] != "unknown":
        raise ArtifactError("original RAFT checkpoint commercial-use verdict must remain unknown")
    if manifest["licenses"]["checkpoint"]["redistribution_permitted"] != "unknown":
        raise ArtifactError("original RAFT checkpoint redistribution verdict must remain unknown")
    if manifest["status"] != "excluded":
        raise ArtifactError("D2 original RAFT baseline must remain explicitly excluded")
    if manifest["validation"]["status"] != "passed":
        raise ArtifactError("D2 original RAFT numerical validation must remain passed")
    if manifest["exclusion"]["reason_code"] != "checkpoint_license_terms_unknown":
        raise ArtifactError("D2 original RAFT exclusion must remain checkpoint-terms scoped")
    export = manifest["export"]
    _check_recorded_platforms(manifest)
    if export["platform"] == MACOS_PLATFORM:
        if export["sha256"] != EXPECTED_ARTIFACT_SHA256:
            raise ArtifactError("D2 original RAFT export SHA256 changed")
        if export["size_bytes"] != EXPECTED_ARTIFACT_SIZE:
            raise ArtifactError("D2 original RAFT export size changed")
        if export["export_environment_sha256"] != EXPECTED_EXPORT_ENVIRONMENT_SHA256:
            raise ArtifactError("D2 original RAFT export environment hash changed")
        if manifest["export_environment"]["sha256"] != EXPECTED_EXPORT_ENVIRONMENT_SHA256:
            raise ArtifactError("D2 original RAFT environment hash changed")
    if manifest["validation"]["observed"]["checkpoint_archive_sha256"] != EXPECTED_ARCHIVE_SHA256:
        raise ArtifactError("D2 original RAFT archive hash changed")
    observed = manifest["validation"]["observed"]
    if observed["numerical_gates"] != "passed":
        raise ArtifactError("D2 original RAFT numerical gates are not recorded as passed")
    expected_provider = (
        "CPUExecutionProvider"
        if export["platform"] == MACOS_PLATFORM
        else "CUDAExecutionProvider"
    )
    if observed["provider_validation"]["requested"] != expected_provider:
        raise ArtifactError(
            f"D2 original RAFT {export['platform']} provider validation did not use "
            f"{expected_provider}"
        )
    if observed["provider_validation"]["passed"] is not True:
        raise ArtifactError("D2 original RAFT provider validation did not pass")
    if observed["graph_domains"] != ["ai.onnx"]:
        raise ArtifactError("D2 original RAFT export contains a non-standard graph domain")
    if manifest["validation"]["parity"]["checked"] is not True:
        raise ArtifactError("D2 original RAFT PyTorch/ONNX parity was not checked")
    if manifest["validation"]["shapes"] != {
        "dynamic": True,
        "example": [1, 2, 128, 192],
        "additional": [1, 2, 160, 256],
    }:
        raise ArtifactError("D2 original RAFT dynamic shape evidence changed")
    config = manifest["model"]["config"]
    if config != {
        "small": False,
        "iters": 12,
        "mixed_precision": False,
        "alternate_corr": False,
        "checkpoint_name": "raft-things.pth",
    }:
        raise ArtifactError("original RAFT model configuration changed")

    artifact_path = manifest_path.parent / manifest["export"]["artifact"]
    # The ONNX payload is intentionally absent from source control. If a future export is
    # staged beside this manifest, validate it through the same regular-file/hash/mode gate.
    try:
        artifact_path.lstat()
    except FileNotFoundError:
        pass
    else:
        validate_artifact(manifest, manifest_path, artifact_path)

    print(
        "original RAFT manifest is valid "
        f"(numerical export recorded on {export['platform']}; "
        "checkpoint terms unknown; excluded from shipping/selection/packaging)"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ArtifactError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"check_raft_manifest.py: error: {exc}", file=sys.stderr)
        raise SystemExit(1)
