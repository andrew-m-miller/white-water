#!/usr/bin/env python3
"""SEA-RAFT migration gate over the candidate-neutral artifact manifest."""

from __future__ import annotations

import sys
from pathlib import Path

from artifact_workflow import (  # type: ignore  # pylint: disable=wrong-import-position
    ArtifactError,
    load_manifest,
    validate_artifact,
)


EXPECTED_SOURCE_COMMIT = "9137517ba24e628442aec097d3afe71d03503b75"
EXPECTED_CHECKPOINT_REVISION = "ea21e467a7076978b251e09d55751fcce166c2f8"


def main() -> int:
    manifest_path = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("sea-raft-m.json")
    )
    manifest = load_manifest(manifest_path)

    # These checks protect the Phase 0B record while the generic schema owns all common
    # candidate fields.  They intentionally assert evidence, not a shipping/default choice.
    if manifest["candidate"]["id"] != "sea-raft-m":
        raise ArtifactError("SEA-RAFT migration changed candidate.id")
    if manifest["upstream"]["commit"] != EXPECTED_SOURCE_COMMIT:
        raise ArtifactError("SEA-RAFT migration changed the pinned upstream commit")
    if manifest["checkpoint"]["revision"] != EXPECTED_CHECKPOINT_REVISION:
        raise ArtifactError("SEA-RAFT migration changed the pinned checkpoint revision")
    checkpoint_url = manifest["checkpoint"]["url"]
    if EXPECTED_CHECKPOINT_REVISION not in checkpoint_url or "/main/" in checkpoint_url:
        raise ArtifactError("SEA-RAFT checkpoint URL is not revision-pinned")
    contract = manifest["tensor_contract"]
    if contract["output"]["direction"] != "image1_to_image2":
        raise ArtifactError("SEA-RAFT output direction is not preserved")
    if contract["padding"]["multiple"] != 8:
        raise ArtifactError("SEA-RAFT padding multiple is not preserved")
    if contract["iterations"] != "4_baked_into_graph":
        raise ArtifactError("SEA-RAFT iteration handling is not preserved")
    if manifest["model"]["config"]["scale"] != -1:
        raise ArtifactError("SEA-RAFT model scale is not preserved")
    if contract["upstream_custom_py_input_scale"] != 0.5:
        raise ArtifactError("SEA-RAFT upstream input scale is not preserved")
    if contract["exported_forward_input_scale"] != 1.0:
        raise ArtifactError("SEA-RAFT exported input scale is not preserved")
    if any(value % 8 for value in manifest["validation"]["second_dynamic_shape"][2:]):
        raise ArtifactError("SEA-RAFT dynamic validation shape is not a multiple of eight")

    artifact_path = manifest_path.parent / manifest["export"]["artifact"]
    # A source checkout intentionally omits the ignored ONNX payload.  If anything exists at
    # the declared path, however, validate it fully; this includes broken symlinks via lstat.
    try:
        artifact_path.lstat()
    except FileNotFoundError:
        pass
    else:
        validate_artifact(manifest, manifest_path, artifact_path)

    print("SEA-RAFT manifest migration is valid")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ArtifactError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"check_searaft_manifest.py: error: {exc}", file=sys.stderr)
        raise SystemExit(1)
