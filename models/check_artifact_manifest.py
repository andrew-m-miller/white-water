#!/usr/bin/env python3
"""Validate a candidate-neutral Phase 2.5 artifact manifest and optional payloads."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from artifact_workflow import (  # type: ignore  # pylint: disable=wrong-import-position
    ArtifactError,
    load_manifest,
    validate_all_artifacts,
    validate_artifact,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "artifact",
        type=Path,
        nargs="*",
        help="optional exact artifact path(s); otherwise only the manifest is checked",
    )
    parser.add_argument(
        "--platform",
        help="platform entry to use for each supplied artifact (one artifact only)",
    )
    parser.add_argument(
        "--all-present",
        action="store_true",
        help="validate every declared platform artifact that exists beside the manifest",
    )
    parser.add_argument(
        "--no-protocol",
        action="store_true",
        help="skip the frozen P25-0 tensor-contract compatibility check",
    )
    args = parser.parse_args()
    protocol_path = False if args.no_protocol else None
    manifest = load_manifest(args.manifest, protocol_path=protocol_path)
    if args.platform and len(args.artifact) != 1:
        parser.error("--platform requires exactly one artifact path")
    for artifact in args.artifact:
        validate_artifact(manifest, args.manifest, artifact, platform=args.platform)
    if args.all_present:
        validate_all_artifacts(manifest, args.manifest)
    print(
        f"artifact manifest valid: candidate={manifest['candidate']['id']} "
        f"platform={manifest['export']['platform']} mode=0644"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ArtifactError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"check_artifact_manifest.py: error: {exc}", file=sys.stderr)
        raise SystemExit(1)

