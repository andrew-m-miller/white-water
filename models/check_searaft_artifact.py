#!/usr/bin/env python3
"""Compatibility wrapper for the candidate-neutral artifact checker.

The Phase 0B command name remains valid for existing CI and host instructions; all validation
now goes through the shared candidate-neutral workflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from artifact_workflow import (  # type: ignore  # pylint: disable=wrong-import-position
    ArtifactError,
    load_manifest,
    validate_artifact,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("artifact", type=Path, nargs="+")
    parser.add_argument("--platform", help="platform artifact entry; defaults to manifest platform")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    for artifact in args.artifact:
        validate_artifact(manifest, args.manifest, artifact, platform=args.platform)
    print(
        f"artifact permissions/hash valid: {len(args.artifact)} file(s), mode 0644, "
        f"SHA256 {manifest['export']['sha256']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ArtifactError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"check_searaft_artifact.py: error: {exc}", file=sys.stderr)
        raise SystemExit(1)
