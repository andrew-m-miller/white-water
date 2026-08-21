#!/usr/bin/env python3
"""Verify a SEA-RAFT model/manifest pair, including payload file permissions.

The mode check is intentional: a CI job running as the file owner can still open a ``0600``
file, so a content/hash-only test would miss the exact failure Flame sees. Each checked file
must be a regular, world-readable ``0644`` file and the model must match the manifest hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import sys


EXPECTED_MODE = 0o644


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is missing or not a regular file: {path}")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != EXPECTED_MODE:
        raise RuntimeError(
            f"{label} has mode {mode:04o}; expected 0644 so Flame can read the payload: {path}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="SEA-RAFT manifest to validate")
    parser.add_argument(
        "artifact",
        type=Path,
        nargs="+",
        help="one or more model files to validate against the manifest hash",
    )
    args = parser.parse_args()

    require_file(args.manifest, "manifest")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_hash = manifest["export"]["sha256"]
    expected_size = manifest["export"]["size_bytes"]
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise RuntimeError("manifest export.sha256 is missing or malformed")
    if not isinstance(expected_size, int) or expected_size <= 0:
        raise RuntimeError("manifest export.size_bytes is missing or malformed")

    for artifact in args.artifact:
        require_file(artifact, "SEA-RAFT artifact")
        actual_size = artifact.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(
                f"SEA-RAFT artifact size is {actual_size}, expected {expected_size}: {artifact}"
            )
        actual_hash = sha256_file(artifact)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"SEA-RAFT artifact SHA256 is {actual_hash}, expected {expected_hash}: {artifact}"
            )

    print(
        f"SEA-RAFT artifact permissions/hash valid: {len(args.artifact)} file(s), mode 0644, "
        f"SHA256 {expected_hash}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, KeyError, TypeError, ValueError) as exc:
        print(f"check_searaft_artifact.py: error: {exc}", file=sys.stderr)
        raise SystemExit(1)
