#!/usr/bin/env python3
"""Materialize the P25-6 carried candidate/artifact-map templates from the linux export.

Finding A (PR #21 review): the checked-in ``bakeoff/p25-6/inputs/candidate-entries.json`` and
``bakeoff/p25-6/inputs/artifact-map.json`` must NOT ship a macOS-arm64 identity.  The candidate
ONNX is not committed; CI exports a fresh ``linux-x86_64`` artifact and rewrites
``models/sea-raft-m.json`` with a ``linux-x86_64`` ``platform_artifacts`` row before packaging.
On that target the driver's ``validate_manifest_artifact`` re-hashes the *packaged linux* ONNX and
raises ``artifact_hash_mismatch`` against any carried macOS binding (``platform: macos-arm64`` plus
the macOS ``artifact_sha256``/``manifest_sha256``/``export_environment_sha256``) before any profile
runs.

So the checked-in inputs are platform-neutral PLACEHOLDER templates: every platform-specific
identity field carries the :data:`PLACEHOLDER` sentinel, and CI fills them from the freshly
exported linux ``models/sea-raft-m.json`` immediately before packaging -- exactly as the admission
document is materialized after export (``scripts/ci-p25-6-qualify.sh`` + ``.github/workflows/ci.yml``).
The linux artifact SHA256 is therefore *bound from the generated manifest*, never hardcoded as a
reproducible-forever constant, so a re-export cannot silently drift the shipped identity.

This module is the single source of truth for that fill: ``scripts/ci-p25-6-qualify.sh`` calls the
CLI, and ``tools/p25_5/p25_6_inputs_tests.py`` calls the same functions against a staged linux
manifest fixture -- so the inputs test is authoritative over exactly what CI ships.

The non-identity fields (``candidate_id``, ``status``, ``measurement_status``, ``source_commit``,
``checkpoint_sha256``, ``measurement_providers``, ``exclusion_reason`` and every license/notice
surface) are provenance/legal content that does not change with the export platform; they are
carried verbatim and are never touched here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

# Sentinel every platform-specific identity field carries in the checked-in template. Chosen to be
# obviously non-hex and non-integer so a stale binding can never be mistaken for a real value, and
# so the report-v2 candidate schema (hex patterns, positive-int size) rejects an un-materialized
# template rather than silently shipping it.
PLACEHOLDER = "materialize-from-linux-manifest"

# candidate-entries identity fields filled from the exported linux platform_artifacts row.
CANDIDATE_IDENTITY_FIELDS: tuple[str, ...] = (
    "artifact_sha256",
    "export_environment_sha256",
    "manifest_sha256",
    "artifact_size_bytes",
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class MaterializeError(RuntimeError):
    """Fail-closed error for an unusable manifest or a template that is not a clean placeholder."""


def _linux_export_row(manifest: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    """Return the (platform, platform_artifacts row) for the manifest's exported linux target.

    Refuses any non-linux ``export.platform`` so this can never re-bind the shipped inputs to the
    checked-in macOS export: it must be handed the freshly exported linux manifest.
    """

    export = manifest.get("export")
    if not isinstance(export, Mapping):
        raise MaterializeError("manifest has no export object")
    platform = export.get("platform")
    if not isinstance(platform, str) or not platform.startswith("linux"):
        raise MaterializeError(
            f"manifest export.platform must be a linux target (got {platform!r}); "
            "materialization requires the freshly exported linux-x86_64 manifest"
        )
    rows = export.get("platform_artifacts")
    if not isinstance(rows, list) or not rows:
        raise MaterializeError("manifest export.platform_artifacts must be a non-empty list")
    row = next(
        (r for r in rows if isinstance(r, Mapping) and r.get("platform") == platform),
        None,
    )
    if row is None:
        raise MaterializeError(f"manifest has no platform_artifacts row for {platform!r}")
    return platform, row


def _require_hex64(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise MaterializeError(f"linux manifest {label} is not a lowercase 64-hex digest: {value!r}")
    return value


def _manifest_file_sha256(manifest_path: Path) -> str:
    return hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()


def _linux_identity(manifest: Mapping[str, Any], manifest_path: Path) -> dict[str, Any]:
    """The exact platform-specific identity the carried candidate entry must ship on linux."""

    _platform, row = _linux_export_row(manifest)
    size = row.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise MaterializeError(f"linux manifest export size_bytes must be a positive int (got {size!r})")
    return {
        "artifact_sha256": _require_hex64(row.get("sha256"), "export.sha256"),
        "export_environment_sha256": _require_hex64(
            row.get("export_environment_sha256"), "export.export_environment_sha256"
        ),
        "manifest_sha256": _manifest_file_sha256(manifest_path),
        "artifact_size_bytes": size,
    }


def _candidate_id(manifest: Mapping[str, Any]) -> str:
    candidate = manifest.get("candidate")
    candidate_id = candidate.get("id") if isinstance(candidate, Mapping) else None
    if not isinstance(candidate_id, str) or not candidate_id:
        raise MaterializeError("manifest candidate.id must be a non-empty string")
    return candidate_id


def materialize_candidate_entries(
    entries: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    manifest_path: Path | str,
) -> list[dict[str, Any]]:
    """Fill the manifest candidate's placeholder identity fields from the linux export row.

    Every carried identity field for that candidate MUST equal :data:`PLACEHOLDER` first; a
    non-placeholder value means someone hardcoded a platform binding into the checked-in template,
    which is exactly the defect this guards, so it fails closed rather than overwriting silently.
    """

    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)) or not entries:
        raise MaterializeError("candidate-entries.json must be a non-empty list")
    candidate_id = _candidate_id(manifest)
    identity = _linux_identity(manifest, Path(manifest_path))

    materialized: list[dict[str, Any]] = []
    filled = False
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise MaterializeError("each candidate-entries.json item must be an object")
        item = dict(entry)
        if item.get("candidate_id") == candidate_id:
            for field in CANDIDATE_IDENTITY_FIELDS:
                if item.get(field) != PLACEHOLDER:
                    raise MaterializeError(
                        f"candidate {candidate_id!r} field {field!r} must be the {PLACEHOLDER!r} "
                        f"placeholder in the checked-in template (got {item.get(field)!r}); refusing "
                        "to overwrite a hardcoded platform binding"
                    )
                item[field] = identity[field]
            filled = True
        materialized.append(item)
    if not filled:
        raise MaterializeError(
            f"candidate-entries.json has no entry for manifest candidate {candidate_id!r}"
        )
    return materialized


def materialize_artifact_map(
    artifact_map: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the manifest candidate's artifact-map ``platform`` to the exported linux target.

    ``artifact-map.json`` carries no hashes -- ``validate_manifest_artifact`` re-hashes the packaged
    ONNX -- so its only platform-specific field is ``platform``, which selects the manifest
    ``platform_artifacts`` row.  A macOS ``platform`` is precisely what makes the driver select the
    macOS row and raise ``artifact_hash_mismatch`` on the linux box.
    """

    if not isinstance(artifact_map, Mapping) or not artifact_map:
        raise MaterializeError("artifact-map.json must be a non-empty object")
    candidate_id = _candidate_id(manifest)
    platform, _row = _linux_export_row(manifest)

    materialized: dict[str, Any] = {}
    filled = False
    for cid, entry in artifact_map.items():
        if not isinstance(entry, Mapping):
            raise MaterializeError(f"artifact-map entry for {cid!r} must be an object")
        item = dict(entry)
        if cid == candidate_id:
            if item.get("platform") != PLACEHOLDER:
                raise MaterializeError(
                    f"artifact-map entry {cid!r} platform must be the {PLACEHOLDER!r} placeholder "
                    f"in the checked-in template (got {item.get('platform')!r}); refusing to "
                    "overwrite a hardcoded platform binding"
                )
            item["platform"] = platform
            filled = True
        materialized[cid] = item
    if not filled:
        raise MaterializeError(
            f"artifact-map.json has no entry for manifest candidate {candidate_id!r}"
        )
    return materialized


def _load_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MaterializeError(f"cannot read JSON from {path}: {exc}") from exc


def _write_json(path: Path, data: Any) -> None:
    # Preserve field order (dicts keep insertion order); do not sort so the materialized file stays
    # human-readable and diff-stable against the checked-in template.
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.chmod(path, 0o644)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="freshly exported linux models/sea-raft-m.json")
    parser.add_argument("--candidate-entries", required=True, type=Path)
    parser.add_argument("--artifact-map", required=True, type=Path)
    parser.add_argument(
        "--out-candidate-entries",
        type=Path,
        help="output path for the materialized candidate-entries.json (default: overwrite in place)",
    )
    parser.add_argument(
        "--out-artifact-map",
        type=Path,
        help="output path for the materialized artifact-map.json (default: overwrite in place)",
    )
    args = parser.parse_args(argv)

    manifest = _load_json(args.manifest)
    entries = _load_json(args.candidate_entries)
    artifact_map = _load_json(args.artifact_map)

    materialized_entries = materialize_candidate_entries(entries, manifest, args.manifest)
    materialized_map = materialize_artifact_map(artifact_map, manifest)

    out_entries = args.out_candidate_entries or args.candidate_entries
    out_map = args.out_artifact_map or args.artifact_map
    _write_json(out_entries, materialized_entries)
    _write_json(out_map, materialized_map)

    platform, row = _linux_export_row(manifest)
    print(f"P25-6 inputs materialized for {platform}:")
    print(f"  candidate-entries: {out_entries}")
    print(f"  artifact-map:      {out_map}")
    print(f"  artifact_sha256:   {row['sha256']}")
    print(f"  artifact_size:     {row['size_bytes']} bytes")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MaterializeError as exc:
        print(f"p25_6_materialize_inputs.py: error: {exc}", file=sys.stderr)
        raise SystemExit(1)
