#!/usr/bin/env python3
"""Validate a hash-bound, operator-supplied P25-5 legal-review attestation.

The attestation is evidence that a human reviewed the exact candidate identities that a
qualification run is about to measure.  It is deliberately not a legal decision made by CI:
the operator supplies the JSON and its SHA256 at workflow dispatch time.  This module only
checks that the input is explicit, complete, and bound to the active protocol, candidate
source/checkpoint identities, and exact manifest licence evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Sequence


LEGAL_REVIEW_SCHEMA_ID = "whitewater-p25-legal-review-v1"
ACTIVE_PROTOCOL_ID = "whitewater-p25-v2"
LEGAL_SURFACES = ("code", "checkpoint", "backbone")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_MODE = 0o644


class LegalReviewError(ValueError):
    """A malformed, unbound, or incomplete operator attestation."""


@dataclass(frozen=True)
class CandidateIdentity:
    candidate_id: str
    source_commit: str
    checkpoint_sha256: str
    licenses_sha256: str


@dataclass(frozen=True)
class LegalReview:
    sha256: str
    protocol_sha256: str
    reviewed_surfaces: tuple[str, ...]
    candidate_identities: Mapping[str, CandidateIdentity]
    reviewer: str
    reviewed_at: str
    statement: str


def canonical_sha256(value: Any) -> str:
    """Hash one JSON value using the repository's deterministic JSON encoding."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate object members instead of letting the last one silently win."""

    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise LegalReviewError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise LegalReviewError(f"could not hash {path}: {exc}") from exc
    return digest.hexdigest()


def _require_regular_0644(path: Path, label: str) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise LegalReviewError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise LegalReviewError(f"{label} must not be a symlink: {path}")
    if not stat.S_ISREG(info.st_mode):
        raise LegalReviewError(f"{label} must be a regular file: {path}")
    mode = stat.S_IMODE(info.st_mode)
    if mode != EXPECTED_MODE:
        raise LegalReviewError(f"{label} has mode {mode:04o}; expected 0644: {path}")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise LegalReviewError(f"{label} must be a non-empty string without NUL")
    return value.strip()


def _sha(value: Any, label: str) -> str:
    value = _text(value, label)
    if _SHA256.fullmatch(value) is None:
        raise LegalReviewError(f"{label} must be a lowercase SHA256")
    return value


def _commit(value: Any, label: str) -> str:
    value = _text(value, label)
    if _COMMIT.fullmatch(value) is None:
        raise LegalReviewError(f"{label} must be a lowercase 40-hex commit")
    return value


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    _require_regular_0644(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object_no_duplicates)
    except LegalReviewError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise LegalReviewError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise LegalReviewError(f"{label} must be a JSON object")
    return value


def _validate_reviewed_at(value: Any) -> str:
    text = _text(value, "reviewed_at")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LegalReviewError("reviewed_at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise LegalReviewError("reviewed_at must include an explicit timezone")
    return text


def load_legal_review(
    review_path: Path | str,
    expected_sha256: str | None,
    *,
    protocol_path: Path | str,
) -> LegalReview:
    """Load and validate one operator attestation, including its external SHA256 binding."""

    path = Path(review_path)
    if expected_sha256 is None:
        raise LegalReviewError("the legal-review SHA256 is required; CI will not infer it")
    expected_sha = _sha(expected_sha256, "expected legal-review SHA256")
    _require_regular_0644(path, "legal-review file")
    actual_sha = _sha256_file(path)
    if actual_sha != expected_sha:
        raise LegalReviewError(
            f"legal-review SHA256 mismatch: expected {expected_sha}, got {actual_sha}"
        )

    protocol_file = Path(protocol_path)
    protocol = _load_json(protocol_file, "protocol")
    if protocol.get("protocol_id") != ACTIVE_PROTOCOL_ID:
        raise LegalReviewError(f"protocol must identify active {ACTIVE_PROTOCOL_ID}")
    protocol_sha = _sha256_file(protocol_file)

    document = _load_json(path, "legal-review file")
    required = {
        "schema_id",
        "protocol_id",
        "protocol_sha256",
        "candidate_identities",
        "reviewed_surfaces",
        "reviewed",
        "reviewer",
        "reviewed_at",
        "statement",
    }
    if set(document) != required:
        missing = sorted(required - set(document))
        extra = sorted(set(document) - required)
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if extra:
            details.append("unknown: " + ", ".join(extra))
        raise LegalReviewError("legal-review fields are not exact (" + "; ".join(details) + ")")
    if document["schema_id"] != LEGAL_REVIEW_SCHEMA_ID:
        raise LegalReviewError(f"legal-review schema must be {LEGAL_REVIEW_SCHEMA_ID}")
    if document["protocol_id"] != ACTIVE_PROTOCOL_ID:
        raise LegalReviewError(f"legal-review protocol must identify active {ACTIVE_PROTOCOL_ID}")
    if _sha(document["protocol_sha256"], "legal-review protocol_sha256") != protocol_sha:
        raise LegalReviewError(
            "legal-review protocol_sha256 does not match the active protocol bytes"
        )
    if document["reviewed"] is not True:
        raise LegalReviewError("legal-review reviewed must be explicit true")

    raw_surfaces = document["reviewed_surfaces"]
    if not isinstance(raw_surfaces, list) or not all(isinstance(item, str) and item for item in raw_surfaces):
        raise LegalReviewError("legal-review reviewed_surfaces must be a string array")
    if len(raw_surfaces) != len(set(raw_surfaces)):
        raise LegalReviewError("legal-review reviewed_surfaces must be unique")
    unknown = sorted(set(raw_surfaces) - set(LEGAL_SURFACES))
    if unknown:
        raise LegalReviewError("unknown legal-review surface(s): " + ", ".join(unknown))
    missing = [surface for surface in LEGAL_SURFACES if surface not in raw_surfaces]
    if missing:
        raise LegalReviewError(
            "legal-review must explicitly cover every legal surface; missing: "
            + ", ".join(missing)
        )
    surfaces = tuple(surface for surface in LEGAL_SURFACES if surface in raw_surfaces)

    raw_identities = document["candidate_identities"]
    if not isinstance(raw_identities, list) or not raw_identities:
        raise LegalReviewError("legal-review candidate_identities must be a non-empty array")
    identities: dict[str, CandidateIdentity] = {}
    identity_fields = {"candidate_id", "source_commit", "checkpoint_sha256", "licenses_sha256"}
    for index, raw_identity in enumerate(raw_identities):
        label = f"candidate_identities[{index}]"
        if not isinstance(raw_identity, dict):
            raise LegalReviewError(f"{label} must be an object")
        if set(raw_identity) != identity_fields:
            raise LegalReviewError(
                f"{label} fields must be exactly "
                "candidate_id/source_commit/checkpoint_sha256/licenses_sha256"
            )
        candidate_id = _text(raw_identity["candidate_id"], f"{label}.candidate_id")
        if candidate_id in identities:
            raise LegalReviewError(f"duplicate legal-review candidate identity: {candidate_id}")
        identities[candidate_id] = CandidateIdentity(
            candidate_id=candidate_id,
            source_commit=_commit(raw_identity["source_commit"], f"{label}.source_commit"),
            checkpoint_sha256=_sha(raw_identity["checkpoint_sha256"], f"{label}.checkpoint_sha256"),
            licenses_sha256=_sha(raw_identity["licenses_sha256"], f"{label}.licenses_sha256"),
        )

    reviewer = _text(document["reviewer"], "reviewer")
    reviewed_at = _validate_reviewed_at(document["reviewed_at"])
    statement = _text(document["statement"], "statement")
    return LegalReview(
        sha256=actual_sha,
        protocol_sha256=protocol_sha,
        reviewed_surfaces=surfaces,
        candidate_identities=identities,
        reviewer=reviewer,
        reviewed_at=reviewed_at,
        statement=statement,
    )


def validate_candidate_identities(
    review: LegalReview,
    manifests: Mapping[str, Mapping[str, Any]],
) -> None:
    """Bind an attestation to exact candidate and licence identities in manifests."""

    manifest_ids: set[str] = set()
    for supplied_id, manifest in manifests.items():
        candidate = manifest.get("candidate")
        if not isinstance(candidate, dict):
            raise LegalReviewError(f"{supplied_id}: manifest candidate must be an object")
        candidate_id = _text(candidate.get("id"), f"{supplied_id}: manifest candidate.id")
        if candidate_id != supplied_id:
            raise LegalReviewError(
                f"manifest key {supplied_id!r} does not match candidate id {candidate_id!r}"
            )
        manifest_ids.add(candidate_id)
        identity = review.candidate_identities.get(candidate_id)
        if identity is None:
            raise LegalReviewError(f"legal-review has no identity for candidate {candidate_id}")
        upstream = manifest.get("upstream")
        checkpoint = manifest.get("checkpoint")
        if not isinstance(upstream, dict) or not isinstance(checkpoint, dict):
            raise LegalReviewError(f"{candidate_id}: manifest identity sections are missing")
        if upstream.get("commit") != identity.source_commit:
            raise LegalReviewError(f"{candidate_id}: legal-review source commit does not match manifest")
        if checkpoint.get("sha256") != identity.checkpoint_sha256:
            raise LegalReviewError(f"{candidate_id}: legal-review checkpoint SHA256 does not match manifest")
        licenses = manifest.get("licenses")
        if not isinstance(licenses, dict):
            raise LegalReviewError(f"{candidate_id}: manifest licenses object is missing")
        if canonical_sha256(licenses) != identity.licenses_sha256:
            raise LegalReviewError(f"{candidate_id}: legal-review licenses SHA256 does not match manifest")
    review_ids = set(review.candidate_identities)
    if review_ids != manifest_ids:
        missing = sorted(manifest_ids - review_ids)
        extra = sorted(review_ids - manifest_ids)
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if extra:
            details.append("unknown: " + ", ".join(extra))
        raise LegalReviewError("legal-review candidate identities do not exactly match manifests (" + "; ".join(details) + ")")


def _load_manifest_identity(path: Path, expected_id: str) -> Mapping[str, Any]:
    manifest = _load_json(path, f"manifest {expected_id}")
    candidate = manifest.get("candidate")
    if not isinstance(candidate, dict) or candidate.get("id") != expected_id:
        raise LegalReviewError(f"manifest {expected_id} does not identify candidate {expected_id}")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        action="append",
        nargs=2,
        metavar=("CANDIDATE_ID", "PATH"),
        required=True,
        help="candidate ID and exact manifest path; repeat for every candidate",
    )
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--sha256", required=True, help="SHA256 supplied by the operator for the exact review file")
    args = parser.parse_args(argv)
    try:
        review = load_legal_review(args.review, args.sha256, protocol_path=args.protocol)
        manifests = {
            candidate_id: _load_manifest_identity(Path(path), candidate_id)
            for candidate_id, path in args.manifest
        }
        if len(manifests) != len(args.manifest):
            raise LegalReviewError("duplicate --manifest candidate ID")
        validate_candidate_identities(review, manifests)
        print(
            json.dumps(
                {
                    "legal_review_sha256": review.sha256,
                    "protocol_sha256": review.protocol_sha256,
                    "reviewed_surfaces": list(review.reviewed_surfaces),
                    "candidate_ids": sorted(review.candidate_identities),
                },
                sort_keys=True,
            )
        )
        return 0
    except (LegalReviewError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"p25_5 legal review: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
