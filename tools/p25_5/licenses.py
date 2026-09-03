#!/usr/bin/env python3
"""Collect deterministic candidate and runtime licence/notice inputs.

This module is intentionally a local, fail-closed boundary.  It never downloads a licence or
tries to infer a licence grant from a package name.  Candidate input declarations name the
exact evidence files reviewed for each manifest surface.  Runtime declarations name the exact
licence and notice files for every package in an explicit conda lock, and may additionally name
non-conda components (such as a native ORT archive and bridge).  The installed ``conda-meta``
records and component payload hashes are checked before any output is produced.

The two commands write small, reproducible bundles:

``candidate``
    ``LICENSES.txt``, ``NOTICES.txt`` and ``candidate-license-inventory.json``.

``runtime``
    ``LICENSES.txt``, ``NOTICES.txt`` and ``runtime-license-inventory.json``.

Evidence with identical bytes is included once in an aggregate file and listed against every
surface/package that uses it.  The aggregate files are attribution indexes plus the exact local
evidence bytes; no legal conclusion is created by this tool.

The runtime inventory also carries a relocation-stable payload identity.  Its canonical walk
excludes only ``conda-meta/``, any ``__pycache__/`` directory, and files ending in ``.pyc``;
conda-pack activation/unpack files remain included and therefore reviewed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from email.parser import Parser
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit


LICENSE_INPUT_SCHEMA_ID = "whitewater-p25-candidate-license-input-v1"
RUNTIME_INPUT_SCHEMA_ID = "whitewater-p25-runtime-license-input-v1"
RUNTIME_INVENTORY_SCHEMA_ID = "whitewater-p25-runtime-license-inventory-v1"
RUNTIME_CONTENT_SCHEMA_ID = "whitewater-p25-runtime-content-v2"
RUNTIME_REVIEW_SCHEMA_ID = "whitewater-p25-runtime-legal-review-v1"
RUNTIME_COMPONENT_INPUT_SCHEMA_ID = "whitewater-p25-runtime-component-input-v1"
RUNTIME_SUPPLEMENT_INPUT_SCHEMA_ID = "whitewater-p25-runtime-license-supplement-v1"
CANDIDATE_INVENTORY_SCHEMA_ID = "whitewater-p25-candidate-license-inventory-v1"
ARTIFACT_SCHEMA_ID = "whitewater-p25-artifact-v1"
CANDIDATE_SURFACES = ("code", "checkpoint", "backbone")
EXPECTED_MODE = 0o644
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_URL_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_UNKNOWN_LICENSES = frozenset({"", "unknown", "other", "n/a", "na", "none"})
# Conservative map from common non-SPDX licence spellings to their SPDX identifier. Only
# UNAMBIGUOUS spellings are listed: a bare "BSD License" (which does not distinguish 2- vs
# 3-clause) is deliberately absent and left exactly as the distribution declared it rather than
# guessed. Keys are casefolded.
_LICENSE_ALIASES = {
    "mit": "MIT",
    "mit license": "MIT",
    "apache 2.0": "Apache-2.0",
    "apache-2.0": "Apache-2.0",
    "apache license 2.0": "Apache-2.0",
    "apache software license": "Apache-2.0",
    "bsd-3-clause": "BSD-3-Clause",
    "bsd 3-clause": "BSD-3-Clause",
    "new bsd license": "BSD-3-Clause",
    "bsd-2-clause": "BSD-2-Clause",
    "python-2.0": "Python-2.0",
    "psf-2.0": "Python-2.0",
    "python software foundation license": "Python-2.0",
    "mozilla public license 2.0 (mpl 2.0)": "MPL-2.0",
    "mpl-2.0": "MPL-2.0",
    "the unlicense (unlicense)": "Unlicense",
    "isc license (iscl)": "ISC",
}


def _looks_like_license_identifier(value: str) -> bool:
    """A short single-line token is an identifier; a multi-line or long value is licence TEXT.

    Some wheels (e.g. numpy) dump the full licence text into the free-text ``License`` field.
    That is not a usable identifier, so it is ignored in favour of the ``License ::`` classifier.
    """

    return bool(value) and "\n" not in value and len(value) <= 64


def _normalize_license_token(value: str) -> str:
    """Reduce a declared licence label to a tidy identifier without inventing terms.

    Strips a trove-classifier prefix (``License :: OSI Approved :: X`` -> ``X``) and maps only
    unambiguous non-SPDX spellings to their SPDX id; anything else is returned unchanged.
    """

    token = value.strip()
    if "::" in token:
        token = token.split("::")[-1].strip()
    return _LICENSE_ALIASES.get(token.casefold(), token)
_RUNTIME_CONTENT_EXCLUDED_ROOTS = ("conda-meta",)
_RUNTIME_CONTENT_EXCLUDED_DIRECTORIES = ("__pycache__",)
_RUNTIME_CONTENT_EXCLUDED_SUFFIXES = (".pyc",)
_RUNTIME_PREFIX_TOKEN = b"<WHITEWATER-P25-RUNTIME-PREFIX>"


class LicenseInputError(ValueError):
    """A malformed, incomplete, or unbound licence/notice input."""


@dataclass(frozen=True)
class EvidenceFile:
    path: Path
    sha256: str
    size_bytes: int
    content: bytes


@dataclass(frozen=True)
class CandidateSurface:
    surface: str
    license_id: str
    commercial_use_permitted: str
    redistribution_permitted: str
    source: str
    license: EvidenceFile
    notice: EvidenceFile


@dataclass(frozen=True)
class CandidateInput:
    candidate_id: str
    manifest_path: Path
    manifest_sha256: str
    licenses_sha256: str
    manifest: Mapping[str, Any]
    surfaces: tuple[CandidateSurface, ...]


@dataclass(frozen=True)
class RuntimePackage:
    source: str
    package_url: str
    canonical_url: str
    name: str
    version: str
    build: str
    license_id: str
    license_family: str | None
    metadata_path: Path
    metadata_sha256: str
    license: tuple[EvidenceFile, ...]
    notice: tuple[EvidenceFile, ...]


@dataclass(frozen=True)
class RuntimeComponent:
    component_id: str
    version: str
    source: str
    license_id: str
    payload_path: str
    payload_sha256: str
    license: tuple[EvidenceFile, ...]
    notice: tuple[EvidenceFile, ...]


def canonical_sha256(value: Any) -> str:
    """Hash one JSON value with the repository's canonical JSON encoding."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise LicenseInputError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _load_json(path: Path, label: str) -> Any:
    _require_regular(path, label)
    try:
        return json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys
        )
    except LicenseInputError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise LicenseInputError(f"{label} is not valid JSON: {exc}") from exc


def _require_regular(path: Path, label: str, mode: int = EXPECTED_MODE, *, require_mode: bool = True) -> None:
    # ``require_mode=False`` keeps the regular-file and symlink-rejection guards but skips the
    # permission-bits check. It is used for upstream files whose mode we do not control -- conda
    # package-cache license evidence (info/licenses/*, mode set by conda-forge) and pip dist-info
    # METADATA (mode set by the wheel/pip; e.g. protobuf ships it 0755). Integrity for those files
    # is the content and its recorded SHA256, still enforced (by _read_evidence, or by the
    # metadata_sha256 binding for dist metadata). Every file we own or carry (aggregated
    # RUNTIME-LICENSES/NOTICES, supplement/component/harvested evidence, the explicit lock, reviews,
    # the inventory) keeps the default 0644 requirement.
    try:
        info = path.lstat()
    except OSError as exc:
        raise LicenseInputError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise LicenseInputError(f"{label} must not be a symlink: {path}")
    if not stat.S_ISREG(info.st_mode):
        raise LicenseInputError(f"{label} must be a regular file: {path}")
    actual = stat.S_IMODE(info.st_mode)
    if require_mode and actual != mode:
        raise LicenseInputError(f"{label} has mode {actual:04o}; expected {mode:04o}: {path}")


def _require_directory(path: Path, label: str) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise LicenseInputError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise LicenseInputError(f"{label} must be a non-symlink directory: {path}")


def _text(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(ord(character) < 0x20 for character in value)
    ):
        raise LicenseInputError(f"{label} must be a non-empty string without control characters")
    return value.strip()


def _hash(value: Any, label: str) -> str:
    text = _text(value, label)
    if _SHA256.fullmatch(text) is None:
        raise LicenseInputError(f"{label} must be a lowercase SHA256")
    return text


def _validate_reviewed_at(value: Any) -> str:
    """Require the same timezone-qualified ISO-8601 timestamp as candidate review input."""

    text = _text(value, "runtime legal-review reviewed_at")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LicenseInputError(
            "runtime legal-review reviewed_at must be an ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise LicenseInputError(
            "runtime legal-review reviewed_at must include an explicit timezone"
        )
    return text


def _resolve_local(base: Path, value: Any, label: str) -> Path:
    text = _text(value, label)
    if _URL_SCHEME.match(text):
        raise LicenseInputError(f"{label} must be a local path; downloads are not permitted")
    path = Path(text)
    # Normalize lexical ``..`` components without following a final symlink.  The latter is
    # important: _require_regular must see and reject a symlink rather than silently checking
    # the target that it points at.
    candidate = path if path.is_absolute() else base / path
    return Path(os.path.abspath(candidate))


def _read_evidence(path: Path, expected_sha: str, label: str, *, require_mode: bool = True) -> EvidenceFile:
    _require_regular(path, label, require_mode=require_mode)
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                chunks.append(chunk)
    except OSError as exc:
        raise LicenseInputError(f"could not read {label}: {path}: {exc}") from exc
    content = b"".join(chunks)
    actual = digest.hexdigest()
    if actual != expected_sha:
        raise LicenseInputError(
            f"{label} SHA256 mismatch: expected {expected_sha}, got {actual}: {path}"
        )
    if not content:
        raise LicenseInputError(f"{label} must not be empty: {path}")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LicenseInputError(f"{label} must be UTF-8 text: {path}") from exc
    if "\x00" in text:
        raise LicenseInputError(f"{label} must not contain NUL: {path}")
    return EvidenceFile(path, actual, len(content), content)


def _require_exact_keys(value: Mapping[str, Any], keys: Iterable[str], label: str) -> None:
    expected = set(keys)
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if extra:
            details.append("unknown: " + ", ".join(extra))
        raise LicenseInputError(f"{label} fields are not exact (" + "; ".join(details) + ")")


def _load_manifest(path: Path, expected_candidate_id: str) -> tuple[Mapping[str, Any], str]:
    value = _load_json(path, "candidate manifest")
    if not isinstance(value, dict):
        raise LicenseInputError("candidate manifest must be a JSON object")
    if value.get("schema_id") != ARTIFACT_SCHEMA_ID:
        raise LicenseInputError(f"candidate manifest schema_id must be {ARTIFACT_SCHEMA_ID}")
    candidate = value.get("candidate")
    if not isinstance(candidate, dict) or candidate.get("id") != expected_candidate_id:
        raise LicenseInputError(
            f"candidate manifest id does not match declared candidate {expected_candidate_id!r}"
        )
    licenses = value.get("licenses")
    if not isinstance(licenses, dict) or set(licenses) != set(CANDIDATE_SURFACES):
        raise LicenseInputError("candidate manifest licenses must contain exactly code/checkpoint/backbone")
    for surface in CANDIDATE_SURFACES:
        entry = licenses[surface]
        if not isinstance(entry, dict):
            raise LicenseInputError(f"candidate manifest licenses.{surface} must be an object")
        for field in ("license", "commercial_use_permitted", "redistribution_permitted"):
            _text(entry.get(field), f"candidate manifest licenses.{surface}.{field}")
        if entry["commercial_use_permitted"] not in {"yes", "no", "unknown"}:
            raise LicenseInputError(
                f"candidate manifest licenses.{surface}.commercial_use_permitted is invalid"
            )
        if entry["redistribution_permitted"] not in {"yes", "no", "unknown"}:
            raise LicenseInputError(
                f"candidate manifest licenses.{surface}.redistribution_permitted is invalid"
            )
        _text(entry.get("source"), f"candidate manifest licenses.{surface}.source")
    return value, hashlib.sha256(path.read_bytes()).hexdigest()


def load_candidate_input(path: Path | str) -> CandidateInput:
    """Load an exact candidate evidence declaration and all referenced files."""

    input_path = Path(path)
    document = _load_json(input_path, "candidate licence input")
    if not isinstance(document, dict):
        raise LicenseInputError("candidate licence input must be a JSON object")
    _require_exact_keys(
        document,
        {"schema_id", "candidate_id", "manifest", "manifest_sha256", "licenses_sha256", "surfaces"},
        "candidate licence input",
    )
    if document["schema_id"] != LICENSE_INPUT_SCHEMA_ID:
        raise LicenseInputError(f"candidate licence input schema_id must be {LICENSE_INPUT_SCHEMA_ID}")
    candidate_id = _text(document["candidate_id"], "candidate_id")
    manifest_sha = _hash(document["manifest_sha256"], "manifest_sha256")
    licenses_sha = _hash(document["licenses_sha256"], "licenses_sha256")
    manifest_path = _resolve_local(input_path.parent, document["manifest"], "manifest")
    manifest, actual_manifest_sha = _load_manifest(manifest_path, candidate_id)
    if actual_manifest_sha != manifest_sha:
        raise LicenseInputError(
            f"candidate manifest SHA256 mismatch: expected {manifest_sha}, got {actual_manifest_sha}"
        )
    actual_licenses_sha = canonical_sha256(manifest["licenses"])
    if actual_licenses_sha != licenses_sha:
        raise LicenseInputError(
            f"candidate manifest licenses SHA256 mismatch: expected {licenses_sha}, got {actual_licenses_sha}"
        )
    raw_surfaces = document["surfaces"]
    if not isinstance(raw_surfaces, dict) or set(raw_surfaces) != set(CANDIDATE_SURFACES):
        raise LicenseInputError("candidate licence input surfaces must contain exactly code/checkpoint/backbone")
    surfaces: list[CandidateSurface] = []
    for surface in CANDIDATE_SURFACES:
        raw = raw_surfaces[surface]
        if not isinstance(raw, dict):
            raise LicenseInputError(f"surfaces.{surface} must be an object")
        _require_exact_keys(
            raw,
            {"license_file", "license_sha256", "notice_file", "notice_sha256"},
            f"surfaces.{surface}",
        )
        license_path = _resolve_local(input_path.parent, raw["license_file"], f"surfaces.{surface}.license_file")
        notice_path = _resolve_local(input_path.parent, raw["notice_file"], f"surfaces.{surface}.notice_file")
        license_file = _read_evidence(
            license_path,
            _hash(raw["license_sha256"], f"surfaces.{surface}.license_sha256"),
            f"surfaces.{surface} licence evidence",
        )
        notice_file = _read_evidence(
            notice_path,
            _hash(raw["notice_sha256"], f"surfaces.{surface}.notice_sha256"),
            f"surfaces.{surface} notice evidence",
        )
        manifest_surface = manifest["licenses"][surface]
        surfaces.append(
            CandidateSurface(
                surface=surface,
                license_id=manifest_surface["license"],
                commercial_use_permitted=manifest_surface["commercial_use_permitted"],
                redistribution_permitted=manifest_surface["redistribution_permitted"],
                source=manifest_surface["source"],
                license=license_file,
                notice=notice_file,
            )
        )
    return CandidateInput(
        candidate_id=candidate_id,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        licenses_sha256=licenses_sha,
        manifest=manifest,
        surfaces=tuple(surfaces),
    )


def _canonical_url(value: Any, label: str) -> tuple[str, str]:
    text = _text(value, label)
    parsed = urlsplit(text)
    if parsed.scheme != "https" or not parsed.netloc or parsed.query:
        raise LicenseInputError(f"{label} must be an HTTPS package URL without a query")
    if parsed.fragment and re.fullmatch(r"(?:sha256=)?[0-9a-f]{64}", parsed.fragment) is None:
        raise LicenseInputError(f"{label} has an invalid SHA256 fragment")
    canonical = text.split("#", 1)[0]
    if canonical.endswith("/") or "/" not in urlsplit(canonical).path:
        raise LicenseInputError(f"{label} must identify a package artifact")
    return canonical, text


def _canonical_runtime_identity(value: Any, label: str) -> tuple[str, str, str]:
    """Return (source, internal key, display identity) for conda or pip metadata.

    The evaluator runtime is usually assembled by conda, but a locked environment may carry a
    Python distribution installed from a wheel (including ONNX Runtime GPU builds).  Treating
    every package as a conda URL would either omit that distribution or invent a conda licence
    source, so pip identities are explicit ``pip://name==version`` records instead.
    """

    text = _text(value, label)
    if text.startswith("pip://"):
        payload = text[len("pip://") :]
        if payload.count("==") != 1:
            raise LicenseInputError(f"{label} pip identity must be pip://name==version")
        name, version = payload.split("==", 1)
        if not name or not version or any(character.isspace() for character in payload):
            raise LicenseInputError(f"{label} pip identity must contain a name and version")
        normalized_name = re.sub(r"[-_.]+", "-", name).casefold()
        if re.fullmatch(r"[a-z0-9][a-z0-9-]*", normalized_name) is None:
            raise LicenseInputError(f"{label} pip package name is unsafe")
        display = f"pip://{normalized_name}=={version}"
        return "pip", f"pip:{normalized_name}=={version}", display
    canonical, original = _canonical_url(text, label)
    return "conda", f"conda:{canonical}", original


def parse_explicit_lock(path: Path | str, expected_sha256: str) -> tuple[str, dict[str, str]]:
    """Return the lock SHA and canonical-package-url -> original URL mapping."""

    lock_path = Path(path)
    expected = _hash(expected_sha256, "expected conda lock SHA256")
    _require_regular(lock_path, "conda explicit lock")
    content = lock_path.read_bytes()
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected:
        raise LicenseInputError(f"conda lock SHA256 mismatch: expected {expected}, got {actual}")
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise LicenseInputError("conda explicit lock must be UTF-8 text") from exc
    explicit_index = None
    for index, line in enumerate(lines):
        if line.strip() == "@EXPLICIT":
            if explicit_index is not None:
                raise LicenseInputError("conda explicit lock contains duplicate @EXPLICIT markers")
            explicit_index = index
    if explicit_index is None:
        raise LicenseInputError("conda lock must be an explicit spec containing @EXPLICIT")
    packages: dict[str, str] = {}
    for index, line in enumerate(lines[explicit_index + 1 :], explicit_index + 2):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        canonical, original = _canonical_url(text, f"conda lock line {index}")
        if canonical in packages:
            raise LicenseInputError(f"conda lock contains duplicate package URL: {canonical}")
        packages[canonical] = original
    if not packages:
        raise LicenseInputError("conda explicit lock contains no package URLs")
    return actual, packages


def _metadata_packages(prefix: Path, lock_urls: Mapping[str, str]) -> list[tuple[Mapping[str, Any], Path, str, str]]:
    _require_directory(prefix, "conda environment prefix")
    metadata_dir = prefix / "conda-meta"
    _require_directory(metadata_dir, "conda environment conda-meta directory")
    paths = sorted(metadata_dir.glob("*.json"), key=lambda item: item.name)
    if not paths:
        raise LicenseInputError("conda environment has no conda-meta/*.json package records")
    records: list[tuple[Mapping[str, Any], Path, str, str]] = []
    seen_urls: set[str] = set()
    for path in paths:
        raw = _load_json(path, f"conda metadata {path.name}")
        if not isinstance(raw, dict):
            raise LicenseInputError(f"conda metadata {path.name} must be a JSON object")
        for field in ("name", "version", "build", "url", "license"):
            _text(raw.get(field), f"conda metadata {path.name}.{field}")
        canonical, original = _canonical_url(raw["url"], f"conda metadata {path.name}.url")
        if canonical not in lock_urls:
            raise LicenseInputError(f"conda metadata package is absent from explicit lock: {canonical}")
        if canonical in seen_urls:
            raise LicenseInputError(f"duplicate conda metadata package URL: {canonical}")
        seen_urls.add(canonical)
        license_id = raw["license"].strip()
        if license_id.casefold() in _UNKNOWN_LICENSES:
            raise LicenseInputError(
                f"conda metadata {path.name} has no usable licence identifier; refusing to infer terms"
            )
        records.append((raw, path, f"conda:{canonical}", lock_urls[canonical]))
    missing = sorted(set(lock_urls) - seen_urls)
    if missing:
        raise LicenseInputError("explicit lock packages missing from conda-meta: " + ", ".join(missing))
    return records


def _pip_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def _pip_metadata_packages(prefix: Path, conda_records: Sequence[RuntimePackage]) -> list[RuntimePackage]:
    """Read installed wheel metadata without assuming any particular distribution is conda-owned."""

    # ``conda_records`` is already validated; use its normalized name/version identity for the
    # normal path so a conda-provided dist-info is not counted twice.
    conda_identities = {(_pip_name(item.name), item.version) for item in conda_records}
    # A conda env may expose the same site-packages under more than one path (e.g. a
    # ``lib/python3.1 -> python3.12`` compatibility symlink), so collapse the roots by their real
    # path before scanning to avoid counting every distribution twice.
    site_roots: list[Path] = []
    seen_roots: set[str] = set()
    for candidate in sorted(prefix.glob("lib/python*/site-packages"), key=lambda item: str(item)):
        real = os.path.realpath(candidate)
        if real in seen_roots:
            continue
        seen_roots.add(real)
        site_roots.append(candidate)
    records: list[RuntimePackage] = []
    seen: dict[str, str] = {}
    for site_root in site_roots:
        _require_directory(site_root, "runtime Python site-packages directory")
        candidates = sorted(
            [*site_root.glob("*.dist-info"), *site_root.glob("*.egg-info")],
            key=lambda item: item.name,
        )
        for distribution_dir in candidates:
            _require_directory(distribution_dir, "runtime Python distribution metadata directory")
            metadata_path = distribution_dir / ("METADATA" if distribution_dir.name.endswith(".dist-info") else "PKG-INFO")
            # The dist-info METADATA mode is set by the wheel/pip, not by us (e.g. protobuf ships
            # its METADATA 0755), so it is not our integrity property. The distribution is bound by
            # this file's content SHA256 (metadata_sha256) below; read it mode-agnostically, keeping
            # the regular-file and symlink guards, exactly like upstream conda cache licence files.
            _require_regular(metadata_path, "runtime Python distribution metadata", require_mode=False)
            try:
                metadata_bytes = metadata_path.read_bytes()
                message = Parser().parsestr(metadata_bytes.decode("utf-8"))
            except (OSError, UnicodeError, ValueError) as exc:
                raise LicenseInputError(
                    f"runtime Python distribution metadata is invalid: {metadata_path}: {exc}"
                ) from exc
            name = _text(message.get("Name"), f"{metadata_path}.Name")
            version = _text(message.get("Version"), f"{metadata_path}.Version")
            identity_name = _pip_name(name)
            if (identity_name, version) in conda_identities:
                # The conda package record is authoritative for a distribution installed by
                # conda; do not count its dist-info as an unbound second package.
                continue
            identity = f"pip:{identity_name}=={version}"
            metadata_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
            if identity in seen:
                # The same distribution reached through two paths (e.g. a duplicated or symlinked
                # tree not collapsed by realpath) is the same package; tolerate an identical
                # dist-info and reject only a genuine conflicting second install.
                if seen[identity] == metadata_sha256:
                    continue
                raise LicenseInputError(f"duplicate runtime Python distribution: {identity}")
            seen[identity] = metadata_sha256
            # PEP 639 `License-Expression` is the modern canonical SPDX field and takes
            # precedence over the deprecated free-text `License` field and the `License ::`
            # trove classifiers.  Reading it is not inference: it is the distribution's own
            # declared SPDX identifier.  Newer wheels (e.g. packaging>=24, anyio>=4) carry only
            # this field, so without it the whole pip stack fails closed.  The free-text `License`
            # field is used only when it is a short identifier, not when a wheel dumps its full
            # licence text there (e.g. numpy), in which case the `License ::` classifier is used.
            # The result is tidied by _normalize_license_token, which maps only unambiguous
            # non-SPDX spellings and never guesses an ambiguous one.
            candidate = (message.get("License-Expression") or "").strip()
            if candidate.casefold() in _UNKNOWN_LICENSES:
                license_field = (message.get("License") or "").strip()
                candidate = license_field if _looks_like_license_identifier(license_field) else ""
            if candidate.casefold() in _UNKNOWN_LICENSES:
                classifiers = message.get_all("Classifier", [])
                license_classifiers = [
                    item
                    for item in classifiers
                    if isinstance(item, str) and item.startswith("License ::") and "::" in item
                ]
                if license_classifiers:
                    candidate = license_classifiers[-1]
            license_id = _normalize_license_token(candidate)
            if license_id.casefold() in _UNKNOWN_LICENSES:
                raise LicenseInputError(
                    f"runtime Python distribution {identity} has no usable licence identifier; "
                    "refusing to infer terms"
                )
            records.append(
                RuntimePackage(
                    source="pip",
                    package_url=f"pip://{identity_name}=={version}",
                    canonical_url=identity,
                    name=name,
                    version=version,
                    build="pip",
                    license_id=license_id,
                    license_family=None,
                    metadata_path=metadata_path,
                    metadata_sha256=metadata_sha256,
                    license=(),
                    notice=(),
                )
            )
    return sorted(records, key=lambda item: item.canonical_url)


_PIP_LICENSE_NAME = re.compile(r"^(?:license|licence|copying|copyright)", re.IGNORECASE)
_PIP_NOTICE_NAME = re.compile(r"^(?:notice|authors)", re.IGNORECASE)


def _harvest_pip_license_files(
    dist_info: Path, message: Any
) -> tuple[list[Path], list[Path]]:
    """Collect a wheel's own bundled licence and notice files from its dist-info.

    PEP 639 wheels record ``License-File`` headers whose files are installed under
    ``<dist-info>/licenses/``; older wheels drop ``LICENSE``/``COPYING`` at the dist-info
    root.  Nothing is fetched or inferred: only files the wheel actually ships are read, and
    each must be a non-symlink, non-empty, UTF-8 regular file to be admitted.  A distribution
    that bundles nothing returns empty lists and is handled by the caller (supplement or fail
    closed).
    """

    def _usable(path: Path) -> bool:
        try:
            info = path.lstat()
        except OSError:
            return False
        if not stat.S_ISREG(info.st_mode) or info.st_size == 0:
            return False
        try:
            content = path.read_bytes()
        except OSError:
            return False
        if not content:
            return False
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            return False
        return "\x00" not in text

    candidates: list[Path] = []
    seen: set[Path] = set()

    def _add(path: Path) -> None:
        resolved = Path(os.path.abspath(path))
        if resolved in seen:
            return
        if _usable(path):
            seen.add(resolved)
            candidates.append(path)

    for raw in message.get_all("License-File", []) or []:
        if not isinstance(raw, str):
            continue
        rel = raw.strip()
        if not rel:
            continue
        relative = Path(rel)
        if relative.is_absolute() or any(part in {"", "..", "."} for part in relative.parts):
            continue
        _add(dist_info / "licenses" / relative)
        _add(dist_info / relative)

    if not candidates:
        licenses_dir = dist_info / "licenses"
        if licenses_dir.is_dir():
            for item in sorted(licenses_dir.rglob("*"), key=lambda p: p.as_posix()):
                _add(item)

    if not candidates:
        for item in sorted(dist_info.iterdir(), key=lambda p: p.name):
            if _PIP_LICENSE_NAME.match(item.name) or _PIP_NOTICE_NAME.match(item.name):
                _add(item)

    license_files: list[Path] = []
    notice_files: list[Path] = []
    for path in candidates:
        if _PIP_NOTICE_NAME.match(path.name):
            notice_files.append(path)
        else:
            license_files.append(path)
    return license_files, notice_files


def _parse_evidence_declarations(
    value: Any, label: str, *, allow_empty: bool = False
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        expectation = "array" if allow_empty else "non-empty array"
        raise LicenseInputError(f"{label} must be an {expectation}")
    parsed: list[tuple[str, str]] = []
    for file_index, item in enumerate(value):
        item_label = f"{label}[{file_index}]"
        if not isinstance(item, dict):
            raise LicenseInputError(f"{item_label} must be an object")
        _require_exact_keys(item, {"path", "sha256"}, item_label)
        parsed.append(
            (
                _text(item["path"], f"{item_label}.path"),
                _hash(item["sha256"], f"{item_label}.sha256"),
            )
        )
    if len({item[0] for item in parsed}) != len(parsed):
        raise LicenseInputError(f"{label} contains duplicate paths")
    return tuple(parsed)


def _load_runtime_input(
    path: Path, expected_lock_sha: str
) -> tuple[
    str,
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    document = _load_json(path, "runtime licence input")
    if not isinstance(document, dict):
        raise LicenseInputError("runtime licence input must be a JSON object")
    required_keys = {"schema_id", "lock_sha256", "packages", "components"}
    unknown_keys = set(document) - required_keys - {"technical_components"}
    missing_keys = required_keys - set(document)
    if missing_keys or unknown_keys:
        details: list[str] = []
        if missing_keys:
            details.append("missing: " + ", ".join(sorted(missing_keys)))
        if unknown_keys:
            details.append("unknown: " + ", ".join(sorted(unknown_keys)))
        raise LicenseInputError("runtime licence input fields are not exact (" + "; ".join(details) + ")")
    if document["schema_id"] != RUNTIME_INPUT_SCHEMA_ID:
        raise LicenseInputError(f"runtime licence input schema_id must be {RUNTIME_INPUT_SCHEMA_ID}")
    declared_lock_sha = _hash(document["lock_sha256"], "runtime input lock_sha256")
    expected = _hash(expected_lock_sha, "expected conda lock SHA256")
    if declared_lock_sha != expected:
        raise LicenseInputError(
            f"runtime licence input lock_sha256 does not match expected lock SHA256: {declared_lock_sha}"
        )
    raw_packages = document["packages"]
    if not isinstance(raw_packages, list) or not raw_packages:
        raise LicenseInputError("runtime licence input packages must be a non-empty array")
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_packages):
        label = f"runtime packages[{index}]"
        if not isinstance(raw, dict):
            raise LicenseInputError(f"{label} must be an object")
        source, identity, display = _canonical_runtime_identity(raw["package_url"], f"{label}.package_url")
        allowed_keys = {"package_url", "license_files", "notice_files"}
        if source == "pip":
            allowed_keys.add("metadata_sha256")
        _require_exact_keys(raw, allowed_keys, label)
        if identity in result:
            raise LicenseInputError(f"duplicate runtime package declaration: {display}")

        metadata_sha256 = None
        if source == "pip":
            # A wheel is not represented by the conda explicit lock.  Bind the installed
            # distribution metadata explicitly so a same-name replacement cannot be silently
            # treated as the reviewed package.  The packed runtime hash then binds the rest of
            # the wheel payload.
            if "metadata_sha256" not in raw:
                raise LicenseInputError(f"{label} pip package requires metadata_sha256")
            metadata_sha256 = _hash(raw["metadata_sha256"], f"{label}.metadata_sha256")
        elif "metadata_sha256" in raw:
            raise LicenseInputError(f"{label} conda package must not carry metadata_sha256")
        result[identity] = {
            "source": source,
            "display": display,
            "metadata_sha256": metadata_sha256,
            "license": _parse_evidence_declarations(raw["license_files"], f"{label}.license_files"),
            "notice": _parse_evidence_declarations(
                raw["notice_files"], f"{label}.notice_files", allow_empty=True
            ),
        }
    raw_components = document["components"]
    if not isinstance(raw_components, list):
        raise LicenseInputError("runtime licence input components must be an array")
    components: dict[str, dict[str, Any]] = {}
    safe_component_id = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    for index, raw in enumerate(raw_components):
        label = f"runtime components[{index}]"
        if not isinstance(raw, dict):
            raise LicenseInputError(f"{label} must be an object")
        _require_exact_keys(
            raw,
            {
                "component_id",
                "version",
                "source",
                "license",
                "payload_path",
                "payload_sha256",
                "license_files",
                "notice_files",
            },
            label,
        )
        component_id = _text(raw["component_id"], f"{label}.component_id")
        if safe_component_id.fullmatch(component_id) is None:
            raise LicenseInputError(f"{label}.component_id is unsafe")
        if component_id in components:
            raise LicenseInputError(f"duplicate runtime component declaration: {component_id}")
        license_id = _text(raw["license"], f"{label}.license")
        if license_id.casefold() in _UNKNOWN_LICENSES:
            raise LicenseInputError(
                f"{label}.license has no usable identifier; refusing to infer terms"
            )
        payload_path = _text(raw["payload_path"], f"{label}.payload_path")
        payload = Path(payload_path)
        if payload.is_absolute() or any(part in {"", ".", ".."} for part in payload.parts):
            raise LicenseInputError(f"{label}.payload_path must be a relative prefix path without traversal")
        components[component_id] = {
            "component_id": component_id,
            "version": _text(raw["version"], f"{label}.version"),
            "source": _text(raw["source"], f"{label}.source"),
            "license": license_id,
            "payload_path": payload_path,
            "payload_sha256": _hash(raw["payload_sha256"], f"{label}.payload_sha256"),
            "license_files": _parse_evidence_declarations(
                raw["license_files"], f"{label}.license_files"
            ),
            "notice_files": _parse_evidence_declarations(
                raw["notice_files"], f"{label}.notice_files", allow_empty=True
            ),
        }
    raw_technical_components = document.get("technical_components", [])
    if not isinstance(raw_technical_components, list):
        raise LicenseInputError("runtime technical_components must be an array")
    technical_components: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_technical_components):
        label = f"runtime technical_components[{index}]"
        if not isinstance(raw, dict):
            raise LicenseInputError(f"{label} must be an object")
        _require_exact_keys(
            raw,
            {"component_id", "source", "source_sha256", "payload_path", "payload_sha256"},
            label,
        )
        component_id = _text(raw["component_id"], f"{label}.component_id")
        if safe_component_id.fullmatch(component_id) is None:
            raise LicenseInputError(f"{label}.component_id is unsafe")
        if component_id in technical_components or component_id in components:
            raise LicenseInputError(f"duplicate runtime component declaration: {component_id}")
        source = _text(raw["source"], f"{label}.source")
        source_path = Path(source)
        if (
            _URL_SCHEME.match(source)
            or source_path.is_absolute()
            or any(part in {"", ".", ".."} for part in source_path.parts)
        ):
            raise LicenseInputError(f"{label}.source must be a relative local path without traversal")
        payload_path = _text(raw["payload_path"], f"{label}.payload_path")
        payload = Path(payload_path)
        if payload.is_absolute() or any(part in {"", ".", ".."} for part in payload.parts):
            raise LicenseInputError(f"{label}.payload_path must be a relative prefix path without traversal")
        technical_components[component_id] = {
            "component_id": component_id,
            "source": source,
            "source_sha256": _hash(raw["source_sha256"], f"{label}.source_sha256"),
            "payload_path": payload_path,
            "payload_sha256": _hash(raw["payload_sha256"], f"{label}.payload_sha256"),
        }
    return declared_lock_sha, result, components, technical_components


def _runtime_evidence(
    declarations: tuple[tuple[str, str], ...], base: Path, label: str, *, require_mode: bool = True
) -> tuple[EvidenceFile, ...]:
    # ``require_mode=False`` is used only when re-reading upstream conda package-cache license
    # evidence (info/licenses/*), whose mode conda-forge controls; the regular-file/symlink guards
    # and the content SHA256 still bind it. Supplement evidence we author is read with the default
    # (its 0644 mode is enforced at generate time), and component/pip/carried reads stay strict.
    values: list[EvidenceFile] = []
    for index, (raw_path, expected_sha) in enumerate(declarations):
        path = _resolve_local(base, raw_path, f"{label}[{index}].path")
        values.append(_read_evidence(path, expected_sha, f"{label}[{index}].evidence", require_mode=require_mode))
    return tuple(values)


def _resolve_prefix_path(prefix: Path, value: Any, label: str) -> Path:
    """Resolve a component path under the validated runtime prefix without following links."""

    text = _text(value, label)
    relative = Path(text)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise LicenseInputError(f"{label} must be a relative prefix path without traversal")
    current = prefix
    for part in relative.parts:
        current = current / part
        try:
            info = current.lstat()
        except OSError as exc:
            raise LicenseInputError(f"{label} is missing: {current}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise LicenseInputError(f"{label} must not traverse a symlink: {current}")
    return current


def component_payload_sha256(path: Path | str) -> str:
    """Hash one non-symlink component file/tree using stable paths, modes and bytes.

    A component is deliberately not treated as a conda package.  The digest binds the exact
    non-conda payload copied into the runtime prefix (for example the official ORT archive
    directory or the native bridge), while the component's separate evidence declarations bind
    the license and notice text.
    """

    root = Path(path)
    try:
        root_info = root.lstat()
    except OSError as exc:
        raise LicenseInputError(f"runtime component payload is missing: {root}") from exc
    if stat.S_ISLNK(root_info.st_mode):
        raise LicenseInputError(f"runtime component payload must not be a symlink: {root}")
    if not stat.S_ISREG(root_info.st_mode) and not stat.S_ISDIR(root_info.st_mode):
        raise LicenseInputError(f"runtime component payload must be a regular file or directory: {root}")
    entries: list[tuple[str, Path, int, int]] = []
    if stat.S_ISREG(root_info.st_mode):
        entries.append((root.name, root, stat.S_IMODE(root_info.st_mode), root_info.st_size))
    else:
        for item in sorted(root.rglob("*"), key=lambda candidate: candidate.relative_to(root).as_posix()):
            info = item.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise LicenseInputError(f"runtime component payload must not contain a symlink: {item}")
            if stat.S_ISDIR(info.st_mode):
                continue
            if not stat.S_ISREG(info.st_mode):
                raise LicenseInputError(f"runtime component payload contains a non-regular file: {item}")
            entries.append(
                (item.relative_to(root).as_posix(), item, stat.S_IMODE(info.st_mode), info.st_size)
            )
    digest = hashlib.sha256()
    digest.update(b"whitewater-p25-runtime-component-v1\0")
    digest.update((b"file\0" if stat.S_ISREG(root_info.st_mode) else b"directory\0"))
    for relative, item, mode, size_bytes in entries:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(f"{mode:04o}:{size_bytes}\0".encode("ascii"))
        try:
            with item.open("rb") as stream:
                actual_size = 0
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                    actual_size += len(chunk)
        except OSError as exc:
            raise LicenseInputError(f"could not read runtime component payload: {item}: {exc}") from exc
        if actual_size != size_bytes:
            raise LicenseInputError(
                f"runtime component payload changed while hashing: {item}"
            )
    return digest.hexdigest()


def _canonical_runtime_bytes(content: bytes, prefixes: Sequence[Path | str]) -> bytes:
    """Normalize known conda prefixes in a runtime file before hashing it.

    ``conda-unpack`` is allowed to rewrite an embedded build prefix during relocation.  The
    final runtime prefix is the only explicit substitution input; no arbitrary path substitution
    is performed.  Sorting longest-first prevents a shorter prefix from consuming the beginning
    of a longer nested path.
    """

    normalized = content
    encoded_prefixes = {
        os.fsencode(os.path.abspath(os.fspath(prefix)).rstrip(os.sep))
        for prefix in prefixes
        if os.fspath(prefix)
    }
    for encoded in sorted(encoded_prefixes, key=len, reverse=True):
        if encoded:
            normalized = normalized.replace(encoded, _RUNTIME_PREFIX_TOKEN)
    return normalized


def runtime_content_identity(prefix: Path | str) -> dict[str, Any]:
    """Return a relocation-stable identity for the installed runtime payload.

    The conda package cache is outside the prefix and ``conda-pack`` may omit ``conda-meta``;
    generated ``__pycache__`` directories and ``*.pyc`` files are also excluded because Python
    may create them while the runtime is smoke-tested.  Every other regular file and symlink is
    represented by relative path, mode, canonical byte hash and canonical byte length.
    Consequently an added, removed, replaced or mode-changed payload entry fails the post-pack
    check, while the destination prefix rewrite remains admissible.
    """

    root = Path(prefix)
    _require_directory(root, "runtime content prefix")
    prefixes = [root]
    entries: list[dict[str, Any]] = []

    def is_excluded(relative: Path) -> bool:
        if not relative.parts:
            return False
        if relative.parts[0] in _RUNTIME_CONTENT_EXCLUDED_ROOTS:
            return True
        if any(part in _RUNTIME_CONTENT_EXCLUDED_DIRECTORIES for part in relative.parts):
            return True
        return relative.name.endswith(_RUNTIME_CONTENT_EXCLUDED_SUFFIXES)

    for current, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        current_relative = current_path.relative_to(root)
        if is_excluded(current_relative):
            directory_names[:] = []
            file_names[:] = []
            continue
        directory_names.sort()
        file_names.sort()
        # os.walk places symlinked directories in directory_names.  Record them as symlink
        # entries and remove them so the walk cannot ever follow a link outside the prefix.
        symlink_directories: list[str] = []
        for name in list(directory_names):
            item = current_path / name
            info = item.lstat()
            if stat.S_ISLNK(info.st_mode):
                directory_names.remove(name)
                symlink_directories.append(name)
        for name in [*symlink_directories, *file_names]:
            item = current_path / name
            relative = item.relative_to(root)
            if is_excluded(relative):
                continue
            info = item.lstat()
            mode = stat.S_IMODE(info.st_mode)
            if stat.S_ISLNK(info.st_mode):
                value = _canonical_runtime_bytes(os.fsencode(os.readlink(item)), prefixes)
                kind = "symlink"
            elif stat.S_ISREG(info.st_mode):
                try:
                    value = _canonical_runtime_bytes(item.read_bytes(), prefixes)
                except OSError as exc:
                    raise LicenseInputError(f"could not read runtime content: {item}") from exc
                kind = "file"
            else:
                raise LicenseInputError(
                    f"runtime content contains unsupported non-regular entry: {item}"
                )
            entries.append(
                {
                    "path": relative.as_posix(),
                    "kind": kind,
                    "mode": f"{mode:04o}",
                    "sha256": hashlib.sha256(value).hexdigest(),
                    "size_bytes": len(value),
                }
            )
    entries.sort(key=lambda item: item["path"])
    digest = canonical_sha256(
        {
            "schema_id": RUNTIME_CONTENT_SCHEMA_ID,
            "excluded_roots": list(_RUNTIME_CONTENT_EXCLUDED_ROOTS),
            "excluded_directories": list(_RUNTIME_CONTENT_EXCLUDED_DIRECTORIES),
            "excluded_suffixes": list(_RUNTIME_CONTENT_EXCLUDED_SUFFIXES),
            "entries": entries,
        }
    )
    return {
        "schema_id": RUNTIME_CONTENT_SCHEMA_ID,
        "excluded_roots": list(_RUNTIME_CONTENT_EXCLUDED_ROOTS),
        "excluded_directories": list(_RUNTIME_CONTENT_EXCLUDED_DIRECTORIES),
        "excluded_suffixes": list(_RUNTIME_CONTENT_EXCLUDED_SUFFIXES),
        "file_count": len(entries),
        "sha256": digest,
    }


def _runtime_component_evidence(
    declarations: tuple[tuple[str, str], ...], prefix: Path, label: str
) -> tuple[EvidenceFile, ...]:
    values: list[EvidenceFile] = []
    for index, (raw_path, expected_sha) in enumerate(declarations):
        path = _resolve_prefix_path(prefix, raw_path, f"{label}[{index}].path")
        values.append(_read_evidence(path, expected_sha, f"{label}[{index}].evidence"))
    return tuple(values)


def _package_cache_archive_stem(url: str) -> str:
    name = Path(urlsplit(url).path).name
    if name.endswith(".tar.bz2"):
        return name[: -len(".tar.bz2")]
    if name.endswith(".conda"):
        return name[: -len(".conda")]
    raise LicenseInputError(f"conda package URL has an unsupported archive suffix: {url}")


def _find_cached_package(package_cache: Path, raw: Mapping[str, Any], label: str) -> Path:
    stem = _package_cache_archive_stem(raw["url"])
    candidates = sorted(
        (item for item in package_cache.glob(stem) if not item.is_symlink()), key=lambda item: str(item)
    )
    if len(candidates) != 1:
        raise LicenseInputError(
            f"{label} must have exactly one extracted package-cache directory named {stem!r} "
            f"(found {len(candidates)})"
        )
    package_dir = candidates[0]
    _require_directory(package_dir, f"{label} package-cache directory")
    index_path = package_dir / "info" / "index.json"
    index = _load_json(index_path, f"{label} package-cache info/index.json")
    if not isinstance(index, dict):
        raise LicenseInputError(f"{label} package-cache info/index.json must be an object")
    for field in ("name", "version", "build", "license"):
        _text(index.get(field), f"{label} package-cache index.{field}")
    for field in ("name", "version", "build", "license"):
        if field == "license":
            matches = index[field].strip() == raw[field].strip()
        else:
            matches = index[field] == raw[field]
        if not matches:
            raise LicenseInputError(
                f"{label} package-cache metadata {field} does not match conda-meta"
            )
    return package_dir


def _cached_license_files(package_dir: Path, label: str) -> tuple[dict[str, str], ...]:
    licenses_dir = package_dir / "info" / "licenses"
    _require_directory(licenses_dir, f"{label} package-cache license directory")
    files: list[dict[str, str]] = []
    for path in sorted(licenses_dir.rglob("*"), key=lambda item: item.relative_to(licenses_dir).as_posix()):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise LicenseInputError(f"{label} package-cache license tree contains a symlink: {path}")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise LicenseInputError(f"{label} package-cache license tree contains a non-regular file: {path}")
        # Upstream conda-forge license files (e.g. intel-gmmlib's LICENSE.md at 0755) may not be
        # 0644; their mode is not our integrity property. Bind them by content + SHA256 and keep
        # the symlink/regular-file guards, but do not require a specific mode. Only this
        # package-cache read path is mode-agnostic.
        evidence = _read_evidence(
            path,
            hashlib.sha256(path.read_bytes()).hexdigest(),
            f"{label} license evidence",
            require_mode=False,
        )
        files.append({"path": str(path), "sha256": evidence.sha256})
    if not files:
        raise LicenseInputError(f"{label} package-cache license directory contains no files")
    return tuple(files)


def _runtime_manifest_components(prefix: Path, path: Path) -> list[dict[str, Any]]:
    """Build the explicit ORT component declaration from the checked-in runtime manifest.

    The manifest supplies identity/source and the expected payload location; the installed
    prefix supplies the actual payload and official text files.  We require exactly one ORT
    license and one ThirdPartyNotices file rather than guessing from package metadata.
    """

    document = _load_json(path, "runtime input manifest")
    if not isinstance(document, dict):
        raise LicenseInputError("runtime input manifest must be a JSON object")
    ort = document.get("onnxruntime_cuda12")
    if not isinstance(ort, dict):
        raise LicenseInputError("runtime input manifest must contain onnxruntime_cuda12")
    for field in ("version", "archive_url", "archive_sha256", "license", "payload_root"):
        _text(ort.get(field), f"runtime input manifest onnxruntime_cuda12.{field}")
    _hash(ort["archive_sha256"], "runtime input manifest ONNX Runtime archive_sha256")
    payload_path = _resolve_prefix_path(prefix, ort["payload_root"], "ONNX Runtime payload_root")
    if not payload_path.is_dir():
        raise LicenseInputError("ONNX Runtime payload_root must be a directory")
    license_candidates = []
    notice_candidates = []
    for item in sorted(payload_path.rglob("*"), key=lambda candidate: str(candidate)):
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise LicenseInputError(f"ONNX Runtime component contains a symlink: {item}")
        if not stat.S_ISREG(info.st_mode):
            continue
        lowered = item.name.casefold()
        if lowered in {"license", "license.txt"}:
            license_candidates.append(item)
        if lowered in {"thirdpartynotices.txt", "third-party-notices.txt"}:
            notice_candidates.append(item)
    if len(license_candidates) != 1:
        raise LicenseInputError(
            f"ONNX Runtime component must contain exactly one LICENSE file (found {len(license_candidates)})"
        )
    if len(notice_candidates) != 1:
        raise LicenseInputError(
            "ONNX Runtime component must contain exactly one ThirdPartyNotices file "
            f"(found {len(notice_candidates)})"
        )
    def prefix_relative(item: Path) -> str:
        return item.relative_to(prefix).as_posix()
    return [
        {
            "component_id": "onnxruntime-cuda12",
            "version": ort["version"],
            "source": ort["archive_url"],
            "license": ort["license"],
            "payload_path": ort["payload_root"],
            "payload_sha256": component_payload_sha256(payload_path),
            "license_files": [
                {
                    "path": prefix_relative(license_candidates[0]),
                    "sha256": hashlib.sha256(license_candidates[0].read_bytes()).hexdigest(),
                }
            ],
            "notice_files": [
                {
                    "path": prefix_relative(notice_candidates[0]),
                    "sha256": hashlib.sha256(notice_candidates[0].read_bytes()).hexdigest(),
                }
            ],
        }
    ]


def _runtime_manifest_technical_components(prefix: Path, path: Path) -> list[dict[str, Any]]:
    """Build first-party technical component identities from the checked-in runtime manifest.

    These records are deliberately separate from ``components``: they carry no licence or
    notice assertion.  The native bridge is White Water's build output, so its reviewed identity
    is the checked-in source path/hash plus the exact payload path/hash in the installed prefix.
    The workflow verifies the source hash before compiling; this function binds those same
    values into the human-reviewed runtime inventory and verifies the resulting payload.
    """

    document = _load_json(path, "runtime input manifest")
    if not isinstance(document, dict):
        raise LicenseInputError("runtime input manifest must be a JSON object")
    raw_bridge = document.get("native_bridge")
    if raw_bridge is None:
        return []
    if not isinstance(raw_bridge, dict):
        raise LicenseInputError("runtime input manifest native_bridge must be an object")
    _require_exact_keys(
        raw_bridge,
        {"source", "source_sha256", "payload"},
        "runtime input manifest native_bridge",
    )
    source = _text(raw_bridge["source"], "runtime input manifest native_bridge.source")
    source_path = Path(source)
    if (
        _URL_SCHEME.match(source)
        or source_path.is_absolute()
        or any(part in {"", ".", ".."} for part in source_path.parts)
    ):
        raise LicenseInputError(
            "runtime input manifest native_bridge.source must be a relative local path without traversal"
        )
    source_sha = _hash(
        raw_bridge["source_sha256"], "runtime input manifest native_bridge.source_sha256"
    )
    payload_path = _text(raw_bridge["payload"], "runtime input manifest native_bridge.payload")
    payload = _resolve_prefix_path(prefix, payload_path, "native bridge payload")
    return [
        {
            "component_id": "whitewater-native-ort-bridge",
            "source": source,
            "source_sha256": source_sha,
            "payload_path": payload_path,
            "payload_sha256": component_payload_sha256(payload),
        }
    ]


def generate_runtime_input(
    prefix: Path | str,
    lock_path: Path | str,
    package_cache: Path | str,
    output_path: Path | str,
    expected_lock_sha256: str,
    *,
    components_path: Path | str | None = None,
    components_manifest_path: Path | str | None = None,
    supplement_path: Path | str | None = None,
) -> dict[str, Any]:
    """Generate a runtime declaration from the exact conda cache and optional components.

    Conda package metadata is taken from the extracted cache's ``info/index.json`` and license
    files, not from a package-name allowlist or a guessed URL.  A package without an actual
    ``info/licenses`` tree fails closed; ``notice_files`` is intentionally an explicit empty
    list because this generator never fabricates a notice.  Non-conda component declarations are
    supplied separately and are validated by :func:`collect_runtime` against the installed prefix.
    """

    prefix_path = Path(prefix)
    _require_directory(prefix_path, "conda environment prefix")
    cache_path = Path(package_cache)
    _require_directory(cache_path, "conda package cache")
    lock_sha, lock_urls = parse_explicit_lock(lock_path, expected_lock_sha256)
    metadata = _metadata_packages(prefix_path, lock_urls)
    supplements: dict[str, dict[str, Any]] = {}
    if supplement_path is not None:
        supplement_file = Path(supplement_path)
        supplement_document = _load_json(supplement_file, "runtime license supplement")
        if not isinstance(supplement_document, dict):
            raise LicenseInputError("runtime license supplement must be a JSON object")
        _require_exact_keys(
            supplement_document,
            {"schema_id", "packages"},
            "runtime license supplement",
        )
        if supplement_document["schema_id"] != RUNTIME_SUPPLEMENT_INPUT_SCHEMA_ID:
            raise LicenseInputError(
                f"runtime license supplement schema_id must be {RUNTIME_SUPPLEMENT_INPUT_SCHEMA_ID}"
            )
        raw_supplements = supplement_document["packages"]
        if not isinstance(raw_supplements, list):
            raise LicenseInputError("runtime license supplement packages must be an array")
        for index, raw in enumerate(raw_supplements):
            label = f"runtime license supplement packages[{index}]"
            if not isinstance(raw, dict):
                raise LicenseInputError(f"{label} must be an object")
            _require_exact_keys(raw, {"package_url", "license_files", "notice_files"}, label)
            source, identity, display = _canonical_runtime_identity(raw["package_url"], f"{label}.package_url")
            if source not in ("conda", "pip"):
                raise LicenseInputError(
                    f"{label}.package_url must be a conda package URL or pip://name==version"
                )
            if identity in supplements:
                raise LicenseInputError(f"duplicate runtime license supplement package: {display}")
            supplements[identity] = {
                "license": _parse_evidence_declarations(
                    raw["license_files"], f"{label}.license_files"
                ),
                "notice": _parse_evidence_declarations(
                    raw["notice_files"], f"{label}.notice_files", allow_empty=True
                ),
                "base": supplement_file.parent,
            }
    packages: list[dict[str, Any]] = []
    used_supplements: set[str] = set()
    for raw, _metadata_path, _identity, package_url in metadata:
        identity = f"conda:{_canonical_url(raw['url'], 'conda metadata URL')[0]}"
        package_dir = _find_cached_package(cache_path, raw, f"conda {raw['name']}")
        try:
            cached_license_files = _cached_license_files(package_dir, f"conda {raw['name']}")
        except LicenseInputError as exc:
            if "package-cache license directory" not in str(exc):
                raise
            supplement = supplements.get(identity)
            if supplement is None:
                raise
            used_supplements.add(identity)
            validated_evidence: dict[str, tuple[EvidenceFile, ...]] = {}
            for evidence_kind in ("license", "notice"):
                declarations = supplement[evidence_kind]
                if evidence_kind == "license" and not declarations:
                    raise LicenseInputError(
                        f"runtime license supplement for {raw['name']} has no license evidence"
                    )
                validated_evidence[evidence_kind] = _runtime_evidence(
                    declarations,
                    supplement["base"],
                    f"runtime license supplement {raw['name']} {evidence_kind}",
                )
            packages.append(
                {
                    "package_url": package_url,
                    "license_files": [
                        {"path": str(item.path), "sha256": item.sha256}
                        for item in validated_evidence["license"]
                    ],
                    "notice_files": [
                        {"path": str(item.path), "sha256": item.sha256}
                        for item in validated_evidence["notice"]
                    ],
                }
            )
            continue
        packages.append(
            {
                "package_url": package_url,
                "license_files": list(cached_license_files),
                "notice_files": [],
            }
        )
    # Harvest each installed pip distribution's own bundled licence/notice evidence.  conda-pack
    # ships the full export environment, so every wheel (torch, numpy, onnxruntime, ...) is part of
    # the runtime and must appear in the inventory with its own declared terms.  Evidence is copied
    # verbatim from the wheel's dist-info into a staging tree beside the generated input so that
    # collect_runtime reads exactly these bytes; a wheel bundling no licence file falls back to a
    # pip:// supplement entry and otherwise fails closed by name.
    conda_shims = [
        SimpleNamespace(name=raw["name"], version=raw["version"]) for raw, _p, _i, _u in metadata
    ]
    pip_records = _pip_metadata_packages(prefix_path, conda_shims)
    pip_evidence_root = Path(output_path).parent / "pip-license-evidence"
    input_base = Path(os.path.abspath(Path(output_path).parent))
    missing_pip_evidence: list[str] = []
    for record in pip_records:
        dist_info = record.metadata_path.parent
        message = Parser().parsestr(record.metadata_path.read_bytes().decode("utf-8"))
        license_srcs, notice_srcs = _harvest_pip_license_files(dist_info, message)
        supplement = supplements.get(record.canonical_url)
        if not license_srcs and supplement is None:
            # Report every wheel that bundles no licence file in one pass so the whole set can be
            # added to the pip supplement at once rather than one CI cycle per package.
            missing_pip_evidence.append(f"pip://{_pip_name(record.name)}=={record.version}")
            continue
        if not license_srcs and supplement is not None:
            used_supplements.add(record.canonical_url)
            lic = _runtime_evidence(
                supplement["license"], supplement["base"], f"pip supplement {record.name} license"
            )
            noti = _runtime_evidence(
                supplement["notice"], supplement["base"], f"pip supplement {record.name} notice"
            )
            license_files = [{"path": str(item.path), "sha256": item.sha256} for item in lic]
            notice_files = [{"path": str(item.path), "sha256": item.sha256} for item in noti]
        elif license_srcs:
            safe = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{_pip_name(record.name)}-{record.version}")
            dest_dir = pip_evidence_root / safe
            dest_dir.mkdir(parents=True, exist_ok=True)
            license_files = []
            notice_files = []
            for kind, srcs, target in (
                ("license", license_srcs, license_files),
                ("notice", notice_srcs, notice_files),
            ):
                for src in srcs:
                    content = src.read_bytes()
                    dest = dest_dir / f"{kind}-{src.name}"
                    collision = 1
                    while dest.exists():
                        dest = dest_dir / f"{kind}-{collision}-{src.name}"
                        collision += 1
                    dest.write_bytes(content)
                    dest.chmod(0o644)
                    target.append(
                        {
                            "path": Path(os.path.abspath(dest)).relative_to(input_base).as_posix(),
                            "sha256": hashlib.sha256(content).hexdigest(),
                        }
                    )
        packages.append(
            {
                "package_url": record.package_url,
                "metadata_sha256": record.metadata_sha256,
                "license_files": license_files,
                "notice_files": notice_files,
            }
        )
    if missing_pip_evidence:
        raise LicenseInputError(
            "pip runtime distributions bundle no licence file; declare each in the runtime licence "
            "supplement as pip://name==version: " + ", ".join(sorted(missing_pip_evidence))
        )
    unused_supplements = sorted(set(supplements) - used_supplements)
    if unused_supplements:
        raise LicenseInputError(
            "runtime license supplement contains unused packages (cache or wheel already carries "
            "licence evidence): " + ", ".join(unused_supplements)
        )
    components: list[Any] = []
    technical_components: list[Any] = []
    if components_path is not None:
        component_file = Path(components_path)
        document = _load_json(component_file, "runtime component input")
        if not isinstance(document, dict):
            raise LicenseInputError("runtime component input must be a JSON object")
        _require_exact_keys(document, {"schema_id", "components"}, "runtime component input")
        if document["schema_id"] != RUNTIME_COMPONENT_INPUT_SCHEMA_ID:
            raise LicenseInputError(
                f"runtime component input schema_id must be {RUNTIME_COMPONENT_INPUT_SCHEMA_ID}"
            )
        if not isinstance(document["components"], list):
            raise LicenseInputError("runtime component input components must be an array")
        components = document["components"]
    if components_manifest_path is not None:
        if components_path is not None:
            raise LicenseInputError("runtime component input and runtime manifest are mutually exclusive")
        manifest_path = Path(components_manifest_path)
        components = _runtime_manifest_components(prefix_path, manifest_path)
        technical_components = _runtime_manifest_technical_components(prefix_path, manifest_path)
    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise LicenseInputError(f"generated runtime input already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_id": RUNTIME_INPUT_SCHEMA_ID,
        "lock_sha256": lock_sha,
        "packages": packages,
        "components": components,
        "technical_components": technical_components,
    }
    content = (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    output_sha, output_size = _write_new(output, content, "generated runtime licence input")
    # Parse once through the exact input boundary, including component schema/identity checks.
    _load_runtime_input(output, lock_sha)
    return {
        "lock_sha256": lock_sha,
        "package_count": len(packages),
        "component_count": len(components),
        "technical_component_count": len(technical_components),
        "input_sha256": output_sha,
        "input_size_bytes": output_size,
    }


def validate_runtime_review(
    path: Path | str, expected_sha256: str, inventory_sha256: str
) -> dict[str, Any]:
    """Validate Andrew Miller's hash-bound approval of one generated runtime inventory."""

    review_path = Path(path)
    expected_file_sha = _hash(expected_sha256, "runtime legal-review SHA256")
    expected_inventory_sha = _hash(inventory_sha256, "runtime inventory SHA256")
    _require_regular(review_path, "runtime legal-review file")
    actual_file_sha = hashlib.sha256(review_path.read_bytes()).hexdigest()
    if actual_file_sha != expected_file_sha:
        raise LicenseInputError(
            f"runtime legal-review SHA256 mismatch: expected {expected_file_sha}, got {actual_file_sha}"
        )
    document = _load_json(review_path, "runtime legal-review file")
    if not isinstance(document, dict):
        raise LicenseInputError("runtime legal-review file must be a JSON object")
    _require_exact_keys(
        document,
        {"schema_id", "reviewer", "reviewed", "reviewed_at", "inventory_sha256", "statement"},
        "runtime legal-review file",
    )
    if document["schema_id"] != RUNTIME_REVIEW_SCHEMA_ID:
        raise LicenseInputError(f"runtime legal-review schema_id must be {RUNTIME_REVIEW_SCHEMA_ID}")
    if document["reviewer"] != "Andrew Miller":
        raise LicenseInputError("runtime legal-review reviewer must be Andrew Miller")
    if document["reviewed"] is not True:
        raise LicenseInputError("runtime legal-review reviewed must be true")
    _validate_reviewed_at(document["reviewed_at"])
    declared_inventory_sha = _hash(
        document["inventory_sha256"], "runtime legal-review inventory_sha256"
    )
    if declared_inventory_sha != expected_inventory_sha:
        raise LicenseInputError(
            "runtime legal-review inventory_sha256 does not match generated runtime inventory: "
            f"expected {expected_inventory_sha}, got {declared_inventory_sha}"
        )
    _text(document["statement"], "runtime legal-review statement")
    return {
        "schema_id": RUNTIME_REVIEW_SCHEMA_ID,
        "reviewer": "Andrew Miller",
        "inventory_sha256": expected_inventory_sha,
        "review_sha256": expected_file_sha,
    }


def collect_runtime(
    prefix: Path | str,
    lock_path: Path | str,
    input_path: Path | str,
    output_dir: Path | str,
    expected_lock_sha256: str,
    *,
    license_name: str = "LICENSES.txt",
    notice_name: str = "NOTICES.txt",
) -> dict[str, Any]:
    """Validate one installed runtime prefix and write its deterministic notice bundle."""

    # Keep the operator-supplied prefix lexical so _require_directory can reject a symlinked
    # environment root instead of silently following it.
    prefix_path = Path(prefix)
    lock_file = Path(lock_path)
    input_file = Path(input_path)
    lock_sha, lock_urls = parse_explicit_lock(lock_file, expected_lock_sha256)
    declared_sha, declarations, component_declarations, technical_declarations = _load_runtime_input(
        input_file, lock_sha
    )
    if declared_sha != lock_sha:
        raise LicenseInputError("runtime input lock SHA256 does not match the lock bytes")
    records: list[RuntimePackage] = []
    for raw, metadata_path, identity, package_url in _metadata_packages(prefix_path, lock_urls):
        declared = declarations.get(identity)
        if declared is None:
            raise LicenseInputError(f"runtime package lacks explicit licence/notice evidence: {identity}")
        records.append(
            RuntimePackage(
                source="conda",
                package_url=package_url,
                canonical_url=identity,
                name=raw["name"],
                version=raw["version"],
                build=raw["build"],
                license_id=raw["license"].strip(),
                license_family=(raw.get("license_family") if isinstance(raw.get("license_family"), str) else None),
                metadata_path=metadata_path,
                metadata_sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                # Conda-package licence/notice evidence is upstream third-party text (a
                # package-cache info/licenses/* file, or a supplement file we authored). Its mode is
                # not our integrity property -- the content SHA256 is, and it is re-verified here --
                # so this read is mode-agnostic. Supplement files still had their 0644 mode enforced
                # at generate time.
                license=_runtime_evidence(
                    declared["license"], input_file.parent, f"{raw['name']} licence", require_mode=False
                ),
                notice=_runtime_evidence(
                    declared["notice"], input_file.parent, f"{raw['name']} notice", require_mode=False
                ),
            )
        )
    records.sort(key=lambda item: item.canonical_url)
    components: list[RuntimeComponent] = []
    for component_id in sorted(component_declarations):
        declared = component_declarations[component_id]
        payload_path = _resolve_prefix_path(
            prefix_path, declared["payload_path"], f"runtime component {component_id}.payload_path"
        )
        actual_payload_sha = component_payload_sha256(payload_path)
        if actual_payload_sha != declared["payload_sha256"]:
            raise LicenseInputError(
                f"runtime component payload SHA256 mismatch for {component_id}: "
                f"expected {declared['payload_sha256']}, got {actual_payload_sha}"
            )
        components.append(
            RuntimeComponent(
                component_id=component_id,
                version=declared["version"],
                source=declared["source"],
                license_id=declared["license"],
                payload_path=declared["payload_path"],
                payload_sha256=actual_payload_sha,
                license=_runtime_component_evidence(
                    declared["license_files"], prefix_path, f"runtime component {component_id} licence"
                ),
                notice=_runtime_component_evidence(
                    declared["notice_files"], prefix_path, f"runtime component {component_id} notice"
                ),
            )
        )
    technical_components: list[dict[str, str]] = []
    for component_id in sorted(technical_declarations):
        declared = technical_declarations[component_id]
        payload_path = _resolve_prefix_path(
            prefix_path, declared["payload_path"], f"runtime technical component {component_id}.payload_path"
        )
        actual_payload_sha = component_payload_sha256(payload_path)
        if actual_payload_sha != declared["payload_sha256"]:
            raise LicenseInputError(
                f"runtime technical component payload SHA256 mismatch for {component_id}: "
                f"expected {declared['payload_sha256']}, got {actual_payload_sha}"
            )
        technical_components.append(
            {
                "component_id": component_id,
                "source": declared["source"],
                "source_sha256": declared["source_sha256"],
                "payload_path": declared["payload_path"],
                "payload_sha256": actual_payload_sha,
            }
        )
    # Any pip distribution not represented by a conda package is an independent runtime input.
    # This is where pip-installed onnxruntime-gpu (or any other wheel) is admitted; no package
    # name is special-cased and an undeclared distribution fails closed.
    pip_records = _pip_metadata_packages(prefix_path, records)
    for package in pip_records:
        declared = declarations.get(package.canonical_url)
        if declared is None:
            raise LicenseInputError(
                f"pip runtime package lacks explicit licence/notice evidence: {package.package_url}"
            )
        expected_metadata_sha = declared.get("metadata_sha256")
        if expected_metadata_sha != package.metadata_sha256:
            raise LicenseInputError(
                f"pip metadata SHA256 mismatch for {package.package_url}: "
                f"expected {expected_metadata_sha}, got {package.metadata_sha256}"
            )
        records.append(
            RuntimePackage(
                source="pip",
                package_url=package.package_url,
                canonical_url=package.canonical_url,
                name=package.name,
                version=package.version,
                build="pip",
                license_id=package.license_id,
                license_family=None,
                metadata_path=package.metadata_path,
                metadata_sha256=package.metadata_sha256,
                license=_runtime_evidence(declared["license"], input_file.parent, f"{package.name} licence"),
                notice=_runtime_evidence(declared["notice"], input_file.parent, f"{package.name} notice"),
            )
        )
    expected_identities = {f"conda:{canonical}" for canonical in lock_urls} | {
        package.canonical_url for package in pip_records
    }
    if set(declarations) != expected_identities:
        missing = sorted(expected_identities - set(declarations))
        extra = sorted(set(declarations) - expected_identities)
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if extra:
            details.append("extra: " + ", ".join(extra))
        raise LicenseInputError("runtime licence declarations do not exactly match installed packages (" + "; ".join(details) + ")")
    records.sort(key=lambda item: item.canonical_url)
    return _write_runtime_bundle(
        records,
        components,
        technical_components,
        lock_sha,
        prefix_path,
        output_dir,
        license_name=license_name,
        notice_name=notice_name,
    )


def _aggregate_candidate(input_value: CandidateInput, kind: str) -> bytes:
    groups: dict[str, dict[str, Any]] = {}
    for surface in input_value.surfaces:
        evidence = surface.license if kind == "license" else surface.notice
        item = groups.setdefault(
            evidence.sha256,
            {"content": evidence.content, "labels": [], "license_id": surface.license_id, "source": surface.source},
        )
        item["labels"].append(surface.surface)
    output = bytearray()
    output.extend(f"White Water P25-5 evaluation-only candidate {kind} evidence\n".encode("utf-8"))
    output.extend(b"Exact evidence bytes follow; this file makes no new legal conclusion.\n\n")
    for sha in sorted(groups):
        item = groups[sha]
        surfaces = ",".join(sorted(item["labels"]))
        output.extend(
            (
                f"===== candidate_id={input_value.candidate_id} surfaces={surfaces} "
                f"license={item['license_id']} source={item['source']} evidence_sha256={sha} =====\n"
            ).encode("utf-8")
        )
        output.extend(item["content"])
        if not item["content"].endswith(b"\n"):
            output.extend(b"\n")
        output.extend(b"\n")
    return bytes(output)


def _aggregate_runtime(
    records: Sequence[RuntimePackage], components: Sequence[RuntimeComponent], kind: str
) -> bytes:
    groups: dict[str, dict[str, Any]] = {}
    for package in records:
        evidence = package.license if kind == "license" else package.notice
        for item in evidence:
            group = groups.setdefault(
                item.sha256,
                {"content": item.content, "packages": [], "license_ids": set()},
            )
            group["packages"].append(package.package_url)
            group["license_ids"].add(package.license_id)
    for component in components:
        evidence = component.license if kind == "license" else component.notice
        for item in evidence:
            group = groups.setdefault(
                item.sha256,
                {"content": item.content, "packages": [], "license_ids": set()},
            )
            group["packages"].append(
                f"component://{component.component_id}@{component.version}"
            )
            group["license_ids"].add(component.license_id)
    output = bytearray()
    output.extend(f"White Water P25-5 evaluation-only runtime {kind} evidence\n".encode("utf-8"))
    output.extend(b"Exact local package evidence bytes follow; this file makes no new legal conclusion.\n\n")
    for sha in sorted(groups):
        item = groups[sha]
        packages = ",".join(sorted(set(item["packages"])))
        licenses = ",".join(sorted(item["license_ids"]))
        output.extend(
            (
                f"===== packages={packages} licenses={licenses} evidence_sha256={sha} =====\n"
            ).encode("utf-8")
        )
        output.extend(item["content"])
        if not item["content"].endswith(b"\n"):
            output.extend(b"\n")
        output.extend(b"\n")
    return bytes(output)


def _write_new(path: Path, content: bytes, label: str) -> tuple[str, int]:
    if path.exists() or path.is_symlink():
        raise LicenseInputError(f"{label} already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(content)
        path.chmod(EXPECTED_MODE)
    except OSError as exc:
        raise LicenseInputError(f"could not write {label}: {path}: {exc}") from exc
    _require_regular(path, label)
    return hashlib.sha256(content).hexdigest(), len(content)


def _prepare_output_dir(path: Path) -> None:
    if path.exists() or path.is_symlink():
        _require_directory(path, "licence output directory")
        if any(path.iterdir()):
            raise LicenseInputError(f"licence output directory must be empty: {path}")
    else:
        path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o755)


def _write_json(path: Path, value: Mapping[str, Any], label: str) -> tuple[str, int]:
    content = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    return _write_new(path, content, label)


def _output_name(value: str, label: str) -> str:
    name = _text(value, label)
    path = Path(name)
    if path.is_absolute() or path.name != name or name in {".", ".."}:
        raise LicenseInputError(f"{label} must be one relative filename")
    return name


def collect_candidate(
    input_path: Path | str,
    output_dir: Path | str,
    *,
    license_name: str = "LICENSES.txt",
    notice_name: str = "NOTICES.txt",
) -> dict[str, Any]:
    """Validate candidate evidence and write its deterministic notice bundle."""

    candidate = load_candidate_input(input_path)
    output = Path(output_dir)
    _prepare_output_dir(output)
    license_name = _output_name(license_name, "candidate license output name")
    notice_name = _output_name(notice_name, "candidate notice output name")
    if license_name == notice_name or "candidate-license-inventory.json" in {license_name, notice_name}:
        raise LicenseInputError("candidate license, notice and inventory output names must be distinct")
    licenses_content = _aggregate_candidate(candidate, "license")
    notices_content = _aggregate_candidate(candidate, "notice")
    license_sha, license_size = _write_new(output / license_name, licenses_content, "candidate license output")
    notice_sha, notice_size = _write_new(output / notice_name, notices_content, "candidate notice output")
    surfaces = [
        {
            "surface": item.surface,
            "license": item.license_id,
            "commercial_use_permitted": item.commercial_use_permitted,
            "redistribution_permitted": item.redistribution_permitted,
            "source": item.source,
            "license_evidence_sha256": item.license.sha256,
            "notice_evidence_sha256": item.notice.sha256,
        }
        for item in candidate.surfaces
    ]
    inventory = {
        "schema_id": CANDIDATE_INVENTORY_SCHEMA_ID,
        "candidate_id": candidate.candidate_id,
        "manifest_sha256": candidate.manifest_sha256,
        "licenses_sha256": candidate.licenses_sha256,
        "surfaces": surfaces,
        "outputs": {
            "license": {"path": license_name, "sha256": license_sha, "size_bytes": license_size, "mode": "0644"},
            "notice": {"path": notice_name, "sha256": notice_sha, "size_bytes": notice_size, "mode": "0644"},
        },
    }
    inventory_sha, inventory_size = _write_json(
        output / "candidate-license-inventory.json", inventory, "candidate licence inventory"
    )
    return {
        "candidate_id": candidate.candidate_id,
        "manifest_sha256": candidate.manifest_sha256,
        "licenses_sha256": candidate.licenses_sha256,
        "license_sha256": license_sha,
        "notice_sha256": notice_sha,
        "inventory_sha256": inventory_sha,
        "inventory_size_bytes": inventory_size,
    }


def _write_runtime_bundle(
    records: Sequence[RuntimePackage],
    components: Sequence[RuntimeComponent],
    technical_components: Sequence[Mapping[str, str]],
    lock_sha: str,
    prefix: Path,
    output_dir: Path | str,
    *,
    license_name: str = "LICENSES.txt",
    notice_name: str = "NOTICES.txt",
) -> dict[str, Any]:
    output = Path(output_dir)
    _prepare_output_dir(output)
    license_name = _output_name(license_name, "runtime license output name")
    notice_name = _output_name(notice_name, "runtime notice output name")
    if license_name == notice_name or "runtime-license-inventory.json" in {license_name, notice_name}:
        raise LicenseInputError("runtime license, notice and inventory output names must be distinct")
    licenses_content = _aggregate_runtime(records, components, "license")
    notices_content = _aggregate_runtime(records, components, "notice")
    license_sha, license_size = _write_new(output / license_name, licenses_content, "runtime license output")
    notice_sha, notice_size = _write_new(output / notice_name, notices_content, "runtime notice output")
    packages = [
        {
            "source": package.source,
            "package_url": package.package_url,
            "name": package.name,
            "version": package.version,
            "build": package.build,
            "license": package.license_id,
            "license_family": package.license_family,
            "metadata_sha256": package.metadata_sha256,
            "license_evidence": [
                {"sha256": item.sha256, "size_bytes": item.size_bytes} for item in package.license
            ],
            "notice_evidence": [
                {"sha256": item.sha256, "size_bytes": item.size_bytes} for item in package.notice
            ],
        }
        for package in records
    ]
    component_inventory = [
        {
            "component_id": component.component_id,
            "version": component.version,
            "source": component.source,
            "license": component.license_id,
            "payload_path": component.payload_path,
            "payload_sha256": component.payload_sha256,
            "license_evidence": [
                {"sha256": item.sha256, "size_bytes": item.size_bytes}
                for item in component.license
            ],
            "notice_evidence": [
                {"sha256": item.sha256, "size_bytes": item.size_bytes}
                for item in component.notice
            ],
        }
        for component in components
    ]
    inventory = {
        "schema_id": RUNTIME_INVENTORY_SCHEMA_ID,
        "lock_sha256": lock_sha,
        "package_count": len(packages),
        "packages": packages,
        "component_count": len(component_inventory),
        "components": component_inventory,
        "technical_component_count": len(technical_components),
        "technical_components": [dict(item) for item in technical_components],
        "content": runtime_content_identity(prefix),
        "outputs": {
            "license": {"path": license_name, "sha256": license_sha, "size_bytes": license_size, "mode": "0644"},
            "notice": {"path": notice_name, "sha256": notice_sha, "size_bytes": notice_size, "mode": "0644"},
        },
    }
    inventory_sha, inventory_size = _write_json(
        output / "runtime-license-inventory.json", inventory, "runtime licence inventory"
    )
    return {
        "lock_sha256": lock_sha,
        "package_count": len(packages),
        "license_sha256": license_sha,
        "notice_sha256": notice_sha,
        "inventory_sha256": inventory_sha,
        "inventory_size_bytes": inventory_size,
    }


def verify_runtime_content(
    prefix: Path | str,
    inventory_path: Path | str,
    expected_inventory_sha256: str,
) -> dict[str, Any]:
    """Revalidate an extracted runtime against the human-approved inventory.

    This intentionally does not call :func:`collect_runtime`: conda-pack may omit ``conda-meta``
    and the package cache is never carried.  Generated ``__pycache__`` directories and ``*.pyc``
    files are also excluded because Python may create them while the runtime is smoke-tested.
    Instead the approved inventory supplies the exact canonical payload digest and component
    payload identities; the extracted prefix supplies the bytes to verify.  The destination
    prefix itself is normalized to permit deterministic reruns under different CI paths.
    """

    prefix_path = Path(prefix)
    inventory_file = Path(inventory_path)
    expected_inventory = _hash(expected_inventory_sha256, "runtime inventory SHA256")
    _require_regular(inventory_file, "runtime license inventory")
    actual_inventory = hashlib.sha256(inventory_file.read_bytes()).hexdigest()
    if actual_inventory != expected_inventory:
        raise LicenseInputError(
            f"runtime inventory SHA256 mismatch: expected {expected_inventory}, got {actual_inventory}"
        )
    document = _load_json(inventory_file, "runtime license inventory")
    if not isinstance(document, dict) or document.get("schema_id") != RUNTIME_INVENTORY_SCHEMA_ID:
        raise LicenseInputError(
            f"runtime license inventory schema_id must be {RUNTIME_INVENTORY_SCHEMA_ID}"
        )
    content = document.get("content")
    if not isinstance(content, dict):
        raise LicenseInputError("runtime license inventory must contain canonical content identity")
    _require_exact_keys(
        content,
        {
            "schema_id",
            "excluded_roots",
            "excluded_directories",
            "excluded_suffixes",
            "file_count",
            "sha256",
        },
        "runtime inventory content identity",
    )
    if content["schema_id"] != RUNTIME_CONTENT_SCHEMA_ID:
        raise LicenseInputError(
            f"runtime inventory content schema_id must be {RUNTIME_CONTENT_SCHEMA_ID}"
        )
    if content["excluded_roots"] != list(_RUNTIME_CONTENT_EXCLUDED_ROOTS):
        raise LicenseInputError(
            "runtime inventory content excluded_roots must exactly omit conda-meta only"
        )
    if content["excluded_directories"] != list(_RUNTIME_CONTENT_EXCLUDED_DIRECTORIES):
        raise LicenseInputError(
            "runtime inventory content excluded_directories must exactly omit __pycache__"
        )
    if content["excluded_suffixes"] != list(_RUNTIME_CONTENT_EXCLUDED_SUFFIXES):
        raise LicenseInputError(
            "runtime inventory content excluded_suffixes must exactly omit *.pyc"
        )
    if not isinstance(content["file_count"], int) or content["file_count"] < 0:
        raise LicenseInputError("runtime inventory content file_count must be a non-negative integer")
    expected_content_sha = _hash(content["sha256"], "runtime inventory content SHA256")
    actual_content = runtime_content_identity(prefix_path)
    if actual_content["file_count"] != content["file_count"]:
        raise LicenseInputError(
            "runtime content file count mismatch: "
            f"expected {content['file_count']}, got {actual_content['file_count']}"
        )
    if actual_content["sha256"] != expected_content_sha:
        raise LicenseInputError(
            "runtime content SHA256 mismatch: "
            f"expected {expected_content_sha}, got {actual_content['sha256']}"
        )

    raw_components = document.get("components")
    if not isinstance(raw_components, list):
        raise LicenseInputError("runtime license inventory components must be an array")
    seen_payloads: set[str] = set()
    for index, component in enumerate(raw_components):
        label = f"runtime inventory components[{index}]"
        if not isinstance(component, dict):
            raise LicenseInputError(f"{label} must be an object")
        for field in ("component_id", "payload_path", "payload_sha256"):
            _text(component.get(field), f"{label}.{field}")
        payload_path = component["payload_path"]
        if payload_path in seen_payloads:
            raise LicenseInputError(f"duplicate runtime component payload path: {payload_path}")
        seen_payloads.add(payload_path)
        expected_payload = _hash(component["payload_sha256"], f"{label}.payload_sha256")
        actual_payload = component_payload_sha256(
            _resolve_prefix_path(prefix_path, payload_path, f"{label}.payload_path")
        )
        if actual_payload != expected_payload:
            raise LicenseInputError(
                f"runtime component payload SHA256 mismatch for {component['component_id']}: "
                f"expected {expected_payload}, got {actual_payload}"
            )

    raw_technical = document.get("technical_components", [])
    if not isinstance(raw_technical, list):
        raise LicenseInputError("runtime license inventory technical_components must be an array")
    seen_technical_ids: set[str] = set()
    for index, component in enumerate(raw_technical):
        label = f"runtime inventory technical_components[{index}]"
        if not isinstance(component, dict):
            raise LicenseInputError(f"{label} must be an object")
        _require_exact_keys(
            component,
            {"component_id", "source", "source_sha256", "payload_path", "payload_sha256"},
            label,
        )
        component_id = _text(component["component_id"], f"{label}.component_id")
        if component_id in seen_technical_ids:
            raise LicenseInputError(f"duplicate runtime technical component: {component_id}")
        seen_technical_ids.add(component_id)
        _text(component["source"], f"{label}.source")
        _hash(component["source_sha256"], f"{label}.source_sha256")
        payload_path = _text(component["payload_path"], f"{label}.payload_path")
        expected_payload = _hash(component["payload_sha256"], f"{label}.payload_sha256")
        actual_payload = component_payload_sha256(
            _resolve_prefix_path(prefix_path, payload_path, f"{label}.payload_path")
        )
        if actual_payload != expected_payload:
            raise LicenseInputError(
                f"runtime technical component payload SHA256 mismatch for {component_id}: "
                f"expected {expected_payload}, got {actual_payload}"
            )
    return {
        "inventory_sha256": actual_inventory,
        "content_sha256": actual_content["sha256"],
        "file_count": actual_content["file_count"],
        "component_count": len(raw_components),
        "technical_component_count": len(raw_technical),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    candidate_parser = subparsers.add_parser("candidate", help="collect reviewed candidate evidence")
    candidate_parser.add_argument("--input", required=True, type=Path, help="candidate evidence declaration JSON")
    candidate_parser.add_argument("--output-dir", required=True, type=Path)
    candidate_parser.add_argument("--license-name", default="LICENSES.txt")
    candidate_parser.add_argument("--notice-name", default="NOTICES.txt")

    runtime_parser = subparsers.add_parser("runtime", help="collect conda and component runtime evidence")
    runtime_parser.add_argument("--prefix", required=True, type=Path, help="installed conda environment prefix")
    runtime_parser.add_argument("--lock", required=True, type=Path, help="explicit conda lock/spec")
    runtime_parser.add_argument("--input", required=True, type=Path, help="runtime evidence declaration JSON")
    runtime_parser.add_argument("--lock-sha256", required=True, help="SHA256 of the exact explicit lock")
    runtime_parser.add_argument("--output-dir", required=True, type=Path)
    runtime_parser.add_argument("--license-name", default="LICENSES.txt")
    runtime_parser.add_argument("--notice-name", default="NOTICES.txt")

    generate_parser = subparsers.add_parser(
        "generate-runtime-input", help="generate a runtime declaration from the conda cache"
    )
    generate_parser.add_argument("--prefix", required=True, type=Path)
    generate_parser.add_argument("--lock", required=True, type=Path)
    generate_parser.add_argument("--package-cache", required=True, type=Path)
    generate_parser.add_argument("--lock-sha256", required=True)
    generate_parser.add_argument("--output", required=True, type=Path)
    generate_parser.add_argument("--components", type=Path)
    generate_parser.add_argument("--components-manifest", type=Path)
    generate_parser.add_argument("--supplement", type=Path)

    review_parser = subparsers.add_parser(
        "validate-runtime-review", help="validate Andrew Miller's hash-bound runtime approval"
    )
    review_parser.add_argument("--input", required=True, type=Path, help="runtime legal-review JSON")
    review_parser.add_argument("--sha256", required=True, help="SHA256 of the exact review JSON")
    review_parser.add_argument(
        "--inventory-sha256", required=True, help="SHA256 of the generated runtime inventory"
    )

    verify_parser = subparsers.add_parser(
        "verify-runtime-content",
        help="revalidate an extracted conda-pack runtime against an approved inventory",
    )
    verify_parser.add_argument("--prefix", required=True, type=Path)
    verify_parser.add_argument("--inventory", required=True, type=Path)
    verify_parser.add_argument("--inventory-sha256", required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "candidate":
            result = collect_candidate(
                args.input,
                args.output_dir,
                license_name=args.license_name,
                notice_name=args.notice_name,
            )
        elif args.command == "runtime":
            result = collect_runtime(
                args.prefix,
                args.lock,
                args.input,
                args.output_dir,
                args.lock_sha256,
                license_name=args.license_name,
                notice_name=args.notice_name,
            )
        elif args.command == "generate-runtime-input":
            result = generate_runtime_input(
                args.prefix,
                args.lock,
                args.package_cache,
                args.output,
                args.lock_sha256,
                components_path=args.components,
                components_manifest_path=args.components_manifest,
                supplement_path=args.supplement,
            )
        elif args.command == "validate-runtime-review":
            result = validate_runtime_review(args.input, args.sha256, args.inventory_sha256)
        else:
            result = verify_runtime_content(
                args.prefix,
                args.inventory,
                args.inventory_sha256,
            )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (LicenseInputError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"p25_5 licences: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
