#!/usr/bin/env python3
"""Build and verify a deterministic, dependency-free Phase 2.5 air-gap package.

The package format deliberately keeps technical measurement admission separate from the
shipping/license decision in the v2 report contract.  A package specification must carry an
explicit ``measurement_admitted: true`` decision for every candidate; ``status`` is retained as
the independent shipping decision and is never used to admit a candidate by itself.

All supplied payloads are copied from local regular files.  No URL or download is accepted.  The
evaluator is an explicit package entrypoint and its identity is declared in the specification; it
is not inferred from repository source files.  Evaluator files are executable (0755), opaque
runtime tarballs/shared-library payloads are regular 0644 files, and
model artifacts plus manifests, licences, notices and instructions are always 0644.

The archive is a reproducible gzip-compressed ustar/PAX stream.  Source, staging, archived and
extracted copies are independently checked for regular-file type, expected mode, size and
SHA256.  The external inventory contains the archive SHA256 and the complete carried file list.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import errno
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tarfile
import tempfile
from typing import Any, Iterable, Mapping, Sequence


PACKAGE_SCHEMA_ID = "whitewater-p25-airgap-package-v1"
INVENTORY_SCHEMA_ID = "whitewater-p25-airgap-inventory-v1"
ACTIVE_PROTOCOL_ID = "whitewater-p25-v2"
EXPECTED_FILE_MODE = 0o644
EXPECTED_EXECUTABLE_MODE = 0o755
GENERATED_ADMISSION_DESTINATION = "manifest/measurement-admission.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_URL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


class PackageError(ValueError):
    """A fail-closed, user-facing package validation error."""


@dataclass(frozen=True)
class FileSpec:
    """A local source and its package-relative destination."""

    role: str
    destination: str
    source: Path
    candidate_id: str | None
    mode: int
    # Runtime payloads are opaque inputs (normally a conda-pack tarball).  The expected
    # identity is supplied by the caller rather than inferred only after copying so a
    # replacement runtime cannot be admitted accidentally.
    expected_sha256: str | None = None
    expected_size_bytes: int | None = None


@dataclass(frozen=True)
class FileRecord:
    """An exact file identity carried by a package and its inventory."""

    role: str
    destination: str
    candidate_id: str | None
    sha256: str
    size_bytes: int
    mode: int
    source_path: str | None
    generated: bool = False

    @property
    def mode_token(self) -> str:
        return f"{self.mode:04o}"

    def inventory_dict(self, staging_dir: Path | None = None) -> dict[str, Any]:
        value: dict[str, Any] = {
            "role": self.role,
            "destination": self.destination,
            "candidate_id": self.candidate_id,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "mode": self.mode_token,
            "source_path": self.source_path,
            "generated": self.generated,
        }
        if staging_dir is not None:
            value["staged_path"] = str(staging_dir / Path(self.destination))
        return value


_ROLE_ALLOWED_MODES: dict[str, frozenset[int]] = {
    "evaluator": frozenset({EXPECTED_EXECUTABLE_MODE}),
    "evaluator-support": frozenset({EXPECTED_FILE_MODE}),
    # The evaluator is the only executable entrypoint.  A conda-pack runtime remains an opaque
    # regular 0644 tarball; its internal executables are not outer package members.
    "runtime": frozenset({EXPECTED_FILE_MODE}),
    "model-artifact": frozenset({EXPECTED_FILE_MODE}),
    "candidate-manifest": frozenset({EXPECTED_FILE_MODE}),
    "license": frozenset({EXPECTED_FILE_MODE}),
    "notice": frozenset({EXPECTED_FILE_MODE}),
    "run-instructions": frozenset({EXPECTED_FILE_MODE}),
}


def _no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PackageError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def load_json(path: Path | str) -> Any:
    """Load strict JSON without external dependencies or implicit network access."""

    source = Path(path)

    def reject_constant(value: str) -> None:
        raise PackageError(f"non-finite JSON number {value}")

    try:
        return json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_no_duplicate_pairs,
            parse_constant=reject_constant,
        )
    except PackageError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise PackageError(f"{source}: invalid JSON: {exc}") from exc


def _require_object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise PackageError(f"{label} must be an object")
    return value


def _require_keys(value: Mapping[str, Any], required: Iterable[str], label: str) -> None:
    expected = set(required)
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise PackageError(f"{label} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise PackageError(f"{label} has unknown fields: {', '.join(unknown)}")


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise PackageError(f"{label} must be a non-empty string without NUL")
    return value


def _require_relative_destination(value: Any, label: str) -> str:
    text = _require_string(value, label)
    if "\\" in text or text.startswith("/") or text.endswith("/"):
        raise PackageError(f"{label} must be a normalized relative POSIX path")
    path = PurePosixPath(text)
    if path == PurePosixPath(".") or any(part in {"", ".", ".."} for part in path.parts):
        raise PackageError(f"{label} must not contain '.', '..', or empty path components")
    if str(path) != text:
        raise PackageError(f"{label} must be normalized: {text!r}")
    return text


def _require_candidate_id(value: Any, label: str) -> str:
    text = _require_string(value, label)
    if not _CANDIDATE_ID_RE.fullmatch(text):
        raise PackageError(f"{label} is not a safe candidate identifier: {text!r}")
    return text


def _require_mode(value: Any, label: str) -> int:
    if not isinstance(value, str) or value not in {"0644", "0755"}:
        raise PackageError(f"{label} must be the explicit mode string '0644' or '0755'")
    return int(value, 8)


def _validate_source_text(value: Any, label: str) -> str:
    source = _require_string(value, label)
    if _URL_RE.match(source):
        raise PackageError(f"{label} must be a local path; downloads are not permitted")
    return source


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def _check_regular_mode(path: Path, expected_mode: int, label: str) -> os.stat_result:
    """Require a non-symlink regular file with exactly the expected permission bits."""

    try:
        info = path.lstat()
    except OSError as exc:
        raise PackageError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise PackageError(f"{label} must not be a symlink: {path}")
    _check_stat_regular_mode(info, expected_mode, label, path)
    return info


def _check_stat_regular_mode(
    info: os.stat_result, expected_mode: int, label: str, path: Path | str
) -> None:
    """Validate an already-open file descriptor's identity without following a path."""

    if not stat.S_ISREG(info.st_mode):
        raise PackageError(f"{label} must be a regular file: {path}")
    actual_mode = stat.S_IMODE(info.st_mode)
    if actual_mode != expected_mode:
        raise PackageError(
            f"{label} has mode {actual_mode:04o}; expected {expected_mode:04o}: {path}"
        )


def _same_file_identity(
    first: os.stat_result, second: os.stat_result, label: str, path: Path
) -> None:
    if (first.st_dev, first.st_ino) != (second.st_dev, second.st_ino):
        raise PackageError(f"{label} changed during staging: {path}")


def _sha256_stream(stream: Any) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _hash_file(path: Path, expected_mode: int, label: str) -> tuple[str, int]:
    _check_regular_mode(path, expected_mode, label)
    try:
        with path.open("rb") as stream:
            return _sha256_stream(stream)
    except OSError as exc:
        raise PackageError(f"could not read {label}: {path}: {exc}") from exc


def _copy_source(source: Path, destination: Path, mode: int, label: str) -> tuple[str, int]:
    """Copy a checked source through a no-follow descriptor while preserving its mode.

    A path-only lstat followed by ``Path.open`` has a source-replacement race: an attacker can
    replace the checked regular file with a symlink between those operations.  Open with
    ``O_NOFOLLOW`` where the host provides it, validate the descriptor with ``fstat``, and compare
    the descriptor and path identities before accepting the copy.  On platforms without
    ``O_NOFOLLOW`` the post-open path identity check still fails closed if replacement occurs.
    """

    before = _check_regular_mode(source, mode, label)
    if destination.exists() or destination.is_symlink():
        raise PackageError(f"staging destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    source_fd: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        source_fd = os.open(os.fspath(source), flags)
        opened = os.fstat(source_fd)
        _check_stat_regular_mode(opened, mode, label, source)
        _same_file_identity(before, opened, label, source)
        source_stream = os.fdopen(source_fd, "rb", closefd=True)
        source_fd = None
        with source_stream, destination.open("xb") as output_stream:
            for chunk in iter(lambda: source_stream.read(1024 * 1024), b""):
                output_stream.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            after_fd = os.fstat(source_stream.fileno())
            _check_stat_regular_mode(after_fd, mode, label, source)
            _same_file_identity(opened, after_fd, label, source)
            if after_fd.st_size != size:
                raise PackageError(f"{label} changed size during staging: {source}")
        os.chmod(destination, mode)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise PackageError(f"{label} must not be a symlink: {source}") from exc
        raise PackageError(f"could not stage {label}: {source} -> {destination}: {exc}") from exc
    finally:
        if source_fd is not None:
            try:
                os.close(source_fd)
            except OSError:
                pass
    _check_regular_mode(destination, mode, f"staged {label}")
    # Re-lstat after closing the descriptor.  A path replacement or mode mutation is safer to
    # reject than to silently record a source that did not satisfy the admission rule.
    after_path = _check_regular_mode(source, mode, label)
    _same_file_identity(before, after_path, label, source)
    return digest.hexdigest(), size


def _generated_file(destination: Path, content: bytes, mode: int, label: str) -> tuple[str, int]:
    if destination.exists() or destination.is_symlink():
        raise PackageError(f"generated destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as output_stream:
            output_stream.write(content)
        os.chmod(destination, mode)
    except OSError as exc:
        raise PackageError(f"could not write generated {label}: {destination}: {exc}") from exc
    _check_regular_mode(destination, mode, f"generated {label}")
    return hashlib.sha256(content).hexdigest(), len(content)


def _validate_file_spec(value: Any, index: int, spec_dir: Path, admitted: set[str]) -> FileSpec:
    item = _require_object(value, f"files[{index}]")
    base_keys = {"role", "destination", "source", "candidate_id", "mode"}
    # A runtime is opaque to the outer package verifier.  Require its exact byte identity in
    # the package specification (conda-pack tarballs are the intended runtime input) and carry
    # that identity through the inventory.  The outer verifier never opens or rewrites the
    # runtime's internal archive.
    role_value = item.get("role")
    required_keys = base_keys | ({"sha256", "size_bytes"} if role_value == "runtime" else set())
    _require_keys(item, required_keys, f"files[{index}]")
    role = _require_string(item["role"], f"files[{index}].role")
    if role not in _ROLE_ALLOWED_MODES:
        raise PackageError(f"files[{index}].role is unsupported: {role!r}")
    destination = _require_relative_destination(item["destination"], f"files[{index}].destination")
    source_text = _validate_source_text(item["source"], f"files[{index}].source")
    candidate_raw = item["candidate_id"]
    candidate_id: str | None
    if candidate_raw is None:
        candidate_id = None
    else:
        candidate_id = _require_candidate_id(candidate_raw, f"files[{index}].candidate_id")
        if candidate_id not in admitted:
            raise PackageError(
                f"files[{index}].candidate_id is not explicitly admitted: {candidate_id!r}"
            )
    mode = _require_mode(item["mode"], f"files[{index}].mode")
    if mode not in _ROLE_ALLOWED_MODES[role]:
        allowed = ", ".join(f"{value:04o}" for value in sorted(_ROLE_ALLOWED_MODES[role]))
        raise PackageError(f"files[{index}] role {role!r} requires one of modes {allowed}")
    if role in {"model-artifact", "candidate-manifest"} and candidate_id is None:
        raise PackageError(f"files[{index}] role {role!r} requires candidate_id")
    if role in {"evaluator", "evaluator-support", "runtime", "run-instructions"} and candidate_id is not None:
        raise PackageError(f"files[{index}] role {role!r} must be package-global")
    source = Path(source_text)
    if not source.is_absolute():
        source = spec_dir / source
    expected_sha256: str | None = None
    expected_size_bytes: int | None = None
    if role == "runtime":
        expected_sha256 = _validate_hash(item["sha256"], f"files[{index}].sha256")
        expected_size_bytes = item["size_bytes"]
        if (
            not isinstance(expected_size_bytes, int)
            or isinstance(expected_size_bytes, bool)
            or expected_size_bytes <= 0
        ):
            raise PackageError(f"files[{index}].size_bytes must be a positive integer")
    return FileSpec(
        role,
        destination,
        source,
        candidate_id,
        mode,
        expected_sha256,
        expected_size_bytes,
    )


def _validate_admission(value: Any) -> tuple[list[dict[str, Any]], set[str]]:
    admission = _require_object(value, "admission")
    _require_keys(admission, {"candidates"}, "admission")
    candidates = admission["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise PackageError("admission.candidates must be a non-empty array")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(candidates):
        item = _require_object(value, f"admission.candidates[{index}]")
        required = {"candidate_id", "measurement_status", "measurement_admitted", "status"}
        keys = set(item)
        missing = sorted(required - keys)
        unknown = sorted(keys - required - {"exclusion_reason"})
        if missing:
            raise PackageError(
                f"admission.candidates[{index}] is missing required fields: {', '.join(missing)}"
            )
        if unknown:
            raise PackageError(
                f"admission.candidates[{index}] has unknown fields: {', '.join(unknown)}"
            )
        candidate_id = _require_candidate_id(item["candidate_id"], f"admission.candidates[{index}].candidate_id")
        if candidate_id in seen:
            raise PackageError(f"duplicate admitted candidate: {candidate_id}")
        seen.add(candidate_id)
        if item["measurement_status"] != "measurable":
            raise PackageError(
                f"candidate {candidate_id!r} is not technically measurable; packaging is fail-closed"
            )
        if item["measurement_admitted"] is not True:
            raise PackageError(
                f"candidate {candidate_id!r} lacks explicit measurement admission"
            )
        if item["status"] not in {"eligible", "excluded"}:
            raise PackageError(f"candidate {candidate_id!r} has invalid shipping status")
        if item["status"] == "excluded":
            reason = item.get("exclusion_reason")
            _require_string(reason, f"admission.candidates[{index}].exclusion_reason")
        elif "exclusion_reason" in item:
            raise PackageError(
                f"candidate {candidate_id!r} is shipping-eligible but carries exclusion_reason"
            )
        normalized.append(dict(item))
    return normalized, seen


def load_spec(path: Path | str) -> tuple[dict[str, Any], list[FileSpec]]:
    """Load and fail-closed validate a package specification."""

    spec_path = Path(path)
    value = _require_object(load_json(spec_path), "package specification")
    _require_keys(
        value,
        {"schema_id", "protocol_id", "package_id", "evaluator", "admission", "files"},
        "package specification",
    )
    if value["schema_id"] != PACKAGE_SCHEMA_ID:
        raise PackageError(f"package specification schema_id must be {PACKAGE_SCHEMA_ID!r}")
    if value["protocol_id"] != ACTIVE_PROTOCOL_ID:
        raise PackageError(
            f"package specification protocol_id must be active {ACTIVE_PROTOCOL_ID!r}"
        )
    package_id = _require_string(value["package_id"], "package_id")
    evaluator = _require_object(value["evaluator"], "evaluator")
    _require_keys(evaluator, {"entrypoint", "runtime_identity"}, "evaluator")
    entrypoint = _require_relative_destination(evaluator["entrypoint"], "evaluator.entrypoint")
    runtime_identity = _require_string(evaluator["runtime_identity"], "evaluator.runtime_identity")
    candidates, admitted = _validate_admission(value["admission"])
    raw_files = value["files"]
    if not isinstance(raw_files, list) or not raw_files:
        raise PackageError("files must be a non-empty array")
    spec_dir = spec_path.resolve().parent
    files = [_validate_file_spec(item, index, spec_dir, admitted) for index, item in enumerate(raw_files)]
    destinations = {item.destination for item in files}
    if len(destinations) != len(files):
        raise PackageError("files contains duplicate package destinations")
    if GENERATED_ADMISSION_DESTINATION in destinations:
        raise PackageError(
            f"files may not claim generated destination {GENERATED_ADMISSION_DESTINATION!r}"
        )
    for destination in destinations:
        parts = destination.split("/")
        for index in range(1, len(parts)):
            if "/".join(parts[:index]) in destinations:
                raise PackageError(f"file destination is an ancestor of another destination: {destination}")

    by_role: dict[str, list[FileSpec]] = {}
    for item in files:
        by_role.setdefault(item.role, []).append(item)
    if len(by_role.get("evaluator", [])) != 1:
        raise PackageError("exactly one explicit evaluator file is required")
    evaluator_file = by_role["evaluator"][0]
    if evaluator_file.destination != entrypoint:
        raise PackageError(
            f"evaluator.entrypoint {entrypoint!r} does not match evaluator destination "
            f"{evaluator_file.destination!r}"
        )
    if not by_role.get("runtime"):
        raise PackageError("at least one explicit runtime file is required")
    if len(by_role.get("run-instructions", [])) != 1:
        raise PackageError("exactly one run-instructions file is required")
    for role in ("license", "notice"):
        if not by_role.get(role):
            raise PackageError(f"at least one {role} file is required")
    for candidate_id in admitted:
        manifests = [item for item in files if item.role == "candidate-manifest" and item.candidate_id == candidate_id]
        artifacts = [item for item in files if item.role == "model-artifact" and item.candidate_id == candidate_id]
        if len(manifests) != 1:
            raise PackageError(f"candidate {candidate_id!r} requires exactly one candidate-manifest file")
        if not artifacts:
            raise PackageError(f"candidate {candidate_id!r} requires at least one model-artifact file")
        for role in ("license", "notice"):
            if not any(item.candidate_id in {None, candidate_id} for item in files if item.role == role):
                raise PackageError(f"candidate {candidate_id!r} has no {role} coverage")
    normalized = {
        "schema_id": value["schema_id"],
        "protocol_id": value["protocol_id"],
        "package_id": package_id,
        "evaluator": {"entrypoint": entrypoint, "runtime_identity": runtime_identity},
        "admission": {"candidates": candidates},
    }
    return normalized, files


def _admission_record(spec: Mapping[str, Any]) -> bytes:
    record = {
        "schema_id": "whitewater-p25-measurement-admission-v1",
        "protocol_id": spec["protocol_id"],
        "package_id": spec["package_id"],
        "evaluator": spec["evaluator"],
        "candidates": spec["admission"]["candidates"],
    }
    return (json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _ensure_empty_directory(path: Path, label: str) -> None:
    if path.exists() or path.is_symlink():
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise PackageError(f"{label} must be a directory, not a symlink or file: {path}")
        if any(path.iterdir()):
            raise PackageError(f"{label} must be empty to prevent stale files: {path}")
    else:
        path.mkdir(parents=True, exist_ok=True)


def _write_new_file(path: Path, content: bytes, mode: int, label: str) -> None:
    if path.exists() or path.is_symlink():
        raise PackageError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(content)
        os.chmod(path, mode)
    except OSError as exc:
        raise PackageError(f"could not write {label}: {path}: {exc}") from exc
    _check_regular_mode(path, mode, label)


def _stage_files(spec: Mapping[str, Any], files: Sequence[FileSpec], staging_dir: Path) -> list[FileRecord]:
    records: list[FileRecord] = []
    for item in files:
        destination = staging_dir / Path(item.destination)
        digest, size = _copy_source(item.source, destination, item.mode, item.role)
        if item.role == "runtime" and (
            digest != item.expected_sha256 or size != item.expected_size_bytes
        ):
            raise PackageError(
                f"runtime identity mismatch for {item.destination}: "
                f"expected sha256={item.expected_sha256} size={item.expected_size_bytes}, "
                f"got sha256={digest} size={size}"
            )
        records.append(
            FileRecord(
                item.role,
                item.destination,
                item.candidate_id,
                digest,
                size,
                item.mode,
                str(item.source.resolve()),
            )
        )
    generated_destination = staging_dir / Path(GENERATED_ADMISSION_DESTINATION)
    content = _admission_record(spec)
    digest, size = _generated_file(
        generated_destination, content, EXPECTED_FILE_MODE, "measurement admission"
    )
    records.append(
        FileRecord(
            "admission-record",
            GENERATED_ADMISSION_DESTINATION,
            None,
            digest,
            size,
            EXPECTED_FILE_MODE,
            None,
            generated=True,
        )
    )
    records.sort(key=lambda record: record.destination)
    _validate_manifest_bindings(staging_dir, records)
    return records


def _read_stage_record(stage: Path, record: FileRecord) -> tuple[str, int]:
    return _hash_file(stage / Path(record.destination), record.mode, f"staged {record.destination}")


def _verify_stage_file_set(staging_dir: Path, records: Sequence[FileRecord]) -> None:
    _ensure_directory_for_verify(staging_dir, "staging directory")
    expected = {record.destination for record in records}
    actual: set[str] = set()
    for root, directories, filenames in os.walk(staging_dir, topdown=True, followlinks=False):
        root_path = Path(root)
        for directory in list(directories):
            directory_path = root_path / directory
            if directory_path.is_symlink():
                raise PackageError(f"staging tree contains symlink directory: {directory_path}")
            if not directory_path.is_dir():
                raise PackageError(f"staging tree contains non-directory: {directory_path}")
        for filename in filenames:
            path = root_path / filename
            relative = path.relative_to(staging_dir).as_posix()
            actual.add(relative)
            if relative not in expected:
                raise PackageError(f"staging tree contains unexpected file: {path}")
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise PackageError(f"staging file set differs; missing={missing}, extra={extra}")
    for record in records:
        digest, size = _read_stage_record(staging_dir, record)
        if digest != record.sha256 or size != record.size_bytes:
            raise PackageError(f"staged identity mismatch for {record.destination}")


def _ensure_directory_for_verify(path: Path, label: str) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise PackageError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise PackageError(f"{label} must be a non-symlink directory: {path}")


def _tar_bytes(staging_dir: Path, records: Sequence[FileRecord]) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for record in sorted(records, key=lambda item: item.destination):
            path = staging_dir / Path(record.destination)
            digest, size = _read_stage_record(staging_dir, record)
            if digest != record.sha256 or size != record.size_bytes:
                raise PackageError(f"staged identity changed before archiving: {record.destination}")
            info = tarfile.TarInfo(record.destination)
            info.size = record.size_bytes
            info.mode = record.mode
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.pax_headers = {}
            with path.open("rb") as stream:
                archive.addfile(info, stream)
    compressed = io.BytesIO()
    with gzip.GzipFile(
        fileobj=compressed,
        mode="wb",
        filename="",
        mtime=0,
        compresslevel=9,
    ) as stream:
        stream.write(raw.getvalue())
    return compressed.getvalue()


def _write_archive(path: Path, content: bytes) -> tuple[str, int]:
    _write_new_file(path, content, EXPECTED_FILE_MODE, "archive")
    return _hash_file(path, EXPECTED_FILE_MODE, "archive")


def _inventory_data(
    spec: Mapping[str, Any],
    records: Sequence[FileRecord],
    staging_dir: Path,
    archive_path: Path,
    archive_sha256: str,
    archive_size: int,
) -> dict[str, Any]:
    return {
        "schema_id": INVENTORY_SCHEMA_ID,
        "package_schema_id": spec["schema_id"],
        "protocol_id": spec["protocol_id"],
        "package_id": spec["package_id"],
        "evaluator": spec["evaluator"],
        "admission": spec["admission"],
        "staging_dir": str(staging_dir.resolve()),
        "files": [
            record.inventory_dict(staging_dir.resolve())
            for record in sorted(records, key=lambda item: item.destination)
        ],
        "archive": {
            "path": str(archive_path.resolve()),
            "format": "tar.gz",
            "sha256": archive_sha256,
            "size_bytes": archive_size,
            "mode": "0644",
        },
        "verification": {
            "source": True,
            "staged": True,
            "archive": True,
            "extracted": True,
        },
    }


def _write_inventory(path: Path, data: Mapping[str, Any]) -> None:
    encoded = (json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    _write_new_file(path, encoded, EXPECTED_FILE_MODE, "inventory")


def _validate_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise PackageError(f"{label} must be a lowercase SHA256")
    return value


def _manifest_artifact_name(value: Any, label: str) -> str:
    text = _require_string(value, label)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or not path.name:
        raise PackageError(f"{label} must identify a relative artifact filename")
    return path.name


def _manifest_artifact_identity(value: Any, label: str) -> tuple[str, str, int, str]:
    """Read the exact export identity needed to bind a manifest to its carried artifact."""

    manifest = _require_object(value, label)
    if manifest.get("schema_id") != "whitewater-p25-artifact-v1":
        raise PackageError(f"{label}.schema_id must be whitewater-p25-artifact-v1")
    export = _require_object(manifest.get("export"), f"{label}.export")
    for field in ("artifact", "sha256", "size_bytes", "mode"):
        if field not in export:
            raise PackageError(f"{label}.export is missing required field: {field}")
    artifact = _manifest_artifact_name(export["artifact"], f"{label}.export.artifact")
    sha256 = _validate_hash(export["sha256"], f"{label}.export.sha256")
    size_bytes = export["size_bytes"]
    if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes <= 0:
        raise PackageError(f"{label}.export.size_bytes must be a positive integer")
    if export["mode"] != "0644":
        raise PackageError(f"{label}.export.mode must be 0644 for a carried model artifact")

    candidate = _require_object(manifest.get("candidate"), f"{label}.candidate")
    if candidate.get("id") is None:
        raise PackageError(f"{label}.candidate.id is required")

    # The detailed artifact manifests carry a platform list.  When present, the selected
    # top-level export must be represented by exactly one platform record; otherwise a package
    # could silently carry a different platform artifact under the same candidate name.
    platform_artifacts = export.get("platform_artifacts")
    if platform_artifacts is not None:
        if not isinstance(platform_artifacts, list) or not platform_artifacts:
            raise PackageError(f"{label}.export.platform_artifacts must be a non-empty array")
        platform = _require_string(export.get("platform"), f"{label}.export.platform")
        selected = []
        for index, entry_value in enumerate(platform_artifacts):
            entry = _require_object(entry_value, f"{label}.export.platform_artifacts[{index}]")
            for field in ("platform", "artifact", "sha256", "size_bytes", "mode"):
                if field not in entry:
                    raise PackageError(
                        f"{label}.export.platform_artifacts[{index}] is missing required field: {field}"
                    )
            entry_platform = _require_string(
                entry["platform"], f"{label}.export.platform_artifacts[{index}].platform"
            )
            if entry_platform == platform:
                selected.append(entry)
        if len(selected) != 1:
            raise PackageError(
                f"{label}.export.platform_artifacts must have exactly one selected {platform!r} entry"
            )
        selected_identity = _manifest_artifact_identity(
            {
                "schema_id": "whitewater-p25-artifact-v1",
                "candidate": {"id": candidate["id"]},
                "export": selected[0],
            },
            f"{label}.export.platform_artifacts[{platform}]",
        )
        if selected_identity != (artifact, sha256, size_bytes, "0644"):
            raise PackageError(
                f"{label}.export top-level identity disagrees with selected platform record"
            )
    return artifact, sha256, size_bytes, "0644"


def _validate_manifest_bindings(staging_dir: Path, records: Sequence[FileRecord]) -> None:
    """Bind each carried candidate manifest to one exact carried model artifact."""

    candidate_ids = {
        record.candidate_id
        for record in records
        if record.role == "candidate-manifest" and record.candidate_id is not None
    }
    for candidate_id in sorted(candidate_ids):
        manifests = [
            record
            for record in records
            if record.role == "candidate-manifest" and record.candidate_id == candidate_id
        ]
        artifacts = [
            record
            for record in records
            if record.role == "model-artifact" and record.candidate_id == candidate_id
        ]
        if len(manifests) != 1:
            raise PackageError(f"candidate {candidate_id!r} requires one candidate manifest")
        manifest_record = manifests[0]
        manifest_path = staging_dir / Path(manifest_record.destination)
        manifest = load_json(manifest_path)
        candidate = _require_object(manifest.get("candidate"), f"{manifest_path}.candidate")
        manifest_candidate_id = _require_candidate_id(
            candidate.get("id"), f"{manifest_path}.candidate.id"
        )
        if manifest_candidate_id != candidate_id:
            raise PackageError(
                f"candidate manifest {manifest_path} identifies {manifest_candidate_id!r}, "
                f"not carried candidate {candidate_id!r}"
            )
        artifact_name, artifact_sha, artifact_size, artifact_mode = _manifest_artifact_identity(
            manifest, str(manifest_path)
        )
        matching = [
            artifact
            for artifact in artifacts
            if Path(artifact.destination).name == artifact_name
            and artifact.sha256 == artifact_sha
            and artifact.size_bytes == artifact_size
            and artifact.mode_token == artifact_mode
        ]
        if len(matching) != 1:
            raise PackageError(
                f"candidate manifest {manifest_path} does not match exactly one carried "
                f"artifact (basename={artifact_name!r}, sha256={artifact_sha}, size={artifact_size}, "
                f"mode={artifact_mode})"
            )


def _record_from_inventory(value: Any, index: int) -> FileRecord:
    item = _require_object(value, f"inventory.files[{index}]")
    _require_keys(
        item,
        {"role", "destination", "candidate_id", "sha256", "size_bytes", "mode", "source_path", "generated", "staged_path"},
        f"inventory.files[{index}]",
    )
    role = _require_string(item["role"], f"inventory.files[{index}].role")
    if role == "admission-record":
        allowed_modes = {EXPECTED_FILE_MODE}
    elif role in _ROLE_ALLOWED_MODES:
        allowed_modes = set(_ROLE_ALLOWED_MODES[role])
    else:
        raise PackageError(f"inventory.files[{index}] has unsupported role {role!r}")
    destination = _require_relative_destination(item["destination"], f"inventory.files[{index}].destination")
    candidate_raw = item["candidate_id"]
    candidate_id = None if candidate_raw is None else _require_candidate_id(candidate_raw, f"inventory.files[{index}].candidate_id")
    digest = _validate_hash(item["sha256"], f"inventory.files[{index}].sha256")
    size = item["size_bytes"]
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise PackageError(f"inventory.files[{index}].size_bytes must be a non-negative integer")
    mode = _require_mode(item["mode"], f"inventory.files[{index}].mode")
    if mode not in allowed_modes:
        raise PackageError(f"inventory.files[{index}] mode is not allowed for role {role!r}")
    source_path = item["source_path"]
    if source_path is not None:
        source_path = _require_string(source_path, f"inventory.files[{index}].source_path")
    if not isinstance(item["generated"], bool):
        raise PackageError(f"inventory.files[{index}].generated must be boolean")
    if role == "admission-record" and (not item["generated"] or source_path is not None):
        raise PackageError("generated admission-record identity is malformed")
    if role != "admission-record" and item["generated"]:
        raise PackageError(f"only the admission-record may be generated: {destination}")
    _require_string(item["staged_path"], f"inventory.files[{index}].staged_path")
    return FileRecord(role, destination, candidate_id, digest, size, mode, source_path, item["generated"])


def _validate_record_semantics(
    records: Sequence[FileRecord],
    evaluator: Mapping[str, Any],
    admission: Mapping[str, Any],
    label: str,
) -> None:
    """Check that an inventory still represents a complete admitted package.

    Hash checking alone is not enough: a tampered inventory could relabel a valid model as a
    different candidate or omit a candidate while retaining a byte-for-byte archive.  Reapply
    the same role/count/binding rules used for the source specification before accepting it.
    """

    _, admitted = _validate_admission(admission)
    by_role: dict[str, list[FileRecord]] = {}
    for record in records:
        by_role.setdefault(record.role, []).append(record)
        if record.role in {"model-artifact", "candidate-manifest"} and record.candidate_id not in admitted:
            raise PackageError(f"{label} binds {record.role} to an unadmitted candidate")
        if record.role in {
            "evaluator",
            "evaluator-support",
            "runtime",
            "run-instructions",
            "admission-record",
        } and record.candidate_id is not None:
            raise PackageError(f"{label} binds package-global role {record.role} to a candidate")
        if record.role in {"license", "notice"} and record.candidate_id not in admitted | {None}:
            raise PackageError(f"{label} binds {record.role} to an unadmitted candidate")
    if len(by_role.get("evaluator", [])) != 1:
        raise PackageError(f"{label} requires exactly one evaluator")
    if by_role["evaluator"][0].destination != evaluator["entrypoint"]:
        raise PackageError(f"{label} evaluator entrypoint does not match evaluator file")
    if not by_role.get("runtime"):
        raise PackageError(f"{label} requires at least one runtime file")
    if len(by_role.get("run-instructions", [])) != 1:
        raise PackageError(f"{label} requires exactly one run-instructions file")
    if not by_role.get("license") or not by_role.get("notice"):
        raise PackageError(f"{label} requires licence and notice coverage")
    if len(by_role.get("admission-record", [])) != 1:
        raise PackageError(f"{label} requires exactly one generated admission record")
    for candidate_id in admitted:
        manifests = [
            record
            for record in records
            if record.role == "candidate-manifest" and record.candidate_id == candidate_id
        ]
        artifacts = [
            record
            for record in records
            if record.role == "model-artifact" and record.candidate_id == candidate_id
        ]
        if len(manifests) != 1 or not artifacts:
            raise PackageError(f"{label} has incomplete files for candidate {candidate_id!r}")
        for role in ("license", "notice"):
            if not any(
                record.role == role and record.candidate_id in {None, candidate_id}
                for record in records
            ):
                raise PackageError(f"{label} has no {role} coverage for {candidate_id!r}")


def _load_inventory(path: Path) -> dict[str, Any]:
    value = _require_object(load_json(path), "inventory")
    _require_keys(
        value,
        {"schema_id", "package_schema_id", "protocol_id", "package_id", "evaluator", "admission", "staging_dir", "files", "archive", "verification"},
        "inventory",
    )
    if value["schema_id"] != INVENTORY_SCHEMA_ID:
        raise PackageError(f"inventory schema_id must be {INVENTORY_SCHEMA_ID!r}")
    if value["package_schema_id"] != PACKAGE_SCHEMA_ID:
        raise PackageError("inventory package_schema_id does not identify the package format")
    if value["protocol_id"] != ACTIVE_PROTOCOL_ID:
        raise PackageError("inventory protocol_id is not the active P25-5 protocol")
    _require_string(value["package_id"], "inventory.package_id")
    evaluator = _require_object(value["evaluator"], "inventory.evaluator")
    _require_keys(evaluator, {"entrypoint", "runtime_identity"}, "inventory.evaluator")
    _require_relative_destination(evaluator["entrypoint"], "inventory.evaluator.entrypoint")
    _require_string(evaluator["runtime_identity"], "inventory.evaluator.runtime_identity")
    _require_string(value["staging_dir"], "inventory.staging_dir")
    admission = _require_object(value["admission"], "inventory.admission")
    _validate_admission(admission)
    files_value = value["files"]
    if not isinstance(files_value, list) or not files_value:
        raise PackageError("inventory.files must be a non-empty array")
    records = [_record_from_inventory(item, index) for index, item in enumerate(files_value)]
    destinations = [record.destination for record in records]
    if len(set(destinations)) != len(destinations):
        raise PackageError("inventory.files contains duplicate destinations")
    if GENERATED_ADMISSION_DESTINATION not in set(destinations):
        raise PackageError("inventory is missing generated measurement admission record")
    _validate_record_semantics(records, evaluator, admission, "inventory")
    staging_root = Path(value["staging_dir"])
    for index, item in enumerate(files_value):
        expected_staged = str(staging_root / Path(records[index].destination))
        if item["staged_path"] != expected_staged:
            raise PackageError(f"inventory.files[{index}].staged_path does not match staging_dir")
    archive = _require_object(value["archive"], "inventory.archive")
    _require_keys(archive, {"path", "format", "sha256", "size_bytes", "mode"}, "inventory.archive")
    _require_string(archive["path"], "inventory.archive.path")
    if archive["format"] != "tar.gz":
        raise PackageError("inventory.archive.format must be tar.gz")
    _validate_hash(archive["sha256"], "inventory.archive.sha256")
    if not isinstance(archive["size_bytes"], int) or isinstance(archive["size_bytes"], bool) or archive["size_bytes"] <= 0:
        raise PackageError("inventory.archive.size_bytes must be a positive integer")
    if archive["mode"] != "0644":
        raise PackageError("inventory.archive.mode must be 0644")
    verification = _require_object(value["verification"], "inventory.verification")
    _require_keys(verification, {"source", "staged", "archive", "extracted"}, "inventory.verification")
    if any(verification[key] is not True for key in verification):
        raise PackageError("inventory verification flags must all be true")
    # Keep normalized records available to callers without changing the on-disk schema.
    normalized = dict(value)
    normalized["_records"] = records
    return normalized


def _verify_archive_members(
    archive_path: Path,
    records: Sequence[FileRecord],
    admission_record: bytes | None = None,
) -> None:
    expected = {record.destination: record for record in records}
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            actual: set[str] = set()
            for member in archive:
                destination = _require_relative_destination(member.name, "archive member name")
                if destination in actual:
                    raise PackageError(f"archive contains duplicate member: {destination}")
                actual.add(destination)
                if destination not in expected:
                    raise PackageError(f"archive contains unexpected member: {destination}")
                if not member.isreg():
                    raise PackageError(f"archive member is not a regular file: {destination}")
                record = expected[destination]
                if stat.S_IMODE(member.mode) != record.mode:
                    raise PackageError(
                        f"archive member {destination} has mode {stat.S_IMODE(member.mode):04o}; "
                        f"expected {record.mode:04o}"
                    )
                if member.size != record.size_bytes:
                    raise PackageError(f"archive member size mismatch: {destination}")
                stream = archive.extractfile(member)
                if stream is None:
                    raise PackageError(f"archive member cannot be read: {destination}")
                if record.role == "admission-record":
                    content = stream.read()
                    if admission_record is None or content != admission_record:
                        raise PackageError("archive admission record does not match inventory admission")
                    digest, size = hashlib.sha256(content).hexdigest(), len(content)
                else:
                    digest, size = _sha256_stream(stream)
                if digest != record.sha256 or size != record.size_bytes:
                    raise PackageError(f"archive member identity mismatch: {destination}")
            if actual != set(expected):
                raise PackageError(
                    f"archive member set differs; missing={sorted(set(expected) - actual)}, "
                    f"extra={sorted(actual - set(expected))}"
                )
    except (OSError, tarfile.TarError) as exc:
        raise PackageError(f"could not inspect archive {archive_path}: {exc}") from exc


def _extract_archive(archive_path: Path, extract_dir: Path, records: Sequence[FileRecord]) -> None:
    _ensure_empty_directory(extract_dir, "extraction directory")
    expected = {record.destination: record for record in records}
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive:
                destination = _require_relative_destination(member.name, "archive member name")
                if destination not in expected or not member.isreg():
                    raise PackageError(f"archive member cannot be safely extracted: {destination}")
                output = extract_dir / Path(destination)
                output.parent.mkdir(parents=True, exist_ok=True)
                if output.exists() or output.is_symlink():
                    raise PackageError(f"duplicate extraction destination: {output}")
                stream = archive.extractfile(member)
                if stream is None:
                    raise PackageError(f"archive member cannot be read: {destination}")
                with output.open("xb") as output_stream:
                    shutil.copyfileobj(stream, output_stream, length=1024 * 1024)
                os.chmod(output, expected[destination].mode)
    except (OSError, tarfile.TarError) as exc:
        raise PackageError(f"could not extract archive {archive_path}: {exc}") from exc
    _verify_extracted_files(extract_dir, records)


def _verify_extracted_files(extract_dir: Path, records: Sequence[FileRecord]) -> None:
    _verify_stage_file_set(extract_dir, records)


def verify_package(
    archive_path: Path | str,
    inventory_path: Path | str,
    *,
    staging_dir: Path | str | None = None,
    extract_dir: Path | str | None = None,
    verify_sources: bool = False,
) -> dict[str, Any]:
    """Verify an inventory, archive, optional staging/source copies and extraction.

    ``verify_sources`` is opt-in because the inventory intentionally records build-machine
    source paths that are normally unavailable on an air-gapped destination.  The builder always
    verifies sources before it emits the inventory.
    """

    archive = Path(archive_path)
    inventory_file = Path(inventory_path)
    _check_regular_mode(inventory_file, EXPECTED_FILE_MODE, "inventory")
    inventory = _load_inventory(inventory_file)
    records: list[FileRecord] = inventory["_records"]
    archive_sha, archive_size = _hash_file(archive, EXPECTED_FILE_MODE, "archive")
    declared = inventory["archive"]
    if archive_sha != declared["sha256"] or archive_size != declared["size_bytes"]:
        raise PackageError("archive SHA256 or size does not match inventory")
    admission_record = _admission_record(
        {
            "protocol_id": inventory["protocol_id"],
            "package_id": inventory["package_id"],
            "evaluator": inventory["evaluator"],
            "admission": inventory["admission"],
        }
    )
    _verify_archive_members(archive, records, admission_record)
    stage = Path(staging_dir) if staging_dir is not None else None
    if stage is not None:
        _verify_stage_file_set(stage, records)
        _validate_manifest_bindings(stage, records)
    if verify_sources:
        for record in records:
            if record.generated:
                continue
            if record.source_path is None:
                raise PackageError(f"non-generated record lacks source path: {record.destination}")
            digest, size = _hash_file(Path(record.source_path), record.mode, f"source {record.destination}")
            if digest != record.sha256 or size != record.size_bytes:
                raise PackageError(f"source identity mismatch: {record.destination}")
    temporary_extract: tempfile.TemporaryDirectory[str] | None = None
    if extract_dir is None:
        temporary_extract = tempfile.TemporaryDirectory(prefix="whitewater-p25-extract-")
        extraction = Path(temporary_extract.name)
    else:
        extraction = Path(extract_dir)
    try:
        _extract_archive(archive, extraction, records)
        _validate_manifest_bindings(extraction, records)
    finally:
        if temporary_extract is not None:
            temporary_extract.cleanup()
    return {
        "package_id": inventory["package_id"],
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive_size,
        "file_count": len(records),
        "staged_verified": stage is not None,
        "sources_verified": verify_sources,
        "extracted_verified": True,
    }


def build_package(
    spec_path: Path | str,
    *,
    staging_dir: Path | str,
    archive_path: Path | str,
    inventory_path: Path | str | None = None,
) -> dict[str, Any]:
    """Build and fully verify an air-gap package from local, explicitly admitted inputs."""

    spec_file = Path(spec_path)
    spec, files = load_spec(spec_file)
    stage = Path(staging_dir)
    archive = Path(archive_path)
    if _is_within(archive, stage) or (inventory_path is not None and _is_within(Path(inventory_path), stage)):
        raise PackageError("archive and inventory outputs must be outside the staging directory")
    _ensure_empty_directory(stage, "staging directory")
    records = _stage_files(spec, files, stage)
    _verify_stage_file_set(stage, records)
    archive_content = _tar_bytes(stage, records)
    archive_sha, archive_size = _write_archive(archive, archive_content)
    inventory = _inventory_data(spec, records, stage, archive, archive_sha, archive_size)
    inventory_file = Path(inventory_path) if inventory_path is not None else archive.with_name(archive.name + ".inventory.json")
    _write_inventory(inventory_file, inventory)
    # Verify the exact carried archive and staged tree, including an extraction round-trip.  The
    # source paths are checked in _stage_files; the optional source pass makes that invariant
    # explicit for callers who want the additional post-build audit.
    verify_package(archive, inventory_file, staging_dir=stage, verify_sources=True)
    return {
        "package_id": spec["package_id"],
        "archive_path": str(archive.resolve()),
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive_size,
        "inventory_path": str(inventory_file.resolve()),
        "staging_dir": str(stage.resolve()),
        "file_count": len(records),
    }


def _build_cli(args: argparse.Namespace) -> int:
    result = build_package(
        args.spec,
        staging_dir=args.staging_dir,
        archive_path=args.archive,
        inventory_path=args.inventory,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def _verify_cli(args: argparse.Namespace) -> int:
    result = verify_package(
        args.archive,
        args.inventory,
        staging_dir=args.staging_dir,
        extract_dir=args.extract_dir,
        verify_sources=args.verify_sources,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build", help="stage and build a deterministic tar.gz")
    build_parser.add_argument("spec", type=Path)
    build_parser.add_argument("--staging-dir", required=True, type=Path)
    build_parser.add_argument("--archive", required=True, type=Path)
    build_parser.add_argument("--inventory", type=Path)
    build_parser.set_defaults(function=_build_cli)
    verify_parser = subparsers.add_parser("verify", help="verify archive, inventory and extraction")
    verify_parser.add_argument("archive", type=Path)
    verify_parser.add_argument("inventory", type=Path)
    verify_parser.add_argument("--staging-dir", type=Path)
    verify_parser.add_argument("--extract-dir", type=Path)
    verify_parser.add_argument("--verify-sources", action="store_true")
    verify_parser.set_defaults(function=_verify_cli)
    args = parser.parse_args(argv)
    try:
        return args.function(args)
    except (PackageError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"p25_5 package: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
