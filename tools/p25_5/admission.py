#!/usr/bin/env python3
"""Generate a deterministic, fail-closed P25-5 measurement-admission document.

This tool is the narrow seam between exact candidate artifact manifests and the evaluator
packager.  It validates the active protocol-v2 document, then delegates manifest/schema/tensor
contract and exact artifact checks to ``models.artifact_workflow``.  Shipping/license status and
technical measurement status are emitted separately:

* a shipping candidate with complete permissive terms can be ``eligible`` and ``measurable``;
* a validation baseline or license-excluded candidate can be ``excluded`` but ``measurable``;
* an absent artifact is emitted as ``unavailable`` only with the explicit ``allow_missing``
  request, and never silently admitted to measurement.

The output is a candidate-admission document rather than a full result report.  Its candidate
objects use the v2 report vocabulary plus ``measurement_admitted`` for the downstream package
boundary.  No network access or model-family inference occurs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
ACTIVE_PROTOCOL_ID = "whitewater-p25-v2"
ADMISSION_SCHEMA_ID = "whitewater-p25-admission-v1"
EXPECTED_MODE = 0o644
_PROVIDER_ORDER = ("cpu", "cuda", "coreml")
_LICENSE_SURFACES = ("code", "checkpoint", "backbone")
_COMMERCIAL_MAP = {
    "yes": "commercial_use_permitted",
    "no": "not_permitted",
    "unknown": "unknown",
}
_REDISTRIBUTION_MAP = {
    "yes": "permitted",
    "no": "not_permitted",
    "unknown": "unknown",
}
_FAILURE_TYPES = {
    "artifact_missing",
    "artifact_hash_mismatch",
    "license_not_permitted",
    "license_unknown",
    "export_not_reproducible",
    "other",
}


class AdmissionError(ValueError):
    """A fail-closed admission or input error."""


@dataclass(frozen=True)
class CandidateInput:
    """One exact manifest, artifact and optional manifest platform selection."""

    manifest_path: Path
    artifact_path: Path
    platform: str | None = None


def _load_private_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module {module_name}: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_artifact_workflow():
    """Load the existing artifact framework without changing ``sys.path``."""

    exclusion_path = ROOT / "models" / "exclusion_contract.py"
    artifact_path = ROOT / "models" / "artifact_workflow.py"
    previous = sys.modules.get("exclusion_contract")
    exclusion = _load_private_module("_whitewater_p25_admission_exclusion_contract", exclusion_path)
    sys.modules["exclusion_contract"] = exclusion
    try:
        return _load_private_module("_whitewater_p25_admission_artifact_workflow", artifact_path)
    finally:
        if previous is None:
            sys.modules.pop("exclusion_contract", None)
        else:
            sys.modules["exclusion_contract"] = previous


def _load_validator():
    return _load_private_module(
        "_whitewater_p25_admission_bakeoff_validator",
        ROOT / "tools" / "bakeoff" / "validator.py",
    )


artifact_workflow = _load_artifact_workflow()
validator = _load_validator()


def _require_object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise AdmissionError(f"{label} must be an object")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise AdmissionError(f"{label} must be a non-empty string without NUL")
    return value


def _failure(kind: str, message: str) -> dict[str, str]:
    if kind not in _FAILURE_TYPES:
        raise AdmissionError(f"unsupported typed exclusion reason: {kind}")
    return {"type": kind, "message": message}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular_0644(path: Path, label: str) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise AdmissionError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise AdmissionError(f"{label} must not be a symlink: {path}")
    if not stat.S_ISREG(info.st_mode):
        raise AdmissionError(f"{label} must be a regular file: {path}")
    mode = stat.S_IMODE(info.st_mode)
    if mode != EXPECTED_MODE:
        raise AdmissionError(f"{label} has mode {mode:04o}; expected 0644: {path}")


def _load_protocol(protocol_path: Path) -> tuple[dict[str, Any], str, Mapping[str, Any]]:
    _require_regular_0644(protocol_path, "protocol")
    protocol = artifact_workflow.load_json(protocol_path)
    if not isinstance(protocol, dict) or protocol.get("protocol_id") != ACTIVE_PROTOCOL_ID:
        raise AdmissionError(f"protocol must identify active {ACTIVE_PROTOCOL_ID}")
    protocol_schema = artifact_workflow.load_json(ROOT / "bakeoff" / "protocol-v2.schema.json")
    report_schema = artifact_workflow.load_json(ROOT / "bakeoff" / "report-v2.schema.json")
    try:
        validator.validate_protocol_consistency(protocol, protocol_schema, report_schema)
    except (validator.ValidationError, KeyError, TypeError, ValueError) as exc:
        raise AdmissionError(f"invalid active protocol-v2 document: {exc}") from exc
    return protocol, _sha256_file(protocol_path), report_schema


def _provider_tokens(protocol: Mapping[str, Any], providers: Sequence[str]) -> list[str]:
    if not providers:
        raise AdmissionError("at least one explicitly measured provider token is required")
    if len(set(providers)) != len(providers):
        raise AdmissionError("measured provider tokens must be unique")
    known = {entry["token"] for entry in protocol["providers"]}
    unknown = sorted(set(providers) - known)
    if unknown:
        raise AdmissionError(f"unknown measured provider token(s): {', '.join(unknown)}")
    return [token for token in _PROVIDER_ORDER if token in providers]


def _reviewed_surface_tokens(reviewed_surfaces: Sequence[str]) -> list[str]:
    """Validate and canonicalize explicit legal-review attestations.

    The artifact manifest's ``audit`` fields are free-form evidence strings.  They are not
    boolean attestations for the report-v2 ``redistribution_terms_reviewed`` field, so this
    admission seam accepts only explicit, operator-supplied surface tokens.
    """

    if len(set(reviewed_surfaces)) != len(reviewed_surfaces):
        raise AdmissionError("reviewed legal surface tokens must be unique")
    unknown = sorted(set(reviewed_surfaces) - set(_LICENSE_SURFACES))
    if unknown:
        raise AdmissionError(f"unknown reviewed legal surface token(s): {', '.join(unknown)}")
    missing = [surface for surface in _LICENSE_SURFACES if surface not in reviewed_surfaces]
    if missing:
        raise AdmissionError(
            "explicit review attestations are required for every legal surface; "
            f"missing: {', '.join(missing)}"
        )
    return [surface for surface in _LICENSE_SURFACES if surface in reviewed_surfaces]


def _selected_export(manifest: Mapping[str, Any], platform: str | None) -> Mapping[str, Any]:
    selected_platform = platform or manifest["export"]["platform"]
    entries = [
        entry
        for entry in manifest["export"]["platform_artifacts"]
        if entry["platform"] == selected_platform
    ]
    if len(entries) != 1:
        raise AdmissionError(
            f"manifest has {len(entries)} artifact entries for platform {selected_platform!r}; expected one"
        )
    return entries[0]


def _candidate_role(protocol: Mapping[str, Any], candidate_id: str) -> str:
    for entry in protocol["candidate_ids"]:
        if entry["id"] == candidate_id:
            return entry["role"]
    raise AdmissionError(f"candidate {candidate_id!r} is not in active protocol-v2")


def _validate_role(protocol_role: str, manifest_role: str, candidate_id: str) -> None:
    if protocol_role == "validation-baseline":
        if manifest_role != "validation-baseline":
            raise AdmissionError(
                f"{candidate_id}: protocol validation-baseline requires manifest role validation-baseline"
            )
        return
    if protocol_role == "shipping-candidate":
        # P25-3 records an evaluation-only shipping candidate as role=excluded when its legal
        # terms are unresolved.  That remains measurable, but never becomes shipping-eligible.
        if manifest_role not in {"shipping-candidate", "excluded"}:
            raise AdmissionError(
                f"{candidate_id}: manifest role {manifest_role!r} does not match shipping-candidate protocol role"
            )
        return
    raise AdmissionError(f"{candidate_id}: unsupported protocol role {protocol_role!r}")


def _legal_surfaces(
    manifest: Mapping[str, Any], reviewed_surfaces: set[str]
) -> tuple[dict[str, str], dict[str, str], dict[str, bool]]:
    commercial: dict[str, str] = {}
    redistribution: dict[str, str] = {}
    reviewed: dict[str, bool] = {}
    for surface in _LICENSE_SURFACES:
        entry = _require_object(manifest["licenses"][surface], f"licenses.{surface}")
        commercial_value = entry.get("commercial_use_permitted")
        redistribution_value = entry.get("redistribution_permitted")
        if commercial_value not in _COMMERCIAL_MAP:
            raise AdmissionError(f"licenses.{surface}.commercial_use_permitted is invalid")
        if redistribution_value not in _REDISTRIBUTION_MAP:
            raise AdmissionError(f"licenses.{surface}.redistribution_permitted is invalid")
        # artifact-v1 requires ``audit`` as evidence text, but it does not define that free-text
        # field as a boolean attestation of the v2 redistribution review decision.  Only the
        # caller's explicit ``--reviewed-surface`` attestations may produce true here.
        commercial[surface] = _COMMERCIAL_MAP[commercial_value]
        redistribution[surface] = _REDISTRIBUTION_MAP[redistribution_value]
        reviewed[surface] = surface in reviewed_surfaces
    return commercial, redistribution, reviewed


def _shipping_exclusion_reason(
    manifest: Mapping[str, Any],
    protocol_role: str,
    commercial: Mapping[str, str],
    redistribution: Mapping[str, str],
    reviewed: Mapping[str, bool],
    *,
    artifact_missing: bool = False,
) -> dict[str, str]:
    if artifact_missing:
        return _failure("artifact_missing", "exact exported artifact is absent; shipping admission is unavailable")
    if protocol_role == "validation-baseline":
        return _failure("other", "validation-baseline is evaluation-only and cannot be shipping-eligible")
    exclusion = manifest.get("exclusion")
    reason_code = exclusion.get("reason_code") if isinstance(exclusion, dict) else None
    if reason_code == "checkpoint_license_terms_unknown":
        return _failure("license_unknown", "manifest excludes shipping because checkpoint terms are unknown")
    if reason_code == "export_or_operator_failure":
        return _failure("export_not_reproducible", "manifest records an export or operator failure")
    if any(value == "not_permitted" for value in commercial.values()) or any(
        value == "not_permitted" for value in redistribution.values()
    ):
        return _failure("license_not_permitted", "one or more legal surfaces prohibit shipping")
    if any(value == "unknown" for value in commercial.values()) or any(
        value == "unknown" for value in redistribution.values()
    ):
        return _failure("license_unknown", "one or more legal surfaces remain unknown")
    if not all(reviewed.values()):
        return _failure(
            "other",
            "redistribution terms were not explicitly reviewed for every legal surface",
        )
    return _failure("other", "manifest is not explicitly shipping-admitted")


def _manifest_candidate(
    protocol: Mapping[str, Any],
    protocol_role: str,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    selected: Mapping[str, Any],
    providers: Sequence[str],
    reviewed_surfaces: set[str],
) -> dict[str, Any]:
    candidate = _require_object(manifest["candidate"], "manifest.candidate")
    candidate_id = _require_string(candidate.get("id"), "manifest.candidate.id")
    _validate_role(protocol_role, candidate.get("role"), candidate_id)
    commercial, redistribution, reviewed = _legal_surfaces(manifest, reviewed_surfaces)

    source_commit = manifest["upstream"]["commit"]
    checkpoint_sha256 = manifest["checkpoint"]["sha256"]
    export_environment_sha256 = selected["export_environment_sha256"]
    artifact_sha256 = selected["sha256"]
    artifact_size = selected["size_bytes"]
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise AdmissionError(f"{candidate_id}: source commit is not an exact 40-hex identity")
    for label, value in (
        ("checkpoint SHA256", checkpoint_sha256),
        ("artifact SHA256", artifact_sha256),
        ("export environment SHA256", export_environment_sha256),
    ):
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise AdmissionError(f"{candidate_id}: {label} is not an exact lowercase SHA256")
    if not isinstance(artifact_size, int) or isinstance(artifact_size, bool) or artifact_size <= 0:
        raise AdmissionError(f"{candidate_id}: artifact size is not positive")

    entry: dict[str, Any] = {
        "candidate_id": candidate_id,
        "status": "excluded",
        "measurement_status": "measurable",
        "source_commit": source_commit,
        "checkpoint_sha256": checkpoint_sha256,
        "artifact_sha256": artifact_sha256,
        "export_environment_sha256": export_environment_sha256,
        "manifest_sha256": _sha256_file(manifest_path),
        "artifact_size_bytes": artifact_size,
        "measurement_providers": list(providers),
        "measurement_admitted": True,
        "license_verdicts": commercial,
        "redistribution_permitted": redistribution,
        "redistribution_terms_reviewed": reviewed,
    }
    backbone_sha256 = manifest["backbone"].get("checkpoint_sha256")
    if manifest["backbone"].get("applicable") is True and backbone_sha256 is None:
        raise AdmissionError(f"{candidate_id}: applicable backbone lacks an exact checkpoint SHA256")
    if backbone_sha256 is not None:
        if not isinstance(backbone_sha256, str) or len(backbone_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in backbone_sha256
        ):
            raise AdmissionError(f"{candidate_id}: backbone checkpoint SHA256 is not exact")
        entry["backbone_sha256"] = backbone_sha256

    shipping_eligible = (
        protocol_role == "shipping-candidate"
        and candidate.get("role") == "shipping-candidate"
        and manifest["status"] != "excluded"
        and all(value == "commercial_use_permitted" for value in commercial.values())
        and all(value == "permitted" for value in redistribution.values())
        and all(reviewed.values())
    )
    if shipping_eligible:
        entry["status"] = "eligible"
    else:
        entry["exclusion_reason"] = _shipping_exclusion_reason(
            manifest, protocol_role, commercial, redistribution, reviewed
        )
    return entry


def _unavailable_candidate(
    protocol_role: str,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    providers: Sequence[str],
    reviewed_surfaces: set[str],
) -> dict[str, Any]:
    candidate = _require_object(manifest["candidate"], "manifest.candidate")
    candidate_id = _require_string(candidate.get("id"), "manifest.candidate.id")
    commercial, redistribution, reviewed = _legal_surfaces(manifest, reviewed_surfaces)
    shipping_reason = _shipping_exclusion_reason(
        manifest,
        protocol_role,
        commercial,
        redistribution,
        reviewed,
        artifact_missing=True,
    )
    if protocol_role == "validation-baseline":
        shipping_reason = _shipping_exclusion_reason(
            manifest, protocol_role, commercial, redistribution, reviewed
        )
    return {
        "candidate_id": candidate_id,
        "status": "excluded",
        "measurement_status": "unavailable",
        "exclusion_reason": shipping_reason,
        "measurement_exclusion_reason": _failure(
            "artifact_missing",
            f"exact artifact for {candidate_id} is missing: {manifest_path}",
        ),
    }


def _validate_v2_candidate_entry(
    candidate: Mapping[str, Any],
    report_schema: Mapping[str, Any],
    label: str,
) -> None:
    """Check generated candidate fields against the frozen report-v2 candidate schema.

    ``measurement_admitted`` belongs only to this admission document and is intentionally
    removed before validation.  Every measurable entry is required to carry explicit review
    attestations for all three legal surfaces, so the actual values satisfy the frozen
    report-v2 candidate schema without inference or coercion.
    """

    schema_defs = report_schema.get("$defs")
    if not isinstance(schema_defs, Mapping) or not isinstance(schema_defs.get("candidate"), Mapping):
        raise AdmissionError("report-v2 schema has no candidate definition")
    candidate_for_schema = dict(candidate)
    candidate_for_schema.pop("measurement_admitted", None)
    try:
        validator.validate(
            candidate_for_schema,
            schema_defs["candidate"],
            path=label,
            root=report_schema,
        )
    except (validator.ValidationError, KeyError, TypeError, ValueError) as exc:
        raise AdmissionError(f"generated candidate does not satisfy report-v2 semantics: {label}: {exc}") from exc


def _artifact_missing(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return True
    except OSError as exc:
        raise AdmissionError(f"could not inspect artifact path {path}: {exc}") from exc
    # A dangling symlink is not an absent artifact: it is a forbidden file type and must fail
    # through the existing artifact validator instead of being converted to unavailable evidence.
    return False


def generate_admission(
    protocol_path: Path | str,
    candidates: Sequence[CandidateInput],
    providers: Sequence[str],
    *,
    allow_missing: bool = False,
    reviewed_surfaces: Sequence[str] = (),
) -> dict[str, Any]:
    """Generate a deterministic candidate-admission document from exact local inputs."""

    protocol_file = Path(protocol_path)
    protocol, protocol_sha256, report_schema = _load_protocol(protocol_file)
    measurement_providers = _provider_tokens(protocol, providers)
    reviewed_surface_order = _reviewed_surface_tokens(reviewed_surfaces)
    reviewed_surface_set = set(reviewed_surface_order)
    if not candidates:
        raise AdmissionError("at least one exact manifest/artifact input is required")

    known_ids = {entry["id"]: entry["role"] for entry in protocol["candidate_ids"]}
    entries: dict[str, dict[str, Any]] = {}
    for candidate_input in candidates:
        manifest_path = Path(candidate_input.manifest_path)
        artifact_path = Path(candidate_input.artifact_path)
        _require_regular_0644(manifest_path, "manifest")
        try:
            manifest = artifact_workflow.load_manifest(
                manifest_path,
                protocol_path=ROOT / "bakeoff" / "protocol-v2.json",
            )
        except (artifact_workflow.ArtifactError, artifact_workflow.ValidationError, OSError, KeyError, TypeError, ValueError) as exc:
            raise AdmissionError(f"invalid candidate manifest {manifest_path}: {exc}") from exc
        candidate = _require_object(manifest.get("candidate"), f"{manifest_path}.candidate")
        candidate_id = _require_string(candidate.get("id"), f"{manifest_path}.candidate.id")
        if candidate_id not in known_ids:
            raise AdmissionError(f"candidate {candidate_id!r} is not in active protocol-v2")
        if candidate_id in entries:
            raise AdmissionError(f"duplicate candidate input: {candidate_id}")
        protocol_role = known_ids[candidate_id]
        _validate_role(protocol_role, candidate.get("role"), candidate_id)
        selected = _selected_export(manifest, candidate_input.platform)
        expected_artifact_name = Path(selected["artifact"]).name
        if artifact_path.name != expected_artifact_name:
            raise AdmissionError(
                f"{candidate_id}: exact artifact basename {artifact_path.name!r} does not match "
                f"manifest {expected_artifact_name!r}"
            )
        if _artifact_missing(artifact_path):
            if not allow_missing:
                raise AdmissionError(
                    f"{candidate_id}: artifact is missing; pass --allow-missing only to record unavailable evidence"
                )
            entries[candidate_id] = _unavailable_candidate(
                protocol_role,
                manifest,
                manifest_path,
                measurement_providers,
                reviewed_surface_set,
            )
            _validate_v2_candidate_entry(entries[candidate_id], report_schema, candidate_id)
            continue
        try:
            artifact_workflow.validate_artifact(
                manifest,
                manifest_path,
                artifact_path,
                platform=candidate_input.platform,
            )
        except (artifact_workflow.ArtifactError, OSError, KeyError, TypeError, ValueError) as exc:
            raise AdmissionError(f"invalid exact artifact for {candidate_id}: {exc}") from exc
        entries[candidate_id] = _manifest_candidate(
            protocol,
            protocol_role,
            manifest,
            manifest_path,
            selected,
            measurement_providers,
            reviewed_surface_set,
        )
        _validate_v2_candidate_entry(entries[candidate_id], report_schema, candidate_id)

    protocol_order = {entry["id"]: index for index, entry in enumerate(protocol["candidate_ids"])}
    ordered_entries = [entries[candidate_id] for candidate_id in sorted(entries, key=protocol_order.__getitem__)]
    return {
        "schema_id": ADMISSION_SCHEMA_ID,
        "protocol_id": ACTIVE_PROTOCOL_ID,
        "protocol_sha256": protocol_sha256,
        "measurement_providers": measurement_providers,
        "candidates": ordered_entries,
    }


def _check_existing_destination(destination: Path, *, replace: bool) -> None:
    """Reject unsafe destination types and enforce no-clobber by default."""

    try:
        info = destination.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise AdmissionError(f"could not inspect admission output {destination}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise AdmissionError(f"admission output must not be a symlink: {destination}")
    if not stat.S_ISREG(info.st_mode):
        raise AdmissionError(f"admission output must be a regular file: {destination}")
    if not replace:
        raise AdmissionError(
            f"admission output already exists: {destination}; pass replace=True or --replace"
        )


def _fsync_parent_directory(parent: Path) -> None:
    """Best-effort directory fsync after publishing a new output entry."""

    try:
        directory_fd = os.open(parent, os.O_RDONLY)
    except OSError:
        return
    try:
        try:
            os.fsync(directory_fd)
        except OSError:
            # Some supported filesystems do not permit fsync on directory descriptors; the
            # file itself was already fsynced before publication.
            pass
    finally:
        os.close(directory_fd)


def write_admission(
    path: Path | str,
    document: Mapping[str, Any],
    *,
    replace: bool = False,
) -> None:
    """Write canonical JSON with mode 0644 using an atomic, no-clobber publication."""

    destination = Path(path)
    if destination.is_symlink():
        raise AdmissionError(f"admission output must not be a symlink: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _check_existing_destination(destination, replace=replace)
    content = (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    temporary: Path | None = None
    descriptor: int | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(destination.parent),
        )
        temporary = Path(temporary_name)
        os.fchmod(descriptor, EXPECTED_MODE)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())

        if replace:
            # The explicit replace mode is for a caller that has consciously opted into
            # replacing an existing regular output.  Unsafe existing types were rejected
            # immediately before the temporary file was created.
            os.replace(temporary, destination)
            temporary = None
        else:
            # A hard link gives POSIX no-clobber semantics: a destination appearing after the
            # initial lstat cannot be overwritten.  Both entries are in the destination
            # directory, so this does not cross filesystems.
            try:
                os.link(temporary, destination, follow_symlinks=False)
            except FileExistsError as exc:
                raise AdmissionError(f"admission output already exists: {destination}") from exc
            os.unlink(temporary)
            temporary = None
        _fsync_parent_directory(destination.parent)
    except AdmissionError:
        raise
    except OSError as exc:
        raise AdmissionError(f"could not write admission document {destination}: {exc}") from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
    _require_regular_0644(destination, "admission output")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=3,
        metavar=("MANIFEST", "ARTIFACT", "PLATFORM"),
        required=True,
        help="exact manifest, artifact and platform; use '-' for the manifest-declared platform",
    )
    parser.add_argument("--provider", action="append", required=True, help="explicit measured provider token")
    parser.add_argument("--allow-missing", action="store_true", help="record absent artifacts as unavailable evidence")
    parser.add_argument(
        "--reviewed-surface",
        action="append",
        default=[],
        metavar="SURFACE",
        help="explicitly attest legal review for code, checkpoint and backbone (all required; repeatable)",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="replace an existing regular output; default is atomic no-clobber",
    )
    args = parser.parse_args(argv)
    try:
        inputs = [
            CandidateInput(Path(manifest), Path(artifact), None if platform == "-" else platform)
            for manifest, artifact, platform in args.candidate
        ]
        document = generate_admission(
            args.protocol,
            inputs,
            args.provider,
            allow_missing=args.allow_missing,
            reviewed_surfaces=args.reviewed_surface,
        )
        write_admission(args.output, document, replace=args.replace)
        print(json.dumps({"output": str(args.output.resolve()), "candidate_count": len(document["candidates"])}, sort_keys=True))
        return 0
    except (AdmissionError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"p25_5 admission: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
