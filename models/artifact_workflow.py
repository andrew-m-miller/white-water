#!/usr/bin/env python3
"""Shared Phase 2.5 artifact-manifest and payload validation.

This module is deliberately dependency-light.  Exporters for individual model families own
their model import and graph construction, but provenance, manifest shape, platform identity,
and payload publication are one workflow.  Keeping those checks here prevents a candidate
export from quietly acquiring weaker hash or permission semantics than SEA-RAFT.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping

try:
    from .exclusion_contract import validate_exclusion_contract
except ImportError:  # Direct script imports keep the existing dependency-light entry points working.
    from exclusion_contract import validate_exclusion_contract


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "models" / "artifact-v1.schema.json"
PROTOCOL_PATH = ROOT / "bakeoff" / "protocol-v1.json"
EXPECTED_MODE = 0o644
_HEX64 = set("0123456789abcdef")

# The protocol validator is the dependency-free JSON Schema implementation already used by
# the air-gapped bake-off gate.  Load it under a private, unique name so this module also works
# when an exporter is invoked from an arbitrary current directory without shadowing another
# interpreter-global ``validator`` module.
_VALIDATOR_PATH = ROOT / "tools" / "bakeoff" / "validator.py"
_VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "_whitewater_p25_bakeoff_validator",
    _VALIDATOR_PATH,
)
if _VALIDATOR_SPEC is None or _VALIDATOR_SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"could not load bake-off validator: {_VALIDATOR_PATH}")
_VALIDATOR_MODULE = importlib.util.module_from_spec(_VALIDATOR_SPEC)
_VALIDATOR_SPEC.loader.exec_module(_VALIDATOR_MODULE)

ValidationError = _VALIDATOR_MODULE.ValidationError
canonical_sha256 = _VALIDATOR_MODULE.canonical_sha256
load_json = _VALIDATOR_MODULE.load_json
validate = _VALIDATOR_MODULE.validate


class ArtifactError(ValueError):
    """A stable failure for manifest, contract, or payload validation."""


def sha256_file(path: Path) -> str:
    """Hash a regular file without following a symlink supplied by the caller."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_regular_mode(path: Path, label: str) -> None:
    """Require an actual regular, exactly-0644 file.

    ``Path.is_file`` follows symlinks, and ``stat`` follows them too.  The Flame staging bug
    this gate exists to prevent is therefore easy to reintroduce unless the first operation is
    an ``lstat`` and the regular-file bit is checked explicitly.
    """

    try:
        info = path.lstat()
    except OSError as exc:
        raise ArtifactError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise ArtifactError(f"{label} must not be a symlink: {path}")
    if not stat.S_ISREG(info.st_mode):
        raise ArtifactError(f"{label} is not a regular file: {path}")
    mode = stat.S_IMODE(info.st_mode)
    if mode != EXPECTED_MODE:
        raise ArtifactError(
            f"{label} has mode {mode:04o}; expected 0644 so Flame can read the payload: {path}"
        )


def _path_is_relative_file(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactError(f"{path} must be a non-empty relative path")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ArtifactError(f"{path} must stay below the manifest directory")
    return value


def _sha(value: Any, path: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or len(value) != 64 or set(value) - _HEX64:
        raise ArtifactError(f"{path} must be a lowercase SHA256 or null")


def _positive_size(value: Any, path: str) -> None:
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
        raise ArtifactError(f"{path} must be a positive integer or null")


def environment_sha256(environment: Mapping[str, Any]) -> str:
    """Hash the export environment without its derived ``sha256`` member."""

    value = dict(environment)
    value.pop("sha256", None)
    return canonical_sha256(value)


def _canonical_channels(value: Any) -> str | list[str]:
    if isinstance(value, list):
        return "".join(value)
    return value


def _validate_input_pair(inputs: list[Mapping[str, Any]]) -> None:
    first = inputs[0]
    second = inputs[1]
    for field in ("dtype", "layout", "channels"):
        first_value = _canonical_channels(first[field]) if field == "channels" else first[field]
        second_value = _canonical_channels(second[field]) if field == "channels" else second[field]
        if first_value != second_value:
            raise ArtifactError(
                f"tensor_contract.inputs image1 and image2 differ in {field}: "
                f"{first_value!r} != {second_value!r}"
            )


def _canonical_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the P25-0 contract vocabulary represented by a detailed manifest.

    SEA-RAFT's original manifest called its post-crop vectors ``input_pixels``.  Its canonical
    ``caller-replication-crop`` policy explicitly crops the result to the unpadded extent, so
    this is equivalent to the frozen protocol's ``unpadded_analysis_pixels`` token. Keeping
    that spelling in the migrated manifest preserves the Phase 0B record while still rejecting
    genuinely incompatible candidates.
    """

    contract = manifest["tensor_contract"]
    inputs = contract["inputs"]
    output = contract["output"]
    units = output["units"]
    padding_policy = contract["padding"]["policy"]
    if units == "input_pixels" and padding_policy in {
        "caller-replication-crop",
        "caller-reflection-crop",
    }:
        units = "unpadded_analysis_pixels"

    channels = _canonical_channels(inputs[0]["channels"])
    return {
        "batch": contract["batch"],
        "input_dtype": inputs[0]["dtype"],
        "input_layout": inputs[0]["layout"],
        "input_channels": channels,
        "input_pair": [item["name"] for item in inputs],
        "output_dtype": output["dtype"],
        "output_layout": output["layout"],
        "output_channels": output["channels"],
        "output_direction": output["direction"],
        "output_units": units,
        # Keep both inputs in the canonical representation.  The frozen protocol names the
        # pair's shared dtype/layout/channel contract as scalars, but retaining each member
        # here makes a second-input drift visible instead of silently inheriting input[0].
        "input_contracts": [
            {
                "name": item["name"],
                "dtype": item["dtype"],
                "layout": item["layout"],
                "channels": _canonical_channels(item["channels"]),
            }
            for item in inputs
        ],
    }


def _validate_contract_compatibility(manifest: Mapping[str, Any], protocol: Mapping[str, Any]) -> None:
    expected = protocol["eligibility"]["required_tensor_contract"]
    actual = _canonical_contract(manifest)
    differences = [
        f"{key}: expected {expected.get(key)!r}, got {actual.get(key)!r}"
        for key in expected
        if actual.get(key) != expected.get(key)
    ]
    for index, item in enumerate(actual["input_contracts"]):
        for field, expected_key in (
            ("dtype", "input_dtype"),
            ("layout", "input_layout"),
            ("channels", "input_channels"),
        ):
            if item[field] != expected[expected_key]:
                differences.append(
                    f"input[{index}].{field}: expected {expected[expected_key]!r}, got {item[field]!r}"
                )
    if any(
        actual["input_contracts"][0][field] != actual["input_contracts"][1][field]
        for field in ("dtype", "layout", "channels")
    ):
        differences.append("image1 and image2 input tensor contracts differ")
    if differences:
        raise ArtifactError(
            "tensor contract is incompatible with frozen P25-0 contract (" + "; ".join(differences) + ")"
        )


def _validate_export_identity(manifest: Mapping[str, Any]) -> None:
    export = manifest["export"]
    root_platform = export["platform"]
    _path_is_relative_file(export["artifact"], "$.export.artifact")
    _sha(export["sha256"], "$.export.sha256")
    _positive_size(export["size_bytes"], "$.export.size_bytes")
    _sha(export["export_environment_sha256"], "$.export.export_environment_sha256")

    entries = export["platform_artifacts"]
    seen: set[str] = set()
    matching: list[Mapping[str, Any]] = []
    for index, entry in enumerate(entries):
        path = f"$.export.platform_artifacts[{index}]"
        platform = entry["platform"]
        if platform in seen:
            raise ArtifactError(f"{path}.platform duplicates platform {platform!r}")
        seen.add(platform)
        _path_is_relative_file(entry["artifact"], f"{path}.artifact")
        _sha(entry["sha256"], f"{path}.sha256")
        _positive_size(entry["size_bytes"], f"{path}.size_bytes")
        _sha(entry["export_environment_sha256"], f"{path}.export_environment_sha256")
        if "export_environment" in entry:
            environment = entry["export_environment"]
            if environment["sha256"] != environment_sha256(environment):
                raise ArtifactError(f"{path}.export_environment.sha256 does not match its fields")
            if environment["sha256"] != entry["export_environment_sha256"]:
                raise ArtifactError(f"{path}.export_environment disagrees with its hash")
        if platform == root_platform:
            matching.append(entry)
    if len(matching) != 1:
        raise ArtifactError(
            "$.export.platform must have exactly one matching platform_artifacts entry"
        )
    selected = matching[0]
    for field in ("artifact", "sha256", "size_bytes", "export_environment_sha256"):
        if selected[field] != export[field]:
            raise ArtifactError(
                f"$.export.{field} disagrees with platform entry {root_platform!r}"
            )

    status = manifest["status"]
    if status == "provenance_pinned_export_pending":
        if export["sha256"] is not None or export["size_bytes"] is not None:
            raise ArtifactError("pending export must not claim an artifact hash or size")
    elif status != "excluded" and (export["sha256"] is None or export["size_bytes"] is None):
        raise ArtifactError("validated export must record an artifact hash and size")


def _validate_numerical_validation(manifest: Mapping[str, Any]) -> None:
    validation = manifest["validation"]
    status = validation["status"]
    manifest_status = manifest["status"]
    expected_statuses = {
        "provenance_pinned_export_pending": {"pending"},
        "export_validated": {"passed"},
        "host_probe_pending": {"passed"},
        "host_probe_cpu_cuda_passed": {"passed"},
        # A candidate can be numerically validated yet excluded from admission for a
        # non-numerical gate such as unresolved checkpoint licensing.  Its pass evidence stays
        # intact while the manifest's top-level status and candidate role remain ``excluded``.
        "excluded": {"pending", "failed", "passed"},
    }
    if status not in expected_statuses[manifest_status]:
        raise ArtifactError(
            f"manifest status {manifest_status!r} is incoherent with numerical validation status {status!r}"
        )
    direction_threshold_fields = (
        "translation_pixels",
        "translation_x_fraction_min",
        "translation_abs_y_max",
    )
    configured_thresholds = [
        field for field in direction_threshold_fields if field in validation
    ]
    if configured_thresholds and len(configured_thresholds) != len(direction_threshold_fields):
        missing = [field for field in direction_threshold_fields if field not in validation]
        raise ArtifactError(
            "direction thresholds must be configured as a complete triplet; "
            f"missing {', '.join(missing)}"
        )
    if status != "passed":
        return
    required = ("identity", "directions", "shapes", "parity")
    missing = [name for name in required if name not in validation]
    if missing or validation.get("observed") is None:
        names = ", ".join(missing) if missing else "observed"
        raise ArtifactError(f"passed numerical validation is missing {names}")
    identity = validation["identity"]
    if identity["passed"] is not True:
        raise ArtifactError("passed numerical validation has identity.passed=false")
    identity_limit = validation.get("identity_median_epe_max")
    if identity_limit is not None and identity["median_epe_px"] > identity_limit:
        raise ArtifactError("passed numerical validation exceeds identity_median_epe_max")
    parity = validation["parity"]
    if parity["checked"] is not True:
        raise ArtifactError("passed numerical validation has parity.checked=false")
    parity_limits = (
        ("mean_abs", "onnx_pytorch_mean_abs_max"),
        ("p99_abs", "onnx_pytorch_p99_abs_max"),
        ("p999_abs", "onnx_pytorch_p999_abs_max"),
        ("max_abs", "onnx_pytorch_max_abs_max"),
    )
    for measured, limit_name in parity_limits:
        limit = validation.get(limit_name)
        if limit is not None and parity[measured] > limit:
            raise ArtifactError(f"passed numerical validation exceeds {limit_name}")
    directions = validation["directions"]
    minimum_primary = None
    maximum_cross = None
    if configured_thresholds:
        fraction = validation["translation_x_fraction_min"]
        if fraction < 0:
            raise ArtifactError("translation_x_fraction_min must be non-negative")
        minimum_primary = abs(validation["translation_pixels"]) * fraction
        maximum_cross = validation["translation_abs_y_max"]
    for name in ("forward", "reverse"):
        evidence = directions[name]
        sign = evidence["expected_sign"]
        axis = sign[-1]
        component = evidence["median_dx_px"] if axis == "x" else evidence["median_dy_px"]
        cross_component = evidence["median_dy_px"] if axis == "x" else evidence["median_dx_px"]
        if sign.startswith("positive_") and component <= 0:
            raise ArtifactError(f"{name} direction evidence contradicts expected {sign}")
        if sign.startswith("negative_") and component >= 0:
            raise ArtifactError(f"{name} direction evidence contradicts expected {sign}")
        if minimum_primary is not None:
            if sign.startswith("positive_") and component < minimum_primary:
                raise ArtifactError(
                    f"{name} primary motion {component} is below the configured minimum "
                    f"{minimum_primary}"
                )
            if sign.startswith("negative_") and component > -minimum_primary:
                raise ArtifactError(
                    f"{name} primary motion {component} is below the configured minimum "
                    f"{minimum_primary}"
                )
            if abs(cross_component) > maximum_cross:
                raise ArtifactError(
                    f"{name} transverse motion {cross_component} exceeds the configured maximum "
                    f"{maximum_cross}"
                )
    forward_sign = directions["forward"]["expected_sign"]
    reverse_sign = directions["reverse"]["expected_sign"]
    if forward_sign[-1] != reverse_sign[-1]:
        raise ArtifactError("forward and reverse direction evidence must use the same axis")
    if forward_sign.startswith("positive_") == reverse_sign.startswith("positive_"):
        raise ArtifactError("forward and reverse direction evidence must have opposite signs")


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any] | None = None,
) -> None:
    """Validate schema and cross-field invariants for any candidate manifest."""

    schema = load_json(SCHEMA_PATH)
    validate(manifest, schema)
    validate_exclusion_contract(manifest)
    if manifest["export_environment"]["sha256"] != environment_sha256(manifest["export_environment"]):
        raise ArtifactError("export_environment.sha256 does not match its canonical environment fields")
    if manifest["export"]["export_environment_sha256"] != manifest["export_environment"]["sha256"]:
        raise ArtifactError("export.export_environment_sha256 does not match export_environment.sha256")
    _validate_export_identity(manifest)
    _validate_numerical_validation(manifest)

    inputs = manifest["tensor_contract"]["inputs"]
    if [item["name"] for item in inputs] != ["image1", "image2"]:
        raise ArtifactError("tensor_contract.inputs must be ordered image1, image2")
    confidence_present = manifest["tensor_contract"]["confidence"]["present"]
    output_confidence = manifest["tensor_contract"]["output"].get("confidence", False)
    if bool(output_confidence) != bool(confidence_present):
        raise ArtifactError("confidence contract disagrees between output and confidence fields")
    _validate_input_pair(inputs)
    if protocol is not None:
        _validate_contract_compatibility(manifest, protocol)


def load_manifest(
    path: Path,
    *,
    check_file: bool = True,
    protocol_path: Path | None = None,
) -> dict[str, Any]:
    if check_file:
        require_regular_mode(path, "manifest")
    manifest = load_json(path)
    if not isinstance(manifest, dict):
        raise ArtifactError("manifest root must be an object")
    protocol = load_json(protocol_path or PROTOCOL_PATH) if protocol_path is not False else None
    validate_manifest(manifest, protocol=protocol)
    return manifest


def _selected_export(manifest: Mapping[str, Any], platform: str | None) -> Mapping[str, Any]:
    export = manifest["export"]
    selected_platform = platform or export["platform"]
    for entry in export["platform_artifacts"]:
        if entry["platform"] == selected_platform:
            return entry
    raise ArtifactError(f"manifest has no artifact for platform {selected_platform!r}")


def validate_artifact(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    artifact_path: Path | None = None,
    *,
    platform: str | None = None,
) -> None:
    """Validate one exact staged artifact against its platform entry."""

    selected = _selected_export(manifest, platform)
    if selected["sha256"] is None or selected["size_bytes"] is None:
        raise ArtifactError("cannot validate a pending export without hash and size")
    if artifact_path is None:
        artifact_path = manifest_path.parent / selected["artifact"]
    require_regular_mode(artifact_path, "artifact")
    actual_size = artifact_path.stat().st_size
    if actual_size != selected["size_bytes"]:
        raise ArtifactError(
            f"artifact size is {actual_size}, expected {selected['size_bytes']}: {artifact_path}"
        )
    actual_hash = sha256_file(artifact_path)
    if actual_hash != selected["sha256"]:
        raise ArtifactError(
            f"artifact SHA256 is {actual_hash}, expected {selected['sha256']}: {artifact_path}"
        )


def validate_all_artifacts(manifest: Mapping[str, Any], manifest_path: Path) -> None:
    """Validate every declared platform artifact that exists beside the manifest."""

    for entry in manifest["export"]["platform_artifacts"]:
        if entry["sha256"] is None:
            continue
        validate_artifact(
            manifest,
            manifest_path,
            manifest_path.parent / entry["artifact"],
            platform=entry["platform"],
        )


def publish_file(source: Path, destination: Path) -> None:
    """Atomically publish a non-secret artifact with the host-readable mode."""

    try:
        source.chmod(EXPECTED_MODE)
    except OSError as exc:
        raise ArtifactError(f"could not set published artifact mode to 0644: {source}") from exc
    os.replace(source, destination)


def update_platform_export(
    manifest: dict[str, Any],
    *,
    platform: str,
    artifact: Path,
    environment: Mapping[str, Any],
) -> None:
    """Record one platform's bytes while retaining hashes for other exporters."""

    env = dict(environment)
    env["sha256"] = environment_sha256(env)
    digest = sha256_file(artifact)
    size = artifact.stat().st_size
    entry = {
        "platform": platform,
        "artifact": artifact.name,
        "sha256": digest,
        "size_bytes": size,
        "mode": "0644",
        "export_environment_sha256": env["sha256"],
        "export_environment": env,
    }
    entries = [item for item in manifest["export"].get("platform_artifacts", []) if item["platform"] != platform]
    entries.append(entry)
    entries.sort(key=lambda item: item["platform"])
    export = manifest["export"]
    export["platform"] = platform
    export["artifact"] = artifact.name
    export["sha256"] = digest
    export["size_bytes"] = size
    export["mode"] = "0644"
    export["export_environment_sha256"] = env["sha256"]
    export["platform_artifacts"] = entries
    manifest["export_environment"] = env
    manifest["status"] = "host_probe_pending"


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Validate then write a manifest and restore its exact public-file mode."""

    validate_manifest(manifest, protocol=load_json(PROTOCOL_PATH))
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        temporary.chmod(EXPECTED_MODE)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
