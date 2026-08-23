#!/usr/bin/env python3
"""Deterministic gate for the WAFT/Twins provenance-only exclusion record.

This check intentionally performs no network access and does not reuse the artifact-v1
manifest.  P25-1 requires exact checkpoint bytes for every artifact row; this record instead
keeps the candidate's exact source lead and the unresolved checkpoint/backbone evidence typed
without making tensor, export, or qualification claims.
"""

from __future__ import annotations

from pathlib import Path
import sys
import importlib.util

from exclusion_contract import validate_exclusion_contract

VALIDATOR_PATH = Path(__file__).resolve().parents[1] / "tools" / "bakeoff" / "validator.py"
VALIDATOR_SPEC = importlib.util.spec_from_file_location("_whitewater_waft_validator", VALIDATOR_PATH)
if VALIDATOR_SPEC is None or VALIDATOR_SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"could not load validator: {VALIDATOR_PATH}")
VALIDATOR = importlib.util.module_from_spec(VALIDATOR_SPEC)
VALIDATOR_SPEC.loader.exec_module(VALIDATOR)
ArtifactError = ValueError


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models" / "waft-twins.json"
SCHEMA = ROOT / "models" / "waft-exclusion-v1.schema.json"
EXPECTED_COMMIT = "b152ff1cad1af8c185ee7b141997c48ff3334c87"
EXPECTED_MODEL_ZOO = "https://drive.google.com/drive/folders/1joBWKGoH2RUdCgcge8Tz2osOHcQUX5m_"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactError(message)


def main() -> int:
    manifest = VALIDATOR.load_json(MANIFEST)
    schema = VALIDATOR.load_json(SCHEMA)
    VALIDATOR.validate(manifest, schema)
    validate_exclusion_contract(manifest)
    require(manifest["candidate"]["id"] == "waft-twins", "unexpected WAFT candidate id")
    require(
        manifest["status"] == "excluded",
        "WAFT row must remain an exclusion until exact weights are identified",
    )
    require(manifest["upstream"]["commit"] == EXPECTED_COMMIT, "WAFT source revision changed")
    require(
        manifest["upstream"]["repository"] == "https://github.com/princeton-vl/WAFT.git",
        "WAFT source repository changed",
    )
    require(manifest["upstream"]["license"] == "BSD-3-Clause", "source licence audit changed")
    checkpoint = manifest["checkpoint"]
    require(checkpoint["repository"] == EXPECTED_MODEL_ZOO, "model-zoo folder changed")
    archive = checkpoint["observed_public_object"]
    require(archive["size_bytes"] == 3702705327, "observed a2.zip size changed")
    require(
        archive["sha256"] == "23282e0bf25e29e182ccedba8dc11969654c0658d06407edfc3932663109f62b",
        "a2.zip SHA256 changed",
    )
    require(
        checkpoint["identity_status"] == "resolved",
        "selected WAFT checkpoint must remain file-identity pinned",
    )
    selected = checkpoint["selected_member"]
    require(selected["path"] == "waftv2-ckpts/twins/zero-shot.pth", "selected member changed")
    require(selected["size_bytes"] == 544230582, "selected member size changed")
    require(
        selected["sha256"] == "f750cd15281fc30de477723438ff4a67fe1591deac4ab0eb9b366e27c827e070",
        "selected member SHA256 changed",
    )
    inventory = checkpoint["archive_inventory"]
    require(len(inventory) == 16, "A2 archive inventory is incomplete")
    require(
        {item["path"] for item in inventory if item["path"].startswith("waftv2-ckpts/twins/")}
        == {
            "waftv2-ckpts/twins/",
            "waftv2-ckpts/twins/kitti.pth",
            "waftv2-ckpts/twins/spring.pth",
            "waftv2-ckpts/twins/zero-shot.pth",
            "waftv2-ckpts/twins/sintel.pth",
        },
        "Twins archive inventory changed",
    )
    require(manifest["license_surfaces"]["checkpoint"]["license"] == "unknown",
            "checkpoint licence must not be inferred")
    require(manifest["license_surfaces"]["checkpoint"]["commercial_use_permitted"] == "unknown",
            "checkpoint commercial-use verdict must remain unknown")
    require(manifest["license_surfaces"]["checkpoint"]["redistribution_permitted"] == "unknown",
            "checkpoint redistribution verdict must remain unknown")

    backbone = manifest["backbone"]
    require(backbone["identity_status"] == "bundled_in_checkpoint",
            "backbone packaging identity changed")
    require(
        backbone["revision"] == "9985cdd56ac6164db09e464008c512fb7b75228a"
        and backbone["checkpoint_sha256"] == "2fbe754f6e595bd07294f381302e0b3f1d449176977e35e4f30200fe4f3bcf97",
        "bundled timm backbone reference changed",
    )
    reference = backbone["reference_weight"]
    require(reference["size_bytes"] == 397122010, "reference timm weight size changed")
    require(reference["license"] == "Apache-2.0", "backbone licence audit changed")
    require(manifest["license_surfaces"]["backbone"]["license"] == "Apache-2.0",
            "backbone licence must remain Apache-2.0")
    require(manifest["license_surfaces"]["backbone"]["commercial_use_permitted"] == "yes",
            "backbone commercial-use verdict changed")
    require(manifest["license_surfaces"]["backbone"]["redistribution_permitted"] == "yes",
            "backbone redistribution verdict changed")
    verification = manifest["verification"]
    require(
        verification["state_dict_keys"] == 699
        and verification["encoder_backbone_keys"] == 380
        and verification["missing_keys"] == 0
        and verification["unexpected_keys"] == 0
        and verification["pretrained_initialization"] is False,
        "strict checkpoint-loading evidence changed",
    )
    require(
        manifest["exclusion"]["reason_code"] == "checkpoint_license_terms_unknown",
        "WAFT exclusion must be terms-scoped after checkpoint identity resolution",
    )
    require(manifest["exclusion"]["decision"] == "not_eligible_for_p25_bakeoff_or_shipping",
            "WAFT exclusion decision changed")
    print("WAFT/Twins provenance-only exclusion record: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ArtifactError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"check_waft_manifest.py: error: {exc}", file=sys.stderr)
        raise SystemExit(1)
