#!/usr/bin/env python3
"""Deterministic, dependency-light gates for the P25-3D candidate manifests."""

from __future__ import annotations

from pathlib import Path
import sys

from artifact_workflow import ArtifactError, load_manifest  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
SEA_RAFT = ROOT / "models" / "sea-raft-m.json"
RAFT = ROOT / "models" / "raft-original.json"

SEA_SOURCE_COMMIT = "9137517ba24e628442aec097d3afe71d03503b75"
SEA_CHECKPOINT_REVISION = "ea21e467a7076978b251e09d55751fcce166c2f8"
SEA_ARTIFACT_SHA256 = "23cc2c850d3c116df193a24ff9ae7722d5635cd04e75dd8aeb20d7e13e4f59f1"
SEA_ARTIFACT_SIZE = 78840944
SEA_CHECKPOINT_SHA256 = "cb8cfbf14c5e0f6734b64add383708b7ff68cc6089a0007c67165d4761346102"
RAFT_SOURCE_COMMIT = "2888e15a51fa41140771d3f498ed8023cff098d1"
RAFT_CHECKPOINT_SHA256 = "fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_sea_raft() -> None:
    manifest = load_manifest(SEA_RAFT)
    _require(manifest["upstream"]["commit"] == SEA_SOURCE_COMMIT, "SEA source pin changed")
    _require(
        manifest["checkpoint"]["revision"] == SEA_CHECKPOINT_REVISION,
        "SEA checkpoint revision changed",
    )
    _require(
        manifest["checkpoint"]["sha256"] == SEA_CHECKPOINT_SHA256,
        "SEA checkpoint hash changed",
    )
    _require(
        manifest["export"]["sha256"] == SEA_ARTIFACT_SHA256,
        "SEA measured export hash changed",
    )
    _require(
        manifest["export"]["size_bytes"] == SEA_ARTIFACT_SIZE,
        "SEA measured export size changed",
    )
    _require(
        manifest["status"] == "host_probe_cpu_cuda_passed",
        "SEA measured host status changed",
    )
    _require(manifest["backbone"]["applicable"] is False, "SEA backbone must be not applicable")
    _require(manifest["backbone"]["identity"] is None, "SEA backbone identity must be null")

    licenses = manifest["licenses"]
    _require(licenses["code"]["license"] == "BSD-3-Clause", "SEA code licence changed")
    _require(licenses["code"]["commercial_use_permitted"] == "yes", "SEA code use is unaudited")
    _require(licenses["code"]["redistribution_permitted"] == "yes", "SEA code redistribution is unaudited")
    _require(
        SEA_SOURCE_COMMIT in licenses["code"]["audit"],
        "SEA code audit is not bound to the pinned source revision",
    )
    _require(licenses["checkpoint"]["license"] == "BSD-3-Clause", "SEA checkpoint licence changed")
    _require(licenses["checkpoint"]["commercial_use_permitted"] == "yes", "SEA checkpoint use is unaudited")
    _require(
        licenses["checkpoint"]["redistribution_permitted"] == "yes",
        "SEA checkpoint redistribution is unaudited",
    )
    _require(
        SEA_CHECKPOINT_REVISION in licenses["checkpoint"]["audit"],
        "SEA checkpoint audit is not bound to the pinned revision",
    )
    _require(licenses["backbone"]["license"] == "not-applicable", "SEA backbone licence surface changed")
    _require(
        "not a licence conclusion" in licenses["backbone"]["notes"],
        "SEA backbone audit must disclose that no framework licence is being concluded",
    )


def check_original_raft() -> None:
    manifest = load_manifest(RAFT)
    _require(manifest["candidate"]["id"] == "raft-original", "RAFT candidate id changed")
    _require(manifest["candidate"]["role"] == "validation-baseline", "RAFT role changed")
    _require(manifest["upstream"]["commit"] == RAFT_SOURCE_COMMIT, "RAFT source pin changed")
    _require(
        manifest["checkpoint"]["sha256"] == RAFT_CHECKPOINT_SHA256,
        "RAFT checkpoint hash changed",
    )
    _require(manifest["checkpoint"]["license"] == "unknown", "RAFT checkpoint terms were guessed")
    _require(
        manifest["licenses"]["checkpoint"]["commercial_use_permitted"] == "unknown",
        "RAFT checkpoint commercial-use terms were guessed",
    )
    _require(
        manifest["licenses"]["checkpoint"]["redistribution_permitted"] == "unknown",
        "RAFT checkpoint redistribution terms were guessed",
    )
    _require(
        manifest["status"] == "provenance_pinned_export_pending",
        "RAFT pending status changed",
    )
    _require(manifest["validation"]["status"] == "pending", "RAFT validation status changed")
    _require(manifest["export"]["sha256"] is None, "RAFT pending export must not claim bytes")
    _require(manifest["export"]["size_bytes"] is None, "RAFT pending export must not claim size")
    _require(
        manifest["validation"]["observed"]["reason_type"]
        == "checkpoint_terms_unresolved_and_export_not_run",
        "RAFT pending reason must remain typed",
    )


def main() -> int:
    check_sea_raft()
    check_original_raft()
    print("P25-3D SEA-RAFT audit and original-RAFT provenance manifests: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, ArtifactError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"p25_3_candidate_manifest_tests.py: error: {exc}", file=sys.stderr)
        raise SystemExit(1)
