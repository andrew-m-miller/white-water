#!/usr/bin/env python3
"""Focused tests for the shared typed exclusion contract."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from exclusion_contract import (  # type: ignore  # pylint: disable=wrong-import-position
    ExclusionContractError,
    validate_exclusion_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def expect_failure(label: str, manifest: dict) -> None:
    try:
        validate_exclusion_contract(manifest)
    except ExclusionContractError:
        return
    raise AssertionError(f"invalid exclusion contract unexpectedly passed: {label}")


def main() -> int:
    excluded = {
        "status": "excluded",
        "exclusion": {"reason_code": "checkpoint_license_terms_unknown"},
    }
    validate_exclusion_contract(excluded)

    missing = {"status": "excluded"}
    expect_failure("excluded record without reason", missing)

    note_only = {
        "status": "excluded",
        "notes": ["admission_status=excluded_checkpoint_license_terms_unknown"],
    }
    expect_failure("note sentinel is not a reason", note_only)

    unknown = copy.deepcopy(excluded)
    unknown["exclusion"]["reason_code"] = "checkpoint_terms_unavailable"
    expect_failure("unknown reason code", unknown)

    nonexcluded = {
        "status": "export_validated",
        "exclusion": {"reason_code": "checkpoint_license_terms_unknown"},
    }
    expect_failure("non-excluded record with exclusion", nonexcluded)

    technical = {
        "status": "excluded",
        "validation": {"status": "failed"},
        "exclusion": {"reason_code": "export_or_operator_failure"},
    }
    validate_exclusion_contract(technical)

    # The WAFT provenance record is delivered by its independent candidate workstream.  When
    # that optional record is present, exercise its migration too; the shared contract itself
    # remains fully covered above on this branch.
    waft_path = ROOT / "models" / "waft-twins.json"
    if waft_path.exists():
        waft = json.loads(waft_path.read_text(encoding="utf-8"))
        validate_exclusion_contract(waft)
        old_waft = copy.deepcopy(waft)
        old_waft["exclusion"].pop("reason_code")
        old_waft["exclusion"]["reasons"] = ["checkpoint_terms_unavailable"]
        expect_failure("WAFT legacy reasons array", old_waft)

    print("shared exclusion contract tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
