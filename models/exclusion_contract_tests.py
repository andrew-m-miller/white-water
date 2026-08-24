#!/usr/bin/env python3
"""Focused tests for the shared typed exclusion contract."""

from __future__ import annotations

import copy

from exclusion_contract import (  # type: ignore  # pylint: disable=wrong-import-position
    ExclusionContractError,
    validate_exclusion_contract,
)


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

    # Keep this shared test independent of candidate worktrees.  The WAFT-specific checker
    # covers the real provenance record and its migration from the legacy reasons array.
    waft_shaped = {
        "status": "excluded",
        "candidate": {"id": "waft-twins"},
        "exclusion": {"reason_code": "checkpoint_license_terms_unknown"},
    }
    validate_exclusion_contract(waft_shaped)
    legacy_shaped = copy.deepcopy(waft_shaped)
    legacy_shaped["exclusion"].pop("reason_code")
    legacy_shaped["exclusion"]["reasons"] = ["checkpoint_terms_unavailable"]
    expect_failure("legacy reasons array", legacy_shaped)

    print("shared exclusion contract tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
