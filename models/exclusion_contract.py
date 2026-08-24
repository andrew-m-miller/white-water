"""Shared typed exclusion contract for Phase 2.5 candidate records."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping


class ExclusionContractError(ValueError):
    """A manifest does not satisfy the shared exclusion contract."""


class ExclusionReason(str, Enum):
    """The deliberately small reason vocabulary shared by candidate records."""

    CHECKPOINT_LICENSE_TERMS_UNKNOWN = "checkpoint_license_terms_unknown"
    EXPORT_OR_OPERATOR_FAILURE = "export_or_operator_failure"


REASON_CODES = frozenset(reason.value for reason in ExclusionReason)


def validate_exclusion_contract(manifest: Mapping[str, Any]) -> None:
    """Require ``exclusion.reason_code`` exactly when the record is excluded.

    JSON Schema owns the detailed shape for each record type.  This helper owns the
    cross-field rule shared by artifact-v1 and the provenance-only WAFT record, so a
    free-form note or an untyped reason cannot stand in for an exclusion decision.
    """

    status = manifest.get("status")
    has_exclusion = "exclusion" in manifest
    if status == "excluded":
        exclusion = manifest.get("exclusion")
        if not isinstance(exclusion, Mapping):
            raise ExclusionContractError(
                "excluded record requires an exclusion object with reason_code"
            )
        reason_code = exclusion.get("reason_code")
        if not isinstance(reason_code, str) or reason_code not in REASON_CODES:
            raise ExclusionContractError(
                "excluded record requires a known typed exclusion.reason_code"
            )
        return
    if has_exclusion:
        raise ExclusionContractError(
            "non-excluded record must not contain an exclusion object"
        )


__all__ = [
    "ExclusionContractError",
    "ExclusionReason",
    "REASON_CODES",
    "validate_exclusion_contract",
]
