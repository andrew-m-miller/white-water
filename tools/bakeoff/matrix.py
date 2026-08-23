#!/usr/bin/env python3
"""Deterministic Phase 2.5 matrix planning.

This module owns only selection admission and expansion.  It does not read or write runner
state, inspect artifacts, execute inference, or emit reports.  The protocol and corpus objects
are passed in by the caller so their already-frozen order remains the source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from numbers import Real
from typing import Any, Mapping, Sequence

try:
    from . import geometry
except ImportError:  # pragma: no cover - supports direct air-gapped invocation
    import geometry  # type: ignore

from .validator import canonical_sha256


class MatrixFailure(ValueError):
    """Stable, reportable matrix-planning failure."""

    def __init__(self, kind: str, message: str):
        self.kind = kind
        self.reason = kind
        self.failure_type = "matrix_failure"
        self.message = message
        super().__init__(f"{kind}: {message}")


@dataclass(frozen=True, order=True)
class CellKey:
    """The six identity axes of one planned bake-off cell."""

    candidate: str
    shot: str
    conditioning: str
    cap: str
    provider: str
    host_load: str

    def as_dict(self) -> dict[str, str]:
        return {
            "candidate": self.candidate,
            "shot": self.shot,
            "conditioning": self.conditioning,
            "cap": self.cap,
            "provider": self.provider,
            "host_load": self.host_load,
        }


@dataclass(frozen=True)
class MatrixPlan:
    """A signed selector and its exact deterministic Cartesian expansion."""

    selector: dict[str, Any]
    cells: tuple[CellKey, ...]
    excluded_candidate_ids: tuple[str, ...]

    @property
    def matrix_sha256(self) -> str:
        return self.selector["matrix_sha256"]


_SELECTION_KEYS = {
    "candidate_ids",
    "shot_ids",
    "conditioning_tokens",
    "cap_tokens",
    "providers",
}
_HOST_ORDER = {"idle": 0, "live_flame": 1, "not_applicable": 2}


def _fail(kind: str, message: str) -> None:
    raise MatrixFailure(kind, message)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        _fail("invalid_value", f"{path} must be a non-empty string")
    return value


def _has_duplicates(values: Sequence[Any]) -> bool:
    return any(value == earlier for index, value in enumerate(values) for earlier in values[:index])


def _ordered_ids(
    requested: Any, available: Sequence[str], path: str
) -> tuple[str, ...]:
    if not _is_sequence(requested) or not requested:
        _fail("empty_selection", f"{path} must be a non-empty sequence")
    requested_ids = []
    for index, value in enumerate(requested):
        requested_ids.append(_nonempty_string(value, f"{path}[{index}]"))
    if _has_duplicates(requested_ids):
        _fail("duplicate_selection", f"{path} contains duplicate values")
    unknown = [value for value in requested_ids if value not in available]
    if unknown:
        _fail("unknown_selection", f"{path} contains unknown value {unknown[0]!r}")
    selected = tuple(value for value in available if value in set(requested_ids))
    if not selected:
        _fail("empty_selection", f"{path} selects no available values")
    return selected


def _candidate_selection(
    requested: Any, eligible: Sequence[str], excluded: Sequence[str]
) -> tuple[str, ...]:
    path = "selections.candidate_ids"
    if not _is_sequence(requested) or not requested:
        _fail("empty_selection", f"{path} must be a non-empty sequence")
    requested_ids = [_nonempty_string(value, f"{path}[{index}]") for index, value in enumerate(requested)]
    if _has_duplicates(requested_ids):
        _fail("duplicate_selection", f"{path} contains duplicate values")
    for candidate_id in requested_ids:
        if candidate_id in excluded:
            _fail("excluded_candidate", f"{candidate_id!r} is excluded and cannot be selected")
        if candidate_id not in eligible:
            _fail("unknown_candidate", f"{candidate_id!r} is not an eligible candidate")
    requested_set = set(requested_ids)
    return tuple(candidate_id for candidate_id in eligible if candidate_id in requested_set)


def _measurement_candidate_selection(
    requested: Any, measurable: Sequence[str], unavailable: Sequence[str]
) -> tuple[str, ...]:
    """Select v2 candidates by technical measurement admission, not shipping status."""

    path = "selections.candidate_ids"
    if not _is_sequence(requested) or not requested:
        _fail("empty_selection", f"{path} must be a non-empty sequence")
    requested_ids = [_nonempty_string(value, f"{path}[{index}]") for index, value in enumerate(requested)]
    if _has_duplicates(requested_ids):
        _fail("duplicate_selection", f"{path} contains duplicate values")
    for candidate_id in requested_ids:
        if candidate_id in unavailable:
            _fail(
                "unavailable_candidate",
                f"{candidate_id!r} has measurement_status=unavailable and cannot be selected",
            )
        if candidate_id not in measurable:
            _fail("unknown_candidate", f"{candidate_id!r} is not a measurable candidate")
    requested_set = set(requested_ids)
    return tuple(candidate_id for candidate_id in measurable if candidate_id in requested_set)


def _protocol_ids(protocol: Mapping[str, Any], key: str, value_key: str | None = None) -> list[str]:
    values = protocol.get(key)
    if not _is_sequence(values) or not values:
        _fail("protocol_shape", f"protocol.{key} must be a non-empty sequence")
    result = []
    for index, value in enumerate(values):
        if value_key is None:
            result.append(_nonempty_string(value, f"protocol.{key}[{index}]"))
        else:
            if not isinstance(value, Mapping):
                _fail("protocol_shape", f"protocol.{key}[{index}] must be an object")
            result.append(_nonempty_string(value.get(value_key), f"protocol.{key}[{index}].{value_key}"))
    if len(result) != len(set(result)):
        _fail("protocol_shape", f"protocol.{key} contains duplicate values")
    return result


def _corpus_shot_order(corpus: Mapping[str, Any]) -> tuple[str, ...]:
    partitions = corpus.get("partitions")
    if not _is_sequence(partitions) or not partitions:
        _fail("corpus_shape", "corpus.partitions must be a non-empty sequence")
    result: list[str] = []
    for partition_index, partition in enumerate(partitions):
        if not isinstance(partition, Mapping):
            _fail("corpus_shape", f"corpus.partitions[{partition_index}] must be an object")
        shots = partition.get("shots")
        if not _is_sequence(shots):
            _fail("corpus_shape", f"corpus.partitions[{partition_index}].shots must be a sequence")
        for shot_index, shot in enumerate(shots):
            if not isinstance(shot, Mapping):
                _fail("corpus_shape", f"corpus shot {partition_index}/{shot_index} must be an object")
            shot_id = _nonempty_string(shot.get("id"), f"corpus shot {partition_index}/{shot_index}.id")
            if shot_id in result:
                _fail("corpus_shape", f"corpus contains duplicate shot {shot_id!r}")
            result.append(shot_id)
    if not result:
        _fail("corpus_shape", "corpus contains no shots")
    return tuple(result)


def _corpus_shot_map(corpus: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Return the validated metadata needed for candidate-specific geometry admission."""

    result: dict[str, Mapping[str, Any]] = {}
    partitions = corpus.get("partitions")
    if not _is_sequence(partitions):
        _fail("corpus_shape", "corpus.partitions must be a non-empty sequence")
    for partition_index, partition in enumerate(partitions):
        if not isinstance(partition, Mapping) or not _is_sequence(partition.get("shots")):
            _fail("corpus_shape", f"corpus.partitions[{partition_index}].shots must be a sequence")
        for shot_index, shot in enumerate(partition["shots"]):
            if not isinstance(shot, Mapping):
                _fail("corpus_shape", f"corpus shot {partition_index}/{shot_index} must be an object")
            shot_id = _nonempty_string(shot.get("id"), f"corpus shot {partition_index}/{shot_index}.id")
            result[shot_id] = shot
    return result


def _candidate_orders(
    protocol: Mapping[str, Any], candidate_entries: Any
) -> tuple[list[str], list[str], dict[str, Mapping[str, Any]]]:
    protocol_order = _protocol_ids(protocol, "candidate_ids", "id")
    if not _is_sequence(candidate_entries) or not candidate_entries:
        _fail("candidate_shape", "candidate entries must be a non-empty sequence")
    entries: dict[str, Mapping[str, Any]] = {}
    for index, entry in enumerate(candidate_entries):
        if not isinstance(entry, Mapping):
            _fail("candidate_shape", f"candidate_entries[{index}] must be an object")
        candidate_id = _nonempty_string(entry.get("candidate_id"), f"candidate_entries[{index}].candidate_id")
        if candidate_id in entries:
            _fail("duplicate_candidate", f"candidate {candidate_id!r} appears more than once")
        if candidate_id not in protocol_order:
            _fail("unknown_candidate", f"candidate {candidate_id!r} is absent from protocol")
        status = entry.get("status")
        if status not in {"eligible", "excluded"}:
            _fail("candidate_status", f"candidate {candidate_id!r} has invalid status")
        entries[candidate_id] = entry
    eligible = [candidate for candidate in protocol_order if entries.get(candidate, {}).get("status") == "eligible"]
    excluded = [candidate for candidate in protocol_order if entries.get(candidate, {}).get("status") == "excluded"]
    if not eligible:
        _fail("no_eligible_candidates", "candidate entries contain no eligible candidate")
    return eligible, excluded, entries


def _measurement_candidate_orders(
    protocol: Mapping[str, Any], candidate_entries: Any,
) -> tuple[list[str], list[str], dict[str, Mapping[str, Any]]]:
    """Return v2 measurable/unavailable order while preserving shipping status separately."""

    protocol_order = _protocol_ids(protocol, "candidate_ids", "id")
    role_map = {
        entry["id"]: entry.get("role")
        for entry in protocol.get("candidate_ids", [])
        if isinstance(entry, Mapping) and isinstance(entry.get("id"), str)
    }
    if not _is_sequence(candidate_entries) or not candidate_entries:
        _fail("candidate_shape", "candidate entries must be a non-empty sequence")
    entries: dict[str, Mapping[str, Any]] = {}
    for index, entry in enumerate(candidate_entries):
        if not isinstance(entry, Mapping):
            _fail("candidate_shape", f"candidate_entries[{index}] must be an object")
        candidate_id = _nonempty_string(entry.get("candidate_id"), f"candidate_entries[{index}].candidate_id")
        if candidate_id in entries:
            _fail("duplicate_candidate", f"candidate {candidate_id!r} appears more than once")
        if candidate_id not in protocol_order:
            _fail("unknown_candidate", f"candidate {candidate_id!r} is absent from protocol")
        if entry.get("status") not in {"eligible", "excluded"}:
            _fail("candidate_status", f"candidate {candidate_id!r} has invalid status")
        measurement_status = entry.get("measurement_status")
        if measurement_status not in {"measurable", "unavailable"}:
            _fail("measurement_status", f"candidate {candidate_id!r} has invalid measurement_status")
        if entry.get("status") == "eligible" and measurement_status != "measurable":
            _fail("candidate_status", f"shipping-eligible candidate {candidate_id!r} must be measurable")
        if entry.get("status") == "eligible" and role_map.get(candidate_id) != "shipping-candidate":
            _fail("candidate_role", f"validation-baseline candidate {candidate_id!r} cannot be shipping-eligible")
        entries[candidate_id] = entry
    measurable = [
        candidate for candidate in protocol_order
        if entries.get(candidate, {}).get("measurement_status") == "measurable"
    ]
    unavailable = [
        candidate for candidate in protocol_order
        if entries.get(candidate, {}).get("measurement_status") == "unavailable"
    ]
    if not measurable:
        _fail("no_measurable_candidates", "candidate entries contain no measurable candidate")
    return measurable, unavailable, entries


def _provider_selection(
    protocol: Mapping[str, Any], requested: Any, profile: str, environment: str
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    provider_entries = protocol.get("providers")
    if not _is_sequence(provider_entries) or not provider_entries:
        _fail("protocol_shape", "protocol.providers must be a non-empty sequence")
    provider_map: dict[str, Mapping[str, Any]] = {}
    provider_order: list[str] = []
    for index, provider in enumerate(provider_entries):
        if not isinstance(provider, Mapping):
            _fail("protocol_shape", f"protocol.providers[{index}] must be an object")
        token = _nonempty_string(provider.get("token"), f"protocol.providers[{index}].token")
        if token in provider_map:
            _fail("protocol_shape", f"protocol.providers contains duplicate {token!r}")
        _nonempty_string(provider.get("environment"), f"protocol.providers[{index}].environment")
        provider_map[token] = provider
        provider_order.append(token)
    if not _is_sequence(requested) or not requested:
        _fail("empty_selection", "selections.providers must be a non-empty sequence")
    requested_map: dict[str, Any] = {}
    for index, entry in enumerate(requested):
        if not isinstance(entry, Mapping) or set(entry) != {"token", "host_loads"}:
            _fail("provider_selection", f"selections.providers[{index}] must contain token and host_loads only")
        token = _nonempty_string(entry.get("token"), f"selections.providers[{index}].token")
        if token in requested_map:
            _fail("duplicate_selection", f"selections.providers contains duplicate {token!r}")
        if token not in provider_map:
            _fail("unknown_selection", f"selections.providers contains unknown provider {token!r}")
        requested_map[token] = entry.get("host_loads")
    normalized: list[dict[str, Any]] = []
    flattened: list[tuple[str, str]] = []
    selected_environments = set()
    for token in provider_order:
        if token not in requested_map:
            continue
        provider = provider_map[token]
        selected_environments.add(provider.get("environment"))
        host_loads = requested_map[token]
        if not _is_sequence(host_loads) or not host_loads:
            _fail("empty_selection", f"host loads for {token!r} must be non-empty")
        if _has_duplicates(host_loads):
            _fail("duplicate_selection", f"host loads for {token!r} contain duplicates")
        if token != "cuda":
            if list(host_loads) != ["not_applicable"]:
                _fail("host_load", f"{token} may select only not_applicable")
            ordered_hosts = ["not_applicable"]
        else:
            unknown_hosts = [host for host in host_loads if host not in ("idle", "live_flame")]
            if unknown_hosts:
                _fail("host_load", f"CUDA has invalid host load {unknown_hosts[0]!r}")
            if any(not isinstance(host, str) for host in host_loads):
                _fail("host_load", "CUDA host loads must be strings")
            if profile == "final" and set(host_loads) != {"idle", "live_flame"}:
                _fail("host_load", "final CUDA selection requires idle and live_flame")
            ordered_hosts = sorted(host_loads, key=lambda host: _HOST_ORDER[host])
        normalized.append({"token": token, "host_loads": ordered_hosts})
        flattened.extend((token, host) for host in ordered_hosts)
    if not normalized:
        _fail("empty_selection", "selections.providers selects no providers")
    if len(selected_environments) != 1 or next(iter(selected_environments)) != environment:
        _fail("environment", "selected providers must share the requested environment")
    return normalized, flattened


def _validate_provider_caps(
    protocol: Mapping[str, Any], providers: Sequence[Mapping[str, Any]], caps: Sequence[str]
) -> None:
    provider_map = {provider["token"]: provider for provider in protocol["providers"]}
    for provider in providers:
        supported = provider_map[provider["token"]].get("cap_tokens", [])
        for cap in caps:
            if cap not in supported:
                _fail("provider_cap", f"provider {provider['token']!r} does not support cap {cap!r}")


def _validate_candidate_constraints(
    protocol: Mapping[str, Any],
    candidate_ids: Sequence[str],
    candidate_entries: Mapping[str, Mapping[str, Any]],
    shot_map: Mapping[str, Mapping[str, Any]],
    cap_tokens: Sequence[str],
    providers: Sequence[Mapping[str, Any]],
) -> None:
    """Reject matrix selections that would create an unsupported candidate row."""

    raw_constraints = protocol.get("candidate_constraints", [])
    if not _is_sequence(raw_constraints):
        # v1 and small hand-built protocol fixtures predate the v2 constraint field.
        return
    constraints: dict[str, Mapping[str, Any]] = {}
    for index, constraint in enumerate(raw_constraints):
        if not isinstance(constraint, Mapping):
            _fail("protocol_shape", f"protocol.candidate_constraints[{index}] must be an object")
        candidate_id = _nonempty_string(
            constraint.get("candidate_id"), f"protocol.candidate_constraints[{index}].candidate_id",
        )
        if candidate_id in constraints:
            _fail("protocol_shape", f"protocol.candidate_constraints contains duplicate {candidate_id!r}")
        constraints[candidate_id] = constraint

    cap_map = {
        cap["token"]: cap
        for cap in protocol.get("analysis_caps", [])
        if isinstance(cap, Mapping) and isinstance(cap.get("token"), str)
    }
    provider_tokens = [provider["token"] for provider in providers]
    for candidate_id in candidate_ids:
        candidate_entry = candidate_entries[candidate_id]
        measurement_providers = candidate_entry.get("measurement_providers")
        constraint = constraints.get(candidate_id)
        if constraint is not None and measurement_providers is None:
            _fail(
                "provider_unavailable",
                f"candidate {candidate_id!r} has no provider-specific technical measurement evidence",
            )
        if measurement_providers is not None:
            if not _is_sequence(measurement_providers) or not measurement_providers:
                _fail("measurement_provider", f"candidate {candidate_id!r} has no measurable providers")
            for provider_token in provider_tokens:
                if provider_token not in measurement_providers:
                    _fail(
                        "provider_unavailable",
                        f"candidate {candidate_id!r} is not technically measurable on provider {provider_token!r}",
                    )

        if constraint is None:
            continue
        allowed_providers = constraint.get("providers", [])
        allowed_caps = constraint.get("cap_tokens", [])
        for provider_token in provider_tokens:
            if provider_token not in allowed_providers:
                _fail(
                    "candidate_capability",
                    f"candidate {candidate_id!r} does not support provider {provider_token!r}",
                )
        for cap_token in cap_tokens:
            if cap_token not in allowed_caps:
                _fail(
                    "candidate_capability",
                    f"candidate {candidate_id!r} does not support cap {cap_token!r}",
                )
            cap = cap_map.get(cap_token)
            required = constraint.get("required_geometry")
            if not isinstance(cap, Mapping) or not isinstance(required, Mapping):
                _fail("candidate_capability", f"candidate {candidate_id!r} has an invalid geometry constraint")
            lattice = cap.get("lattice")
            if not isinstance(lattice, Mapping) or any(
                lattice.get(field) != required.get(field)
                for field in ("analysis_width", "analysis_height", "canonical_aspect_ratio")
            ):
                _fail("candidate_capability", f"candidate {candidate_id!r} disagrees with cap {cap_token!r} lattice")
            for shot_id, shot in shot_map.items():
                try:
                    analysis_width, analysis_height = geometry.analysis_dimensions(
                        shot["width"], shot["height"], shot["pixel_aspect_ratio"],
                        cap["decimal_megapixels"],
                    )
                    canonical_aspect = (
                        float(shot["width"]) * float(shot["pixel_aspect_ratio"])
                    ) / float(shot["height"])
                except (KeyError, TypeError, ValueError, ZeroDivisionError, geometry.GeometryFailure) as exc:
                    _fail("candidate_geometry", f"candidate {candidate_id!r} cannot compute shot {shot_id!r} geometry: {exc}")
                if (analysis_width, analysis_height) != (
                    required.get("analysis_width"), required.get("analysis_height"),
                ):
                    _fail(
                        "candidate_geometry",
                        f"candidate {candidate_id!r} requires {required.get('analysis_width')}x{required.get('analysis_height')} "
                        f"but shot {shot_id!r} computes {analysis_width}x{analysis_height}",
                    )
                if required.get("canonical_aspect_ratio") == "16:9" and not math.isclose(
                    canonical_aspect, 16.0 / 9.0, rel_tol=0.0, abs_tol=1e-12,
                ):
                    _fail(
                        "candidate_geometry",
                        f"candidate {candidate_id!r} requires canonical 16:9 geometry but shot {shot_id!r} is {canonical_aspect!r}",
                    )


def _validate_final_coverage(
    corpus: Mapping[str, Any], shots: Sequence[str], caps: Sequence[str], providers: Sequence[Mapping[str, Any]], profile: str
) -> None:
    if profile != "final":
        return
    if "mp2" not in caps:
        _fail("final_coverage", "final matrix must select mp2")
    if not any(provider["token"] == "cuda" for provider in providers):
        _fail("final_coverage", "final matrix must select CUDA")
    selected = set(shots)
    targets = {
        "fhd-1920x1080-par1": (1920, 1080),
        "uhd-3840x2160-par1": (3840, 2160),
    }
    found = {case_id: False for case_id in targets}
    for partition in corpus["partitions"]:
        for shot in partition["shots"]:
            if shot.get("id") not in selected:
                continue
            case_id = shot.get("case_id")
            if case_id in targets and shot.get("width") == targets[case_id][0] and shot.get("height") == targets[case_id][1]:
                pixel_aspect = shot.get("pixel_aspect_ratio")
                if isinstance(pixel_aspect, Real) and not isinstance(pixel_aspect, bool) and math.isclose(float(pixel_aspect), 1.0, rel_tol=0.0, abs_tol=1e-12):
                    found[case_id] = True
    missing = [case_id for case_id, present in found.items() if not present]
    if missing:
        _fail("final_coverage", f"final matrix is missing {missing[0]} PAR1 shot")


def build_matrix(
    protocol: Mapping[str, Any],
    corpus: Mapping[str, Any],
    candidate_entries: Any,
    selections: Mapping[str, Any],
    profile: str,
    environment: str,
) -> MatrixPlan:
    """Validate explicit selections and expand their exact six-axis Cartesian product."""

    if profile not in {"smoke", "screen", "final"}:
        _fail("profile", f"unknown profile {profile!r}")
    environment = _nonempty_string(environment, "environment")
    if not isinstance(selections, Mapping) or set(selections) != _SELECTION_KEYS:
        _fail("selection_shape", "selections must contain exactly the five explicit matrix axes")

    if protocol.get("protocol_id") == "whitewater-p25-v2":
        measurable, unavailable, candidate_entry_map = _measurement_candidate_orders(protocol, candidate_entries)
        candidate_ids = _measurement_candidate_selection(
            selections["candidate_ids"], measurable, unavailable,
        )
        omitted_candidate_ids = unavailable
    else:
        eligible, excluded, candidate_entry_map = _candidate_orders(protocol, candidate_entries)
        candidate_ids = _candidate_selection(selections["candidate_ids"], eligible, excluded)
        omitted_candidate_ids = excluded
    shot_ids = _ordered_ids(selections["shot_ids"], _corpus_shot_order(corpus), "selections.shot_ids")
    conditioning_ids = _protocol_ids(protocol, "conditioning", "token")
    conditioning_tokens = _ordered_ids(
        selections["conditioning_tokens"], conditioning_ids, "selections.conditioning_tokens"
    )
    cap_ids = _protocol_ids(protocol, "analysis_caps", "token")
    cap_tokens = _ordered_ids(selections["cap_tokens"], cap_ids, "selections.cap_tokens")
    providers, provider_loads = _provider_selection(protocol, selections["providers"], profile, environment)
    corpus_shot_map = _corpus_shot_map(corpus)
    _validate_candidate_constraints(
        protocol,
        candidate_ids,
        candidate_entry_map,
        {shot_id: corpus_shot_map[shot_id] for shot_id in shot_ids},
        cap_tokens,
        providers,
    )
    _validate_provider_caps(protocol, providers, cap_tokens)
    _validate_final_coverage(corpus, shot_ids, cap_tokens, providers, profile)

    selector_payload = {
        "candidate_ids": list(candidate_ids),
        "shot_ids": list(shot_ids),
        "conditioning_tokens": list(conditioning_tokens),
        "cap_tokens": list(cap_tokens),
        "providers": providers,
    }
    selector = dict(selector_payload)
    selector["matrix_sha256"] = canonical_sha256(selector_payload)
    cells = tuple(
        CellKey(candidate, shot, conditioning, cap, provider, host_load)
        for candidate, shot, conditioning, cap, (provider, host_load) in product(
            candidate_ids, shot_ids, conditioning_tokens, cap_tokens, provider_loads
        )
    )
    return MatrixPlan(selector, cells, tuple(omitted_candidate_ids))


__all__ = ["CellKey", "MatrixFailure", "MatrixPlan", "build_matrix"]
