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

    eligible, excluded, _ = _candidate_orders(protocol, candidate_entries)
    candidate_ids = _candidate_selection(selections["candidate_ids"], eligible, excluded)
    shot_ids = _ordered_ids(selections["shot_ids"], _corpus_shot_order(corpus), "selections.shot_ids")
    conditioning_ids = _protocol_ids(protocol, "conditioning", "token")
    conditioning_tokens = _ordered_ids(
        selections["conditioning_tokens"], conditioning_ids, "selections.conditioning_tokens"
    )
    cap_ids = _protocol_ids(protocol, "analysis_caps", "token")
    cap_tokens = _ordered_ids(selections["cap_tokens"], cap_ids, "selections.cap_tokens")
    providers, provider_loads = _provider_selection(protocol, selections["providers"], profile, environment)
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
    return MatrixPlan(selector, cells, tuple(excluded))


__all__ = ["CellKey", "MatrixFailure", "MatrixPlan", "build_matrix"]
