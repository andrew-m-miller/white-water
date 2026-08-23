#!/usr/bin/env python3
"""Strict, atomic resume state for the offline Phase 2.5 runner.

The state format is intentionally internal to the runner.  It binds an exact identity to an
ordered matrix plan, and each transition is committed by a same-directory fsync/replace.  A
public load treats an ``in_progress`` entry as interrupted work and atomically returns it to
``pending`` so the next attempt cannot silently claim completion.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping

from .matrix import CellKey, MatrixPlan
from .validator import canonical_sha256, load_json


class ResumeFailure(ValueError):
    """Stable, reportable resume-state failure."""

    def __init__(self, kind: str, message: str):
        self.kind = kind
        self.reason = kind
        self.failure_type = "resume_failure"
        self.message = message
        super().__init__(f"{kind}: {message}")


_STATE_KEYS = {"schema_version", "identity", "identity_sha256", "entries"}
_ENTRY_KEYS = {"cell", "state"}
_CELL_KEYS = {"candidate", "shot", "conditioning", "cap", "provider", "host_load"}
_STATES = {"pending", "in_progress", "complete"}


def _fail(kind: str, message: str) -> None:
    raise ResumeFailure(kind, message)


def _reject_nonfinite(value: Any, path: str = "$", seen: set[int] | None = None) -> None:
    """Reject non-JSON values, nonfinite numbers, non-string keys, and cycles."""

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail("nonfinite", f"{path} contains a nonfinite number")
        return
    if isinstance(value, Mapping):
        if not isinstance(value, dict):
            _fail("json_value", f"{path} must use a plain JSON object")
        if seen is None:
            seen = set()
        marker = id(value)
        if marker in seen:
            _fail("identity_shape", f"{path} contains a cycle")
        seen.add(marker)
        for key, child in value.items():
            if not isinstance(key, str):
                _fail("json_value", f"{path} contains a non-string object key")
            _reject_nonfinite(child, f"{path}.{key}", seen)
        seen.remove(marker)
        return
    if isinstance(value, list):
        seen = seen if seen is not None else set()
        marker = id(value)
        if marker in seen:
            _fail("identity_shape", f"{path} contains a cycle")
        seen.add(marker)
        for index, child in enumerate(value):
            _reject_nonfinite(child, f"{path}[{index}]", seen)
        seen.remove(marker)
        return
    _fail("json_value", f"{path} contains a non-JSON value")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("shape", f"{path} must be a JSON object")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        _fail("shape", f"{path} must be a non-empty string")
    return value


def _cell_from_json(value: Any, path: str) -> CellKey:
    mapping = _mapping(value, path)
    if set(mapping) != _CELL_KEYS:
        _fail("cell_shape", f"{path} must contain exactly the six CellKey fields")
    return CellKey(
        _string(mapping["candidate"], f"{path}.candidate"),
        _string(mapping["shot"], f"{path}.shot"),
        _string(mapping["conditioning"], f"{path}.conditioning"),
        _string(mapping["cap"], f"{path}.cap"),
        _string(mapping["provider"], f"{path}.provider"),
        _string(mapping["host_load"], f"{path}.host_load"),
    )


def _expected_cells(plan: MatrixPlan) -> tuple[CellKey, ...]:
    cells = tuple(plan.cells)
    if any(not isinstance(cell, CellKey) for cell in cells):
        _fail("plan_shape", "MatrixPlan contains a non-CellKey entry")
    if len(cells) != len(set(cells)):
        _fail("plan_shape", "MatrixPlan contains duplicate CellKeys")
    if not cells:
        _fail("plan_shape", "MatrixPlan contains no cells")
    return cells


def _validate_identity(identity: Any) -> dict[str, Any]:
    mapping = _mapping(identity, "identity")
    _reject_nonfinite(mapping)
    return dict(mapping)


def _validate_state_header(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the state fields needed before expanding a caller's plan."""

    if not isinstance(mapping["schema_version"], int) or isinstance(mapping["schema_version"], bool) or mapping["schema_version"] != 1:
        _fail("schema_version", "state schema_version must be integer 1")
    identity = _validate_identity(mapping["identity"])
    identity_sha256 = mapping["identity_sha256"]
    if not isinstance(identity_sha256, str) or len(identity_sha256) != 64 or identity_sha256 != identity_sha256.lower():
        _fail("identity_hash", "identity_sha256 must be a lowercase SHA256")
    try:
        int(identity_sha256, 16)
    except ValueError as exc:
        raise ResumeFailure("identity_hash", "identity_sha256 must be hexadecimal") from exc
    if canonical_sha256(identity) != identity_sha256:
        _fail("identity_hash", "identity_sha256 does not match identity")
    return identity


def _validate_result(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail("result_shape", f"{path} must be a JSON object")
    mapping = value
    _reject_nonfinite(mapping, path)
    return dict(mapping)


def _validate_state_shape(state: Any, plan: MatrixPlan) -> tuple[dict[str, Any], tuple[CellKey, ...]]:
    mapping = _mapping(state, "state")
    if set(mapping) != _STATE_KEYS:
        _fail("state_shape", "state must contain exactly schema_version, identity, identity_sha256, and entries")
    identity = _validate_state_header(mapping)
    entries = mapping["entries"]
    if not isinstance(entries, list) or len(entries) != len(plan.cells):
        _fail("cell_count", "state entries must contain exactly one entry per plan cell")
    expected = _expected_cells(plan)
    seen: set[CellKey] = set()
    actual: list[CellKey] = []
    for index, entry in enumerate(entries):
        entry_mapping = _mapping(entry, f"entries[{index}]")
        if not set(entry_mapping).issubset(_ENTRY_KEYS | {"result"}) or not _ENTRY_KEYS.issubset(entry_mapping):
            _fail("entry_shape", f"entries[{index}] must contain exactly cell and state, plus optional result")
        cell = _cell_from_json(entry_mapping["cell"], f"entries[{index}].cell")
        if cell in seen:
            _fail("duplicate_cell", f"entries[{index}] duplicates CellKey {cell!r}")
        seen.add(cell)
        actual.append(cell)
        status = entry_mapping["state"]
        if status not in _STATES:
            _fail("status", f"entries[{index}].state is not a permitted state")
        if status == "complete":
            if set(entry_mapping) != _ENTRY_KEYS | {"result"}:
                _fail("result_shape", f"complete entries[{index}] must carry only a result object")
            if "result" not in entry_mapping:
                _fail("result_shape", f"complete entries[{index}] require a result object")
            _validate_result(entry_mapping["result"], f"entries[{index}].result")
        elif set(entry_mapping) != _ENTRY_KEYS:
            _fail("result_shape", f"only complete entries may carry a result")
    if tuple(actual) != expected:
        _fail("cell_order", "state entries must match MatrixPlan CellKeys exactly in plan order")
    return dict(mapping), tuple(actual)


def _validate_expected_identity(state: Mapping[str, Any], expected_identity: Mapping[str, Any]) -> None:
    expected = _validate_identity(expected_identity)
    if canonical_sha256(state["identity"]) != canonical_sha256(expected):
        _fail("identity_mismatch", "resume identity differs from expected identity")


def _validate_identity_before_plan(state: Any, expected_identity: Mapping[str, Any]) -> None:
    """Reject a different run identity before comparing its plan cell expansion.

    A changed matrix normally also changes the state entry cells.  Checking the identity first
    keeps that case reportable as an identity mismatch rather than leaking an incidental cell
    order/count error, while malformed state still receives the normal shape validation below.
    """

    if not isinstance(state, Mapping) or set(state) != _STATE_KEYS:
        return
    identity = _validate_state_header(state)
    _validate_expected_identity({"identity": identity}, expected_identity)


def _check_existing_file(path: Path, *, missing_ok: bool) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        if missing_ok:
            return
        _fail("missing_state", f"resume state does not exist: {path}")
    except OSError as exc:
        raise ResumeFailure("state_path", f"cannot inspect resume state {path}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode):
        _fail("symlink_state", "resume state symlinks are not permitted")
    if not stat.S_ISREG(info.st_mode):
        _fail("nonregular_state", "resume state must be a regular file")
    if stat.S_IMODE(info.st_mode) != 0o644:
        _fail("state_mode", "resume state mode must be exactly 0644")


def _atomic_write(path: Path, state: Mapping[str, Any]) -> None:
    _check_existing_file(path, missing_ok=True)
    parent = path.parent
    if not parent.is_dir():
        _fail("state_path", f"resume state parent is not a directory: {parent}")
    encoded = (json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    descriptor = -1
    temporary: Path | None = None
    directory_descriptor = -1
    try:
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(parent))
        temporary = Path(temporary_name)
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_descriptor = os.open(str(parent), os.O_RDONLY)
        os.fsync(directory_descriptor)
    except OSError as exc:
        raise ResumeFailure("atomic_write", f"cannot atomically write resume state: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if directory_descriptor >= 0:
            os.close(directory_descriptor)
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _save_validated(path: Path, state: Mapping[str, Any], plan: MatrixPlan) -> dict[str, Any]:
    validated, _ = _validate_state_shape(state, plan)
    _atomic_write(path, validated)
    return validated


def create_state(path: Path | str, identity: Mapping[str, Any], plan: MatrixPlan) -> dict[str, Any]:
    """Create and atomically persist a pending state for every plan cell."""

    path = Path(path)
    try:
        path.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise ResumeFailure("state_path", f"cannot inspect resume state {path}: {exc}") from exc
    else:
        _check_existing_file(path, missing_ok=False)
        _fail("state_exists", f"resume state already exists: {path}")
    normalized_identity = _validate_identity(identity)
    state = {
        "schema_version": 1,
        "identity": normalized_identity,
        "identity_sha256": canonical_sha256(normalized_identity),
        "entries": [{"cell": cell.as_dict(), "state": "pending"} for cell in _expected_cells(plan)],
    }
    return _save_validated(path, state, plan)


def _read_validated(path: Path, expected_identity: Mapping[str, Any], plan: MatrixPlan, *, recover: bool) -> dict[str, Any]:
    _check_existing_file(path, missing_ok=False)
    try:
        state = load_json(path)
    except (OSError, ValueError) as exc:
        raise ResumeFailure("invalid_json", str(exc)) from exc
    _validate_identity_before_plan(state, expected_identity)
    validated, _ = _validate_state_shape(state, plan)
    _validate_expected_identity(validated, expected_identity)
    if recover and any(entry["state"] == "in_progress" for entry in validated["entries"]):
        recovered = {
            **validated,
            "entries": [
                {"cell": entry["cell"], "state": "pending"}
                if entry["state"] == "in_progress"
                else entry
                for entry in validated["entries"]
            ],
        }
        return _save_validated(path, recovered, plan)
    return validated


def load_state(path: Path | str, expected_identity: Mapping[str, Any], plan: MatrixPlan) -> dict[str, Any]:
    """Load a state, recovering every interrupted in-progress entry to pending."""

    return _read_validated(Path(path), expected_identity, plan, recover=True)


def _transition(
    path: Path | str,
    expected_identity: Mapping[str, Any],
    plan: MatrixPlan,
    cell: CellKey,
    current: str,
    target: str,
    result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(cell, CellKey):
        _fail("cell", "transition cell must be a CellKey")
    state_path = Path(path)
    state = _read_validated(state_path, expected_identity, plan, recover=False)
    expected = _expected_cells(plan)
    try:
        index = expected.index(cell)
    except ValueError as exc:
        raise ResumeFailure("unknown_cell", f"CellKey is absent from MatrixPlan: {cell!r}") from exc
    entry = state["entries"][index]
    if entry["state"] != current:
        _fail("illegal_transition", f"CellKey {cell!r} is {entry['state']}, expected {current}")
    replacement: dict[str, Any] = {"cell": entry["cell"], "state": target}
    if target == "complete":
        if result is None:
            _fail("result_shape", "complete transition requires a result object")
        replacement["result"] = _validate_result(result, "result")
    state["entries"][index] = replacement
    return _save_validated(state_path, state, plan)


def mark_in_progress(path: Path | str, expected_identity: Mapping[str, Any], plan: MatrixPlan, cell: CellKey) -> dict[str, Any]:
    """Atomically transition one pending cell to in_progress."""

    return _transition(path, expected_identity, plan, cell, "pending", "in_progress")


def mark_complete(
    path: Path | str,
    expected_identity: Mapping[str, Any],
    plan: MatrixPlan,
    cell: CellKey,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically transition one in-progress cell to complete with an object result."""

    return _transition(path, expected_identity, plan, cell, "in_progress", "complete", result)


__all__ = ["ResumeFailure", "create_state", "load_state", "mark_complete", "mark_in_progress"]
