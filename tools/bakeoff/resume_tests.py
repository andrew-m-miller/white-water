#!/usr/bin/env python3
"""Focused strictness and atomicity tests for runner resume state."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import stat
import tempfile
from types import MappingProxyType
from unittest.mock import patch

from . import resume as resume_module
from .matrix import CellKey, MatrixPlan
from .resume import (
    ResumeFailure,
    create_state,
    load_state,
    mark_complete,
    mark_in_progress,
)


CELLS = (
    CellKey("candidate-a", "shot-a", "cond-a", "mp0_5", "cpu", "not_applicable"),
    CellKey("candidate-a", "shot-b", "cond-a", "mp0_5", "cpu", "not_applicable"),
)
PLAN = MatrixPlan({"matrix_sha256": "a" * 64}, CELLS, ())
IDENTITY = {
    "protocol_sha256": "1" * 64,
    "corpus_sha256": "2" * 64,
    "matrix_sha256": "3" * 64,
    "profile": "smoke",
    "environment": "el8-x86_64",
}


def _failure(kind: str, callback) -> None:
    try:
        callback()
    except ResumeFailure as failure:
        assert failure.kind == kind, (failure.kind, kind)
        assert failure.reason == kind
        assert failure.failure_type == "resume_failure"
    else:
        raise AssertionError(f"expected ResumeFailure({kind})")


def _raw_state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_raw(path: Path, value: str | dict, mode: int = 0o644) -> None:
    if isinstance(value, dict):
        value = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.write_text(value, encoding="utf-8")
    os.chmod(path, mode)


def _new_state(path: Path) -> dict:
    if path.exists() or path.is_symlink():
        path.unlink()
    return create_state(path, IDENTITY, PLAN)


def _ref(cell, suffix: str = "a") -> dict[str, object]:
    return {
        "schema_version": 1,
        "identity_sha256": "1" * 64,
        "cell_id": cell.candidate + "/" + cell.shot,
        "cell_sha256": "2" * 64,
        "attempt_id": "attempt-" + suffix,
        "manifest_sha256": "3" * 64,
    }


def test_create_determinism_and_shape() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-resume-") as temporary:
        path = Path(temporary) / "state.json"
        first = _new_state(path)
        first_bytes = path.read_bytes()
        _failure("state_exists", lambda: create_state(path, IDENTITY, PLAN))
        second_path = Path(temporary) / "state-copy.json"
        _new_state(second_path)
        assert first_bytes == second_path.read_bytes()
        assert first_bytes == path.read_bytes()
        assert stat.S_IMODE(path.stat().st_mode) == 0o644
        assert sorted(path.parent.iterdir()) == sorted([path, second_path])
        assert first["schema_version"] == 2
        assert [entry["state"] for entry in first["entries"]] == ["pending", "pending"]
        assert all(set(entry) == {"cell", "state"} for entry in first["entries"])


def test_create_collision_at_publication_does_not_clobber() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-resume-") as temporary:
        directory = Path(temporary)
        path = directory / "state.json"
        competitor_bytes = b"competitor state\n"

        def publish_with_competitor(_temporary: Path, destination: Path) -> None:
            assert destination == path
            path.write_bytes(competitor_bytes)
            os.chmod(path, 0o644)
            raise FileExistsError("publication collision")

        with patch.object(resume_module.os, "link", side_effect=publish_with_competitor):
            _failure("state_exists", lambda: create_state(path, IDENTITY, PLAN))
        assert path.read_bytes() == competitor_bytes
        assert list(directory.glob(f".{path.name}.*.tmp")) == []


def test_identity_and_cell_strictness() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-resume-") as temporary:
        path = Path(temporary) / "state.json"
        _new_state(path)
        changed_identity = dict(IDENTITY, profile="final")
        _failure("identity_mismatch", lambda: load_state(path, changed_identity, PLAN))

        _new_state(path)
        tampered = _raw_state(path)
        tampered["identity"]["profile"] = "tampered"
        _write_raw(path, tampered)
        _failure("identity_hash", lambda: load_state(path, IDENTITY, PLAN))

        _new_state(path)
        tampered = _raw_state(path)
        tampered["schema_version"] = 1
        _write_raw(path, tampered)
        _failure("schema_version", lambda: load_state(path, IDENTITY, PLAN))

        _new_state(path)
        tampered = _raw_state(path)
        tampered["identity_sha256"] = "0" * 64
        _write_raw(path, tampered)
        _failure("identity_hash", lambda: load_state(path, IDENTITY, PLAN))

        _new_state(path)
        tampered = _raw_state(path)
        tampered["entries"][1] = copy.deepcopy(tampered["entries"][0])
        _write_raw(path, tampered)
        _failure("duplicate_cell", lambda: load_state(path, IDENTITY, PLAN))

        _new_state(path)
        tampered = _raw_state(path)
        tampered["entries"].reverse()
        _write_raw(path, tampered)
        _failure("cell_order", lambda: load_state(path, IDENTITY, PLAN))

        _new_state(path)
        tampered = _raw_state(path)
        tampered["entries"][0]["cell"]["extra"] = "bad"
        _write_raw(path, tampered)
        _failure("cell_shape", lambda: load_state(path, IDENTITY, PLAN))

        _new_state(path)
        tampered = _raw_state(path)
        tampered["entries"].pop()
        _write_raw(path, tampered)
        _failure("cell_count", lambda: load_state(path, IDENTITY, PLAN))

        _new_state(path)
        tampered = _raw_state(path)
        tampered["entries"][0]["state"] = "unknown"
        _write_raw(path, tampered)
        _failure("status", lambda: load_state(path, IDENTITY, PLAN))


def test_json_and_result_strictness() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-resume-") as temporary:
        path = Path(temporary) / "state.json"
        _new_state(path)
        duplicate = '{"schema_version":2,"identity":{},"identity":{},"identity_sha256":"%s","entries":[]}' % ("0" * 64)
        _write_raw(path, duplicate)
        _failure("invalid_json", lambda: load_state(path, IDENTITY, PLAN))

        nan_json = '{"schema_version":2,"identity":{"bad":NaN},"identity_sha256":"%s","entries":[]}' % ("0" * 64)
        _write_raw(path, nan_json)
        _failure("invalid_json", lambda: load_state(path, IDENTITY, PLAN))

        _new_state(path)
        tampered = _raw_state(path)
        tampered["entries"][0]["result"] = {"value": 1}
        _write_raw(path, tampered)
        _failure("result_shape", lambda: load_state(path, IDENTITY, PLAN))

        _new_state(path)
        tampered = _raw_state(path)
        tampered["entries"][0]["state"] = "complete"
        _write_raw(path, tampered)
        _failure("result_shape", lambda: load_state(path, IDENTITY, PLAN))

        _new_state(path)
        tampered = _raw_state(path)
        tampered["entries"][0]["state"] = "complete"
        tampered["entries"][0]["result"] = {"bad": float("nan")}
        _write_raw(path, tampered)
        _failure("invalid_json", lambda: load_state(path, IDENTITY, PLAN))

        _new_state(path)
        non_string_identity = dict(IDENTITY)
        non_string_identity[1] = "bad"
        _failure("json_value", lambda: create_state(path.with_name("non-string.json"), non_string_identity, PLAN))

        _new_state(path)
        cyclic_result = {}
        cyclic_result["cycle"] = cyclic_result
        mark_in_progress(path, IDENTITY, PLAN, CELLS[0])
        _failure("identity_shape", lambda: mark_complete(path, IDENTITY, PLAN, CELLS[0], cyclic_result, _ref(CELLS[0])))

        mapping_proxy = MappingProxyType({"nested": MappingProxyType({"value": 1})})
        _failure("json_value", lambda: create_state(path.with_name("mapping-proxy.json"), mapping_proxy, PLAN))


def test_v1_state_is_refused_instead_of_silently_migrated() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-resume-") as temporary:
        path = Path(temporary) / "state.json"
        _new_state(path)
        legacy = _raw_state(path)
        legacy["schema_version"] = 1
        legacy["entries"][0]["state"] = "complete"
        legacy["entries"][0]["result"] = {"legacy": True}
        _write_raw(path, legacy)
        try:
            load_state(path, IDENTITY, PLAN)
        except ResumeFailure as failure:
            assert failure.kind == "schema_version"
            assert "schema_version 1" in str(failure)
            assert "refused" in str(failure)
        else:
            raise AssertionError("legacy v1 state must not be migrated")


def test_complete_entries_require_a_well_shaped_exact_ref() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-resume-") as temporary:
        path = Path(temporary) / "state.json"
        _new_state(path)
        tampered = _raw_state(path)
        tampered["entries"][0]["state"] = "complete"
        tampered["entries"][0]["result"] = {"ok": True}
        _write_raw(path, tampered)
        _failure("artifact_ref_shape", lambda: load_state(path, IDENTITY, PLAN))

        _new_state(path)
        mark_in_progress(path, IDENTITY, PLAN, CELLS[0])
        _failure(
            "artifact_ref_shape",
            lambda: mark_complete(path, IDENTITY, PLAN, CELLS[0], {"ok": True}, {"bad": True}),
        )


def test_interrupted_recovery() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-resume-") as temporary:
        path = Path(temporary) / "state.json"
        _new_state(path)
        mark_in_progress(path, IDENTITY, PLAN, CELLS[0])
        assert _raw_state(path)["entries"][0]["state"] == "in_progress"
        recovered = load_state(path, IDENTITY, PLAN)
        assert recovered["entries"][0]["state"] == "pending"
        assert recovered["entries"][1]["state"] == "pending"
        assert sorted(path.parent.iterdir()) == [path]


def test_transitions_and_known_cells() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-resume-") as temporary:
        path = Path(temporary) / "state.json"
        _new_state(path)
        _failure("illegal_transition", lambda: mark_complete(path, IDENTITY, PLAN, CELLS[0], {"ok": True}, _ref(CELLS[0])))
        _failure("unknown_cell", lambda: mark_in_progress(path, IDENTITY, PLAN, CellKey("no", "no", "no", "no", "no", "no")))
        mark_in_progress(path, IDENTITY, PLAN, CELLS[0])
        complete = mark_complete(path, IDENTITY, PLAN, CELLS[0], {"ok": True, "samples": [1, 2]}, _ref(CELLS[0]))
        assert complete["entries"][0]["state"] == "complete"
        assert complete["entries"][0]["result"] == {"ok": True, "samples": [1, 2]}
        _failure("illegal_transition", lambda: mark_in_progress(path, IDENTITY, PLAN, CELLS[0]))
        mark_in_progress(path, IDENTITY, PLAN, CELLS[1])
        _failure("result_shape", lambda: mark_complete(path, IDENTITY, PLAN, CELLS[1], [], _ref(CELLS[1])))
        _failure("illegal_transition", lambda: mark_in_progress(path, IDENTITY, PLAN, CELLS[1]))


def test_destination_security() -> None:
    with tempfile.TemporaryDirectory(prefix="whitewater-resume-") as temporary:
        directory = Path(temporary)
        target = directory / "target.json"
        target.write_text("{}", encoding="utf-8")
        link = directory / "link.json"
        link.symlink_to(target)
        _failure("symlink_state", lambda: load_state(link, IDENTITY, PLAN))
        _failure("symlink_state", lambda: create_state(link, IDENTITY, PLAN))

        subdirectory = directory / "directory.json"
        subdirectory.mkdir()
        _failure("nonregular_state", lambda: load_state(subdirectory, IDENTITY, PLAN))
        _failure("nonregular_state", lambda: create_state(subdirectory, IDENTITY, PLAN))

        mode_path = directory / "mode.json"
        _new_state(mode_path)
        os.chmod(mode_path, 0o600)
        _failure("state_mode", lambda: load_state(mode_path, IDENTITY, PLAN))


def main() -> int:
    test_create_determinism_and_shape()
    test_create_collision_at_publication_does_not_clobber()
    test_identity_and_cell_strictness()
    test_json_and_result_strictness()
    test_v1_state_is_refused_instead_of_silently_migrated()
    test_complete_entries_require_a_well_shaped_exact_ref()
    test_interrupted_recovery()
    test_transitions_and_known_cells()
    test_destination_security()
    print("P25-4 resume tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
