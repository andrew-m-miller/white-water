#!/usr/bin/env python3
"""Focused lifecycle tests for the dependency-free bake-off coordinator."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import tempfile
from types import MappingProxyType
from unittest.mock import patch
import unittest

from . import coordinator as coordinator_module
from .coordinator import CommittedExecution, CoordinatorFailure, IncompleteFailure, RunCoordinator
from .matrix import CellKey, MatrixPlan
from .resume import create_state, load_state, mark_complete, mark_in_progress


CELLS = (
    # Deliberately not sorted by CellKey: MatrixPlan order is the contract.
    CellKey("candidate-b", "shot-z", "cond-b", "mp2", "cuda", "live_flame"),
    CellKey("candidate-a", "shot-a", "cond-a", "mp0_5", "cpu", "not_applicable"),
    CellKey("candidate-c", "shot-m", "cond-a", "mp0_5", "coreml", "not_applicable"),
)
PLAN = MatrixPlan({"matrix_sha256": "a" * 64}, CELLS, ())
IDENTITY = {
    "protocol_sha256": "1" * 64,
    "corpus_sha256": "2" * 64,
    "matrix_sha256": "3" * 64,
    "profile": "screen",
    "environment": "el8-x86_64",
}


def _result(cell: CellKey, *, status: str = "pass", failure=None, **extra):
    result = {
        "candidate_id": cell.candidate,
        "shot_id": cell.shot,
        "conditioning_token": cell.conditioning,
        "cap_token": cell.cap,
        "provider": cell.provider,
        "host_load": cell.host_load,
        "status": status,
    }
    if failure is not None:
        result["failure"] = failure
    result.update(extra)
    return result


def _ref(cell: CellKey, suffix: str = "a") -> dict[str, str | int]:
    return {
        "schema_version": 1,
        "identity_sha256": "1" * 64,
        "cell_id": json.dumps(cell.as_dict(), sort_keys=True, separators=(",", ":")),
        "cell_sha256": "2" * 64,
        "attempt_id": f"attempt-{suffix}",
        "manifest_sha256": "3" * 64,
    }


def _execution(cell: CellKey, *, suffix: str = "a", result=None) -> CommittedExecution:
    return CommittedExecution(_result(cell) if result is None else result, _ref(cell, suffix))


def _validate_ref(_cell, _result, _ref_value):
    return None


def _association_ref(cell: CellKey, result: dict) -> dict:
    ref = _ref(cell)
    canonical = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    ref["manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return ref


def _association_validator(cell: CellKey, result: dict, ref: dict) -> None:
    expected_cell_id = json.dumps(cell.as_dict(), sort_keys=True, separators=(",", ":"))
    if ref.get("cell_id") != expected_cell_id:
        raise ValueError("artifact ref belongs to a different cell")
    canonical = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if ref.get("manifest_sha256") != hashlib.sha256(canonical.encode("utf-8")).hexdigest():
        raise ValueError("artifact ref does not prove this result")


def _new_state(directory: str) -> Path:
    path = Path(directory) / "state.json"
    create_state(path, IDENTITY, PLAN)
    return path


class CoordinatorTests(unittest.TestCase):
    def _count_validations(self):
        calls = []
        original = coordinator_module._validate_result

        def count(result, cell):
            calls.append(cell)
            return original(result, cell)

        return calls, count

    def test_pending_cells_run_in_exact_plan_order_and_records_are_ordered(self):
        with tempfile.TemporaryDirectory(prefix="whitewater-coordinator-") as directory:
            path = _new_state(directory)
            seen = []

            def execute(cell):
                raw = json.loads(path.read_text(encoding="utf-8"))
                index = CELLS.index(cell)
                self.assertEqual(raw["entries"][index]["state"], "in_progress")
                seen.append(cell)
                return CommittedExecution(
                    _result(cell, marker=len(seen)), _ref(cell, str(len(seen)))
                )

            records = RunCoordinator(path, IDENTITY, PLAN, execute, _validate_ref).run()
            self.assertEqual(seen, list(CELLS))
            self.assertEqual([record["cell"] for record in records], [cell.as_dict() for cell in CELLS])
            self.assertEqual([record["result"]["marker"] for record in records], [1, 2, 3])

    def test_run_semantically_validates_each_fresh_result_once(self):
        with tempfile.TemporaryDirectory(prefix="whitewater-coordinator-") as directory:
            path = _new_state(directory)
            calls, count = self._count_validations()
            with patch.object(coordinator_module, "_validate_result", side_effect=count):
                RunCoordinator(path, IDENTITY, PLAN, lambda cell: _execution(cell), _validate_ref).run()
            self.assertEqual(calls, list(CELLS))

    def test_resumed_run_validates_existing_and_fresh_results_once_each(self):
        with tempfile.TemporaryDirectory(prefix="whitewater-coordinator-") as directory:
            path = _new_state(directory)
            mark_in_progress(path, IDENTITY, PLAN, CELLS[0])
            mark_complete(path, IDENTITY, PLAN, CELLS[0], _result(CELLS[0]), _ref(CELLS[0]))
            calls, count = self._count_validations()
            with patch.object(coordinator_module, "_validate_result", side_effect=count):
                records = RunCoordinator(path, IDENTITY, PLAN, lambda cell: _execution(cell), _validate_ref).run()
            self.assertEqual(calls, list(CELLS))
            self.assertEqual([record["cell"] for record in records], [cell.as_dict() for cell in CELLS])

    def test_public_completed_record_paths_validate_each_persisted_result_once(self):
        with tempfile.TemporaryDirectory(prefix="whitewater-coordinator-") as directory:
            path = _new_state(directory)
            for cell in CELLS:
                mark_in_progress(path, IDENTITY, PLAN, cell)
                mark_complete(path, IDENTITY, PLAN, cell, _result(cell), _ref(cell))

            for read_records in (
                lambda: RunCoordinator(path, IDENTITY, PLAN, lambda cell: _execution(cell), _validate_ref).completed_records(),
                lambda: coordinator_module.completed_records(path, IDENTITY, PLAN, _validate_ref),
            ):
                with self.subTest(path=read_records):
                    calls, count = self._count_validations()
                    with patch.object(coordinator_module, "_validate_result", side_effect=count):
                        records = read_records()
                    self.assertEqual(calls, list(CELLS))
                    self.assertEqual(len(records), len(CELLS))

    def test_interruption_leaves_cell_in_progress_and_next_invocation_recovers(self):
        with tempfile.TemporaryDirectory(prefix="whitewater-coordinator-") as directory:
            path = _new_state(directory)
            first_seen = []

            def interrupt(cell):
                first_seen.append(cell)
                if cell == CELLS[1]:
                    raise RuntimeError("interrupted")
                return CommittedExecution(_result(cell), _ref(cell))

            with self.assertRaises(RuntimeError):
                RunCoordinator(path, IDENTITY, PLAN, lambda cell: (
                    interrupt(cell) if cell != CELLS[1] else interrupt(cell)
                ), _validate_ref).run()
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(raw["entries"][1]["state"], "in_progress")
            interrupted = load_state(path, IDENTITY, PLAN)
            self.assertEqual(first_seen, [CELLS[0], CELLS[1]])
            self.assertEqual(interrupted["entries"][0]["state"], "complete")
            self.assertEqual(interrupted["entries"][1]["state"], "pending")

            resumed_seen = []
            records = RunCoordinator(
                path, IDENTITY, PLAN, lambda cell: CommittedExecution(
                    (resumed_seen.append(cell), _result(cell))[1], _ref(cell)
                ), _validate_ref
            ).run()
            self.assertEqual(resumed_seen, [CELLS[1], CELLS[2]])
            self.assertEqual(len(records), len(CELLS))

    def test_complete_cells_are_never_rerun(self):
        with tempfile.TemporaryDirectory(prefix="whitewater-coordinator-") as directory:
            path = _new_state(directory)
            mark_in_progress(path, IDENTITY, PLAN, CELLS[0])
            mark_complete(path, IDENTITY, PLAN, CELLS[0], _result(CELLS[0]), _ref(CELLS[0]))
            seen = []
            RunCoordinator(
                path, IDENTITY, PLAN, lambda cell: CommittedExecution(
                    (seen.append(cell), _result(cell))[1], _ref(cell)
                ), _validate_ref
            ).run()
            self.assertEqual(seen, [CELLS[1], CELLS[2]])

    def test_executor_result_validation_is_typed_and_leaves_in_progress(self):
        malformed_results = (
            (MappingProxyType(_result(CELLS[0])), "result_shape"),
            (_result(CELLS[0], shot_id="wrong"), "cell_mismatch"),
            ({**_result(CELLS[0]), "status": "unknown"}, "result_status"),
            ({**_result(CELLS[0]), "metric": float("nan")}, "nonfinite"),
            ({**_result(CELLS[0]), "values": (1, 2)}, "json_value"),
            ({**_result(CELLS[0]), "failure": None}, "failure_shape"),
            (_result(CELLS[0], status="fail"), "failure_missing"),
            (_result(CELLS[0], status="skip", failure={"message": "not typed"}), "failure_shape"),
            (_result(CELLS[0], status="pass", failure={"type": "other", "message": "bad"}), "failure_shape"),
            (_result(CELLS[0], status="fail", failure={"type": "unknown", "message": "bad"}), "failure_type"),
        )
        for malformed, kind in malformed_results:
            with self.subTest(kind=kind):
                with tempfile.TemporaryDirectory(prefix="whitewater-coordinator-") as directory:
                    path = _new_state(directory)
                    with self.assertRaises(CoordinatorFailure) as context:
                        RunCoordinator(
                            path, IDENTITY, PLAN,
                            lambda cell, value=malformed: CommittedExecution(value, _ref(cell)),
                            _validate_ref,
                        ).run()
                    self.assertEqual(context.exception.kind, kind)
                    self.assertEqual(context.exception.failure_type, "coordinator_failure")
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    self.assertEqual(raw["entries"][0]["state"], "in_progress")
                    # Read the raw file through the normal loader after the assertion: this
                    # both proves the transition was durable and exercises recovery semantics.
                    recovered = load_state(path, IDENTITY, PLAN)
                    self.assertEqual(recovered["entries"][0]["state"], "pending")

    def test_legacy_result_only_executor_is_rejected_and_cannot_complete(self):
        with tempfile.TemporaryDirectory(prefix="whitewater-coordinator-") as directory:
            path = _new_state(directory)
            with self.assertRaises(CoordinatorFailure) as context:
                RunCoordinator(
                    path, IDENTITY, PLAN, lambda cell: _result(cell), _validate_ref,
                ).run()
            self.assertEqual(context.exception.kind, "executor_contract")
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(raw["entries"][0]["state"], "in_progress")

    def test_existing_refs_are_validated_and_public_records_omit_them(self):
        with tempfile.TemporaryDirectory(prefix="whitewater-coordinator-") as directory:
            path = _new_state(directory)
            for cell in CELLS:
                mark_in_progress(path, IDENTITY, PLAN, cell)
                mark_complete(path, IDENTITY, PLAN, cell, _result(cell), _ref(cell))
            seen_refs = []

            def validator(_cell, _result, ref):
                seen_refs.append(ref)

            coordinator = RunCoordinator(path, IDENTITY, PLAN, lambda cell: _execution(cell), validator)
            records = coordinator.run()
            self.assertEqual(len(seen_refs), len(CELLS))
            self.assertTrue(all("artifact_ref" not in record for record in records))
            internal = coordinator.completed_records_with_refs()
            self.assertTrue(all("artifact_ref" in record for record in internal))

    def test_ref_validator_failure_refuses_tampered_or_missing_generation(self):
        with tempfile.TemporaryDirectory(prefix="whitewater-coordinator-") as directory:
            path = _new_state(directory)
            mark_in_progress(path, IDENTITY, PLAN, CELLS[0])
            mark_complete(path, IDENTITY, PLAN, CELLS[0], _result(CELLS[0]), _ref(CELLS[0]))

            def reject(_ref_value):
                raise RuntimeError("exact result artifact is missing")

            with self.assertRaises(CoordinatorFailure) as context:
                RunCoordinator(path, IDENTITY, PLAN, lambda cell: _execution(cell), reject).run()
            self.assertEqual(context.exception.kind, "artifact_ref")

    def test_valid_refs_swapped_between_completed_cells_are_refused(self):
        with tempfile.TemporaryDirectory(prefix="whitewater-coordinator-") as directory:
            path = _new_state(directory)
            for cell in CELLS[:2]:
                result = _result(cell)
                mark_in_progress(path, IDENTITY, PLAN, cell)
                mark_complete(path, IDENTITY, PLAN, cell, result, _association_ref(cell, result))
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["entries"][0]["artifact_ref"], raw["entries"][1]["artifact_ref"] = (
                raw["entries"][1]["artifact_ref"], raw["entries"][0]["artifact_ref"]
            )
            path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaises(CoordinatorFailure) as context:
                RunCoordinator(path, IDENTITY, PLAN, lambda cell: _execution(cell), _association_validator).run()
            self.assertEqual(context.exception.kind, "artifact_ref")

    def test_semantically_valid_result_edit_is_refused_by_exact_ref_validator(self):
        with tempfile.TemporaryDirectory(prefix="whitewater-coordinator-") as directory:
            path = _new_state(directory)
            cell = CELLS[0]
            original = _result(cell, marker=1)
            mark_in_progress(path, IDENTITY, PLAN, cell)
            mark_complete(path, IDENTITY, PLAN, cell, original, _association_ref(cell, original))
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["entries"][0]["result"]["marker"] = 2
            path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaises(CoordinatorFailure) as context:
                RunCoordinator(path, IDENTITY, PLAN, lambda current: _execution(current), _association_validator).run()
            self.assertEqual(context.exception.kind, "artifact_ref")

    def test_non_pass_results_require_and_preserve_typed_failure(self):
        with tempfile.TemporaryDirectory(prefix="whitewater-coordinator-") as directory:
            path = _new_state(directory)
            failure = {
                "type": "runtime_error",
                "message": "provider stopped",
                "retryable": True,
                "stage": "inference",
            }
            def execute(cell):
                return _result(cell, status="fail", failure=failure)

            records = RunCoordinator(
                path, IDENTITY, PLAN,
                lambda cell: CommittedExecution(execute(cell), _ref(cell)),
                _validate_ref,
            ).run()
            self.assertTrue(all(record["result"]["status"] == "fail" for record in records))
            self.assertTrue(all(record["result"]["failure"] == failure for record in records))

    def test_completed_records_require_all_cells_complete(self):
        with tempfile.TemporaryDirectory(prefix="whitewater-coordinator-") as directory:
            path = _new_state(directory)
            coordinator = RunCoordinator(path, IDENTITY, PLAN, lambda cell: _execution(cell), _validate_ref)
            with self.assertRaises(IncompleteFailure) as context:
                coordinator.completed_records()
            self.assertEqual(context.exception.kind, "incomplete")
            self.assertEqual(context.exception.reason, "incomplete")


if __name__ == "__main__":
    unittest.main()
