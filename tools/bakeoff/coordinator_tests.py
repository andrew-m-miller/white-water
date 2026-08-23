#!/usr/bin/env python3
"""Focused lifecycle tests for the dependency-free bake-off coordinator."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import MappingProxyType
import unittest

from .coordinator import CoordinatorFailure, IncompleteFailure, RunCoordinator
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


def _new_state(directory: str) -> Path:
    path = Path(directory) / "state.json"
    create_state(path, IDENTITY, PLAN)
    return path


class CoordinatorTests(unittest.TestCase):
    def test_pending_cells_run_in_exact_plan_order_and_records_are_ordered(self):
        with tempfile.TemporaryDirectory(prefix="whitewater-coordinator-") as directory:
            path = _new_state(directory)
            seen = []

            def execute(cell):
                raw = json.loads(path.read_text(encoding="utf-8"))
                index = CELLS.index(cell)
                self.assertEqual(raw["entries"][index]["state"], "in_progress")
                seen.append(cell)
                return _result(cell, marker=len(seen))

            records = RunCoordinator(path, IDENTITY, PLAN, execute).run()
            self.assertEqual(seen, list(CELLS))
            self.assertEqual([record["cell"] for record in records], [cell.as_dict() for cell in CELLS])
            self.assertEqual([record["result"]["marker"] for record in records], [1, 2, 3])

    def test_interruption_leaves_cell_in_progress_and_next_invocation_recovers(self):
        with tempfile.TemporaryDirectory(prefix="whitewater-coordinator-") as directory:
            path = _new_state(directory)
            first_seen = []

            def interrupt(cell):
                first_seen.append(cell)
                if cell == CELLS[1]:
                    raise RuntimeError("interrupted")
                return _result(cell)

            with self.assertRaises(RuntimeError):
                RunCoordinator(path, IDENTITY, PLAN, interrupt).run()
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(raw["entries"][1]["state"], "in_progress")
            interrupted = load_state(path, IDENTITY, PLAN)
            self.assertEqual(first_seen, [CELLS[0], CELLS[1]])
            self.assertEqual(interrupted["entries"][0]["state"], "complete")
            self.assertEqual(interrupted["entries"][1]["state"], "pending")

            resumed_seen = []
            records = RunCoordinator(
                path, IDENTITY, PLAN, lambda cell: (resumed_seen.append(cell), _result(cell))[1]
            ).run()
            self.assertEqual(resumed_seen, [CELLS[1], CELLS[2]])
            self.assertEqual(len(records), len(CELLS))

    def test_complete_cells_are_never_rerun(self):
        with tempfile.TemporaryDirectory(prefix="whitewater-coordinator-") as directory:
            path = _new_state(directory)
            mark_in_progress(path, IDENTITY, PLAN, CELLS[0])
            mark_complete(path, IDENTITY, PLAN, CELLS[0], _result(CELLS[0]))
            seen = []
            RunCoordinator(
                path, IDENTITY, PLAN, lambda cell: (seen.append(cell), _result(cell))[1]
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
                        RunCoordinator(path, IDENTITY, PLAN, lambda _cell, value=malformed: value).run()
                    self.assertEqual(context.exception.kind, kind)
                    self.assertEqual(context.exception.failure_type, "coordinator_failure")
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    self.assertEqual(raw["entries"][0]["state"], "in_progress")
                    # Read the raw file through the normal loader after the assertion: this
                    # both proves the transition was durable and exercises recovery semantics.
                    recovered = load_state(path, IDENTITY, PLAN)
                    self.assertEqual(recovered["entries"][0]["state"], "pending")

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

            records = RunCoordinator(path, IDENTITY, PLAN, execute).run()
            self.assertTrue(all(record["result"]["status"] == "fail" for record in records))
            self.assertTrue(all(record["result"]["failure"] == failure for record in records))

    def test_completed_records_require_all_cells_complete(self):
        with tempfile.TemporaryDirectory(prefix="whitewater-coordinator-") as directory:
            path = _new_state(directory)
            coordinator = RunCoordinator(path, IDENTITY, PLAN, lambda cell: _result(cell))
            with self.assertRaises(IncompleteFailure) as context:
                coordinator.completed_records()
            self.assertEqual(context.exception.kind, "incomplete")
            self.assertEqual(context.exception.reason, "incomplete")


if __name__ == "__main__":
    unittest.main()
