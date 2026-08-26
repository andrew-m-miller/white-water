#!/usr/bin/env python3
"""Integration tests for the dependency-free bake-off run session."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from .coordinator import CommittedExecution
from .matrix import build_matrix
from .reporting import ReportFailure
from .resume import ResumeFailure
from .session import SessionFailure, resume_identity, run_session
from .validator import validate_report_consistency


ROOT = Path(__file__).resolve().parents[2]


def _load(relative: str):
    with (ROOT / relative).open(encoding="utf-8") as stream:
        return json.load(stream)


class SessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = _load("bakeoff/protocol-v1.json")
        self.corpus = _load("bakeoff/fixtures/positive/corpus-v1.json")
        self.fixture = _load("bakeoff/fixtures/positive/report-v1.json")
        self.report_schema = _load("bakeoff/report-v1.schema.json")
        self.corpus_schema = _load("bakeoff/corpus-v1.schema.json")
        generated = {
            "schema_version", "protocol_id", "corpus_sha256", "matrix",
            "candidates", "results", "summary",
        }
        self.metadata = {
            key: value for key, value in self.fixture.items() if key not in generated
        }
        self.candidates = copy.deepcopy(self.fixture["candidates"])
        self.selections = {
            "candidate_ids": ["sea-raft-m"],
            "shot_ids": ["syn-identity"],
            "conditioning_tokens": ["native-clamp01-v1"],
            "cap_tokens": ["mp0_5"],
            "providers": [{"token": "cpu", "host_loads": ["not_applicable"]}],
        }

    def _result_for(self, cell):
        result = copy.deepcopy(self.fixture["results"][0])
        result["candidate_id"] = cell.candidate
        result["shot_id"] = cell.shot
        result["conditioning_token"] = cell.conditioning
        result["cap_token"] = cell.cap
        result["provider"] = cell.provider
        result["host_load"] = cell.host_load
        return result

    def _execution_for(self, cell, result=None):
        return CommittedExecution(
            self._result_for(cell) if result is None else result,
            {
                "schema_version": 1,
                "identity_sha256": "1" * 64,
                "cell_id": cell.candidate + "/" + cell.shot,
                "cell_sha256": "2" * 64,
                "attempt_id": "attempt-" + cell.conditioning,
                "manifest_sha256": "3" * 64,
            },
        )

    @staticmethod
    def _validate_ref(_cell, _result, _ref):
        return None

    def _run(self, directory: Path, selections=None, metadata=None, candidates=None, executor=None):
        selected = self.selections if selections is None else selections
        report_metadata = self.metadata if metadata is None else metadata
        candidate_entries = self.candidates if candidates is None else candidates
        if executor is None:
            executor = lambda cell: self._execution_for(cell)
        return run_session(
            self.protocol,
            self.corpus,
            self.report_schema,
            self.corpus_schema,
            candidate_entries,
            selected,
            "screen",
            "el8-x86_64",
            report_metadata,
            directory / "state.json",
            directory / "report.json",
            directory / "report.csv",
            executor,
            artifact_ref_validator=self._validate_ref,
        )

    def test_initial_run_returns_valid_report_and_deterministic_pair(self):
        with tempfile.TemporaryDirectory(prefix="whitewater-session-") as first_dir, tempfile.TemporaryDirectory(prefix="whitewater-session-") as second_dir:
            first = self._run(Path(first_dir))
            second = self._run(Path(second_dir))
            self.assertEqual(first, self.fixture)
            validate_report_consistency(
                first,
                self.protocol,
                self.report_schema,
                self.corpus,
                self.corpus_schema,
            )
            self.assertEqual(
                (Path(first_dir) / "report.json").read_bytes(),
                (Path(second_dir) / "report.json").read_bytes(),
            )
            self.assertEqual(
                (Path(first_dir) / "report.csv").read_bytes(),
                (Path(second_dir) / "report.csv").read_bytes(),
            )

    def test_interruption_resumes_without_rerunning_complete_cells(self):
        selections = copy.deepcopy(self.selections)
        selections["conditioning_tokens"] = ["native-clamp01-v1", "signed-log-v1"]
        plan = build_matrix(
            self.protocol,
            self.corpus,
            self.candidates,
            selections,
            "screen",
            "el8-x86_64",
        )
        with tempfile.TemporaryDirectory(prefix="whitewater-session-") as temporary:
            directory = Path(temporary)
            first_seen = []

            def interrupt(cell):
                first_seen.append(cell)
                if len(first_seen) == 2:
                    raise RuntimeError("interrupted")
                return self._execution_for(cell)

            with self.assertRaises(RuntimeError):
                self._run(directory, selections=selections, executor=interrupt)
            raw = json.loads((directory / "state.json").read_text(encoding="utf-8"))
            self.assertEqual([entry["state"] for entry in raw["entries"]], ["complete", "in_progress"])
            self.assertFalse((directory / "report.json").exists())

            resumed_seen = []

            def resume(cell):
                resumed_seen.append(cell)
                return self._execution_for(cell)

            resumed_metadata = copy.deepcopy(self.metadata)
            resumed_metadata.update({
                "report_id": "p25-resumed-report",
                "started_utc": "2026-08-23T10:00:00Z",
                "completed_utc": "2026-08-23T10:01:00Z",
                "warnings": ["recovered after interruption"],
            })
            report = self._run(
                directory,
                selections=selections,
                metadata=resumed_metadata,
                executor=resume,
            )
            self.assertEqual(resumed_seen, [plan.cells[1]])
            self.assertEqual(report["report_id"], "p25-resumed-report")
            self.assertEqual(report["warnings"], ["recovered after interruption"])
            self.assertEqual(
                [result["conditioning_token"] for result in report["results"]],
                [cell.conditioning for cell in plan.cells],
            )

    def test_changed_selection_metadata_and_candidate_reject_existing_state(self):
        cases = (
            ("selection", {**self.selections, "conditioning_tokens": ["signed-log-v1"]}, self.metadata, self.candidates, "identity_mismatch"),
            ("runner", self.selections, {**self.metadata, "runner": {**self.metadata["runner"], "version": "changed"}}, self.candidates, "identity_mismatch"),
            ("hardware", self.selections, {**self.metadata, "hardware": {**self.metadata["hardware"], "driver": "changed"}}, self.candidates, "identity_mismatch"),
            ("candidate", self.selections, self.metadata, [{**self.candidates[0], "artifact_sha256": "9" * 64}], "identity_mismatch"),
        )
        for label, selections, metadata, candidates, expected_kind in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory(prefix="whitewater-session-") as temporary:
                directory = Path(temporary)
                with self.assertRaises(RuntimeError):
                    self._run(directory, executor=lambda _cell: (_ for _ in ()).throw(RuntimeError("stop")))
                with self.assertRaises(ResumeFailure) as context:
                    self._run(
                        directory,
                        selections=selections,
                        metadata=metadata,
                        candidates=candidates,
                    )
                self.assertEqual(context.exception.kind, expected_kind)

    def test_incomplete_run_never_assembles_or_publishes(self):
        with tempfile.TemporaryDirectory(prefix="whitewater-session-") as temporary:
            directory = Path(temporary)
            with self.assertRaises(RuntimeError):
                self._run(directory, executor=lambda _cell: (_ for _ in ()).throw(RuntimeError("failed")))
            self.assertFalse((directory / "report.json").exists())
            self.assertFalse((directory / "report.csv").exists())

    def test_existing_outputs_are_protected_by_no_clobber_default(self):
        with tempfile.TemporaryDirectory(prefix="whitewater-session-") as temporary:
            directory = Path(temporary)
            self._run(directory)
            json_path, csv_path = directory / "report.json", directory / "report.csv"
            original_json, original_csv = json_path.read_bytes(), csv_path.read_bytes()
            called = []
            with self.assertRaises(ReportFailure) as context:
                self._run(directory, executor=lambda cell: (called.append(cell), self._execution_for(cell))[1])
            self.assertEqual(context.exception.kind, "output_exists")
            self.assertEqual(called, [])
            self.assertEqual(json_path.read_bytes(), original_json)
            self.assertEqual(csv_path.read_bytes(), original_csv)

    def test_state_symlink_and_nonregular_path_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="whitewater-session-") as temporary:
            directory = Path(temporary)
            target = directory / "target.json"
            target.write_text("{}", encoding="utf-8")
            (directory / "state.json").symlink_to(target)
            with self.assertRaises(ResumeFailure) as context:
                self._run(directory)
            self.assertEqual(context.exception.kind, "symlink_state")

        with tempfile.TemporaryDirectory(prefix="whitewater-session-") as temporary:
            directory = Path(temporary)
            (directory / "state.json").mkdir()
            with self.assertRaises(ResumeFailure) as context:
                self._run(directory)
            self.assertEqual(context.exception.kind, "nonregular_state")

    def test_metadata_profile_and_environment_must_match_before_execution(self):
        for field, value, kind in (
            ("profile", "final", "metadata_profile"),
            ("environment", "macos-arm64", "metadata_environment"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory(prefix="whitewater-session-") as temporary:
                metadata = copy.deepcopy(self.metadata)
                metadata[field] = value
                called = []
                with self.assertRaises(SessionFailure) as context:
                    self._run(
                        Path(temporary),
                        metadata=metadata,
                        executor=lambda cell: (called.append(cell), self._execution_for(cell))[1],
                    )
                self.assertEqual(context.exception.kind, kind)
                self.assertEqual(called, [])
                self.assertFalse((Path(temporary) / "state.json").exists())

    def test_identity_canonicalization_rejects_nonfinite_values_as_typed_failure(self):
        plan = build_matrix(
            self.protocol,
            self.corpus,
            self.candidates,
            self.selections,
            "screen",
            "el8-x86_64",
        )
        malformed_inputs = []
        bad_protocol = copy.deepcopy(self.protocol)
        bad_protocol["identity_marker"] = float("nan")
        malformed_inputs.append(("protocol", bad_protocol, self.corpus, self.report_schema, self.corpus_schema, self.candidates, self.metadata))
        bad_corpus = copy.deepcopy(self.corpus)
        bad_corpus["cycle"] = bad_corpus
        malformed_inputs.append(("corpus", self.protocol, bad_corpus, self.report_schema, self.corpus_schema, self.candidates, self.metadata))
        bad_report_schema = copy.deepcopy(self.report_schema)
        bad_report_schema["nonfinite_marker"] = float("nan")
        malformed_inputs.append(("report_schema", self.protocol, self.corpus, bad_report_schema, self.corpus_schema, self.candidates, self.metadata))
        bad_corpus_schema = copy.deepcopy(self.corpus_schema)
        bad_corpus_schema["nonjson_marker"] = object()
        malformed_inputs.append(("corpus_schema", self.protocol, self.corpus, self.report_schema, bad_corpus_schema, self.candidates, self.metadata))
        bad_candidates = copy.deepcopy(self.candidates)
        bad_candidates[0]["nonfinite_marker"] = float("nan")
        malformed_inputs.append(("candidate_entries", self.protocol, self.corpus, self.report_schema, self.corpus_schema, bad_candidates, self.metadata))
        bad_metadata = copy.deepcopy(self.metadata)
        bad_metadata["cycle"] = bad_metadata
        malformed_inputs.append(("report_metadata", self.protocol, self.corpus, self.report_schema, self.corpus_schema, self.candidates, bad_metadata))
        for label, protocol, corpus, report_schema, corpus_schema, candidates, metadata in malformed_inputs:
            with self.subTest(label=label):
                with self.assertRaises(SessionFailure) as context:
                    resume_identity(
                        protocol,
                        corpus,
                        report_schema,
                        corpus_schema,
                        candidates,
                        plan,
                        "screen",
                        "el8-x86_64",
                        metadata,
                    )
                self.assertEqual(context.exception.kind, "identity_json")
                self.assertEqual(context.exception.failure_type, "session_failure")

    def test_state_inspection_oserror_is_typed(self):
        with tempfile.TemporaryDirectory(prefix="whitewater-session-") as temporary:
            state_path = Path(temporary) / "state.json"
            with mock.patch.object(Path, "lstat", side_effect=OSError("unreadable")):
                with self.assertRaises(SessionFailure) as context:
                    self._run(Path(temporary))
            self.assertEqual(context.exception.kind, "state_path")


if __name__ == "__main__":
    unittest.main()
