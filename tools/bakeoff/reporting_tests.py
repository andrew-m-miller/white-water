#!/usr/bin/env python3
"""Focused tests for deterministic Phase 2.5 report assembly and publication."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock

from . import reporting
from .matrix import CellKey, MatrixPlan
from .reporting import (
    CSV_HEADER,
    ReportFailure,
    assemble_report,
    render_csv,
    render_json,
    write_report_pair,
)
from .validator import validate_report_consistency


ROOT = Path(__file__).resolve().parents[2]


def _load(relative: str):
    with (ROOT / relative).open(encoding="utf-8") as stream:
        return json.load(stream)


class ReportingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = _load("bakeoff/protocol-v1.json")
        self.corpus = _load("bakeoff/fixtures/positive/corpus-v1.json")
        self.fixture = _load("bakeoff/fixtures/positive/report-v1.json")
        self.report_schema = _load("bakeoff/report-v1.schema.json")
        self.corpus_schema = _load("bakeoff/corpus-v1.schema.json")
        self.cell = CellKey(
            "sea-raft-m", "syn-identity", "native-clamp01-v1", "mp0_5", "cpu", "not_applicable"
        )
        self.plan = MatrixPlan(self.fixture["matrix"], (self.cell,), ())
        excluded = {
            "schema_version", "protocol_id", "corpus_sha256", "matrix",
            "candidates", "results", "summary",
        }
        self.metadata = {key: value for key, value in self.fixture.items() if key not in excluded}

    def assembled(self, results=None):
        return assemble_report(
            self.protocol, self.corpus, self.report_schema, self.corpus_schema,
            self.metadata, self.fixture["candidates"], self.plan,
            self.fixture["results"] if results is None else results,
        )

    def test_positive_assembly_matches_fixture_and_validator(self):
        report = self.assembled()
        self.assertEqual(report, self.fixture)
        validate_report_consistency(
            report, self.protocol, self.report_schema, self.corpus, self.corpus_schema
        )

    def test_results_normalize_to_plan_order_and_resume_records_are_supported(self):
        result = dict(self.fixture["results"][0])
        resume = {"state": "complete", "cell": self.cell.as_dict(), "result": result}
        self.assertEqual(self.assembled([resume]), self.fixture)

    def test_metadata_cannot_supply_generated_fields(self):
        for field in ("schema_version", "protocol_id", "corpus_sha256", "matrix", "candidates", "results"):
            with self.subTest(field=field):
                metadata = {**self.metadata, field: self.fixture.get(field, 1)}
                with self.assertRaises(ReportFailure) as context:
                    assemble_report(
                        self.protocol, self.corpus, self.report_schema, self.corpus_schema,
                        metadata, self.fixture["candidates"], self.plan, self.fixture["results"],
                    )
                self.assertEqual(context.exception.kind, "metadata_shape")

    def test_csv_has_frozen_header_and_canonical_nested_columns(self):
        report = self.assembled()
        payload = render_csv(report)
        self.assertEqual(payload.decode("utf-8").count("\n"), 2)
        rows = list(csv.reader(payload.decode("utf-8").splitlines()))
        self.assertEqual(tuple(rows[0]), CSV_HEADER)
        self.assertEqual(len(rows), 2)
        nested = rows[1][CSV_HEADER.index("steady_samples_ms_json")]
        self.assertEqual(json.loads(nested), report["results"][0]["timing"]["steady_samples_ms"])
        self.assertIn('"', payload.decode("utf-8"))

    def test_csv_rejects_malformed_public_inputs_with_typed_failures(self):
        report = self.assembled()
        for malformed, kind in (
            ({**report, "results": None}, "result_shape"),
            ({**report, "results": [None]}, "result_shape"),
            ({**report, "results": [{**report["results"][0], "geometry": []}]}, "result_shape"),
            ({**report, "results": [{**report["results"][0], "timing": "bad"}]}, "result_shape"),
            ({**report, "results": [{**report["results"][0], "metrics": 1}]}, "result_shape"),
            ({**report, "results": [{**report["results"][0], "resource": False}]}, "result_shape"),
            ({**report, "results": [{**report["results"][0], "environment": ["bad"]}]}, "result_shape"),
        ):
            with self.subTest(kind=kind, malformed=malformed.get("results")):
                with self.assertRaises(ReportFailure) as context:
                    render_csv(malformed)
                self.assertEqual(context.exception.kind, kind)

    def test_deterministic_pair_bytes_and_modes(self):
        report = self.assembled()
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            paths = []
            for directory in (Path(first), Path(second)):
                json_path, csv_path = directory / "report.json", directory / "report.csv"
                write_report_pair(
                    json_path, csv_path, report, self.protocol, self.report_schema,
                    self.corpus, self.corpus_schema,
                )
                paths.append((json_path, csv_path))
                self.assertEqual(stat.S_IMODE(json_path.stat().st_mode), 0o644)
                self.assertEqual(stat.S_IMODE(csv_path.stat().st_mode), 0o644)
                self.assertEqual(list(directory.glob(".*.tmp")), [])
            self.assertEqual(paths[0][0].read_bytes(), paths[1][0].read_bytes())
            self.assertEqual(paths[0][1].read_bytes(), paths[1][1].read_bytes())
            self.assertEqual(json.loads(render_json(report)), report)

    def test_missing_duplicate_extra_and_mismatched_results_fail_typed(self):
        for results, kind in (([], "result_count"), (self.fixture["results"] + self.fixture["results"], "result_count")):
            with self.subTest(kind=kind):
                with self.assertRaises(ReportFailure) as context:
                    self.assembled(results)
                self.assertEqual(context.exception.kind, kind)
        wrong = dict(self.fixture["results"][0])
        wrong["shot_id"] = "not-a-plan-shot"
        with self.assertRaises(ReportFailure) as context:
            self.assembled([wrong])
        self.assertIn(context.exception.kind, {"extra_result", "validation"})
        resume = {"state": "in_progress", "cell": self.cell.as_dict(), "result": self.fixture["results"][0]}
        with self.assertRaises(ReportFailure) as context:
            self.assembled([resume])
        self.assertEqual(context.exception.kind, "incomplete_result")

    def test_nonfinite_and_cell_mismatch_fail_typed(self):
        bad = dict(self.fixture["results"][0])
        bad["metrics"] = dict(bad["metrics"])
        bad["metrics"]["endpoint_error_px"] = float("nan")
        with self.assertRaises(ReportFailure) as context:
            self.assembled([bad])
        self.assertEqual(context.exception.kind, "nonfinite")
        resume = {
            "state": "complete",
            "cell": self.cell.as_dict(),
            "result": {**self.fixture["results"][0], "shot_id": "other-shot"},
        }
        with self.assertRaises(ReportFailure) as context:
            self.assembled([resume])
        self.assertEqual(context.exception.kind, "cell_mismatch")

    def test_destination_protection(self):
        report = self.assembled()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            json_path, csv_path = directory / "report.json", directory / "report.csv"
            write_report_pair(
                json_path, csv_path, report, self.protocol, self.report_schema,
                self.corpus, self.corpus_schema,
            )
            with self.assertRaises(ReportFailure) as context:
                write_report_pair(
                    json_path, directory / "second.csv", report, self.protocol,
                    self.report_schema, self.corpus, self.corpus_schema,
                )
            self.assertEqual(context.exception.kind, "output_exists")
            link = directory / "link.json"
            link.symlink_to(json_path)
            with self.assertRaises(ReportFailure) as context:
                write_report_pair(
                    link, directory / "link.csv", report, self.protocol,
                    self.report_schema, self.corpus, self.corpus_schema,
                )
            self.assertEqual(context.exception.kind, "symlink_output")
            with self.assertRaises(ReportFailure) as context:
                write_report_pair(
                    directory, directory / "dir.csv", report, self.protocol,
                    self.report_schema, self.corpus, self.corpus_schema,
                )
            self.assertEqual(context.exception.kind, "nonregular_output")
            mode_path = directory / "mode.json"
            mode_path.write_bytes(b"old")
            os.chmod(mode_path, 0o600)
            with self.assertRaises(ReportFailure) as context:
                write_report_pair(
                    mode_path, directory / "mode.csv", report, self.protocol,
                    self.report_schema, self.corpus, self.corpus_schema,
                )
            self.assertEqual(context.exception.kind, "output_mode")

    def test_no_clobber_collision_at_publication_preserves_competitor(self):
        report = self.assembled()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            json_path, csv_path = directory / "report.json", directory / "report.csv"
            competitor = b"created by another writer"

            def racing_link(source, destination, **kwargs):
                Path(destination).write_bytes(competitor)
                raise FileExistsError(destination)

            with mock.patch.object(reporting.os, "link", side_effect=racing_link):
                with self.assertRaises(ReportFailure) as context:
                    write_report_pair(
                        json_path, csv_path, report, self.protocol, self.report_schema,
                        self.corpus, self.corpus_schema,
                    )
            self.assertEqual(context.exception.kind, "output_exists")
            self.assertEqual(json_path.read_bytes(), competitor)
            self.assertFalse(csv_path.exists())
            self.assertEqual(list(directory.glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
