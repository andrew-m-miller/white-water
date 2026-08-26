#!/usr/bin/env python3
"""P25-6 carried driver-input template tests.

WP3 carries ready-to-run, schema-valid driver inputs under ``bakeoff/p25-6/inputs/`` so the
airgapped operator does not have to author them from prose (the older ``candidate_id`` +
``measurement_providers``-only candidate shape is rejected by the carried protocol-v2 matrix
validator before any cell runs).  These tests load each carried template and prove it validates
against the exact contracts the driver enforces on the box:

* every ``candidate-entries.json`` entry carries both v2 decisions (``status`` and
  ``measurement_status``) and validates against the carried report-v2 candidate schema;
* ``corpus.template.json`` validates against the carried corpus-v1 schema;
* the ``smoke`` and ``screen`` selections expand through the real ``matrix.build_matrix`` against
  protocol-v2, the carried candidate entries and the carried corpus template;
* the NeuFlow shared-lattice screen is spelled exactly as protocol-v2 prescribes but is correctly
  rejected until ``neuflow-v2`` is admitted and added to the candidate entries;
* ``artifact-map.json`` points at the carried manifest + ONNX destinations; and
* every carried input is registered in ``package-spec.json`` at mode 0644.

They import only the dependency-free planning/validation modules (``matrix``, ``validator``), so
they run in the ordinary suite without numpy, onnxruntime, OpenImageIO, pynvml, or a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
import stat
import unittest

try:
    from tools.bakeoff import matrix as matrix_module
    from tools.bakeoff import validator as validator_module
except ImportError:  # Direct execution from the repo root.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.bakeoff import matrix as matrix_module
    from tools.bakeoff import validator as validator_module


REPO_ROOT = Path(__file__).resolve().parents[2]
BAKEOFF = REPO_ROOT / "bakeoff"
INPUTS_DIR = BAKEOFF / "p25-6" / "inputs"
PACKAGE_SPEC = BAKEOFF / "p25-6" / "package-spec.json"

PROTOCOL = json.loads((BAKEOFF / "protocol-v2.json").read_text(encoding="utf-8"))
REPORT_SCHEMA = json.loads((BAKEOFF / "report-v2.schema.json").read_text(encoding="utf-8"))
CORPUS_SCHEMA = json.loads((BAKEOFF / "corpus-v1.schema.json").read_text(encoding="utf-8"))

_SELECTION_AXES = ("candidate_ids", "shot_ids", "conditioning_tokens", "cap_tokens", "providers")

CARRIED_INPUTS = (
    "candidate-entries.json",
    "artifact-map.json",
    "selection-smoke.json",
    "selection-screen.json",
    "selection-screen-neuflow-shared-lattice.json",
    "report-metadata.json",
    "corpus.template.json",
)


def _load(name: str):
    return json.loads((INPUTS_DIR / name).read_text(encoding="utf-8"))


def _selection_axes(selection):
    return {key: selection[key] for key in _SELECTION_AXES}


class P25_6InputTemplateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = _load("candidate-entries.json")
        self.corpus = _load("corpus.template.json")
        self.spec = json.loads(PACKAGE_SPEC.read_text(encoding="utf-8"))
        self.spec_by_destination = {item["destination"]: item for item in self.spec["files"]}

    # -- candidate entries -------------------------------------------------------------------

    def test_candidate_entries_validate_report_v2_and_carry_both_statuses(self) -> None:
        self.assertIsInstance(self.candidates, list)
        self.assertTrue(self.candidates)
        candidate_schema = REPORT_SCHEMA["$defs"]["candidate"]
        for entry in self.candidates:
            # Both v2 decisions must be present -- this is the exact defect the template fixes.
            self.assertIn("status", entry)
            self.assertIn("measurement_status", entry)
            # Full report-v2 candidate validation (the oneOf branch for its status pair, all
            # required identity/license fields, hex patterns, etc.).
            validator_module.validate(entry, candidate_schema, root=REPORT_SCHEMA)

    def test_carried_candidate_is_measurable_sea_raft_m(self) -> None:
        ids = {entry["candidate_id"] for entry in self.candidates}
        self.assertEqual(ids, {"sea-raft-m"})
        entry = self.candidates[0]
        self.assertEqual(entry["measurement_status"], "measurable")
        self.assertIn("cpu", entry["measurement_providers"])

    def test_candidate_identity_matches_committed_manifest(self) -> None:
        """The template's identity hashes are the real reviewed values, not fabricated."""

        manifest = json.loads((REPO_ROOT / "models" / "sea-raft-m.json").read_text(encoding="utf-8"))
        entry = self.candidates[0]
        self.assertEqual(entry["artifact_sha256"], manifest["export"]["sha256"])
        self.assertEqual(entry["artifact_size_bytes"], manifest["export"]["size_bytes"])
        self.assertEqual(entry["export_environment_sha256"], manifest["export"]["export_environment_sha256"])
        self.assertEqual(entry["checkpoint_sha256"], manifest["checkpoint"]["sha256"])
        self.assertEqual(entry["source_commit"], manifest["upstream"]["commit"])

    # -- corpus ------------------------------------------------------------------------------

    def test_corpus_template_validates_corpus_v1(self) -> None:
        validator_module.validate(self.corpus, CORPUS_SCHEMA, root=CORPUS_SCHEMA)

    def test_corpus_declares_the_selection_shots(self) -> None:
        shot_ids = {
            shot["id"]
            for partition in self.corpus["partitions"]
            for shot in partition["shots"]
        }
        for needed in ("prod-smoke-sample", "syn-fhd-1920x1080-par1", "syn-uhd-3840x2160-par1"):
            self.assertIn(needed, shot_ids)

    # -- selections build through the real matrix planner ------------------------------------

    def test_smoke_and_screen_selections_build_matrix(self) -> None:
        for name in ("selection-smoke.json", "selection-screen.json"):
            selection = _load(name)
            plan = matrix_module.build_matrix(
                PROTOCOL,
                self.corpus,
                self.candidates,
                _selection_axes(selection),
                selection["profile"],
                selection["environment"],
            )
            self.assertTrue(plan.cells, f"{name} produced an empty matrix")

    def test_selection_axes_are_protocol_v2_valid(self) -> None:
        protocol_candidate_ids = {c["id"] for c in PROTOCOL["candidate_ids"]}
        conditioning = {c["token"] for c in PROTOCOL["conditioning"]}
        caps = {c["token"] for c in PROTOCOL["analysis_caps"]}
        providers = {p["token"] for p in PROTOCOL["providers"]}
        for name in ("selection-smoke.json", "selection-screen.json"):
            selection = _load(name)
            self.assertIn(selection["profile"], {"smoke", "screen", "final"})
            self.assertEqual(selection["environment"], "el8-x86_64")
            self.assertTrue(set(selection["candidate_ids"]) <= protocol_candidate_ids)
            self.assertTrue(set(selection["conditioning_tokens"]) <= conditioning)
            self.assertTrue(set(selection["cap_tokens"]) <= caps)
            for provider in selection["providers"]:
                self.assertIn(provider["token"], providers)

    # -- NeuFlow shared lattice: carried, prescribed, but gated on admission -----------------

    def test_neuflow_shared_lattice_matches_protocol_and_requires_admission(self) -> None:
        selection = _load("selection-screen-neuflow-shared-lattice.json")
        protocol_candidate_ids = {c["id"] for c in PROTOCOL["candidate_ids"]}
        # Its axes are real protocol-v2 values ...
        self.assertTrue(set(selection["candidate_ids"]) <= protocol_candidate_ids)
        self.assertEqual(selection["cap_tokens"], ["mp0_331776"])
        self.assertIn("neuflow-v2", selection["candidate_ids"])
        # ... but neuflow-v2 is not carried in candidate-entries.json, so the matrix must reject
        # it rather than silently scheduling an unadmitted candidate.
        carried_ids = {entry["candidate_id"] for entry in self.candidates}
        self.assertNotIn("neuflow-v2", carried_ids)
        with self.assertRaises(matrix_module.MatrixFailure):
            matrix_module.build_matrix(
                PROTOCOL,
                self.corpus,
                self.candidates,
                _selection_axes(selection),
                selection["profile"],
                selection["environment"],
            )

    # -- artifact map ------------------------------------------------------------------------

    def test_artifact_map_points_at_carried_manifest_and_onnx(self) -> None:
        artifact_map = _load("artifact-map.json")
        self.assertEqual(set(artifact_map), {"sea-raft-m"})
        entry = artifact_map["sea-raft-m"]
        self.assertEqual(entry["manifest"], "models/sea-raft-m/manifest.json")
        self.assertEqual(entry["artifact"], "models/sea-raft-m/sea-raft-m-opset17.onnx")
        # Those destinations are the carried candidate manifest + model artifact.
        manifest_item = self.spec_by_destination.get(entry["manifest"])
        artifact_item = self.spec_by_destination.get(entry["artifact"])
        self.assertIsNotNone(manifest_item)
        self.assertIsNotNone(artifact_item)
        self.assertEqual(manifest_item["role"], "candidate-manifest")
        self.assertEqual(artifact_item["role"], "model-artifact")

    # -- report metadata ---------------------------------------------------------------------

    def test_report_metadata_shape_and_marked_placeholders(self) -> None:
        metadata = _load("report-metadata.json")
        runner = metadata["runner"]
        hardware = metadata["hardware"]
        # Known-fixed fields are concrete.
        self.assertEqual(runner["name"], "ww-bakeoff")
        self.assertEqual(
            runner["runtime"], self.spec["evaluator"]["runtime_identity"]
        )
        self.assertEqual(hardware["platform"], "linux")
        self.assertEqual(hardware["architecture"], "x86_64")
        # Host/CI-specific values are clearly-marked placeholders (never fabricated hashes).
        for field in ("source_commit", "evaluator_sha256", "runtime_sha256"):
            self.assertIn("REPLACE_WITH", runner[field], field)
        for field in ("os_release", "cpu", "gpu", "driver"):
            self.assertIn("REPLACE_WITH", hardware[field], field)

    # -- carried in the package spec ---------------------------------------------------------

    def test_every_input_is_carried_at_0644(self) -> None:
        for name in CARRIED_INPUTS:
            destination = f"inputs/{name}"
            item = self.spec_by_destination.get(destination)
            self.assertIsNotNone(item, f"{destination} is not carried in package-spec.json")
            self.assertEqual(item["role"], "evaluator-support", destination)
            self.assertEqual(item["mode"], "0644", destination)
            source = (BAKEOFF / "p25-6" / item["source"]).resolve()
            self.assertTrue(source.is_file(), source)
            self.assertFalse(source.is_symlink(), source)
            self.assertEqual(stat.S_IMODE(source.stat().st_mode), 0o644, source)


if __name__ == "__main__":
    unittest.main()
