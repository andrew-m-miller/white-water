#!/usr/bin/env python3
"""P25-6 carried driver-input template tests.

WP3 carries ready-to-run, schema-valid driver inputs under ``bakeoff/p25-6/inputs/`` so the
airgapped operator does not have to author them from prose (the older ``candidate_id`` +
``measurement_providers``-only candidate shape is rejected by the carried protocol-v2 matrix
validator before any cell runs).  These tests are the authoritative gate over every carried
input: each one must validate against the exact contract the driver enforces on the box, each
carried selection must expand through the real ``matrix.build_matrix`` for its declared profile,
and the candidate/artifact-map identity must be target-correct (linux, CI-materialized).

Two review findings on PR #21 shaped the current shape and are enforced here:

* Finding A -- the checked-in ``candidate-entries.json``/``artifact-map.json`` must NOT ship a
  macOS-arm64 identity.  The candidate ONNX is not committed; CI exports a fresh ``linux-x86_64``
  artifact and rewrites ``models/sea-raft-m.json`` before packaging, so these two files are
  platform-neutral PLACEHOLDER templates whose platform-specific identity fields are filled from
  the generated linux manifest by ``tools/p25_5/p25_6_materialize_inputs.py`` (called from
  ``scripts/ci-p25-6-qualify.sh``).  The tests assert the checked-in files carry the placeholder
  (never the macOS binding) and that materializing against a staged linux manifest fixture yields
  a report-v2-valid entry bound to that manifest's linux row.
* Finding B -- an exact ``final`` selection is carried (``selection-final.json``) so the operator
  never has to hand-edit a carried selection into a final.  The tests build all three carried
  selections (smoke, screen, final) and assert the final one satisfies the final-coverage rule.

They import only the dependency-free planning/validation modules (``matrix``, ``validator``) plus
the pure-Python materializer, so they run in the ordinary suite without numpy, onnxruntime,
OpenImageIO, pynvml, or a GPU.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
import tempfile
import unittest

try:
    from tools.bakeoff import matrix as matrix_module
    from tools.bakeoff import validator as validator_module
    from tools.p25_5 import p25_6_materialize_inputs as materialize_module
except ImportError:  # Direct execution from the repo root.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.bakeoff import matrix as matrix_module
    from tools.bakeoff import validator as validator_module
    from tools.p25_5 import p25_6_materialize_inputs as materialize_module


REPO_ROOT = Path(__file__).resolve().parents[2]
BAKEOFF = REPO_ROOT / "bakeoff"
INPUTS_DIR = BAKEOFF / "p25-6" / "inputs"
PACKAGE_SPEC = BAKEOFF / "p25-6" / "package-spec.json"

PROTOCOL = json.loads((BAKEOFF / "protocol-v2.json").read_text(encoding="utf-8"))
REPORT_SCHEMA = json.loads((BAKEOFF / "report-v2.schema.json").read_text(encoding="utf-8"))
CORPUS_SCHEMA = json.loads((BAKEOFF / "corpus-v1.schema.json").read_text(encoding="utf-8"))

PLACEHOLDER = materialize_module.PLACEHOLDER
# The stale macOS export identity the shipped inputs must never carry (models/sea-raft-m.json).
MACOS_ARTIFACT_SHA256 = "23cc2c850d3c116df193a24ff9ae7722d5635cd04e75dd8aeb20d7e13e4f59f1"

_SELECTION_AXES = ("candidate_ids", "shot_ids", "conditioning_tokens", "cap_tokens", "providers")

CARRIED_SELECTIONS = (
    "selection-smoke.json",
    "selection-screen.json",
    "selection-final.json",
)

CARRIED_INPUTS = (
    "candidate-entries.json",
    "artifact-map.json",
    "selection-smoke.json",
    "selection-screen.json",
    "selection-screen-neuflow-shared-lattice.json",
    "selection-final.json",
    "report-metadata.json",
    "corpus.template.json",
)


def _load(name: str):
    return json.loads((INPUTS_DIR / name).read_text(encoding="utf-8"))


def _selection_axes(selection):
    return {key: selection[key] for key in _SELECTION_AXES}


def _staged_linux_manifest():
    """A minimal freshly-exported linux-x86_64 manifest fixture with obviously-synthetic hashes.

    The identity hex is fake on purpose: these tests prove the materializer copies the exported
    linux row through faithfully and yields a schema-valid entry, not the real (unfabricated)
    linux hashes -- those come from CI's actual export.
    """

    linux_env_sha = "b" * 64
    return {
        "schema_id": "whitewater-p25-artifact-v1",
        "candidate": {"id": "sea-raft-m", "role": "measurement-candidate"},
        "export": {
            "artifact": "sea-raft-m-opset17.onnx",
            "platform": "linux-x86_64",
            "sha256": "a" * 64,
            "size_bytes": 78840999,
            "mode": "0644",
            "export_environment_sha256": linux_env_sha,
            "platform_artifacts": [
                {
                    "platform": "linux-x86_64",
                    "artifact": "sea-raft-m-opset17.onnx",
                    "sha256": "a" * 64,
                    "size_bytes": 78840999,
                    "mode": "0644",
                    "export_environment_sha256": linux_env_sha,
                }
            ],
        },
    }


class P25_6InputTemplateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = _load("candidate-entries.json")
        self.artifact_map = _load("artifact-map.json")
        self.corpus = _load("corpus.template.json")
        self.spec = json.loads(PACKAGE_SPEC.read_text(encoding="utf-8"))
        self.spec_by_destination = {item["destination"]: item for item in self.spec["files"]}

    # -- candidate entries: checked-in is a linux-materializable placeholder -----------------

    def test_candidate_entries_carry_both_statuses(self) -> None:
        self.assertIsInstance(self.candidates, list)
        self.assertTrue(self.candidates)
        for entry in self.candidates:
            # Both v2 decisions must be present in the carried template regardless of platform.
            self.assertIn("status", entry)
            self.assertIn("measurement_status", entry)

    def test_carried_candidate_is_measurable_sea_raft_m(self) -> None:
        ids = {entry["candidate_id"] for entry in self.candidates}
        self.assertEqual(ids, {"sea-raft-m"})
        entry = self.candidates[0]
        self.assertEqual(entry["measurement_status"], "measurable")
        self.assertIn("cpu", entry["measurement_providers"])

    def test_checked_in_candidate_entries_are_placeholder_not_macos(self) -> None:
        """Finding A: the shipped template must NOT carry the macOS-arm64 export identity."""

        entry = self.candidates[0]
        for field in materialize_module.CANDIDATE_IDENTITY_FIELDS:
            self.assertEqual(
                entry.get(field),
                PLACEHOLDER,
                f"{field} must be the linux-materialization placeholder, not a shipped binding",
            )
        # Explicitly assert the stale macOS artifact hash is gone.
        self.assertNotEqual(entry.get("artifact_sha256"), MACOS_ARTIFACT_SHA256)
        # Non-identity provenance/legal fields are carried verbatim (never materialized).
        manifest = json.loads((REPO_ROOT / "models" / "sea-raft-m.json").read_text(encoding="utf-8"))
        self.assertEqual(entry["checkpoint_sha256"], manifest["checkpoint"]["sha256"])
        self.assertEqual(entry["source_commit"], manifest["upstream"]["commit"])

    def test_checked_in_artifact_map_is_placeholder_not_macos(self) -> None:
        """Finding A: artifact-map platform must not select the macOS row on the linux box."""

        entry = self.artifact_map["sea-raft-m"]
        self.assertEqual(entry.get("platform"), PLACEHOLDER)
        self.assertNotEqual(entry.get("platform"), "macos-arm64")

    def test_materialized_candidate_entry_binds_linux_manifest_and_validates(self) -> None:
        """Materializing against a staged linux manifest yields a report-v2-valid linux entry."""

        manifest = _staged_linux_manifest()
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "sea-raft-m.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            materialized = materialize_module.materialize_candidate_entries(
                self.candidates, manifest, manifest_path
            )
            expected_manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

        entry = materialized[0]
        row = manifest["export"]["platform_artifacts"][0]
        self.assertEqual(entry["artifact_sha256"], row["sha256"])
        self.assertEqual(entry["export_environment_sha256"], row["export_environment_sha256"])
        self.assertEqual(entry["artifact_size_bytes"], row["size_bytes"])
        self.assertEqual(entry["manifest_sha256"], expected_manifest_sha)
        # It is now the real linux identity, not the macOS one, and it is a full report-v2 candidate.
        self.assertNotEqual(entry["artifact_sha256"], MACOS_ARTIFACT_SHA256)
        validator_module.validate(entry, REPORT_SCHEMA["$defs"]["candidate"], root=REPORT_SCHEMA)

    def test_materialized_artifact_map_binds_linux_platform(self) -> None:
        manifest = _staged_linux_manifest()
        materialized = materialize_module.materialize_artifact_map(self.artifact_map, manifest)
        self.assertEqual(materialized["sea-raft-m"]["platform"], "linux-x86_64")
        # Manifest/artifact package-relative paths are unchanged by materialization.
        self.assertEqual(materialized["sea-raft-m"]["manifest"], "models/sea-raft-m/manifest.json")
        self.assertEqual(
            materialized["sea-raft-m"]["artifact"], "models/sea-raft-m/sea-raft-m-opset17.onnx"
        )

    def test_materialize_refuses_non_linux_manifest(self) -> None:
        """Handing the checked-in macOS manifest to the materializer fails closed."""

        macos_manifest = json.loads(
            (REPO_ROOT / "models" / "sea-raft-m.json").read_text(encoding="utf-8")
        )
        self.assertEqual(macos_manifest["export"]["platform"], "macos-arm64")
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "sea-raft-m.json"
            manifest_path.write_text(json.dumps(macos_manifest), encoding="utf-8")
            with self.assertRaises(materialize_module.MaterializeError):
                materialize_module.materialize_candidate_entries(
                    self.candidates, macos_manifest, manifest_path
                )
            with self.assertRaises(materialize_module.MaterializeError):
                materialize_module.materialize_artifact_map(self.artifact_map, macos_manifest)

    def test_materialize_refuses_hardcoded_identity(self) -> None:
        """A template that hardcodes a real hash (rather than the placeholder) is rejected."""

        manifest = _staged_linux_manifest()
        tampered = [dict(self.candidates[0])]
        tampered[0]["artifact_sha256"] = MACOS_ARTIFACT_SHA256
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "sea-raft-m.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(materialize_module.MaterializeError):
                materialize_module.materialize_candidate_entries(tampered, manifest, manifest_path)

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

    def test_all_carried_selections_build_matrix(self) -> None:
        """smoke, screen AND final each expand for their own declared profile."""

        for name in CARRIED_SELECTIONS:
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

    def test_final_selection_satisfies_final_coverage(self) -> None:
        """Finding B: the carried final selection satisfies matrix._validate_final_coverage.

        build_matrix runs that rule internally and raises when it is not met, so a successful
        plan already proves coverage; this additionally pins the exact axes the rule requires
        (profile=final, CUDA idle+live_flame, cap mp2, FHD+UHD PAR1 shots).
        """

        selection = _load("selection-final.json")
        self.assertEqual(selection["profile"], "final")
        self.assertEqual(selection["cap_tokens"], ["mp2"])
        self.assertEqual(
            selection["shot_ids"], ["syn-fhd-1920x1080-par1", "syn-uhd-3840x2160-par1"]
        )
        self.assertEqual(len(selection["providers"]), 1)
        provider = selection["providers"][0]
        self.assertEqual(provider["token"], "cuda")
        self.assertEqual(set(provider["host_loads"]), {"idle", "live_flame"})

        plan = matrix_module.build_matrix(
            PROTOCOL,
            self.corpus,
            self.candidates,
            _selection_axes(selection),
            "final",
            selection["environment"],
        )
        # mp2 + CUDA on both host loads across the FHD and UHD PAR1 shots.
        self.assertEqual({cell.cap for cell in plan.cells}, {"mp2"})
        self.assertEqual({cell.provider for cell in plan.cells}, {"cuda"})
        self.assertEqual({cell.host_load for cell in plan.cells}, {"idle", "live_flame"})
        self.assertEqual(
            {cell.shot for cell in plan.cells},
            {"syn-fhd-1920x1080-par1", "syn-uhd-3840x2160-par1"},
        )

    def test_final_coverage_rejects_dropping_a_required_shot(self) -> None:
        """Sanity: the same axes minus the UHD shot are rejected, proving the rule is live."""

        selection = _load("selection-final.json")
        axes = _selection_axes(selection)
        axes["shot_ids"] = ["syn-fhd-1920x1080-par1"]
        with self.assertRaises(matrix_module.MatrixFailure):
            matrix_module.build_matrix(
                PROTOCOL, self.corpus, self.candidates, axes, "final", selection["environment"]
            )

    def test_selection_axes_are_protocol_v2_valid(self) -> None:
        protocol_candidate_ids = {c["id"] for c in PROTOCOL["candidate_ids"]}
        conditioning = {c["token"] for c in PROTOCOL["conditioning"]}
        caps = {c["token"] for c in PROTOCOL["analysis_caps"]}
        providers = {p["token"] for p in PROTOCOL["providers"]}
        for name in CARRIED_SELECTIONS:
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
        entry = self.artifact_map["sea-raft-m"]
        self.assertEqual(set(self.artifact_map), {"sea-raft-m"})
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
