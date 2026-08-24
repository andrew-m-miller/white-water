#!/usr/bin/env python3
"""Focused tests for P25-5 candidate admission generation."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest

from .admission import (
    ACTIVE_PROTOCOL_ID,
    AdmissionError,
    CandidateInput,
    generate_admission,
    write_admission,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "bakeoff" / "protocol-v2.json"
POSITIVE_MANIFEST = ROOT / "models" / "fixtures" / "positive" / "artifact-v1.json"
POSITIVE_ARTIFACT = ROOT / "models" / "fixtures" / "positive" / "valid.bin"


class AdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="whitewater-p25-admission-test-")
        self.root = Path(self.temp.name)
        self.manifest = self.root / "candidate.json"
        self.artifact = self.root / "valid.bin"
        shutil.copy2(POSITIVE_MANIFEST, self.manifest)
        shutil.copy2(POSITIVE_ARTIFACT, self.artifact)
        self.manifest.chmod(0o644)
        self.artifact.chmod(0o644)
        self._mutate_manifest(
            candidate_id="sea-raft-m",
            candidate_role="shipping-candidate",
            status="host_probe_cpu_cuda_passed",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _read_manifest(self) -> dict:
        return json.loads(self.manifest.read_text(encoding="utf-8"))

    def _write_manifest(self, value: dict) -> None:
        self.manifest.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        self.manifest.chmod(0o644)

    def _mutate_manifest(self, *, candidate_id: str, candidate_role: str, status: str) -> None:
        value = self._read_manifest()
        value["candidate"]["id"] = candidate_id
        value["candidate"]["role"] = candidate_role
        value["status"] = status
        if status == "excluded":
            value["exclusion"] = {"reason_code": "checkpoint_license_terms_unknown"}
        else:
            value.pop("exclusion", None)
        self._write_manifest(value)

    def _input(self, artifact: Path | None = None, platform: str | None = None) -> CandidateInput:
        return CandidateInput(self.manifest, artifact or self.artifact, platform)

    def _generate(
        self,
        candidates: list[CandidateInput] | None = None,
        providers: tuple[str, ...] = ("cpu",),
        **kwargs,
    ) -> dict:
        return generate_admission(
            PROTOCOL,
            candidates or [self._input()],
            providers,
            reviewed_surfaces=("code", "checkpoint", "backbone"),
            **kwargs,
        )

    def test_shipping_candidate_is_eligible_and_identity_bound(self) -> None:
        value = self._read_manifest()
        value["backbone"]["applicable"] = True
        value["backbone"]["checkpoint_sha256"] = "a" * 64
        self._write_manifest(value)
        document = self._generate(providers=("cuda", "cpu"))
        self.assertEqual(document["protocol_id"], ACTIVE_PROTOCOL_ID)
        self.assertEqual(document["measurement_providers"], ["cpu", "cuda"])
        candidate = document["candidates"][0]
        self.assertEqual(candidate["candidate_id"], "sea-raft-m")
        self.assertEqual(candidate["status"], "eligible")
        self.assertEqual(candidate["measurement_status"], "measurable")
        self.assertTrue(candidate["measurement_admitted"])
        self.assertEqual(candidate["artifact_size_bytes"], self.artifact.stat().st_size)
        self.assertEqual(candidate["artifact_sha256"], hashlib.sha256(self.artifact.read_bytes()).hexdigest())
        self.assertEqual(candidate["manifest_sha256"], hashlib.sha256(self.manifest.read_bytes()).hexdigest())
        self.assertEqual(candidate["backbone_sha256"], "a" * 64)
        self.assertNotIn("exclusion_reason", candidate)

    def test_validation_baseline_is_excluded_but_measurable_with_legal_surfaces(self) -> None:
        self._mutate_manifest(
            candidate_id="raft-original",
            candidate_role="validation-baseline",
            status="excluded",
        )
        value = self._read_manifest()
        value["licenses"]["checkpoint"]["commercial_use_permitted"] = "unknown"
        value["licenses"]["checkpoint"]["redistribution_permitted"] = "unknown"
        self._write_manifest(value)
        document = self._generate()
        candidate = document["candidates"][0]
        self.assertEqual(candidate["status"], "excluded")
        self.assertEqual(candidate["measurement_status"], "measurable")
        self.assertEqual(candidate["exclusion_reason"]["type"], "other")
        self.assertEqual(candidate["license_verdicts"]["checkpoint"], "unknown")
        self.assertEqual(candidate["redistribution_permitted"]["checkpoint"], "unknown")
        self.assertTrue(candidate["measurement_admitted"])

    def test_missing_artifact_requires_explicit_unavailable_request(self) -> None:
        missing = self.root / "valid.bin.missing"
        value = self._read_manifest()
        value["export"]["artifact"] = missing.name
        value["export"]["platform_artifacts"][0]["artifact"] = missing.name
        self._write_manifest(value)
        with self.assertRaisesRegex(AdmissionError, "allow-missing"):
            generate_admission(
                PROTOCOL,
                [self._input(missing)],
                ["cpu"],
                reviewed_surfaces=("code", "checkpoint", "backbone"),
            )

        document = self._generate([self._input(missing)], allow_missing=True)
        candidate = document["candidates"][0]
        self.assertEqual(candidate["measurement_status"], "unavailable")
        self.assertNotIn("measurement_admitted", candidate)
        self.assertNotIn("measurement_providers", candidate)
        self.assertEqual(candidate["measurement_exclusion_reason"]["type"], "artifact_missing")

    def test_hash_mismatch_is_not_downgraded_to_unavailable(self) -> None:
        self.artifact.write_bytes(b"changed bytes\n")
        self.artifact.chmod(0o644)
        with self.assertRaisesRegex(AdmissionError, "invalid exact artifact"):
            self._generate(allow_missing=True)

    def test_manifest_role_enforcement_prevents_baseline_shipping(self) -> None:
        self._mutate_manifest(
            candidate_id="raft-original",
            candidate_role="shipping-candidate",
            status="host_probe_cpu_cuda_passed",
        )
        with self.assertRaisesRegex(AdmissionError, "requires manifest role validation-baseline"):
            self._generate()

    def test_shipping_candidate_with_unknown_checkpoint_terms_is_excluded(self) -> None:
        value = self._read_manifest()
        value["status"] = "excluded"
        value["exclusion"] = {"reason_code": "checkpoint_license_terms_unknown"}
        value["licenses"]["checkpoint"]["commercial_use_permitted"] = "unknown"
        value["licenses"]["checkpoint"]["redistribution_permitted"] = "unknown"
        self._write_manifest(value)
        document = self._generate()
        candidate = document["candidates"][0]
        self.assertEqual(candidate["status"], "excluded")
        self.assertEqual(candidate["exclusion_reason"]["type"], "license_unknown")
        self.assertEqual(candidate["measurement_status"], "measurable")

    def test_provider_tokens_are_explicit_and_fail_closed(self) -> None:
        with self.assertRaisesRegex(AdmissionError, "at least one"):
            self._generate(providers=())
        with self.assertRaisesRegex(AdmissionError, "unknown measured provider"):
            self._generate(providers=("tpu",))
        with self.assertRaisesRegex(AdmissionError, "unique"):
            self._generate(providers=("cpu", "cpu"))

    def test_legal_review_attestations_are_explicit_and_complete(self) -> None:
        with self.assertRaisesRegex(AdmissionError, "every legal surface"):
            generate_admission(PROTOCOL, [self._input()], ["cpu"])
        with self.assertRaisesRegex(AdmissionError, "missing: backbone"):
            generate_admission(
                PROTOCOL,
                [self._input()],
                ["cpu"],
                reviewed_surfaces=("code", "checkpoint"),
            )
        with self.assertRaisesRegex(AdmissionError, "unknown reviewed legal surface"):
            generate_admission(
                PROTOCOL,
                [self._input()],
                ["cpu"],
                reviewed_surfaces=("code", "checkpoint", "backbone", "terms"),
            )

    def test_multiple_candidates_are_ordered_by_protocol_and_output_is_deterministic(self) -> None:
        second_manifest = self.root / "second.json"
        second_artifact = self.root / "second" / "valid.bin"
        second_artifact.parent.mkdir()
        shutil.copy2(self.manifest, second_manifest)
        shutil.copy2(self.artifact, second_artifact)
        second_manifest.chmod(0o644)
        second_artifact.chmod(0o644)
        second_value = json.loads(second_manifest.read_text(encoding="utf-8"))
        second_value["candidate"]["id"] = "raft-original"
        second_value["candidate"]["role"] = "validation-baseline"
        second_value["status"] = "excluded"
        second_value["exclusion"] = {"reason_code": "checkpoint_license_terms_unknown"}
        second_manifest.write_text(
            json.dumps(second_value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        second_manifest.chmod(0o644)
        first = self._generate(
            [CandidateInput(second_manifest, second_artifact), self._input()]
        )
        second = self._generate(
            [self._input(), CandidateInput(second_manifest, second_artifact)]
        )
        self.assertEqual(first, second)
        self.assertEqual(
            [entry["candidate_id"] for entry in first["candidates"]],
            ["sea-raft-m", "raft-original"],
        )

    def test_output_is_canonical_0644_json(self) -> None:
        document = self._generate()
        output = self.root / "admission.json"
        write_admission(output, document)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o644)
        self.assertEqual(
            output.read_text(encoding="utf-8"),
            json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )

    def test_output_defaults_to_atomic_no_clobber_and_explicit_replace(self) -> None:
        document = self._generate()
        output = self.root / "safe-output.json"
        write_admission(output, document)
        original = output.read_bytes()
        with self.assertRaisesRegex(AdmissionError, "already exists"):
            write_admission(output, {"changed": True})
        self.assertEqual(output.read_bytes(), original)

        replacement = {"changed": True}
        write_admission(output, replacement, replace=True)
        self.assertEqual(
            output.read_text(encoding="utf-8"),
            json.dumps(replacement, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o644)

    def test_output_rejects_existing_symlink_and_nonregular(self) -> None:
        target = self.root / "target.json"
        target.write_text("do not touch\n", encoding="utf-8")
        symlink = self.root / "symlink.json"
        symlink.symlink_to(target)
        with self.assertRaisesRegex(AdmissionError, "must not be a symlink"):
            write_admission(symlink, self._generate())
        self.assertEqual(target.read_text(encoding="utf-8"), "do not touch\n")

        directory = self.root / "directory.json"
        directory.mkdir()
        with self.assertRaisesRegex(AdmissionError, "regular file"):
            write_admission(directory, self._generate())

    def test_cli_writes_admission_document(self) -> None:
        output = self.root / "cli-admission.json"
        script = ROOT / "tools" / "p25_5" / "admission.py"
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--protocol",
                str(PROTOCOL),
                "--candidate",
                str(self.manifest),
                str(self.artifact),
                "-",
                "--provider",
                "cpu",
                "--reviewed-surface",
                "code",
                "--reviewed-surface",
                "checkpoint",
                "--reviewed-surface",
                "backbone",
                "--output",
                str(output),
            ],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        document = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(document["protocol_id"], ACTIVE_PROTOCOL_ID)


if __name__ == "__main__":
    unittest.main()
