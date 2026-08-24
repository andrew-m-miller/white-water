#!/usr/bin/env python3
"""Focused tests for the P25-5 hash-bound legal-review input seam."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from .legal_review import (
    ACTIVE_PROTOCOL_ID,
    LEGAL_REVIEW_SCHEMA_ID,
    LegalReviewError,
    canonical_sha256,
    load_legal_review,
    validate_candidate_identities,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "bakeoff" / "protocol-v2.json"
MANIFEST = ROOT / "models" / "sea-raft-m.json"


class LegalReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="whitewater-p25-legal-review-test-")
        self.root = Path(self.temp.name)
        self.manifest = self.root / "sea-raft-m.json"
        shutil.copy2(MANIFEST, self.manifest)
        self.manifest.chmod(0o644)
        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.document = {
            "schema_id": LEGAL_REVIEW_SCHEMA_ID,
            "protocol_id": ACTIVE_PROTOCOL_ID,
            "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
            "candidate_identities": [
                {
                    "candidate_id": "sea-raft-m",
                    "source_commit": "9137517ba24e628442aec097d3afe71d03503b75",
                    "checkpoint_sha256": "cb8cfbf14c5e0f6734b64add383708b7ff68cc6089a0007c67165d4761346102",
                    "licenses_sha256": canonical_sha256(manifest["licenses"]),
                }
            ],
            "reviewed_surfaces": ["code", "checkpoint", "backbone"],
            "reviewed": True,
            "reviewer": "operator@example.invalid",
            "reviewed_at": "2026-08-24T12:00:00Z",
            "statement": "I reviewed the exact source and checkpoint identities for this measurement.",
        }
        self.review = self.root / "legal-review.json"
        self._write_review()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_review(self) -> str:
        self.review.write_text(
            json.dumps(self.document, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        self.review.chmod(0o644)
        return hashlib.sha256(self.review.read_bytes()).hexdigest()

    def _load(self, expected_sha: str | None = None):
        expected_sha = expected_sha or hashlib.sha256(self.review.read_bytes()).hexdigest()
        return load_legal_review(self.review, expected_sha, protocol_path=PROTOCOL)

    def test_valid_review_is_hash_bound_and_candidate_bound(self) -> None:
        review = self._load()
        self.assertEqual(review.reviewed_surfaces, ("code", "checkpoint", "backbone"))
        self.assertEqual(review.reviewer, "operator@example.invalid")
        validate_candidate_identities(
            review,
            {"sea-raft-m": json.loads(self.manifest.read_text(encoding="utf-8"))},
        )

    def test_external_hash_is_required_and_must_match_exact_bytes(self) -> None:
        with self.assertRaisesRegex(LegalReviewError, "SHA256 is required"):
            load_legal_review(self.review, None, protocol_path=PROTOCOL)
        with self.assertRaisesRegex(LegalReviewError, "SHA256 mismatch"):
            load_legal_review(self.review, "0" * 64, protocol_path=PROTOCOL)

    def test_protocol_hash_and_explicit_review_are_required(self) -> None:
        self.document["protocol_sha256"] = "0" * 64
        expected_sha = self._write_review()
        with self.assertRaisesRegex(LegalReviewError, "does not match the active protocol"):
            load_legal_review(self.review, expected_sha, protocol_path=PROTOCOL)

        self.document["protocol_sha256"] = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
        self.document["reviewed"] = False
        expected_sha = self._write_review()
        with self.assertRaisesRegex(LegalReviewError, "reviewed must be explicit true"):
            load_legal_review(self.review, expected_sha, protocol_path=PROTOCOL)

        self.document["reviewed"] = True
        self.document["reviewed_surfaces"] = ["code", "checkpoint"]
        expected_sha = self._write_review()
        with self.assertRaisesRegex(LegalReviewError, "missing: backbone"):
            load_legal_review(self.review, expected_sha, protocol_path=PROTOCOL)

    def test_candidate_identity_mismatch_is_rejected(self) -> None:
        review = self._load()
        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        changed = copy.deepcopy(manifest)
        changed["upstream"]["commit"] = "a" * 40
        with self.assertRaisesRegex(LegalReviewError, "source commit does not match"):
            validate_candidate_identities(review, {"sea-raft-m": changed})

        extra = copy.deepcopy(self.document)
        extra["candidate_identities"].append(
            {
                "candidate_id": "raft-original",
                "source_commit": "a" * 40,
                "checkpoint_sha256": "b" * 64,
                "licenses_sha256": "c" * 64,
            }
        )
        self.document = extra
        expected_sha = self._write_review()
        extra_review = self._load(expected_sha)
        with self.assertRaisesRegex(LegalReviewError, "do not exactly match manifests"):
            validate_candidate_identities(
                extra_review,
                {"sea-raft-m": json.loads(self.manifest.read_text(encoding="utf-8"))},
            )

    def test_duplicate_json_object_keys_are_rejected(self) -> None:
        self.review.write_text(
            '{"schema_id":"whitewater-p25-legal-review-v1",'
            '"schema_id":"whitewater-p25-legal-review-v1"}\n',
            encoding="utf-8",
        )
        self.review.chmod(0o644)
        expected_sha = hashlib.sha256(self.review.read_bytes()).hexdigest()
        with self.assertRaisesRegex(LegalReviewError, "duplicate JSON object key"):
            load_legal_review(self.review, expected_sha, protocol_path=PROTOCOL)

    def test_manifest_license_identity_mismatch_is_rejected(self) -> None:
        review = self._load()
        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        manifest["licenses"]["checkpoint"]["audit"] += " changed"
        with self.assertRaisesRegex(LegalReviewError, "licenses SHA256 does not match"):
            validate_candidate_identities(review, {"sea-raft-m": manifest})

    def test_cli_validates_the_same_contract_used_by_workflow(self) -> None:
        expected_sha = hashlib.sha256(self.review.read_bytes()).hexdigest()
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "p25_5" / "legal_review.py"),
                "--protocol",
                str(PROTOCOL),
                "--manifest",
                "sea-raft-m",
                str(self.manifest),
                "--review",
                str(self.review),
                "--sha256",
                expected_sha,
            ],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"legal_review_sha256"', result.stdout)

    def test_workflow_consumes_attestation_instead_of_hardcoded_surface_flags(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        for token in (
            "p25_5_legal_review_file",
            "p25_5_legal_review_sha256",
            "p25_5_candidate_license_input",
            "p25_5_runtime_legal_review_file",
            "p25_5_runtime_legal_review_sha256",
            "P25_5_LEGAL_REVIEW_FILE",
            "P25_5_LEGAL_REVIEW_SHA256",
            "P25_5_CANDIDATE_LICENSE_INPUT",
            "P25_5_RUNTIME_LICENSE_INPUT",
            "P25_5_RUNTIME_LEGAL_REVIEW_FILE",
            "P25_5_RUNTIME_LEGAL_REVIEW_SHA256",
            "--legal-review-file",
            "--legal-review-sha256",
        ):
            self.assertIn(token, workflow)
        for surface in ("code", "checkpoint", "backbone"):
            self.assertNotIn(f"--reviewed-surface {surface}", workflow)

    def test_workflow_uses_el8_compatible_curl_options(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        self.assertNotIn(
            "--retry-all-errors",
            workflow,
            "the EL8 qualification container carries curl 7.61.1",
        )

    def test_workflow_checks_the_literal_relative_ort_runpath(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        self.assertIn("grep -F '$ORIGIN/onnxruntime'", workflow)
        self.assertNotIn("grep -F '\\$ORIGIN/onnxruntime'", workflow)

    def test_workflow_resolves_conda_unpack_with_the_relocated_python(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        self.assertIn(
            'env PATH="$RUNNER_TEMP/p25-5-runtime-extracted/bin:$PATH"',
            workflow,
        )

    def test_workflow_runs_the_glibc_tree_gate_with_bash(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        self.assertIn(
            "- name: Check glibc 2.28 baseline for runtime and evaluator ELFs\n"
            "        shell: bash\n"
            "        run: |",
            workflow,
        )


if __name__ == "__main__":
    unittest.main()
