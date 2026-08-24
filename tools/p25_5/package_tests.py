#!/usr/bin/env python3
"""Focused unit tests for the Phase 2.5 air-gap package builder/verifier."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

try:
    from . import package as package_module
    from .package import PackageError, build_package, load_spec, verify_package
except ImportError:  # Direct execution: ``python tools/p25_5/package_tests.py``.
    import package as package_module  # type: ignore
    from package import PackageError, build_package, load_spec, verify_package  # type: ignore


P25_5_RUNTIME_IDENTITY = (
    "python-3.11;microsoft-onnxruntime-linux-x64-gpu_cuda12-1.29.0+whitewater-native-bridge;"
    "conda-pack;el8-x86_64"
)


class PackageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="whitewater-p25-package-test-")
        self.root = Path(self.temp.name)
        self.sources = self.root / "sources"
        self.sources.mkdir()
        self._write("evaluator", b"#!/bin/sh\nexit 0\n", 0o755)
        self._write("evaluator_support.py", b"def run_case():\n    return None\n", 0o644)
        self._write("evaluator_launcher", b"#!/bin/sh\nexit 0\n", 0o755)
        # This is intentionally just opaque bytes.  The outer package must not unpack or
        # rewrite a conda-pack runtime; its exact hash and size are admitted below.
        self._write("runtime.tar.gz", b"opaque-conda-pack-runtime\x00", 0o644)
        self._write("model.onnx", b"model bytes\n", 0o644)
        model_sha, model_size = self._sha_size(self.sources / "model.onnx")
        candidate_manifest = {
            "schema_id": "whitewater-p25-artifact-v1",
            "candidate": {"id": "candidate-a", "role": "shipping-candidate"},
            "export": {
                "artifact": "model.onnx",
                "sha256": model_sha,
                "size_bytes": model_size,
                "mode": "0644",
                "platform": "fixture",
                "platform_artifacts": [
                    {
                        "platform": "fixture",
                        "artifact": "model.onnx",
                        "sha256": model_sha,
                        "size_bytes": model_size,
                        "mode": "0644",
                    }
                ],
            },
        }
        self._write(
            "candidate.json",
            (json.dumps(candidate_manifest, sort_keys=True) + "\n").encode("utf-8"),
            0o644,
        )
        self._write("LICENSE.txt", b"license text\n", 0o644)
        self._write("NOTICE.txt", b"notice text\n", 0o644)
        self._write(
            "RUN.md",
            b"tar -xf runtime/conda-pack.tar.gz -C runtime-env\n"
            b"runtime-env/bin/conda-unpack\n"
            b"evaluator/ww-flow-eval\n",
            0o644,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write(self, name: str, content: bytes, mode: int) -> Path:
        path = self.sources / name
        path.write_bytes(content)
        path.chmod(mode)
        return path

    def _sha_size(self, path: Path) -> tuple[str, int]:
        content = path.read_bytes()
        return hashlib.sha256(content).hexdigest(), len(content)

    def _spec_value(self, *, measurement_status: str = "measurable", admitted: bool = True) -> dict:
        runtime_sha, runtime_size = self._sha_size(self.sources / "runtime.tar.gz")
        return {
            "schema_id": "whitewater-p25-airgap-package-v1",
            "protocol_id": "whitewater-p25-v2",
            "package_id": "fixture-airgap",
            "evaluator": {
                "entrypoint": "evaluator/ww-flow-eval",
                "runtime_identity": "conda-pack-opaque-fixture-v1",
            },
            "admission": {
                "candidates": [
                    {
                        "candidate_id": "candidate-a",
                        "measurement_status": measurement_status,
                        "measurement_admitted": admitted,
                        "status": "excluded",
                        "exclusion_reason": "checkpoint_license_terms_unknown",
                    }
                ]
            },
            "files": [
                {
                    "role": "evaluator",
                    "destination": "evaluator/ww-flow-eval",
                    "source": str(self.sources / "evaluator"),
                    "candidate_id": None,
                    "mode": "0755",
                },
                {
                    "role": "runtime",
                    "destination": "runtime/conda-pack.tar.gz",
                    "source": str(self.sources / "runtime.tar.gz"),
                    "candidate_id": None,
                    "mode": "0644",
                    "sha256": runtime_sha,
                    "size_bytes": runtime_size,
                },
                {
                    "role": "evaluator-support",
                    "destination": "evaluator/evaluator_support.py",
                    "source": str(self.sources / "evaluator_support.py"),
                    "candidate_id": None,
                    "mode": "0644",
                },
                {
                    "role": "model-artifact",
                    "destination": "models/candidate-a/model.onnx",
                    "source": str(self.sources / "model.onnx"),
                    "candidate_id": "candidate-a",
                    "mode": "0644",
                },
                {
                    "role": "candidate-manifest",
                    "destination": "models/candidate-a/manifest.json",
                    "source": str(self.sources / "candidate.json"),
                    "candidate_id": "candidate-a",
                    "mode": "0644",
                },
                {
                    "role": "license",
                    "destination": "legal/LICENSE.txt",
                    "source": str(self.sources / "LICENSE.txt"),
                    "candidate_id": None,
                    "mode": "0644",
                },
                {
                    "role": "notice",
                    "destination": "legal/NOTICE.txt",
                    "source": str(self.sources / "NOTICE.txt"),
                    "candidate_id": None,
                    "mode": "0644",
                },
                {
                    "role": "run-instructions",
                    "destination": "RUN.md",
                    "source": str(self.sources / "RUN.md"),
                    "candidate_id": None,
                    "mode": "0644",
                },
            ],
        }

    def _write_spec(self, value: dict, name: str = "package.json") -> Path:
        path = self.root / name
        path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        path.chmod(0o644)
        return path

    def _build(self, value: dict | None = None, suffix: str = "") -> tuple[dict, Path, Path, Path]:
        spec_path = self._write_spec(value or self._spec_value(), f"package{suffix}.json")
        staging = self.root / f"staging{suffix}"
        archive = self.root / f"package{suffix}.tar.gz"
        inventory = self.root / f"package{suffix}.inventory.json"
        result = build_package(
            spec_path,
            staging_dir=staging,
            archive_path=archive,
            inventory_path=inventory,
        )
        return result, staging, archive, inventory

    def test_build_verifies_all_copies_and_keeps_runtime_opaque(self) -> None:
        result, staging, archive, inventory = self._build()
        self.assertEqual(result["archive_sha256"], hashlib.sha256(archive.read_bytes()).hexdigest())
        self.assertEqual(stat.S_IMODE((staging / "evaluator/ww-flow-eval").stat().st_mode), 0o755)
        self.assertEqual(stat.S_IMODE((staging / "runtime/conda-pack.tar.gz").stat().st_mode), 0o644)
        self.assertEqual(stat.S_IMODE((staging / "models/candidate-a/model.onnx").stat().st_mode), 0o644)
        verified = verify_package(
            archive,
            inventory,
            staging_dir=staging,
            extract_dir=self.root / "extracted",
            verify_sources=True,
        )
        self.assertEqual(verified["file_count"], 9)  # 8 supplied files + generated admission record.
        extracted_runtime = self.root / "extracted/runtime/conda-pack.tar.gz"
        self.assertEqual(extracted_runtime.read_bytes(), (self.sources / "runtime.tar.gz").read_bytes())
        self.assertEqual(stat.S_IMODE(extracted_runtime.stat().st_mode), 0o644)

    def test_archive_is_deterministic(self) -> None:
        _, _, archive_a, _ = self._build(suffix="-a")
        _, _, archive_b, _ = self._build(suffix="-b")
        self.assertEqual(archive_a.read_bytes(), archive_b.read_bytes())
        self.assertEqual(
            hashlib.sha256(archive_a.read_bytes()).hexdigest(),
            hashlib.sha256(archive_b.read_bytes()).hexdigest(),
        )

    def test_excluded_but_measurable_candidate_is_admitted(self) -> None:
        result, _, _, _ = self._build()
        self.assertEqual(result["package_id"], "fixture-airgap")

    def test_measurement_status_is_fail_closed(self) -> None:
        unavailable = self._spec_value(measurement_status="unavailable")
        with self.assertRaisesRegex(PackageError, "not technically measurable"):
            load_spec(self._write_spec(unavailable))

    def test_explicit_measurement_admission_is_required(self) -> None:
        not_admitted = self._spec_value(admitted=False)
        with self.assertRaisesRegex(PackageError, "lacks explicit measurement admission"):
            load_spec(self._write_spec(not_admitted))

    def test_shipping_status_does_not_substitute_for_measurement_admission(self) -> None:
        not_admitted = self._spec_value(admitted=False)
        not_admitted["admission"]["candidates"][0]["status"] = "eligible"
        not_admitted["admission"]["candidates"][0].pop("exclusion_reason")
        with self.assertRaisesRegex(PackageError, "lacks explicit measurement admission"):
            load_spec(self._write_spec(not_admitted))

    def test_runtime_sha256_and_size_are_required_and_checked(self) -> None:
        value = self._spec_value()
        runtime = next(item for item in value["files"] if item["role"] == "runtime")
        runtime["sha256"] = "0" * 64
        with self.assertRaisesRegex(PackageError, "runtime identity mismatch"):
            self._build(value)

        missing = self._spec_value()
        missing_runtime = next(item for item in missing["files"] if item["role"] == "runtime")
        del missing_runtime["size_bytes"]
        with self.assertRaisesRegex(PackageError, "missing required fields: size_bytes"):
            load_spec(self._write_spec(missing))

    def test_model_mode_must_be_0644(self) -> None:
        model = self.sources / "model.onnx"
        model.chmod(0o600)
        with self.assertRaisesRegex(PackageError, "expected 0644"):
            self._build()

    def test_evaluator_mode_is_distinct_and_must_be_0755(self) -> None:
        evaluator = self.sources / "evaluator"
        evaluator.chmod(0o644)
        with self.assertRaisesRegex(PackageError, "expected 0755"):
            self._build()

    def test_evaluator_support_is_global_0644_and_optional(self) -> None:
        _, staging, archive, inventory = self._build()
        support = staging / "evaluator/evaluator_support.py"
        self.assertTrue(support.is_file())
        self.assertEqual(stat.S_IMODE(support.stat().st_mode), 0o644)
        with tarfile.open(archive, "r:gz") as stream:
            member = stream.getmember("evaluator/evaluator_support.py")
            self.assertEqual(stat.S_IMODE(member.mode), 0o644)
        extracted = self.root / "extracted-support"
        verify_package(archive, inventory, staging_dir=staging, extract_dir=extracted)
        self.assertEqual(
            stat.S_IMODE((extracted / "evaluator/evaluator_support.py").stat().st_mode),
            0o644,
        )

        # Support is deliberately not an alternate entrypoint role: a 0755 support record is
        # rejected at spec admission rather than being copied through as a second executable.
        executable_support = self._spec_value()
        support_entry = next(
            item for item in executable_support["files"] if item["role"] == "evaluator-support"
        )
        support_entry["mode"] = "0755"
        with self.assertRaisesRegex(
            PackageError, r"role 'evaluator-support' requires one of modes 0644"
        ):
            load_spec(self._write_spec(executable_support, "executable-support.json"))

        # A post-publication chmod is also rejected at the staged-copy boundary.
        support.chmod(0o755)
        with self.assertRaisesRegex(PackageError, "expected 0644"):
            verify_package(archive, inventory, staging_dir=staging)

        without_support = self._spec_value()
        without_support["files"] = [
            item for item in without_support["files"] if item["role"] != "evaluator-support"
        ]
        result, _, _, _ = self._build(without_support, suffix="-without-support")
        self.assertEqual(result["package_id"], "fixture-airgap")

        bound_support = self._spec_value()
        support_entry = next(
            item for item in bound_support["files"] if item["role"] == "evaluator-support"
        )
        support_entry["candidate_id"] = "candidate-a"
        with self.assertRaisesRegex(PackageError, "must be package-global"):
            load_spec(self._write_spec(bound_support, "bound-support.json"))

    def test_executable_support_source_cannot_be_declared_as_support(self) -> None:
        value = self._spec_value()
        value["files"].append(
            {
                "role": "evaluator-support",
                "destination": "scripts/evaluator-launcher",
                "source": str(self.sources / "evaluator_launcher"),
                "candidate_id": None,
                "mode": "0755",
            }
        )
        with self.assertRaisesRegex(
            PackageError, r"role 'evaluator-support' requires one of modes 0644"
        ):
            load_spec(self._write_spec(value, "executable-support.json"))

    def test_symlink_and_nonregular_sources_are_rejected(self) -> None:
        value = self._spec_value()
        model = self.sources / "model.onnx"
        model.unlink()
        model.symlink_to(self.sources / "candidate.json")
        with self.assertRaisesRegex(PackageError, "must not be a symlink"):
            self._build(value)

        model.unlink()
        model.mkdir()
        with self.assertRaisesRegex(PackageError, "must be a regular file"):
            self._build(self._spec_value(), suffix="-nonregular")

    def test_source_replacement_between_lstat_and_open_is_rejected(self) -> None:
        value = self._spec_value()
        source = self.sources / "model.onnx"
        real_open = package_module.os.open

        def replace_before_open(path, flags, *arguments, **keywords):
            if Path(path) == source:
                source.unlink()
                source.symlink_to(self.sources / "candidate.json")
            return real_open(path, flags, *arguments, **keywords)

        with patch.object(package_module.os, "open", side_effect=replace_before_open):
            with self.assertRaisesRegex(PackageError, "must not be a symlink"):
                self._build(value, suffix="-source-race")

    def test_candidate_manifest_must_match_carried_artifact_identity(self) -> None:
        value = self._spec_value()
        manifest_path = self.sources / "candidate.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["export"]["sha256"] = "0" * 64
        manifest["export"]["platform_artifacts"][0]["sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
        manifest_path.chmod(0o644)
        with self.assertRaisesRegex(PackageError, "does not match exactly one carried artifact"):
            self._build(value, suffix="-manifest-mismatch")

        # A stale selected platform record is also rejected before an archive can be emitted.
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_sha, model_size = self._sha_size(self.sources / "model.onnx")
        manifest["export"]["sha256"] = model_sha
        manifest["export"]["platform_artifacts"][0]["size_bytes"] = model_size + 1
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
        manifest_path.chmod(0o644)
        with self.assertRaisesRegex(PackageError, "selected platform record"):
            self._build(value, suffix="-platform-mismatch")

    def test_destination_traversal_and_url_sources_are_rejected(self) -> None:
        traversal = self._spec_value()
        traversal["files"][2]["destination"] = "../model.onnx"
        with self.assertRaisesRegex(PackageError, "must not contain"):
            load_spec(self._write_spec(traversal))

        url = self._spec_value()
        url["files"][2]["source"] = "https://example.invalid/model.onnx"
        with self.assertRaisesRegex(PackageError, "downloads are not permitted"):
            load_spec(self._write_spec(url))

    def test_archive_and_staging_tampering_are_detected(self) -> None:
        _, staging, archive, inventory = self._build()
        model_stage = staging / "models/candidate-a/model.onnx"
        model_stage.write_bytes(b"tampered\n")
        model_stage.chmod(0o644)
        with self.assertRaisesRegex(PackageError, "staged identity mismatch"):
            verify_package(archive, inventory, staging_dir=staging)

        # Rebuild fresh, then alter the archive after its SHA was recorded.
        _, _, archive, inventory = self._build(self._spec_value(), suffix="-tamper")
        bytes_value = bytearray(archive.read_bytes())
        bytes_value[-1] ^= 0x01
        archive.write_bytes(bytes(bytes_value))
        archive.chmod(0o644)
        with self.assertRaisesRegex(PackageError, "archive SHA256 or size"):
            verify_package(archive, inventory)

    def test_archive_and_inventory_modes_are_checked(self) -> None:
        _, _, archive, inventory = self._build()
        archive.chmod(0o600)
        with self.assertRaisesRegex(PackageError, "expected 0644"):
            verify_package(archive, inventory)
        archive.chmod(0o644)
        inventory.chmod(0o600)
        with self.assertRaisesRegex(PackageError, "expected 0644"):
            verify_package(archive, inventory)

    def test_staging_tree_must_not_contain_unlisted_files(self) -> None:
        _, staging, archive, inventory = self._build()
        extra = staging / "unexpected.bin"
        extra.write_bytes(b"extra")
        extra.chmod(0o644)
        with self.assertRaisesRegex(PackageError, "unexpected file"):
            verify_package(archive, inventory, staging_dir=staging)

    def test_inventory_rechecks_candidate_binding_and_admission_record(self) -> None:
        _, _, archive, inventory = self._build()
        value = json.loads(inventory.read_text(encoding="utf-8"))
        model = next(item for item in value["files"] if item["role"] == "model-artifact")
        model["candidate_id"] = "not-admitted"
        inventory.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        inventory.chmod(0o644)
        with self.assertRaisesRegex(PackageError, "unadmitted candidate"):
            verify_package(archive, inventory)

        # Rebuild before the second mutation so the first malformed inventory does not affect
        # the archive, then alter only the carried admission decision in the inventory.
        _, _, archive, inventory = self._build(self._spec_value(), suffix="-admission-tamper")
        value = json.loads(inventory.read_text(encoding="utf-8"))
        value["admission"]["candidates"][0]["status"] = "eligible"
        value["admission"]["candidates"][0].pop("exclusion_reason")
        inventory.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        inventory.chmod(0o644)
        with self.assertRaisesRegex(PackageError, "admission record does not match"):
            verify_package(archive, inventory)

    def test_existing_extraction_directory_is_not_overwritten(self) -> None:
        _, _, archive, inventory = self._build()
        extraction = self.root / "already-there"
        extraction.mkdir()
        (extraction / "keep").write_bytes(b"keep")
        with self.assertRaisesRegex(PackageError, "must be empty"):
            verify_package(archive, inventory, extract_dir=extraction)

    def test_cli_build_and_verify_round_trip(self) -> None:
        spec = self._write_spec(self._spec_value(), "cli-package.json")
        staging = self.root / "cli-staging"
        archive = self.root / "cli-package.tar.gz"
        inventory = self.root / "cli-package.inventory.json"
        script = Path(__file__).with_name("package.py")
        built = subprocess.run(
            [
                sys.executable,
                str(script),
                "build",
                str(spec),
                "--staging-dir",
                str(staging),
                "--archive",
                str(archive),
                "--inventory",
                str(inventory),
            ],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(built.returncode, 0, built.stderr)
        verified = subprocess.run(
            [
                sys.executable,
                str(script),
                "verify",
                str(archive),
                str(inventory),
                "--staging-dir",
                str(staging),
                "--extract-dir",
                str(self.root / "cli-extracted"),
                "--verify-sources",
            ],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(verified.returncode, 0, verified.stderr)

    def test_checked_in_searaft_template_has_explicit_closure_and_only_ci_markers(self) -> None:
        root = Path(__file__).resolve().parents[2]
        template_path = root / "bakeoff" / "p25-5" / "package-spec.json"
        run_path = root / "bakeoff" / "p25-5" / "RUN-P25-5.txt"
        template = json.loads(template_path.read_text(encoding="utf-8"))
        self.assertEqual(template["schema_id"], "whitewater-p25-airgap-package-v1")
        self.assertEqual(template["protocol_id"], "whitewater-p25-v2")
        self.assertEqual(template["admission"]["candidates"], "__P25_5_ADMISSION_CANDIDATES__")
        self.assertEqual(template["evaluator"]["runtime_identity"], P25_5_RUNTIME_IDENTITY)

        runtime = [item for item in template["files"] if item.get("role") == "runtime"]
        self.assertEqual(len(runtime), 1)
        self.assertEqual(
            runtime[0],
            {
                "role": "runtime",
                "destination": "runtime/whitewater-p25-5-runtime.tar.gz",
                "source": "__P25_5_RUNTIME_ARCHIVE__",
                "candidate_id": None,
                "mode": "0644",
                "sha256": "__P25_5_RUNTIME_SHA256__",
                "size_bytes": "__P25_5_RUNTIME_SIZE_BYTES__",
            },
        )

        markers = sorted(set(re.findall(r"__P25_5_[A-Z0-9_]+__", json.dumps(template))))
        self.assertEqual(
            markers,
            [
                "__P25_5_ADMISSION_CANDIDATES__",
                "__P25_5_CANDIDATE_INVENTORY_SOURCE__",
                "__P25_5_CANDIDATE_LICENSE_SOURCE__",
                "__P25_5_CANDIDATE_NOTICE_SOURCE__",
                "__P25_5_RUNTIME_ARCHIVE__",
                "__P25_5_RUNTIME_INVENTORY_SOURCE__",
                "__P25_5_RUNTIME_LICENSE_SOURCE__",
                "__P25_5_RUNTIME_NOTICE_SOURCE__",
                "__P25_5_RUNTIME_REVIEW_SOURCE__",
                "__P25_5_RUNTIME_SHA256__",
                "__P25_5_RUNTIME_SIZE_BYTES__",
            ],
        )

        by_destination = {item["destination"]: item for item in template["files"]}
        required = {
            "tools/bakeoff/evaluator.py",
            "tools/bakeoff/native_ort.py",
            "scripts/ww-bakeoff-airgap",
            "tools/bakeoff/conditioning.py",
            "tools/bakeoff/geometry.py",
            "tools/bakeoff/measurement.py",
            "tools/bakeoff/metrics.py",
            "tools/bakeoff/padding.py",
            "tools/bakeoff/pfm.py",
            "tools/bakeoff/validator.py",
            "models/artifact_workflow.py",
            "models/exclusion_contract.py",
            "models/artifact-v1.schema.json",
            "bakeoff/corpus-v1.schema.json",
            "bakeoff/protocol-v2.schema.json",
            "bakeoff/report-v2.schema.json",
            "bakeoff/protocol-v2.json",
            "models/sea-raft-m/manifest.json",
            "models/sea-raft-m/sea-raft-m-opset17.onnx",
            "legal/SEA-RAFT-LICENSE.txt",
            "legal/SEA-RAFT-NOTICE.txt",
            "legal/candidate-license-inventory.json",
            "legal/RUNTIME-LICENSES.txt",
            "legal/RUNTIME-NOTICES.txt",
            "legal/runtime-license-inventory.json",
            "legal/runtime-inputs.json",
            "legal/runtime-legal-review.json",
            "RUN-P25-5.txt",
            "runtime/whitewater-p25-5-runtime.tar.gz",
        }
        self.assertTrue(required.issubset(by_destination))
        self.assertEqual(template["evaluator"]["entrypoint"], "scripts/ww-bakeoff-airgap")
        self.assertEqual(by_destination["scripts/ww-bakeoff-airgap"]["mode"], "0755")
        self.assertEqual(by_destination["scripts/ww-bakeoff-airgap"]["role"], "evaluator")
        self.assertEqual(by_destination["tools/bakeoff/evaluator.py"]["mode"], "0644")
        self.assertEqual(
            by_destination["tools/bakeoff/evaluator.py"]["role"], "evaluator-support"
        )
        self.assertEqual(by_destination["tools/bakeoff/native_ort.py"]["mode"], "0644")
        self.assertEqual(
            by_destination["tools/bakeoff/native_ort.py"]["role"], "evaluator-support"
        )
        self.assertEqual(
            by_destination["tools/bakeoff/native_ort.py"]["source"],
            "../../tools/bakeoff/native_ort.py",
        )
        self.assertEqual(by_destination["models/artifact_workflow.py"]["mode"], "0644")
        self.assertEqual(
            [item["destination"] for item in template["files"] if item["mode"] == "0755"],
            ["scripts/ww-bakeoff-airgap"],
        )
        self.assertEqual(
            by_destination["legal/SEA-RAFT-LICENSE.txt"]["source"],
            "__P25_5_CANDIDATE_LICENSE_SOURCE__",
        )
        self.assertEqual(
            by_destination["legal/SEA-RAFT-NOTICE.txt"]["source"],
            "__P25_5_CANDIDATE_NOTICE_SOURCE__",
        )
        self.assertNotIn("legal-review-sea-raft-m.json", {
            item.get("source") for item in template["files"]
        })

        run_text = run_path.read_text(encoding="utf-8")
        for required_text in (
            "evaluation-only",
            "scripts/ww-bakeoff-airgap verify",
            "--manifest models/sea-raft-m/manifest.json",
            "--artifact models/sea-raft-m/sea-raft-m-opset17.onnx",
            "--protocol bakeoff/protocol-v2.json",
            "__P25_5_ADMISSION_CANDIDATES__",
            "__P25_5_RUNTIME_ARCHIVE__",
            "__P25_5_RUNTIME_SHA256__",
            "__P25_5_RUNTIME_SIZE_BYTES__",
            "SEA-RAFT-LICENSE.txt",
            "SEA-RAFT-NOTICE.txt",
            "RUNTIME-LICENSES.txt",
            "RUNTIME-NOTICES.txt",
            "runtime-license-inventory.json",
            "runtime-legal-review.json",
        ):
            self.assertIn(required_text, run_text)

    def test_packaged_evaluator_import_uses_carried_native_adapter(self) -> None:
        """Import the extracted evaluator and exercise its native-runtime fallback.

        This is intentionally a package-level smoke rather than a native bridge test: it proves
        the evaluator's relative ``native_ort`` import survives packaging while replacing the
        loader with a sentinel, so no Linux ORT shared objects or GPU are required.
        """

        root = Path(__file__).resolve().parents[2]
        template_path = root / "bakeoff" / "p25-5" / "package-spec.json"
        template = json.loads(template_path.read_text(encoding="utf-8"))
        by_destination = {item["destination"]: item for item in template["files"]}
        required_support = {
            "tools/bakeoff/__init__.py",
            "tools/bakeoff/conditioning.py",
            "tools/bakeoff/evaluator.py",
            "tools/bakeoff/geometry.py",
            "tools/bakeoff/measurement.py",
            "tools/bakeoff/metrics.py",
            "tools/bakeoff/native_ort.py",
            "tools/bakeoff/padding.py",
            "tools/bakeoff/pfm.py",
            "tools/bakeoff/validator.py",
        }
        self.assertTrue(required_support.issubset(by_destination))

        value = self._spec_value()
        for index, destination in enumerate(sorted(required_support)):
            template_entry = by_destination[destination]
            source = (template_path.parent / template_entry["source"]).resolve()
            self.assertTrue(source.is_file(), source)
            fixture_source = self._write(
                f"packaged-evaluator-support-{index}.py",
                source.read_bytes(),
                0o644,
            )
            value["files"].append(
                {
                    "role": "evaluator-support",
                    "destination": destination,
                    "source": str(fixture_source),
                    "candidate_id": None,
                    "mode": "0644",
                }
            )

        _, staging, archive, inventory = self._build(value, suffix="-packaged-evaluator")
        extracted = self.root / "packaged-evaluator-extracted"
        verify_package(
            archive,
            inventory,
            staging_dir=staging,
            extract_dir=extracted,
            verify_sources=True,
        )

        smoke = r'''
import builtins
import importlib
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
evaluator = importlib.import_module("tools.bakeoff.evaluator")
native = importlib.import_module("tools.bakeoff.native_ort")
assert Path(evaluator.__file__).resolve() == root / "tools/bakeoff/evaluator.py"
assert Path(native.__file__).resolve() == root / "tools/bakeoff/native_ort.py"

sentinel = object()
native.load_runtime = lambda: sentinel
real_import = builtins.__import__

def blocked(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and (name == "onnxruntime" or name.startswith("onnxruntime.")):
        raise ImportError("onnxruntime intentionally blocked by package smoke")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = blocked
assert evaluator._onnxruntime() is sentinel
'''
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        result = subprocess.run(
            [sys.executable, "-c", smoke, str(extracted)],
            cwd=self.root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)


if __name__ == "__main__":
    unittest.main()
