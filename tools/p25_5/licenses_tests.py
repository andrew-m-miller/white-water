#!/usr/bin/env python3
"""Focused tests for the P25-5 candidate/runtime licence input boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from .licenses import (
    CANDIDATE_SURFACES,
    LICENSE_INPUT_SCHEMA_ID,
    RUNTIME_INPUT_SCHEMA_ID,
    LicenseInputError,
    _read_evidence,
    canonical_sha256,
    component_payload_sha256,
    collect_candidate,
    collect_runtime,
    generate_runtime_input,
    verify_runtime_content,
    validate_runtime_review,
)


ROOT = Path(__file__).resolve().parents[2]
SEA_RAFT_MANIFEST = ROOT / "models" / "sea-raft-m.json"


class LicenseInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="whitewater-p25-license-test-")
        self.root = Path(self.temp.name)
        self.candidate_root = self.root / "candidate"
        self.candidate_root.mkdir()
        self.manifest = self.candidate_root / "manifest.json"
        shutil.copy2(SEA_RAFT_MANIFEST, self.manifest)
        self.manifest.chmod(0o644)
        self.license_file = self.candidate_root / "BSD-3-Clause.txt"
        self.license_file.write_text("BSD-3-Clause evidence\n", encoding="utf-8")
        self.license_file.chmod(0o644)
        self.notice_files: dict[str, Path] = {}
        for surface in CANDIDATE_SURFACES:
            path = self.candidate_root / f"{surface}-notice.txt"
            path.write_text(f"SEA-RAFT {surface} notice evidence\n", encoding="utf-8")
            path.chmod(0o644)
            self.notice_files[surface] = path
        self.candidate_input = self.root / "candidate-input.json"
        self._write_candidate_input()

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _write_json(self, path: Path, value: object) -> None:
        path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        path.chmod(0o644)

    def _write_candidate_input(self) -> None:
        manifest_value = json.loads(self.manifest.read_text(encoding="utf-8"))
        surfaces = {}
        for surface in CANDIDATE_SURFACES:
            surfaces[surface] = {
                "license_file": str(self.license_file.relative_to(self.root)),
                "license_sha256": self._sha(self.license_file),
                "notice_file": str(self.notice_files[surface].relative_to(self.root)),
                "notice_sha256": self._sha(self.notice_files[surface]),
            }
        self._write_json(
            self.candidate_input,
            {
                "schema_id": LICENSE_INPUT_SCHEMA_ID,
                "candidate_id": "sea-raft-m",
                "manifest": str(self.manifest.relative_to(self.root)),
                "manifest_sha256": self._sha(self.manifest),
                "licenses_sha256": canonical_sha256(manifest_value["licenses"]),
                "surfaces": surfaces,
            },
        )

    def test_candidate_bundle_is_deterministic_and_deduplicates_identical_license_bytes(self) -> None:
        first = collect_candidate(self.candidate_input, self.root / "candidate-out-a")
        second = collect_candidate(self.candidate_input, self.root / "candidate-out-b")
        self.assertEqual(first, second)
        first_dir = self.root / "candidate-out-a"
        self.assertEqual(
            (first_dir / "LICENSES.txt").read_bytes(),
            (self.root / "candidate-out-b/LICENSES.txt").read_bytes(),
        )
        license_text = (first_dir / "LICENSES.txt").read_text(encoding="utf-8")
        self.assertEqual(license_text.count("evidence_sha256="), 1)
        notice_text = (first_dir / "NOTICES.txt").read_text(encoding="utf-8")
        self.assertEqual(notice_text.count("evidence_sha256="), 3)
        inventory = json.loads((first_dir / "candidate-license-inventory.json").read_text(encoding="utf-8"))
        self.assertEqual(inventory["schema_id"], "whitewater-p25-candidate-license-inventory-v1")
        self.assertEqual(stat.S_IMODE((first_dir / "LICENSES.txt").stat().st_mode), 0o644)
        self.assertEqual(stat.S_IMODE((first_dir / "NOTICES.txt").stat().st_mode), 0o644)

    def test_checked_in_sea_raft_candidate_evidence_is_hash_bound(self) -> None:
        checked_input = ROOT / "bakeoff" / "p25-5" / "candidate-license-input.json"
        result = collect_candidate(checked_input, self.root / "checked-candidate-out")
        self.assertEqual(result["candidate_id"], "sea-raft-m")
        self.assertEqual(result["manifest_sha256"], self._sha(SEA_RAFT_MANIFEST))

    def test_checked_in_runtime_supplement_is_exactly_lock_bound(self) -> None:
        supplement_path = ROOT / "bakeoff" / "p25-5" / "runtime-license-supplement.json"
        lock_path = ROOT / "bakeoff" / "p25-5" / "conda-el8-x86_64.lock"
        supplement = json.loads(supplement_path.read_text(encoding="utf-8"))
        lock_urls = {
            line.strip()
            for line in lock_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith(("#", "@"))
        }
        expected_names = {
            "libgcc",
            "libgfortran",
            "libgfortran5",
            "libgomp",
            "libsqlite",
            "libstdcxx",
        }
        records = supplement["packages"]
        self.assertEqual(len(records), len(expected_names))
        actual_names = set()
        for record in records:
            self.assertIn(record["package_url"], lock_urls)
            archive_name = record["package_url"].rsplit("/", 1)[-1].split("#", 1)[0]
            actual_names.add(archive_name.split("-", 1)[0])
            self.assertEqual(record["notice_files"], [])
            self.assertTrue(record["license_files"])
            for evidence in record["license_files"]:
                path = supplement_path.parent / evidence["path"]
                self.assertTrue(path.is_file() and not path.is_symlink())
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)
                self.assertEqual(self._sha(path), evidence["sha256"])
        self.assertEqual(actual_names, expected_names)

    def test_candidate_output_names_can_bind_existing_package_spec(self) -> None:
        result = collect_candidate(
            self.candidate_input,
            self.root / "candidate-named-out",
            license_name="SEA-RAFT-LICENSE.txt",
            notice_name="SEA-RAFT-NOTICE.txt",
        )
        output = self.root / "candidate-named-out"
        self.assertTrue((output / "SEA-RAFT-LICENSE.txt").is_file())
        self.assertTrue((output / "SEA-RAFT-NOTICE.txt").is_file())
        inventory = json.loads((output / "candidate-license-inventory.json").read_text(encoding="utf-8"))
        self.assertEqual(inventory["outputs"]["license"]["path"], "SEA-RAFT-LICENSE.txt")
        self.assertEqual(inventory["outputs"]["notice"]["path"], "SEA-RAFT-NOTICE.txt")
        self.assertEqual(result["license_sha256"], self._sha(output / "SEA-RAFT-LICENSE.txt"))

    def test_candidate_requires_exact_manifest_and_evidence_hashes(self) -> None:
        value = json.loads(self.candidate_input.read_text(encoding="utf-8"))
        value["manifest_sha256"] = "0" * 64
        self._write_json(self.candidate_input, value)
        with self.assertRaisesRegex(LicenseInputError, "manifest SHA256 mismatch"):
            collect_candidate(self.candidate_input, self.root / "out-manifest-hash")

        self._write_candidate_input()
        value = json.loads(self.candidate_input.read_text(encoding="utf-8"))
        value["surfaces"]["code"]["license_sha256"] = "0" * 64
        self._write_json(self.candidate_input, value)
        with self.assertRaisesRegex(LicenseInputError, "licence evidence SHA256 mismatch"):
            collect_candidate(self.candidate_input, self.root / "out-evidence-hash")

    def test_candidate_requires_all_three_surfaces_and_rejects_symlink_evidence(self) -> None:
        value = json.loads(self.candidate_input.read_text(encoding="utf-8"))
        del value["surfaces"]["backbone"]
        self._write_json(self.candidate_input, value)
        with self.assertRaisesRegex(LicenseInputError, "exactly code/checkpoint/backbone"):
            collect_candidate(self.candidate_input, self.root / "out-missing-surface")

        self._write_candidate_input()
        self.license_file.unlink()
        self.license_file.symlink_to(self.notice_files["code"])
        with self.assertRaisesRegex(LicenseInputError, "must not be a symlink"):
            collect_candidate(self.candidate_input, self.root / "out-symlink")

    def _runtime_fixture(self) -> tuple[Path, Path, Path, str]:
        prefix = self.root / "env"
        metadata = prefix / "conda-meta"
        if prefix.exists():
            shutil.rmtree(prefix)
        metadata.mkdir(parents=True)
        lock = self.root / "environment.lock"
        package_rows = [
            (
                "python",
                "3.11.0",
                "h1",
                "BSD-3-Clause",
                "https://packages.example.invalid/linux-64/python-3.11.0-h1.conda#sha256=" + "a" * 64,
            ),
            (
                "openssl",
                "3.0.0",
                "h2",
                "Apache-2.0",
                "https://packages.example.invalid/linux-64/openssl-3.0.0-h2.conda#sha256=" + "b" * 64,
            ),
        ]
        lock.write_text(
            "# platform: linux-64\n@EXPLICIT\n"
            + "\n".join(row[4] for row in package_rows)
            + "\n",
            encoding="utf-8",
        )
        lock.chmod(0o644)
        package_declarations = []
        for name, version, build, license_id, url in package_rows:
            metadata_path = metadata / f"{name}-{version}-{build}.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "name": name,
                        "version": version,
                        "build": build,
                        "build_number": 0,
                        "url": url.split("#", 1)[0],
                        "license": license_id,
                        "license_family": license_id.split("-", 1)[0],
                    },
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            metadata_path.chmod(0o644)
            license_path = self.root / f"{name}-LICENSE.txt"
            license_path.write_text(f"{license_id} licence evidence\n", encoding="utf-8")
            license_path.chmod(0o644)
            notice_path = self.root / f"{name}-NOTICE.txt"
            notice_path.write_text(f"{name} notice evidence\n", encoding="utf-8")
            notice_path.chmod(0o644)
            package_declarations.append(
                {
                    "package_url": url,
                    "license_files": [
                        {"path": str(license_path), "sha256": self._sha(license_path)}
                    ],
                    "notice_files": [
                        {"path": str(notice_path), "sha256": self._sha(notice_path)}
                    ],
                }
            )
        lock_sha = self._sha(lock)
        runtime_input = self.root / "runtime-input.json"
        self._write_json(
            runtime_input,
            {
                "schema_id": RUNTIME_INPUT_SCHEMA_ID,
                "lock_sha256": lock_sha,
                "packages": package_declarations,
                "components": [],
            },
        )
        return prefix, lock, runtime_input, lock_sha

    def test_runtime_bundle_binds_lock_metadata_and_is_deterministic(self) -> None:
        prefix, lock, runtime_input, lock_sha = self._runtime_fixture()
        first = collect_runtime(prefix, lock, runtime_input, self.root / "runtime-out-a", lock_sha)
        second = collect_runtime(prefix, lock, runtime_input, self.root / "runtime-out-b", lock_sha)
        self.assertEqual(first, second)
        first_dir = self.root / "runtime-out-a"
        inventory = json.loads((first_dir / "runtime-license-inventory.json").read_text(encoding="utf-8"))
        self.assertEqual(inventory["schema_id"], "whitewater-p25-runtime-license-inventory-v1")
        self.assertEqual(inventory["package_count"], 2)
        self.assertEqual(inventory["content"]["excluded_roots"], ["conda-meta"])
        self.assertEqual(inventory["content"]["excluded_directories"], ["__pycache__"])
        self.assertEqual(inventory["content"]["excluded_suffixes"], [".pyc"])
        self.assertEqual((first_dir / "LICENSES.txt").read_bytes(), (self.root / "runtime-out-b/LICENSES.txt").read_bytes())

    def test_runtime_content_revalidation_survives_relocation_and_missing_conda_meta(self) -> None:
        prefix, lock, runtime_input, lock_sha = self._runtime_fixture()
        payload = prefix / "bin" / "runtime-prefix.txt"
        payload.parent.mkdir(parents=True)
        payload.write_text(f"prefix={prefix}\n", encoding="utf-8")
        payload.chmod(0o644)
        # These are the files conda-pack adds to the final extracted prefix. They are part of
        # the reviewed identity; only Python's runtime-generated bytecode is excluded.
        for name in ("activate", "activate.fish", "deactivate", "conda-unpack"):
            activation = prefix / "bin" / name
            activation.write_text(f"#!/bin/sh\nPREFIX={prefix}\n", encoding="utf-8")
            activation.chmod(0o755)
        pycache = prefix / "lib" / "python3.11" / "__pycache__"
        pycache.mkdir(parents=True)
        (pycache / "generated.cpython-311.pyc").write_bytes(b"generated bytecode")
        (pycache / "generated.cpython-311.pyc").chmod(0o644)
        output = self.root / "runtime-content-out"
        collect_runtime(prefix, lock, runtime_input, output, lock_sha)
        inventory = output / "runtime-license-inventory.json"
        inventory_sha = self._sha(inventory)

        relocated = self.root / "relocated-runtime"
        shutil.copytree(prefix, relocated)
        relocated_payload = relocated / "bin" / "runtime-prefix.txt"
        relocated_payload.write_text(
            relocated_payload.read_text(encoding="utf-8").replace(str(prefix), str(relocated)),
            encoding="utf-8",
        )
        relocated_payload.chmod(0o644)
        for name in ("activate", "activate.fish", "deactivate", "conda-unpack"):
            activation = relocated / "bin" / name
            activation.write_text(
                activation.read_text(encoding="utf-8").replace(str(prefix), str(relocated)),
                encoding="utf-8",
            )
            activation.chmod(0o755)
        relocated_pycache = relocated / "lib" / "python3.11" / "__pycache__"
        (relocated_pycache / "newly-generated.cpython-311.pyc").write_bytes(b"new bytecode")
        (relocated_pycache / "newly-generated.cpython-311.pyc").chmod(0o644)
        shutil.rmtree(relocated / "conda-meta")
        result = verify_runtime_content(
            relocated,
            inventory,
            inventory_sha,
        )
        self.assertEqual(result["inventory_sha256"], inventory_sha)
        self.assertGreater(result["file_count"], 0)

        relocated_payload.write_text("tampered\n", encoding="utf-8")
        relocated_payload.chmod(0o644)
        with self.assertRaisesRegex(LicenseInputError, "runtime content SHA256 mismatch"):
            verify_runtime_content(relocated, inventory, inventory_sha)

    def test_runtime_inventory_requires_canonical_content_identity(self) -> None:
        prefix, lock, runtime_input, lock_sha = self._runtime_fixture()
        output = self.root / "runtime-content-required"
        collect_runtime(prefix, lock, runtime_input, output, lock_sha)
        inventory = output / "runtime-license-inventory.json"
        value = json.loads(inventory.read_text(encoding="utf-8"))
        del value["content"]
        self._write_json(inventory, value)
        with self.assertRaisesRegex(LicenseInputError, "canonical content identity"):
            verify_runtime_content(prefix, inventory, self._sha(inventory))

    def test_runtime_rejects_lock_mismatch_or_incomplete_evidence(self) -> None:
        prefix, lock, runtime_input, lock_sha = self._runtime_fixture()
        with self.assertRaisesRegex(LicenseInputError, "lock SHA256 mismatch"):
            collect_runtime(prefix, lock, runtime_input, self.root / "out-lock-hash", "0" * 64)

        value = json.loads(runtime_input.read_text(encoding="utf-8"))
        value["packages"].pop()
        self._write_json(runtime_input, value)
        with self.assertRaisesRegex(LicenseInputError, "lacks explicit licence/notice evidence"):
            collect_runtime(prefix, lock, runtime_input, self.root / "out-missing-package", lock_sha)

    def test_runtime_rejects_unknown_metadata_license_and_lock_set_mismatch(self) -> None:
        prefix, lock, runtime_input, lock_sha = self._runtime_fixture()
        metadata_path = next((prefix / "conda-meta").glob("openssl-*.json"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["license"] = "unknown"
        metadata_path.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        metadata_path.chmod(0o644)
        with self.assertRaisesRegex(LicenseInputError, "no usable licence identifier"):
            collect_runtime(prefix, lock, runtime_input, self.root / "out-unknown-license", lock_sha)

        prefix, lock, runtime_input, lock_sha = self._runtime_fixture()
        lock.write_text(lock.read_text(encoding="utf-8") + "https://packages.example.invalid/linux-64/extra-1-h3.conda#sha256=" + "c" * 64 + "\n", encoding="utf-8")
        lock.chmod(0o644)
        changed_sha = self._sha(lock)
        value = json.loads(runtime_input.read_text(encoding="utf-8"))
        value["lock_sha256"] = changed_sha
        self._write_json(runtime_input, value)
        with self.assertRaisesRegex(LicenseInputError, "missing from conda-meta"):
            collect_runtime(prefix, lock, runtime_input, self.root / "out-lock-set", changed_sha)

    def test_runtime_input_generator_uses_cached_license_files_and_no_fake_notices(self) -> None:
        prefix, lock, _runtime_input, lock_sha = self._runtime_fixture()
        cache = self.root / "conda-pkgs"
        for package_name in ("python-3.11.0-h1", "openssl-3.0.0-h2"):
            package_dir = cache / package_name
            licenses_dir = package_dir / "info" / "licenses"
            licenses_dir.mkdir(parents=True)
            metadata = json.loads(
                next((prefix / "conda-meta").glob(f"{package_name}.json")).read_text(encoding="utf-8")
            )
            (package_dir / "info" / "index.json").write_text(
                json.dumps(
                    {
                        "name": metadata["name"],
                        "version": metadata["version"],
                        "build": metadata["build"],
                        "license": metadata["license"],
                    },
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (package_dir / "info" / "index.json").chmod(0o644)
            license_file = licenses_dir / "LICENSE"
            license_file.write_text(f"{metadata['license']} cached evidence\n", encoding="utf-8")
            license_file.chmod(0o644)
        generated = self.root / "generated-runtime-input.json"
        result = generate_runtime_input(prefix, lock, cache, generated, lock_sha)
        self.assertEqual(result["package_count"], 2)
        generated_value = json.loads(generated.read_text(encoding="utf-8"))
        self.assertEqual(generated_value["components"], [])
        self.assertTrue(all(item["notice_files"] == [] for item in generated_value["packages"]))
        collect_runtime(prefix, lock, generated, self.root / "generated-runtime-out", lock_sha)

        component_root = prefix / "lib" / "whitewater" / "ort-cuda12" / "onnxruntime"
        component_root.mkdir(parents=True)
        ort_license = component_root / "LICENSE"
        ort_license.write_text("MIT component evidence\n", encoding="utf-8")
        ort_license.chmod(0o644)
        ort_notice = component_root / "ThirdPartyNotices.txt"
        ort_notice.write_text("ORT notices\n", encoding="utf-8")
        ort_notice.chmod(0o644)
        bridge_payload = prefix / "lib" / "whitewater" / "ort-cuda12" / "libwhitewater_ort_bridge.so"
        bridge_payload.write_bytes(b"native bridge payload\n")
        bridge_payload.chmod(0o755)
        component_manifest = self.root / "runtime-inputs.json"
        self._write_json(
            component_manifest,
            {
                "schema_id": "whitewater-p25-runtime-inputs-v1",
                "onnxruntime_cuda12": {
                    "version": "1.29.0",
                    "archive_url": "https://example.invalid/ort-1.29.0.tgz",
                    "archive_sha256": "c" * 64,
                    "license": "MIT",
                    "payload_root": "lib/whitewater/ort-cuda12/onnxruntime",
                },
                "native_bridge": {
                    "source": "native_bridge.cpp",
                    "source_sha256": "d" * 64,
                    "payload": "lib/whitewater/ort-cuda12/libwhitewater_ort_bridge.so",
                },
            },
        )
        generated_with_component = self.root / "generated-runtime-component-input.json"
        generate_runtime_input(
            prefix,
            lock,
            cache,
            generated_with_component,
            lock_sha,
            components_manifest_path=component_manifest,
        )
        generated_component_value = json.loads(generated_with_component.read_text(encoding="utf-8"))
        self.assertEqual(generated_component_value["components"][0]["component_id"], "onnxruntime-cuda12")
        self.assertEqual(
            generated_component_value["technical_components"][0]["component_id"],
            "whitewater-native-ort-bridge",
        )
        self.assertEqual(
            generated_component_value["technical_components"][0]["source_sha256"], "d" * 64
        )
        collect_runtime(
            prefix,
            lock,
            generated_with_component,
            self.root / "generated-runtime-component-out",
            lock_sha,
        )
        component_inventory = json.loads(
            (
                self.root
                / "generated-runtime-component-out/runtime-license-inventory.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(component_inventory["technical_component_count"], 1)
        self.assertEqual(
            component_inventory["technical_components"][0]["payload_path"],
            "lib/whitewater/ort-cuda12/libwhitewater_ort_bridge.so",
        )

        missing_package_dir = cache / "python-3.11.0-h1" / "info" / "licenses"
        shutil.rmtree(missing_package_dir)
        supplement_file = self.root / "python-supplement-license.txt"
        supplement_file.write_text("supplemented python license\n", encoding="utf-8")
        supplement_file.chmod(0o644)
        python_metadata = json.loads(
            (prefix / "conda-meta/python-3.11.0-h1.json").read_text(encoding="utf-8")
        )
        supplement = self.root / "runtime-license-supplement.json"
        self._write_json(
            supplement,
            {
                "schema_id": "whitewater-p25-runtime-license-supplement-v1",
                "packages": [
                    {
                        "package_url": python_metadata["url"],
                        "license_files": [
                            {"path": supplement_file.name, "sha256": self._sha(supplement_file)}
                        ],
                        "notice_files": [],
                    }
                ],
            },
        )
        supplemented = self.root / "generated-sidecar" / "runtime-supplemented.json"
        generate_runtime_input(
            prefix,
            lock,
            cache,
            supplemented,
            lock_sha,
            supplement_path=supplement,
        )
        supplemented_value = json.loads(supplemented.read_text(encoding="utf-8"))
        python_record = next(
            item for item in supplemented_value["packages"] if "/python-3.11.0-h1.conda" in item["package_url"]
        )
        self.assertTrue(Path(python_record["license_files"][0]["path"]).is_absolute())
        collect_runtime(prefix, lock, supplemented, self.root / "generated-runtime-supplemented-out", lock_sha)

    def _build_package_cache(self, prefix: Path, cache: Path) -> dict[str, Path]:
        """Populate a conda package cache from the fixture's conda-meta and return license paths."""

        license_paths: dict[str, Path] = {}
        for package_name in ("python-3.11.0-h1", "openssl-3.0.0-h2"):
            package_dir = cache / package_name
            licenses_dir = package_dir / "info" / "licenses"
            licenses_dir.mkdir(parents=True)
            metadata = json.loads(
                next((prefix / "conda-meta").glob(f"{package_name}.json")).read_text(encoding="utf-8")
            )
            (package_dir / "info" / "index.json").write_text(
                json.dumps(
                    {
                        "name": metadata["name"],
                        "version": metadata["version"],
                        "build": metadata["build"],
                        "license": metadata["license"],
                    },
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (package_dir / "info" / "index.json").chmod(0o644)
            license_file = licenses_dir / "LICENSE"
            license_file.write_text(f"{metadata['license']} cached evidence\n", encoding="utf-8")
            license_file.chmod(0o644)
            license_paths[package_name] = license_file
        return license_paths

    def test_generate_harvests_pip_distribution_license_evidence(self) -> None:
        # conda-pack ships the full export env, so every installed wheel must land in the
        # inventory. generate-runtime-input harvests each wheel's own bundled licence (PEP 639
        # License-File under dist-info/licenses/, keyed off the modern License-Expression field),
        # and a wheel that bundles nothing falls back to a pip:// supplement entry.
        prefix, lock, _runtime_input, lock_sha = self._runtime_fixture()
        cache = self.root / "conda-pkgs"
        self._build_package_cache(prefix, cache)
        site = prefix / "lib" / "python3.11" / "site-packages"

        harvested = site / "torchy-2.7.0.dist-info"
        (harvested / "licenses").mkdir(parents=True)
        (harvested / "METADATA").write_text(
            "Metadata-Version: 2.4\nName: torchy\nVersion: 2.7.0\n"
            "License-Expression: BSD-3-Clause\nLicense-File: LICENSE\n\n",
            encoding="utf-8",
        )
        # Wheels/pip control the METADATA mode (protobuf ships it 0755); it is bound by content
        # SHA, so a non-0644 dist metadata file must be tolerated.
        (harvested / "METADATA").chmod(0o755)
        (harvested / "licenses" / "LICENSE").write_text(
            "torchy BSD-3-Clause bundled text\n", encoding="utf-8"
        )
        (harvested / "licenses" / "LICENSE").chmod(0o644)

        bundleless = site / "barepkg-1.0.dist-info"
        bundleless.mkdir(parents=True)
        (bundleless / "METADATA").write_text(
            "Metadata-Version: 2.4\nName: barepkg\nVersion: 1.0\nLicense-Expression: MIT\n\n",
            encoding="utf-8",
        )
        (bundleless / "METADATA").chmod(0o644)
        supp_license = self.root / "barepkg-LICENSE.txt"
        supp_license.write_text("barepkg MIT supplement evidence\n", encoding="utf-8")
        supp_license.chmod(0o644)
        supplement = self.root / "pip-supplement.json"
        self._write_json(
            supplement,
            {
                "schema_id": "whitewater-p25-runtime-license-supplement-v1",
                "packages": [
                    {
                        "package_url": "pip://barepkg==1.0",
                        "license_files": [
                            {"path": supp_license.name, "sha256": self._sha(supp_license)}
                        ],
                        "notice_files": [],
                    }
                ],
            },
        )

        generated = self.root / "gen" / "runtime-pip-input.json"
        result = generate_runtime_input(
            prefix, lock, cache, generated, lock_sha, supplement_path=supplement
        )
        self.assertEqual(result["package_count"], 4)  # 2 conda + 2 pip
        generated_value = json.loads(generated.read_text(encoding="utf-8"))
        torchy = next(
            item for item in generated_value["packages"] if item["package_url"] == "pip://torchy==2.7.0"
        )
        self.assertIn("metadata_sha256", torchy)
        self.assertTrue(torchy["license_files"][0]["path"].startswith("pip-license-evidence/"))

        collect_runtime(prefix, lock, generated, self.root / "pip-harvest-out", lock_sha)
        inventory = json.loads(
            (self.root / "pip-harvest-out/runtime-license-inventory.json").read_text(encoding="utf-8")
        )
        self.assertEqual(inventory["package_count"], 4)
        pip_ids = {
            item["package_url"]
            for item in inventory["packages"]
            if item["package_url"].startswith("pip://")
        }
        self.assertEqual(pip_ids, {"pip://torchy==2.7.0", "pip://barepkg==1.0"})
        aggregated = (self.root / "pip-harvest-out/LICENSES.txt").read_bytes()
        self.assertIn(b"torchy BSD-3-Clause bundled text", aggregated)
        self.assertIn(b"barepkg MIT supplement evidence", aggregated)

        # A wheel that bundles no licence and has no supplement entry fails closed by name.
        (site / "orphan-9.9.dist-info").mkdir(parents=True)
        (site / "orphan-9.9.dist-info" / "METADATA").write_text(
            "Metadata-Version: 2.4\nName: orphan\nVersion: 9.9\nLicense-Expression: MIT\n\n",
            encoding="utf-8",
        )
        (site / "orphan-9.9.dist-info" / "METADATA").chmod(0o644)
        with self.assertRaisesRegex(LicenseInputError, "no licence file"):
            generate_runtime_input(
                prefix, lock, cache, self.root / "gen-orphan" / "input.json", lock_sha,
                supplement_path=supplement,
            )

    def test_upstream_package_cache_license_mode_is_tolerated_but_owned_evidence_is_not(self) -> None:
        # conda-forge ships some upstream license files non-0644 (e.g. intel-gmmlib's LICENSE.md at
        # 0755). Their mode is not our integrity property, so the package-cache license-evidence read
        # must accept them while still binding content + SHA256. Everything we own or carry keeps the
        # 0644 requirement, and symlinks are rejected on every path.
        prefix, lock, _runtime_input, lock_sha = self._runtime_fixture()
        cache = self.root / "conda-pkgs"
        license_paths = self._build_package_cache(prefix, cache)
        # An upstream license file with mode 0755 must be tolerated and bound by content hash.
        upstream_license = license_paths["openssl-3.0.0-h2"]
        upstream_license.chmod(0o755)
        upstream_sha = self._sha(upstream_license)
        generated = self.root / "generated-mode-tolerant.json"
        result = generate_runtime_input(prefix, lock, cache, generated, lock_sha)
        self.assertEqual(result["package_count"], 2)
        generated_value = json.loads(generated.read_text(encoding="utf-8"))
        openssl_record = next(
            item for item in generated_value["packages"] if "/openssl-3.0.0-h2.conda" in item["package_url"]
        )
        self.assertEqual(openssl_record["license_files"][0]["sha256"], upstream_sha)
        collect_runtime(prefix, lock, generated, self.root / "generated-mode-tolerant-out", lock_sha)

        # A symlinked package-cache license file is still rejected even though the mode is relaxed.
        symlink_cache = self.root / "conda-pkgs-symlink"
        symlink_paths = self._build_package_cache(prefix, symlink_cache)
        victim = symlink_paths["openssl-3.0.0-h2"]
        real_target = self.root / "outside-license.txt"
        real_target.write_text("Apache-2.0 cached evidence\n", encoding="utf-8")
        victim.unlink()
        victim.symlink_to(real_target)
        with self.assertRaises(LicenseInputError) as symlink_error:
            generate_runtime_input(
                prefix, lock, symlink_cache, self.root / "generated-symlink.json", lock_sha
            )
        self.assertIn("symlink", str(symlink_error.exception))

        # Evidence we own or carry still requires 0644: only the package-cache read path is relaxed.
        owned = self.root / "owned-evidence.txt"
        owned.write_text("carried aggregate evidence\n", encoding="utf-8")
        owned.chmod(0o755)
        owned_sha = self._sha(owned)
        with self.assertRaises(LicenseInputError) as owned_error:
            _read_evidence(owned, owned_sha, "carried output")
        self.assertIn("mode", str(owned_error.exception))
        # The same file is accepted only on the mode-agnostic upstream path.
        tolerated = _read_evidence(owned, owned_sha, "package-cache evidence", require_mode=False)
        self.assertEqual(tolerated.sha256, owned_sha)
        # A symlink is rejected on the mode-agnostic path too.
        owned_symlink = self.root / "owned-evidence-symlink.txt"
        owned_symlink.symlink_to(owned)
        with self.assertRaises(LicenseInputError) as symlink_owned_error:
            _read_evidence(owned_symlink, owned_sha, "package-cache evidence", require_mode=False)
        self.assertIn("symlink", str(symlink_owned_error.exception))

    def test_runtime_notice_list_can_be_explicitly_empty(self) -> None:
        prefix, lock, runtime_input, lock_sha = self._runtime_fixture()
        value = json.loads(runtime_input.read_text(encoding="utf-8"))
        value["packages"][0]["notice_files"] = []
        self._write_json(runtime_input, value)
        collect_runtime(prefix, lock, runtime_input, self.root / "empty-notice-out", lock_sha)
        inventory = json.loads(
            (self.root / "empty-notice-out/runtime-license-inventory.json").read_text(encoding="utf-8")
        )
        python_entry = next(item for item in inventory["packages"] if item["name"] == "python")
        self.assertEqual(python_entry["notice_evidence"], [])

    def test_runtime_accepts_pip_distribution_without_treating_it_as_conda(self) -> None:
        prefix, lock, runtime_input, lock_sha = self._runtime_fixture()
        site = prefix / "lib" / "python3.11" / "site-packages"
        dist_info = site / "onnxruntime_gpu-1.29.0.dist-info"
        dist_info.mkdir(parents=True)
        metadata = dist_info / "METADATA"
        metadata.write_text(
            "Metadata-Version: 2.1\n"
            "Name: onnxruntime-gpu\n"
            "Version: 1.29.0\n"
            "License: MIT\n\n",
            encoding="utf-8",
        )
        metadata.chmod(0o644)
        pip_license = self.root / "onnxruntime-gpu-LICENSE.txt"
        pip_license.write_text("onnxruntime-gpu MIT licence evidence\n", encoding="utf-8")
        pip_license.chmod(0o644)
        pip_notice = self.root / "onnxruntime-gpu-NOTICE.txt"
        pip_notice.write_text("onnxruntime-gpu notice evidence\n", encoding="utf-8")
        pip_notice.chmod(0o644)
        value = json.loads(runtime_input.read_text(encoding="utf-8"))
        value["packages"].append(
            {
                "package_url": "pip://onnxruntime-gpu==1.29.0",
                "metadata_sha256": self._sha(metadata),
                "license_files": [
                    {"path": str(pip_license), "sha256": self._sha(pip_license)}
                ],
                "notice_files": [
                    {"path": str(pip_notice), "sha256": self._sha(pip_notice)}
                ],
            }
        )
        self._write_json(runtime_input, value)
        result = collect_runtime(prefix, lock, runtime_input, self.root / "runtime-pip-out", lock_sha)
        self.assertEqual(result["package_count"], 3)
        inventory = json.loads(
            (self.root / "runtime-pip-out/runtime-license-inventory.json").read_text(encoding="utf-8")
        )
        pip_entries = [item for item in inventory["packages"] if item["package_url"].startswith("pip://")]
        self.assertEqual(len(pip_entries), 1)
        self.assertEqual(pip_entries[0]["package_url"], "pip://onnxruntime-gpu==1.29.0")

        value["packages"] = [item for item in value["packages"] if not item["package_url"].startswith("pip://")]
        self._write_json(runtime_input, value)
        with self.assertRaisesRegex(LicenseInputError, "pip runtime package lacks explicit"):
            collect_runtime(prefix, lock, runtime_input, self.root / "runtime-pip-missing", lock_sha)

    def test_runtime_binds_non_conda_component_payload_and_evidence(self) -> None:
        prefix, lock, runtime_input, lock_sha = self._runtime_fixture()
        component_root = prefix / "lib" / "whitewater" / "ort-cuda12"
        component_root.mkdir(parents=True)
        license_path = component_root / "LICENSE"
        license_path.write_text("component BSD evidence\n", encoding="utf-8")
        license_path.chmod(0o644)
        notice_path = component_root / "ThirdPartyNotices.txt"
        notice_path.write_text("component third-party notices\n", encoding="utf-8")
        notice_path.chmod(0o644)
        value = json.loads(runtime_input.read_text(encoding="utf-8"))
        value["components"] = [
            {
                "component_id": "onnxruntime-cuda12",
                "version": "1.29.0",
                "source": "microsoft-official-cuda12-archive",
                "license": "BSD-3-Clause",
                "payload_path": "lib/whitewater/ort-cuda12",
                "payload_sha256": component_payload_sha256(component_root),
                "license_files": [
                    {"path": "lib/whitewater/ort-cuda12/LICENSE", "sha256": self._sha(license_path)}
                ],
                "notice_files": [
                    {
                        "path": "lib/whitewater/ort-cuda12/ThirdPartyNotices.txt",
                        "sha256": self._sha(notice_path),
                    }
                ],
            }
        ]
        self._write_json(runtime_input, value)
        result = collect_runtime(prefix, lock, runtime_input, self.root / "runtime-component-out", lock_sha)
        self.assertEqual(result["package_count"], 2)
        inventory = json.loads(
            (self.root / "runtime-component-out/runtime-license-inventory.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(inventory["component_count"], 1)
        self.assertEqual(inventory["components"][0]["component_id"], "onnxruntime-cuda12")
        self.assertIn("component://onnxruntime-cuda12@1.29.0", (self.root / "runtime-component-out/LICENSES.txt").read_text(encoding="utf-8"))

        notice_path.write_text("tampered\n", encoding="utf-8")
        notice_path.chmod(0o644)
        with self.assertRaisesRegex(LicenseInputError, "payload SHA256 mismatch"):
            collect_runtime(prefix, lock, runtime_input, self.root / "runtime-component-tampered", lock_sha)

    def test_component_payload_hash_streams_and_preserves_digest_contract(self) -> None:
        payload = self.root / "large-component.bin"
        payload_bytes = bytes(range(256)) * (8192 + 1)
        payload.write_bytes(payload_bytes)
        payload.chmod(0o755)

        expected = hashlib.sha256()
        expected.update(b"whitewater-p25-runtime-component-v1\0")
        expected.update(b"file\0")
        expected.update(payload.name.encode("utf-8"))
        expected.update(b"\0")
        expected.update(f"0755:{len(payload_bytes)}\0".encode("ascii"))
        expected.update(payload_bytes)

        reader = mock.mock_open(read_data=payload_bytes)
        with (
            mock.patch.object(Path, "read_bytes", side_effect=AssertionError("read_bytes used")),
            mock.patch.object(Path, "open", reader),
        ):
            actual = component_payload_sha256(payload)

        self.assertEqual(actual, expected.hexdigest())
        read_calls = reader.return_value.read.call_args_list
        self.assertGreater(len(read_calls), 1)
        self.assertTrue(all(call.args == (1024 * 1024,) for call in read_calls))

    def test_runtime_review_is_hash_bound_to_generated_inventory_and_reviewer(self) -> None:
        prefix, lock, runtime_input, lock_sha = self._runtime_fixture()
        output = self.root / "runtime-review-out"
        collect_runtime(prefix, lock, runtime_input, output, lock_sha)
        inventory_sha = self._sha(output / "runtime-license-inventory.json")
        review = self.root / "runtime-legal-review.json"
        self._write_json(
            review,
            {
                "schema_id": "whitewater-p25-runtime-legal-review-v1",
                "reviewer": "Andrew Miller",
                "reviewed": True,
                "reviewed_at": "2026-08-24T20:00:00Z",
                "inventory_sha256": inventory_sha,
                "statement": "I approve this exact runtime license and notice inventory for P25-5 evaluation-only use.",
            },
        )
        review_sha = self._sha(review)
        result = validate_runtime_review(review, review_sha, inventory_sha)
        self.assertEqual(result["inventory_sha256"], inventory_sha)
        with self.assertRaisesRegex(LicenseInputError, "does not match generated runtime inventory"):
            validate_runtime_review(review, review_sha, "0" * 64)
        value = json.loads(review.read_text(encoding="utf-8"))
        value["reviewer"] = "not Andrew Miller"
        self._write_json(review, value)
        with self.assertRaisesRegex(LicenseInputError, "reviewer must be Andrew Miller"):
            validate_runtime_review(review, self._sha(review), inventory_sha)

    def test_runtime_review_requires_timezone_qualified_iso8601_timestamp(self) -> None:
        prefix, lock, runtime_input, lock_sha = self._runtime_fixture()
        output = self.root / "runtime-review-timestamp-out"
        collect_runtime(prefix, lock, runtime_input, output, lock_sha)
        inventory_sha = self._sha(output / "runtime-license-inventory.json")
        review = self.root / "runtime-legal-review-timestamp.json"
        document = {
            "schema_id": "whitewater-p25-runtime-legal-review-v1",
            "reviewer": "Andrew Miller",
            "reviewed": True,
            "reviewed_at": "2026-08-24T20:00:00Z",
            "inventory_sha256": inventory_sha,
            "statement": "I approve this exact runtime license and notice inventory for P25-5 evaluation-only use.",
        }

        self._write_json(review, document)
        self.assertEqual(
            validate_runtime_review(review, self._sha(review), inventory_sha)["inventory_sha256"],
            inventory_sha,
        )

        document["reviewed_at"] = "2026-08-24T20:00:00"
        self._write_json(review, document)
        with self.assertRaisesRegex(LicenseInputError, "must include an explicit timezone"):
            validate_runtime_review(review, self._sha(review), inventory_sha)

        document["reviewed_at"] = "not-an-iso8601-timestamp"
        self._write_json(review, document)
        with self.assertRaisesRegex(LicenseInputError, "must be an ISO-8601 timestamp"):
            validate_runtime_review(review, self._sha(review), inventory_sha)

    def test_cli_emits_json_result(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "p25_5" / "licenses.py"),
                "candidate",
                "--input",
                str(self.candidate_input),
                "--output-dir",
                str(self.root / "cli-output"),
            ],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["candidate_id"], "sea-raft-m")


if __name__ == "__main__":
    unittest.main()
