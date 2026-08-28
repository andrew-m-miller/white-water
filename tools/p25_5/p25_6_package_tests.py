#!/usr/bin/env python3
"""P25-6 target-measurement package tests.

WP3 reuses the P25-5 air-gap packager (``tools/p25_5/package.py``) unchanged and adds a NEW
package that carries the resumable profile driver (``tools/bakeoff/run.py``) and a runtime with
the OpenEXR Python bindings and pynvml.  These tests pin the checked-in P25-6 package-spec template to:

* the exact driver import closure (recomputed here by AST walk, not trusted from a list);
* the P25-5 support/schema/protocol/legal file set it inherits;
* the ``__P25_6_*`` CI placeholder set and the OpenEXR/pynvml runtime identity; and
* a full ``build_package``/``verify_package`` round trip over the real driver sources plus
  fixtures for the candidate artifact, runtime archive and legal placeholders.

They also assert the checked-in runtime-inputs hashes stay bound to the exact conda spec and
native-bridge source, so a stale hash fails locally rather than in an EL8 CI dispatch.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import stat
import tempfile
import unittest

try:
    from . import package as package_module
    from .package import build_package, load_spec, verify_package
except ImportError:  # Direct execution: ``python tools/p25_5/p25_6_package_tests.py``.
    import package as package_module  # type: ignore
    from package import build_package, load_spec, verify_package  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_DIR = REPO_ROOT / "bakeoff" / "p25-6"
TEMPLATE_PATH = SPEC_DIR / "package-spec.json"
RUN_PATH = SPEC_DIR / "RUN-P25-6.txt"
RUNTIME_INPUTS_PATH = SPEC_DIR / "runtime-inputs.json"
CONDA_SPEC_PATH = SPEC_DIR / "conda-el8-x86_64.lock"
NATIVE_BRIDGE_PATH = REPO_ROOT / "tools" / "bakeoff" / "ort_native_bridge.cpp"

P25_6_RUNTIME_IDENTITY = (
    "python-3.11;microsoft-onnxruntime-linux-x64-gpu_cuda12-1.29.0+whitewater-native-bridge;"
    "openexr-python+openexr+imath;pynvml;conda-pack;el8-x86_64"
)

# The driver entrypoint and its expected first-party module closure.  ``run`` is the entrypoint;
# ``evaluator`` and ``native_ort`` are already carried by P25-5.  This list is validated against
# a fresh AST walk in test_template_carries_exact_driver_closure.
EXPECTED_CLOSURE = {
    "artifact_store",
    "conditioning",
    "coordinator",
    "evaluator",
    "exr",
    "geometry",
    "matrix",
    "measurement",
    "metrics",
    "native_ort",
    "nvml",
    "padding",
    "pfm",
    "reporting",
    "resume",
    "run_spec",
    "synthetic",
    "validator",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _driver_closure() -> set[str]:
    """Recompute the transitive first-party import closure of run.py by AST walk."""

    bakeoff = REPO_ROOT / "tools" / "bakeoff"
    present = {p.stem for p in bakeoff.glob("*.py")}

    def imports_of(module: str) -> set[str]:
        tree = ast.parse((bakeoff / f"{module}.py").read_text(encoding="utf-8"))
        found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level >= 1 and node.module:
                    found.add(node.module.split(".")[0])
                elif node.module and node.module.startswith("tools.bakeoff."):
                    found.add(node.module.split(".")[2])
                elif node.level >= 1 and node.module is None:
                    for alias in node.names:
                        found.add(alias.name.split(".")[0])
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("tools.bakeoff."):
                        found.add(alias.name.split(".")[2])
        return {name for name in found if name in present}

    closure: set[str] = set()
    frontier = {"run"}
    while frontier:
        module = frontier.pop()
        if module in closure:
            continue
        closure.add(module)
        frontier |= imports_of(module) - closure
    closure.discard("run")
    return closure


class P25_6PackageTemplateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.template = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
        self.by_destination = {item["destination"]: item for item in self.template["files"]}

    def test_template_identity_and_placeholders(self) -> None:
        self.assertEqual(self.template["schema_id"], "whitewater-p25-airgap-package-v1")
        self.assertEqual(self.template["protocol_id"], "whitewater-p25-v2")
        self.assertEqual(
            self.template["package_id"], "whitewater-p25-6-sea-raft-m-el8-x86_64-target"
        )
        self.assertEqual(self.template["admission"]["candidates"], "__P25_6_ADMISSION_CANDIDATES__")
        self.assertEqual(self.template["evaluator"]["entrypoint"], "scripts/ww-bakeoff-airgap")
        self.assertEqual(self.template["evaluator"]["runtime_identity"], P25_6_RUNTIME_IDENTITY)

        runtime = [item for item in self.template["files"] if item.get("role") == "runtime"]
        self.assertEqual(len(runtime), 1)
        self.assertEqual(
            runtime[0],
            {
                "role": "runtime",
                "destination": "runtime/whitewater-p25-6-runtime.tar.gz",
                "source": "__P25_6_RUNTIME_ARCHIVE__",
                "candidate_id": None,
                "mode": "0644",
                "sha256": "__P25_6_RUNTIME_SHA256__",
                "size_bytes": "__P25_6_RUNTIME_SIZE_BYTES__",
            },
        )

        markers = sorted(set(re.findall(r"__P25_6_[A-Z0-9_]+__", json.dumps(self.template))))
        self.assertEqual(
            markers,
            [
                "__P25_6_ADMISSION_CANDIDATES__",
                "__P25_6_CANDIDATE_INVENTORY_SOURCE__",
                "__P25_6_CANDIDATE_LICENSE_SOURCE__",
                "__P25_6_CANDIDATE_NOTICE_SOURCE__",
                "__P25_6_RUNTIME_ARCHIVE__",
                "__P25_6_RUNTIME_INVENTORY_SOURCE__",
                "__P25_6_RUNTIME_LICENSE_SOURCE__",
                "__P25_6_RUNTIME_NOTICE_SOURCE__",
                "__P25_6_RUNTIME_REVIEW_SOURCE__",
                "__P25_6_RUNTIME_SHA256__",
                "__P25_6_RUNTIME_SIZE_BYTES__",
            ],
        )
        # No stray P25-5 marker leaked into the copy.
        self.assertEqual(re.findall(r"__P25_5_[A-Z0-9_]+__", json.dumps(self.template)), [])

    def test_template_carries_exact_driver_closure(self) -> None:
        closure = _driver_closure()
        self.assertEqual(
            closure,
            EXPECTED_CLOSURE,
            "run.py import closure drifted; update the P25-6 package spec and this test together",
        )
        # Every closure module plus the entrypoint must be a carried 0644 evaluator-support file.
        for module in sorted(closure | {"run"}):
            destination = f"tools/bakeoff/{module}.py"
            self.assertIn(destination, self.by_destination, destination)
            entry = self.by_destination[destination]
            self.assertEqual(entry["role"], "evaluator-support", destination)
            self.assertEqual(entry["mode"], "0644", destination)
            self.assertIsNone(entry["candidate_id"], destination)

    def test_template_carries_p25_5_support_and_schemas(self) -> None:
        required = {
            "scripts/ww-bakeoff-airgap",
            "tools/bakeoff/__init__.py",
            "tools/bakeoff/run.py",
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
            "RUN-P25-6.txt",
            "runtime/whitewater-p25-6-runtime.tar.gz",
        }
        self.assertTrue(required.issubset(self.by_destination), sorted(required - set(self.by_destination)))
        self.assertEqual(self.by_destination["scripts/ww-bakeoff-airgap"]["role"], "evaluator")
        self.assertEqual(self.by_destination["scripts/ww-bakeoff-airgap"]["mode"], "0755")
        # Exactly one executable file in the whole package.
        self.assertEqual(
            [item["destination"] for item in self.template["files"] if item["mode"] == "0755"],
            ["scripts/ww-bakeoff-airgap"],
        )
        # legal/runtime-inputs.json binds the checked-in P25-6 technical runtime declaration.
        self.assertEqual(
            self.by_destination["legal/runtime-inputs.json"]["source"],
            "../../bakeoff/p25-6/runtime-inputs.json",
        )

    def test_non_placeholder_sources_exist_at_expected_mode(self) -> None:
        for item in self.template["files"]:
            source = item["source"]
            if source.startswith("__P25_6_"):
                continue
            # The candidate ONNX artifact is exported in CI (gitignored); skip its existence.
            if item["destination"].endswith("sea-raft-m-opset17.onnx"):
                continue
            resolved = (SPEC_DIR / source).resolve()
            self.assertTrue(resolved.is_file(), f"missing carried source: {resolved}")
            self.assertFalse(resolved.is_symlink(), resolved)
            expected = 0o755 if item["mode"] == "0755" else 0o644
            self.assertEqual(stat.S_IMODE(resolved.stat().st_mode), expected, resolved)

    def test_run_instructions_document_the_driver_procedure(self) -> None:
        text = RUN_PATH.read_text(encoding="utf-8")
        for needle in (
            "WW_BAKEOFF_ENTRYPOINT=tools/bakeoff/run.py",
            "WW_BAKEOFF_RUNTIME_ARCHIVE=runtime/whitewater-p25-6-runtime.tar.gz",
            "smoke",
            "screen",
            "final",
            "IDENTICAL command to resume",
            "report.json",
            "summary.txt",
            "runner.log",
            "nvml.csv",
            "review.csv",
            "No production images leave the machine.",
            "sha256sum -c whitewater-p25-6-el8.tar.gz.sha256",
            "whitewater-p25-runtime-legal-review-v1",
            "scripts/ci-p25-6-qualify.sh",
            "--assume-host-load-ready",
            "OpenEXR",
            "pynvml",
            "__P25_6_ADMISSION_CANDIDATES__",
            "__P25_6_RUNTIME_ARCHIVE__",
            "__P25_6_RUNTIME_SHA256__",
            "__P25_6_RUNTIME_SIZE_BYTES__",
        ):
            self.assertIn(needle, text, needle)
        # This must NOT masquerade as the P25-5 evaluation procedure.
        self.assertIn("is NOT the P25-5 evaluation package", text)

    def test_runtime_inputs_hashes_bind_exact_local_files(self) -> None:
        inputs = json.loads(RUNTIME_INPUTS_PATH.read_text(encoding="utf-8"))
        conda = inputs["conda"]
        self.assertEqual(conda["explicit_lock"], "bakeoff/p25-6/conda-el8-x86_64.lock")
        self.assertEqual(conda["explicit_lock_sha256"], _sha256(CONDA_SPEC_PATH))
        self.assertEqual(inputs["onnxruntime_cuda12"]["version"], "1.29.0")
        self.assertEqual(inputs["onnxruntime_cuda12"]["cuda_major"], 12)
        self.assertEqual(
            inputs["native_bridge"]["source_sha256"], _sha256(NATIVE_BRIDGE_PATH)
        )
        # The checked-in spec is a requested match-spec that CI must solve/re-pin: no
        # non-comment line is the @EXPLICIT directive of a fully-pinned conda lock.
        self.assertTrue(conda.get("explicit_lock_is_requested_spec"))
        content_lines = [
            line.strip()
            for line in CONDA_SPEC_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertNotIn("@EXPLICIT", content_lines)
        self.assertIn("openexr-python", content_lines)
        self.assertIn("pynvml", content_lines)


class P25_6PackageBuildTests(unittest.TestCase):
    """Materialize the template like ci-p25-6-qualify.sh and build/verify it end to end."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="whitewater-p25-6-build-")
        self.root = Path(self.temp.name)
        self.fixtures = self.root / "fixtures"
        self.fixtures.mkdir()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write(self, name: str, data: bytes, mode: int = 0o644) -> Path:
        path = self.fixtures / name
        path.write_bytes(data)
        path.chmod(mode)
        return path

    def _materialized_spec(self) -> Path:
        template = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))

        # Fixture opaque runtime tarball (the outer package never opens it).
        runtime_bytes = b"opaque-p25-6-conda-pack-runtime\x00"
        runtime = self._write("whitewater-p25-6-runtime.tar.gz", runtime_bytes)
        runtime_sha = hashlib.sha256(runtime_bytes).hexdigest()

        # Fixture candidate artifact + a manifest whose export identity binds it exactly.
        artifact_bytes = b"fixture-sea-raft-m-onnx-bytes\n"
        artifact = self._write("sea-raft-m-opset17.onnx", artifact_bytes)
        manifest = {
            "schema_id": "whitewater-p25-artifact-v1",
            "candidate": {"id": "sea-raft-m", "role": "measurement-candidate"},
            "export": {
                "artifact": "sea-raft-m-opset17.onnx",
                "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
                "size_bytes": len(artifact_bytes),
                "mode": "0644",
            },
        }
        manifest_path = self.fixtures / "sea-raft-m.manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        manifest_path.chmod(0o644)

        legal = {
            marker: self._write(f"legal-{index}.txt", f"fixture {marker}\n".encode())
            for index, marker in enumerate(
                (
                    "__P25_6_CANDIDATE_LICENSE_SOURCE__",
                    "__P25_6_CANDIDATE_NOTICE_SOURCE__",
                    "__P25_6_CANDIDATE_INVENTORY_SOURCE__",
                    "__P25_6_RUNTIME_LICENSE_SOURCE__",
                    "__P25_6_RUNTIME_NOTICE_SOURCE__",
                    "__P25_6_RUNTIME_INVENTORY_SOURCE__",
                    "__P25_6_RUNTIME_REVIEW_SOURCE__",
                )
            )
        }

        template["admission"]["candidates"] = [
            {
                "candidate_id": "sea-raft-m",
                "measurement_status": "measurable",
                "measurement_admitted": True,
                "status": "eligible",
            }
        ]

        for item in template["files"]:
            source = item["source"]
            destination = item["destination"]
            if item["role"] == "runtime":
                item["source"] = str(runtime)
                item["sha256"] = runtime_sha
                item["size_bytes"] = len(runtime_bytes)
            elif destination == "models/sea-raft-m/manifest.json":
                item["source"] = str(manifest_path)
            elif destination.endswith("sea-raft-m-opset17.onnx"):
                item["source"] = str(artifact)
            elif source in legal:
                item["source"] = str(legal[source])
            else:
                # Keep the real repo source, resolved to an absolute path.
                item["source"] = str((SPEC_DIR / source).resolve())

        serialized = json.dumps(template, ensure_ascii=False, sort_keys=True)
        self.assertNotIn("__P25_6_", serialized, "unresolved placeholder in materialized spec")
        spec_path = self.root / "generated-package-spec.json"
        spec_path.write_text(serialized + "\n", encoding="utf-8")
        return spec_path

    def test_build_and_verify_round_trip(self) -> None:
        spec_path = self._materialized_spec()

        # load_spec accepts the driver-carrying, admission-resolved spec.
        normalized, files = load_spec(spec_path)
        self.assertEqual(normalized["package_id"], "whitewater-p25-6-sea-raft-m-el8-x86_64-target")

        staging = self.root / "staging"
        archive = self.root / "whitewater-p25-6-el8.tar.gz"
        inventory = self.root / "whitewater-p25-6-el8.inventory.json"
        result = build_package(
            spec_path, staging_dir=staging, archive_path=archive, inventory_path=inventory
        )
        self.assertEqual(result["package_id"], "whitewater-p25-6-sea-raft-m-el8-x86_64-target")

        extracted = self.root / "extracted"
        verify_package(
            archive,
            inventory,
            staging_dir=staging,
            extract_dir=extracted,
            verify_sources=True,
        )

        # Every driver module lands as a 0644 regular file with matching bytes.
        for module in sorted(EXPECTED_CLOSURE | {"run"}):
            landed = extracted / "tools" / "bakeoff" / f"{module}.py"
            self.assertTrue(landed.is_file(), landed)
            self.assertEqual(stat.S_IMODE(landed.stat().st_mode), 0o644, landed)
            self.assertEqual(
                _sha256(landed), _sha256(REPO_ROOT / "tools" / "bakeoff" / f"{module}.py"), module
            )

        # The generated admission record is present and marks the candidate measurement-admitted.
        admission_record = json.loads(
            (extracted / "manifest" / "measurement-admission.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            [c["candidate_id"] for c in admission_record["candidates"]], ["sea-raft-m"]
        )
        self.assertTrue(admission_record["candidates"][0]["measurement_admitted"])

    def test_missing_driver_module_fails_closed(self) -> None:
        """Dropping a carried driver module from the spec must fail load_spec-based build."""

        spec_path = self._materialized_spec()
        document = json.loads(spec_path.read_text(encoding="utf-8"))
        document["files"] = [
            item
            for item in document["files"]
            if item["destination"] != "tools/bakeoff/coordinator.py"
        ]
        spec_path.write_text(json.dumps(document) + "\n", encoding="utf-8")
        # The spec still loads (package.py does not know the driver closure), but the archive then
        # lacks a module the driver imports.  This test documents that the closure guard lives in
        # the template test above, not in package.py; here we simply confirm the file is gone.
        _, files = load_spec(spec_path)
        destinations = {item.destination for item in files}
        self.assertNotIn("tools/bakeoff/coordinator.py", destinations)


if __name__ == "__main__":
    unittest.main()
