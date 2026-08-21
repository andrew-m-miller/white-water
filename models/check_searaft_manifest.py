#!/usr/bin/env python3
"""Dependency-free structural gate for the Phase 0B SEA-RAFT manifest."""

import json
import hashlib
from pathlib import Path
import re
import sys


manifest_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("sea-raft-m.json")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
hex64 = re.compile(r"[0-9a-f]{64}").fullmatch
hex40 = re.compile(r"[0-9a-f]{40}").fullmatch

assert manifest["schema_version"] == 1
assert hex40(manifest["upstream"]["commit"])
assert hex40(manifest["checkpoint"]["revision"])
assert hex64(manifest["checkpoint"]["sha256"])
assert str(manifest["checkpoint"]["revision"]) in manifest["checkpoint"]["url"]
assert "/main/" not in manifest["checkpoint"]["url"]
assert manifest["checkpoint"]["size_bytes"] > 0
assert manifest["tensor_contract"]["output"]["direction"] == "image1_to_image2"
assert manifest["tensor_contract"]["padding"]["multiple"] == 8
assert manifest["tensor_contract"]["iterations"] == "4_baked_into_graph"
assert manifest["model"]["config"]["scale"] == -1
assert manifest["tensor_contract"]["upstream_custom_py_input_scale"] == 0.5
assert manifest["tensor_contract"]["exported_forward_input_scale"] == 1.0
assert all(value % 8 == 0 for value in manifest["validation"]["second_dynamic_shape"][2:])

artifact_hash = manifest["export"]["sha256"]
if manifest["status"] == "provenance_pinned_export_pending":
    assert artifact_hash is None
    assert manifest["validation"]["observed"] is None
else:
    assert manifest["status"] in {"host_probe_pending", "host_probe_cpu_cuda_passed"}
    assert hex64(artifact_hash)
    assert manifest["export"]["size_bytes"] > 0
    assert manifest["validation"]["observed"] is not None
    artifact_path = manifest_path.with_name(manifest["export"]["artifact"])
    if artifact_path.exists():
        assert artifact_path.stat().st_size == manifest["export"]["size_bytes"]
        digest = hashlib.sha256()
        with artifact_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        assert digest.hexdigest() == artifact_hash

print("SEA-RAFT M manifest structure is valid")
