#!/usr/bin/env python3
"""Dependency-free tests for WAFT export gates and failure-safe manifest updates."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from artifact_workflow import load_manifest
from export_waft import (  # type: ignore  # pylint: disable=wrong-import-position
    BlockerCode,
    TechnicalBlocker,
    gate_onnx_graph,
    load_pinned_config,
    record_failure,
    _platform_id,
    validate_provider_device,
    verify_provider_selection,
    verify_provenance,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models" / "waft-twins-artifact.json"


class Node:
    def __init__(self, op_type: str, domain: str = "") -> None:
        self.op_type = op_type
        self.domain = domain


class Opset:
    def __init__(self, domain: str = "", version: int = 17) -> None:
        self.domain = domain
        self.version = version


class Graph:
    def __init__(self, nodes: list[Node], opsets: list[Opset] | None = None) -> None:
        self.node = nodes
        self.opset_import = opsets or [Opset()]


class Model:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph


def expect_blocker(function, code: str) -> None:
    try:
        function()
    except TechnicalBlocker as exc:
        assert exc.code == code, (exc.code, code)
    else:
        raise AssertionError(f"expected technical blocker {code}")


def main() -> int:
    # The gate accepts standard-domain nodes and records the exact operator/domain evidence.
    observed = gate_onnx_graph(Model(Graph([Node("Add"), Node("Mul")])), expected_opset=17)
    assert observed["version"] == "onnx-standard-domain-v1"
    assert observed["foreign_domains"] == []
    assert observed["operators"] == ["Add", "Mul"]

    expect_blocker(
        lambda: gate_onnx_graph(Model(Graph([Node("GridSample", "com.microsoft")])), expected_opset=17),
        BlockerCode.OPERATOR_DOMAIN,
    )
    expect_blocker(
        lambda: gate_onnx_graph(Model(Graph([Node("Add")], [Opset(version=18)])), expected_opset=17),
        BlockerCode.OPERATOR_DOMAIN,
    )
    expect_blocker(
        lambda: validate_provider_device("cpu", "CUDAExecutionProvider"),
        BlockerCode.CONFIG,
    )
    expect_blocker(
        lambda: verify_provider_selection("CUDAExecutionProvider", ["CPUExecutionProvider"]),
        BlockerCode.DEPENDENCY,
    )
    expect_blocker(lambda: _platform_id("not-a-real-platform"), BlockerCode.PLATFORM_IDENTITY)

    # A missing local checkout is typed and does not require any ML dependency or network.
    manifest = load_manifest(MANIFEST)
    expect_blocker(
        lambda: verify_provenance(manifest, Path("/no/such/waft-checkout"), Path("/no/such/checkpoint.pth")),
        BlockerCode.MISSING_INPUT,
    )
    expect_blocker(
        lambda: load_pinned_config(manifest, ROOT, Path("/tmp/modified-waft-config.json")),
        BlockerCode.CONFIG,
    )
    with tempfile.TemporaryDirectory(prefix="whitewater-waft-config-") as config_root:
        config_checkout = Path(config_root)
        config_path = config_checkout / "config/a2/twins/chairs-things.json"
        config_path.parent.mkdir(parents=True)
        config_path.write_text('{"feature_encoder":"twins","iterative_module":"vits","iters":5}\n', encoding="utf-8")
        expect_blocker(
            lambda: load_pinned_config(manifest, config_checkout, None),
            BlockerCode.CONFIG,
        )
    with tempfile.TemporaryDirectory(prefix="whitewater-waft-dirty-") as dirty_root:
        dirty_checkout = Path(dirty_root)
        subprocess.run(["git", "init", "-q", str(dirty_checkout)], check=True)
        (dirty_checkout / "untracked.txt").write_text("dirty\n", encoding="utf-8")
        from export_waft import _require_clean_worktree  # type: ignore

        expect_blocker(
            lambda: _require_clean_worktree(dirty_checkout),
            BlockerCode.SOURCE_DIRTY,
        )

    # Failure recording is atomic, preserves status=excluded and does not invent artifact bytes.
    with tempfile.TemporaryDirectory(prefix="whitewater-waft-export-tests-") as temporary:
        temporary_path = Path(temporary)
        manifest_path = temporary_path / "manifest.json"
        shutil.copy2(MANIFEST, manifest_path)
        blocker = TechnicalBlocker(
            BlockerCode.DEPENDENCY,
            "dependencies",
            "test blocker",
            {"missing": "torch"},
        )
        assert record_failure(manifest_path, copy.deepcopy(manifest), blocker) is True
        failed = load_manifest(manifest_path)
        assert failed["status"] == "excluded"
        assert failed["export"]["sha256"] is None
        assert failed["export"]["size_bytes"] is None
        assert failed["validation"]["status"] == "pending"
        assert failed["validation"]["observed"]["technical_blocker"]["code"] == BlockerCode.DEPENDENCY

    # An ONNX Runtime inference failure (e.g. the dynamic-shape broadcast mismatch) is now a typed
    # blocker, so --record-failure can capture it instead of the run dying on a raw traceback.
    with tempfile.TemporaryDirectory(prefix="whitewater-waft-export-tests-ort-") as temporary:
        manifest_path = Path(temporary) / "manifest.json"
        shutil.copy2(MANIFEST, manifest_path)
        ort_blocker = TechnicalBlocker(
            BlockerCode.ONNX_RUNTIME,
            "CPUExecutionProvider_runtime_validation",
            "ONNX Runtime raised while running the exported graph",
            {"exception": "RuntimeException", "input_shape": [1, 3, 160, 256]},
        )
        assert record_failure(manifest_path, copy.deepcopy(manifest), ort_blocker) is True
        recorded = load_manifest(manifest_path)
        assert recorded["validation"]["status"] == "pending"
        tb = recorded["validation"]["observed"]["technical_blocker"]
        assert tb["code"] == BlockerCode.ONNX_RUNTIME
        assert tb["stage"] == "CPUExecutionProvider_runtime_validation"

    print("WAFT export gates and failure-safe manifest tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
