#!/usr/bin/env python3
"""Dependency-light contract tests for the NeuFlow v2 exporter and export record."""

from __future__ import annotations

import ast
from pathlib import Path

try:
    from artifact_workflow import load_manifest, validate_artifact  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - package import path
    from .artifact_workflow import load_manifest, validate_artifact  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "models" / "neuflow-v2.json"
EXPORTER_PATH = ROOT / "models" / "export_neuflow_v2.py"
REQUIREMENTS_PATH = ROOT / "models" / "requirements-neuflow-export.txt"


def _calls(tree: ast.AST, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == name)
        )
    ]


def _keyword_constant(call: ast.Call, name: str):
    for keyword in call.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant):
            return keyword.value.value
    return None


def main() -> int:
    manifest = load_manifest(MANIFEST_PATH)
    if manifest["status"] not in {"provenance_pinned_export_pending", "export_validated", "excluded"}:
        raise AssertionError(f"unexpected NeuFlow manifest status: {manifest['status']}")

    source = EXPORTER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(EXPORTER_PATH))
    init_calls = _calls(tree, "init_bhwd")
    if not any(_keyword_constant(call, "amp") is False for call in init_calls):
        raise AssertionError("exporter does not make float32 init_bhwd state explicit")
    export_calls = _calls(tree, "export")
    if not export_calls or any(
        keyword.arg == "dynamic_axes" for call in export_calls for keyword in call.keywords
    ):
        raise AssertionError("fixed-shape exporter must not advertise dynamic axes")
    if "iters_s16=1" not in source or "iters_s8=8" not in source:
        raise AssertionError("exporter does not freeze the official iteration counts")
    if "scaled_dot_product_attention" not in source or "torch.softmax" not in source:
        raise AssertionError("exporter does not document its portable SDPA lowering")

    requirements = REQUIREMENTS_PATH.read_text(encoding="utf-8")
    for pin in ("torch==2.0.1", "torchvision==0.15.2", "onnx==1.14.1", "onnxruntime==1.16.3"):
        if pin not in requirements:
            raise AssertionError(f"missing pinned export dependency: {pin}")

    artifact_path = MANIFEST_PATH.parent / manifest["export"]["artifact"]
    if artifact_path.exists():
        validate_artifact(manifest, MANIFEST_PATH, artifact_path)
        artifact_note = "artifact hash/size/mode validated"
    else:
        artifact_note = "large artifact intentionally absent from source checkout"
    print(f"NeuFlow v2 exporter contract: PASS ({artifact_note})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
