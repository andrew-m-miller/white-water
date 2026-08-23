#!/usr/bin/env python3
"""Dependency-light contract tests for the NeuFlow v2 exporter and export record."""

from __future__ import annotations

import ast
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

try:
    from artifact_workflow import ArtifactError, load_manifest, validate_artifact  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - package import path
    from .artifact_workflow import ArtifactError, load_manifest, validate_artifact  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "models" / "neuflow-v2.json"
EXPORTER_PATH = ROOT / "models" / "export_neuflow_v2.py"
REQUIREMENTS_PATH = ROOT / "models" / "requirements-neuflow-export.txt"
sys.path.insert(0, str(EXPORTER_PATH.parent))
import export_neuflow_v2 as exporter  # noqa: E402


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


def _expect_error(callback, error_type, text: str) -> None:
    try:
        callback()
    except error_type as exc:
        if text not in str(exc):
            raise AssertionError(f"expected {text!r} in error, got {exc!r}") from exc
    else:
        raise AssertionError(f"expected {error_type.__name__} containing {text!r}")


def _fake_git_run(status: str):
    def run(command, **_kwargs):
        if command[-2:] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout=exporter.EXPECTED_SOURCE_COMMIT + "\n")
        if "status" in command:
            return SimpleNamespace(stdout=status)
        raise AssertionError(f"unexpected git command: {command}")

    return run


def _test_provenance_invariants(manifest) -> None:
    with tempfile.TemporaryDirectory(prefix="neuflow-export-contract-") as directory:
        root = Path(directory)
        upstream = root / "upstream"
        (upstream / ".git").mkdir(parents=True)
        checkpoint_target = root / "checkpoint-target.pth"
        checkpoint_target.write_bytes(b"checkpoint")
        checkpoint_link = root / "checkpoint-link.pth"
        checkpoint_link.symlink_to(checkpoint_target)

        with patch.object(exporter.subprocess, "run", side_effect=_fake_git_run("")):
            _expect_error(
                lambda: exporter.verify_provenance(manifest, upstream, checkpoint_link),
                ArtifactError,
                "must not be a symlink",
            )

        bad_mode = root / "checkpoint-bad-mode.pth"
        bad_mode.write_bytes(b"checkpoint")
        bad_mode.chmod(0o600)
        with patch.object(exporter.subprocess, "run", side_effect=_fake_git_run("")):
            _expect_error(
                lambda: exporter.verify_provenance(manifest, upstream, bad_mode),
                ArtifactError,
                "expected 0644",
            )

        with patch.object(
            exporter.subprocess,
            "run",
            side_effect=_fake_git_run(" M NeuFlow/neuflow.py\n?? local-edit.txt\n"),
        ):
            _expect_error(
                lambda: exporter.verify_provenance(manifest, upstream, checkpoint_target),
                RuntimeError,
                "dirty",
            )


def _test_numerically_validated_failure_is_immutable(manifest) -> None:
    if (
        manifest["status"] != "excluded"
        or manifest["candidate"]["role"] != "excluded"
        or manifest["validation"]["observed"].get("numerical_status") != "passed"
    ):
        raise AssertionError("negative validated-failure test requires the current excluded admission record")
    validated_copy = copy.deepcopy(manifest)
    with tempfile.TemporaryDirectory(prefix="neuflow-failure-contract-") as directory:
        destination = Path(directory) / "manifest.json"
        _expect_error(
            lambda: exporter.record_failure(
                destination, validated_copy, "onnx_export", RuntimeError("synthetic failure")
            ),
            ArtifactError,
            "fresh pending manifest",
        )
        if destination.exists():
            raise AssertionError("record_failure wrote over a validated manifest")
    if validated_copy["status"] != "excluded":
        raise AssertionError("record_failure mutated the excluded manifest in memory")
    if validated_copy["export"]["sha256"] is None:
        raise AssertionError("validated export identity was cleared by a rejected failure")

    pending_copy = copy.deepcopy(manifest)
    pending_copy["status"] = "provenance_pinned_export_pending"
    pending_copy["candidate"]["role"] = "shipping-candidate"
    pending_copy["export"]["sha256"] = None
    pending_copy["export"]["size_bytes"] = None
    for entry in pending_copy["export"]["platform_artifacts"]:
        entry["sha256"] = None
        entry["size_bytes"] = None
    pending_copy["validation"]["status"] = "pending"
    pending_copy["validation"]["observed"] = None
    with tempfile.TemporaryDirectory(prefix="neuflow-pending-failure-") as directory:
        destination = Path(directory) / "manifest.json"
        exporter.record_failure(
            destination, pending_copy, "onnx_export", RuntimeError("synthetic failure")
        )
        recorded = load_manifest(destination)
        if recorded["status"] != "excluded" or recorded["validation"]["status"] != "failed":
            raise AssertionError("pending export failure was not recorded as excluded/failed")
        if recorded["validation"]["observed"]["failure_stage"] != "onnx_export":
            raise AssertionError("typed failure stage was not retained")
        if recorded["export"]["sha256"] is not None or recorded["export"]["size_bytes"] is not None:
            raise AssertionError("excluded pending export acquired an artifact identity")


def main() -> int:
    manifest = load_manifest(MANIFEST_PATH)
    if manifest["status"] not in {"provenance_pinned_export_pending", "export_validated", "excluded"}:
        raise AssertionError(f"unexpected NeuFlow manifest status: {manifest['status']}")
    if manifest["status"] != "excluded":
        raise AssertionError("unknown checkpoint terms must leave NeuFlow admission explicitly excluded")
    if manifest["candidate"]["role"] != "excluded":
        raise AssertionError("checkpoint-license exclusion must mark candidate.role=excluded")
    if "admission_status=excluded_checkpoint_license_terms_unknown" not in manifest.get("notes", []):
        raise AssertionError("checkpoint-license exclusion lacks an explicit admission note")
    observed = manifest["validation"].get("observed")
    if manifest["validation"]["status"] != "passed" or not isinstance(observed, dict):
        raise AssertionError("checkpoint-license exclusion must preserve passed numerical status")
    if observed.get("numerical_status") != "passed":
        raise AssertionError("checkpoint-license exclusion must preserve numerical pass evidence")

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
    validation_calls = _calls(tree, "_record_validation")
    if len(validation_calls) != 1:
        raise AssertionError(
            f"update_manifest must record validation exactly once, found {len(validation_calls)} calls"
        )
    if "iters_s16=1" not in source or "iters_s8=8" not in source:
        raise AssertionError("exporter does not freeze the official iteration counts")
    if "scaled_dot_product_attention" not in source or "torch.softmax" not in source:
        raise AssertionError("exporter does not document its portable SDPA lowering")

    requirements = REQUIREMENTS_PATH.read_text(encoding="utf-8")
    for pin in ("torch==2.0.1", "torchvision==0.15.2", "onnx==1.14.1", "onnxruntime==1.16.3"):
        if pin not in requirements:
            raise AssertionError(f"missing pinned export dependency: {pin}")

    _test_provenance_invariants(manifest)
    _test_numerically_validated_failure_is_immutable(manifest)

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
