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
import check_neuflow_manifest as checker  # noqa: E402
from exclusion_contract import ExclusionReason  # noqa: E402
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
        or manifest["exclusion"]["reason_code"]
        != ExclusionReason.CHECKPOINT_LICENSE_TERMS_UNKNOWN.value
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
        if recorded["exclusion"]["reason_code"] != ExclusionReason.EXPORT_OR_OPERATOR_FAILURE.value:
            raise AssertionError("technical export failure did not acquire its typed exclusion reason")
        old_argv = sys.argv
        sys.argv = ["check_neuflow_manifest.py", str(destination)]
        try:
            if checker.main() != 0:
                raise AssertionError("NeuFlow checker rejected technical failure output")
        finally:
            sys.argv = old_argv


def _test_update_manifest_preserves_checkpoint_admission(manifest) -> None:
    """A passing export must still pass the NeuFlow-specific excluded-state gate."""

    if manifest["licenses"]["checkpoint"]["commercial_use_permitted"] != "unknown":
        raise AssertionError("regression fixture no longer has unknown checkpoint commercial terms")
    if manifest["licenses"]["checkpoint"]["redistribution_permitted"] != "unknown":
        raise AssertionError("regression fixture no longer has unknown checkpoint redistribution terms")

    updated = copy.deepcopy(manifest)
    observed = copy.deepcopy(updated["validation"]["observed"])
    for field in ("numerical_status", "admission_status", "admission_reason"):
        observed.pop(field, None)

    with tempfile.TemporaryDirectory(prefix="neuflow-update-admission-") as directory:
        root = Path(directory)
        destination = root / "manifest.json"
        output = root / "synthetic.onnx"
        output.write_bytes(b"synthetic validated export")
        output.chmod(0o644)

        exporter.update_manifest(
            destination,
            updated,
            output,
            observed,
            "macos-arm64",
        )
        recorded = load_manifest(destination)
        recorded_observed = recorded["validation"]["observed"]
        if recorded["status"] != "excluded" or recorded["candidate"]["role"] != "excluded":
            raise AssertionError("passing export promoted a checkpoint-admission exclusion")
        if recorded["exclusion"]["reason_code"] != ExclusionReason.CHECKPOINT_LICENSE_TERMS_UNKNOWN.value:
            raise AssertionError("exporter did not emit the typed checkpoint exclusion reason")
        if recorded["validation"]["status"] != "passed":
            raise AssertionError("checkpoint-admission exclusion lost numerical pass status")
        if recorded_observed.get("numerical_status") != "passed":
            raise AssertionError("exporter did not preserve numerical pass evidence")
        if any(field in recorded_observed for field in ("admission_status", "admission_reason")):
            raise AssertionError("exporter retained legacy observed admission fields")
        if any(note.startswith("admission_status=") for note in recorded["notes"]):
            raise AssertionError("exporter retained the legacy free-text admission sentinel")

        # Exercise the candidate-specific checker as well as the shared manifest gate. The
        # synthetic payload has the same contract as a real export, so only its expected hash
        # and size need to be substituted for this dependency-free regression fixture.
        digest = exporter.sha256_file(output)
        old_argv = sys.argv
        sys.argv = ["check_neuflow_manifest.py", str(destination)]
        try:
            with patch.object(checker, "EXPECTED_ARTIFACT_SHA256", digest), patch.object(
                checker, "EXPECTED_ARTIFACT_SIZE", output.stat().st_size
            ):
                if checker.main() != 0:
                    raise AssertionError("NeuFlow checker rejected exporter admission output")
        finally:
            sys.argv = old_argv


def _test_contiguous_onnx_inputs() -> None:
    class FakeTensor:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class FakeNumpy:
        def __init__(self):
            self.calls = []

        def ascontiguousarray(self, value):
            self.calls.append(value)
            return ("contiguous", value)

    original = object()
    numpy = FakeNumpy()
    converted = exporter._contiguous_numpy(FakeTensor(original), numpy)
    if numpy.calls != [original] or converted != ("contiguous", original):
        raise AssertionError("exporter did not normalize ONNX inputs through ascontiguousarray")


def _test_provider_selection_guards() -> None:
    """Provider qualification must fail on unavailable providers and silent fallback."""

    _expect_error(
        lambda: exporter._validate_provider_selection(
            "CUDAExecutionProvider", ["CPUExecutionProvider"]
        ),
        RuntimeError,
        "unavailable",
    )
    _expect_error(
        lambda: exporter._validate_provider_selection(
            "CUDAExecutionProvider",
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            ["CPUExecutionProvider"],
        ),
        RuntimeError,
        "select CUDAExecutionProvider first",
    )
    exporter._validate_provider_selection(
        "CUDAExecutionProvider",
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    exporter._validate_provider_selection(
        "CPUExecutionProvider", ["CPUExecutionProvider"], ["CPUExecutionProvider"]
    )
    _expect_error(
        lambda: checker._validate_provider_evidence(
            "linux-x86_64",
            {
                "provider_validation": {
                    "requested": "CUDAExecutionProvider",
                    "selected": ["CPUExecutionProvider"],
                    "passed": True,
                }
            },
        ),
        ArtifactError,
        "first-select CUDAExecutionProvider",
    )
    _expect_error(
        lambda: checker._validate_provider_evidence(
            "macos-arm64",
            {
                "provider_validation": {
                    "requested": "CUDAExecutionProvider",
                    "selected": ["CUDAExecutionProvider"],
                    "passed": True,
                }
            },
        ),
        ArtifactError,
        "CPUExecutionProvider",
    )


def _test_linux_cuda_manifest_path(manifest) -> None:
    """A Linux artifact may qualify CUDA, while the checked-in macOS row stays CPU-only."""

    linux = copy.deepcopy(manifest)
    observed = copy.deepcopy(linux["validation"]["observed"])
    observed["provider_validation"] = {
        "requested": "CUDAExecutionProvider",
        "available": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "selected": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "passed": True,
    }
    observed["environment"]["provider"] = "CUDAExecutionProvider"
    observed["environment"]["available_providers"] = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    observed["environment"]["selected_providers"] = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    with tempfile.TemporaryDirectory(prefix="neuflow-linux-cuda-qualification-") as directory:
        root = Path(directory)
        destination = root / "manifest.json"
        output = root / "neuflow-linux.onnx"
        output.write_bytes(b"synthetic Linux CUDA-qualified artifact")
        output.chmod(0o644)
        exporter.update_manifest(
            destination,
            linux,
            output,
            observed,
            "linux-x86_64",
        )
        old_argv = sys.argv
        sys.argv = ["check_neuflow_manifest.py", str(destination)]
        try:
            if checker.main() != 0:
                raise AssertionError("checker rejected an exact Linux CUDA evidence row")
        finally:
            sys.argv = old_argv


def main() -> int:
    manifest = load_manifest(MANIFEST_PATH)
    if manifest["status"] not in {"provenance_pinned_export_pending", "export_validated", "excluded"}:
        raise AssertionError(f"unexpected NeuFlow manifest status: {manifest['status']}")
    if manifest["status"] != "excluded":
        raise AssertionError("unknown checkpoint terms must leave NeuFlow admission explicitly excluded")
    if manifest["candidate"]["role"] != "excluded":
        raise AssertionError("checkpoint-license exclusion must mark candidate.role=excluded")
    if manifest.get("exclusion", {}).get("reason_code") != ExclusionReason.CHECKPOINT_LICENSE_TERMS_UNKNOWN.value:
        raise AssertionError("checkpoint-license exclusion lacks its typed reason")
    observed = manifest["validation"].get("observed")
    if manifest["validation"]["status"] != "passed" or not isinstance(observed, dict):
        raise AssertionError("checkpoint-license exclusion must preserve passed numerical status")
    if observed.get("numerical_status") != "passed":
        raise AssertionError("checkpoint-license exclusion must preserve numerical pass evidence")
    if any(field in observed for field in ("admission_status", "admission_reason")):
        raise AssertionError("checked-in manifest retains legacy observed admission fields")
    if any(note.startswith("admission_status=") for note in manifest.get("notes", [])):
        raise AssertionError("checked-in manifest retains the legacy free-text admission sentinel")
    if manifest["tensor_contract"]["spatial_dimensions"] != (
        "fixed_shape_only; exactly 432x768; no dynamic or other-shape support"
    ):
        raise AssertionError("NeuFlow fixed evaluation-shape contract is not explicit")
    if manifest["validation"]["shapes"] != {
        "dynamic": False,
        "example": [1, 2, 432, 768],
        "additional": [1, 2, 432, 768],
    }:
        raise AssertionError("NeuFlow manifest does not record the constrained fixed shape")
    observed_provider = manifest["validation"]["observed"].get("provider_validation")
    if observed_provider != {
        "requested": "CPUExecutionProvider",
        "available": ["AzureExecutionProvider", "CPUExecutionProvider"],
        "selected": ["CPUExecutionProvider"],
        "passed": True,
    }:
        raise AssertionError("NeuFlow provider qualification evidence changed")

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
    if source.count("_contiguous_numpy(a, np)") != 1 or source.count("_contiguous_numpy(b, np)") != 1:
        raise AssertionError("run_onnx does not make both ONNX inputs contiguous")
    if "def _validate_fixed_io" not in source or "requested ONNX Runtime provider is unavailable" not in source:
        raise AssertionError("exporter lacks fixed-IO/provider qualification hooks")
    if "CUDA is intended for a later" not in source:
        raise AssertionError("exporter does not distinguish later EL8/CUDA qualification")

    requirements = REQUIREMENTS_PATH.read_text(encoding="utf-8")
    for pin in ("torch==2.0.1", "torchvision==0.15.2", "onnx==1.14.1", "onnxruntime==1.16.3"):
        if pin not in requirements:
            raise AssertionError(f"missing pinned export dependency: {pin}")

    _test_provenance_invariants(manifest)
    _test_numerically_validated_failure_is_immutable(manifest)
    _test_update_manifest_preserves_checkpoint_admission(manifest)
    _test_contiguous_onnx_inputs()
    _test_provider_selection_guards()
    _test_linux_cuda_manifest_path(manifest)

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
