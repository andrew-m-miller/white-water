#!/usr/bin/env python3
"""Dependency-light tests for original-RAFT acquisition/export declarations."""

from __future__ import annotations

import argparse
import hashlib
import copy
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import warnings
import zipfile

from artifact_workflow import ArtifactError, load_manifest, sha256_file  # type: ignore
import export_raft  # type: ignore
import fetch_raft_checkpoint  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models" / "raft-original.json"


class _FakeTensorInfo:
    def __init__(self, name: str, tensor_type: str, shape: list[object]):
        self.name = name
        self.type = tensor_type
        self.shape = shape


class _FakeOrtSession:
    def __init__(self, inputs: list[_FakeTensorInfo], outputs: list[_FakeTensorInfo]):
        self._inputs = inputs
        self._outputs = outputs

    def get_inputs(self) -> list[_FakeTensorInfo]:
        return self._inputs

    def get_outputs(self) -> list[_FakeTensorInfo]:
        return self._outputs


class _FakeDownloadResponse:
    def __init__(self, payload: bytes):
        self._stream = io.BytesIO(payload)

    def __enter__(self) -> io.BytesIO:
        return self._stream

    def __exit__(self, *args: object) -> None:
        self._stream.close()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _expect_failure(callback) -> None:
    try:
        callback()
    except ArtifactError:
        return
    raise AssertionError("expected acquisition failure")


def _check_advertised_io_contract() -> None:
    """Reject symbolic channel dimensions while allowing dynamic batch/spatial axes."""

    def session(input_channels: object = 3, output_channels: object = 2) -> _FakeOrtSession:
        return _FakeOrtSession(
            [
                _FakeTensorInfo("image1", "tensor(float)", [None, input_channels, "height", "width"]),
                _FakeTensorInfo("image2", "tensor(float)", [None, 3, "height", "width"]),
            ],
            [_FakeTensorInfo("flow", "tensor(float)", [None, output_channels, "height", "width"])],
        )

    advertised = export_raft._validate_advertised_io(session(), {})
    assert advertised["inputs"][0]["shape"] == [None, 3, "height", "width"]
    assert advertised["outputs"][0]["shape"] == [None, 2, "height", "width"]

    for channels in ("channels", None, 4):
        _expect_runtime_failure(
            lambda channels=channels: export_raft._validate_advertised_io(
                session(input_channels=channels), {}
            )
        )
    for channels in ("channels", None, 3):
        _expect_runtime_failure(
            lambda channels=channels: export_raft._validate_advertised_io(
                session(output_channels=channels), {}
            )
        )


def _check_download_archive_semantics() -> None:
    """Exercise explicit archive-copy safety without contacting the pinned URL."""

    payload = b"deterministic-raft-checkpoint-fixture"
    with tempfile.TemporaryDirectory(prefix="whitewater-raft-download-") as directory:
        root = Path(directory)
        archive_source = root / "source.zip"
        with zipfile.ZipFile(archive_source, "w", compression=zipfile.ZIP_STORED) as zipped:
            zipped.writestr(fetch_raft_checkpoint.MEMBER_NAME, payload)
        archive_bytes = archive_source.read_bytes()
        archive_sha256 = _sha256(archive_bytes)
        retained = root / "retained.zip"
        calls: list[str] = []
        previous_urlopen = fetch_raft_checkpoint.urllib.request.urlopen

        def fake_urlopen(url: str):
            calls.append(url)
            return _FakeDownloadResponse(archive_bytes)

        fetch_raft_checkpoint.urllib.request.urlopen = fake_urlopen
        try:
            fetch_raft_checkpoint.download_archive(retained, expected_sha256=archive_sha256)
            assert retained.read_bytes() == archive_bytes
            assert stat.S_IMODE(retained.stat().st_mode) == 0o644
            first_call_count = len(calls)

            # An exact, verified retained copy is idempotent and does not need another network
            # request, which is useful when an operator reruns acquisition on an airgapped box.
            fetch_raft_checkpoint.download_archive(retained, expected_sha256=archive_sha256)
            assert len(calls) == first_call_count

            retained.write_bytes(b"do-not-clobber")
            retained.chmod(0o644)
            before = retained.read_bytes()
            _expect_failure(
                lambda: fetch_raft_checkpoint.download_archive(
                    retained, expected_sha256=archive_sha256
                )
            )
            assert retained.read_bytes() == before
            assert len(calls) == first_call_count
        finally:
            fetch_raft_checkpoint.urllib.request.urlopen = previous_urlopen


def _check_default_download_is_temporary() -> None:
    """Ensure the default --download flow never leaves cwd/models.zip behind."""

    with tempfile.TemporaryDirectory(prefix="whitewater-raft-download-main-") as directory:
        root = Path(directory)
        output = root / "raft-things.pth"
        seen: dict[str, Path] = {}
        previous_parse_args = fetch_raft_checkpoint.parse_args
        previous_download_archive = fetch_raft_checkpoint.download_archive
        previous_extract = fetch_raft_checkpoint.extract_verified_member
        previous_cwd = Path.cwd()

        def fake_download(path: Path) -> None:
            seen["archive"] = path
            path.write_bytes(b"archive")
            path.chmod(0o644)

        def fake_extract(archive: Path, destination: Path) -> None:
            seen["extracted_archive"] = archive
            destination.write_bytes(b"checkpoint")
            destination.chmod(0o644)

        fetch_raft_checkpoint.parse_args = lambda: argparse.Namespace(
            archive=None,
            archive_copy=None,
            download=True,
            output=output,
        )
        fetch_raft_checkpoint.download_archive = fake_download
        fetch_raft_checkpoint.extract_verified_member = fake_extract
        os.chdir(root)
        try:
            assert fetch_raft_checkpoint.main() == 0
        finally:
            os.chdir(previous_cwd)
            fetch_raft_checkpoint.parse_args = previous_parse_args
            fetch_raft_checkpoint.download_archive = previous_download_archive
            fetch_raft_checkpoint.extract_verified_member = previous_extract

        temporary_archive = seen["archive"]
        assert temporary_archive == seen["extracted_archive"]
        assert temporary_archive != root / "models.zip"
        assert not temporary_archive.exists()
        assert not (root / "models.zip").exists()
        assert output.read_bytes() == b"checkpoint"


def _check_success_manifest_update() -> None:
    """Ensure the numerical-success exporter path writes the shared exclusion contract."""

    with tempfile.TemporaryDirectory(prefix="whitewater-raft-success-record-") as directory:
        root = Path(directory)
        manifest_path = root / "generated.json"
        output = root / "generated.onnx"
        generated = copy.deepcopy(load_manifest(MANIFEST))
        manifest_path.write_text(json.dumps(generated, indent=2) + "\n", encoding="utf-8")
        manifest_path.chmod(0o644)
        output.write_bytes(b"deterministic-export-fixture")
        output.chmod(0o644)

        observed = copy.deepcopy(generated["validation"]["observed"])
        metrics = observed.pop("metrics")
        observed.update(
            {
                "identity_median_epe": metrics["identity_median_epe"],
                "forward_median": metrics["forward_median"],
                "reverse_median": metrics["reverse_median"],
                "onnx_pytorch_mean_abs": metrics["onnx_pytorch_mean_abs"],
                "onnx_pytorch_p99_abs": metrics["onnx_pytorch_p99_abs"],
                "onnx_pytorch_p999_abs": metrics["onnx_pytorch_p999_abs"],
                "onnx_pytorch_max_abs": metrics["onnx_pytorch_max_abs"],
                "second_dynamic_shape": generated["validation"]["shapes"]["additional"],
            }
        )
        export_raft.update_manifest(
            manifest_path,
            generated,
            output,
            observed,
            "macos-arm64",
        )
        recorded = load_manifest(manifest_path)
        assert recorded["status"] == "excluded"
        assert recorded["candidate"]["role"] == "validation-baseline"
        assert recorded["exclusion"]["reason_code"] == "checkpoint_license_terms_unknown"
        assert recorded["validation"]["status"] == "passed"
        assert "reason_type" not in recorded["validation"]["observed"]


def main() -> int:
    manifest = load_manifest(MANIFEST)
    assert manifest["export"]["script"] == "models/export_raft.py"
    assert manifest["model"]["config"]["iters"] == 12
    assert manifest["tensor_contract"]["padding"] == {
        "multiple": 8,
        "policy": "caller-replication-crop",
    }
    assert manifest["status"] == "excluded"
    assert manifest["candidate"]["role"] == "validation-baseline"
    assert manifest["exclusion"]["reason_code"] == "checkpoint_license_terms_unknown"
    assert manifest["validation"]["status"] == "passed"
    assert manifest["validation"]["observed"]["numerical_gates"] == "passed"
    assert "reason_type" not in manifest["validation"]["observed"]

    _check_advertised_io_contract()
    _check_download_archive_semantics()
    _check_default_download_is_temporary()
    _check_success_manifest_update()

    payload = b"deterministic-raft-checkpoint-fixture"
    with tempfile.TemporaryDirectory(prefix="whitewater-raft-acquisition-") as directory:
        root = Path(directory)
        archive = root / "models.zip"
        output = root / "raft-things.pth"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zipped:
            zipped.writestr(fetch_raft_checkpoint.MEMBER_NAME, payload)
        archive.chmod(0o644)
        archive_hash = sha256_file(archive)
        archive.chmod(0o600)
        _expect_failure(lambda: fetch_raft_checkpoint.verify_archive(archive, expected_sha256=archive_hash))
        archive.chmod(0o644)
        fetch_raft_checkpoint.extract_verified_member(
            archive,
            output,
            expected_size=len(payload),
            expected_sha256=_sha256(payload),
            expected_archive_sha256=archive_hash,
        )
        assert output.read_bytes() == payload
        assert stat.S_IMODE(output.stat().st_mode) == 0o644
        # Extraction is idempotent and refuses to replace an existing different file.
        fetch_raft_checkpoint.extract_verified_member(
            archive,
            output,
            expected_size=len(payload),
            expected_sha256=_sha256(payload),
            expected_archive_sha256=archive_hash,
        )
        symlink_target = root / "target.pth"
        symlink_target.write_bytes(payload)
        symlink_target.chmod(0o644)
        symlink = root / "symlink.pth"
        symlink.symlink_to(symlink_target)
        _expect_failure(
            lambda: fetch_raft_checkpoint.extract_verified_member(
                archive,
                symlink,
                expected_size=len(payload),
                expected_sha256=_sha256(payload),
                expected_archive_sha256=archive_hash,
            )
        )

        duplicate = root / "duplicate.zip"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(duplicate, "w", compression=zipfile.ZIP_STORED) as zipped:
                zipped.writestr(fetch_raft_checkpoint.MEMBER_NAME, payload)
                zipped.writestr(fetch_raft_checkpoint.MEMBER_NAME, payload)
        duplicate.chmod(0o644)
        _expect_failure(
            lambda: fetch_raft_checkpoint.extract_verified_member(
                duplicate,
                root / "duplicate.pth",
                expected_size=len(payload),
                expected_sha256=_sha256(payload),
                expected_archive_sha256=sha256_file(duplicate),
            )
        )

        source = root / "source"
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        tracked = source / "tracked.py"
        tracked.write_text("print('pinned')\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(source), "add", "tracked.py"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "-c",
                "user.name=White Water Test",
                "-c",
                "user.email=whitewater@example.invalid",
                "commit",
                "-q",
                "-m",
                "fixture",
            ],
            check=True,
        )
        assert export_raft.checked_out_commit(source)
        (source / "untracked.py").write_text("untracked\n", encoding="utf-8")
        _expect_runtime_failure(lambda: export_raft.checked_out_commit(source))

        pending = copy.deepcopy(manifest)
        pending["status"] = "provenance_pinned_export_pending"
        pending.pop("exclusion")
        pending["export"]["sha256"] = None
        pending["export"]["size_bytes"] = None
        for entry in pending["export"]["platform_artifacts"]:
            entry["sha256"] = None
            entry["size_bytes"] = None
        pending["validation"] = {
            "status": "pending",
            "observed": {
                "fixture": "provenance_pinned_export_pending",
            },
        }
        pending_path = root / "pending.json"
        pending_path.write_text(json.dumps(pending, indent=2) + "\n", encoding="utf-8")
        pending_path.chmod(0o644)
        provenance_failure = subprocess.run(
            [
                sys.executable,
                str(ROOT / "models" / "export_raft.py"),
                "--upstream",
                str(source),
                "--checkpoint",
                str(root / "missing-checkpoint.pth"),
                "--manifest",
                str(pending_path),
                "--verify-provenance-only",
                "--update-manifest",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert provenance_failure.returncode != 0
        failed = load_manifest(pending_path)
        assert failed["status"] == "excluded"
        assert failed["candidate"]["role"] == "validation-baseline"
        assert failed["exclusion"]["reason_code"] == "export_or_operator_failure"
        assert failed["validation"]["status"] == "failed"
        assert failed["validation"]["observed"]["stage"] == "provenance"
        assert failed["validation"]["observed"]["source_commit_verified"] is False
        assert failed["validation"]["observed"]["checkpoint_verified"] is False
        assert "reason_type" not in failed["validation"]["observed"]

        (source / "untracked.py").unlink()
        tracked.write_text("dirty\n", encoding="utf-8")
        _expect_runtime_failure(lambda: export_raft.checked_out_commit(source))

        validated_rerun_path = root / "validated-rerun.json"
        validated_rerun_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        validated_rerun_path.chmod(0o644)
        validated_rerun_before = validated_rerun_path.read_bytes()
        later_failure = subprocess.run(
            [
                sys.executable,
                str(ROOT / "models" / "export_raft.py"),
                "--upstream",
                str(source),
                "--checkpoint",
                str(root / "missing-checkpoint.pth"),
                "--manifest",
                str(validated_rerun_path),
                "--verify-provenance-only",
                "--update-manifest",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert later_failure.returncode != 0
        assert validated_rerun_path.read_bytes() == validated_rerun_before
        preserved_rerun = load_manifest(validated_rerun_path)
        assert preserved_rerun["export"]["sha256"] == manifest["export"]["sha256"]
        assert preserved_rerun["validation"]["observed"]["numerical_gates"] == "passed"

    with tempfile.TemporaryDirectory(prefix="whitewater-raft-failure-guard-") as directory:
        validated_path = Path(directory) / "validated.json"
        validated_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        validated_path.chmod(0o644)
        validated_before = validated_path.read_bytes()
        _expect_failure(
            lambda: export_raft.record_failure(
                validated_path,
                load_manifest(validated_path),
                "simulated later operator failure",
                stage="operator_validation",
                source_verified=True,
                checkpoint_verified=True,
            )
        )
        assert validated_path.read_bytes() == validated_before
        preserved = load_manifest(validated_path)
        assert preserved["status"] == "excluded"
        assert preserved["export"]["sha256"] == manifest["export"]["sha256"]
        assert preserved["validation"]["observed"]["numerical_gates"] == "passed"

    requirements = (ROOT / "models" / "requirements-raft-export.txt").read_text(encoding="utf-8")
    for requirement in (
        "torch==2.2.0",
        "numpy==1.26.4",
        "scipy==1.12.0",
        "onnx==1.15.0",
        "onnxruntime==1.29.0",
    ):
        assert requirement in requirements
    print("original RAFT acquisition/export declarations: PASS")
    return 0


def _expect_runtime_failure(callback) -> None:
    try:
        callback()
    except RuntimeError:
        return
    raise AssertionError("expected dirty source checkout failure")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, ArtifactError, OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"raft_export_tests.py: error: {exc}")
        raise SystemExit(1)
