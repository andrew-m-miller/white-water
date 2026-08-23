#!/usr/bin/env python3
"""Dependency-light tests for original-RAFT acquisition/export declarations."""

from __future__ import annotations

import hashlib
import copy
import json
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


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _expect_failure(callback) -> None:
    try:
        callback()
    except ArtifactError:
        return
    raise AssertionError("expected acquisition failure")


def main() -> int:
    manifest = load_manifest(MANIFEST)
    assert manifest["export"]["script"] == "models/export_raft.py"
    assert manifest["model"]["config"]["iters"] == 12
    assert manifest["tensor_contract"]["padding"] == {
        "multiple": 8,
        "policy": "caller-replication-crop",
    }
    assert manifest["status"] == "excluded"
    assert manifest["validation"]["observed"]["numerical_gates"] == "passed"

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
        pending["export"]["sha256"] = None
        pending["export"]["size_bytes"] = None
        for entry in pending["export"]["platform_artifacts"]:
            entry["sha256"] = None
            entry["size_bytes"] = None
        pending["validation"] = {
            "status": "pending",
            "observed": {
                "typed_status": "provenance_pinned_export_pending",
                "reason_type": "test_fixture",
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
        assert failed["validation"]["status"] == "failed"
        assert failed["validation"]["observed"]["stage"] == "provenance"
        assert failed["validation"]["observed"]["source_commit_verified"] is False
        assert failed["validation"]["observed"]["checkpoint_verified"] is False

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
