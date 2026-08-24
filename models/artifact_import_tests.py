#!/usr/bin/env python3
"""Regression gate for artifact-workflow imports and outside-root CLI use."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
POSITIVE = ROOT / "models" / "fixtures" / "positive" / "artifact-v1.json"


def _assert_import_does_not_mutate_sys_path(module_name: str) -> None:
    before = list(sys.path)
    module = importlib.import_module(module_name)
    if sys.path != before:
        raise AssertionError(f"{module_name} import changed sys.path")
    importlib.reload(module)
    if sys.path != before:
        raise AssertionError(f"{module_name} reload changed sys.path")


def _assert_private_validator_spec_import_does_not_mutate_sys_path() -> None:
    validator_path = ROOT / "tools" / "bakeoff" / "validator.py"
    spec = importlib.util.spec_from_file_location(
        "_whitewater_private_validator_regression", validator_path
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not create validator spec for {validator_path}")
    before = list(sys.path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if sys.path != before:
        raise AssertionError("private validator spec import changed sys.path")
    if module._expected_analysis_dimensions(3, 3, 1.0, 0.000008) != (2, 3):
        raise AssertionError("private validator spec import loaded the wrong geometry helper")


def _run_cli(script: Path, *arguments: str, cwd: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(script), *arguments],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"{script.name} failed outside repo root ({result.returncode}): "
            f"{result.stdout}{result.stderr}"
        )


def main() -> int:
    # Running this file by absolute path puts models/ on sys.path, so this covers the
    # standalone imports used by the checker and exporter scripts.
    _assert_import_does_not_mutate_sys_path("artifact_workflow")
    _assert_private_validator_spec_import_does_not_mutate_sys_path()

    package_environment = os.environ.copy()
    package_environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT), package_environment.get("PYTHONPATH", "")) if part
    )
    package_code = """
import importlib
import sys
before = list(sys.path)
module = importlib.import_module('models.artifact_workflow')
assert sys.path == before, 'package import changed sys.path'
importlib.reload(module)
assert sys.path == before, 'package reload changed sys.path'
"""
    package_result = subprocess.run(
        [sys.executable, "-c", package_code],
        cwd=ROOT.parent,
        env=package_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if package_result.returncode != 0:
        raise AssertionError(
            f"models.artifact_workflow import failed: "
            f"{package_result.stdout}{package_result.stderr}"
        )

    with tempfile.TemporaryDirectory(prefix="whitewater-artifact-import-") as temporary:
        outside_root = Path(temporary)
        _run_cli(
            ROOT / "models" / "check_artifact.py",
            str(POSITIVE),
            "--no-protocol",
            cwd=outside_root,
        )
        _run_cli(
            ROOT / "models" / "check_searaft_manifest.py",
            str(ROOT / "models" / "sea-raft-m.json"),
            cwd=outside_root,
        )
        _run_cli(
            ROOT / "models" / "check_raft_manifest.py",
            str(ROOT / "models" / "raft-original.json"),
            cwd=outside_root,
        )

    print("artifact workflow import and outside-root CLI gates: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, subprocess.SubprocessError) as exc:
        print(f"artifact_import_tests.py: error: {exc}", file=sys.stderr)
        raise SystemExit(1)
