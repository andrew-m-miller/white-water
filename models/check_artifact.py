#!/usr/bin/env python3
"""Short compatibility entry point for the candidate-neutral artifact checker."""

from __future__ import annotations

try:
    from check_artifact_manifest import main  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - imported as models.check_artifact
    from .check_artifact_manifest import main  # type: ignore


if __name__ == "__main__":
    raise SystemExit(main())

