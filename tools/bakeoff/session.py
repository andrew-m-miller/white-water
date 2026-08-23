#!/usr/bin/env python3
"""Dependency-free orchestration for one resumable Phase 2.5 bake-off session.

This module composes matrix planning, resume state, lifecycle coordination, report assembly, and
publication.  It deliberately does not know how frames, models, or metrics are produced: the
caller supplies one executor callback for each planned cell.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .coordinator import Executor, RunCoordinator
from .matrix import MatrixPlan, build_matrix
from .reporting import assemble_report, write_report_pair
from .resume import create_state, load_state
from .validator import canonical_sha256


class SessionFailure(ValueError):
    """Stable, reportable session-input failure."""

    def __init__(self, kind: str, message: str):
        self.kind = kind
        self.reason = kind
        self.failure_type = "session_failure"
        self.message = message
        super().__init__(f"{kind}: {message}")


def _fail(kind: str, message: str) -> None:
    raise SessionFailure(kind, message)


def _canonical_hash(value: Any, path: str) -> str:
    try:
        return canonical_sha256(value)
    except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError) as exc:
        raise SessionFailure("identity_json", f"{path} is not a finite JSON value: {exc}") from exc


def _metadata_identity(report_metadata: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(report_metadata, Mapping):
        _fail("metadata_shape", "report_metadata must be an object")
    for field in ("runner", "hardware"):
        if field not in report_metadata or not isinstance(report_metadata[field], Mapping):
            _fail("metadata_shape", f"report_metadata.{field} must be an object")
    metadata_hash = _canonical_hash(report_metadata, "report_metadata")
    # Keep the material environment records visible in state identity while also binding every
    # supplied metadata field (timestamps, report id, warnings, and any future frozen metadata).
    return {
        "runner": dict(report_metadata["runner"]),
        "hardware": dict(report_metadata["hardware"]),
        "report_metadata_sha256": metadata_hash,
    }


def resume_identity(
    protocol: Mapping[str, Any],
    corpus: Mapping[str, Any],
    report_schema: Mapping[str, Any],
    corpus_schema: Mapping[str, Any],
    candidate_entries: Any,
    plan: MatrixPlan,
    profile: str,
    environment: str,
    report_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the strict deterministic identity bound to one planned session."""

    if not isinstance(report_metadata, Mapping):
        _fail("metadata_shape", "report_metadata must be an object")
    if report_metadata.get("profile") != profile:
        _fail("metadata_profile", "report_metadata.profile must match the explicit profile")
    if report_metadata.get("environment") != environment:
        _fail("metadata_environment", "report_metadata.environment must match the explicit environment")
    metadata_identity = _metadata_identity(report_metadata)
    return {
        "protocol_sha256": _canonical_hash(protocol, "protocol"),
        "corpus_sha256": _canonical_hash(corpus, "corpus"),
        "report_schema_sha256": _canonical_hash(report_schema, "report_schema"),
        "corpus_schema_sha256": _canonical_hash(corpus_schema, "corpus_schema"),
        "candidate_entries_sha256": _canonical_hash(candidate_entries, "candidate_entries"),
        "matrix_sha256": plan.matrix_sha256,
        "profile": profile,
        "environment": environment,
        **metadata_identity,
    }


def _ensure_state(
    state_path: Path,
    identity: Mapping[str, Any],
    plan: MatrixPlan,
) -> None:
    """Create absent state or load an existing state through resume's path checks."""

    try:
        state_path.lstat()
    except FileNotFoundError:
        create_state(state_path, identity, plan)
        return
    except OSError as exc:
        raise SessionFailure("state_path", f"cannot inspect resume state {state_path}: {exc}") from exc
    # load_state rejects symlinks, directories, bad modes, malformed state, plan changes, and
    # identity changes with the existing typed ResumeFailure vocabulary.
    load_state(state_path, identity, plan)


def run_session(
    protocol: Mapping[str, Any],
    corpus: Mapping[str, Any],
    report_schema: Mapping[str, Any],
    corpus_schema: Mapping[str, Any],
    candidate_entries: Any,
    selections: Mapping[str, Any],
    profile: str,
    environment: str,
    report_metadata: Mapping[str, Any],
    state_path: Path | str,
    json_path: Path | str,
    csv_path: Path | str,
    executor: Executor,
    *,
    replace: bool = False,
) -> dict[str, Any]:
    """Run, assemble, and publish one complete validated bake-off report.

    ``replace`` is exposed only as an explicit opt-in; the default preserves report publication's
    no-clobber contract.  Executor exceptions propagate unchanged and prevent report assembly.
    """

    plan = build_matrix(
        protocol,
        corpus,
        candidate_entries,
        selections,
        profile,
        environment,
    )
    identity = resume_identity(
        protocol,
        corpus,
        report_schema,
        corpus_schema,
        candidate_entries,
        plan,
        profile,
        environment,
        report_metadata,
    )
    state_destination = Path(state_path)
    _ensure_state(state_destination, identity, plan)

    records = RunCoordinator(
        state_destination,
        identity,
        plan,
        executor,
    ).run()
    report = assemble_report(
        protocol,
        corpus,
        report_schema,
        corpus_schema,
        report_metadata,
        candidate_entries,
        plan,
        records,
    )
    write_report_pair(
        json_path,
        csv_path,
        report,
        protocol,
        report_schema,
        corpus,
        corpus_schema,
        replace=replace,
    )
    return report


__all__ = ["SessionFailure", "resume_identity", "run_session"]
