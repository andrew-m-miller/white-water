#!/usr/bin/env python3
"""Dependency-free lifecycle coordination for the Phase 2.5 bake-off.

The coordinator owns ordering and state transitions only.  A caller supplies the work for one
``CellKey`` through an executor callback; image loading, inference, and metric calculation stay
outside this module.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .matrix import CellKey, MatrixPlan
from .resume import load_state, mark_complete, mark_in_progress


class CoordinatorFailure(ValueError):
    """Stable, reportable lifecycle or executor-result failure."""

    def __init__(self, kind: str, message: str):
        self.kind = kind
        self.reason = kind
        self.failure_type = "coordinator_failure"
        self.message = message
        super().__init__(f"{kind}: {message}")


class IncompleteFailure(CoordinatorFailure):
    """Raised when completed records are requested before every cell is complete."""

    def __init__(self, message: str):
        super().__init__("incomplete", message)


@dataclass(frozen=True)
class CommittedExecution:
    """The only executor result that can complete a cell.

    ``result`` is the public report record; ``artifact_ref`` is the exact immutable artifact
    generation that proves it.  Keeping these together makes it impossible for a caller to
    accidentally persist a result without the generation that produced it.
    """

    result: Mapping[str, Any]
    artifact_ref: Mapping[str, Any]


Executor = Callable[[CellKey], CommittedExecution]
# The validator receives the complete association being persisted.  Validating only the ref
# proves that *some* generation exists, but cannot reject a valid generation swapped between
# cells or a result edited while retaining the same ref.
ArtifactRefValidator = Callable[[CellKey, Mapping[str, Any], Mapping[str, Any]], Any]

_IDENTITY_FIELDS = (
    ("candidate_id", "candidate"),
    ("shot_id", "shot"),
    ("conditioning_token", "conditioning"),
    ("cap_token", "cap"),
    ("provider", "provider"),
    ("host_load", "host_load"),
)
_STATUSES = {"pass", "fail", "skip"}
_FAILURE_FIELDS = {"type", "message", "retryable", "stage"}
_FAILURE_TYPES = frozenset({
    "artifact_missing",
    "artifact_hash_mismatch",
    "license_not_permitted",
    "license_unknown",
    "provider_unavailable",
    "unsupported_tensor_contract",
    "wrong_direction",
    "export_not_reproducible",
    "missing_input",
    "input_invalid",
    "conditioning_failure",
    "cap_unavailable",
    "out_of_memory",
    "runtime_error",
    "nonfinite_output",
    "repeated_run_instability",
    "quality_gate_failed",
    "operator_cancelled",
    "not_attempted",
    "other",
})


def _fail(kind: str, message: str) -> None:
    raise CoordinatorFailure(kind, message)


def _validate_json(value: Any, path: str = "$", seen: set[int] | None = None) -> None:
    """Validate the strict JSON subset accepted from an executor.

    ``type(...) is ...`` is intentional: an executor result must be made of plain Python JSON
    values, rather than custom mappings, tuples, decimals, or other values that merely happen
    to be serializable by a particular encoder.
    """

    value_type = type(value)
    if value is None or value_type in (str, bool, int):
        return
    if value_type is float:
        if not math.isfinite(value):
            _fail("nonfinite", f"{path} contains a nonfinite number")
        return
    if value_type is dict:
        if seen is None:
            seen = set()
        marker = id(value)
        if marker in seen:
            _fail("json_value", f"{path} contains a cycle")
        seen.add(marker)
        for key, child in value.items():
            if type(key) is not str:
                _fail("json_value", f"{path} contains a non-string object key")
            _validate_json(child, f"{path}.{key}", seen)
        seen.remove(marker)
        return
    if value_type is list:
        if seen is None:
            seen = set()
        marker = id(value)
        if marker in seen:
            _fail("json_value", f"{path} contains a cycle")
        seen.add(marker)
        for index, child in enumerate(value):
            _validate_json(child, f"{path}[{index}]", seen)
        seen.remove(marker)
        return
    _fail("json_value", f"{path} contains a non-JSON value")


def _nonempty_string(value: Any, path: str) -> None:
    if type(value) is not str or not value:
        _fail("result_identity", f"{path} must be a non-empty string")


def _validate_failure(failure: Any, path: str) -> None:
    if type(failure) is not dict:
        _fail("failure_shape", f"{path} must be a plain JSON object")
    unknown = set(failure) - _FAILURE_FIELDS
    if unknown:
        _fail("failure_shape", f"{path} contains unsupported field {sorted(unknown)[0]!r}")
    if "type" not in failure or "message" not in failure:
        _fail("failure_shape", f"{path} requires type and message")
    _nonempty_string(failure["type"], f"{path}.type")
    if failure["type"] not in _FAILURE_TYPES:
        _fail("failure_type", f"{path}.type is not a permitted report failure type")
    _nonempty_string(failure["message"], f"{path}.message")
    if "retryable" in failure and type(failure["retryable"]) is not bool:
        _fail("failure_shape", f"{path}.retryable must be boolean")
    if "stage" in failure:
        _nonempty_string(failure["stage"], f"{path}.stage")


def _validate_result(result: Any, cell: CellKey) -> dict[str, Any]:
    """Validate one executor result and return the unchanged plain object."""

    if type(result) is not dict:
        _fail("result_shape", "executor result must be a plain JSON object")
    _validate_json(result, "result")
    for result_field, cell_field in _IDENTITY_FIELDS:
        if result_field not in result:
            _fail("result_identity", f"result is missing {result_field}")
        value = result[result_field]
        _nonempty_string(value, f"result.{result_field}")
        if value != getattr(cell, cell_field):
            _fail(
                "cell_mismatch",
                f"result.{result_field} does not match CellKey.{cell_field}",
            )
    status = result.get("status")
    if type(status) is not str or status not in _STATUSES:
        _fail("result_status", "result.status must be pass, fail, or skip")
    if status == "pass":
        if "failure" in result:
            _fail("failure_shape", "passing result must not carry a failure")
    else:
        if "failure" not in result:
            _fail("failure_missing", "non-pass result requires a typed failure")
        _validate_failure(result["failure"], "result.failure")
    return result


def _record(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Return a fresh public wrapper for one already-validated complete entry."""

    # Copy the top-level objects so callers cannot mutate the state mapping held by a caller or
    # the in-memory transition result.  Nested values are JSON data and are not shared with the
    # persisted file.
    return {
        "cell": deepcopy(entry["cell"]),
        "state": "complete",
        "result": deepcopy(entry["result"]),
    }


def _record_with_ref(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Return an internal record retaining the exact ref for publication/recovery callers."""

    return {
        "cell": deepcopy(entry["cell"]),
        "state": "complete",
        "result": deepcopy(entry["result"]),
        "artifact_ref": deepcopy(entry["artifact_ref"]),
    }


def _validate_artifact_ref(
    cell: CellKey,
    result: Mapping[str, Any],
    artifact_ref: Any,
    validator: ArtifactRefValidator,
) -> dict[str, Any]:
    if not isinstance(artifact_ref, Mapping):
        _fail("artifact_ref_shape", "executor artifact_ref must be a mapping")
    # The resume module performs the strict serialized shape check.  Make a detached plain
    # mapping before invoking the store so a validator cannot mutate executor-owned state.
    detached = deepcopy(dict(artifact_ref))
    try:
        accepted = validator(cell, deepcopy(result), detached)
    except CoordinatorFailure:
        raise
    except Exception as exc:  # noqa: BLE001 - the injected store has backend-specific errors
        _fail("artifact_ref", f"artifact_ref validation failed for {cell!r}: {exc}")
    if accepted is False:
        _fail("artifact_ref", f"artifact_ref validator rejected the exact reference for {cell!r}")
    return detached


def _validated_execution(
    execution: Any,
    cell: CellKey,
    validator: ArtifactRefValidator,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(execution, CommittedExecution):
        _fail(
            "executor_contract",
            "executor must return CommittedExecution(result, artifact_ref); "
            "legacy result-only mappings cannot complete a cell",
        )
    validated_result = _validate_result(execution.result, cell)
    validated_ref = _validate_artifact_ref(cell, validated_result, execution.artifact_ref, validator)
    return deepcopy(validated_result), validated_ref


def _completed_records(
    state: Mapping[str, Any],
    plan: MatrixPlan,
    artifact_ref_validator: ArtifactRefValidator,
    *,
    include_refs: bool = False,
) -> list[dict[str, Any]]:
    entries = state["entries"]
    incomplete = [
        (index, entry["cell"])
        for index, entry in enumerate(entries)
        if entry["state"] != "complete"
    ]
    if incomplete:
        index, cell = incomplete[0]
        raise IncompleteFailure(f"cell {index} is not complete: {cell!r}")

    records: list[dict[str, Any]] = []
    for cell, entry in zip(plan.cells, entries):
        _validate_result(entry["result"], cell)
        _validate_artifact_ref(
            cell, entry["result"], entry["artifact_ref"], artifact_ref_validator,
        )
        records.append(_record_with_ref(entry) if include_refs else _record(entry))
    return records


class RunCoordinator:
    """Run pending cells in plan order and publish only validated results."""

    def __init__(
        self,
        state_path: Path | str,
        expected_identity: Mapping[str, Any],
        plan: MatrixPlan,
        executor: Executor,
        artifact_ref_validator: ArtifactRefValidator,
    ) -> None:
        if not callable(executor):
            raise CoordinatorFailure("executor", "executor must be callable")
        if not callable(artifact_ref_validator):
            raise CoordinatorFailure("artifact_ref_validator", "artifact_ref_validator must be callable")
        self.state_path = Path(state_path)
        self.expected_identity = expected_identity
        self.plan = plan
        self.executor = executor
        self.artifact_ref_validator = artifact_ref_validator

    def run(self) -> list[dict[str, Any]]:
        """Resume the state file, execute pending cells, and return ordered complete records.

        The in-progress transition is deliberately outside the executor call.  Any exception
        from the executor, or any validation error from its result, therefore leaves that cell
        in ``in_progress`` for ``load_state`` to recover on the next invocation.
        """

        state = load_state(self.state_path, self.expected_identity, self.plan)
        records: list[dict[str, Any] | None] = [None] * len(self.plan.cells)
        for index, cell in enumerate(self.plan.cells):
            entry = state["entries"][index]
            if entry["state"] == "complete":
                # ``load_state`` checks only JSON shape and finiteness.  Existing complete
                # records therefore need their coordinator-level semantic check exactly once
                # during this invocation.
                _validate_result(entry["result"], cell)
                _validate_artifact_ref(
                    cell, entry["result"], entry["artifact_ref"], self.artifact_ref_validator,
                )
                records[index] = _record(entry)
                continue
            # load_state recovers interrupted work, so a non-complete entry here must be pending.
            state = mark_in_progress(
                self.state_path, self.expected_identity, self.plan, cell
            )
            execution = self.executor(cell)
            validated, artifact_ref = _validated_execution(
                execution, cell, self.artifact_ref_validator,
            )
            state = mark_complete(
                self.state_path,
                self.expected_identity,
                self.plan,
                cell,
                validated,
                artifact_ref,
            )
            records[index] = _record(state["entries"][index])
        # Every plan cell is either an existing complete record or was completed above.  Avoid a
        # trailing load/semantic-validation pass: the transition writes are already durable and
        # the wrappers are fresh copies of the validated records.
        return [record for record in records if record is not None]

    def completed_records(self) -> list[dict[str, Any]]:
        """Return complete resume records in MatrixPlan order, or raise ``IncompleteFailure``."""

        state = load_state(self.state_path, self.expected_identity, self.plan)
        return _completed_records(state, self.plan, self.artifact_ref_validator)

    def completed_records_with_refs(self) -> list[dict[str, Any]]:
        """Return complete records including exact refs for internal publication/recovery code."""

        state = load_state(self.state_path, self.expected_identity, self.plan)
        return _completed_records(
            state, self.plan, self.artifact_ref_validator, include_refs=True,
        )


def run(
    state_path: Path | str,
    expected_identity: Mapping[str, Any],
    plan: MatrixPlan,
    executor: Executor,
    artifact_ref_validator: ArtifactRefValidator,
) -> list[dict[str, Any]]:
    """Convenience wrapper around :class:`RunCoordinator`."""

    return RunCoordinator(
        state_path, expected_identity, plan, executor, artifact_ref_validator,
    ).run()


def completed_records(
    state_path: Path | str,
    expected_identity: Mapping[str, Any],
    plan: MatrixPlan,
    artifact_ref_validator: ArtifactRefValidator,
) -> list[dict[str, Any]]:
    """Expose persisted complete records without executing any cell."""

    state = load_state(Path(state_path), expected_identity, plan)
    return _completed_records(state, plan, artifact_ref_validator)


__all__ = [
    "CoordinatorFailure",
    "CommittedExecution",
    "ArtifactRefValidator",
    "Executor",
    "IncompleteFailure",
    "RunCoordinator",
    "completed_records",
    "run",
]
