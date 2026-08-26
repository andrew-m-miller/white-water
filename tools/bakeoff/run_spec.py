#!/usr/bin/env python3
"""Immutable run specifications and deterministic identity hashing.

The profile driver has two different kinds of data which used to travel together in a
single report-metadata object:

* stable evaluation inputs -- protocol/corpus content, matrix selection, exact candidate
  artifacts, runner/runtime versions, hardware and measurement settings, plus semantic report
  inputs such as warnings and operator-supplied summary/score metadata; and
* publication metadata -- report ids, timestamps, command lines, output locations and logs
  generated while a run is being published.

``RunSpec`` makes that boundary explicit.  Every value in its ``stable_inputs`` object is
identity-bearing; there is deliberately no stable-field allowlist or volatile-key denylist.
Adding a new stable input therefore changes the identity automatically.  Publication data is
held by ``PublicationMetadata`` and is never included in the identity hash.

Every spec carries ``IDENTITY_SCHEMA_VERSION`` as the named
``identity_schema_version`` stable field.  Bump that module constant when the identity format or
driver semantics change.  ``from_mapping`` accepts unknown fields as stable by default, while
``from_inputs`` is an intentionally explicit convenience surface for the currently known config
fields and must be updated when a new one becomes part of the driver contract.

This module has no dependency on the runner or on third-party packages.  It accepts the strict
JSON value subset used by the bake-off state/report code: ``None``, strings, booleans, finite
integers/floats, plain dictionaries with string keys, and lists.  Tuples, sets, paths, custom
objects, non-string keys, cycles, and non-finite floats are rejected before ``json.dumps`` is
called so canonicalization cannot silently accept a Python value which is not JSON.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping


IDENTITY_SCHEMA_VERSION = 1
_PUBLICATION_SEMANTIC_FIELDS = frozenset({"warnings", "summary", "report_inputs"})


class RunSpecError(ValueError):
    """Raised when run-spec or publication metadata is not strict JSON."""

    def __init__(self, kind: str, message: str):
        self.kind = kind
        self.reason = kind
        self.failure_type = "run_spec_error"
        self.message = message
        super().__init__(f"{kind}: {message}")


def _fail(kind: str, message: str) -> None:
    raise RunSpecError(kind, message)


def _validate_json(value: Any, path: str = "$", seen: set[int] | None = None) -> None:
    """Validate the strict JSON subset accepted by this module.

    ``type(...) is ...`` is intentional.  Python's JSON encoder accepts tuples and some
    mapping subclasses even though they are not the plain JSON values used by the runner's
    persistence contracts.  Rejecting those values here makes identity formation fail at the
    boundary rather than depend on encoder-specific behaviour.
    """

    value_type = type(value)
    if value is None or value_type in (str, bool, int):
        return
    if value_type is float:
        if not math.isfinite(value):
            _fail("nonfinite", f"{path} contains a nonfinite number")
        return
    if value_type is dict:
        active = seen if seen is not None else set()
        marker = id(value)
        if marker in active:
            _fail("json_value", f"{path} contains a cycle")
        active.add(marker)
        for key, child in value.items():
            if type(key) is not str:
                _fail("json_value", f"{path} contains a non-string object key")
            _validate_json(child, f"{path}.{key}", active)
        active.remove(marker)
        return
    if value_type is list:
        active = seen if seen is not None else set()
        marker = id(value)
        if marker in active:
            _fail("json_value", f"{path} contains a cycle")
        active.add(marker)
        for index, child in enumerate(value):
            _validate_json(child, f"{path}[{index}]", active)
        active.remove(marker)
        return
    _fail("json_value", f"{path} contains a non-JSON value ({value_type.__name__})")


def canonical_json(value: Any) -> str:
    """Return one deterministic JSON representation of ``value``.

    Object keys are sorted, insignificant whitespace is removed, UTF-8 characters are retained,
    and ``allow_nan=False`` is kept as a second guard against non-finite numbers.  The returned
    string has no trailing newline; callers that persist it can choose their own file framing.
    """

    _validate_json(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError) as exc:  # defensive: validation ran first
        raise RunSpecError("json_value", f"cannot canonicalize JSON value: {exc}") from exc


def canonical_json_bytes(value: Any) -> bytes:
    """Return canonical JSON encoded as UTF-8 bytes."""

    return _utf8_bytes(canonical_json(value))


def _utf8_bytes(encoded: str) -> bytes:
    """Encode canonical JSON text, converting lone-surrogate failures to RunSpecError."""

    try:
        return encoded.encode("utf-8")
    except UnicodeEncodeError as exc:
        # A lone surrogate can pass through ``json.dumps(ensure_ascii=False)`` as a Python
        # string but cannot be represented in the UTF-8 bytes whose hash defines identity.
        raise RunSpecError("json_value", f"JSON contains an invalid Unicode string: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    """Return the SHA256 of the canonical JSON representation of ``value``."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _decode_canonical(encoded: str) -> dict[str, Any]:
    """Decode an internally stored canonical object into a fresh mutable copy."""

    value = json.loads(encoded)
    # ``RunSpec`` and ``PublicationMetadata`` only canonicalize top-level objects.  This check
    # is deliberately redundant with construction, but keeps the public properties defensive
    # if the implementation is changed later.
    if type(value) is not dict:
        raise RunSpecError("json_value", "stored metadata is not a JSON object")
    return value


def _publication_value(value: PublicationMetadata | Mapping[str, Any] | None) -> PublicationMetadata:
    if value is None:
        return PublicationMetadata({})
    if isinstance(value, PublicationMetadata):
        return value
    return PublicationMetadata(value)


def _stable_with_identity_version(stable_inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Copy stable inputs and ensure every spec carries the current identity schema field."""

    stable = dict(stable_inputs)
    if "identity_schema_version" not in stable:
        stable["identity_schema_version"] = IDENTITY_SCHEMA_VERSION
    else:
        value = stable["identity_schema_version"]
        if type(value) is not int or value < 1:
            _fail("identity_schema_version", "identity_schema_version must be a positive integer")
    return stable


@dataclass(frozen=True, slots=True, init=False)
class PublicationMetadata:
    """Generated/operator-facing metadata deliberately excluded from run identity.

    The mapping is kept as canonical JSON internally, so mutating the caller's input mapping or
    a mapping returned by :meth:`as_dict` cannot mutate a ``RunSpec``.  This intentionally has a
    generic shape for non-semantic publication details such as report ids, timestamps, command
    lines, output paths and logs.  Semantic report inputs belong in ``RunSpec.report_inputs``
    instead and must affect identity.
    """

    _canonical: str = field(repr=False)

    def __init__(self, values: Mapping[str, Any] | None = None):
        if values is None:
            values = {}
        if type(values) is not dict:
            _fail("json_value", "publication metadata must be a plain JSON object")
        semantic_fields = sorted(_PUBLICATION_SEMANTIC_FIELDS.intersection(values))
        if semantic_fields:
            _fail(
                "publication_field",
                f"{semantic_fields[0]!r} is semantic report input and belongs in RunSpec.report_inputs",
            )
        encoded = canonical_json(values)
        # Validate the exact bytes that persistence/hash consumers will use now, rather than
        # allowing a lone surrogate to survive construction until ``sha256`` or file output.
        _utf8_bytes(encoded)
        object.__setattr__(self, "_canonical", encoded)

    @property
    def canonical_json(self) -> str:
        """Canonical JSON for publication metadata (never used for identity)."""

        return self._canonical

    @property
    def sha256(self) -> str:
        """Hash of publication metadata, for diagnostics only.

        This is intentionally not the run identity.  Callers should use
        ``RunSpec.identity_sha256`` when deciding whether work can be resumed.
        """

        return hashlib.sha256(_utf8_bytes(self._canonical)).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        """Return a fresh mutable copy suitable for report assembly."""

        return _decode_canonical(self._canonical)

    to_dict = as_dict


# These are the stable inputs currently owned by ``tools/bakeoff/run.py``.  The tuple is
# documentation and a convenience for callers constructing a spec; it is *not* an allowlist.
# ``RunSpec.from_mapping`` accepts additional stable fields and hashes them automatically.
KNOWN_STABLE_INPUTS: tuple[str, ...] = (
    "identity_schema_version",
    "protocol",
    "corpus",
    "candidate_entries",
    "selection",
    "matrix",
    "artifacts",
    "report_schema",
    "corpus_schema",
    "environment",
    "profile",
    "runner",
    "hardware",
    "chain_offsets",
    "measurement",
    "report_inputs",
)


@dataclass(frozen=True, slots=True, init=False)
class RunSpec:
    """Immutable stable run inputs plus separately held publication metadata.

    Construct with ``RunSpec(stable_inputs, publication=...)`` or the more descriptive
    ``RunSpec.from_mapping``/``RunSpec.from_inputs`` factories.  Every key and nested value in
    ``stable_inputs`` is included in the canonical identity payload.  There is no silent
    filtering of unknown keys, so adding a stable input cannot accidentally leave resume state
    reusable under the old identity.
    """

    _stable_canonical: str = field(repr=False)
    _identity_hash: str = field(repr=False)
    publication: PublicationMetadata = field(repr=False)

    def __init__(
        self,
        stable_inputs: Mapping[str, Any],
        publication: PublicationMetadata | Mapping[str, Any] | None = None,
    ):
        if type(stable_inputs) is not dict:
            _fail("json_value", "stable inputs must be a plain JSON object")
        stable = _stable_with_identity_version(stable_inputs)
        encoded = canonical_json(stable)
        object.__setattr__(self, "_stable_canonical", encoded)
        object.__setattr__(self, "_identity_hash", hashlib.sha256(_utf8_bytes(encoded)).hexdigest())
        object.__setattr__(self, "publication", _publication_value(publication))
        # Keep the invariant at the immutable identity boundary.  This is deliberately a
        # typed check rather than a bare ``assert`` so optimized Python cannot remove it.
        self.assert_identity()

    @classmethod
    def from_mapping(
        cls,
        stable_inputs: Mapping[str, Any],
        *,
        publication: PublicationMetadata | Mapping[str, Any] | None = None,
    ) -> RunSpec:
        """Build a spec from a complete stable-input mapping.

        The mapping is copied through canonical JSON at construction.  Every supplied key is
        identity-bearing by default, and only values passed through ``publication`` are excluded.
        Unknown keys are therefore safe when using this generic boundary.  The named
        ``from_inputs`` convenience surface is intentionally narrower and must be updated when
        a new known driver configuration surface is introduced.
        """

        return cls(stable_inputs, publication=publication)

    @classmethod
    def from_inputs(
        cls,
        *,
        protocol: Any,
        corpus: Any,
        candidate_entries: Any,
        selection: Any,
        matrix: Any,
        artifacts: Any,
        report_schema: Any,
        corpus_schema: Any,
        environment: Any,
        profile: Any,
        runner: Any,
        hardware: Any,
        chain_offsets: Any,
        measurement: Any,
        report_inputs: Any,
        identity_schema_version: int = IDENTITY_SCHEMA_VERSION,
        extra_stable: Mapping[str, Any] | None = None,
        publication: PublicationMetadata | Mapping[str, Any] | None = None,
    ) -> RunSpec:
        """Build a spec from the current profile-driver stable input set.

        ``report_inputs`` contains semantic report content such as warnings and operator-supplied
        summary/score metadata; changing it must invalidate resume state.  ``extra_stable`` is
        intentionally available for a new result-affecting input before the convenience
        signature is updated.  Overwriting one of the named fields is rejected so a caller
        cannot accidentally create two competing values for one identity field.
        """

        stable = {
            "protocol": protocol,
            "corpus": corpus,
            "candidate_entries": candidate_entries,
            "selection": selection,
            "matrix": matrix,
            "artifacts": artifacts,
            "report_schema": report_schema,
            "corpus_schema": corpus_schema,
            "environment": environment,
            "profile": profile,
            "runner": runner,
            "hardware": hardware,
            "chain_offsets": chain_offsets,
            "measurement": measurement,
            "report_inputs": report_inputs,
            "identity_schema_version": identity_schema_version,
        }
        if extra_stable is not None:
            if type(extra_stable) is not dict:
                _fail("json_value", "extra_stable must be a plain JSON object")
            overlap = sorted(set(stable).intersection(extra_stable))
            if overlap:
                _fail("stable_field", f"extra_stable overlaps named field {overlap[0]!r}")
            stable.update(extra_stable)
        return cls(stable, publication=publication)

    @property
    def canonical_json(self) -> str:
        """Canonical JSON containing stable inputs only."""

        return self._stable_canonical

    @property
    def canonical_bytes(self) -> bytes:
        """UTF-8 bytes of :attr:`canonical_json`."""

        return _utf8_bytes(self._stable_canonical)

    @property
    def identity_sha256(self) -> str:
        """SHA256 of stable canonical JSON; publication metadata is excluded."""

        return self._identity_hash

    def assert_identity(self) -> None:
        """Raise ``RunSpecError`` unless the stored identity matches stable inputs.

        Callers that pass the spec to another identity-bearing subsystem can use this one
        boundary check instead of reimplementing the hash comparison.  The public comparison is
        intentionally against :func:`canonical_sha256` of a detached ``stable_inputs`` copy,
        which also verifies that the persisted representation decodes to the same JSON value.
        """

        expected = canonical_sha256(self.stable_inputs)
        if self.identity_sha256 != expected:
            _fail(
                "identity_hash",
                "identity_sha256 does not match canonical_sha256(stable_inputs)",
            )

    @property
    def stable_inputs(self) -> dict[str, Any]:
        """Return a fresh mutable copy of all stable inputs."""

        return _decode_canonical(self._stable_canonical)

    # These aliases make the distinction explicit at call sites that use "identity" language.
    identity_payload = stable_inputs
    as_identity = stable_inputs

    def with_publication(
        self, publication: PublicationMetadata | Mapping[str, Any] | None
    ) -> RunSpec:
        """Return an equivalent immutable spec with replacement publication metadata."""

        return RunSpec(self.stable_inputs, publication=publication)

    def as_record(self, *, include_publication: bool = False) -> dict[str, Any]:
        """Return a persistence-friendly record with an explicit publication boundary.

        The identity hash is computed only from ``stable``.  Including publication data here is
        useful for a diagnostic sidecar, but callers must not hash this complete record when
        deciding resume compatibility.
        """

        record: dict[str, Any] = {
            "stable": self.stable_inputs,
            "identity_sha256": self.identity_sha256,
        }
        if include_publication:
            record["publication"] = self.publication.as_dict()
        return record


__all__ = [
    "IDENTITY_SCHEMA_VERSION",
    "KNOWN_STABLE_INPUTS",
    "PublicationMetadata",
    "RunSpec",
    "RunSpecError",
    "canonical_json",
    "canonical_json_bytes",
    "canonical_sha256",
]
