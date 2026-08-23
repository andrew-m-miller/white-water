#!/usr/bin/env python3
"""Small, explicit caller-padding seam for the Phase 2.5 bake-off.

P25-1 owns the candidate manifest and will declare each artifact's caller-side
padding mode.  This module intentionally does not import or interpret that manifest;
the runner passes the declared string through :func:`pad_rows`.  Keeping this seam
small prevents the current SEA-RAFT replication-padding contract from being silently
replaced by the core preprocessing layer's reflect padding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


# These are the tokens emitted by candidate artifact manifests.  The migrated
# SEA-RAFT manifest uses ``caller-replication-crop``; keeping that spelling here
# means the runner can pass the declaration through without candidate-specific
# translation.  The short names remain accepted as compatibility aliases for
# older P25-2 callers, but normalized results always carry the manifest token.
PADDING_POLICIES = ("caller-replication-crop", "caller-reflection-crop")
_PADDING_ALIASES = {
    "replication": "caller-replication-crop",
    "reflect": "caller-reflection-crop",
}


class PaddingPolicyError(ValueError):
    """Raised for an unknown or malformed caller-padding policy."""


@dataclass(frozen=True)
class PaddedRows:
    """Padded row-major pixels and the crop needed to recover the source."""

    rows: tuple[tuple[Any, ...], ...]
    source_width: int
    source_height: int
    pad_left: int
    pad_right: int
    pad_bottom: int
    pad_top: int
    policy: str

    @property
    def width(self) -> int:
        return self.source_width + self.pad_left + self.pad_right

    @property
    def height(self) -> int:
        return self.source_height + self.pad_bottom + self.pad_top

    @property
    def crop(self) -> tuple[int, int, int, int]:
        """``(x, y, width, height)`` in the padded image, bottom-left rows."""

        return (self.pad_left, self.pad_bottom, self.source_width, self.source_height)


def normalize_policy(policy: str) -> str:
    if not isinstance(policy, str):
        raise PaddingPolicyError(
            f"padding policy must be one of {', '.join(PADDING_POLICIES)}"
        )
    normalized = _PADDING_ALIASES.get(policy, policy)
    if normalized not in PADDING_POLICIES:
        raise PaddingPolicyError(
            f"padding policy must be one of {', '.join(PADDING_POLICIES)}"
        )
    return normalized


def _mirror_index(index: int, size: int) -> int:
    # Match src/core/flow/Preprocess.cpp: reflection is about the edge pixel,
    # without repeating that edge in the first padded sample.
    if size <= 1:
        return 0
    if 0 <= index < size:
        return index
    period = 2 * (size - 1)
    folded = index % period
    return folded if folded < size else period - folded


def _validate_rows(rows: Sequence[Sequence[Any]]) -> tuple[int, int]:
    if not isinstance(rows, Sequence) or not rows:
        raise ValueError("padding source must contain at least one row")
    if not all(isinstance(row, Sequence) and row for row in rows):
        raise ValueError("padding source rows must be non-empty sequences")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("padding source rows must have equal width")
    return width, len(rows)


def pad_rows(
    rows: Sequence[Sequence[Any]],
    *,
    left: int = 0,
    right: int = 0,
    bottom: int = 0,
    top: int = 0,
    policy: str,
    multiple: int = 1,
) -> PaddedRows:
    """Pad bottom-left row-major pixels by replication or edge reflection.

    The source rows are not mutated.  ``caller-replication-crop`` clamps coordinates
    to the nearest source pixel and is the policy used by the current SEA-RAFT caller.
    ``caller-reflection-crop`` mirrors without repeating edge pixels and matches core
    preprocessing.  The short ``replication``/``reflect`` spellings are accepted only
    as compatibility aliases and normalize to the canonical tokens.
    A policy is mandatory by design: callers must not accidentally compare a model
    using replication against one using reflect.  When ``multiple`` is greater than
    one, the requested right/top halo is extended as needed so the returned tensor
    dimensions are exact multiples.  The returned side counts are the actual crop
    seam, including those extensions.
    """

    policy = normalize_policy(policy)
    width, height = _validate_rows(rows)
    sides = (left, right, bottom, top)
    if any(isinstance(side, bool) or not isinstance(side, int) or side < 0 for side in sides):
        raise ValueError("padding sides must be non-negative integers")
    if isinstance(multiple, bool) or not isinstance(multiple, int) or multiple <= 0:
        raise ValueError("padding multiple must be a positive integer")

    minimum_width = width + left + right
    minimum_height = height + bottom + top
    padded_width = ((minimum_width + multiple - 1) // multiple) * multiple
    padded_height = ((minimum_height + multiple - 1) // multiple) * multiple
    actual_right = right + padded_width - minimum_width
    actual_top = top + padded_height - minimum_height
    if padded_width % multiple != 0 or padded_height % multiple != 0:
        raise AssertionError("caller padding failed to produce requested multiple")

    output: list[tuple[Any, ...]] = []
    for output_y in range(padded_height):
        source_y = output_y - bottom
        if policy == "caller-replication-crop":
            source_y = min(height - 1, max(0, source_y))
        else:
            source_y = _mirror_index(source_y, height)
        source_row = rows[source_y]
        padded_row: list[Any] = []
        for output_x in range(padded_width):
            source_x = output_x - left
            if policy == "caller-replication-crop":
                source_x = min(width - 1, max(0, source_x))
            else:
                source_x = _mirror_index(source_x, width)
            padded_row.append(source_row[source_x])
        output.append(tuple(padded_row))

    return PaddedRows(
        tuple(output), width, height, left, actual_right, bottom, actual_top, policy
    )


__all__ = [
    "PADDING_POLICIES",
    "PaddingPolicyError",
    "PaddedRows",
    "normalize_policy",
    "pad_rows",
]
