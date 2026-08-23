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


PADDING_POLICIES = ("replication", "reflect")


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
    if not isinstance(policy, str) or policy not in PADDING_POLICIES:
        raise PaddingPolicyError(
            f"padding policy must be one of {', '.join(PADDING_POLICIES)}"
        )
    return policy


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
) -> PaddedRows:
    """Pad bottom-left row-major pixels by replication or edge reflection.

    The source rows are not mutated.  ``replication`` clamps coordinates to the
    nearest source pixel and is the policy used by the current SEA-RAFT caller.
    ``reflect`` mirrors without repeating edge pixels and matches core preprocessing.
    A policy is mandatory by design: callers must not accidentally compare a model
    using replication against one using reflect.
    """

    policy = normalize_policy(policy)
    width, height = _validate_rows(rows)
    sides = (left, right, bottom, top)
    if any(isinstance(side, bool) or not isinstance(side, int) or side < 0 for side in sides):
        raise ValueError("padding sides must be non-negative integers")

    output: list[tuple[Any, ...]] = []
    for output_y in range(height + bottom + top):
        source_y = output_y - bottom
        if policy == "replication":
            source_y = min(height - 1, max(0, source_y))
        else:
            source_y = _mirror_index(source_y, height)
        source_row = rows[source_y]
        padded_row: list[Any] = []
        for output_x in range(width + left + right):
            source_x = output_x - left
            if policy == "replication":
                source_x = min(width - 1, max(0, source_x))
            else:
                source_x = _mirror_index(source_x, width)
            padded_row.append(source_row[source_x])
        output.append(tuple(padded_row))

    return PaddedRows(
        tuple(output), width, height, left, right, bottom, top, policy
    )


__all__ = [
    "PADDING_POLICIES",
    "PaddingPolicyError",
    "PaddedRows",
    "normalize_policy",
    "pad_rows",
]
