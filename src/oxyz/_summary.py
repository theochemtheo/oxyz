"""Display and equality shared by the public dataclasses.

A dataclass `__repr__` over a dict of arrays prints its contents (500 KB for a
100k-atom frame with a string column) and a dataclass `__eq__` raises
`ValueError` on an ndarray value. Every public repr is built from these
instead, so each is one line whose length depends on how many names an object
carries, never on how many atoms or frames.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from _typeshed import DataclassInstance

_MAX_ENTRIES = 8
"""How many entries a repr shows before eliding the rest: enough for a typical
frame's columns, short of the ~16 metadata keys a heavy comment line carries."""


def summarise(value: object) -> str:
    """Describe one value by dtype and shape, never by content."""
    if isinstance(value, np.ndarray):
        return f"{value.dtype}{list(value.shape)}"
    if isinstance(value, list):
        # A column list is list[str] or list[list[str]] by contract; take the
        # tag from the first element rather than asserting `str` over a
        # hand-built list of something else.
        first = value[0] if value else ""
        if isinstance(first, list):
            inner = type(first[0]).__name__ if first else "str"
            return f"{inner}[{len(value)}, {len(first)}]"
        return f"{type(first).__name__}[{len(value)}]"
    return type(value).__name__


def mapping_repr[V](
    mapping: Mapping[str, V], render: Callable[[V], str] = summarise
) -> str:
    """Render a name->value map as `{'pos': float64[96, 3], ...}`.

    Insertion order, at most `_MAX_ENTRIES` entries, then `, +N more`.
    """
    items = list(mapping.items())
    shown = ", ".join(
        f"{name!r}: {render(value)}" for name, value in items[:_MAX_ENTRIES]
    )
    elided = len(items) - _MAX_ENTRIES
    if elided > 0:
        shown = f"{shown}, +{elided} more"
    return "{" + shown + "}"


def atom_range(n_atoms: np.ndarray) -> str:
    """Render per-frame atom counts as `, n_atoms=lo..hi`, or `""` if empty.

    Always a range, even when `lo == hi`, so it cannot be misread as one
    frame's count.
    """
    if not n_atoms.size:
        return ""
    return f", n_atoms={int(n_atoms.min())}..{int(n_atoms.max())}"


def nondefault_repr(obj: DataclassInstance) -> str:
    """Render a dataclass as its constructor, leaving out defaulted fields.

    A field with no default is always shown.
    """
    shown = [
        f"{f.name}={getattr(obj, f.name)!r}"
        for f in dataclasses.fields(obj)
        if f.repr
        and (f.default is dataclasses.MISSING or getattr(obj, f.name) != f.default)
    ]
    return f"{type(obj).__name__}({', '.join(shown)})"


def values_equal(a: object, b: object) -> bool:
    """Report whether two stored values hold the same data, NaN equal to NaN.

    Shapes must match; two float arrays compare bit-exact with `equal_nan`, so
    a NaN column equals itself. Never raises, because `__eq__` must return a
    bool: a value numpy cannot coerce, or an object array whose elements have
    no unambiguous truth, simply equals nothing but itself.
    """
    if a is b:
        return True
    try:
        left, right = np.asarray(a), np.asarray(b)
    except (ValueError, TypeError):  # a ragged hand-built value equals nothing
        return False
    if left.shape != right.shape:
        return False
    try:
        if left.dtype.kind == "f" and right.dtype.kind == "f":
            # equal_nan needs both sides float: np.isnan raises on strings, and
            # an integer array has no NaN to preserve.
            return bool(np.array_equal(left, right, equal_nan=True))
        return bool(np.array_equal(left, right))
    except ValueError:
        # An object array of arrays: `bool()` over the elementwise comparison
        # is ambiguous, so the two are not demonstrably equal.
        return False


def mappings_equal(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    """Report whether both maps carry the same names and equal values.

    Key *sets*, not key order: `oxyz.write` imposes its own column order, so a
    round-tripped frame that reordered nothing else stays equal.
    """
    if set(left) != set(right):
        return False
    return all(values_equal(value, right[name]) for name, value in left.items())
