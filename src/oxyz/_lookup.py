"""Mapping-like lookup by name across a columns side and a metadata side.

`Frame`, `Batch`, `Schema`, and `SchemaSpec` share one definition of the rules:
columns are searched before metadata, and a name on both sides is an error
rather than a silent pick.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

from oxyz._rust import OxyzError

if TYPE_CHECKING:
    from collections.abc import Mapping


class AmbiguousNameError(OxyzError):
    """A name is both a column and a metadata key, so a lookup cannot choose.

    Raised by `frame[name]` and `.get(name)` (and the same on `Batch`, `Schema`
    and `SchemaSpec`), never by `name in frame`. Deliberately not a `KeyError`,
    so neither `get` nor `except KeyError` can mistake it for absence.

    Attributes
    ----------
    name
        The name found on both sides.
    """

    def __init__(self, message: str, *, name: str) -> None:
        super().__init__(message)
        self.name = name


class NameLookup[C, M]:
    """Mapping-like access by name over a host's columns and metadata.

    Not a `Mapping`: iterating the names is disabled and no `__len__` counts
    them. With both, numpy reads a list of hosts as a sequence of key names, so
    `rng.choice(frames, k)` would return names rather than frames. `Batch`
    instead defines both over its frames, as the sequence it is.
    """

    __slots__ = ()
    __iter__ = None

    def _sides(self) -> tuple[Mapping[str, C], Mapping[str, M]]:
        raise NotImplementedError

    def __getitem__(self, name: str) -> C | M:
        """Return the column or metadata entry called `name`, as stored.

        Raises `KeyError` when neither side has it, and
        `oxyz.AmbiguousNameError` when both do.
        """
        if not isinstance(name, str):
            raise TypeError(
                f"{type(self).__name__} keys are column and metadata names "
                f"(str), not {type(name).__name__}"
            )
        columns, metadata = self._sides()
        if name in columns:
            if name in metadata:
                raise AmbiguousNameError(
                    f"{name!r} is both a column and a metadata key of this "
                    f"{type(self).__name__}; look it up in .columns or .metadata",
                    name=name,
                )
            return columns[name]
        if name in metadata:
            return metadata[name]
        raise KeyError(name)

    def __contains__(self, name: object) -> bool:
        """Report whether `name` is a column or a metadata key. Never raises."""
        columns, metadata = self._sides()
        return name in columns or name in metadata

    @overload
    def get(self, name: str) -> C | M | None: ...
    @overload
    def get[D](self, name: str, default: D) -> C | M | D: ...
    def get(self, name: str, default: object = None) -> object:
        """Return `self[name]`, or `default` when neither side has `name`.

        A name on both sides still raises `oxyz.AmbiguousNameError`: it is
        ambiguous, not absent.
        """
        try:
            return self[name]
        except KeyError:
            return default

    def keys(self) -> list[str]:
        """Column names, then metadata names, each once, in stored order."""
        columns, metadata = self._sides()
        return list(dict.fromkeys([*columns, *metadata]))
