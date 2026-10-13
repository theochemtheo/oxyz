"""Lookup by name across columns, then metadata; a name on both is an error."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

from oxyz._rust import OxyzError

if TYPE_CHECKING:
    from collections.abc import Mapping


class AmbiguousNameError(OxyzError):
    """A name is both a column and a metadata key, so a lookup cannot choose.

    Not a `KeyError`, so `get` and `except KeyError` cannot mistake it for
    absence. `name in frame` never raises it.

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

    Not iterable and without `__len__`, so numpy treats a list of hosts as
    hosts, not as key names.
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

        A name on both sides still raises `oxyz.AmbiguousNameError`.
        """
        try:
            return self[name]
        except KeyError:
            return default

    def keys(self) -> list[str]:
        """Column names, then metadata names, each once, in stored order."""
        columns, metadata = self._sides()
        return list(dict.fromkeys([*columns, *metadata]))
