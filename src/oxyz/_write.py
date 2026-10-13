"""Writing frames: the unified `write` and the incremental `Writer`."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Self, cast

import numpy as np

from oxyz import _rust
from oxyz._convert import FieldError
from oxyz._frames import ColumnValues, Compression, Frame, MetadataValue

if TYPE_CHECKING:
    from pathlib import Path

    from ase import Atoms

# A single item `write` and `Writer.write` accept. A `type` statement keeps
# ase unimported and has no docstring slot.
type Writable = Frame | Atoms

__all__ = ["Writable", "Writer", "write"]


def write(
    path: str | Path,
    obj: Writable | Iterable[Writable],
    *,
    append: bool = False,
    compression: Compression = "infer",
    level: int | None = None,
    threads: int | None = None,
) -> None:
    """Write frames to `path`, encoding from the path's extension (overridable).

    Columns are written `species`, `pos`, then the rest; the comment line
    `Lattice`, `pbc`, `Properties`, then the remaining metadata. Reals use the
    shortest round-trippable form, so `read` then `write` is lossless. A frame
    without `species` and `pos` columns is rejected.

    Parameters
    ----------
    path : str or Path
        Output location. The codec follows the extension (`.xyz`, `.extxyz`,
        `.gz`, `.zip`, `.tar`, `.tar.gz`) unless `compression` overrides it.
        `"-"` writes to stdout.
    obj : Writable or iterable of Writable
        A single `Frame`/`ase.Atoms`, or an iterable mixing them.
    append : bool, optional
        Add to an existing plain or gzip file. Rejected for archives and
        stdout.
    compression : Compression, optional
        Codec, overriding the one inferred from `path`. zstd cannot be
        written.
    level : int, optional
        Deflate level, `0..=9`, for the compressed codecs.
    threads : int, optional
        Workers to serialise frames across; `None` (the default) uses every
        core, `1` is serial. The bytes are identical; parallel serialisation
        holds every frame's text in memory before writing.

    Examples
    --------
    >>> import tempfile, pathlib, oxyz
    >>> frames = oxyz.read(DATA / "water.extxyz")
    >>> with tempfile.TemporaryDirectory() as d:
    ...     out = pathlib.Path(d) / "out.extxyz"
    ...     oxyz.write(out, frames)
    ...     len(oxyz.read(out))
    3
    """
    payloads = [_payload(item) for item in _items(obj)]
    _rust.write(str(path), payloads, compression, level, append, threads)


class Writer:
    """Write frames incrementally, in bounded memory for the streaming codecs.

    A context manager; exiting, or `close`, finalises the file. Options and
    accepted items match `write`.

    Examples
    --------
    >>> import tempfile, pathlib, oxyz
    >>> frames = oxyz.read(DATA / "water.extxyz")
    >>> with tempfile.TemporaryDirectory() as d:
    ...     out = pathlib.Path(d) / "out.extxyz"
    ...     with oxyz.Writer(out) as writer:
    ...         for frame in frames:
    ...             writer.write(frame)
    ...     len(oxyz.read(out))
    3
    """

    def __init__(
        self,
        path: str | Path,
        *,
        append: bool = False,
        compression: Compression = "infer",
        level: int | None = None,
        batch: int | None = None,
    ) -> None:
        """Open `path` for incremental writing.

        Parameters
        ----------
        path : str or Path
            Output location. The codec follows the extension unless
            `compression` overrides it.
        append : bool, optional
            Add to an existing plain or gzip file. Rejected for archives and
            stdout.
        compression : Compression, optional
            Codec, overriding the one inferred from `path`.
        level : int, optional
            Deflate level, `0..=9`, for the compressed codecs.
        batch : int, optional
            Buffer `batch` frames and serialise each full batch in parallel.
            `None` (the default) writes each frame as it comes. The output is
            identical.
        """
        self._inner = _rust.FrameWriter(str(path), compression, level, append, batch)
        self._path = str(path)
        options = {
            "append": (append, False),
            "compression": (compression, "infer"),
            "level": (level, None),
            "batch": (batch, None),
        }
        self._options = "".join(
            f", {name}={value!r}"
            for name, (value, default) in options.items()
            if value != default
        )
        self._frames_written = 0
        self._closed = False

    def __repr__(self) -> str:
        """Name the file, its non-default options, and progress so far."""
        closed = ", closed=True" if self._closed else ""
        return (
            f"Writer({self._path!r}{self._options}, "
            f"frames_written={self._frames_written}{closed})"
        )

    def write(self, obj: Writable | Iterable[Writable]) -> None:
        """Write `obj`: a `Frame`/`ase.Atoms`, or an iterable mixing them."""
        for item in _items(obj):
            self._inner.write(_payload(item))
            self._frames_written += 1

    def close(self) -> None:
        """Finalise the encoder and close the file. Idempotent."""
        self._inner.close()
        self._closed = True

    def __enter__(self) -> Self:
        """Return `self`; writing happens via `write`."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Finalise and close the file, via `close`."""
        self.close()


def _items(obj: Writable | Iterable[Writable]) -> list[Writable]:
    """Normalise `obj` to a list of writables.

    `ase.Atoms` iterates over its atoms, so it is matched as one item first.
    """
    if isinstance(obj, Frame) or _looks_like(obj, "ase.atoms", "Atoms"):
        return [cast("Writable", obj)]
    if isinstance(obj, Iterable):
        return list(obj)
    return [cast("Writable", obj)]


def _payload(obj: Writable) -> _rust.FrameData:
    """Convert `obj` to the binding's payload, or raise `TypeError`."""
    if isinstance(obj, Frame):
        return _frame_payload(obj)
    if _looks_like(obj, "ase.atoms", "Atoms"):
        from oxyz.ase import from_atoms

        return _frame_payload(from_atoms(cast("Atoms", obj)))
    raise TypeError(
        f"oxyz.write cannot write {type(obj).__name__!r}; expected a Frame, an "
        "ase.Atoms, or an iterable of them"
    )


def _frame_payload(frame: Frame) -> _rust.FrameData:
    """Convert `frame` to canonical dtypes, checking row counts against `n_atoms`.

    The encoder would panic on a short column and drop atoms from a long one.
    A 0-D value counts as one row.
    """
    columns: dict[str, ColumnValues] = {}
    for name, values in frame.columns.items():
        try:
            rows = 1 if getattr(values, "ndim", 1) == 0 else len(values)
        except TypeError:
            raise FieldError(
                f"column {name!r} is not a per-atom sequence; "
                f"expected an array or a list of strings, got {type(values).__name__}"
            ) from None
        if rows != frame.n_atoms:
            raise FieldError(
                f"column {name!r} has {rows} rows, but the frame declares "
                f"n_atoms={frame.n_atoms}"
            )
        columns[name] = _canonical_column(values)
    return {
        "n_atoms": frame.n_atoms,
        "columns": columns,
        "metadata": {
            key: _canonical_meta(value) for key, value in frame.metadata.items()
        },
    }


def _canonical_column(values: ColumnValues) -> ColumnValues:
    """Coerce `values` via `_canonical_array` if it is an array; else pass through."""
    return _canonical_array(values) if isinstance(values, np.ndarray) else values


def _canonical_meta(value: MetadataValue) -> MetadataValue:
    """Coerce `value` via `_canonical_array` if it is an array; else pass through."""
    return _canonical_array(value) if isinstance(value, np.ndarray) else value


def _canonical_array(array: np.ndarray) -> np.ndarray | list:
    """Coerce `array` to a dtype the binding moves without copying.

    Floats become float64 and ints int64; a string array becomes a list.
    """
    kind = array.dtype.kind
    if kind == "f":
        return np.asarray(array, dtype=np.float64)
    if kind in "iu":
        return np.asarray(array, dtype=np.int64)
    if kind in "US":
        return array.tolist()
    return array


def _looks_like(obj: object, module: str, name: str) -> bool:
    """Report whether `obj`'s type or any base is `module.name`, without importing."""
    return any(
        base.__module__ == module and base.__name__ == name
        for base in type(obj).__mro__
    )
