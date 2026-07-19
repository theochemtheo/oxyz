"""Writing frames back out: the unified `write` and the incremental `Writer`.

Both accept a `Frame`, an `ase.Atoms`, or an iterable mixing them, and dispatch
per item — a `Frame` passes straight through; anything else is routed to a
lazily-imported converter, so the base install never imports ASE. Serialisation,
ordering, and compression all live in the Rust core; this layer only marshals.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Self, cast

import numpy as np

import oxyz._rust as _rust
from oxyz._frames import ColumnValues, Compression, Frame, MetadataValue

if TYPE_CHECKING:
    from pathlib import Path

    from ase import Atoms

# A public type alias for a single item `write`/`Writer.write` accept: a `Frame`
# or an `ase.Atoms`. Both functions also accept an `Iterable[Writable]` mixing
# the two. Declared with PEP 695 `type`, so the right-hand side is evaluated
# lazily — exporting `oxyz.Writable` does not import ase, and `type` statements
# carry no docstring slot, hence this comment rather than a docstring.
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

    Each item of `obj` is converted independently. Columns are written
    `species`, `pos`, then the rest; the comment line is `Lattice`, `pbc`,
    `Properties`, then remaining metadata. Reals use the shortest
    round-trippable form, so `read` then `write` is lossless. A frame without
    both a `species` and a `pos` column is rejected.

    Parameters
    ----------
    path : str or Path
        Output location. The codec follows the extension — `.xyz`, `.extxyz`,
        and the compressed forms `.gz`/`.zip`/`.tar`/`.tar.gz` — unless
        `compression` overrides it. `"-"` writes to stdout.
    obj : Writable or iterable of Writable
        A single `Frame`/`ase.Atoms`, or an iterable mixing them.
    append : bool, optional
        Add to an existing file, for the formats that allow it (plain,
        gzip). Rejected for the archive codecs and for stdout.
    compression : Compression, optional
        Codec, overriding the one inferred from `path`. Writing `.zst` is
        not yet supported.
    level : int, optional
        Deflate level, `0..=9`, for the compressed codecs.
    threads : int, optional
        Workers to serialise frames across. `None` (the default) uses every
        core; `1` is serial. Output bytes are identical either way; the cost
        is peak memory, since frames are serialised before being written.

    Examples
    --------
    >>> import tempfile, pathlib, oxyz
    >>> frames = oxyz.read("examples/data/water.extxyz")
    >>> with tempfile.TemporaryDirectory() as d:
    ...     out = pathlib.Path(d) / "out.extxyz"
    ...     oxyz.write(out, frames)
    ...     len(oxyz.read(out))
    3
    """
    payloads = [_payload(item) for item in _items(obj)]
    _rust.write(str(path), payloads, compression, level, append, threads)


class Writer:
    """Write frames incrementally, in constant memory for the streaming codecs.

    A context manager: entering opens the encoder, `write` accepts frames as
    they are produced, and exiting (or calling `close`) finalises and closes
    the file. Options match `write`; dispatch is the same, so `.write` also
    takes a single `Frame`/`ase.Atoms` or an iterable mixing them.

    Examples
    --------
    >>> import tempfile, pathlib, oxyz
    >>> frames = oxyz.read("examples/data/water.extxyz")
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
            Add to an existing file, for the formats that allow it (plain,
            gzip). Rejected for the archive codecs and for stdout.
        compression : Compression, optional
            Codec, overriding the one inferred from `path`.
        level : int, optional
            Deflate level, `0..=9`, for the compressed codecs.
        batch : int, optional
            Buffer `batch` frames and serialise each full batch in parallel,
            trading a little memory for throughput. `None` (the default)
            streams each frame out in constant memory instead. The output is
            identical either way; peak extra memory is one batch.
        """
        self._inner = _rust.FrameWriter(str(path), compression, level, append, batch)

    def write(self, obj: Writable | Iterable[Writable]) -> None:
        """Write `obj` to the file.

        Parameters
        ----------
        obj : Writable or iterable of Writable
            A single `Frame`/`ase.Atoms`, or an iterable mixing them; each
            item is converted and written independently.
        """
        for item in _items(obj):
            self._inner.write(_payload(item))

    def close(self) -> None:
        """Finalise the encoder and close the file. Idempotent."""
        self._inner.close()

    def __enter__(self) -> Self:
        """Return `self`; writing happens via `write`."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Finalise and close the file, via `close`."""
        self.close()


def _items(obj: Writable | Iterable[Writable]) -> list[Writable]:
    """Normalise `obj` to a list of writables.

    A single writable becomes a one-item list; any other iterable is consumed
    in order. `ase.Atoms` is itself iterable (over its atoms), so it must be
    recognised as a single item before the iterable check.
    """
    if isinstance(obj, Frame) or _looks_like(obj, "ase.atoms", "Atoms"):
        return [cast("Writable", obj)]
    if isinstance(obj, Iterable):
        return list(obj)
    # Not writable and not iterable: defer to _payload for a clear error.
    return [cast("Writable", obj)]


def _payload(obj: Writable) -> _rust.FrameData:
    """Convert `obj` to the payload shape the Rust binding writes.

    Raises `TypeError` for anything that is neither a `Frame` nor an
    `ase.Atoms`.
    """
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
    """Convert `frame`'s columns and metadata to their canonical dtypes."""
    return {
        "n_atoms": frame.n_atoms,
        "columns": {
            name: _canonical_column(values) for name, values in frame.columns.items()
        },
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
    """Coerce `array` to the dtype the binding moves without copying.

    Floats become float64, ints become int64, and bool passes through; a
    string array becomes a (nested) list, the form string columns cross in.
    `asarray` skips the copy when the dtype already matches, the common case
    for reader-produced frames.
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
    """Report whether `obj`'s type, or any base, is `module.name`.

    An isinstance check that does not import the module, so a non-ASE object
    never imports ASE.
    """
    return any(
        base.__module__ == module and base.__name__ == name
        for base in type(obj).__mro__
    )
