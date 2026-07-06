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

    Writable = Frame | Atoms

__all__ = ["Writer", "write"]


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

    `obj` is a single `Frame`/`ase.Atoms` or an iterable mixing them; each item
    is converted independently. The codec follows the extension — `.xyz`,
    `.extxyz`, and the compressed forms `.gz`/`.zip`/`.tar`/`.tar.gz` — or is
    forced by `compression`. `path="-"` writes to stdout.

    Columns are written `species`, `pos`, then the rest; the comment line is
    `Lattice`, `pbc`, `Properties`, then remaining metadata. Reals use the
    shortest round-trippable form, so `read` then `write` is lossless. A frame
    without both a `species` and a `pos` column is rejected.

    `level` (`0..=9`) tunes the deflate codecs; `append=True` adds to an existing
    file for the formats that allow it (plain, gzip), and is rejected for the
    archive codecs and for stdout. Writing `.zst` is not yet supported.

    `threads` spreads serialisation over workers — `None` (the default) uses
    every core, `1` is serial. The output bytes are identical either way; the
    cost is peak memory, since the frames are serialised before being written.
    """
    payloads = [_payload(item) for item in _items(obj)]
    _rust.write(str(path), payloads, compression, level, append, threads)


class Writer:
    """Incremental writer: `write` frames as they are produced, in constant
    memory for the streaming codecs. Use as a context manager so the encoder is
    finalised on exit.

    Options match `write`; the dispatch is the same, so `.write` also takes a
    single `Frame`/`ase.Atoms` or an iterable mixing them.

    `batch` trades a little memory for serialisation throughput: with
    `batch=None` (the default) each frame streams straight out in constant
    memory; with `batch=n` frames are buffered `n` at a time and each full batch
    is serialised in parallel before being written. The output is identical;
    peak extra memory is one batch.
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
        self._inner = _rust.FrameWriter(str(path), compression, level, append, batch)

    def write(self, obj: Writable | Iterable[Writable]) -> None:
        for item in _items(obj):
            self._inner.write(_payload(item))

    def close(self) -> None:
        """Finalise the encoder and close the file. Idempotent."""
        self._inner.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _items(obj: Writable | Iterable[Writable]) -> list[Writable]:
    """A single writable becomes a one-item list; any other iterable is consumed
    in order. `ase.Atoms` is itself iterable (over its atoms), so it must be
    recognised as a single item before the iterable check."""
    if isinstance(obj, Frame) or _looks_like(obj, "ase.atoms", "Atoms"):
        return [cast("Writable", obj)]
    if isinstance(obj, Iterable):
        return list(obj)
    # Not writable and not iterable: defer to _payload for a clear error.
    return [cast("Writable", obj)]


def _payload(obj: Writable) -> _rust.FrameData:
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
    return _canonical_array(values) if isinstance(values, np.ndarray) else values


def _canonical_meta(value: MetadataValue) -> MetadataValue:
    return _canonical_array(value) if isinstance(value, np.ndarray) else value


def _canonical_array(array: np.ndarray) -> np.ndarray | list:
    """Coerce a numpy array to the dtype the binding moves without copying —
    float64 / int64 / bool — and turn a string array into a (nested) list, the
    form string columns cross in. `asarray` skips the copy when the dtype already
    matches, the common case for reader-produced frames."""
    kind = array.dtype.kind
    if kind == "f":
        return np.asarray(array, dtype=np.float64)
    if kind in "iu":
        return np.asarray(array, dtype=np.int64)
    if kind in "US":
        return array.tolist()
    return array


def _looks_like(obj: object, module: str, name: str) -> bool:
    """Whether `obj`'s type, or any base, is `module.name` — an isinstance check
    that does not import the module (so a non-ASE object never imports ASE)."""
    return any(
        base.__module__ == module and base.__name__ == name
        for base in type(obj).__mro__
    )
