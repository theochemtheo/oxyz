from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import atomflow._rust as _rust

if TYPE_CHECKING:
    from ase import Atoms

ColumnValues = np.ndarray | list[str] | list[list[str]]
MetadataValue = float | int | bool | str | np.ndarray | list[str]


@dataclass(frozen=True, slots=True)
class Frame:
    """One parsed extxyz frame: per-atom columns plus comment-line metadata.

    Both dicts preserve file order. Column names and metadata values are kept
    exactly as written in the file; aliasing (``force`` vs ``forces``) and
    conversions (Fortran-order ``Lattice`` to a 3x3 cell) belong to a later
    normalisation layer.
    """

    n_atoms: int
    columns: dict[str, ColumnValues]
    metadata: dict[str, MetadataValue]

    def to_ase(self) -> Atoms:
        """Convert to `ase.Atoms` (requires the optional `ase` extra)."""
        from atomflow.ase import to_atoms

        return to_atoms(self)


def read_first_frame(path: str | Path) -> Frame:
    return _frame_from_data(_rust.read_first_frame(str(path)))


def read_frames(path: str | Path, *, threads: int | None = None) -> list[Frame]:
    """Read every frame. Parses on all cores by default; `threads=1` streams
    serially. Results and errors are identical regardless of `threads`."""
    data = _rust.read_frames(str(path), threads)
    return [_frame_from_data(frame) for frame in data]


class IndexedFrames:
    """Random-access reader: scans on open, then reads frames in any order.

    Internal for now — the public random-access surface is `atomflow.scan`
    plus the completed index grammar in `atomflow.ase`.
    """

    def __init__(self, path: str | Path) -> None:
        self._inner = _rust.IndexedFrames(str(path))

    def __len__(self) -> int:
        return len(self._inner)

    def get(self, frame_index: int) -> Frame:
        return _frame_from_data(self._inner.get(frame_index))


def iter_frames(path: str | Path) -> Iterator[Frame]:
    """Stream frames one at a time, in constant memory.

    The file stays open while iterating and closes when the iterator is
    dropped. After a parse error the stream position is untrustworthy, so
    iteration ends: the error is raised once, then StopIteration.
    """
    for data in _rust.FrameIter(str(path)):
        yield _frame_from_data(data)


def _frame_from_data(data: _rust.FrameData) -> Frame:
    return Frame(
        n_atoms=data["n_atoms"],
        columns=data["columns"],
        metadata=data["metadata"],
    )
