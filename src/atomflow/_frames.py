from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import atomflow._rust as _rust

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


def read_first_frame(path: str | Path) -> Frame:
    data = _rust.read_first_frame(str(path))

    return Frame(
        n_atoms=data["n_atoms"],
        columns=data["columns"],
        metadata=data["metadata"],
    )
