from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import oxyz._rust as _rust
from oxyz._stats import AtomCountStats


@dataclass(frozen=True, slots=True)
class FrameIndex(AtomCountStats):
    """Structural facts from a scan: frame offsets and declared atom counts.

    Nothing is parsed beyond the count lines, so this is cheap even for files
    where a full read is not. Statistics are derived from the stored counts;
    `mean_atoms`/`median_atoms`/`std_atoms` come from `AtomCountStats`, with
    `std_atoms` the population standard deviation. All statistics are None for
    an empty file.
    """

    offsets: np.ndarray
    n_atoms: np.ndarray

    @property
    def n_frames(self) -> int:
        return len(self.n_atoms)

    @property
    def total_atoms(self) -> int:
        return int(self.n_atoms.sum())

    @property
    def min_atoms(self) -> int | None:
        return int(self.n_atoms.min()) if self.n_frames else None

    @property
    def max_atoms(self) -> int | None:
        return int(self.n_atoms.max()) if self.n_frames else None


def scan(path: str | Path) -> FrameIndex:
    """Scan a file's structure without parsing any frame contents.

    Returns a `FrameIndex` of per-frame byte offsets and declared atom counts.
    `n_atoms` is `intp` so arithmetic with it does not promote to float64.
    The atom-count statistics (`min_atoms`/`max_atoms`/`mean_atoms`/
    `median_atoms`/`std_atoms`) are `None` for an empty file — the only
    optionals in the result.
    """
    data = _rust.scan(str(path))
    return FrameIndex(offsets=data["offsets"], n_atoms=data["n_atoms"])
