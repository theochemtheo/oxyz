from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import atomflow._rust as _rust


@dataclass(frozen=True, slots=True)
class FrameIndex:
    """Structural facts from a scan: frame offsets and declared atom counts.

    Nothing is parsed beyond the count lines, so this is cheap even for files
    where a full read is not. Statistics are derived from the stored counts;
    `std_atoms` is the population standard deviation. All statistics are
    None for an empty file.
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

    @property
    def mean_atoms(self) -> float | None:
        return float(self.n_atoms.mean()) if self.n_frames else None

    @property
    def median_atoms(self) -> float | None:
        return float(np.median(self.n_atoms)) if self.n_frames else None

    @property
    def std_atoms(self) -> float | None:
        return float(self.n_atoms.std()) if self.n_frames else None


def scan(path: str | Path) -> FrameIndex:
    """Scan a file's structure without parsing any frame contents."""
    data = _rust.scan(str(path))
    return FrameIndex(offsets=data["offsets"], n_atoms=data["n_atoms"])
