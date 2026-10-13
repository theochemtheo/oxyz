from __future__ import annotations

import numpy as np


class AtomCountStats:
    """Distribution statistics over a per-frame atom-count array.

    Shared by the scan and schema results; all are None for an empty file.

    Attributes
    ----------
    n_atoms
        Per-frame atom count, supplied by the host dataclass.
    """

    __slots__ = ()
    n_atoms: np.ndarray

    @property
    def mean_atoms(self) -> float | None:
        """Arithmetic mean of `n_atoms`, or None if empty."""
        return float(self.n_atoms.mean()) if self.n_atoms.size else None

    @property
    def median_atoms(self) -> float | None:
        """Median of `n_atoms`, or None if empty."""
        return float(np.median(self.n_atoms)) if self.n_atoms.size else None

    @property
    def std_atoms(self) -> float | None:
        """Population standard deviation of `n_atoms`, or None if empty."""
        return float(self.n_atoms.std()) if self.n_atoms.size else None
