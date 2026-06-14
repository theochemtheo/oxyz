from __future__ import annotations

import numpy as np


class AtomCountStats:
    """Distribution statistics over a per-frame atom-count array.

    Mixed into the scan and schema results, which both carry an `n_atoms`
    array, so the two report identical numbers from one definition. The mean,
    median, and population standard deviation are derived on demand; all are
    None for an empty file. `min`/`max`/`total` are left to each host, which
    already has them.
    """

    __slots__ = ()
    n_atoms: np.ndarray

    @property
    def mean_atoms(self) -> float | None:
        return float(self.n_atoms.mean()) if self.n_atoms.size else None

    @property
    def median_atoms(self) -> float | None:
        return float(np.median(self.n_atoms)) if self.n_atoms.size else None

    @property
    def std_atoms(self) -> float | None:
        return float(self.n_atoms.std()) if self.n_atoms.size else None
