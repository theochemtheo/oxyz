"""atompack: the specialised binary molecule store (pip: atompack-db).

`serial` loops `get_molecule(i)`; `native` calls `get_molecules_flat`, the
mmap-backed batch path on all cores. The pool size is fixed per process by
RAYON_NUM_THREADS. Positions and forces are stored as float32.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

import oxyz

from .common import build_once, frame_record

INGEST_CHUNK = 1024


def ensure(src: Path) -> Path:
    return build_once(src.with_suffix(".atp"), lambda tmp: _ingest(src, tmp))


def _ingest(src: Path, dest: Path) -> None:
    import atompack  # ty: ignore[unresolved-import]

    db = atompack.Database(str(dest), overwrite=True)
    chunk = []
    for frame in oxyz.iread(src):
        record = frame_record(frame)
        chunk.append(
            atompack.Molecule.from_arrays(
                record["positions"].astype(np.float32),
                record["numbers"],
                energy=record["energy"],
                forces=record["forces"].astype(np.float32),
                cell=record["cell"],
                pbc=tuple(bool(p) for p in record["pbc"]),
            )
        )
        if len(chunk) == INGEST_CHUNK:
            db.add_molecules(chunk)
            chunk = []
    if chunk:
        db.add_molecules(chunk)
    db.flush()


def open_db(path: Path):
    import atompack  # ty: ignore[unresolved-import]

    # mmap-backed read-only handle: the mode whose batch reads parallelise.
    return atompack.Database.open(str(path))


def read_serial(db, indices: Sequence[int]) -> int:
    """Per-item gets, arrays decoded — atompack's own published pattern."""
    total = 0
    for i in indices:
        molecule = db.get_molecule(i)
        arrays = (molecule.positions, molecule.forces, molecule.energy)
        total += len(arrays[0])
    return total


def read_flat(db, indices: Sequence[int]) -> int:
    """One batched read returning stacked arrays."""
    flat = db.get_molecules_flat(list(indices))
    return len(flat["positions"])
