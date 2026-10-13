"""ASE's own database: `ase.db.connect` on SQLite.

Ingest commits row by row, so the first build takes minutes.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import oxyz.ase


def ensure(src: Path) -> Path:
    from .common import build_once

    return build_once(src.with_suffix(".db"), lambda tmp: _ingest(src, tmp))


def _ingest(src: Path, dest: Path) -> None:
    from ase.db import connect

    # The .tmp build path gives connect() nothing to infer a backend from.
    db = connect(dest, type="db")
    for atoms in oxyz.ase.iread(src):
        db.write(atoms)


def open_db(path: Path):
    from ase.db import connect

    return connect(path)


def read_all(db) -> int:
    """Full select, materialising Atoms row by row."""
    return sum(len(row.toatoms()) for row in db.select())


def read_indices(db, indices: Sequence[int]) -> int:
    # ase.db ids are 1-based and assigned in insertion order.
    return sum(len(db.get(id=i + 1).toatoms()) for i in indices)
