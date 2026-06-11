"""ASE's own database: `ase.db.connect` on SQLite, core ASE, no extras.

The common ecosystem path for "I have Atoms, I want a queryable store".
Ingest commits row by row (that is how ase.db writes), so building the
cache the first time takes minutes; it only happens once.
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

    # Explicit type: the build path ends in .tmp, which connect() cannot
    # infer a backend from.
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
