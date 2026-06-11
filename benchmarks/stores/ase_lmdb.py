"""Hand-rolled LMDB of pickled `ase.Atoms` records, one per key.

The pattern behind fairchem's `.aselmdb` files, without the fairchem
(and so torch) dependency: this is what people deploy when they want
Atoms back out of LMDB.
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path

import oxyz.ase

from .common import build_once, key

MAP_SIZE = 8 << 30  # sparse until written


def ensure(src: Path) -> Path:
    return build_once(src.with_suffix(".aselmdb"), lambda tmp: _ingest(src, tmp))


def _ingest(src: Path, dest: Path) -> None:
    import lmdb  # ty: ignore[unresolved-import]

    env = lmdb.open(str(dest), map_size=MAP_SIZE, subdir=False)
    with env.begin(write=True) as txn:
        for i, atoms in enumerate(oxyz.ase.iread(src)):
            txn.put(key(i), pickle.dumps(atoms, protocol=5))
    env.sync()
    env.close()


def open_env(path: Path):
    import lmdb  # ty: ignore[unresolved-import]

    return lmdb.open(str(path), subdir=False, readonly=True, lock=False)


def read_all(env) -> int:
    total = 0
    with env.begin(buffers=True) as txn:
        for _, value in txn.cursor():
            total += len(pickle.loads(value))
    return total


def read_indices(env, indices: Sequence[int]) -> int:
    total = 0
    with env.begin(buffers=True) as txn:
        for i in indices:
            total += len(pickle.loads(txn.get(key(i))))
    return total
