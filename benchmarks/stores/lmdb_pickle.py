"""Hand-rolled LMDB store: one pickled dict of numpy arrays per frame.

Not a library — the pattern dataloader repositories write themselves
when extxyz parsing starts to hurt. Kept deliberately plain: no
compression, default pickles, one key per frame.
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path

import oxyz

from .common import build_once, frame_record, key

MAP_SIZE = 8 << 30  # sparse until written


def ensure(src: Path) -> Path:
    return build_once(src.with_suffix(".lmdb"), lambda tmp: _ingest(src, tmp))


def _ingest(src: Path, dest: Path) -> None:
    import lmdb  # ty: ignore[unresolved-import]

    env = lmdb.open(str(dest), map_size=MAP_SIZE, subdir=False)
    with env.begin(write=True) as txn:
        for i, frame in enumerate(oxyz.iter_frames(src)):
            txn.put(key(i), pickle.dumps(frame_record(frame), protocol=5))
    env.sync()
    env.close()


def open_env(path: Path):
    import lmdb  # ty: ignore[unresolved-import]

    return lmdb.open(str(path), subdir=False, readonly=True, lock=False)


def read_all(env) -> int:
    """Cursor pass over every frame, unpickling each record."""
    total = 0
    with env.begin(buffers=True) as txn:
        for _, value in txn.cursor():
            record = pickle.loads(value)
            total += len(record["numbers"])
    return total


def read_indices(env, indices: Sequence[int]) -> int:
    total = 0
    with env.begin(buffers=True) as txn:
        for i in indices:
            record = pickle.loads(txn.get(key(i)))
            total += len(record["numbers"])
    return total
