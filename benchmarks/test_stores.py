"""oxyz against database-style stores, on one 100k-frame dataset.

The reader-vs-reader suite (test_read.py) compares parsers on the same
text input. This suite asks a different question: how does parsing
extxyz text compare with reading from a binary store that was converted
up front? The stores hold pre-decoded arrays and an index; oxyz gets the
text file and nothing else. oxyz losing here is expected — the margin is
the result.

One stated reason per baseline:

- atompack: the specialised binary molecule store (mmap, rayon).
- lmdb-pickle: the hand-rolled LMDB-of-arrays pattern dataloader
  repositories write themselves.
- ase-sqlite / ase-lmdb: the common ASE ecosystem paths, queryable
  database and pickled-Atoms LMDB.

Groups are split by output contract: `stores/...` rows hand back numpy
arrays (oxyz batches, atompack arrays, unpickled dicts), `stores-ase/...`
rows materialise ase.Atoms. The two are never compared in one table.
Every row streams or gathers its frames and returns a total atom count.

Stores are ingested once into benchmarks/.cache/ during fixture setup
(ase.db commits per row, so the first run takes minutes); ingest is
setup, never a benchmark.
"""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import pytest
from conftest import STORE_ATOMS_PER_FRAME, STORE_N_FRAMES
from test_read import needs_ase, row, run

import oxyz

needs_atompack = pytest.mark.skipif(
    importlib.util.find_spec("atompack") is None, reason="atompack-db not installed"
)

needs_lmdb = pytest.mark.skipif(
    importlib.util.find_spec("lmdb") is None, reason="lmdb not installed"
)

needs_ase_lmdb = pytest.mark.skipif(
    importlib.util.find_spec("lmdb") is None or importlib.util.find_spec("ase") is None,
    reason="lmdb or ase not installed",
)

TOTAL_ATOMS = STORE_N_FRAMES * STORE_ATOMS_PER_FRAME

# The access patterns. Shuffled draws a seeded sample without replacement
# (a dataloader epoch slice); strided is the classic every-Nth subsample.
SHUFFLED_INDICES = random.Random(0x5A3D).sample(range(STORE_N_FRAMES), 2048)
STRIDED_INDICES = range(0, STORE_N_FRAMES, 50)

FULL_SHAPE = (STORE_N_FRAMES, TOTAL_ATOMS)
SHUFFLED_SHAPE = (len(SHUFFLED_INDICES), len(SHUFFLED_INDICES) * STORE_ATOMS_PER_FRAME)
STRIDED_SHAPE = (len(STRIDED_INDICES), len(STRIDED_INDICES) * STORE_ATOMS_PER_FRAME)


# -- store handles, built once per session (ingest happens here) ----------


@pytest.fixture(scope="session")
def atompack_db(store_dataset: Path):
    from stores import atompack

    return atompack.open_db(atompack.ensure(store_dataset))


@pytest.fixture(scope="session")
def lmdb_env(store_dataset: Path):
    from stores import lmdb_pickle

    return lmdb_pickle.open_env(lmdb_pickle.ensure(store_dataset))


@pytest.fixture(scope="session")
def ase_sqlite_db(store_dataset: Path):
    from stores import ase_sqlite

    return ase_sqlite.open_db(ase_sqlite.ensure(store_dataset))


@pytest.fixture(scope="session")
def ase_lmdb_env(store_dataset: Path):
    from stores import ase_lmdb

    return ase_lmdb.open_env(ase_lmdb.ensure(store_dataset))


# -- arrays-out rows -------------------------------------------------------


@row("Batch", "parallel")
def oxyz_sequential(path: Path) -> int:
    total = 0
    for batch in oxyz.iter_batches(path, frames_per_batch=1024):
        total += batch.total_atoms
    return total


@row("Batch", "parallel")
def oxyz_shuffled(path: Path) -> int:
    return oxyz.read_batch(path, SHUFFLED_INDICES).total_atoms


@row("Batch", "parallel")
def oxyz_strided(path: Path) -> int:
    return oxyz.read_batch(path, STRIDED_INDICES).total_atoms


@row("numpy arrays", "serial")
def atompack_serial_sequential(db) -> int:
    from stores import atompack

    return atompack.read_serial(db, range(STORE_N_FRAMES))


@row("numpy arrays", "parallel")
def atompack_native_sequential(db) -> int:
    from stores import atompack

    return atompack.read_flat(db, range(STORE_N_FRAMES))


@row("numpy arrays", "serial")
def atompack_serial_shuffled(db) -> int:
    from stores import atompack

    return atompack.read_serial(db, SHUFFLED_INDICES)


@row("numpy arrays", "parallel")
def atompack_native_shuffled(db) -> int:
    from stores import atompack

    return atompack.read_flat(db, SHUFFLED_INDICES)


@row("numpy arrays", "serial")
def atompack_serial_strided(db) -> int:
    from stores import atompack

    return atompack.read_serial(db, STRIDED_INDICES)


@row("numpy arrays", "parallel")
def atompack_native_strided(db) -> int:
    from stores import atompack

    return atompack.read_flat(db, STRIDED_INDICES)


@row("numpy arrays", "serial")
def lmdb_pickle_sequential(env) -> int:
    from stores import lmdb_pickle

    return lmdb_pickle.read_all(env)


@row("numpy arrays", "serial")
def lmdb_pickle_shuffled(env) -> int:
    from stores import lmdb_pickle

    return lmdb_pickle.read_indices(env, SHUFFLED_INDICES)


@row("numpy arrays", "serial")
def lmdb_pickle_strided(env) -> int:
    from stores import lmdb_pickle

    return lmdb_pickle.read_indices(env, STRIDED_INDICES)


# -- ase.Atoms-out rows ----------------------------------------------------


@row("ase.Atoms", "serial")
def oxyz_to_ase_sequential(path: Path) -> int:
    import oxyz.ase

    return sum(len(atoms) for atoms in oxyz.ase.iread(path))


@row("ase.Atoms", "serial")
def oxyz_to_ase_strided(path: Path) -> int:
    import oxyz.ase

    return sum(len(atoms) for atoms in oxyz.ase.iread(path, "::50"))


@row("ase.Atoms", "serial")
def ase_sqlite_sequential(db) -> int:
    from stores import ase_sqlite

    return ase_sqlite.read_all(db)


@row("ase.Atoms", "serial")
def ase_sqlite_shuffled(db) -> int:
    from stores import ase_sqlite

    return ase_sqlite.read_indices(db, SHUFFLED_INDICES)


@row("ase.Atoms", "serial")
def ase_sqlite_strided(db) -> int:
    from stores import ase_sqlite

    return ase_sqlite.read_indices(db, STRIDED_INDICES)


@row("ase.Atoms", "serial")
def ase_lmdb_sequential(env) -> int:
    from stores import ase_lmdb

    return ase_lmdb.read_all(env)


@row("ase.Atoms", "serial")
def ase_lmdb_shuffled(env) -> int:
    from stores import ase_lmdb

    return ase_lmdb.read_indices(env, SHUFFLED_INDICES)


@row("ase.Atoms", "serial")
def ase_lmdb_strided(env) -> int:
    from stores import ase_lmdb

    return ase_lmdb.read_indices(env, STRIDED_INDICES)


# -- the benchmarks --------------------------------------------------------


SEQUENTIAL_ARRAYS = [
    pytest.param(oxyz_sequential, "store_dataset", id="oxyz-batches"),
    pytest.param(
        atompack_serial_sequential,
        "atompack_db",
        id="atompack-serial",
        marks=needs_atompack,
    ),
    pytest.param(
        atompack_native_sequential,
        "atompack_db",
        id="atompack-native",
        marks=needs_atompack,
    ),
    pytest.param(
        lmdb_pickle_sequential, "lmdb_env", id="lmdb-pickle", marks=needs_lmdb
    ),
]

SHUFFLED_ARRAYS = [
    pytest.param(oxyz_shuffled, "store_dataset", id="oxyz-read-batch"),
    pytest.param(
        atompack_serial_shuffled,
        "atompack_db",
        id="atompack-serial",
        marks=needs_atompack,
    ),
    pytest.param(
        atompack_native_shuffled,
        "atompack_db",
        id="atompack-native",
        marks=needs_atompack,
    ),
    pytest.param(lmdb_pickle_shuffled, "lmdb_env", id="lmdb-pickle", marks=needs_lmdb),
]

STRIDED_ARRAYS = [
    pytest.param(oxyz_strided, "store_dataset", id="oxyz-read-batch"),
    pytest.param(
        atompack_serial_strided,
        "atompack_db",
        id="atompack-serial",
        marks=needs_atompack,
    ),
    pytest.param(
        atompack_native_strided,
        "atompack_db",
        id="atompack-native",
        marks=needs_atompack,
    ),
    pytest.param(lmdb_pickle_strided, "lmdb_env", id="lmdb-pickle", marks=needs_lmdb),
]

SEQUENTIAL_ASE = [
    pytest.param(
        oxyz_to_ase_sequential, "store_dataset", id="oxyz-to-ase", marks=needs_ase
    ),
    pytest.param(
        ase_sqlite_sequential, "ase_sqlite_db", id="ase-sqlite", marks=needs_ase
    ),
    pytest.param(
        ase_lmdb_sequential, "ase_lmdb_env", id="ase-lmdb", marks=needs_ase_lmdb
    ),
]

# No oxyz row: gathering an arbitrary index list as Atoms has no public
# oxyz spelling yet; oxyz's shuffled story is the arrays-out group above.
SHUFFLED_ASE = [
    pytest.param(
        ase_sqlite_shuffled, "ase_sqlite_db", id="ase-sqlite", marks=needs_ase
    ),
    pytest.param(
        ase_lmdb_shuffled, "ase_lmdb_env", id="ase-lmdb", marks=needs_ase_lmdb
    ),
]

STRIDED_ASE = [
    pytest.param(
        oxyz_to_ase_strided, "store_dataset", id="oxyz-to-ase", marks=needs_ase
    ),
    pytest.param(ase_sqlite_strided, "ase_sqlite_db", id="ase-sqlite", marks=needs_ase),
    pytest.param(ase_lmdb_strided, "ase_lmdb_env", id="ase-lmdb", marks=needs_ase_lmdb),
]


@pytest.mark.benchmark(group="stores/sequential", min_rounds=2)
@pytest.mark.parametrize("read,source", SEQUENTIAL_ARRAYS)
def test_sequential_full_pass(benchmark, read, source, request):
    total = run(benchmark, read, request.getfixturevalue(source), shape=FULL_SHAPE)
    assert total == TOTAL_ATOMS


@pytest.mark.benchmark(group="stores/shuffled")
@pytest.mark.parametrize("read,source", SHUFFLED_ARRAYS)
def test_shuffled_sample(benchmark, read, source, request):
    total = run(benchmark, read, request.getfixturevalue(source), shape=SHUFFLED_SHAPE)
    assert total == SHUFFLED_SHAPE[1]


@pytest.mark.benchmark(group="stores/strided")
@pytest.mark.parametrize("read,source", STRIDED_ARRAYS)
def test_strided_read(benchmark, read, source, request):
    total = run(benchmark, read, request.getfixturevalue(source), shape=STRIDED_SHAPE)
    assert total == STRIDED_SHAPE[1]


@pytest.mark.benchmark(group="stores-ase/sequential", min_rounds=2)
@pytest.mark.parametrize("read,source", SEQUENTIAL_ASE)
def test_sequential_full_pass_to_ase(benchmark, read, source, request):
    total = run(benchmark, read, request.getfixturevalue(source), shape=FULL_SHAPE)
    assert total == TOTAL_ATOMS


@pytest.mark.benchmark(group="stores-ase/shuffled")
@pytest.mark.parametrize("read,source", SHUFFLED_ASE)
def test_shuffled_sample_to_ase(benchmark, read, source, request):
    total = run(benchmark, read, request.getfixturevalue(source), shape=SHUFFLED_SHAPE)
    assert total == SHUFFLED_SHAPE[1]


@pytest.mark.benchmark(group="stores-ase/strided")
@pytest.mark.parametrize("read,source", STRIDED_ASE)
def test_strided_read_to_ase(benchmark, read, source, request):
    total = run(benchmark, read, request.getfixturevalue(source), shape=STRIDED_SHAPE)
    assert total == STRIDED_SHAPE[1]
