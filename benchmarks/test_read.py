"""End-to-end read benchmarks, one row per library per workload.

Record a run: uv run benchmarks/run.py --benchmark-autosave
Smoke only:   uv run benchmarks/run.py --benchmark-disable
Render:       uv run benchmarks/report.py

run.py is a PEP 723 script that supplies its own environment (CPython
3.13, the comparison libraries, a release build of this checkout's oxyz);
conftest refuses debug builds as a backstop. Saves land in .benchmarks/
(gitignored); numbers are only comparable across runs on one machine.

Each reader carries a `@row(output, mode)` tag describing what it
materialises (numpy frames / ase.Atoms / Batch) and whether it parses on
all cores or one; report.py renders those columns straight from the save,
and RESULTS.md is never edited by hand. cextxyz and ase-extxyz rows skip
where the libraries are missing (they publish no CPython 3.14 wheels).
"""

from __future__ import annotations

import functools
import importlib.util
from pathlib import Path

import pytest

import oxyz
from conftest import MAD_N_FRAMES

needs_ase = pytest.mark.skipif(
    importlib.util.find_spec("ase") is None, reason="ase not installed"
)

needs_cextxyz = pytest.mark.skipif(
    importlib.util.find_spec("extxyz") is None,
    reason="libAtoms extxyz not installed (no CPython 3.14 wheels)",
)

needs_ase_extxyz = pytest.mark.skipif(
    importlib.util.find_spec("ase_extxyz") is None
    or importlib.util.find_spec("ase") is None,
    reason="ase-extxyz not installed (no CPython 3.14 wheels)",
)


THREAD_SWEEP = (1, 2, 4, 8)


def row(output: str, mode: str, threads: int | None = None):
    """Tag a reader with the output/mode columns report.py renders.

    Swept rows also record their thread count; plot.py folds rows that
    differ only in threads into one reader with a bar per count.
    """

    def tag(fn):
        info: dict[str, str | int] = {"output": output, "mode": mode}
        if threads is not None:
            info["threads"] = threads
        fn.row_info = info
        return fn

    return tag


@functools.cache
def file_shape(path: Path) -> tuple[int, int]:
    """(n_frames, n_atoms) for a fixture, from the structural scan."""
    index = oxyz.scan(path)
    return index.n_frames, index.total_atoms


def run(benchmark, read, path: Path, shape: tuple[int, int] | str | None = "file"):
    """Benchmark `read(path)`, recording its row tags in the save.

    `shape` is the work the row does, recorded as n_frames/n_atoms so
    report.py can derive frames/s and atoms/s. The default records the
    whole file; rows reading a subset pass an explicit (n_frames, n_atoms);
    rows whose cost does not scale with frames read (first/last/scan) pass
    None and get no throughput columns.
    """
    benchmark.extra_info.update(read.row_info)
    if shape == "file":
        shape = file_shape(path)
    if shape is not None:
        n_frames, n_atoms = shape
        benchmark.extra_info.update(n_frames=int(n_frames), n_atoms=int(n_atoms))
    return benchmark(read, path)


def oxyz_read_all_with(threads: int | None):
    @row("numpy frames", "serial" if threads == 1 else "parallel", threads=threads)
    def read(path: Path) -> list:
        return oxyz.read(path, threads=threads)

    return read


def oxyz_read_all_schema_with(conformance: oxyz.Conformance):
    @row("numpy frames", f"schema-{conformance}")
    def read(path: Path) -> list:
        return oxyz.read(path, schema=read.spec, conformance=conformance)

    read.spec = None  # set per file by the test below, before timing starts
    return read


@row("numpy frames", "serial")
def oxyz_iter_read_all(path: Path) -> list:
    # The constant-memory streaming path; collected so every read_all row
    # does the same total work.
    return list(oxyz.iread(path))


@row("numpy frames", "serial")
def oxyz_read_first(path: Path) -> object:
    return oxyz.read(path, 0)


@row("ase.Atoms", "serial")
def ase_read_all(path: Path) -> list:
    from ase.io import read

    frames = read(path, index=":", format="extxyz")
    # `read` returns Atoms | list[Atoms] depending on `index`; narrow it.
    assert isinstance(frames, list)
    return frames


def oxyz_to_ase_read_all_with(threads: int | None):
    @row("ase.Atoms", "serial" if threads == 1 else "parallel", threads=threads)
    def read(path: Path) -> list:
        from oxyz.ase import read as ase_read

        # slice(None) rather than ":" picks the precisely-typed overload.
        return ase_read(path, index=slice(None), threads=threads)

    return read


@row("ase.Atoms", "serial")
def oxyz_to_ase_read_first(path: Path) -> object:
    from oxyz.ase import read

    return read(path, index=0)


@row("ase.Atoms", "serial")
def oxyz_to_ase_read_last(path: Path) -> object:
    from oxyz.ase import read

    return read(path, index=-1)


@row("ase.Atoms", "serial")
def ase_read_last(path: Path) -> object:
    from ase.io import read

    return read(path, index=-1, format="extxyz")


@row("ase.Atoms", "serial")
def ase_read_first(path: Path) -> object:
    from ase.io import read

    return read(path, index=0, format="extxyz")


@row("numpy frames", "serial")
def cextxyz_read_all(path: Path) -> list:
    # Not installable on 3.14 (needs_cextxyz skips there), so unresolvable
    # for ty, which checks under the 3.14 venv.
    from extxyz import read_dicts  # ty: ignore[unresolved-import]

    frames = read_dicts(str(path))
    # `read_dicts` returns Frame | list[Frame] depending on count; narrow it.
    assert isinstance(frames, list)
    return frames


@row("numpy frames", "serial")
def cextxyz_read_first(path: Path) -> object:
    from extxyz import iread_dicts  # ty: ignore[unresolved-import]

    return next(iter(iread_dicts(str(path))))


@row("ase.Atoms", "serial")
def cextxyz_to_ase_read_all(path: Path) -> list:
    # ase-extxyz registers the C parser as an ASE IO plugin: same Atoms
    # output as the `ase` row, parsing via cextxyz.
    from ase.io import read

    frames = read(path, index=":", format="cextxyz")
    assert isinstance(frames, list)
    return frames


@row("ase.Atoms", "serial")
def cextxyz_to_ase_read_first(path: Path) -> object:
    from ase.io import read

    return read(path, index=0, format="cextxyz")


@row("ase.Atoms", "serial")
def cextxyz_to_ase_read_last(path: Path) -> object:
    from ase.io import read

    return read(path, index=-1, format="cextxyz")


READ_ALL = [
    # threads=None is the default call: every core the machine has. The sweep
    # stops at 8, so without this row no measurement reaches the core count the
    # all-core label names.
    pytest.param(oxyz_read_all_with(None), id="oxyz"),
    *(pytest.param(oxyz_read_all_with(t), id=f"oxyz-{t}t") for t in THREAD_SWEEP),
    pytest.param(oxyz_iter_read_all, id="oxyz-iter"),
    pytest.param(oxyz_to_ase_read_all_with(None), id="oxyz-to-ase", marks=needs_ase),
    pytest.param(
        oxyz_to_ase_read_all_with(1), id="oxyz-to-ase-serial", marks=needs_ase
    ),
    pytest.param(ase_read_all, id="ase", marks=needs_ase),
    pytest.param(cextxyz_read_all, id="cextxyz", marks=needs_cextxyz),
    pytest.param(cextxyz_to_ase_read_all, id="cextxyz-to-ase", marks=needs_ase_extxyz),
]

READ_FIRST = [
    pytest.param(oxyz_read_first, id="oxyz"),
    pytest.param(oxyz_to_ase_read_first, id="oxyz-to-ase", marks=needs_ase),
    pytest.param(ase_read_first, id="ase", marks=needs_ase),
    pytest.param(cextxyz_read_first, id="cextxyz", marks=needs_cextxyz),
    pytest.param(
        cextxyz_to_ase_read_first, id="cextxyz-to-ase", marks=needs_ase_extxyz
    ),
]


@pytest.mark.benchmark(group="read_all/many_small_frames")
@pytest.mark.parametrize("read", READ_ALL)
def test_read_all_many_small_frames(benchmark, read, many_small_frames):
    frames = run(benchmark, read, many_small_frames)
    assert len(frames) == 2_000


@pytest.mark.benchmark(group="read_all/large_frames")
@pytest.mark.parametrize("read", READ_ALL)
def test_read_all_large_frames(benchmark, read, large_frames):
    frames = run(benchmark, read, large_frames)
    assert len(frames) == 4


@pytest.mark.benchmark(group="read_all/metadata_heavy")
@pytest.mark.parametrize("read", READ_ALL)
def test_read_all_metadata_heavy(benchmark, read, metadata_heavy):
    frames = run(benchmark, read, metadata_heavy)
    assert len(frames) == 2_000


@pytest.mark.benchmark(group="read_all/mace_mixed")
@pytest.mark.parametrize("read", READ_ALL)
def test_read_all_mace_mixed(benchmark, read, mace_mixed):
    frames = run(benchmark, read, mace_mixed)
    assert len(frames) == 1_000


SCHEMA_READ = [
    pytest.param(oxyz_read_all_with(None), id="baseline"),
    pytest.param(oxyz_read_all_schema_with("required"), id="schema-required"),
    pytest.param(oxyz_read_all_schema_with("strict"), id="schema-strict"),
]


# Isolates per-frame validation cost from parsing: all three rows parse the
# same file the same way, differing only in whether/how strictly read_frames
# checks each frame against a schema. The schema itself is derived from the
# file (the common case: validate what training data actually looks like)
# and computed once per row, before timing starts, so the measured cost is
# validation alone, not schema inference.
@pytest.mark.benchmark(group="schema_read/mace_mixed")
@pytest.mark.parametrize("read", SCHEMA_READ)
def test_schema_read_overhead(benchmark, read, mace_mixed):
    if getattr(read, "spec", "unset") is None:
        read.spec = oxyz.infer_schema(mace_mixed).to_spec()
    frames = run(benchmark, read, mace_mixed)
    assert len(frames) == 1_000


@pytest.mark.benchmark(group="read_first/large_frames")
@pytest.mark.parametrize("read", READ_FIRST)
def test_read_first_of_large_file(benchmark, read, large_frames):
    frame = run(benchmark, read, large_frames, shape=None)
    assert frame is not None


def oxyz_read_batch_strided_with(threads: int):
    @row("Batch", "serial" if threads == 1 else "parallel", threads=threads)
    def read(path: Path) -> object:
        return oxyz.read_batch(path, range(0, 2_000, 20), threads=threads)

    return read


@row("ase.Atoms", "serial")
def oxyz_to_ase_read_strided(path: Path) -> list:
    from oxyz.ase import read

    return read(path, index=slice(None, None, 20))


@row("ase.Atoms", "serial")
def ase_read_strided(path: Path) -> list:
    from ase.io import read

    frames = read(path, index="::20", format="extxyz")
    assert isinstance(frames, list)
    return frames


@row("ase.Atoms", "serial")
def cextxyz_to_ase_read_strided(path: Path) -> list:
    from ase.io import read

    frames = read(path, index="::20", format="cextxyz")
    assert isinstance(frames, list)
    return frames


# Selective read: every 20th frame (100 of 2000). The oxyz rows seek via
# the byte-offset index. Output contracts: `oxyz-read-batch` returns one
# Batch, the other rows lists of Atoms.
@pytest.mark.benchmark(group="selective/many_small_frames")
@pytest.mark.parametrize(
    "read",
    [
        *(
            pytest.param(oxyz_read_batch_strided_with(t), id=f"oxyz-read-batch-{t}t")
            for t in THREAD_SWEEP
        ),
        pytest.param(oxyz_to_ase_read_strided, id="oxyz-to-ase", marks=needs_ase),
        pytest.param(ase_read_strided, id="ase", marks=needs_ase),
        pytest.param(
            cextxyz_to_ase_read_strided, id="cextxyz-to-ase", marks=needs_ase_extxyz
        ),
    ],
)
def test_selective_read_of_many_small_frames(benchmark, read, many_small_frames):
    # Throughput is over the 100 frames actually read, not the whole file.
    index = oxyz.scan(many_small_frames)
    selected = index.n_atoms[::20]
    result = run(
        benchmark, read, many_small_frames, shape=(len(selected), selected.sum())
    )
    assert result is not None


@row("Batch", "serial")
def oxyz_read_batch_last(path: Path) -> object:
    # The large_frames fixture has 4 frames; read_batch takes absolute
    # indices, so "last" is spelled 3. One frame requested, so threads
    # would have nothing to parallelise; serial keeps the row honest.
    return oxyz.read_batch(path, [3], threads=1)


READ_LAST = [
    pytest.param(oxyz_read_batch_last, id="oxyz-read-batch"),
    pytest.param(oxyz_to_ase_read_last, id="oxyz-to-ase", marks=needs_ase),
    pytest.param(ase_read_last, id="ase", marks=needs_ase),
    pytest.param(cextxyz_to_ase_read_last, id="cextxyz-to-ase", marks=needs_ase_extxyz),
]


# Exercises the structural scan: seek to the last frame instead of parsing
# the whole file (which is what index=-1 costs without an index).
@pytest.mark.benchmark(group="read_last/large_frames")
@pytest.mark.parametrize("read", READ_LAST)
def test_read_last_frame_of_large_file(benchmark, read, large_frames):
    frame = run(benchmark, read, large_frames, shape=None)
    assert frame is not None


@row("FrameIndex", "serial")
def oxyz_scan(path: Path) -> object:
    return oxyz.scan(path)


# The structural scan underlies iread_batch planning, IndexedFrames open,
# and ASE-style negative/strided indexing; it should sit far above parse
# throughput to earn its keep.
@pytest.mark.benchmark(group="scan/many_small_frames")
@pytest.mark.parametrize("read", [pytest.param(oxyz_scan, id="oxyz-scan")])
def test_scan_many_small_frames(benchmark, read, many_small_frames):
    index = run(benchmark, read, many_small_frames, shape=None)
    assert index.n_frames == 2_000


@pytest.mark.benchmark(group="scan/large_frames")
@pytest.mark.parametrize("read", [pytest.param(oxyz_scan, id="oxyz-scan")])
def test_scan_large_frames(benchmark, read, large_frames):
    index = run(benchmark, read, large_frames, shape=None)
    assert index.n_frames == 4


def oxyz_sequential_batches_with(threads: int):
    @row("Batch", "serial" if threads == 1 else "parallel", threads=threads)
    def batched(path: Path) -> int:
        total = 0
        for batch in oxyz.iread_batch(path, frames_per_batch=64, threads=threads):
            total += batch.total_atoms
        return total

    return batched


def oxyz_shuffled_atom_batches_with(threads: int):
    @row("Batch", "serial" if threads == 1 else "parallel", threads=threads)
    def batched(path: Path) -> int:
        batches = oxyz.iread_batch(
            path, atoms_per_batch=2048, shuffle=True, seed=0, threads=threads
        )
        return sum(batch.total_atoms for batch in batches)

    return batched


# No ASE rows: ASE has no batch concept. Tracked against read_all[oxyz]
# (same parse work, per-frame objects) as the informal baseline.
@pytest.mark.benchmark(group="batches/many_small_frames")
@pytest.mark.parametrize(
    "batched_read",
    [
        *(
            pytest.param(
                oxyz_sequential_batches_with(t), id=f"sequential-64-frames-{t}t"
            )
            for t in THREAD_SWEEP
        ),
        *(
            pytest.param(
                oxyz_shuffled_atom_batches_with(t), id=f"shuffled-2048-atoms-{t}t"
            )
            for t in THREAD_SWEEP
        ),
    ],
)
def test_batched_read_of_many_small_frames(benchmark, batched_read, many_small_frames):
    assert run(benchmark, batched_read, many_small_frames) > 0


# The only real dataset in the suite: chemically diverse frames that disagree
# on schema, at a size no generated fixture reaches. min_rounds=1 because a
# single ase row here builds 180k Atoms objects off 303.5 MiB of text.
@pytest.mark.benchmark(group="real_data/mad", min_rounds=1)
@pytest.mark.parametrize("read", READ_ALL)
def test_read_all_mad_full(benchmark, read, mad_full):
    frames = run(benchmark, read, mad_full)
    # The frame count doubles as a guard that the file on disk is this dataset.
    assert len(frames) == MAD_N_FRAMES


@pytest.mark.benchmark(group="real_data/mad_scan", min_rounds=1)
@pytest.mark.parametrize("read", [pytest.param(oxyz_scan, id="oxyz-scan")])
def test_scan_mad_full(benchmark, read, mad_full):
    index = run(benchmark, read, mad_full, shape=None)
    assert index.n_frames == MAD_N_FRAMES
