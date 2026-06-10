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

import importlib.util
from pathlib import Path

import pytest

import oxyz

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


def row(output: str, mode: str):
    """Tag a reader with the output/mode columns report.py renders."""

    def tag(fn):
        fn.row_info = {"output": output, "mode": mode}
        return fn

    return tag


def run(benchmark, read, path: Path):
    """Benchmark `read(path)`, recording its row tags in the save."""
    benchmark.extra_info.update(read.row_info)
    return benchmark(read, path)


@row("numpy frames", "parallel")
def oxyz_read_all(path: Path) -> list:
    return oxyz.read_frames(path)


@row("numpy frames", "serial")
def oxyz_read_all_serial(path: Path) -> list:
    return oxyz.read_frames(path, threads=1)


@row("numpy frames", "serial")
def oxyz_read_first(path: Path) -> object:
    return oxyz.read_first_frame(path)


@row("ase.Atoms", "serial")
def ase_read_all(path: Path) -> list:
    from ase.io import read

    frames = read(path, index=":", format="extxyz")
    # `read` returns Atoms | list[Atoms] depending on `index`; narrow it.
    assert isinstance(frames, list)
    return frames


@row("ase.Atoms", "serial")
def oxyz_to_ase_read_all(path: Path) -> list:
    from oxyz.ase import read

    # slice(None) rather than ":" picks the precisely-typed overload.
    return read(path, index=slice(None))


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
    pytest.param(oxyz_read_all, id="oxyz"),
    pytest.param(oxyz_read_all_serial, id="oxyz-serial"),
    pytest.param(oxyz_to_ase_read_all, id="oxyz-to-ase", marks=needs_ase),
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


@pytest.mark.benchmark(group="read_first/large_frames")
@pytest.mark.parametrize("read", READ_FIRST)
def test_read_first_frame_of_large_file(benchmark, read, large_frames):
    frame = run(benchmark, read, large_frames)
    assert frame is not None


@row("Batch", "parallel")
def oxyz_read_batch_strided(path: Path) -> object:
    return oxyz.read_batch(path, range(0, 2_000, 20))


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
        pytest.param(oxyz_read_batch_strided, id="oxyz-read-batch"),
        pytest.param(oxyz_to_ase_read_strided, id="oxyz-to-ase", marks=needs_ase),
        pytest.param(ase_read_strided, id="ase", marks=needs_ase),
        pytest.param(
            cextxyz_to_ase_read_strided, id="cextxyz-to-ase", marks=needs_ase_extxyz
        ),
    ],
)
def test_selective_read_of_many_small_frames(benchmark, read, many_small_frames):
    result = run(benchmark, read, many_small_frames)
    assert result is not None


READ_LAST = [
    pytest.param(oxyz_to_ase_read_last, id="oxyz-to-ase", marks=needs_ase),
    pytest.param(ase_read_last, id="ase", marks=needs_ase),
    pytest.param(cextxyz_to_ase_read_last, id="cextxyz-to-ase", marks=needs_ase_extxyz),
]


# Exercises the structural scan: seek to the last frame instead of parsing
# the whole file (which is what index=-1 costs without an index).
@pytest.mark.benchmark(group="read_last/large_frames")
@pytest.mark.parametrize("read", READ_LAST)
def test_read_last_frame_of_large_file(benchmark, read, large_frames):
    frame = run(benchmark, read, large_frames)
    assert frame is not None


@row("Batch", "parallel")
def oxyz_sequential_batches(path: Path) -> int:
    total = 0
    for batch in oxyz.iter_batches(path, frames_per_batch=64):
        total += batch.total_atoms
    return total


@row("Batch", "parallel")
def oxyz_shuffled_atom_batches(path: Path) -> int:
    batches = oxyz.iter_batches(path, atoms_per_batch=2048, shuffle=True, seed=0)
    return sum(batch.total_atoms for batch in batches)


# No ASE rows: ASE has no batch concept. Tracked against read_all[oxyz]
# (same parse work, per-frame objects) as the informal baseline.
@pytest.mark.benchmark(group="batches/many_small_frames")
@pytest.mark.parametrize(
    "batched_read",
    [
        pytest.param(oxyz_sequential_batches, id="sequential-64-frames"),
        pytest.param(oxyz_shuffled_atom_batches, id="shuffled-2048-atoms"),
    ],
)
def test_batched_read_of_many_small_frames(benchmark, batched_read, many_small_frames):
    assert run(benchmark, batched_read, many_small_frames) > 0
