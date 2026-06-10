"""End-to-end read benchmarks, one row per library per workload.

Run:        uv run pytest benchmarks/ --benchmark-autosave
Compare:    uv run pytest-benchmark compare
Smoke only: uv run pytest benchmarks/ --benchmark-disable

uv rebuilds the extension in release when Rust sources change (cache-keys
in pyproject.toml), and conftest refuses to run against a debug build —
which a manual `maturin develop` can still install unnoticed.

Results land in .benchmarks/ (gitignored); numbers are only comparable
across runs on the same machine.

Fairness: output contracts differ. The `atomflow` row returns Frame
dataclasses holding numpy arrays; the `ase` row builds full `Atoms`
objects. The `atomflow-to-ase` row is the like-for-like comparison with
`ase`: same `Atoms` output, via `atomflow.ase.read`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import atomflow

needs_ase = pytest.mark.skipif(
    importlib.util.find_spec("ase") is None, reason="ase not installed"
)


def atomflow_read_all(path: Path) -> list:
    return atomflow.read_frames(path)


def atomflow_read_first(path: Path) -> object:
    return atomflow.read_first_frame(path)


def ase_read_all(path: Path) -> list:
    from ase.io import read

    frames = read(path, index=":", format="extxyz")
    # `read` returns Atoms | list[Atoms] depending on `index`; narrow it.
    assert isinstance(frames, list)
    return frames


def atomflow_to_ase_read_all(path: Path) -> list:
    from atomflow.ase import read

    # slice(None) rather than ":" picks the precisely-typed overload.
    return read(path, index=slice(None))


def atomflow_to_ase_read_first(path: Path) -> object:
    from atomflow.ase import read

    return read(path, index=0)


def atomflow_to_ase_read_last(path: Path) -> object:
    from atomflow.ase import read

    return read(path, index=-1)


def ase_read_last(path: Path) -> object:
    from ase.io import read

    return read(path, index=-1, format="extxyz")


def ase_read_first(path: Path) -> object:
    from ase.io import read

    return read(path, index=0, format="extxyz")


READ_ALL = [
    pytest.param(atomflow_read_all, id="atomflow"),
    pytest.param(atomflow_to_ase_read_all, id="atomflow-to-ase", marks=needs_ase),
    pytest.param(ase_read_all, id="ase", marks=needs_ase),
]

READ_FIRST = [
    pytest.param(atomflow_read_first, id="atomflow"),
    pytest.param(atomflow_to_ase_read_first, id="atomflow-to-ase", marks=needs_ase),
    pytest.param(ase_read_first, id="ase", marks=needs_ase),
]


@pytest.mark.benchmark(group="read_all/many_small_frames")
@pytest.mark.parametrize("read", READ_ALL)
def test_read_all_many_small_frames(benchmark, read, many_small_frames):
    frames = benchmark(read, many_small_frames)
    assert len(frames) == 2_000


@pytest.mark.benchmark(group="read_all/large_frames")
@pytest.mark.parametrize("read", READ_ALL)
def test_read_all_large_frames(benchmark, read, large_frames):
    frames = benchmark(read, large_frames)
    assert len(frames) == 4


@pytest.mark.benchmark(group="read_first/large_frames")
@pytest.mark.parametrize("read", READ_FIRST)
def test_read_first_frame_of_large_file(benchmark, read, large_frames):
    frame = benchmark(read, large_frames)
    assert frame is not None


READ_LAST = [
    pytest.param(atomflow_to_ase_read_last, id="atomflow-to-ase", marks=needs_ase),
    pytest.param(ase_read_last, id="ase", marks=needs_ase),
]


# Exercises the structural scan: seek to the last frame instead of parsing
# the whole file (which is what index=-1 costs without an index).
@pytest.mark.benchmark(group="read_last/large_frames")
@pytest.mark.parametrize("read", READ_LAST)
def test_read_last_frame_of_large_file(benchmark, read, large_frames):
    frame = benchmark(read, large_frames)
    assert frame is not None


def atomflow_sequential_batches(path: Path) -> int:
    total = 0
    for batch in atomflow.iter_batches(path, frames_per_batch=64):
        total += batch.total_atoms
    return total


def atomflow_shuffled_atom_batches(path: Path) -> int:
    batches = atomflow.iter_batches(path, atoms_per_batch=2048, shuffle=True, seed=0)
    return sum(batch.total_atoms for batch in batches)


# No ASE rows: ASE has no batch concept. Tracked against read_all[atomflow]
# (same parse work, per-frame objects) as the informal baseline.
@pytest.mark.benchmark(group="batches/many_small_frames")
@pytest.mark.parametrize(
    "batched_read",
    [
        pytest.param(atomflow_sequential_batches, id="sequential-64-frames"),
        pytest.param(atomflow_shuffled_atom_batches, id="shuffled-2048-atoms"),
    ],
)
def test_batched_read_of_many_small_frames(benchmark, batched_read, many_small_frames):
    assert benchmark(batched_read, many_small_frames) > 0
