"""Scaling sweeps: throughput vs thread count, and time vs file size across
two families. Recorded into the same save JSON as the scenario suite; the
curve renderer in plot.py reads the scaling_* groups.

Record: uv run benchmarks/run.py --benchmark-autosave
Smoke:  uv run benchmarks/run.py --benchmark-disable benchmarks/test_scaling.py
Render: uv run benchmarks/plot.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from test_read import (
    needs_ase,
    needs_ase_extxyz,
    needs_cextxyz,
    row,
    run,
)

import conftest
import oxyz

# Competitors are far slower; cap them so the top sweep points (oxyz-only)
# don't drag a full run past budget. The gap is already vast by the cap.
COMPETITOR_MAX_DATASET_FRAMES = 100_000
COMPETITOR_MAX_SYSTEM_ATOMS = 100_000

# Thread axis: 1..N over the machine's logical cores, a handful of points.
_CPU = os.cpu_count() or 8
THREAD_POINTS = sorted({1, 2, 3, 4, 6, 8, _CPU})

# Representative sizes for the thread sweep — large enough that parallelism
# across frames (dataset) / within the file (system) actually shows.
THREAD_DATASET_FRAMES = 100_000
THREAD_SYSTEM_ATOMS = 200_000


# --- readers ---------------------------------------------------------------


@row("numpy frames", "parallel")
def oxyz_parallel(path: Path) -> list:
    return oxyz.read(path)


@row("numpy frames", "serial")
def oxyz_serial(path: Path) -> list:
    return oxyz.read(path, threads=1)


@row("numpy frames", "serial")
def cextxyz_read(path: Path) -> list:
    from extxyz import read_dicts  # ty: ignore[unresolved-import]

    frames = read_dicts(str(path))
    assert isinstance(frames, list)
    return frames


@row("ase.Atoms", "serial")
def cextxyz_to_ase(path: Path) -> list:
    from ase.io import read

    frames = read(path, index=":", format="cextxyz")
    assert isinstance(frames, list)
    return frames


@row("ase.Atoms", "serial")
def ase_read(path: Path) -> list:
    from ase.io import read

    frames = read(path, index=":", format="extxyz")
    assert isinstance(frames, list)
    return frames


def _size_readers(size: int, cap: int):
    """Readers for one size point: oxyz always; competitors below the cap."""
    readers = [
        pytest.param(oxyz_parallel, id="oxyz"),
        pytest.param(oxyz_serial, id="oxyz-serial"),
    ]
    if size <= cap:
        readers += [
            pytest.param(cextxyz_read, id="cextxyz", marks=needs_cextxyz),
            pytest.param(cextxyz_to_ase, id="cextxyz-to-ase", marks=needs_ase_extxyz),
            pytest.param(ase_read, id="ase", marks=needs_ase),
        ]
    return readers


# --- size families: flat (reader, size) params so competitors drop above ---
# their cap and each combination is its own benchmark row. `pytest.param`
# carries the id and skip marks straight through.


def _size_params(sizes, cap):
    params = []
    for size in sizes:
        for p in _size_readers(size, cap):
            params.append(
                pytest.param(p.values[0], size, id=f"{p.id}-{size}", marks=p.marks)
            )
    return params


DATASET_PARAMS = _size_params(
    conftest.DATASET_SIZE_FRAMES, COMPETITOR_MAX_DATASET_FRAMES
)
SYSTEM_PARAMS = _size_params(conftest.SYSTEM_SIZE_ATOMS, COMPETITOR_MAX_SYSTEM_ATOMS)


@pytest.mark.parametrize("reader,n_frames", DATASET_PARAMS)
def test_scaling_dataset(benchmark, reader, n_frames):
    path = conftest.sweep_dataset_size_file(n_frames)
    total_atoms = sum(conftest.dataset_frame_atoms(n_frames))
    benchmark.group = f"scaling_dataset/{n_frames}"
    result = run(benchmark, reader, path, shape=(n_frames, total_atoms))
    assert len(result) == n_frames


@pytest.mark.parametrize("reader,n_atoms", SYSTEM_PARAMS)
def test_scaling_system(benchmark, reader, n_atoms):
    path = conftest.sweep_system_size_file(n_atoms)
    benchmark.group = f"scaling_system/{n_atoms}"
    frames = conftest.SYSTEM_SIZE_FRAMES
    result = run(benchmark, reader, path, shape=(frames, frames * n_atoms))
    assert len(result) == frames


# --- thread sweep (oxyz only, one row per thread count) --------------------


def _oxyz_threads(n: int):
    @row("numpy frames", "serial" if n == 1 else "parallel", threads=n)
    def read(path: Path) -> list:
        return oxyz.read(path, threads=n)

    return read


THREAD_PARAMS = [
    pytest.param(_oxyz_threads(n), n, id=f"oxyz-{n}t") for n in THREAD_POINTS
]


@pytest.mark.parametrize("reader,threads", THREAD_PARAMS)
def test_scaling_threads_dataset(benchmark, reader, threads):
    path = conftest.sweep_dataset_size_file(THREAD_DATASET_FRAMES)
    total_atoms = sum(conftest.dataset_frame_atoms(THREAD_DATASET_FRAMES))
    benchmark.group = "scaling_threads/dataset"
    result = run(benchmark, reader, path, shape=(THREAD_DATASET_FRAMES, total_atoms))
    assert len(result) == THREAD_DATASET_FRAMES


@pytest.mark.parametrize("reader,threads", THREAD_PARAMS)
def test_scaling_threads_system(benchmark, reader, threads):
    path = conftest.sweep_system_size_file(THREAD_SYSTEM_ATOMS)
    benchmark.group = "scaling_threads/system"
    frames = conftest.SYSTEM_SIZE_FRAMES
    result = run(benchmark, reader, path, shape=(frames, frames * THREAD_SYSTEM_ATOMS))
    assert len(result) == frames
