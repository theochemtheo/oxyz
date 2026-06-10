from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import atomflow._rust as _rust
from atomflow._frames import ColumnValues
from atomflow._scan import scan


@dataclass(frozen=True, slots=True)
class Batch:
    """Frames concatenated atom-major, CSR-style (PyG's batch layout).

    `columns` holds per-atom arrays with `total_atoms` rows; `metadata` holds
    per-frame arrays with `n_frames` rows. Frame `i` occupies rows
    `offsets[i]:offsets[i + 1]` of every column. `frame_indices` records
    which file frame each batch entry came from — provenance for shuffled
    training batches.
    """

    columns: dict[str, ColumnValues]
    metadata: dict[str, ColumnValues]
    offsets: np.ndarray
    frame_indices: np.ndarray

    @property
    def n_frames(self) -> int:
        return len(self.offsets) - 1

    @property
    def total_atoms(self) -> int:
        return int(self.offsets[-1])

    @property
    def n_atoms(self) -> np.ndarray:
        return np.diff(self.offsets)

    @property
    def ptr(self) -> np.ndarray:
        """Alias of `offsets`, under its PyG name."""
        return self.offsets

    @property
    def batch(self) -> np.ndarray:
        """Per-atom frame id within this batch (PyG's `batch` vector)."""
        return np.repeat(np.arange(self.n_frames), self.n_atoms)


def read_batch(
    path: str | Path, indices: Sequence[int], *, threads: int | None = None
) -> Batch:
    """Gather the given frames (in order, repeats allowed) into one batch.

    Scans the file on every call; for repeated gathers from one file, prefer
    `iter_batches`, which scans once. `threads=None` parses on every core,
    `threads=1` serially; the batch is identical either way.
    """
    plan = [int(i) for i in indices]
    reader = _rust.IndexedFrames(str(path))
    return _batch_from_data(reader.get_batch(plan, threads), plan)


def iter_batches(
    path: str | Path,
    *,
    frames_per_batch: int | None = None,
    atoms_per_batch: int | None = None,
    shuffle: bool = False,
    seed: int | None = None,
    threads: int | None = None,
) -> Iterator[Batch]:
    """Read a file as a sequence of batches.

    Exactly one of `frames_per_batch` (fixed structure count) or
    `atoms_per_batch` (greedy packing to a total-atom budget; a frame larger
    than the budget gets a batch to itself) must be given. `shuffle` draws
    frames in a seeded random order via the byte-offset index instead of
    file order.

    Batch composition depends only on the file, the knobs, and the seed —
    never on `threads`, which sets parsing parallelism within each batch
    (None: all cores; 1: serial, which for unshuffled `frames_per_batch`
    streams the file without scanning).
    """
    if (frames_per_batch is None) == (atoms_per_batch is None):
        raise ValueError("pass exactly one of frames_per_batch or atoms_per_batch")
    if frames_per_batch is not None and frames_per_batch < 1:
        raise ValueError("frames_per_batch must be at least 1")
    if atoms_per_batch is not None and atoms_per_batch < 1:
        raise ValueError("atoms_per_batch must be at least 1")
    if seed is not None and not shuffle:
        raise ValueError("seed requires shuffle=True")

    if frames_per_batch is not None and not shuffle and threads == 1:
        return _sequential_batches(path, frames_per_batch)
    return _planned_batches(
        path, frames_per_batch, atoms_per_batch, shuffle, seed, threads
    )


def _sequential_batches(path: str | Path, frames_per_batch: int) -> Iterator[Batch]:
    """Streamed file-order batches: constant memory, no scan."""
    start = 0
    for data in _rust.BatchIter(str(path), frames_per_batch):
        n_frames = len(data["offsets"]) - 1
        yield _batch_from_data(data, range(start, start + n_frames))
        start += n_frames


def _planned_batches(
    path: str | Path,
    frames_per_batch: int | None,
    atoms_per_batch: int | None,
    shuffle: bool,
    seed: int | None,
    threads: int | None,
) -> Iterator[Batch]:
    """Index-backed batches over a frame order planned up front.

    Planning is serial and happens before any parsing, so `threads` cannot
    influence which frames land in which batch."""
    n_atoms = scan(path).n_atoms
    order = np.arange(len(n_atoms), dtype=np.intp)
    if shuffle:
        order = np.random.default_rng(seed).permutation(order)

    if frames_per_batch is not None:
        plans = [
            [int(i) for i in order[start : start + frames_per_batch]]
            for start in range(0, len(order), frames_per_batch)
        ]
    else:
        assert atoms_per_batch is not None
        plans = _greedy_atom_plans(order, n_atoms, atoms_per_batch)

    reader = _rust.IndexedFrames(str(path))
    for plan in plans:
        yield _batch_from_data(reader.get_batch(plan, threads), plan)


def _greedy_atom_plans(
    order: np.ndarray, n_atoms: np.ndarray, atoms_per_batch: int
) -> list[list[int]]:
    """Fill batches in `order` until the next frame would exceed the budget."""
    plans: list[list[int]] = []
    current: list[int] = []
    total = 0
    for i in order:
        count = int(n_atoms[i])
        if current and total + count > atoms_per_batch:
            plans.append(current)
            current, total = [], 0
        current.append(int(i))
        total += count
    if current:
        plans.append(current)
    return plans


def _batch_from_data(data: _rust.BatchData, frame_indices: Sequence[int]) -> Batch:
    return Batch(
        columns=data["columns"],
        metadata=data["metadata"],
        offsets=data["offsets"],
        frame_indices=np.asarray(frame_indices, dtype=np.intp),
    )
