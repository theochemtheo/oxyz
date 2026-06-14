from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import oxyz

DATA_DIR = Path(__file__).parent / "data"
VARYING = DATA_DIR / "varying_atom_counts.xyz"


def as_array(value: object) -> np.ndarray:
    """Same ty-limitation shim as test_extxyz.as_array; delete with the canary."""
    assert isinstance(value, np.ndarray)
    return value


def test_sequential_batches_chunk_the_file() -> None:
    batches = list(oxyz.iter_batches(VARYING, frames_per_batch=2))

    assert len(batches) == 2
    first, last = batches
    assert first.n_frames == 2
    assert first.total_atoms == 4
    assert_array_equal(first.offsets, [0, 3, 4])
    assert_array_equal(first.frame_indices, [0, 1])
    assert_array_equal(last.frame_indices, [2])


def test_batch_columns_concatenate_frames() -> None:
    frames = oxyz.read_frames(VARYING)
    (batch,) = oxyz.iter_batches(VARYING, frames_per_batch=3)

    stacked = np.vstack([as_array(frame.columns["pos"]) for frame in frames])
    assert_allclose(as_array(batch.columns["pos"]), stacked)
    assert batch.columns["species"] == [
        s for frame in frames for s in frame.columns["species"]
    ]
    assert_allclose(as_array(batch.metadata["energy"]), [-76.3, -13.6, -31.8])


def test_batch_derived_properties() -> None:
    (batch,) = oxyz.iter_batches(VARYING, frames_per_batch=3)

    assert_array_equal(batch.n_atoms, [3, 1, 2])
    assert_array_equal(batch.ptr, batch.offsets)
    assert_array_equal(batch.batch, [0, 0, 0, 1, 2, 2])


def test_atom_budget_packs_greedily() -> None:
    batches = list(oxyz.iter_batches(VARYING, atoms_per_batch=4))
    assert [list(b.frame_indices) for b in batches] == [[0, 1], [2]]

    # A frame above the budget still gets a batch to itself.
    batches = list(oxyz.iter_batches(VARYING, atoms_per_batch=2))
    assert [list(b.frame_indices) for b in batches] == [[0], [1], [2]]


def test_shuffled_batches_are_seeded_and_partition_the_file() -> None:
    def plan(seed: int) -> list[list[int]]:
        return [
            list(b.frame_indices)
            for b in oxyz.iter_batches(
                VARYING, atoms_per_batch=4, shuffle=True, seed=seed
            )
        ]

    assert plan(0) == plan(0)

    flat = sorted(i for batch in plan(0) for i in batch)
    assert flat == [0, 1, 2]


def test_read_batch_gathers_in_requested_order() -> None:
    frames = oxyz.read_frames(VARYING)
    batch = oxyz.read_batch(VARYING, [2, 0])

    assert_array_equal(batch.frame_indices, [2, 0])
    assert_array_equal(batch.offsets, [0, 2, 5])
    stacked = np.vstack(
        [as_array(frames[2].columns["pos"]), as_array(frames[0].columns["pos"])]
    )
    assert_allclose(as_array(batch.columns["pos"]), stacked)
    assert_allclose(as_array(batch.metadata["energy"]), [-31.8, -76.3])


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"frames_per_batch": 2, "atoms_per_batch": 4},
        {"frames_per_batch": 0},
        {"atoms_per_batch": 0},
        {"frames_per_batch": 2, "seed": 0},
        {"frames_per_batch": 2, "threads": 0},
    ],
)
def test_invalid_batching_arguments(kwargs) -> None:
    with pytest.raises(ValueError):
        oxyz.iter_batches(VARYING, **kwargs)


def test_zero_threads_is_rejected() -> None:
    # threads=0 would read as "all cores" in rayon; require None or >= 1.
    with pytest.raises(ValueError, match="threads must be"):
        oxyz.read_batch(VARYING, [0], threads=0)
    with pytest.raises(ValueError, match="threads must be"):
        oxyz.read_frames(VARYING, threads=0)


def assert_batches_equal(left: oxyz.Batch, right: oxyz.Batch) -> None:
    assert_array_equal(left.frame_indices, right.frame_indices)
    assert_array_equal(left.offsets, right.offsets)
    assert set(left.columns) == set(right.columns)
    for name, values in right.columns.items():
        if isinstance(values, np.ndarray):
            assert_array_equal(as_array(left.columns[name]), values)
        else:
            assert left.columns[name] == values
    assert set(left.metadata) == set(right.metadata)
    for key, values in right.metadata.items():
        if isinstance(values, np.ndarray):
            assert_array_equal(as_array(left.metadata[key]), values)
        else:
            assert left.metadata[key] == values


def test_threads_never_change_batch_composition() -> None:
    """Same seed, same file: identical batches at any thread count."""

    def batches(threads: int | None) -> list[oxyz.Batch]:
        return list(
            oxyz.iter_batches(
                VARYING, atoms_per_batch=4, shuffle=True, seed=7, threads=threads
            )
        )

    serial = batches(1)
    for threads in (None, 4):
        for left, right in zip(batches(threads), serial, strict=True):
            assert_batches_equal(left, right)


def test_read_batch_ignores_damage_past_the_last_requested_frame(
    tmp_path: Path,
) -> None:
    """The partial-read promise: only the needed file prefix is inspected."""
    path = tmp_path / "tail.extxyz"
    good = "1\nProperties=species:S:1:pos:R:3 energy=-1\nH 0 0 0\n"
    path.write_text(good * 2 + "garbage\n")

    batch = oxyz.read_batch(path, [0, 1])
    assert batch.n_frames == 2

    # A whole-file read must still reject the damage.
    with pytest.raises(ValueError, match="invalid atom count"):
        oxyz.read_frames(path)


def test_read_batch_out_of_range_raises_index_error(tmp_path: Path) -> None:
    path = tmp_path / "short.extxyz"
    path.write_text("1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n")

    with pytest.raises(IndexError, match="frame index 3 out of range"):
        oxyz.read_batch(path, [0, 3])


def test_read_batch_negative_index_raises_index_error() -> None:
    # Negative indices are not supported; they raise the documented
    # IndexError rather than leaking pyo3's OverflowError.
    with pytest.raises(IndexError, match="frame index -1 out of range"):
        oxyz.read_batch(VARYING, [0, -1])


def test_read_batch_threads_are_equivalent() -> None:
    serial = oxyz.read_batch(VARYING, [2, 0, 1], threads=1)
    parallel = oxyz.read_batch(VARYING, [2, 0, 1], threads=4)
    assert_batches_equal(parallel, serial)


def test_sequential_batches_match_across_thread_counts() -> None:
    streamed = list(oxyz.iter_batches(VARYING, frames_per_batch=2, threads=1))
    planned = list(oxyz.iter_batches(VARYING, frames_per_batch=2))
    for left, right in zip(planned, streamed, strict=True):
        assert_batches_equal(left, right)


def test_int_real_metadata_promotes_to_float(tmp_path: Path) -> None:
    path = tmp_path / "promote.extxyz"
    path.write_text(
        "1\nProperties=species:S:1:pos:R:3 energy=-1\nH 0 0 0\n"
        "1\nProperties=species:S:1:pos:R:3 energy=-1.5\nH 0 0 0\n"
    )
    (batch,) = oxyz.iter_batches(path, frames_per_batch=2)

    energy = as_array(batch.metadata["energy"])
    assert energy.dtype == np.float64
    assert_allclose(energy, [-1.0, -1.5])


def test_schema_drift_within_a_batch_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "drift.extxyz"
    path.write_text(
        "1\nProperties=species:S:1:pos:R:3 energy=-1\nH 0 0 0\n"
        "1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n"
    )
    with pytest.raises(ValueError, match="missing metadata"):
        list(oxyz.iter_batches(path, frames_per_batch=2))

    # Batches that never span the drift are still readable.
    assert len(list(oxyz.iter_batches(path, frames_per_batch=1))) == 2
