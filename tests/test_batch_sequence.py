"""`Batch` as a sequence of frames: `len`, `batch[i]`, slicing, gathering.

Basic indexing shares memory, as numpy's does: `batch[i]` and `batch[a:b]` are
views onto the batch's numeric arrays. A step slice, an index list, or a mask
gathers rows from non-adjacent frames, so it copies. String columns are Python
lists and are always copied; scalars come out as the Python values
`oxyz.read` gives.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest

import oxyz
from oxyz import Batch, Frame

DATA_DIR = Path(__file__).parent / "data"
VARYING = DATA_DIR / "varying_atom_counts.xyz"  # 3, 1, 2 atoms; forces, energy

# Every fixture a batch can hold (one schema across its frames), covering
# string, bool, 2-D-string-metadata, and per-frame array metadata.
BATCHABLE = [
    "atomic_numbers_z.extxyz",
    "id_and_selection.extxyz",
    "mace_isolated_atom_and_head.xyz",
    "mace_ref_energy_forces_stress.xyz",
    "molecule_type_labels.extxyz",
    "newstyle_array_metadata.extxyz",
    "no_lattice_molecule.xyz",
    "per_atom_boolean.extxyz",
    "quoted_strings_booleans_scalars.extxyz",
    "schema_conformant.extxyz",
    "two_frame_same_schema.xyz",
    "varying_atom_counts.xyz",
    "varying_density.extxyz",
]


@pytest.fixture
def batch() -> Batch:
    return oxyz.read_batch(VARYING)


@pytest.mark.parametrize("name", BATCHABLE)
def test_each_frame_of_a_batch_is_the_frame_read_from_the_file(name: str) -> None:
    path = DATA_DIR / name
    batch = oxyz.read_batch(path)
    read = oxyz.read(path)

    assert len(batch) == len(read)
    iterated = list(batch)
    for i in range(len(batch)):
        expected = read[int(batch.frame_indices[i])]
        assert batch[i] == expected
        assert iterated[i] == expected
        # repr carries dtype, shape and Python scalar type, which == ignores.
        assert repr(batch[i]) == repr(expected)


def test_negative_index_counts_from_the_end(batch: Batch) -> None:
    assert batch[-1] == batch[2]
    assert batch[np.int64(1)] == batch[1]


@pytest.mark.parametrize("index", [3, -4])
def test_out_of_range_index_is_an_index_error(batch: Batch, index: int) -> None:
    with pytest.raises(IndexError, match="out of range for a batch of 3"):
        batch[index]


@pytest.mark.parametrize("key", [1.0, True], ids=["float", "bool"])
def test_a_float_or_bool_index_is_a_type_error(batch: Batch, key: object) -> None:
    with pytest.raises(TypeError):
        batch[key]  # ty: ignore[invalid-argument-type]


def test_a_frame_shares_the_batch_numeric_arrays(batch: Batch) -> None:
    second = batch[2]
    pos = batch.columns["pos"]
    assert isinstance(pos, np.ndarray)

    assert np.shares_memory(second["pos"], pos)
    assert np.shares_memory(second["Lattice"], batch.metadata["Lattice"])
    second.columns["forces"][:] = 7.0  # ty: ignore[invalid-assignment]
    assert np.all(np.asarray(batch.columns["forces"])[4:6] == 7.0)


def test_a_frame_copies_strings_and_unboxes_scalars(batch: Batch) -> None:
    first = batch[0]
    species = cast("list[str]", first.columns["species"])

    species[0] = "Xx"
    assert batch.columns["species"][0] != "Xx"
    assert type(first.metadata["energy"]) is float
    assert first.n_atoms == 3


def test_a_frame_copies_the_rows_of_string_array_metadata() -> None:
    batch = oxyz.read_batch(DATA_DIR / "newstyle_array_metadata.extxyz")
    tags = cast("list[str]", batch[0].metadata["tags"])

    tags.append("extra")
    assert batch.metadata["tags"][0] == batch[0].metadata["tags"]
    assert "extra" not in batch.metadata["tags"][0]


def test_a_frame_has_its_own_dicts(batch: Batch) -> None:
    first = batch[0]
    first.metadata["note"] = "mine"

    assert "note" not in batch.metadata
    assert "note" not in batch[0].metadata


def test_a_contiguous_slice_is_a_batch_of_views(batch: Batch) -> None:
    tail = batch[1:]

    assert tail == oxyz.read_batch(VARYING, [1, 2])
    assert tail.offsets.tolist() == [0, 1, 3]
    assert np.shares_memory(tail["pos"], batch["pos"])
    assert np.shares_memory(tail["energy"], batch["energy"])


@pytest.mark.parametrize(
    "key",
    [
        slice(None, None, 2),
        [0, 2],
        (0, 2),
        np.array([0, 2]),
        np.array([True, False, True]),
    ],
    ids=["step-slice", "list", "tuple", "array", "mask"],
)
def test_a_non_contiguous_selection_gathers_a_copy(batch: Batch, key: object) -> None:
    picked = batch[key]  # ty: ignore[invalid-argument-type]

    assert picked == oxyz.read_batch(VARYING, [0, 2])
    assert not np.shares_memory(picked["pos"], batch["pos"])


def test_a_gather_may_repeat_and_reorder_frames(batch: Batch) -> None:
    picked = batch[[2, 0, 2]]

    assert picked.frame_indices.tolist() == [2, 0, 2]
    assert picked[0] == picked[2] == batch[2]
    assert picked[1] == batch[0]


def test_a_mask_filters_by_a_per_frame_field(batch: Batch) -> None:
    energy = np.asarray(batch["energy"])
    low = batch[energy < energy.max()]

    assert len(low) == 2
    assert np.all(np.asarray(low["energy"]) < energy.max())


def test_a_mask_of_the_wrong_length_is_an_index_error(batch: Batch) -> None:
    with pytest.raises(IndexError, match="mask"):
        batch[np.array([True, False])]


@pytest.mark.parametrize("positions", [[0, 3], [0, -4]])
def test_an_out_of_range_gather_is_an_index_error(
    batch: Batch, positions: list[int]
) -> None:
    with pytest.raises(IndexError, match="out of range"):
        batch[positions]


def test_a_gather_counts_negative_positions_from_the_end(batch: Batch) -> None:
    assert batch[[-1, 0]] == batch[[2, 0]]


def test_a_gather_keeps_file_provenance_not_batch_positions(batch: Batch) -> None:
    # In `batch[1:]`, position 0 is file frame 1.
    assert batch[1:][[1, 0]].frame_indices.tolist() == [2, 1]
    assert batch[1:][0] == oxyz.read(VARYING, 1)


@pytest.mark.parametrize("key", [slice(2, 2), [], np.array([], dtype=int)])
def test_an_empty_selection_keeps_every_key_with_no_rows(
    batch: Batch, key: object
) -> None:
    empty = batch[key]  # ty: ignore[invalid-argument-type]

    assert len(empty) == 0
    assert empty.keys() == batch.keys()
    assert np.asarray(empty["pos"]).shape == (0, 3)
    assert np.asarray(empty["Lattice"]).shape == (0, 9)
    assert empty.offsets.tolist() == [0]


def test_a_batch_iterates_its_frames_and_still_looks_names_up(batch: Batch) -> None:
    frames = list(batch)

    assert all(isinstance(f, Frame) for f in frames)
    assert [f.n_atoms for f in frames] == [3, 1, 2]
    assert "pos" in batch


def test_writing_a_batch_writes_its_frames(batch: Batch, tmp_path: Path) -> None:
    out = tmp_path / "out.extxyz"
    oxyz.write(out, batch)

    assert oxyz.read(out) == oxyz.read(VARYING)
