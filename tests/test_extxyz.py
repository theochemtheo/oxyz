from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import atomflow

DATA_DIR = Path(__file__).parent / "data"

CORPUS = sorted(path for ext in ("*.xyz", "*.extxyz") for path in DATA_DIR.glob(ext))


def as_array(value: object) -> np.ndarray:
    """Assert that ``value`` is an ndarray and re-type it for ty.

    Exists only because of a current ty limitation: isinstance-narrowing
    ``np.ndarray`` out of a union yields a type that fails numpy's
    ``assert_allclose`` overloads, even though a plain ``np.ndarray`` passes.
    Delete this helper (plain ``assert isinstance`` is enough) once
    test_ty_canary.py fails.
    """
    assert isinstance(value, np.ndarray)
    return value


@pytest.mark.parametrize("path", CORPUS, ids=lambda path: path.name)
def test_every_fixture_converts_to_python(path: Path) -> None:
    frame = atomflow.read_first_frame(path)

    assert frame.n_atoms > 0
    assert frame.columns
    for values in frame.columns.values():
        assert len(values) == frame.n_atoms


def test_read_frames_trajectory() -> None:
    frames = atomflow.read_frames(DATA_DIR / "varying_atom_counts.xyz")

    assert [frame.n_atoms for frame in frames] == [3, 1, 2]
    assert [frame.metadata["energy"] for frame in frames] == [-76.3, -13.6, -31.8]

    last_pos = as_array(frames[2].columns["pos"])
    assert last_pos.shape == (2, 3)


def test_read_frames_error_carries_frame_index(tmp_path: Path) -> None:
    text = (DATA_DIR / "varying_atom_counts.xyz").read_text()
    broken = tmp_path / "broken.xyz"
    broken.write_text(text + "not-a-count\n")

    with pytest.raises(ValueError, match="frame 3"):
        atomflow.read_frames(broken)


@pytest.mark.parametrize("path", CORPUS, ids=lambda path: path.name)
def test_parallel_read_frames_matches_serial(path: Path) -> None:
    serial = atomflow.read_frames(path, threads=1)
    parallel = atomflow.read_frames(path, threads=4)

    assert len(parallel) == len(serial)
    for left, right in zip(parallel, serial, strict=True):
        assert left.n_atoms == right.n_atoms
        assert list(left.columns) == list(right.columns)
        for name, values in right.columns.items():
            if isinstance(values, np.ndarray):
                assert_array_equal(as_array(left.columns[name]), values)
            else:
                assert left.columns[name] == values
        assert list(left.metadata) == list(right.metadata)


def test_iter_frames_streams_the_trajectory() -> None:
    frames = atomflow.iter_frames(DATA_DIR / "varying_atom_counts.xyz")

    assert [frame.n_atoms for frame in frames] == [3, 1, 2]


def test_iter_frames_yields_good_frames_then_raises_then_fuses(tmp_path: Path) -> None:
    text = (DATA_DIR / "varying_atom_counts.xyz").read_text()
    broken = tmp_path / "broken.xyz"
    broken.write_text(text + "not-a-count\n")

    frames = atomflow.iter_frames(broken)
    assert [next(frames).n_atoms for _ in range(3)] == [3, 1, 2]
    with pytest.raises(ValueError, match="frame 3"):
        next(frames)
    with pytest.raises(StopIteration):
        next(frames)


def test_two_iterators_are_independent() -> None:
    path = DATA_DIR / "varying_atom_counts.xyz"
    first, second = atomflow.iter_frames(path), atomflow.iter_frames(path)

    next(first)
    assert next(first).n_atoms == 1
    assert next(second).n_atoms == 3


def test_scan_reports_structure_and_statistics() -> None:
    index = atomflow.scan(DATA_DIR / "varying_atom_counts.xyz")

    assert index.n_frames == 3
    assert index.total_atoms == 6
    assert list(index.n_atoms) == [3, 1, 2]
    assert (index.min_atoms, index.max_atoms) == (1, 3)
    assert index.mean_atoms == 2.0
    assert index.median_atoms == 2.0
    assert index.std_atoms == pytest.approx((2 / 3) ** 0.5)


def test_scan_rejects_structural_garbage(tmp_path: Path) -> None:
    text = (DATA_DIR / "varying_atom_counts.xyz").read_text()
    broken = tmp_path / "broken.xyz"
    broken.write_text(text + "not-a-count\n")

    with pytest.raises(ValueError, match="frame 3"):
        atomflow.scan(broken)


def test_infer_schema_report() -> None:
    report = atomflow.infer_schema(DATA_DIR / "varying_atom_counts.xyz")

    assert "3 frames, 6 atoms (min 1, max 3)" in report
    assert "pos: R:3 (3/3 frames)" in report
    assert "energy: Real (3/3 frames)" in report


def test_read_first_frame_simple_extxyz() -> None:
    frame = atomflow.read_first_frame(DATA_DIR / "simple.extxyz")

    assert frame.n_atoms == 1
    assert list(frame.columns) == ["species", "pos", "forces"]

    assert frame.columns["species"] == ["H"]

    pos = as_array(frame.columns["pos"])
    assert pos.dtype == np.float64
    assert pos.shape == (1, 3)
    assert pos.flags.c_contiguous
    assert_allclose(pos, np.array([[0.0, 0.0, 0.0]]))

    forces = as_array(frame.columns["forces"])
    assert forces.shape == (1, 3)
    assert_allclose(forces, np.array([[0.0, 0.0, 0.0]]))

    assert frame.metadata["energy"] == -1.0
    assert isinstance(frame.metadata["energy"], float)

    # Lattice arrives flat, in as-written order; reshaping and reordering are
    # the normalisation layer's job.
    lattice = as_array(frame.metadata["Lattice"])
    assert lattice.shape == (9,)
    assert_allclose(lattice, np.array([15.0, 0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 15.0]))

    stress = as_array(frame.metadata["stress"])
    assert stress.shape == (6,)
    assert_allclose(stress, np.zeros(6))

    pbc = as_array(frame.metadata["pbc"])
    assert pbc.dtype == np.bool_
    assert_array_equal(pbc, np.array([True, True, True]))

    # Properties is consumed into columns, not duplicated in metadata.
    assert "Properties" not in frame.metadata


def test_nonorthogonal_lattice_preserved_as_written() -> None:
    frame = atomflow.read_first_frame(DATA_DIR / "nonorthogonal.extxyz")

    lattice = as_array(frame.metadata["Lattice"])
    assert_allclose(lattice, np.array([10.0, 1.0, 2.0, 0.0, 11.0, 3.0, 0.0, 0.0, 12.0]))

    pos = as_array(frame.columns["pos"])
    assert pos.shape == (2, 3)
    assert_allclose(pos, np.array([[0.0, 0.1, 0.2], [3.0, 3.1, 3.2]]))


def test_integer_and_string_columns() -> None:
    frame = atomflow.read_first_frame(DATA_DIR / "id_and_selection.extxyz")

    assert list(frame.columns) == ["id", "species", "pos", "selection"]

    ids = as_array(frame.columns["id"])
    assert ids.dtype == np.int64
    assert_array_equal(ids, np.array([10, 11, 12]))

    assert frame.columns["species"] == ["Si", "Si", "O"]

    selection = as_array(frame.columns["selection"])
    assert_array_equal(selection, np.array([1, 0, 1]))


def test_metadata_value_typing() -> None:
    frame = atomflow.read_first_frame(
        DATA_DIR / "quoted_strings_booleans_scalars.extxyz"
    )

    assert frame.metadata["source"] == "generated for parser study"
    assert frame.metadata["split"] == "train"
    assert frame.metadata["converged"] is True
    assert frame.metadata["frozen"] is False
    assert frame.metadata["temperature"] == 298.15

    # bool is a subclass of int in Python, so check step isn't a bool too.
    step = frame.metadata["step"]
    assert step == 12
    assert isinstance(step, int)
    assert not isinstance(step, bool)


def test_bracket_array_metadata() -> None:
    frame = atomflow.read_first_frame(DATA_DIR / "newstyle_array_metadata.extxyz")

    kpoints = as_array(frame.metadata["kpoints"])
    assert kpoints.dtype == np.int64
    assert_array_equal(kpoints, np.array([2, 2, 1]))

    cutoffs = as_array(frame.metadata["cutoffs"])
    assert cutoffs.dtype == np.float64
    assert_allclose(cutoffs, np.array([4.5, 5.0]))

    assert frame.metadata["tags"] == ["slab", "relaxed"]


def test_mace_training_schema_names_preserved() -> None:
    frame = atomflow.read_first_frame(DATA_DIR / "mace_ref_energy_forces_stress.xyz")

    ref_forces = as_array(frame.columns["REF_forces"])
    assert ref_forces.shape == (3, 3)

    assert frame.metadata["REF_energy"] == -76.123
    assert frame.metadata["config_type"] == "Default"

    ref_stress = as_array(frame.metadata["REF_stress"])
    assert ref_stress.shape == (6,)
