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
