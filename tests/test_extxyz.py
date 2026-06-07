from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

import atomflow

DATA_DIR = Path(__file__).parent / "data"


def test_read_first_frame_simple_extxyz() -> None:
    frame = atomflow.read_first_frame(DATA_DIR / "simple.extxyz")

    assert frame.numbers.dtype == np.uint8
    assert frame.numbers.shape == (1,)
    assert_array_equal(frame.numbers, np.array([1], dtype=np.uint8))

    assert frame.positions.dtype == np.float64
    assert frame.positions.shape == (1, 3)
    assert_allclose(frame.positions, np.array([[0.0, 0.0, 0.0]]))

    assert frame.forces.dtype == np.float64
    assert frame.forces.shape == (1, 3)
    assert_allclose(frame.forces, np.array([[0.0, 0.0, 0.0]]))

    assert frame.energy == -1.0

    assert frame.cell.dtype == np.float64
    assert frame.cell.shape == (3, 3)
    assert_allclose(
        frame.cell,
        np.array(
            [
                [15.0, 0.0, 0.0],
                [0.0, 15.0, 0.0],
                [0.0, 0.0, 15.0],
            ]
        ),
    )

    assert frame.stress.dtype == np.float64
    assert frame.stress.shape == (6,)
    assert_allclose(frame.stress, np.zeros(6))

    assert frame.pbc.dtype == np.bool_
    assert frame.pbc.shape == (3,)
    assert_array_equal(frame.pbc, np.array([True, True, True], dtype=np.bool_))


def test_read_first_frame_nonorthogonal_extxyz_row_major_arrays() -> None:
    frame = atomflow.read_first_frame(DATA_DIR / "nonorthogonal.extxyz")

    assert frame.numbers.dtype == np.uint8
    assert frame.numbers.shape == (2,)
    assert frame.numbers.flags.c_contiguous
    assert_array_equal(frame.numbers, np.array([1, 1], dtype=np.uint8))

    assert frame.positions.dtype == np.float64
    assert frame.positions.shape == (2, 3)
    assert frame.positions.flags.c_contiguous
    assert_allclose(
        frame.positions,
        np.array(
            [
                [0.0, 0.1, 0.2],
                [3.0, 3.1, 3.2],
            ],
            dtype=np.float64,
        ),
    )

    assert frame.forces.dtype == np.float64
    assert frame.forces.shape == (2, 3)
    assert frame.forces.flags.c_contiguous
    assert_allclose(
        frame.forces,
        np.array(
            [
                [1.0, 1.1, 1.2],
                [-1.0, -1.1, -1.2],
            ],
            dtype=np.float64,
        ),
    )

    assert frame.cell.dtype == np.float64
    assert frame.cell.shape == (3, 3)
    assert frame.cell.flags.c_contiguous
    assert_allclose(
        frame.cell,
        np.array(
            [
                [10.0, 0.0, 0.0],
                [1.0, 11.0, 0.0],
                [2.0, 3.0, 12.0],
            ],
            dtype=np.float64,
        ),
    )

    assert frame.stress.dtype == np.float64
    assert frame.stress.shape == (6,)
    assert frame.stress.flags.c_contiguous
    assert_allclose(frame.stress, np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))

    assert frame.pbc.dtype == np.bool_
    assert frame.pbc.shape == (3,)
    assert frame.pbc.flags.c_contiguous
    assert_array_equal(frame.pbc, np.array([True, False, True], dtype=np.bool_))
