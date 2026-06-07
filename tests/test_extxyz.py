from __future__ import annotations

from pathlib import Path

import atomflow

DATA_DIR = Path(__file__).parent / "data"


def test_read_first_frame_simple_extxyz() -> None:
    frame = atomflow.read_first_frame(DATA_DIR / "simple.extxyz")

    assert frame.numbers == [1]
    assert frame.positions == [[0.0, 0.0, 0.0]]
    assert frame.forces == [[0.0, 0.0, 0.0]]
    assert frame.energy == -1.0

    assert frame.cell == [
        [15.0, 0.0, 0.0],
        [0.0, 15.0, 0.0],
        [0.0, 0.0, 15.0],
    ]

    assert frame.stress == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert frame.pbc == [True, True, True]
