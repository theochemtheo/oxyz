from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

import atomflow._rust as _rust


@dataclass(frozen=True, slots=True)
class Frame:
    numbers: npt.NDArray[np.uint8]
    positions: npt.NDArray[np.float64]
    forces: npt.NDArray[np.float64]
    energy: float
    cell: npt.NDArray[np.float64]
    stress: npt.NDArray[np.float64]
    pbc: npt.NDArray[np.bool_]


def read_first_frame(path: str | Path) -> Frame:
    data = _rust.read_first_frame(str(path))

    return Frame(
        numbers=data["numbers"],
        positions=data["positions"],
        forces=data["forces"],
        energy=data["energy"],
        cell=data["cell"],
        stress=data["stress"],
        pbc=data["pbc"],
    )
