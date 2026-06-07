from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import atomflow._rust as _rust


@dataclass(frozen=True, slots=True)
class Frame:
    numbers: list[int]
    positions: list[list[float]]
    forces: list[list[float]]
    energy: float
    cell: list[list[float]]
    stress: list[float]
    pbc: list[bool]


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
