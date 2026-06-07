from __future__ import annotations

from typing import TypedDict

class FrameData(TypedDict):
    numbers: list[int]
    positions: list[list[float]]
    forces: list[list[float]]
    energy: float
    cell: list[list[float]]
    stress: list[float]
    pbc: list[bool]

def read_first_frame(path: str) -> FrameData: ...
