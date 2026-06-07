from __future__ import annotations

from typing import TypedDict

import numpy as np
import numpy.typing as npt

class FrameData(TypedDict):
    numbers: npt.NDArray[np.uint8]
    positions: npt.NDArray[np.float64]
    forces: npt.NDArray[np.float64]
    energy: float
    cell: npt.NDArray[np.float64]
    stress: npt.NDArray[np.float64]
    pbc: npt.NDArray[np.bool_]

def read_first_frame(path: str) -> FrameData: ...
