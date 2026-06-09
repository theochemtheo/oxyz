from __future__ import annotations

from typing import TypedDict

import numpy as np

ColumnValues = np.ndarray | list[str] | list[list[str]]
MetadataValue = float | int | bool | str | np.ndarray | list[str]

class FrameData(TypedDict):
    n_atoms: int
    columns: dict[str, ColumnValues]
    metadata: dict[str, MetadataValue]

def read_first_frame(path: str) -> FrameData: ...
