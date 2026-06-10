from __future__ import annotations

from typing import TypedDict

import numpy as np

__build_profile__: str

ColumnValues = np.ndarray | list[str] | list[list[str]]
MetadataValue = float | int | bool | str | np.ndarray | list[str]

class FrameData(TypedDict):
    n_atoms: int
    columns: dict[str, ColumnValues]
    metadata: dict[str, MetadataValue]

class ScanData(TypedDict):
    offsets: np.ndarray
    n_atoms: np.ndarray

class FrameIter:
    def __init__(self, path: str) -> None: ...
    def __iter__(self) -> FrameIter: ...
    def __next__(self) -> FrameData: ...

class IndexedFrames:
    def __init__(self, path: str) -> None: ...
    def __len__(self) -> int: ...
    def get(self, frame_index: int) -> FrameData: ...

def read_first_frame(path: str) -> FrameData: ...
def read_frames(path: str) -> list[FrameData]: ...
def infer_schema(path: str) -> str: ...
def scan(path: str) -> ScanData: ...
