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

class BatchData(TypedDict):
    offsets: np.ndarray
    columns: dict[str, ColumnValues]
    metadata: dict[str, ColumnValues]

class ColumnVariantData(TypedDict):
    kind: str
    width: int
    frames: int

class MetadataVariantData(TypedDict):
    kind: str
    shape: tuple[int, ...]
    frames: int

class ColumnSchemaData(TypedDict):
    name: str
    variants: list[ColumnVariantData]
    frames_present: int
    unified: tuple[str, int] | None

class MetadataSchemaData(TypedDict):
    key: str
    variants: list[MetadataVariantData]
    frames_present: int
    unified: tuple[str, tuple[int, ...]] | None

class SchemaData(TypedDict):
    n_frames: int
    total_atoms: int
    min_atoms: int | None
    max_atoms: int | None
    columns: list[ColumnSchemaData]
    metadata: list[MetadataSchemaData]
    is_consistent: bool
    report: str

class FrameIter:
    def __init__(self, path: str) -> None: ...
    def __iter__(self) -> FrameIter: ...
    def __next__(self) -> FrameData: ...

class IndexedFrames:
    def __init__(self, path: str) -> None: ...
    def __len__(self) -> int: ...
    @property
    def n_atoms(self) -> np.ndarray: ...
    def get(self, frame_index: int) -> FrameData: ...
    def get_batch(
        self, indices: list[int], threads: int | None = None
    ) -> BatchData: ...

class BatchIter:
    def __init__(self, path: str, frames_per_batch: int) -> None: ...
    def __iter__(self) -> BatchIter: ...
    def __next__(self) -> BatchData: ...

def read_first_frame(path: str) -> FrameData: ...
def read_frames(path: str, threads: int | None = None) -> list[FrameData]: ...
def read_batch(
    path: str, indices: list[int], threads: int | None = None
) -> BatchData: ...
def infer_schema(path: str) -> SchemaData: ...
def scan(path: str) -> ScanData: ...
