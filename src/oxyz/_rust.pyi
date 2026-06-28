from __future__ import annotations

from typing import NotRequired, TypedDict

import numpy as np

__build_profile__: str

class ParseError(ValueError):
    """Raised when extxyz content cannot be parsed.

    A `ValueError` subclass. Carries the location of the offending input as
    attributes — `frame_index`, `line_number`, `column` — each `None` when
    the parser cannot pin that dimension down, so callers can find the bad
    frame without parsing the message string.
    """

    frame_index: int | None
    line_number: int | None
    column: str | None

ColumnValues = np.ndarray | list[str] | list[list[str]]
MetadataValue = float | int | bool | str | np.ndarray | list[str]

class FrameData(TypedDict):
    n_atoms: int
    columns: dict[str, ColumnValues]
    metadata: dict[str, MetadataValue]

class ScanData(TypedDict):
    offsets: np.ndarray
    n_atoms: np.ndarray
    # Present only for scan(..., with_volume=True): per-frame |det(Lattice)|,
    # NaN where a frame has no Lattice.
    volumes: NotRequired[np.ndarray]

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
    n_atoms: np.ndarray
    columns: list[ColumnSchemaData]
    metadata: list[MetadataSchemaData]
    is_consistent: bool
    report: str

# `compression` is one of "infer", "none", "gzip", "zstd", "zip"; `member`
# names an entry inside an archive (.zip/.tar/.tar.gz).
class FrameIter:
    def __init__(
        self, path: str, compression: str = "infer", member: str | None = None
    ) -> None: ...
    def __iter__(self) -> FrameIter: ...
    def __next__(self) -> FrameData: ...

class IndexedFrames:
    def __init__(self, path: str, with_volume: bool = False) -> None: ...
    def __len__(self) -> int: ...
    @property
    def n_atoms(self) -> np.ndarray: ...
    @property
    def volumes(self) -> np.ndarray | None: ...
    def get(self, frame_index: int) -> FrameData: ...
    def get_batch(
        self, indices: list[int], threads: int | None = None
    ) -> BatchData: ...

class BatchIter:
    def __init__(
        self,
        path: str,
        frames_per_batch: int,
        compression: str = "infer",
        member: str | None = None,
    ) -> None: ...
    def __iter__(self) -> BatchIter: ...
    def __next__(self) -> BatchData: ...

def read_first_frame(
    path: str, compression: str = "infer", member: str | None = None
) -> FrameData: ...
def read_frames(
    path: str,
    threads: int | None = None,
    compression: str = "infer",
    member: str | None = None,
) -> list[FrameData]: ...
def read_batch(
    path: str,
    indices: list[int] | None = None,
    threads: int | None = None,
    compression: str = "infer",
    member: str | None = None,
) -> BatchData: ...
def infer_schema(
    path: str, compression: str = "infer", member: str | None = None
) -> SchemaData: ...
def scan(
    path: str,
    with_volume: bool = False,
    compression: str = "infer",
    member: str | None = None,
) -> ScanData: ...
def is_compressed(path: str, compression: str = "infer") -> bool: ...
def write(
    path: str,
    frames: list[FrameData],
    compression: str = "infer",
    level: int | None = None,
    append: bool = False,
) -> None: ...

class FrameWriter:
    def __init__(
        self,
        path: str,
        compression: str = "infer",
        level: int | None = None,
        append: bool = False,
    ) -> None: ...
    def write(self, frame: FrameData) -> None: ...
    def close(self) -> None: ...
