from typing import NotRequired, TypedDict, final

import numpy as np

# "debug" or "release", so callers can refuse a debug build.
__build_profile__: str

class OxyzError(ValueError):
    """Base class for every error oxyz raises; a `ValueError` subclass."""

class ParseError(OxyzError):
    """Raised when extxyz content cannot be parsed.

    Carries `frame_index`, `line`, and `column` (the 1-based character column
    of the offending token), each `None` when it cannot be pinned down.
    """

    frame_index: int | None
    line: int | None
    column: int | None

type ColumnValues = np.ndarray | list[str] | list[list[str]]
# 2-D string arrays cross as list[list[str]], mirroring ColumnValues.
type MetadataValue = float | int | bool | str | np.ndarray | list[str] | list[list[str]]

class FrameData(TypedDict):
    """Raw per-frame crossing dict; the Python surface wraps this as `Frame`."""

    n_atoms: int
    columns: dict[str, ColumnValues]
    metadata: dict[str, MetadataValue]

class ScanData(TypedDict):
    """Raw crossing dict from `scan`/`scan_reader`; wrapped as `FrameIndex`."""

    offsets: np.ndarray
    n_atoms: np.ndarray
    # Only with scan(..., with_volume=True); NaN where a frame has no Lattice.
    volumes: NotRequired[np.ndarray]

class BatchData(TypedDict):
    """Raw crossing dict for a concatenated batch; wrapped as `Batch`."""

    offsets: np.ndarray
    columns: dict[str, ColumnValues]
    metadata: dict[str, ColumnValues]

class DeviationData(TypedDict):
    """One projection deviation, before the Python layer adds `frame_index`."""

    axis: str
    name: str
    deviation: str
    expected: str
    found: str | None

# (columns, metadata), each a list of field tuples built in oxyz._project:
# (name, letter, width or shape, required, fill or None).
type ProjectionPlan = tuple[list[tuple], list[tuple]]
# A dropped frame has None in place of its FrameData; deviations report why.
type ProjectedFrame = tuple[FrameData | None, list[DeviationData]]
# Survivors' data, their file indices, and a report per requested frame.
type ProjectedBatch = tuple[BatchData, list[int], list[tuple[int, list[DeviationData]]]]

class ColumnVariantData(TypedDict):
    """Raw crossing dict for one observed (kind, width) of a column."""

    kind: str
    width: int
    frames: int

class MetadataVariantData(TypedDict):
    """Raw crossing dict for one observed (kind, shape) of a metadata key."""

    kind: str
    shape: tuple[int, ...]
    frames: int

class ColumnSchemaData(TypedDict):
    """Raw crossing dict for one per-atom column's observations."""

    name: str
    variants: list[ColumnVariantData]
    frames_present: int
    unified: tuple[str, int] | None

class MetadataSchemaData(TypedDict):
    """Raw crossing dict for one metadata key's observations."""

    key: str
    variants: list[MetadataVariantData]
    frames_present: int
    unified: tuple[str, tuple[int, ...]] | None

class SchemaData(TypedDict):
    """Raw crossing dict from `infer_schema`; wrapped as `Schema`."""

    n_frames: int
    total_atoms: int
    min_atoms: int | None
    max_atoms: int | None
    n_atoms: np.ndarray
    columns: list[ColumnSchemaData]
    metadata: list[MetadataSchemaData]
    is_consistent: bool
    report: str

# `compression`: "infer", "none", "gzip", "zstd", "zip", "tar", "tar.gz" or
# "tar.zst"; `member` names an archive entry.
@final
class FrameIter:
    """Streaming iterator: one frame per `__next__`; fused after an error or EOF."""

    def __new__(
        cls, path: str, compression: str = "infer", member: str | None = None
    ) -> FrameIter: ...
    def __iter__(self) -> FrameIter: ...
    def __next__(self) -> FrameData: ...
    @staticmethod
    def from_reader(
        source: object, codec: str, member: str | None = None
    ) -> FrameIter: ...

@final
class IndexedFrames:
    """Random-access reader: scans on construction, then `get(i)` parses one frame."""

    def __new__(cls, path: str, with_volume: bool = False) -> IndexedFrames: ...
    def __len__(self) -> int: ...
    @property
    def n_atoms(self) -> np.ndarray:
        """Declared atom count per frame, from the scan done at construction."""

    @property
    def volumes(self) -> np.ndarray | None:
        """Per-frame `|det(Lattice)|`, or `None` unless opened `with_volume`."""

    def get(self, frame_index: int) -> FrameData: ...
    def get_batch(
        self, indices: list[int], threads: int | None = None
    ) -> BatchData: ...
    def get_batch_projected(
        self, indices: list[int], plan: ProjectionPlan, threads: int | None = None
    ) -> ProjectedBatch: ...

@final
class BatchIter:
    """Streaming batch iterator; the final batch may be smaller. Fused after errors."""

    def __new__(
        cls,
        path: str,
        frames_per_batch: int,
        compression: str = "infer",
        member: str | None = None,
    ) -> BatchIter: ...
    def __iter__(self) -> BatchIter: ...
    def __next__(self) -> BatchData: ...
    @staticmethod
    def from_reader(
        source: object,
        frames_per_batch: int,
        codec: str,
        member: str | None = None,
    ) -> BatchIter: ...

@final
class FrameIterProjected:
    """Projected `FrameIter`: yields `(FrameData | None, deviations)`."""

    def __new__(
        cls,
        path: str,
        plan: ProjectionPlan,
        compression: str = "infer",
        member: str | None = None,
    ) -> FrameIterProjected: ...
    def __iter__(self) -> FrameIterProjected: ...
    def __next__(self) -> ProjectedFrame: ...
    @staticmethod
    def from_reader(
        source: object,
        plan: ProjectionPlan,
        codec: str,
        member: str | None = None,
    ) -> FrameIterProjected: ...

@final
class BatchIterProjected:
    """Projected `BatchIter`: yields `(BatchData, survivors, reports)`."""

    def __new__(
        cls,
        path: str,
        frames_per_batch: int,
        plan: ProjectionPlan,
        compression: str = "infer",
        member: str | None = None,
    ) -> BatchIterProjected: ...
    def __iter__(self) -> BatchIterProjected: ...
    def __next__(self) -> ProjectedBatch: ...
    @staticmethod
    def from_reader(
        source: object,
        frames_per_batch: int,
        plan: ProjectionPlan,
        codec: str,
        member: str | None = None,
    ) -> BatchIterProjected: ...

def read_first_frame(
    path: str, compression: str = "infer", member: str | None = None
) -> FrameData:
    """Read the first frame."""

def read_frames(
    path: str,
    threads: int | None = None,
    compression: str = "infer",
    member: str | None = None,
) -> list[FrameData]:
    """Read every frame. `threads=1` is serial; output and errors are identical."""

def read_batch(
    path: str,
    indices: list[int] | None = None,
    threads: int | None = None,
    compression: str = "infer",
    member: str | None = None,
) -> BatchData:
    """Gather frames into one batch: the whole file, or `indices` in order."""

def build_batch(frames: list[FrameData]) -> BatchData:
    """Assemble a batch from parsed frame dicts; an empty list gives an empty batch."""

def read_frames_projected(
    path: str,
    threads: int | None = None,
    compression: str = "infer",
    member: str | None = None,
    *,
    plan: ProjectionPlan,
) -> list[ProjectedFrame]:
    """Projected `read_frames`: `(FrameData | None, deviations)` per frame."""

def read_first_frame_projected(
    path: str,
    compression: str = "infer",
    member: str | None = None,
    *,
    plan: ProjectionPlan,
) -> ProjectedFrame:
    """Projected variant of `read_first_frame`."""

def read_frames_projected_reader(
    source: object,
    codec: str,
    member: str | None = None,
    threads: int | None = None,
    *,
    plan: ProjectionPlan,
) -> list[ProjectedFrame]:
    """Reader-source variant of `read_frames_projected`."""

def read_first_frame_projected_reader(
    source: object,
    codec: str,
    member: str | None = None,
    *,
    plan: ProjectionPlan,
) -> ProjectedFrame:
    """Reader-source variant of `read_first_frame_projected`."""

def read_batch_projected(
    path: str,
    indices: list[int] | None = None,
    threads: int | None = None,
    compression: str = "infer",
    member: str | None = None,
    *,
    plan: ProjectionPlan,
) -> ProjectedBatch:
    """Projected `read_batch`: `(BatchData, survivors, reports)`."""

def read_batch_projected_reader(
    source: object,
    codec: str,
    indices: list[int] | None = None,
    threads: int | None = None,
    member: str | None = None,
    *,
    plan: ProjectionPlan,
) -> ProjectedBatch:
    """Reader-source variant of `read_batch_projected`."""

def infer_schema(
    path: str, compression: str = "infer", member: str | None = None
) -> SchemaData:
    """Infer the file's schema as one nested dict."""

def scan(
    path: str,
    with_volume: bool = False,
    compression: str = "infer",
    member: str | None = None,
) -> ScanData:
    """Structurally scan the file; `with_volume=True` adds `|det(Lattice)|`."""

def is_compressed(path: str, compression: str = "infer") -> bool:
    """Whether `path` would be read through a decompressing layer."""

def detect_codec(name: str, head: bytes | None = None) -> str:
    """Infer the codec name from a filename and optional header bytes."""

# `source` yields bytes; `codec` is a `detect_codec` name.
def read_frames_reader(
    source: object,
    codec: str,
    member: str | None = None,
    threads: int | None = None,
) -> list[FrameData]: ...
def read_first_frame_reader(
    source: object,
    codec: str,
    member: str | None = None,
) -> FrameData: ...
def scan_reader(
    source: object,
    codec: str,
    with_volume: bool = False,
    member: str | None = None,
) -> ScanData: ...
def infer_schema_reader(
    source: object,
    codec: str,
    member: str | None = None,
) -> SchemaData: ...
def read_batch_reader(
    source: object,
    codec: str,
    indices: list[int] | None = None,
    threads: int | None = None,
    member: str | None = None,
) -> BatchData: ...
def write(
    path: str,
    frames: list[FrameData],
    compression: str = "infer",
    level: int | None = None,
    append: bool = False,
    threads: int | None = None,
) -> None:
    """Write frames to `path`; the bytes do not depend on `threads`."""

@final
class FrameWriter:
    """Incremental writer behind `oxyz.Writer`.

    `batch=None` streams each frame; `batch=n` buffers `n` frames and
    serialises each full batch in parallel.
    """

    def __new__(
        cls,
        path: str,
        compression: str = "infer",
        level: int | None = None,
        append: bool = False,
        batch: int | None = None,
    ) -> FrameWriter: ...
    def write(self, frame: FrameData) -> None: ...
    def close(self) -> None: ...
