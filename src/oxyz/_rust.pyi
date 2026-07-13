from typing import NotRequired, TypedDict, final

import numpy as np

__build_profile__: str

class OxyzError(ValueError):
    """Base class for every error oxyz raises.

    A `ValueError` subclass, so `except ValueError` still catches everything;
    `except oxyz.OxyzError` narrows to errors this package raised.
    """

class ParseError(OxyzError):
    """Raised when extxyz content cannot be parsed.

    An `OxyzError` (and so a `ValueError`) subclass. Carries the location of
    the offending input as attributes — `frame_index`, `line`, and `column`
    (the 1-based character column of the offending token within its line) —
    each `None` when the parser cannot pin that dimension down, so callers can
    find the bad frame without parsing the message string.
    """

    frame_index: int | None
    line: int | None
    column: int | None

type ColumnValues = np.ndarray | list[str] | list[list[str]]
type MetadataValue = float | int | bool | str | np.ndarray | list[str]

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

class DeviationData(TypedDict):
    axis: str
    name: str
    deviation: str
    expected: str
    found: str | None

# A projection plan crosses as a tuple of two lists, columns then metadata,
# each holding one per-field tuple built in oxyz._project. A column field is
# name, letter, width, required, fill-or-None; a metadata field carries a shape
# tuple in place of width (empty for a scalar, length-one for an array).
type ProjectionPlan = tuple[list[tuple], list[tuple]]
# A dropped frame has None in place of its FrameData; deviations report why.
type ProjectedFrame = tuple[FrameData | None, list[DeviationData]]
# A projected batch: the survivors' data, their file indices, and a
# (frame_index, deviations) report per requested frame (survivors and drops).
type ProjectedBatch = tuple[BatchData, list[int], list[tuple[int, list[DeviationData]]]]

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
@final
class FrameIter:
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
    def __new__(cls, path: str, with_volume: bool = False) -> IndexedFrames: ...
    def __len__(self) -> int: ...
    @property
    def n_atoms(self) -> np.ndarray: ...
    @property
    def volumes(self) -> np.ndarray | None: ...
    def get(self, frame_index: int) -> FrameData: ...
    def get_batch(
        self, indices: list[int], threads: int | None = None
    ) -> BatchData: ...
    def get_batch_projected(
        self, indices: list[int], plan: ProjectionPlan, threads: int | None = None
    ) -> ProjectedBatch: ...

@final
class BatchIter:
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
def build_batch(frames: list[FrameData]) -> BatchData: ...
def read_frames_projected(
    path: str,
    threads: int | None = None,
    compression: str = "infer",
    member: str | None = None,
    *,
    plan: ProjectionPlan,
) -> list[ProjectedFrame]: ...
def read_first_frame_projected(
    path: str,
    compression: str = "infer",
    member: str | None = None,
    *,
    plan: ProjectionPlan,
) -> ProjectedFrame: ...
def read_frames_projected_reader(
    source: object,
    codec: str,
    member: str | None = None,
    threads: int | None = None,
    *,
    plan: ProjectionPlan,
) -> list[ProjectedFrame]: ...
def read_first_frame_projected_reader(
    source: object,
    codec: str,
    member: str | None = None,
    *,
    plan: ProjectionPlan,
) -> ProjectedFrame: ...
def read_batch_projected(
    path: str,
    indices: list[int] | None = None,
    threads: int | None = None,
    compression: str = "infer",
    member: str | None = None,
    *,
    plan: ProjectionPlan,
) -> ProjectedBatch: ...
def read_batch_projected_reader(
    source: object,
    codec: str,
    indices: list[int] | None = None,
    threads: int | None = None,
    member: str | None = None,
    *,
    plan: ProjectionPlan,
) -> ProjectedBatch: ...
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
def detect_codec(name: str, head: bytes | None = None) -> str: ...

# Reader entries: source is any iterator yielding bytes (e.g. obstore's stream).
# codec is one of "plain", "gzip", "zstd", "tar", "tar.gz", "zip".
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
) -> None: ...

@final
class FrameWriter:
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
