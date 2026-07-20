from typing import NotRequired, TypedDict, final

import numpy as np

# "debug" or "release"; lets Python callers (e.g. the benchmark harness) refuse
# a debug build.
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
# np.ndarray covers scalar-width-1 arrays through 2-D numeric/bool arrays
# (shape (rows, cols)); 2-D string arrays cross as list[list[str]], mirroring
# ColumnValues.
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
    # Present only for scan(..., with_volume=True): per-frame |det(Lattice)|,
    # NaN where a frame has no Lattice.
    volumes: NotRequired[np.ndarray]

class BatchData(TypedDict):
    """Raw crossing dict for a concatenated batch; wrapped as `Batch`."""

    offsets: np.ndarray
    columns: dict[str, ColumnValues]
    metadata: dict[str, ColumnValues]

class DeviationData(TypedDict):
    """One projection-plan deviation for a single field, as the Rust projector
    emits it; the Python policy layer attaches `frame_index` and wraps it as a
    `Violation`.
    """

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
    """Raw crossing dict for one observed (kind, width) combination on a
    per-atom column; wrapped as `ColumnVariant`.
    """

    kind: str
    width: int
    frames: int

class MetadataVariantData(TypedDict):
    """Raw crossing dict for one observed (kind, shape) combination on a
    metadata key; wrapped as `MetadataVariant`.
    """

    kind: str
    shape: tuple[int, ...]
    frames: int

class ColumnSchemaData(TypedDict):
    """Raw crossing dict for everything observed about one per-atom column;
    wrapped as `ColumnSchema`.
    """

    name: str
    variants: list[ColumnVariantData]
    frames_present: int
    unified: tuple[str, int] | None

class MetadataSchemaData(TypedDict):
    """Raw crossing dict for everything observed about one metadata key;
    wrapped as `MetadataSchema`.
    """

    key: str
    variants: list[MetadataVariantData]
    frames_present: int
    unified: tuple[str, tuple[int, ...]] | None

class SchemaData(TypedDict):
    """Raw crossing dict from `infer_schema`/`infer_schema_reader`; wrapped as
    `Schema`.
    """

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
    """Streaming iterator: one frame parsed and converted per `__next__`.

    Owns the file handle; closes when the object is dropped. Fused after an
    error or EOF — only raises `StopIteration` from then on.
    """

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
    """Random-access reader: scans on construction, then `get(i)` seeks and
    parses single frames in any order.
    """

    def __new__(cls, path: str, with_volume: bool = False) -> IndexedFrames: ...
    def __len__(self) -> int: ...
    @property
    def n_atoms(self) -> np.ndarray:
        """Declared atom count per frame, from the scan done at construction."""

    @property
    def volumes(self) -> np.ndarray | None:
        """Per-frame cell volume `|det(Lattice)|` from the scan, or `None` if
        opened without `with_volume`; `NaN` for a frame with no `Lattice`.
        """

    def get(self, frame_index: int) -> FrameData: ...
    def get_batch(
        self, indices: list[int], threads: int | None = None
    ) -> BatchData: ...
    def get_batch_projected(
        self, indices: list[int], plan: ProjectionPlan, threads: int | None = None
    ) -> ProjectedBatch: ...

@final
class BatchIter:
    """Streaming batch iterator: `frames_per_batch` frames assembled per
    `__next__`; the final batch may be smaller. Fused after errors.
    """

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
    """Projected variant of `FrameIter`: `__next__` yields `(FrameData | None,
    deviations)` — a dropped frame carries `None` in place of its data.
    """

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
    """Projected variant of `BatchIter`: `__next__` yields `(BatchData,
    survivors, reports)` — the surviving frames' batch, their file indices,
    and a `(frame_index, deviations)` report per requested frame.
    """

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
    """Read every frame. `threads=None` parses on every core; `threads=1` is
    the exact serial streaming read — either way the file is read in a single
    pass and output/errors are identical.
    """

def read_batch(
    path: str,
    indices: list[int] | None = None,
    threads: int | None = None,
    compression: str = "infer",
    member: str | None = None,
) -> BatchData:
    """Gather frames into one batch. `indices=None` reads the whole file in
    file order; a list gathers those frames in request order (repeats
    allowed). Single pass: reads only as far as the last requested frame.
    """

def build_batch(frames: list[FrameData]) -> BatchData:
    """Assemble a batch from already-parsed frame dicts — the inverse of
    reading a batch from a file. An empty list yields the empty batch; a
    non-uniform set raises `ParseError`.
    """

def read_frames_projected(
    path: str,
    threads: int | None = None,
    compression: str = "infer",
    member: str | None = None,
    *,
    plan: ProjectionPlan,
) -> list[ProjectedFrame]:
    """Projected variant of `read_frames`: each element is `(FrameData | None,
    deviations)`.
    """

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
    """Projected variant of `read_batch`: returns `(BatchData, survivors,
    reports)` holding only the frames that survived projection.
    """

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
    """Infer the file's schema: counts, per-column and per-key variant lists
    with unification verdicts, consistency, and the rendered report.
    """

def scan(
    path: str,
    with_volume: bool = False,
    compression: str = "infer",
    member: str | None = None,
) -> ScanData:
    """Structurally scan the file without parsing atom data. `with_volume=True`
    also computes each frame's cell volume `|det(Lattice)|` (`NaN` where a
    frame has no `Lattice`).
    """

def is_compressed(path: str, compression: str = "infer") -> bool:
    """Whether `path` would be read through a decompressing layer under the
    given `compression` — used to refuse random-access batch strategies on a
    non-seekable source.
    """

def detect_codec(name: str, head: bytes | None = None) -> str:
    """Infer the codec name from a filename and optional header bytes."""

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
) -> None:
    """Write frames to `path`. `level` is `0..=9` (codec default when `None`);
    `append` adds to an existing file where the codec allows it; `threads`
    spreads serialisation over workers (`None`: every core, `1`: serial),
    with output bytes identical regardless.
    """

@final
class FrameWriter:
    """Incremental writer: construct it, `write` frames as they come, then
    `close`. Backs `oxyz.Writer`.

    `batch=None` streams each frame straight to the sink one at a time;
    `batch=n` buffers up to `n` frames and serialises each full batch in
    parallel before writing it — bounded extra memory (one batch), output
    bytes unchanged.
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
