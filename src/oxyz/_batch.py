from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import numpy as np

from oxyz import _html, _remote, _rust, _summary
from oxyz._frames import (
    ColumnValues,
    Compression,
    Frame,
    MetadataValue,
    _bind_record,
    _check_threads,
    _open_first_frame,
    _projection,
    _require_schema_for_mode,
)
from oxyz._lookup import NameLookup

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence
    from pathlib import Path

    from oxyz._record_batch import RecordBatch
    from oxyz._record_bind import Binding
    from oxyz._rust import ProjectedBatch, ProjectionPlan
    from oxyz._schema_match import Conformance
    from oxyz._schema_spec import FrameRule, Mode, SchemaSpec

MemoryScaling = Literal["n_atoms", "n_atoms_x_density"]
"""How `iread_batch`'s `memory_scales_with` weighs a frame for balanced-bin
packing: `"n_atoms"` is the atom count; `"n_atoms_x_density"` is
`n_atoms**2 / volume`, a proxy for the neighbour-graph size that drives MLIP
memory."""


class FrameLayout:
    """The CSR frame layout `offsets` describes, shared by both batch kinds."""

    __slots__ = ()

    offsets: np.ndarray

    def __len__(self) -> int:
        """Return the number of frames, `n_frames`."""
        return self.n_frames

    @property
    def n_frames(self) -> int:
        """Number of frames in the batch."""
        return len(self.offsets) - 1

    @property
    def total_atoms(self) -> int:
        """Total atom count across all frames."""
        return int(self.offsets[-1])

    @property
    def n_atoms(self) -> np.ndarray:
        """Per-frame atom counts, `np.diff(offsets)`."""
        return np.diff(self.offsets)

    @property
    def ptr(self) -> np.ndarray:
        """Alias of `offsets`, under its PyG name."""
        return self.offsets

    @property
    def batch(self) -> np.ndarray:
        """Per-atom frame id within this batch (PyG's `batch` vector).

        Recomputed on each access; hoist it out of a hot loop.
        """
        return np.repeat(np.arange(self.n_frames), self.n_atoms)


@dataclass(frozen=True, slots=True, eq=False)
class Batch(NameLookup[ColumnValues, ColumnValues], FrameLayout):  # noqa: PLW1641  a dict of arrays is unhashable by construction
    """Frames concatenated atom-major, CSR-style (PyG's batch layout).

    Frame `i` occupies rows `offsets[i]:offsets[i + 1]` of every column.
    `batch[name]` looks a name up across columns and metadata, as on `Frame`;
    `name in batch` tests names, not frames.

    A batch is a sequence of its frames. Indexing follows numpy: `batch[i]` is
    a `Frame` of views, `batch[a:b]` a `Batch` of views, and a step slice,
    sequence of positions, or boolean mask gathers a new `Batch`, a copy.
    String columns are lists, so these always get new lists.

    Attributes
    ----------
    columns
        Per-atom data keyed by column name, `total_atoms` rows.
    metadata
        Per-frame data keyed by name, `n_frames` rows.
    offsets
        CSR row boundaries, length `n_frames + 1`.
    frame_indices
        The file frame each entry came from.
    n_frames
        Number of frames in the batch.
    total_atoms
        Total atom count across all frames.
    n_atoms
        Per-frame atom counts, `np.diff(offsets)`.
    ptr
        Alias of `offsets`, under its PyG name.
    batch
        Per-atom frame id (PyG's `batch` vector), computed on each access.
    """

    columns: dict[str, ColumnValues]
    metadata: dict[str, ColumnValues]
    offsets: np.ndarray
    frame_indices: np.ndarray

    def __repr__(self) -> str:
        """Summarise the batch's shape without printing any of its data."""
        return (
            f"Batch(n_frames={self.n_frames}, total_atoms={self.total_atoms}"
            f"{_summary.atom_range(self.n_atoms)}, "
            f"columns={_summary.mapping_repr(self.columns)}, "
            f"metadata={_summary.mapping_repr(self.metadata)})"
        )

    def _repr_html_(self) -> str:
        """Render as tables for a notebook, previewing a bounded slice of values."""
        return _html.batch(self)

    def __eq__(self, other: object) -> bool:
        """Compare the layout arrays and every stored value.

        Key order is ignored and NaN compares equal, as on `Frame`.
        """
        if not isinstance(other, Batch):
            return NotImplemented
        return (
            _summary.values_equal(self.offsets, other.offsets)
            and _summary.values_equal(self.frame_indices, other.frame_indices)
            and _summary.mappings_equal(self.columns, other.columns)
            and _summary.mappings_equal(self.metadata, other.metadata)
        )

    def _sides(self) -> tuple[dict[str, ColumnValues], dict[str, ColumnValues]]:
        return self.columns, self.metadata

    def __iter__(self) -> Iterator[Frame]:
        """Yield each frame in turn, as `batch[i]` would."""
        for i in range(self.n_frames):
            yield self._frame(i)

    @overload
    def __getitem__(self, key: str) -> ColumnValues: ...
    @overload
    def __getitem__(self, key: int | np.integer) -> Frame: ...
    @overload
    def __getitem__(self, key: slice | Sequence[int] | np.ndarray) -> Batch: ...
    def __getitem__(
        self, key: str | int | np.integer | slice | Sequence[int] | np.ndarray
    ) -> ColumnValues | Frame | Batch:
        """Look up a field by name, a frame by position, or frames by selection.

        Negative positions count from the end; out of range is `IndexError`.
        """
        if isinstance(key, str):
            return NameLookup.__getitem__(self, key)
        if isinstance(key, slice):
            start, stop, step = key.indices(self.n_frames)
            if step == 1:
                return self._slice(start, max(start, stop))
            return self._gather(np.arange(start, stop, step, dtype=np.intp))
        if isinstance(key, (int, np.integer)) and not isinstance(key, (bool, np.bool_)):
            return self._frame(self._position(int(key)))
        if isinstance(key, (list, tuple, np.ndarray)):
            return self._gather(self._positions(key))
        raise TypeError(
            "Batch keys are names (str), frame positions (int), slices, "
            f"position sequences or boolean masks, not {type(key).__name__}"
        )

    def _position(self, index: int) -> int:
        return position(index, self.n_frames)

    def _positions(self, key: object) -> np.ndarray:
        return positions(key, self.n_frames)

    def _frame(self, i: int) -> Frame:
        n_atoms, columns, metadata = frame_rows(
            self.columns, self.metadata, self.offsets, i
        )
        return Frame(n_atoms=n_atoms, columns=columns, metadata=metadata)

    def _slice(self, start: int, stop: int) -> Batch:
        return Batch(
            *slice_rows(
                self.columns,
                self.metadata,
                self.offsets,
                self.frame_indices,
                start,
                stop,
            )
        )

    def _gather(self, positions: np.ndarray) -> Batch:
        return Batch(
            *gather_rows(
                self.columns, self.metadata, self.offsets, self.frame_indices, positions
            )
        )


type Unbox = Callable[[str, object], object]


def position(index: int, n_frames: int) -> int:
    """Resolve a possibly negative frame index, or raise `IndexError`."""
    n = n_frames
    if not -n <= index < n:
        raise IndexError(f"frame {index} is out of range for a batch of {n} frames")
    return index + n if index < 0 else index


def positions(key: object, n_frames: int) -> np.ndarray:
    """Resolve a boolean mask or integer sequence to non-negative frame positions."""
    n = n_frames
    selection = np.asarray(key)
    if selection.size == 0:
        return np.empty(0, dtype=np.intp)
    if selection.dtype == np.bool_:
        if selection.shape != (n,):
            raise IndexError(
                f"boolean mask of shape {selection.shape} does not match a "
                f"batch of {n} frames"
            )
        return np.flatnonzero(selection)
    if selection.ndim != 1 or selection.dtype.kind not in "iu":
        raise TypeError(
            "frame positions must be a 1-D sequence of integers, not "
            f"{selection.dtype}{list(selection.shape)}"
        )
    out_of_range = (selection < -n) | (selection >= n)
    if out_of_range.any():
        raise IndexError(
            f"frames {selection[out_of_range].tolist()} are out of range "
            f"for a batch of {n} frames"
        )
    return np.where(selection < 0, selection + n, selection).astype(np.intp)


def frame_rows(
    columns: Mapping[str, Any],
    metadata: Mapping[str, Any],
    offsets: np.ndarray,
    i: int,
    unbox: Unbox = lambda _name, value: _unbox(value),
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    """Frame `i`'s atom count, column views and unboxed metadata values."""
    lo, hi = int(offsets[i]), int(offsets[i + 1])
    return (
        hi - lo,
        {name: _fresh(value[lo:hi]) for name, value in columns.items()},
        {name: unbox(name, value[i]) for name, value in metadata.items()},
    )


def slice_rows(
    columns: Mapping[str, Any],
    metadata: Mapping[str, Any],
    offsets: np.ndarray,
    frame_indices: np.ndarray,
    start: int,
    stop: int,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]:
    """Frames `start:stop` as views, with offsets rebased to start at zero."""
    lo, hi = int(offsets[start]), int(offsets[stop])
    return (
        {name: _fresh(value[lo:hi]) for name, value in columns.items()},
        {name: _fresh(value[start:stop]) for name, value in metadata.items()},
        offsets[start : stop + 1] - lo,
        frame_indices[start:stop],
    )


def gather_rows(
    columns: Mapping[str, Any],
    metadata: Mapping[str, Any],
    offsets: np.ndarray,
    frame_indices: np.ndarray,
    picks: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]:
    """Frames at `picks` copied in order, with fresh offsets."""
    starts = offsets[picks]
    counts = offsets[picks + 1] - starts
    gathered = np.zeros(len(picks) + 1, dtype=offsets.dtype)
    np.cumsum(counts, out=gathered[1:])
    # Atom row `k` of the gather is row `k - gathered[f] + starts[f]` of the
    # source, for the frame `f` it falls in.
    rows = np.arange(gathered[-1]) + np.repeat(starts - gathered[:-1], counts)
    return (
        {name: _take(value, rows) for name, value in columns.items()},
        {name: _take(value, picks) for name, value in metadata.items()},
        gathered,
        frame_indices[picks],
    )


def _fresh[V](value: V) -> V:
    """Return `value` with every list in it replaced, so nothing aliases the batch."""
    if isinstance(value, list):
        return [_fresh(item) for item in value]  # ty: ignore[invalid-return-type]
    return value


def _unbox(value: object) -> MetadataValue:
    """Turn one frame's metadata row into what `oxyz.read` would give."""
    if isinstance(value, np.generic):
        # A batch's numeric metadata is float64, int64 or bool.
        return cast("float | int | bool", value.item())
    return cast("MetadataValue", _fresh(value))


def _take(value: ColumnValues, rows: np.ndarray) -> ColumnValues:
    """Gather `rows` of a stored value: fancy indexing, or list picking."""
    if isinstance(value, list):
        return [_fresh(value[row]) for row in rows.tolist()]
    return value[rows]


def _scan_count(
    path: str | Path,
    compression: Compression,
    member: str | None,
    storage_options: _remote.StorageOptions | None,
) -> int:
    """Frame count from a structural scan."""
    from oxyz._scan import scan

    return scan(
        path, compression=compression, member=member, storage_options=storage_options
    ).n_frames


def _resolve_batch_indices(
    index: int | str | slice | Sequence[int],
    path: str | Path,
    compression: Compression,
    member: str | None,
    storage_options: _remote.StorageOptions | None,
) -> list[int] | None:
    """Turn `read_batch`'s `index` into a frame list, or `None` for every frame.

    Only a slice or a negative int needs a scan.
    """
    from oxyz._select import parse_index

    if isinstance(index, str):
        index = parse_index(index)
    if isinstance(index, slice):
        if index.start is None and index.stop is None and index.step is None:
            return None
        n_frames = _scan_count(path, compression, member, storage_options)
        return list(range(*index.indices(n_frames)))
    if isinstance(index, int):
        if index >= 0:
            return [index]
        n_frames = _scan_count(path, compression, member, storage_options)
        resolved = index + n_frames
        if resolved < 0:
            raise IndexError(
                f"frame index {index} out of range: file has {n_frames} frames"
            )
        return [resolved]
    return [int(i) for i in index]


@overload
def read_batch[B: RecordBatch[Any]](
    path: str | Path,
    index: int | str | slice | Sequence[int] = ...,
    *,
    record: type[B],
    threads: int | None = ...,
    schema: SchemaSpec | str | Path | None = ...,
    conformance: Conformance = ...,
    compression: Compression = ...,
    member: str | None = ...,
    storage_options: _remote.StorageOptions | None = ...,
) -> B: ...


@overload
def read_batch(
    path: str | Path,
    index: int | str | slice | Sequence[int] = ...,
    *,
    record: None = ...,
    threads: int | None = ...,
    schema: SchemaSpec | str | Path | None = ...,
    conformance: Conformance = ...,
    mode: Mode | None = ...,
    compression: Compression = ...,
    member: str | None = ...,
    storage_options: _remote.StorageOptions | None = ...,
) -> Batch: ...


def read_batch[B: RecordBatch[Any]](  # noqa: PLR0913  the read/schema/projection options are the contract
    path: str | Path,
    index: int | str | slice | Sequence[int] = ":",
    *,
    record: type[B] | None = None,
    threads: int | None = None,
    schema: SchemaSpec | str | Path | None = None,
    conformance: Conformance = "required",
    mode: Mode | None = None,
    compression: Compression = "infer",
    member: str | None = None,
    storage_options: _remote.StorageOptions | None = None,
) -> Batch | B:
    """Gather frames into one batch.

    The file is read once, and for an explicit selection only as far as the
    last requested frame. To gather repeatedly from one file, `iread_batch`
    scans once and reuses the index.

    With a `schema`, each frame is validated before concatenation; with
    `mode="project"` each is reshaped to the schema instead, which is how to
    batch a mixed-schema file. Under `conformance="warn"` a projected frame
    that cannot be filled is dropped.

    Parameters
    ----------
    path
        File path, or a remote URL (``s3://``, ``gs://``, ``az://``,
        ``hf://`` — needs the ``oxyz[s3]`` extra).
    index
        `read`'s selection grammar, or a sequence of non-negative frame
        indices gathered in order (repeats allowed). Default: every frame.
    record
        A batch class to read into instead of a `Batch`: `RecordBatch[R]`
        for a `Record` subclass `R`, or a declared subclass of it. Frames
        are projected onto the record's fields as `read` with `record=R`
        projects them, and only the selected frames are read. Under
        `conformance="warn"` the batch holds the frames whose records
        `read` returns for the same `index`. An `X | None` field is filled
        where a frame lacks it (NaN for a float field, else its `fill=`,
        which a batch read of an integer, bool or string `X | None` field
        requires) and `present` says which frames carried it. The batch's
        arrays are writable and belong to it alone. An empty selection, or
        one whose frames all drop, gives a batch of no frames with the
        record's fields, where a read without `record` raises. Cannot be
        combined with `mode`; `schema` acts as it does for `read`.
    threads
        `None` parses on every core; `1` is serial. The batch is identical.
    schema
        A `SchemaSpec`, or a path to a schema file.
    conformance
        How a schema deviation is handled: `"strict"`, `"required"` (default),
        or `"warn"`.
    mode
        Overrides the schema's own `mode`; requires `schema`.
    compression
        Forces a codec instead of inferring it from `path`; see `read`.
    member
        Selects one entry from an archive holding more than one, or all of them
        with `"*"`; see `read`.
    storage_options
        Endpoint/credentials for a remote store; see `read`.

    Returns
    -------
    Batch, or B with `record=B`
        The gathered frames, concatenated; with `record`, an instance of
        its batch class.

    Raises
    ------
    oxyz.ParseError
        On malformed input, carrying the frame index and location.
    IndexError
        On an out-of-range or negative explicit index.
    oxyz.SchemaError
        With `record`, as for `read`. Before reading: `schema` disagrees
        with the record, or an integer, bool or string `X | None` field has
        no `fill=`.
    TypeError
        `record` is not a batch class; a `Record` subclass `R` is named
        here as `RecordBatch[R]`.
    ValueError
        `mode` is given with `record`.

    Examples
    --------
    >>> import oxyz
    >>> batch = oxyz.read_batch(DATA / "water.extxyz")
    >>> batch.columns["pos"].shape
    (9, 3)
    >>> batch.metadata["energy"].shape
    (3,)
    """
    _check_threads(threads)
    binding = None
    if record is not None:
        binding = _bind_record(record, schema, mode, batch=True)
    _require_schema_for_mode(schema, mode)
    projection = None
    spec = None
    if binding is not None:
        projection, spec = binding.batch_plan, binding.spec
    elif schema is not None:
        projection, spec = _projection(schema, mode)

    selection = _resolve_batch_indices(
        index, path, compression, member, storage_options
    )
    if selection is not None:
        for i in selection:
            if i < 0:
                # IndexError, not the binding's OverflowError on an unsigned index.
                raise IndexError(
                    f"frame index {i} out of range: indices must be non-negative"
                )

    if projection is not None:
        projected = _read_batch_projected(
            path,
            selection,
            threads=threads,
            compression=compression,
            member=member,
            storage_options=storage_options,
            projection=projection,
            conformance=conformance,
            spec=spec,
            binding=binding,
        )
        return cast("Batch | B", projected)

    if spec is not None:
        return _read_batch_validate(
            path,
            selection,
            spec,
            conformance,
            threads,
            compression,
            member,
            storage_options,
        )

    if _remote.is_remote(path):
        src = _remote.open_source(
            path,
            compression=compression,
            member=member,
            storage_options=storage_options,
        )
        data = _rust.read_batch_reader(
            src.obj, src.codec, selection, threads, src.member
        )
        frame_indices: Sequence[int] = (
            selection if selection is not None else range(len(data["offsets"]) - 1)
        )
        return _batch_from_data(data, frame_indices)
    if selection is None:
        data = _rust.read_batch(str(path), None, threads, compression, member)
        return _batch_from_data(data, range(len(data["offsets"]) - 1))
    return _batch_from_data(
        _rust.read_batch(str(path), selection, threads, compression, member), selection
    )


def _read_batch_projected(  # noqa: PLR0913  the read options plus the projection
    path: str | Path,
    selection: list[int] | None,
    *,
    threads: int | None,
    compression: Compression,
    member: str | None,
    storage_options: _remote.StorageOptions | None,
    projection: ProjectionPlan,
    conformance: Conformance,
    spec: SchemaSpec | None,
    binding: Binding | None,
) -> Batch | RecordBatch[Any]:
    """Project the selected frames into one batch, empty if none survive."""
    if binding is not None and selection is not None and not selection:
        _open_first_frame(path, compression, member, storage_options)
        return binding.empty_batch()
    if _remote.is_remote(path):
        src = _remote.open_source(
            path,
            compression=compression,
            member=member,
            storage_options=storage_options,
        )
        projected = _rust.read_batch_projected_reader(
            src.obj, src.codec, selection, threads, src.member, plan=projection
        )
    else:
        projected = _rust.read_batch_projected(
            str(path), selection, threads, compression, member, plan=projection
        )
    result = _resolve_projected_batch(projected, conformance, spec, binding)
    if result is not None:
        return result
    return _empty_batch() if binding is None else binding.empty_batch()


@overload
def iread_batch[B: RecordBatch[Any]](
    path: str | Path,
    *,
    record: type[B],
    frames_per_batch: int | None = ...,
    atoms_per_batch: int | None = ...,
    memory_scales_with: MemoryScaling | None = ...,
    max_scaler: float | None = ...,
    shuffle: bool = ...,
    seed: int | None = ...,
    threads: int | None = ...,
    schema: SchemaSpec | str | Path | None = ...,
    conformance: Conformance = ...,
    compression: Compression = ...,
    member: str | None = ...,
    storage_options: _remote.StorageOptions | None = ...,
) -> Iterator[B]: ...


@overload
def iread_batch(
    path: str | Path,
    *,
    record: None = ...,
    frames_per_batch: int | None = ...,
    atoms_per_batch: int | None = ...,
    memory_scales_with: MemoryScaling | None = ...,
    max_scaler: float | None = ...,
    shuffle: bool = ...,
    seed: int | None = ...,
    threads: int | None = ...,
    schema: SchemaSpec | str | Path | None = ...,
    conformance: Conformance = ...,
    mode: Mode | None = ...,
    compression: Compression = ...,
    member: str | None = ...,
    storage_options: _remote.StorageOptions | None = ...,
) -> Iterator[Batch]: ...


def iread_batch[B: RecordBatch[Any]](  # noqa: C901, PLR0913, PLR0912  the keyword options are the batching contract
    path: str | Path,
    *,
    record: type[B] | None = None,
    frames_per_batch: int | None = None,
    atoms_per_batch: int | None = None,
    memory_scales_with: MemoryScaling | None = None,
    max_scaler: float | None = None,
    shuffle: bool = False,
    seed: int | None = None,
    threads: int | None = None,
    schema: SchemaSpec | str | Path | None = None,
    conformance: Conformance = "required",
    mode: Mode | None = None,
    compression: Compression = "infer",
    member: str | None = None,
    storage_options: _remote.StorageOptions | None = None,
) -> Iterator[Batch] | Iterator[B]:
    """Read a file as a sequence of batches.

    Exactly one strategy must be given:

    - `frames_per_batch`: a fixed frame count per batch.
    - `atoms_per_batch`: file-order packing to an atom budget.
    - `memory_scales_with`: balanced bins under `max_scaler`, weighting each
      frame by `"n_atoms"` or `"n_atoms_x_density"` (`n_atoms**2 / volume`, a
      proxy for MLIP neighbour-graph memory; `n_atoms` where there is no
      `Lattice`).

    A frame over budget gets a batch to itself. Batch composition depends only
    on the file, the strategy and the seed, never on `threads`.

    A compressed or remote source supports only `frames_per_batch` without
    `shuffle`; the other strategies need random access.

    With `mode="project"`, or with `record`, a batch whose frames all drop
    under `conformance="warn"` is skipped.

    Parameters
    ----------
    path
        File path, an S3-compatible URL, or `"-"` for stdin.
    record
        A batch class to yield instances of instead of `Batch`es:
        `RecordBatch[R]` or a declared subclass of it; see `read_batch`.
        Each batch holds the frames of its share of the file that `read`
        with `record=R` keeps.
    frames_per_batch
        Fixed frame count per batch.
    atoms_per_batch
        Atom budget per batch.
    memory_scales_with
        `"n_atoms"` or `"n_atoms_x_density"`.
    max_scaler
        The per-bin weight budget for `memory_scales_with`.
    shuffle
        Draw frames in a seeded random order; not with `memory_scales_with`.
    seed
        Seeds `shuffle`.
    threads
        Parsing parallelism within each batch; `None` for all cores, `1` for
        serial.
    schema
        A `SchemaSpec`, or a path to a schema file; see `read_batch`.
    conformance
        How a schema deviation is handled; see `read_batch`.
    mode
        Overrides the schema's own `mode`; requires `schema`.
    compression
        Forces a codec instead of inferring it from `path`; see `read`.
    member
        Selects one entry from an archive holding more than one, or all of them
        with `"*"`; see `read`.
    storage_options
        Endpoint/credentials for a remote store; see `read`.

    Returns
    -------
    Iterator[Batch], or Iterator[B] with `record=B`
        Batches in the strategy's order, each with its own `frame_indices`.

    Raises
    ------
    ValueError
        On an invalid combination of options, or a strategy the source cannot
        support; `mode` given with `record`.
    oxyz.SchemaError
        With `record`, as for `read_batch`. A disagreeing `schema`, or an
        `X | None` field that needs a `fill=`, raises from this call, before
        any frame is read.
    TypeError
        `record` is not a batch class; see `read_batch`.

    Examples
    --------
    >>> import oxyz
    >>> for b in oxyz.iread_batch(DATA / "water.extxyz", frames_per_batch=2):
    ...     print(b.columns["pos"].shape[0], b.frame_indices.tolist())
    6 [0, 1]
    3 [2]
    """
    strategies = sum(
        knob is not None
        for knob in (frames_per_batch, atoms_per_batch, memory_scales_with)
    )
    if strategies != 1:
        raise ValueError(
            "pass exactly one of frames_per_batch, atoms_per_batch, "
            "or memory_scales_with"
        )
    if frames_per_batch is not None and frames_per_batch < 1:
        raise ValueError("frames_per_batch must be at least 1")
    if atoms_per_batch is not None and atoms_per_batch < 1:
        raise ValueError("atoms_per_batch must be at least 1")
    if memory_scales_with is not None:
        if memory_scales_with not in ("n_atoms", "n_atoms_x_density"):
            raise ValueError(
                "memory_scales_with must be 'n_atoms' or 'n_atoms_x_density', "
                f"got {memory_scales_with!r}"
            )
        if max_scaler is None or max_scaler <= 0:
            raise ValueError("memory_scales_with requires max_scaler > 0")
        if shuffle:
            raise ValueError("shuffle is not supported with memory_scales_with")
    elif max_scaler is not None:
        raise ValueError("max_scaler requires memory_scales_with")
    if seed is not None and not shuffle:
        raise ValueError("seed requires shuffle=True")
    _check_threads(threads)
    binding = None
    if record is not None:
        binding = _bind_record(record, schema, mode, batch=True)
    _require_schema_for_mode(schema, mode)
    projection = None
    spec = None
    if binding is not None:
        projection, spec = binding.batch_plan, binding.spec
    elif schema is not None:
        projection, spec = _projection(schema, mode)

    remote = _remote.is_remote(path)
    # Non-seekable sources support only streaming frames_per_batch.
    streaming_only = True if remote else _rust.is_compressed(str(path), compression)
    if member is not None and not remote and not streaming_only:
        raise ValueError(
            "member= is only valid for an archive (.zip/.tar/.tar.gz/.tar.zst) source"
        )
    if streaming_only and (
        shuffle or atoms_per_batch is not None or memory_scales_with is not None
    ):
        raise ValueError(
            "a compressed or remote source cannot be randomly accessed: only "
            "frames_per_batch without shuffle is supported; download the file first "
            "to use shuffle, atoms_per_batch, or memory_scales_with"
        )

    if (
        frames_per_batch is not None
        and not shuffle
        and (streaming_only or threads == 1)
    ):
        sequential = _sequential_batches(
            path,
            frames_per_batch,
            compression,
            member,
            storage_options,
            projection=projection,
            conformance=conformance,
            spec=spec,
            binding=binding,
        )
        return cast("Iterator[Batch] | Iterator[B]", sequential)
    planned = _planned_batches(
        path,
        frames_per_batch,
        atoms_per_batch,
        memory_scales_with,
        max_scaler,
        shuffle=shuffle,
        seed=seed,
        threads=threads,
        projection=projection,
        conformance=conformance,
        spec=spec,
        binding=binding,
    )
    return cast("Iterator[Batch] | Iterator[B]", planned)


def _sequential_batches(  # noqa: PLR0913  the source options plus projection
    path: str | Path,
    frames_per_batch: int,
    compression: Compression = "infer",
    member: str | None = None,
    storage_options: _remote.StorageOptions | None = None,
    *,
    projection: ProjectionPlan | None = None,
    conformance: Conformance = "required",
    spec: SchemaSpec | None = None,
    binding: Binding | None = None,
) -> Iterator[Batch | RecordBatch[Any]]:
    """File-order batches streamed without a scan."""
    remote = _remote.is_remote(path)
    src = (
        _remote.open_source(
            path,
            compression=compression,
            member=member,
            storage_options=storage_options,
        )
        if remote
        else None
    )

    if projection is not None:
        if src is not None:
            iterator = _rust.BatchIterProjected.from_reader(
                src.obj, frames_per_batch, projection, src.codec, src.member
            )
        else:
            iterator = _rust.BatchIterProjected(
                str(path), frames_per_batch, projection, compression, member
            )
        for projected in iterator:
            batch = _resolve_projected_batch(projected, conformance, spec, binding)
            if batch is not None:  # a fully-dropped window is skipped
                yield batch
        return

    if spec is not None:
        from oxyz._frames import _frame_from_data
        from oxyz._schema_match import enforce_frame, resolve

        compiled = resolve(spec)
        frames_iter = (
            _rust.FrameIter.from_reader(src.obj, src.codec, src.member)
            if src is not None
            else _rust.FrameIter(str(path), compression, member)
        )
        window: list[_rust.FrameData] = []
        start = 0
        for i, fdata in enumerate(frames_iter):
            enforce_frame(_frame_from_data(fdata), compiled, conformance, i)
            window.append(fdata)
            if len(window) == frames_per_batch:
                yield _batch_from_data(_rust.build_batch(window), range(start, i + 1))
                start, window = i + 1, []
        if window:
            yield _batch_from_data(
                _rust.build_batch(window), range(start, start + len(window))
            )
        return

    if src is not None:
        plain = _rust.BatchIter.from_reader(
            src.obj, frames_per_batch, src.codec, src.member
        )
    else:
        plain = _rust.BatchIter(str(path), frames_per_batch, compression, member)
    start = 0
    for data in plain:
        n_frames = len(data["offsets"]) - 1
        yield _batch_from_data(data, range(start, start + n_frames))
        start += n_frames


def _planned_batches(  # noqa: PLR0913  the planning knobs plus projection
    path: str | Path,
    frames_per_batch: int | None,
    atoms_per_batch: int | None,
    memory_scales_with: MemoryScaling | None,
    max_scaler: float | None,
    *,
    shuffle: bool,
    seed: int | None,
    threads: int | None,
    projection: ProjectionPlan | None = None,
    conformance: Conformance = "required",
    spec: SchemaSpec | None = None,
    binding: Binding | None = None,
) -> Iterator[Batch | RecordBatch[Any]]:
    """Index-backed batches over a frame order planned before any parsing."""
    validate_compiled = None
    if spec is not None and projection is None:
        from oxyz._schema_match import resolve

        validate_compiled = resolve(spec)
    need_volume = memory_scales_with == "n_atoms_x_density"
    reader = _rust.IndexedFrames(str(path), need_volume)
    n_atoms = reader.n_atoms
    order = np.arange(len(n_atoms), dtype=np.intp)
    if shuffle:
        order = np.random.default_rng(seed).permutation(order)

    if frames_per_batch is not None:
        plans = [
            [int(i) for i in order[start : start + frames_per_batch]]
            for start in range(0, len(order), frames_per_batch)
        ]
    elif atoms_per_batch is not None:
        plans = _greedy_atom_plans(order, n_atoms, atoms_per_batch)
    else:
        # Type-narrowing only: iread_batch ensured exactly one strategy.
        assert memory_scales_with is not None  # noqa: S101
        assert max_scaler is not None  # noqa: S101
        weights = _memory_weights(memory_scales_with, n_atoms, reader.volumes)
        plans = _balanced_bins(order, weights, max_scaler)

    for plan in plans:
        if projection is not None:
            projected = reader.get_batch_projected(plan, projection, threads)
            batch = _resolve_projected_batch(projected, conformance, spec, binding)
            if batch is not None:  # a fully-dropped batch is skipped
                yield batch
        elif validate_compiled is not None:
            from oxyz._frames import _frame_from_data
            from oxyz._schema_match import enforce_frame

            frames_data = [reader.get(i) for i in plan]
            for pos, i in enumerate(plan):
                enforce_frame(
                    _frame_from_data(frames_data[pos]),
                    validate_compiled,
                    conformance,
                    i,
                )
            yield _batch_from_data(_rust.build_batch(frames_data), plan)
        else:
            yield _batch_from_data(reader.get_batch(plan, threads), plan)


def _greedy_atom_plans(
    order: np.ndarray, n_atoms: np.ndarray, atoms_per_batch: int
) -> list[list[int]]:
    """Fill batches in `order` until the next frame would exceed the budget."""
    plans: list[list[int]] = []
    current: list[int] = []
    total = 0
    for i in order:
        count = int(n_atoms[i])
        if current and total + count > atoms_per_batch:
            plans.append(current)
            current, total = [], 0
        current.append(int(i))
        total += count
    if current:
        plans.append(current)
    return plans


def _memory_weights(
    metric: MemoryScaling, n_atoms: np.ndarray, volumes: np.ndarray | None
) -> np.ndarray:
    """Per-frame packing weight; `n_atoms` where the volume is missing or <= 0."""
    counts = n_atoms.astype(np.float64)
    if metric == "n_atoms":
        return counts
    # Type-narrowing only: need_volume scanned with volumes.
    assert volumes is not None  # noqa: S101
    with np.errstate(invalid="ignore", divide="ignore"):
        density = counts * counts / volumes
    return np.where(np.isfinite(volumes) & (volumes > 0), density, counts)


def _balanced_bins(
    order: np.ndarray, weights: np.ndarray, max_volume: float
) -> list[list[int]]:
    """Best-fit-decreasing packing, after `torch_sim`'s `to_constant_volume_bins`."""
    entries = sorted(
        ((float(weights[i]), int(i)) for i in order), key=lambda entry: -entry[0]
    )
    bins: list[list[int]] = []
    sums: list[float] = []
    for weight, index in entries:
        candidates = [b for b, total in enumerate(sums) if total + weight <= max_volume]
        if candidates:
            chosen = max(candidates, key=lambda b: sums[b])
        else:
            chosen = len(bins)
            bins.append([])
            sums.append(0.0)
        bins[chosen].append(index)
        sums[chosen] += weight
    return bins


def _batch_from_data(data: _rust.BatchData, frame_indices: Sequence[int]) -> Batch:
    return Batch(
        columns=data["columns"],
        metadata=data["metadata"],
        offsets=data["offsets"],
        frame_indices=np.asarray(frame_indices, dtype=np.intp),
    )


def _empty_batch() -> Batch:
    """Build a batch with no frames, for a fully-dropped projecting read."""
    return Batch(
        columns={},
        metadata={},
        offsets=np.array([0], dtype=np.intp),
        frame_indices=np.array([], dtype=np.intp),
    )


def _resolve_projected_batch(
    projected: ProjectedBatch,
    conformance: Conformance,
    spec: SchemaSpec | None = None,
    binding: Binding | None = None,
) -> Batch | RecordBatch[Any] | None:
    """Apply projection policy to a `read_batch_projected`-shaped result.

    Returns the surviving `Batch`, cast by `binding` when given, or `None`
    when every frame dropped.
    """
    from oxyz._project import enforce_projection

    data, survivors, reports, presence = projected
    survived = set(survivors)
    for req_index, deviations in reports:
        dropped = req_index not in survived
        enforce_projection(deviations, conformance, req_index, dropped=dropped)
    if not survivors:
        return None
    batch = _batch_from_data(data, survivors)
    if binding is not None and conformance == "warn":
        # Under "required" and "strict" every deviation has already raised.
        kept = binding.kept(batch, presence, reports, survived)
        if kept is None:
            return None
        batch, presence = kept
    if spec is not None and spec.frame is not None:
        lattice = None if binding is None else binding.carried("Lattice", presence)
        _enforce_frame_rule_on_batch(batch, spec.frame, conformance, lattice)
    if binding is None:
        return batch
    return binding.cast_batch(batch, presence, owned=True)


def _enforce_frame_rule_on_batch(
    batch: Batch,
    frame_rule: FrameRule,
    conformance: Conformance,
    lattice: np.ndarray | None = None,
) -> None:
    """Check each frame's atom count and `Lattice` against the frame rule.

    `lattice` marks the frames that carry `Lattice`, where the batch's own
    metadata cannot tell; by default every frame carries it if the batch has
    the key.
    """
    import warnings

    from oxyz._schema_match import SchemaError, SchemaWarning, Violation, message

    lo, hi = frame_rule.n_atoms_min, frame_rule.n_atoms_max
    n_atoms = batch.n_atoms
    indices = batch.frame_indices
    if lattice is None:
        lattice = np.full(batch.n_frames, "Lattice" in batch.metadata)
    for pos in range(batch.n_frames):
        idx = int(indices[pos])
        violations: list[Violation] = []
        count = int(n_atoms[pos])
        if (lo is not None and count < lo) or (hi is not None and count > hi):
            bounds = f"[{'' if lo is None else lo}, {'' if hi is None else hi}]"
            violations.append(
                Violation("frame", "n_atoms", "mismatch", bounds, str(count))
            )
        if frame_rule.lattice_required and not lattice[pos]:
            violations.append(
                Violation("frame", "Lattice", "missing", "required", None)
            )
        for violation in violations:
            if conformance in ("strict", "required"):
                raise SchemaError(
                    message(violation, idx), frame_index=idx, name=violation.name
                )
            warnings.warn(message(violation, idx), SchemaWarning, stacklevel=2)


def _materialise_batch_frames(
    path: str | Path,
    selection: list[int] | None,
    threads: int | None,
    compression: Compression,
    member: str | None,
    storage_options: _remote.StorageOptions | None,
) -> tuple[list[_rust.FrameData], list[int]]:
    """Parse the frames a batch will contain, with their file indices."""
    if _remote.is_remote(path):
        src = _remote.open_source(
            path,
            compression=compression,
            member=member,
            storage_options=storage_options,
        )
        all_data = _rust.read_frames_reader(src.obj, src.codec, src.member, threads)
    else:
        all_data = _rust.read_frames(str(path), threads, compression, member)
    if selection is None:
        return all_data, list(range(len(all_data)))
    picked = []
    for i in selection:
        if i >= len(all_data):
            raise IndexError(
                f"frame index {i} out of range: file has {len(all_data)} frames"
            )
        picked.append(all_data[i])
    return picked, list(selection)


def _read_batch_validate(
    path: str | Path,
    selection: list[int] | None,
    spec: SchemaSpec,
    conformance: Conformance,
    threads: int | None,
    compression: Compression,
    member: str | None,
    storage_options: _remote.StorageOptions | None,
) -> Batch:
    """Validate each frame against the schema, then assemble the batch."""
    from oxyz._frames import _frame_from_data
    from oxyz._schema_match import enforce_frame, resolve

    frames_data, frame_indices = _materialise_batch_frames(
        path, selection, threads, compression, member, storage_options
    )
    compiled = resolve(spec)
    for pos, file_index in enumerate(frame_indices):
        enforce_frame(
            _frame_from_data(frames_data[pos]), compiled, conformance, file_index
        )
    return _batch_from_data(_rust.build_batch(frames_data), frame_indices)
