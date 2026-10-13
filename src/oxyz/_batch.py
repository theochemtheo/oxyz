from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast, overload

import numpy as np

from oxyz import _html, _remote, _rust, _summary
from oxyz._frames import (
    ColumnValues,
    Compression,
    Frame,
    MetadataValue,
    _check_threads,
    _projection,
    _require_schema_for_mode,
)
from oxyz._lookup import NameLookup

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from oxyz._rust import ProjectedBatch, ProjectionPlan
    from oxyz._schema_match import Conformance
    from oxyz._schema_spec import FrameRule, Mode, SchemaSpec

MemoryScaling = Literal["n_atoms", "n_atoms_x_density"]
"""How `iread_batch`'s `memory_scales_with` weighs a frame for balanced-bin
packing: `"n_atoms"` is the atom count; `"n_atoms_x_density"` is
`n_atoms**2 / volume`, a proxy for the neighbour-graph size that drives MLIP
memory."""


@dataclass(frozen=True, slots=True, eq=False)
class Batch(NameLookup[ColumnValues, ColumnValues]):  # noqa: PLW1641  a dict of arrays is unhashable by construction
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

    def __len__(self) -> int:
        """Return the number of frames, `n_frames`."""
        return self.n_frames

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
    # ty does not yet accept an overloaded override of NameLookup's signature.
    def __getitem__(  # ty: ignore[invalid-method-override]
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
        n = self.n_frames
        if not -n <= index < n:
            raise IndexError(f"frame {index} is out of range for a batch of {n} frames")
        return index + n if index < 0 else index

    def _positions(self, key: object) -> np.ndarray:
        n = self.n_frames
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

    def _frame(self, i: int) -> Frame:
        lo, hi = int(self.offsets[i]), int(self.offsets[i + 1])
        return Frame(
            n_atoms=hi - lo,
            columns={
                name: _fresh(value[lo:hi]) for name, value in self.columns.items()
            },
            metadata={name: _unbox(value[i]) for name, value in self.metadata.items()},
        )

    def _slice(self, start: int, stop: int) -> Batch:
        lo, hi = int(self.offsets[start]), int(self.offsets[stop])
        return Batch(
            columns={
                name: _fresh(value[lo:hi]) for name, value in self.columns.items()
            },
            metadata={
                name: _fresh(value[start:stop]) for name, value in self.metadata.items()
            },
            offsets=self.offsets[start : stop + 1] - lo,
            frame_indices=self.frame_indices[start:stop],
        )

    def _gather(self, positions: np.ndarray) -> Batch:
        starts = self.offsets[positions]
        counts = self.offsets[positions + 1] - starts
        offsets = np.zeros(len(positions) + 1, dtype=self.offsets.dtype)
        np.cumsum(counts, out=offsets[1:])
        # Atom row `k` of the gather is row `k - offsets[f] + starts[f]` of
        # the batch, for the frame `f` it falls in.
        rows = np.arange(offsets[-1]) + np.repeat(starts - offsets[:-1], counts)
        return Batch(
            columns={name: _take(value, rows) for name, value in self.columns.items()},
            metadata={
                name: _take(value, positions) for name, value in self.metadata.items()
            },
            offsets=offsets,
            frame_indices=self.frame_indices[positions],
        )

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


def read_batch(  # noqa: PLR0913  the read/schema/projection options are the contract
    path: str | Path,
    index: int | str | slice | Sequence[int] = ":",
    *,
    threads: int | None = None,
    schema: SchemaSpec | str | Path | None = None,
    conformance: Conformance = "required",
    mode: Mode | None = None,
    compression: Compression = "infer",
    member: str | None = None,
    storage_options: _remote.StorageOptions | None = None,
) -> Batch:
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
        Selects one entry from an archive holding more than one; see `read`.
    storage_options
        Endpoint/credentials for a remote store; see `read`.

    Returns
    -------
    Batch
        The gathered frames, concatenated.

    Raises
    ------
    oxyz.ParseError
        On malformed input, carrying the frame index and location.
    IndexError
        On an out-of-range or negative explicit index.

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
    _require_schema_for_mode(schema, mode)
    projection = None
    spec = None
    if schema is not None:
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
        if _remote.is_remote(path):
            src = _remote.open_source(
                path,
                compression=compression,
                member=member,
                storage_options=storage_options,
            )
            triple = _rust.read_batch_projected_reader(
                src.obj, src.codec, selection, threads, src.member, plan=projection
            )
        else:
            triple = _rust.read_batch_projected(
                str(path), selection, threads, compression, member, plan=projection
            )
        result = _resolve_projected_batch(triple, conformance, spec)
        return result if result is not None else _empty_batch()

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


def iread_batch(  # noqa: C901, PLR0913, PLR0912  the keyword options are the batching contract
    path: str | Path,
    *,
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
) -> Iterator[Batch]:
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

    With `mode="project"`, a batch whose frames all drop under
    `conformance="warn"` is skipped.

    Parameters
    ----------
    path
        File path, an S3-compatible URL, or `"-"` for stdin.
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
        Selects one entry from an archive holding more than one; see `read`.
    storage_options
        Endpoint/credentials for a remote store; see `read`.

    Returns
    -------
    Iterator[Batch]
        Batches in the strategy's order, each with its own `frame_indices`.

    Raises
    ------
    ValueError
        On an invalid combination of options, or a strategy the source cannot
        support.

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
    _require_schema_for_mode(schema, mode)
    projection = None
    spec = None
    if schema is not None:
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
        return _sequential_batches(
            path,
            frames_per_batch,
            compression,
            member,
            storage_options,
            projection,
            conformance,
            spec,
        )
    return _planned_batches(
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
    )


def _sequential_batches(
    path: str | Path,
    frames_per_batch: int,
    compression: Compression = "infer",
    member: str | None = None,
    storage_options: _remote.StorageOptions | None = None,
    projection: ProjectionPlan | None = None,
    conformance: Conformance = "required",
    spec: SchemaSpec | None = None,
) -> Iterator[Batch]:
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
        for triple in iterator:
            batch = _resolve_projected_batch(triple, conformance, spec)
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
) -> Iterator[Batch]:
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
            triple = reader.get_batch_projected(plan, projection, threads)
            batch = _resolve_projected_batch(triple, conformance, spec)
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
    triple: ProjectedBatch,
    conformance: Conformance,
    spec: SchemaSpec | None = None,
) -> Batch | None:
    """Apply projection policy to a `(batch_data, survivors, reports)` triple.

    Returns the surviving `Batch`, or `None` when every frame dropped.
    """
    from oxyz._project import enforce_projection

    data, survivors, reports = triple
    survived = set(survivors)
    for req_index, deviations in reports:
        dropped = req_index not in survived
        enforce_projection(deviations, conformance, req_index, dropped=dropped)
    if not survivors:
        return None
    batch = _batch_from_data(data, survivors)
    if spec is not None and spec.frame is not None:
        _enforce_frame_rule_on_batch(batch, spec.frame, conformance)
    return batch


def _enforce_frame_rule_on_batch(
    batch: Batch, frame_rule: FrameRule, conformance: Conformance
) -> None:
    """Check each frame's atom count and `Lattice` against the frame rule."""
    import warnings

    from oxyz._schema_match import SchemaError, SchemaWarning, Violation, message

    lo, hi = frame_rule.n_atoms_min, frame_rule.n_atoms_max
    n_atoms = batch.n_atoms
    indices = batch.frame_indices
    lattice_missing = frame_rule.lattice_required and "Lattice" not in batch.metadata
    for pos in range(batch.n_frames):
        idx = int(indices[pos])
        violations: list[Violation] = []
        count = int(n_atoms[pos])
        if (lo is not None and count < lo) or (hi is not None and count > hi):
            bounds = f"[{'' if lo is None else lo}, {'' if hi is None else hi}]"
            violations.append(
                Violation("frame", "n_atoms", "mismatch", bounds, str(count))
            )
        if lattice_missing:
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
