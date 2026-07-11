from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

import oxyz._rust as _rust
from oxyz import _remote
from oxyz._frames import (
    ColumnValues,
    Compression,
    _check_threads,
    _projection,
    _require_schema_for_mode,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from oxyz._rust import ProjectedBatch, ProjectionPlan
    from oxyz._schema_match import Conformance
    from oxyz._schema_spec import FrameRule, Mode, SchemaSpec

MemoryScaling = Literal["n_atoms", "n_atoms_x_density"]


@dataclass(frozen=True, slots=True)
class Batch:
    """Frames concatenated atom-major, CSR-style (PyG's batch layout).

    `columns` holds per-atom arrays with `total_atoms` rows; `metadata` holds
    per-frame arrays with `n_frames` rows. Frame `i` occupies rows
    `offsets[i]:offsets[i + 1]` of every column. `frame_indices` records
    which file frame each batch entry came from — provenance for shuffled
    training batches.
    """

    columns: dict[str, ColumnValues]
    metadata: dict[str, ColumnValues]
    offsets: np.ndarray
    frame_indices: np.ndarray

    @property
    def n_frames(self) -> int:
        return len(self.offsets) - 1

    @property
    def total_atoms(self) -> int:
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


def read_batch(  # noqa: PLR0913  the read/schema/projection options are the contract
    path: str | Path,
    indices: Sequence[int] | None = None,
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

    `indices=None` reads every frame in file order; a sequence gathers those
    frames (in order, repeats allowed). Single pass: the file is read once, and
    for a selection only as far as the last requested frame — structure and
    contents beyond it are never inspected. For repeated gathers from one file
    prefer `iread_batch`, which scans once and reuses the index. `threads=None`
    parses on every core, `threads=1` serially; the batch is identical either
    way.

    Works on a compressed source (the selection still streams in one pass);
    `compression` and `member` are as in `read`.

    A remote URL (``s3://``, ``gs://``, ``az://``) streams the object through
    the same reader (needs the ``oxyz[s3]`` extra); ``storage_options`` passes
    endpoint/credentials to the store.

    `schema` (a `SchemaSpec` or a path) with effective `mode="project"` reshapes
    every frame to the schema before concatenation — the way to batch a
    mixed-schema file. `conformance` (`"strict"`/`"required"`/`"warn"`) governs
    deviations; under `warn` an unfillable frame is dropped from the batch, and a
    read where every frame drops yields an empty batch. `mode` without a `schema`
    is an error. Without `schema`, behaviour is unchanged.

    `schema` with effective validate mode (the default) validates each frame the
    batch contains — columns, metadata, and the frame rule — exactly as the frame
    readers do, before concatenation; a mixed-schema file still cannot batch, so
    use `mode="project"` to reshape it.
    """
    _check_threads(threads)
    _require_schema_for_mode(schema, mode)
    projection = None
    spec = None
    if schema is not None:
        projection, spec = _projection(schema, mode)

    selection = None if indices is None else [int(i) for i in indices]
    if selection is not None:
        for index in selection:
            if index < 0:
                # The Rust binding takes unsigned indices; reject negatives here
                # with the documented out-of-range IndexError rather than leaking
                # pyo3's OverflowError. Negative indexing is not supported.
                raise IndexError(
                    f"frame index {index} out of range: indices must be non-negative"
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
        # Validate-mode schema: validate each constituent frame, then assemble.
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


def iread_batch(  # noqa: C901, PLR0913  the keyword options are the batching contract
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

    Exactly one batching strategy must be given:

    - `frames_per_batch` — a fixed structure count per batch.
    - `atoms_per_batch` — greedy file-order packing to a total-atom budget; a
      frame larger than the budget gets a batch to itself.
    - `memory_scales_with` — pack into balanced bins (best-fit-decreasing) under
      `max_scaler`, weighting each frame by `"n_atoms"` or by `"n_atoms_x_density"`
      (`n_atoms**2 / volume`, a proxy for the neighbour-graph size that drives
      MLIP memory). A frame whose weight exceeds `max_scaler` gets its own bin.
      Density needs the cell volume, read by an opt-in extension of the scan;
      a frame with no `Lattice` falls back to its atom count.

    `shuffle` draws frames in a seeded random order via the byte-offset index
    instead of file order (not available with `memory_scales_with`, whose
    packing defines its own order). Whatever the order, each batch records the
    file frame every entry came from in `frame_indices`.

    Batch composition depends only on the file, the knobs, and the seed —
    never on `threads`, which sets parsing parallelism within each batch
    (None: all cores; 1: serial, which for unshuffled `frames_per_batch`
    streams the file without scanning).

    A compressed source (`.gz`, `.zst`, `.zip`, `.tar.gz`, `.tar`) cannot be
    randomly accessed, so only `frames_per_batch` without `shuffle` is
    supported there — it streams in constant memory. `shuffle`, `atoms_per_batch`
    and `memory_scales_with` all need the byte-offset index and raise on a
    compressed source; decompress the file first. `compression` and `member`
    are as in `read`.

    `schema` with effective `mode="project"` reshapes every frame to the schema
    before batching — the way to batch a mixed-schema file. `conformance` governs
    deviations; under `warn` an unfillable frame is dropped, so a batch whose
    frames all drop is skipped and the batch count is not purely a function of
    the strategy. `mode` without a `schema` is an error.
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
    # A remote URL or a compressed local file is non-seekable, so only the
    # streaming strategy (frames_per_batch without shuffle) is available; the
    # index-backed strategies need random access.
    streaming_only = True if remote else _rust.is_compressed(str(path), compression)
    if member is not None and not remote and not streaming_only:
        raise ValueError(
            "member= is only valid for an archive (.zip/.tar/.tar.gz) source"
        )
    if streaming_only and (
        shuffle or atoms_per_batch is not None or memory_scales_with is not None
    ):
        raise ValueError(
            "a compressed or remote source cannot be randomly accessed: only "
            "frames_per_batch without shuffle is supported; download the file first "
            "to use shuffle, atoms_per_batch, or memory_scales_with"
        )

    # The sequential stream covers unshuffled frames_per_batch; a streaming-only
    # source must take it (no index), and otherwise it is the serial fast path.
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
        shuffle,
        seed,
        threads,
        projection,
        conformance,
        spec,
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
    """Streamed file-order batches: constant memory, no scan."""
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
        # Validate-mode schema: parse and validate each frame, then window the
        # frames into batches (build_batch assembles; a mixed file raises there).
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
    shuffle: bool,
    seed: int | None,
    threads: int | None,
    projection: ProjectionPlan | None = None,
    conformance: Conformance = "required",
    spec: SchemaSpec | None = None,
) -> Iterator[Batch]:
    """Index-backed batches over a frame order planned up front.

    Planning is serial and happens before any parsing, so `threads` cannot
    influence which frames land in which batch. The reader's own index
    supplies the atom counts (and, for density, the cell volumes), so the file
    is scanned exactly once."""
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
        # Type-narrowing only, never control flow: iread_batch already raised
        # unless exactly one strategy is set, so this else is the memory case.
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
    """Per-frame packing weight; see `iread_batch` for the rationale.

    `n_atoms_x_density` is `n_atoms**2 / volume`, falling back to `n_atoms`
    where the volume is missing (`NaN`, a frame with no `Lattice`) or
    non-positive — mirroring `torch_sim`'s `where(volume > 0, ...)`.
    """
    counts = n_atoms.astype(np.float64)
    if metric == "n_atoms":
        return counts
    # Type-narrowing only, never control flow: need_volume opened the scan with
    # volumes, so this is non-None here.
    assert volumes is not None  # noqa: S101
    with np.errstate(invalid="ignore", divide="ignore"):
        density = counts * counts / volumes
    return np.where(np.isfinite(volumes) & (volumes > 0), density, counts)


def _balanced_bins(
    order: np.ndarray, weights: np.ndarray, max_volume: float
) -> list[list[int]]:
    """Best-fit-decreasing bin packing, after `torch_sim`'s
    `to_constant_volume_bins`: heaviest first, into the most-full bin that still
    has room, opening a new bin only when none does. A frame heavier than the
    budget opens (and fills) a bin of its own."""
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
    """A batch with no frames — what a fully-dropped projecting read returns."""
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
    """Apply projection policy to a `(batch_data, survivors, reports)` triple and
    return the surviving `Batch`, or `None` when every frame dropped.

    Reports arrive in request order, so the first strict/required violation
    raised is deterministic and matches the frame readers; `warn` warns per
    deviation and the caller skips a `None` (empty) result. The frame rule, if
    the spec has one, is checked against the projected survivors.
    """
    from oxyz._project import enforce_projection

    data, survivors, reports = triple
    survived = set(survivors)
    for req_index, deviations in reports:
        dropped = req_index not in survived
        enforce_projection(deviations, conformance, req_index, dropped)
    if not survivors:
        return None
    batch = _batch_from_data(data, survivors)
    if spec is not None and spec.frame is not None:
        _enforce_frame_rule_on_batch(batch, spec.frame, conformance)
    return batch


def _enforce_frame_rule_on_batch(
    batch: Batch, frame_rule: FrameRule, conformance: Conformance
) -> None:
    """Check a batch's per-frame structural facts (`n_atoms` bounds,
    `lattice_required`) against the frame rule, in frame order — the batch
    analogue of the frame readers' frame-rule validation. Raises under
    strict/required on the first violation; warns under warn."""
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
    """Parse the frames a batch will contain as `FrameData` dicts, with their
    file indices. A selection reads the whole file and picks (the validate path
    is not on the hot loop); `indices=None` returns every frame in file order."""
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
    """Validate each constituent frame (columns, metadata, frame rule) then
    assemble the batch — the validate-mode analogue of the projecting read."""
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
