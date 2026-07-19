from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, overload

import numpy as np

import oxyz._rust as _rust
from oxyz import _remote

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from pathlib import Path

    from ase import Atoms

    from oxyz._rust import ProjectedFrame, ProjectionPlan
    from oxyz._schema_match import CompiledSpec, Conformance
    from oxyz._schema_spec import Mode, SchemaSpec

ColumnValues = np.ndarray | list[str] | list[list[str]]
"""A per-atom column's values: an array for numeric/bool data, `list[str]` for
a 1-D string column, `list[list[str]]` for a 2-D one."""

# np.ndarray covers 1-D and 2-D numeric/bool metadata (shape (rows, cols) for
# 2-D); 2-D string metadata crosses as list[list[str]], mirroring ColumnValues.
MetadataValue = float | int | bool | str | np.ndarray | list[str] | list[list[str]]
"""A comment-line metadata value: a scalar, an array, or (for 2-D string
metadata) `list[list[str]]`."""

Compression = Literal["infer", "none", "gzip", "zstd", "zip"]
"""How to read a possibly-compressed file. `"infer"` detects the codec from the
extension, falling back to the leading magic bytes; the rest force a codec."""


@dataclass(frozen=True, slots=True)
class Frame:
    """One parsed extxyz frame: per-atom columns plus comment-line metadata.

    Both dicts preserve file order. Column names and metadata values are kept
    exactly as written in the file; aliasing (`force` vs `forces`) and
    conversions (Fortran-order `Lattice` to a 3x3 cell) belong to a later
    normalisation layer. `metadata` is a dict, so a repeated key keeps only its
    last value.

    Attributes
    ----------
    n_atoms
        Number of atoms in the frame (the XYZ file's leading count line).
    columns
        Per-atom data keyed by column name, e.g. `"species"`, `"pos"`,
        `"forces"`, each shaped `(n_atoms, ...)`.
    metadata
        Comment-line key/value pairs, e.g. `"Lattice"`, `"energy"`, `"pbc"`.
    """

    n_atoms: int
    columns: dict[str, ColumnValues]
    metadata: dict[str, MetadataValue]

    def to_atoms(self) -> Atoms:
        """Convert to `ase.Atoms` (requires the optional `ase` extra)."""
        from oxyz.ase import to_atoms

        return to_atoms(self)


@overload
def read(
    path: str | Path,
    index: int,
    *,
    threads: int | None = ...,
    schema: SchemaSpec | str | Path | None = ...,
    conformance: Conformance = ...,
    mode: Mode | None = ...,
    compression: Compression = ...,
    member: str | None = ...,
    storage_options: _remote.StorageOptions | None = ...,
) -> Frame: ...


@overload
def read(
    path: str | Path,
    index: str,
    *,
    threads: int | None = ...,
    schema: SchemaSpec | str | Path | None = ...,
    conformance: Conformance = ...,
    mode: Mode | None = ...,
    compression: Compression = ...,
    member: str | None = ...,
    storage_options: _remote.StorageOptions | None = ...,
) -> Frame | list[Frame]: ...


@overload
def read(
    path: str | Path,
    index: slice | Sequence[int] = ...,
    *,
    threads: int | None = ...,
    schema: SchemaSpec | str | Path | None = ...,
    conformance: Conformance = ...,
    mode: Mode | None = ...,
    compression: Compression = ...,
    member: str | None = ...,
    storage_options: _remote.StorageOptions | None = ...,
) -> list[Frame]: ...


def read(  # noqa: PLR0913  the index/schema/projection/source options are the contract
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
) -> Frame | list[Frame]:
    """Read frames from an extxyz file into numpy-backed `Frame` objects.

    Reads run on all cores by default, and a bounded or single-frame selection
    stops as soon as the last requested frame is read. A compressed path is
    decoded on the fly, so reads stay parallel without decompressing to a
    temporary file; a remote URL streams the object through the same parser. A
    remote or compressed source cannot seek, so a negative or reverse selection
    there reads the whole file and indexes in memory (as ASE does). For
    constant memory over a large file, stream with `iread` instead.

    Parameters
    ----------
    path
        File path, an S3-compatible URL (``s3://``, ``gs://``, ``az://`` —
        needs the ``oxyz[s3]`` extra), or `"-"` for stdin.
    index
        Which frames to return: `":"` (default) reads all; an `int` returns a
        single `Frame`; a slice or slice-string (`"1:10:2"`) or a sequence of
        non-negative ints selects a subset, in order (repeats allowed).
    threads
        `None` parses on every core; `1` is the exact serial path. Results and
        errors are identical either way.
    schema
        A `SchemaSpec`, or a path to a `.json`/`.yaml`/`.toml` schema file,
        validated against each frame as it is read. See `oxyz.SchemaSpec`.
    conformance
        How a schema deviation is handled: `"strict"`, `"required"` (default),
        or `"warn"`.
    mode
        Overrides the schema's own `mode`. `"project"` reshapes each frame to
        the schema (extras dropped, optionals filled); an unfillable frame is
        dropped under `conformance="warn"`.
    compression
        Forces a codec (`"infer"`, `"none"`, `"gzip"`, `"zstd"`, `"zip"`)
        instead of inferring it from `path`.
    member
        Selects one entry from a `.zip`/`.tar`/`.tar.gz` holding more than one.
    storage_options
        Endpoint/credentials for a remote store, falling back to `AWS_*` env
        vars.

    Returns
    -------
    Frame or list[Frame]
        A single `Frame` for an integer `index`, otherwise a list.

    Raises
    ------
    oxyz.ParseError
        On malformed input, carrying the frame index and location.

    Examples
    --------
    >>> import oxyz
    >>> frames = oxyz.read("examples/data/water.extxyz")
    >>> len(frames)
    3
    >>> frames[0].columns["pos"].shape
    (3, 3)
    >>> frames[0].columns["pos"].dtype
    dtype('float64')
    """
    from oxyz._select import frames_for_read, gathered_frames, nth_frame, parse_index

    _check_threads(threads)
    _require_schema_for_mode(schema, mode)
    if isinstance(index, str):
        index = parse_index(index)
    if isinstance(index, int):
        return nth_frame(
            path,
            index,
            schema=schema,
            conformance=conformance,
            mode=mode,
            compression=compression,
            member=member,
            storage_options=storage_options,
        )
    if isinstance(index, slice):
        return list(
            frames_for_read(
                path,
                index,
                threads,
                schema=schema,
                conformance=conformance,
                mode=mode,
                compression=compression,
                member=member,
                storage_options=storage_options,
            )
        )
    return gathered_frames(
        path,
        index,
        threads,
        schema=schema,
        conformance=conformance,
        mode=mode,
        compression=compression,
        member=member,
        storage_options=storage_options,
    )


def iread(
    path: str | Path,
    index: int | str | slice | Sequence[int] = ":",
    *,
    schema: SchemaSpec | str | Path | None = None,
    conformance: Conformance = "required",
    mode: Mode | None = None,
    compression: Compression = "infer",
    member: str | None = None,
    storage_options: _remote.StorageOptions | None = None,
) -> Iterator[Frame]:
    """Iterate frames one at a time, in constant memory.

    Parameters and `index` semantics match `read`; frames are yielded lazily
    rather than materialised into a list, and an int index yields exactly one
    frame. The file stays open while iterating and closes when the iterator is
    dropped. After a parse error the stream position is untrustworthy, so
    iteration ends: the error is raised once, then `StopIteration`. Selecting
    an explicit sequence of frames reads eagerly — an arbitrary set cannot
    stream — while a slice or the default `":"` stays lazy. To materialise
    every frame at once (and in parallel), use `read` instead.

    Parameters
    ----------
    path
        File path, an S3-compatible URL, or `"-"` for stdin.
    index
        Selection grammar; see `read`.
    schema
        A `SchemaSpec`, or a path to a schema file; see `read`.
    conformance
        How a schema deviation is handled; see `read`.
    mode
        Overrides the schema's own `mode`; see `read`.
    compression
        Forces a codec instead of inferring it from `path`; see `read`.
    member
        Selects one entry from an archive holding more than one; see `read`.
    storage_options
        Endpoint/credentials for a remote store; see `read`.

    Returns
    -------
    Iterator[Frame]
        Frames in file order (or in `index` order, for a sequence).

    Raises
    ------
    oxyz.ParseError
        On malformed input, carrying the frame index and location.

    Examples
    --------
    >>> import oxyz
    >>> [f.columns["pos"].shape for f in oxyz.iread("examples/data/water.extxyz")]
    [(3, 3), (3, 3), (3, 3)]
    """
    from oxyz._select import gathered_frames, nth_frame, parse_index, sliced_frames

    _require_schema_for_mode(schema, mode)
    if isinstance(index, str):
        index = parse_index(index)
    if isinstance(index, int):
        return iter(
            (
                nth_frame(
                    path,
                    index,
                    schema=schema,
                    conformance=conformance,
                    mode=mode,
                    compression=compression,
                    member=member,
                    storage_options=storage_options,
                ),
            )
        )
    if isinstance(index, slice):
        return sliced_frames(
            path,
            index,
            schema=schema,
            conformance=conformance,
            mode=mode,
            compression=compression,
            member=member,
            storage_options=storage_options,
        )
    return iter(
        gathered_frames(
            path,
            index,
            None,
            schema=schema,
            conformance=conformance,
            mode=mode,
            compression=compression,
            member=member,
            storage_options=storage_options,
        )
    )


def _check_threads(threads: int | None) -> None:
    """Reject an invalid `threads` value.

    `None` parses on all cores, an integer >= 1 sets the count. Reject 0 and
    negatives rather than letting rayon read `num_threads(0)` as "all cores".
    """
    if threads is not None and threads < 1:
        raise ValueError(f"threads must be a positive integer or None, got {threads!r}")


def _projection(
    schema: SchemaSpec | str | Path, mode: Mode | None
) -> tuple[ProjectionPlan | None, SchemaSpec]:
    """Resolve `schema` and compile its projection plan.

    Returns the crossing plan (`None` under validate mode) and the resolved
    spec, for a reader that was given a `schema`.
    """
    from oxyz._project import compile_projection, effective_mode
    from oxyz._schema_spec import SchemaSpec

    spec = schema if isinstance(schema, SchemaSpec) else SchemaSpec.from_file(schema)
    return compile_projection(spec, effective_mode(spec, mode)), spec


def _require_schema_for_mode(schema: object, mode: Mode | None) -> None:
    if mode is not None and schema is None:
        raise ValueError("mode= requires a schema= to project or validate against")


def _frame_rule_compiled(spec: SchemaSpec) -> CompiledSpec | None:
    """Compile `spec`'s frame rule, if it has one.

    Returns `None` when the spec has no frame rule. Columns and metadata
    already conform after projection, so only frame-axis (`n_atoms`,
    `lattice`) checks bite.
    """
    if spec.frame is None:
        return None
    from oxyz import _schema_match

    return _schema_match.compile_spec(spec)


def _keep_projected(
    raw: Iterable[ProjectedFrame],
    conformance: Conformance,
    spec: SchemaSpec,
    indices: Iterable[int],
) -> list[Frame]:
    """Apply projection policy to `(data, deviations)` items and return the kept frames.

    Items are paired with their file `indices`: raise/warn/drop, then
    frame-rule-check the survivors.
    """
    from oxyz import _schema_match
    from oxyz._project import enforce_projection

    frame_compiled = _frame_rule_compiled(spec)
    out: list[Frame] = []
    for index, (data, deviations) in zip(indices, raw, strict=True):
        keep = enforce_projection(deviations, conformance, index, data is None)
        if keep and data is not None:
            frame = _frame_from_data(data)
            if frame_compiled is not None:
                _schema_match.enforce_frame(frame, frame_compiled, conformance, index)
            out.append(frame)
    return out


def _read_all(
    path: str | Path,
    *,
    threads: int | None = None,
    schema: SchemaSpec | str | Path | None = None,
    conformance: Conformance = "required",
    mode: Mode | None = None,
    compression: Compression = "infer",
    member: str | None = None,
    storage_options: _remote.StorageOptions | None = None,
) -> list[Frame]:
    """Read and materialise every frame.

    The whole-file primitive behind `read`/`iread`; parses on all cores unless
    `threads=1`.
    """
    _check_threads(threads)
    _require_schema_for_mode(schema, mode)
    plan = spec = None
    if schema is not None:
        plan, spec = _projection(schema, mode)

    if plan is not None:
        assert spec is not None  # noqa: S101 — set alongside plan above
        if _remote.is_remote(path):
            src = _remote.open_source(
                path,
                compression=compression,
                member=member,
                storage_options=storage_options,
            )
            raw = _rust.read_frames_projected_reader(
                src.obj, src.codec, src.member, threads, plan=plan
            )
        else:
            raw = _rust.read_frames_projected(
                str(path), threads, compression, member, plan=plan
            )
        return _keep_projected(raw, conformance, spec, range(len(raw)))

    if _remote.is_remote(path):
        src = _remote.open_source(
            path,
            compression=compression,
            member=member,
            storage_options=storage_options,
        )
        data = _rust.read_frames_reader(src.obj, src.codec, src.member, threads)
    else:
        data = _rust.read_frames(str(path), threads, compression, member)
    frames = [_frame_from_data(frame) for frame in data]
    if schema is not None:
        from oxyz import _schema_match

        compiled = _schema_match.resolve(schema)
        for index, frame in enumerate(frames):
            _schema_match.enforce_frame(frame, compiled, conformance, index)
    return frames


def _read_all_sliced(  # noqa: PLR0913  the read/schema/projection options are the contract
    path: str | Path,
    frames: slice,
    threads: int | None = None,
    *,
    schema: SchemaSpec | str | Path | None = None,
    conformance: Conformance = "required",
    mode: Mode | None = None,
    compression: Compression = "infer",
    member: str | None = None,
    storage_options: _remote.StorageOptions | None = None,
) -> list[Frame]:
    """Read every frame, then apply `frames` before wrapping.

    Like `_read_all`, but parse every frame (`threads=None`: all cores), then
    build `Frame` objects only for those the slice keeps — so an unbounded
    forward slice that drops a prefix or steps (`"1000:"`, `"::2"`) does not
    pay to wrap the frames it discards.
    """
    _require_schema_for_mode(schema, mode)
    plan = spec = None
    if schema is not None:
        plan, spec = _projection(schema, mode)

    if plan is not None:
        assert spec is not None  # noqa: S101 — set alongside plan above
        if _remote.is_remote(path):
            src = _remote.open_source(
                path,
                compression=compression,
                member=member,
                storage_options=storage_options,
            )
            raw = _rust.read_frames_projected_reader(
                src.obj, src.codec, src.member, threads, plan=plan
            )
        else:
            raw = _rust.read_frames_projected(
                str(path), threads, compression, member, plan=plan
            )
        indices = range(len(raw))[frames]
        return _keep_projected(raw[frames], conformance, spec, indices)

    if _remote.is_remote(path):
        src = _remote.open_source(
            path,
            compression=compression,
            member=member,
            storage_options=storage_options,
        )
        data = _rust.read_frames_reader(src.obj, src.codec, src.member, threads)
    else:
        data = _rust.read_frames(str(path), threads, compression, member)
    indices = range(len(data))[frames]
    result = [_frame_from_data(frame) for frame in data[frames]]
    if schema is not None:
        from oxyz import _schema_match

        compiled = _schema_match.resolve(schema)
        for index, frame in zip(indices, result, strict=True):
            _schema_match.enforce_frame(frame, compiled, conformance, index)
    return result


class IndexedFrames:
    """Random-access reader: scans on open, then reads frames in any order.

    Internal for now — the public random-access surface is `oxyz.scan`
    plus the completed index grammar in `oxyz.ase`.
    """

    def __init__(self, path: str | Path) -> None:
        self._inner = _rust.IndexedFrames(str(path))

    def __len__(self) -> int:
        return len(self._inner)

    def get(self, frame_index: int) -> Frame:
        return _frame_from_data(self._inner.get(frame_index))


def _iter_all(
    path: str | Path,
    *,
    schema: SchemaSpec | str | Path | None = None,
    conformance: Conformance = "required",
    mode: Mode | None = None,
    compression: Compression = "infer",
    member: str | None = None,
    storage_options: _remote.StorageOptions | None = None,
) -> Iterator[Frame]:
    """Stream every frame in constant memory.

    The whole-file primitive behind `read`/`iread`; the file closes when the
    iterator is dropped.
    """
    _require_schema_for_mode(schema, mode)
    plan = spec = None
    if schema is not None:
        plan, spec = _projection(schema, mode)

    if plan is not None:
        assert spec is not None  # noqa: S101 — set alongside plan above
        from oxyz import _schema_match
        from oxyz._project import enforce_projection

        if _remote.is_remote(path):
            src = _remote.open_source(
                path,
                compression=compression,
                member=member,
                storage_options=storage_options,
            )
            projected = _rust.FrameIterProjected.from_reader(
                src.obj, plan, src.codec, src.member
            )
        else:
            projected = _rust.FrameIterProjected(str(path), plan, compression, member)
        frame_compiled = _frame_rule_compiled(spec)
        for index, (data, deviations) in enumerate(projected):
            keep = enforce_projection(deviations, conformance, index, data is None)
            if keep and data is not None:
                frame = _frame_from_data(data)
                if frame_compiled is not None:
                    _schema_match.enforce_frame(
                        frame, frame_compiled, conformance, index
                    )
                yield frame
        return

    if _remote.is_remote(path):
        src = _remote.open_source(
            path,
            compression=compression,
            member=member,
            storage_options=storage_options,
        )
        iterator = _rust.FrameIter.from_reader(src.obj, src.codec, src.member)
    else:
        iterator = _rust.FrameIter(str(path), compression, member)
    compiled = None
    if schema is not None:
        from oxyz import _schema_match

        compiled = _schema_match.resolve(schema)
    for index, data in enumerate(iterator):
        frame = _frame_from_data(data)
        if compiled is not None:
            _schema_match.enforce_frame(frame, compiled, conformance, index)
        yield frame


def _frame_from_data(data: _rust.FrameData) -> Frame:
    return Frame(
        n_atoms=data["n_atoms"],
        columns=data["columns"],
        metadata=data["metadata"],
    )
