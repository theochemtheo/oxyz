from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast, overload

import numpy as np

from oxyz import _convert, _html, _remote, _rust, _summary
from oxyz._lookup import NameLookup

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from pathlib import Path

    from ase import Atoms
    from typing_extensions import TypeForm

    from oxyz._record import Record
    from oxyz._record_bind import Binding
    from oxyz._rust import ProjectedFrame, ProjectionPlan
    from oxyz._schema_match import CompiledSpec, Conformance
    from oxyz._schema_spec import Mode, SchemaSpec

ColumnValues = np.ndarray | list[str] | list[list[str]]
"""A per-atom column's values: an array for numeric/bool data, `list[str]` for
a 1-D string column, `list[list[str]]` for a 2-D one."""

# Numeric and bool metadata are arrays; 2-D string metadata is list[list[str]].
MetadataValue = float | int | bool | str | np.ndarray | list[str] | list[list[str]]
"""A comment-line metadata value: a scalar, an array, or (for 2-D string
metadata) `list[list[str]]`."""

Compression = Literal[
    "infer", "none", "gzip", "zstd", "zip", "tar", "tar.gz", "tar.zst"
]
"""How to read a possibly-compressed file. `"infer"` detects the codec from the
extension, falling back to the leading magic bytes; the rest force a codec. A
tar has no magic bytes, so forcing is the only way to read one whose name does
not say `.tar`, `.tar.gz` or `.tar.zst`."""


@dataclass(frozen=True, slots=True, eq=False)
class Frame(NameLookup[ColumnValues, MetadataValue]):  # noqa: PLW1641  a dict of arrays is unhashable by construction
    """One parsed extxyz frame: per-atom columns plus comment-line metadata.

    Both dicts keep file order, and names and values are kept as written:
    no aliasing (`force` vs `forces`) and no conversion. A repeated metadata
    key keeps its last value.

    `frame[name]` looks a name up in `columns`, then `metadata`; `in`, `get`,
    `keys` and `dict(frame)` follow suit, and a name in both raises
    `oxyz.AmbiguousNameError`. A frame is not a `Mapping`: it has no `len()`
    and cannot be iterated, so numpy keeps a list of frames as frames.

    The derived accessors resolve a well-known field on each access, and never
    cache or rewrite what the frame stores.

    Attributes
    ----------
    n_atoms
        Number of atoms in the frame.
    columns
        Per-atom data keyed by column name, each shaped `(n_atoms, ...)`.
    metadata
        Comment-line key/value pairs, e.g. `"Lattice"`, `"energy"`, `"pbc"`.
    positions
        The `pos` column.
    numbers
        Atomic numbers as int32.
    symbols
        Chemical symbols.
    cell
        The 3x3 cell in ASE's row-vector convention.
    pbc
        Three periodic-boundary flags.
    """

    n_atoms: int
    columns: dict[str, ColumnValues]
    metadata: dict[str, MetadataValue]

    def __repr__(self) -> str:
        """Summarise the frame's shape without printing any of its data."""
        return (
            f"Frame(n_atoms={self.n_atoms}, "
            f"columns={_summary.mapping_repr(self.columns)}, "
            f"metadata={_summary.mapping_repr(self.metadata)})"
        )

    def _repr_html_(self) -> str:
        """Render as tables for a notebook, previewing a bounded slice of values."""
        return _html.frame(self)

    def __eq__(self, other: object) -> bool:
        """Compare atom counts and every stored value.

        Key order is ignored and NaN compares equal, so a written-then-read frame
        equals the one it came from.
        """
        if not isinstance(other, Frame):
            return NotImplemented
        return (
            self.n_atoms == other.n_atoms
            and _summary.mappings_equal(self.columns, other.columns)
            and _summary.mappings_equal(self.metadata, other.metadata)
        )

    def _sides(self) -> tuple[dict[str, ColumnValues], dict[str, MetadataValue]]:
        return self.columns, self.metadata

    @property
    def positions(self) -> np.ndarray:
        """The `pos` column, shaped `(n_atoms, 3)`.

        The stored array, not a copy. Raises `oxyz.FieldError` without a `pos`
        column.
        """
        return _convert.positions(self.columns)

    @property
    def numbers(self) -> np.ndarray:
        """Atomic numbers as int32, from `Z`, else `numbers`, else `species`.

        Recomputed on each access. Raises `oxyz.FieldError` when no such column
        exists, or a species token is not a chemical symbol.
        """
        return _convert.numbers(self.columns)

    @property
    def symbols(self) -> list[str]:
        """Chemical symbols: the `species` column, else mapped from `numbers`.

        A `species` column is returned as stored. Raises `oxyz.FieldError` when
        there is neither.
        """
        return _convert.symbols(self.columns)

    @property
    def cell(self) -> np.ndarray:
        """The 3x3 float64 cell in ASE's row-vector convention.

        Built from the flat, Fortran-order `metadata["Lattice"]`; all-zero without
        one. Raises `oxyz.FieldError` when `Lattice` is not 9 values.
        """
        return _convert.cell(self.metadata)

    @property
    def pbc(self) -> np.ndarray:
        """Three periodic-boundary flags, from `metadata["pbc"]`.

        Defaults to all-true with a `Lattice` and all-false without; a scalar
        broadcasts. An already-boolean array is returned as stored. Raises
        `oxyz.FieldError` for any other shape.
        """
        return _convert.pbc(self.metadata)

    def to_atoms(self) -> Atoms:
        """Convert to `ase.Atoms` (requires the optional `ase` extra)."""
        from oxyz.ase import to_atoms

        return to_atoms(self)


@overload
def read[R: Record](
    path: str | Path,
    index: int,
    *,
    record: TypeForm[R],
    threads: int | None = ...,
    schema: SchemaSpec | str | Path | None = ...,
    conformance: Conformance = ...,
    compression: Compression = ...,
    member: str | None = ...,
    storage_options: _remote.StorageOptions | None = ...,
) -> R: ...


@overload
def read[R: Record](
    path: str | Path,
    index: str,
    *,
    record: TypeForm[R],
    threads: int | None = ...,
    schema: SchemaSpec | str | Path | None = ...,
    conformance: Conformance = ...,
    compression: Compression = ...,
    member: str | None = ...,
    storage_options: _remote.StorageOptions | None = ...,
) -> R | list[R]: ...


@overload
def read[R: Record](
    path: str | Path,
    index: slice | Sequence[int] = ...,
    *,
    record: TypeForm[R],
    threads: int | None = ...,
    schema: SchemaSpec | str | Path | None = ...,
    conformance: Conformance = ...,
    compression: Compression = ...,
    member: str | None = ...,
    storage_options: _remote.StorageOptions | None = ...,
) -> list[R]: ...


@overload
def read(
    path: str | Path,
    index: int,
    *,
    record: None = ...,
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
    record: None = ...,
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
    record: None = ...,
    threads: int | None = ...,
    schema: SchemaSpec | str | Path | None = ...,
    conformance: Conformance = ...,
    mode: Mode | None = ...,
    compression: Compression = ...,
    member: str | None = ...,
    storage_options: _remote.StorageOptions | None = ...,
) -> list[Frame]: ...


def read[R: Record](  # noqa: PLR0913  the index/schema/projection/source options are the contract
    path: str | Path,
    index: int | str | slice | Sequence[int] = ":",
    *,
    record: TypeForm[R] | None = None,
    threads: int | None = None,
    schema: SchemaSpec | str | Path | None = None,
    conformance: Conformance = "required",
    mode: Mode | None = None,
    compression: Compression = "infer",
    member: str | None = None,
    storage_options: _remote.StorageOptions | None = None,
) -> Frame | list[Frame] | R | list[R]:
    """Read frames from an extxyz file into numpy-backed `Frame` objects.

    Parses on all cores by default, and a selection stops reading after its
    last frame. Compressed and remote sources stream; as they cannot seek, a
    negative or reverse selection there reads the whole file. For bounded
    memory, use `iread`.

    Parameters
    ----------
    path
        File path, a remote URL (``s3://``, ``gs://``, ``az://``, ``hf://``
        — needs the ``oxyz[s3]`` extra), or `"-"` for stdin.
    index
        Which frames to return: `":"` (default) reads all; an `int` returns a
        single `Frame`; a slice or slice-string (`"1:10:2"`) or a sequence of
        non-negative ints selects a subset, in order (repeats allowed).
    record
        A `Record` subclass, or a union of sibling ones (`A | B`, or a
        `type` alias of one), to read each frame into instead of a `Frame`.
        Each frame is projected onto the record's fields, as `Record.spec`
        describes them, and its arrays are writable and belong to that
        record alone. With a union, each frame becomes the one member that
        admits it: of the keys not every member requires, the member
        declares each one the frame carries and requires none it lacks. A
        frame no member admits is an error, or dropped with a warning under
        `conformance="warn"`. A record read checks only the frames `index`
        selects. A non-negative integer or a bounded forward slice stops
        reading after its last frame; any other `index` parses the whole
        file. Cannot be combined with `mode`. With `record`, `schema`
        describes the dataset rather than reshaping it: its frame rule
        applies (one that requires `Lattice` needs a field that reads it,
        such as a `Cell`), and its rule for any key the record reads must
        agree with the field's kind and width or shape (an integer rule
        agrees with a float field unless the record is `strict`); rules for
        other keys are not applied. Under `conformance="warn"`, a frame missing a
        required field, or carrying a field of the wrong kind, is dropped
        with a warning. `index` still counts the file's frames: a slice or
        sequence gives the records of the selected frames that were kept,
        in selection order, so the list may be shorter than the selection,
        and an integer naming a dropped frame raises `IndexError`.
    threads
        `None` parses on every core; `1` is serial. Results and errors are
        identical.
    schema
        A `SchemaSpec`, or a path to a `.json`/`.yaml`/`.toml` schema file,
        validated against each frame. See `oxyz.SchemaSpec`.
    conformance
        How a schema deviation is handled: `"strict"`, `"required"` (default),
        or `"warn"`.
    mode
        Overrides the schema's own `mode`. `"project"` reshapes each frame to
        the schema; an unfillable frame is dropped under `conformance="warn"`.
    compression
        Forces a codec (`"infer"`, `"none"`, `"gzip"`, `"zstd"`, `"zip"`,
        `"tar"`, `"tar.gz"`, `"tar.zst"`) instead of inferring it from `path`.
    member
        Selects one entry from a `.zip`/`.tar`/`.tar.gz`/`.tar.zst` holding
        more than one. `"*"` reads every `.xyz`/`.extxyz` entry in archive order
        as one stream, skipping macOS resource forks. A blank count line ends
        its entry, not the stream, and an entry ending inside a frame raises
        `OSError` naming it. Other errors locate the frame within the stream,
        not the entry.
    storage_options
        Endpoint/credentials for a remote store, falling back to `AWS_*` env
        vars.

    Returns
    -------
    Frame or list[Frame], or R or list[R] with `record=R`
        A single `Frame` for an integer `index`, otherwise a list. With
        `record`, instances of it, or of a union's members, in place of
        frames.

    Raises
    ------
    oxyz.ParseError
        On malformed input, carrying the frame index and location.
    oxyz.SchemaError
        With `record`, under `conformance="strict"` or `"required"`: a frame
        lacks a required field or has one of the wrong kind, with the frame
        index and key, or no member of a union admits it. Before reading:
        `schema` disagrees with the record, or its frame rule requires
        `Lattice` and no field reads it.
    IndexError
        An integer `index` is out of range; with `record`, it names a frame
        dropped under `conformance="warn"`.
    TypeError
        `record` is not a `Record` subclass or a union of sibling ones that
        agree on their shared keys and tell each frame apart; a
        `RecordBatch` class is for `read_batch`.
    ValueError
        `mode` is given with `record`.

    Examples
    --------
    >>> import oxyz
    >>> frames = oxyz.read(DATA / "water.extxyz")
    >>> len(frames)
    3
    >>> frames[0].columns["pos"].shape
    (3, 3)
    >>> frames[0].columns["pos"].dtype
    dtype('float64')
    """
    from oxyz._select import frames_for_read, gathered_frames, nth_frame, parse_index

    _check_threads(threads)
    if record is not None:
        from oxyz._select import read_records

        binding = _bind_record(record, schema, mode)
        return cast(
            "R | list[R]",
            read_records(
                path,
                index,
                binding,
                threads=threads,
                conformance=conformance,
                compression=compression,
                member=member,
                storage_options=storage_options,
            ),
        )
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


@overload
def iread[R: Record](
    path: str | Path,
    index: int | str | slice | Sequence[int] = ...,
    *,
    record: TypeForm[R],
    schema: SchemaSpec | str | Path | None = ...,
    conformance: Conformance = ...,
    compression: Compression = ...,
    member: str | None = ...,
    storage_options: _remote.StorageOptions | None = ...,
) -> Iterator[R]: ...


@overload
def iread(
    path: str | Path,
    index: int | str | slice | Sequence[int] = ...,
    *,
    record: None = ...,
    schema: SchemaSpec | str | Path | None = ...,
    conformance: Conformance = ...,
    mode: Mode | None = ...,
    compression: Compression = ...,
    member: str | None = ...,
    storage_options: _remote.StorageOptions | None = ...,
) -> Iterator[Frame]: ...


def iread[R: Record](  # noqa: PLR0913  the index/schema/projection/source options are the contract
    path: str | Path,
    index: int | str | slice | Sequence[int] = ":",
    *,
    record: TypeForm[R] | None = None,
    schema: SchemaSpec | str | Path | None = None,
    conformance: Conformance = "required",
    mode: Mode | None = None,
    compression: Compression = "infer",
    member: str | None = None,
    storage_options: _remote.StorageOptions | None = None,
) -> Iterator[Frame] | Iterator[R]:
    """Iterate frames one at a time, in memory bounded by the largest frame.

    The file stays open until the iterator is dropped. After a parse error the
    error is raised once and iteration ends. A sequence `index` reads eagerly,
    as an arbitrary set cannot stream.

    Parameters
    ----------
    path
        File path, an S3-compatible URL, or `"-"` for stdin.
    index
        Selection grammar; see `read`.
    record
        A `Record` subclass, or a union of sibling ones, to yield instances
        of instead of frames: the records `read` returns for the same
        arguments, with the same frames dropped. A forward slice streams,
        and a non-negative integer reads no further than its frame; a
        negative integer, a reverse slice or a sequence reads the whole file
        first.
    schema
        A `SchemaSpec`, or a path to a schema file; see `read`.
    conformance
        How a schema deviation is handled; see `read`.
    mode
        Overrides the schema's own `mode`; see `read`.
    compression
        Forces a codec instead of inferring it from `path`; see `read`.
    member
        Selects one entry from an archive holding more than one, or all of them
        with `"*"`; see `read`.
    storage_options
        Endpoint/credentials for a remote store; see `read`.

    Returns
    -------
    Iterator[Frame], or Iterator[R] with `record=R`
        Frames, or records with `record`, in file order (or in `index`
        order, for a sequence).

    Raises
    ------
    oxyz.ParseError
        On malformed input, carrying the frame index and location.
    oxyz.SchemaError
        With `record`, as for `read`. A disagreeing `schema` raises from
        this call, before any frame is read.
    TypeError
        `record` is not a `Record` subclass or a union of sibling ones that
        agree on their shared keys and tell each frame apart; a
        `RecordBatch` class is for `read_batch`.
    ValueError
        `mode` is given with `record`.

    Examples
    --------
    >>> import oxyz
    >>> [f.columns["pos"].shape for f in oxyz.iread(DATA / "water.extxyz")]
    [(3, 3), (3, 3), (3, 3)]
    """
    from oxyz._select import gathered_frames, nth_frame, parse_index, sliced_frames

    if record is not None:
        from oxyz._select import iread_records

        binding = _bind_record(record, schema, mode)
        return cast(
            "Iterator[R]",
            iread_records(
                path,
                index,
                binding,
                conformance=conformance,
                compression=compression,
                member=member,
                storage_options=storage_options,
            ),
        )
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
    """Reject 0 and negatives, which rayon would read as all cores."""
    if threads is not None and threads < 1:
        raise ValueError(f"threads must be a positive integer or None, got {threads!r}")


def _bind_record(
    record: object,
    schema: SchemaSpec | str | Path | None,
    mode: Mode | None,
    *,
    batch: bool = False,
) -> Binding:
    from oxyz._record_bind import bind

    if mode is not None:
        raise ValueError(
            "mode= cannot be combined with record=: a record read always projects"
        )
    return bind(record, schema, batch=batch)


def _projection(
    schema: SchemaSpec | str | Path, mode: Mode | None
) -> tuple[ProjectionPlan | None, SchemaSpec]:
    """Resolve `schema` and its projection plan (`None` under validate mode)."""
    from oxyz._project import compile_projection, effective_mode
    from oxyz._schema_spec import SchemaSpec

    spec = schema if isinstance(schema, SchemaSpec) else SchemaSpec.from_file(schema)
    return compile_projection(spec, effective_mode(spec, mode)), spec


def _require_schema_for_mode(schema: object, mode: Mode | None) -> None:
    if mode is not None and schema is None:
        raise ValueError("mode= requires a schema= to project or validate against")


def _frame_rule_compiled(spec: SchemaSpec) -> CompiledSpec | None:
    """Compile `spec`'s frame rule, or `None` if it has none."""
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
    """Apply projection policy to projected frames; return the kept ones."""
    from oxyz import _schema_match
    from oxyz._project import enforce_projection

    frame_compiled = _frame_rule_compiled(spec)
    out: list[Frame] = []
    for index, (data, deviations, _) in zip(indices, raw, strict=True):
        keep = enforce_projection(deviations, conformance, index, dropped=data is None)
        if keep and data is not None:
            frame = _frame_from_data(data)
            if frame_compiled is not None:
                _schema_match.enforce_frame(frame, frame_compiled, conformance, index)
            out.append(frame)
    return out


def _read_projected(
    path: str | Path,
    plan: ProjectionPlan,
    threads: int | None,
    compression: Compression,
    member: str | None,
    storage_options: _remote.StorageOptions | None,
) -> list[ProjectedFrame]:
    """Project every frame of `path` with `plan`."""
    if _remote.is_remote(path):
        src = _remote.open_source(
            path,
            compression=compression,
            member=member,
            storage_options=storage_options,
        )
        return _rust.read_frames_projected_reader(
            src.obj, src.codec, src.member, threads, plan=plan
        )
    return _rust.read_frames_projected(
        str(path), threads, compression, member, plan=plan
    )


def _stream_projected(
    path: str | Path,
    plan: ProjectionPlan,
    compression: Compression,
    member: str | None,
    storage_options: _remote.StorageOptions | None,
) -> Iterator[ProjectedFrame]:
    """Project the frames of `path` with `plan`, one at a time."""
    if _remote.is_remote(path):
        src = _remote.open_source(
            path,
            compression=compression,
            member=member,
            storage_options=storage_options,
        )
        return iter(
            _rust.FrameIterProjected.from_reader(src.obj, plan, src.codec, src.member)
        )
    return iter(_rust.FrameIterProjected(str(path), plan, compression, member))


def _open_first_frame(
    path: str | Path,
    compression: Compression,
    member: str | None,
    storage_options: _remote.StorageOptions | None,
) -> None:
    """Open `path` and parse its first frame, raising what reading it would."""
    if _remote.is_remote(path):
        src = _remote.open_source(
            path,
            compression=compression,
            member=member,
            storage_options=storage_options,
        )
        frames = _rust.FrameIter.from_reader(src.obj, src.codec, src.member)
    else:
        frames = _rust.FrameIter(str(path), compression, member)
    next(iter(frames), None)


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
    """Read and materialise every frame."""
    _check_threads(threads)
    _require_schema_for_mode(schema, mode)
    plan = spec = None
    if schema is not None:
        plan, spec = _projection(schema, mode)

    if plan is not None:
        assert spec is not None  # noqa: S101 — set alongside plan above
        raw = _read_projected(path, plan, threads, compression, member, storage_options)
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
    if spec is not None:
        from oxyz import _schema_match

        compiled = _schema_match.compile_spec(spec)
        for index, frame in enumerate(frames):
            _schema_match.enforce_frame(frame, compiled, conformance, index)
    return frames


def _read_all_sliced(
    path: str | Path,
    frames: slice,
    threads: int | None = None,
    *,
    compression: Compression = "infer",
    member: str | None = None,
    storage_options: _remote.StorageOptions | None = None,
) -> list[Frame]:
    """Read every frame, wrapping only those `frames` keeps."""
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
    return [_frame_from_data(frame) for frame in data[frames]]


class IndexedFrames:
    """Random-access reader: scans on open, then reads frames in any order."""

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
    """Stream every frame; the file closes when the iterator is dropped."""
    _require_schema_for_mode(schema, mode)
    plan = spec = None
    if schema is not None:
        plan, spec = _projection(schema, mode)

    if plan is not None:
        assert spec is not None  # noqa: S101 — set alongside plan above
        from oxyz import _schema_match
        from oxyz._project import enforce_projection

        projected = _stream_projected(path, plan, compression, member, storage_options)
        frame_compiled = _frame_rule_compiled(spec)
        for index, (data, deviations, _) in enumerate(projected):
            keep = enforce_projection(
                deviations, conformance, index, dropped=data is None
            )
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
    if spec is not None:
        from oxyz import _schema_match

        compiled = _schema_match.compile_spec(spec)
    for index, data in enumerate(iterator):
        frame = _frame_from_data(data)
        if compiled is not None:
            _schema_match.enforce_frame(frame, compiled, conformance, index)
        yield frame


def _cast_projected(
    items: Iterable[tuple[int, ProjectedFrame]],
    conformance: Conformance,
    binding: Binding,
) -> Iterator[Record | None]:
    """Apply projection policy to each frame and cast it; `None` where it dropped.

    A field of the wrong kind drops the frame even where the field has a fill,
    and the binding's cast drops a frame no member of a union admits.
    """
    from oxyz import _schema_match
    from oxyz._project import enforce_projection, mismatched

    frame_compiled = _frame_rule_compiled(binding.spec)
    build = binding.caster(conformance)
    for index, (data, deviations, present) in items:
        dropped = data is None or mismatched(deviations)
        keep = enforce_projection(deviations, conformance, index, dropped=dropped)
        if not keep or data is None:
            yield None
            continue
        frame = _frame_from_data(data)
        if frame_compiled is not None:
            _schema_match.enforce_frame(
                binding.rule_frame(frame, present),
                frame_compiled,
                conformance,
                index,
                frame_rule_only=True,
            )
        yield build(frame, index)


def _frame_from_data(data: _rust.FrameData) -> Frame:
    return Frame(
        n_atoms=data["n_atoms"],
        columns=data["columns"],
        metadata=data["metadata"],
    )
