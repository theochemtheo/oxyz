from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

import oxyz._rust as _rust
from oxyz import _remote

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

    from ase import Atoms

    from oxyz._rust import ProjectedFrame, ProjectionPlan
    from oxyz._schema_match import CompiledSpec, Conformance
    from oxyz._schema_spec import Mode, SchemaSpec

ColumnValues = np.ndarray | list[str] | list[list[str]]
MetadataValue = float | int | bool | str | np.ndarray | list[str]

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
    """

    n_atoms: int
    columns: dict[str, ColumnValues]
    metadata: dict[str, MetadataValue]

    def to_ase(self) -> Atoms:
        """Convert to `ase.Atoms` (requires the optional `ase` extra)."""
        from oxyz.ase import to_atoms

        return to_atoms(self)


def read_first(
    path: str | Path,
    *,
    schema: SchemaSpec | str | Path | None = None,
    conformance: Conformance = "required",
    mode: Mode | None = None,
    compression: Compression = "infer",
    member: str | None = None,
    storage_options: _remote.StorageOptions | None = None,
) -> Frame:
    """Read only the first frame, stopping there.

    Cheaper than `read_frames(path)[0]`, which parses the whole file.

    A compressed path (`.gz`, `.zst`, `.zip`, `.tar.gz`, `.tar`) is decoded on
    the fly; `compression` overrides the codec and `member` names one entry in a
    multi-member archive. See `read_frames` for details.

    A remote URL (``s3://``, ``gs://``, ``az://``) streams the object through
    the same parser (needs the ``oxyz[s3]`` extra); ``storage_options`` passes
    endpoint/credentials to the store, falling back to ``AWS_*`` env vars.

    `schema` (a `SchemaSpec` or a path to a `.json`/`.yaml`/`.toml` file)
    validates the frame; `conformance` is `"strict"`, `"required"` (default),
    or `"warn"`. `mode` (`None`/`"validate"`/`"project"`) overrides the schema's
    own `mode`; under `project` the frame is reshaped to the schema (extras
    dropped, optionals filled). See `oxyz.SchemaSpec`.
    """
    _require_schema_for_mode(schema, mode)
    plan = spec = None
    if schema is not None:
        plan, spec = _projection(schema, mode)
    if _remote.is_remote(path):
        src = _remote.open_source(
            path,
            compression=compression,
            member=member,
            storage_options=storage_options,
        )
        if plan is not None:
            raw = _rust.read_first_frame_projected_reader(
                src.obj, src.codec, src.member, plan=plan
            )
        else:
            data = _rust.read_first_frame_reader(src.obj, src.codec, src.member)
    elif plan is not None:
        raw = _rust.read_first_frame_projected(
            str(path), compression, member, plan=plan
        )
    else:
        data = _rust.read_first_frame(str(path), compression, member)

    if plan is not None:
        assert spec is not None  # noqa: S101 — set alongside plan above
        kept = _keep_projected([raw], conformance, spec, [0])
        if not kept:
            from oxyz._schema_match import SchemaError

            raise SchemaError(
                "the first frame was dropped by projection (an unfillable "
                "required field); no frame to return",
                frame_index=0,
            )
        return kept[0]

    frame = _frame_from_data(data)
    if schema is not None:
        from oxyz import _schema_match

        _schema_match.enforce_frame(
            frame, _schema_match.resolve(schema), conformance, 0
        )
    return frame


def _check_threads(threads: int | None) -> None:
    """`None` parses on all cores, an integer >= 1 sets the count. Reject 0 and
    negatives rather than letting rayon read `num_threads(0)` as "all cores"."""
    if threads is not None and threads < 1:
        raise ValueError(f"threads must be a positive integer or None, got {threads!r}")


def _projection(
    schema: SchemaSpec | str | Path, mode: Mode | None
) -> tuple[ProjectionPlan | None, SchemaSpec]:
    """The crossing plan (`None` under validate mode) and the resolved spec, for
    a reader that was given a `schema`."""
    from oxyz._project import compile_projection, effective_mode
    from oxyz._schema_spec import SchemaSpec

    spec = schema if isinstance(schema, SchemaSpec) else SchemaSpec.from_file(schema)
    return compile_projection(spec, effective_mode(spec, mode)), spec


def _require_schema_for_mode(schema: object, mode: Mode | None) -> None:
    if mode is not None and schema is None:
        raise ValueError("mode= requires a schema= to project or validate against")


def _frame_rule_compiled(spec: SchemaSpec) -> CompiledSpec | None:
    """A compiled spec to check the frame rule against each projected frame, or
    `None` when the spec has no frame rule. Columns and metadata already conform
    after projection, so only frame-axis (`n_atoms`, `lattice`) checks bite."""
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
    """Apply projection policy to `(data, deviations)` items paired with their
    file `indices`: raise/warn/drop, then frame-rule-check the survivors."""
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


def read_frames(
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
    """Read every frame. Parses on all cores by default; `threads=1` streams
    serially. Results and errors are identical regardless of `threads`.

    For constant memory on a large file, stream with `iter_frames`.

    A compressed path is decoded while streaming, so reads stay parallel without
    decompressing to a temporary file. `compression` forces a codec (one of
    `"infer"`, `"none"`, `"gzip"`, `"zstd"`, `"zip"`) rather than inferring it
    from the name. `member` selects one entry from a `.zip`/`.tar`/`.tar.gz`
    holding more than one; with it omitted, an archive must contain exactly one
    extxyz file.

    A remote URL (``s3://``, ``gs://``, ``az://``) streams the object through
    the same parser (needs the ``oxyz[s3]`` extra); ``storage_options`` passes
    endpoint/credentials to the store, falling back to ``AWS_*`` env vars.

    `schema` (a `SchemaSpec` or a path to a `.json`/`.yaml`/`.toml` file)
    validates each frame; `conformance` is `"strict"`, `"required"` (default),
    or `"warn"`. `mode` (`None`/`"validate"`/`"project"`) overrides the schema's
    own `mode`; under `project` each frame is reshaped to the schema (extras
    dropped, optionals filled) and an unfillable frame is dropped under `warn`.
    See `oxyz.SchemaSpec`.
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


def read_frames_sliced(  # noqa: PLR0913  the read/schema/projection options are the contract
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
    """`read_frames`, but apply `frames` before wrapping: parse every frame
    (`threads=None`: all cores), then build `Frame` objects only for those the
    slice keeps.

    The parallel parse still touches the whole file, but a slice that drops a
    prefix or steps (`read(path, "1000:")`, `"::2"`) no longer pays to wrap the
    frames it immediately discards.

    A remote URL (``s3://``, ``gs://``, ``az://``) streams the object through
    the same parser (needs the ``oxyz[s3]`` extra); ``storage_options`` passes
    endpoint/credentials to the store, falling back to ``AWS_*`` env vars.

    `schema` (a `SchemaSpec` or a path to a `.json`/`.yaml`/`.toml` file)
    validates each kept frame against its original index; `conformance` is
    `"strict"`, `"required"` (default), or `"warn"`. `mode`
    (`None`/`"validate"`/`"project"`) overrides the schema's own `mode`; under
    `project` each kept frame is reshaped to the schema. See `oxyz.SchemaSpec`.
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


def iter_frames(
    path: str | Path,
    *,
    schema: SchemaSpec | str | Path | None = None,
    conformance: Conformance = "required",
    mode: Mode | None = None,
    compression: Compression = "infer",
    member: str | None = None,
    storage_options: _remote.StorageOptions | None = None,
) -> Iterator[Frame]:
    """Stream frames one at a time, in constant memory.

    The file stays open while iterating and closes when the iterator is
    dropped. After a parse error the stream position is untrustworthy, so
    iteration ends: the error is raised once, then StopIteration. To
    materialise every frame at once (and in parallel), use `read_frames`.

    A compressed path is decoded while streaming; see `read_frames` for the
    `compression` and `member` options.

    A remote URL (``s3://``, ``gs://``, ``az://``) streams the object through
    the same parser (needs the ``oxyz[s3]`` extra); ``storage_options`` passes
    endpoint/credentials to the store, falling back to ``AWS_*`` env vars.

    `schema` (a `SchemaSpec` or a path to a `.json`/`.yaml`/`.toml` file)
    validates each frame before it is yielded; `conformance` is `"strict"`,
    `"required"` (default), or `"warn"`. `mode` (`None`/`"validate"`/`"project"`)
    overrides the schema's own `mode`; under `project` each frame is reshaped to
    the schema and an unfillable frame is dropped under `warn`. See
    `oxyz.SchemaSpec`.
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
