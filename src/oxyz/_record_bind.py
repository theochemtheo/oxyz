"""Binding a record to a read: its projection plans and its casts."""

from __future__ import annotations

import dataclasses
import functools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast, final, get_origin

import numpy as np

from oxyz._frames import Frame
from oxyz._record import Record
from oxyz._record_batch import RecordBatch, batch_class
from oxyz._schema import KIND_TO_LETTER, Kind, column_sig, metadata_sig
from oxyz._schema_match import SchemaError
from oxyz._schema_spec import SchemaSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    import numpy.typing as npt

    from oxyz._batch import Batch
    from oxyz._record import FieldSpec, Fill
    from oxyz._record_fields import FieldType
    from oxyz._rust import ProjectionPlan
    from oxyz._schema_match import Conformance

type _Fields = tuple[tuple[str, FieldType, FieldSpec[Any]], ...]


@final
@dataclass(frozen=True, eq=False)
class Binding:
    """A record or batch class bound to a read: its plans, its spec and its casts.

    `target` is the class a read builds: `record` for a frame read, a
    `RecordBatch` class of `record` for a batch read.
    """

    target: type
    record: type[Record]
    spec: SchemaSpec
    frame_plan: ProjectionPlan
    fields: _Fields
    reported: Mapping[str, int]
    optional: tuple[tuple[str, int], ...]
    _build: Callable[..., Record] = dataclasses.field(repr=False, compare=False)

    @property
    def batch_plan(self) -> ProjectionPlan:
        """The plan a batch read projects with: `X | None` fields filled, not omitted.

        Raises `SchemaError` for an `X | None` field of a kind other than
        REAL declared without a `fill`.
        """
        return _plan(self.record, self.fields, promote=self.frame_plan[2], batch=True)

    @functools.cached_property
    def _batched(self) -> tuple[tuple[str, FieldType, str], ...]:
        """Each field's name, batch-form field type and file key."""
        return tuple(
            (name, field.batched(), _key(spec)) for name, field, spec in self.fields
        )

    def caster(self, conformance: Conformance) -> Callable[[Frame, int], Record]:  # noqa: ARG002 — a single record's cast never fails
        """Return the hook a frame read applies to each kept frame and its index."""
        return self._build

    def cast_frame(self, frame: Frame, *, owned: bool) -> Record:
        """Build a record from a frame projected with `frame_plan`.

        Unless `owned`, arrays come back read-only.
        """
        return self._build(frame, owned=owned)

    def cast_batch(
        self, batch: Batch, present: npt.NDArray[np.bool_], *, owned: bool
    ) -> RecordBatch[Any]:
        """Build a `target` batch from a batch projected with `batch_plan`.

        `present` is the core's presence report for the batch's frames, one
        column per optional plan entry. Unless `owned`, arrays come back
        read-only.
        """
        target = cast("type[RecordBatch[Any]]", self.target)
        columns, metadata = batch.columns, batch.metadata
        values = {
            name: field.cast(
                (columns if field.axis == "column" else metadata)[key], owned=owned
            )
            for name, field, key in self._batched
        }
        return target._trusted(  # noqa: SLF001 — the reader is trusted
            present={name: present[:, column] for name, column in self.optional},
            offsets=batch.offsets,
            frame_indices=batch.frame_indices,
            **values,
        )

    def empty_batch(self) -> RecordBatch[Any]:
        """Return a `target` batch of no frames, its fields of their batch dtypes."""
        target = cast("type[RecordBatch[Any]]", self.target)
        return target._trusted(  # noqa: SLF001 — the reader is trusted
            present={name: np.zeros(0, dtype=np.bool_) for name, _ in self.optional},
            offsets=np.zeros(1, dtype=np.intp),
            frame_indices=np.zeros(0, dtype=np.intp),
            **{name: field.empty() for name, field, _ in self._batched},
        )

    def carried(
        self, key: str, present: npt.NDArray[np.bool_]
    ) -> npt.NDArray[np.bool_]:
        """Return which projected frames carried metadata `key` in the file.

        `present` is the core's presence report: a row per frame, or one
        frame's row alone.
        """
        for name, field, spec in self.fields:
            if field.axis == "metadata" and spec.key == key:
                if name in self.reported:
                    return present[..., self.reported[name]]
                return np.ones(present.shape[:-1], dtype=np.bool_)
        return np.zeros(present.shape[:-1], dtype=np.bool_)

    def rule_frame(self, frame: Frame, present: Sequence[bool]) -> Frame:
        """Return `frame` as a frame rule sees it: without a filled `Lattice`."""
        if "Lattice" not in frame.metadata or self.carried(
            "Lattice", np.asarray(present, dtype=np.bool_)
        ):
            return frame
        return Frame(n_atoms=frame.n_atoms, columns={}, metadata={})


def bind(
    target: object,
    schema: SchemaSpec | str | Path | None = None,
    *,
    batch: bool | None = None,
) -> Binding:
    """Bind `target` for a read; `schema` describes the dataset.

    `target` is a `Record` subclass for a frame read, or a `RecordBatch`
    subclass or `RecordBatch[R]` for a batch read; `batch` says which the
    reader takes, and `None` accepts either. `RecordBatch[R]` binds as the
    class derived for it.

    Raises `TypeError` for any other `target`, naming the form to use, and
    `SchemaError` where `schema` disagrees with one of the record's fields,
    or its frame rule requires `Lattice` and no field reads that key.
    """
    if _names_batch(target):
        generic = get_origin(target) is RecordBatch
        target = batch_class(target)
        if batch is False:
            record = target.__record__.__name__
            named = f"RecordBatch[{record}]" if generic else target.__name__
            raise TypeError(
                f"record= on a frame read takes a Record subclass, not the batch "
                f"{named}; pass record={record}, or read batches with read_batch"
            )
    elif batch:
        raise TypeError(_not_a_batch(target))
    else:
        target = _record_class(target)
    if schema is not None and not isinstance(schema, SchemaSpec):
        schema = SchemaSpec.from_file(schema)
    if schema is not None and not _hashable(schema):
        # A spec built with lists reads like one built with tuples; it skips the cache.
        return _bind.__wrapped__(target, schema)
    return _bind(target, schema)


def _hashable(value: object) -> bool:
    try:
        hash(value)
    except TypeError:
        return False
    return True


def _names_batch(target: object) -> bool:
    if get_origin(target) is RecordBatch:
        return True
    return (
        isinstance(target, type)
        and issubclass(target, RecordBatch)
        and target is not RecordBatch
    )


def _not_a_batch(target: object) -> str:
    if isinstance(target, type) and issubclass(target, Record) and target is not Record:
        name = target.__name__
        return (
            f"record= on a batch read takes RecordBatch[{name}] or a subclass of "
            f"it, not the record {name}"
        )
    return (
        "record= on a batch read takes a RecordBatch subclass or RecordBatch[R], "
        f"got {target!r}"
    )


def _record_class(target: object) -> type[Record]:
    if not (isinstance(target, type) and issubclass(target, Record)):
        raise TypeError(f"record= takes a Record subclass, got {target!r}")
    if target is Record:
        raise TypeError(
            "record= takes a Record subclass, got Record itself, which declares "
            "no fields"
        )
    return target


@functools.lru_cache(maxsize=64)
def _bind(target: type, schema: SchemaSpec | None) -> Binding:
    record = cast("type[Record]", getattr(target, "__record__", target))
    fields = record.__record_fields__
    strict = record.__record_strict__
    spec = record.spec()
    if schema is not None:
        _check_agrees(record, fields, schema, strict=strict)
        _check_lattice_read(record, fields, schema)
        if schema.frame is not None:
            spec = dataclasses.replace(spec, frame=schema.frame)
    reported = _reported(fields)
    return Binding(
        target=target,
        record=record,
        spec=spec,
        frame_plan=_plan(record, fields, promote=not strict, batch=False),
        fields=fields,
        reported=reported,
        optional=tuple(
            (name, reported[name]) for name, field, _ in fields if field.optional
        ),
        _build=_builder(record, fields, strict=strict),
    )


def _reported(fields: _Fields) -> dict[str, int]:
    """Each field with a column in the core's presence report, and that column.

    The report has a column per optional plan entry, columns then metadata.
    """
    entries = [
        name
        for axis in ("column", "metadata")
        for name, field, spec in fields
        if field.axis == axis and (field.optional or spec.fill is not None)
    ]
    return {name: column for column, name in enumerate(entries)}


def _key(spec: FieldSpec[Any]) -> str:
    assert spec.key is not None  # noqa: S101 — type-narrowing only
    return spec.key


def _check_agrees(
    record: type[Record], fields: _Fields, schema: SchemaSpec, *, strict: bool
) -> None:
    """Raise `SchemaError` where `schema`'s rule for a field's key disagrees with it."""
    for name, field, spec in fields:
        key = _key(spec)
        if field.axis == "column":
            column = schema.rule_for(key, "column")
            if column is None:
                continue
            kind = column.kind
            found = column_sig(column.kind, column.width)
            expected = column_sig(field.kind, field.width)
            agrees = column.width == field.width
        else:
            metadata = schema.rule_for(key, "metadata")
            if metadata is None:
                continue
            kind, shape = metadata.kind, tuple(metadata.shape)
            found = metadata_sig(kind, shape)
            expected = metadata_sig(field.kind, field.shape)
            # A 2-D value is read as its flattened length, so its rule is too.
            agrees = shape == field.shape or (
                len(shape) > 1 and field.shape == (math.prod(shape),)
            )
        promotes = kind == Kind.INT and field.kind == Kind.REAL
        if agrees and (kind == field.kind or (promotes and not strict)):
            continue
        reason = "; the record is strict" if agrees and promotes else ""
        raise SchemaError(
            f"{record.__name__}.{name}: schema= gives {field.axis} {key!r} as "
            f"{found}, the record reads {expected}{reason}",
            name=key,
        )


def _check_lattice_read(
    record: type[Record], fields: _Fields, schema: SchemaSpec
) -> None:
    """Raise `SchemaError` if `schema` requires `Lattice` and no field reads it."""
    if schema.frame is None or not schema.frame.lattice_required:
        return
    if any(
        field.axis == "metadata" and spec.key == "Lattice" for _, field, spec in fields
    ):
        return
    raise SchemaError(
        f"{record.__name__}: schema= requires 'Lattice' in every frame, but no "
        "field reads it; declare a Cell field",
        name="Lattice",
    )


def _plan(
    record: type[Record], fields: _Fields, *, promote: bool, batch: bool
) -> ProjectionPlan:
    columns: list[tuple] = []
    metadata: list[tuple] = []
    for name, field, spec in fields:
        required, fill = _presence(record, name, field, spec, batch=batch)
        letter = KIND_TO_LETTER[field.kind]
        if field.axis == "column":
            columns.append((_key(spec), letter, field.width, required, fill))
        else:
            column_major = field.layout == "matrix3"
            metadata.append(
                (_key(spec), letter, field.shape, required, fill, column_major)
            )
    return columns, metadata, promote


def _presence(
    record: type[Record],
    name: str,
    field: FieldType,
    spec: FieldSpec[Any],
    *,
    batch: bool,
) -> tuple[bool, Fill | None]:
    """Return the plan entry's `(required, fill)`; a required entry has no fill."""
    if not field.optional:
        return spec.fill is None, spec.fill
    if not batch:
        return False, None
    if spec.fill is not None:
        return False, spec.fill
    if field.kind == Kind.REAL:
        return False, math.nan
    raise SchemaError(
        f"{record.__name__}.{name}: an optional {KIND_TO_LETTER[field.kind]} "
        f"field has no natural null; give it a fill= to read it into a batch",
        name=_key(spec),
    )


def _builder(
    record: type[Record], fields: _Fields, *, strict: bool
) -> Callable[..., Record]:
    """Return `build(frame, frame_index=None, *, owned=True)` for `record`.

    It closes over everything it needs, so building looks up no record
    class attribute.
    """
    from oxyz._record_fields import field_value

    new = object.__new__
    assign = object.__setattr__
    steps = tuple(
        (name, field, spec, field.axis == "column") for name, field, spec in fields
    )

    def build(
        frame: Frame,
        frame_index: int | None = None,  # noqa: ARG001 — the hook's signature; a single record's cast never fails
        *,
        owned: bool = True,
    ) -> Record:
        n_atoms = frame.n_atoms
        columns, metadata = frame.columns, frame.metadata
        out = new(record)
        assign(out, "n_atoms", n_atoms)
        for name, field, spec, is_column in steps:
            value = field_value(
                field,
                spec,
                columns if is_column else metadata,
                rows=n_atoms,
                strict=strict,
                owned=owned,
                checked=False,
            )
            assign(out, name, value)
        return out

    return build
