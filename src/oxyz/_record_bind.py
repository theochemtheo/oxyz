"""Binding a record to a read: its projection plans and its per-frame cast."""

from __future__ import annotations

import dataclasses
import functools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from oxyz._record import Record
from oxyz._schema import KIND_TO_LETTER, Kind, column_sig, metadata_sig
from oxyz._schema_match import SchemaError
from oxyz._schema_spec import SchemaSpec

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from oxyz._frames import Frame
    from oxyz._record import FieldSpec, Fill
    from oxyz._record_fields import FieldType
    from oxyz._rust import ProjectionPlan
    from oxyz._schema_match import Conformance

type _Fields = tuple[tuple[str, FieldType, FieldSpec[Any]], ...]


@dataclass(frozen=True)
class Binding:
    """A record class bound to a read: its plans, its spec and its frame cast."""

    target: type
    record: type[Record]
    spec: SchemaSpec
    frame_plan: ProjectionPlan
    fields: _Fields
    optional: tuple[tuple[str, int], ...]
    omitted: frozenset[tuple[str, str]]
    _build: Callable[..., Record] = dataclasses.field(repr=False, compare=False)

    @property
    def batch_plan(self) -> ProjectionPlan:
        """The plan a batch read projects with: `X | None` fields filled, not omitted.

        Raises `SchemaError` for an `X | None` field of a kind other than
        REAL declared without a `fill`.
        """
        return _plan(self.record, self.fields, promote=self.frame_plan[2], batch=True)

    def caster(self, conformance: Conformance) -> Callable[[Frame, int], Record]:  # noqa: ARG002 — a single record's cast never fails
        """Return the hook a frame read applies to each kept frame and its index."""
        return self._build

    def cast_frame(self, frame: Frame, *, owned: bool) -> Record:
        """Build a record from a frame projected with `frame_plan`.

        Unless `owned`, arrays come back read-only.
        """
        return self._build(frame, owned=owned)


def bind(target: object, schema: SchemaSpec | str | Path | None = None) -> Binding:
    """Bind `target` for a read; `schema` describes the dataset.

    Raises `TypeError` unless `target` is a `Record` subclass, and
    `SchemaError` where `schema` disagrees with one of the record's fields.
    """
    if not (isinstance(target, type) and issubclass(target, Record)):
        raise TypeError(f"record= takes a Record subclass, got {target!r}")
    if schema is not None and not isinstance(schema, SchemaSpec):
        schema = SchemaSpec.from_file(schema)
    return _bind(target, schema)


@functools.lru_cache(maxsize=64)
def _bind(record: type[Record], schema: SchemaSpec | None) -> Binding:
    fields = record.__record_fields__
    strict = record.__record_strict__
    spec = record.spec()
    if schema is not None:
        _check_agrees(record, fields, schema, strict=strict)
        if schema.frame is not None:
            spec = dataclasses.replace(spec, frame=schema.frame)
    return Binding(
        target=record,
        record=record,
        spec=spec,
        frame_plan=_plan(record, fields, promote=not strict, batch=False),
        fields=fields,
        optional=_reported(fields),
        omitted=frozenset(
            (field.axis, _key(key)) for _, field, key in fields if field.optional
        ),
        _build=_builder(record, fields, strict=strict),
    )


def _reported(fields: _Fields) -> tuple[tuple[str, int], ...]:
    """Each `X | None` field and its column in the core's presence report.

    The report has a column per optional plan entry, columns then metadata.
    """
    entries = [
        (name, field.optional)
        for axis in ("column", "metadata")
        for name, field, spec in fields
        if field.axis == axis and (field.optional or spec.fill is not None)
    ]
    return tuple(
        (name, column) for column, (name, omits) in enumerate(entries) if omits
    )


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
            metadata.append((_key(spec), letter, field.shape, required, fill))
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
