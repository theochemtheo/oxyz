"""Binding a record to a read: its projection plans and its casts."""

from __future__ import annotations

import dataclasses
import functools
import math
import types
import warnings
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    TypeAliasType,
    Union,
    cast,
    final,
    get_origin,
)

import numpy as np

from oxyz._frames import Frame
from oxyz._project import mismatched
from oxyz._record import Record, spec_of
from oxyz._record_batch import RecordBatch, batch_class
from oxyz._record_union import members_of, union_plan
from oxyz._schema import KIND_TO_LETTER, Kind, column_sig, metadata_sig
from oxyz._schema_match import SchemaError, SchemaWarning
from oxyz._schema_spec import SchemaSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    import numpy.typing as npt

    from oxyz._batch import Batch
    from oxyz._record import FieldSpec, Fill
    from oxyz._record_fields import FieldType
    from oxyz._record_union import Key, UnionPlan
    from oxyz._rust import DeviationData, ProjectionPlan
    from oxyz._schema_match import Conformance

type _Fields = tuple[tuple[str, FieldType, FieldSpec[Any]], ...]


@final
@dataclass(frozen=True, eq=False)
class Binding:
    """A record, union of records or batch class bound to a read.

    It holds the read's plans, its spec and its casts. `target` is the
    class a batch read builds, or the sole member of a frame read's union;
    `record` is that sole member, `None` for a union of several. `fields`
    are the union's merged fields, and `member_fields` each member's own.
    """

    target: type | None
    record: type[Record] | None
    union: UnionPlan
    name: str
    spec: SchemaSpec
    frame_plan: ProjectionPlan
    fields: _Fields
    member_fields: tuple[_Fields, ...]
    reported: Mapping[str, int]
    optional: tuple[tuple[str, int], ...]
    _builds: tuple[Callable[..., Record], ...] = dataclasses.field(
        repr=False, compare=False
    )

    @property
    def batch_plan(self) -> ProjectionPlan:
        """The plan a batch read projects with: optional fields filled, not omitted.

        Raises `SchemaError` for an optional field of a kind other than
        REAL declared without a `fill`.
        """
        return _plan(self.fields, self.union, promote=self.frame_plan[2], batch=True)

    @functools.cached_property
    def _batched(self) -> tuple[tuple[str, FieldType, str], ...]:
        """Each field's name, batch-form field type and file key."""
        return tuple(
            (name, field.batched(), _key(spec)) for name, field, spec in self.fields
        )

    def caster(self, conformance: Conformance) -> Callable[[Frame, int], Record | None]:
        """Return the hook a frame read applies to each kept frame and its index.

        For a union, the frame's matching keys pick the member. A frame no
        member admits raises `SchemaError` under `"strict"` and
        `"required"`; under `"warn"` it warns, and the hook returns `None`.
        """
        if len(self._builds) == 1:
            return self._builds[0]
        return _matcher(self, conformance)

    def cast_frame(self, frame: Frame, *, owned: bool) -> Record:
        """Build a record from a frame projected with `frame_plan`.

        Unless `owned`, arrays come back read-only. Raises `SchemaError` if
        no member of a union admits the frame.
        """
        if len(self._builds) == 1:
            return self._builds[0](frame, owned=owned)
        record = _matcher(self, "required")(frame, None, owned=owned)
        assert record is not None  # noqa: S101 — "required" raises rather than drop
        return record

    def cast_batch(
        self,
        batch: Batch,
        present: npt.NDArray[np.bool_],
        tags: npt.NDArray[np.intp],
        *,
        owned: bool,
    ) -> RecordBatch[Any]:
        """Build a `target` batch from a batch projected with `batch_plan`.

        `present` is the core's presence report for the batch's frames, one
        column per optional plan entry, and `tags` each frame's member, as
        `kept` gives them. Unless `owned`, arrays come back read-only.
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
            tags=tags,
            offsets=batch.offsets,
            frame_indices=batch.frame_indices,
            **values,
        )

    def kept(
        self,
        batch: Batch,
        presence: npt.NDArray[np.bool_],
        reports: list[tuple[int, list[DeviationData]]],
        survived: set[int],
        conformance: Conformance,
    ) -> tuple[Batch, npt.NDArray[np.bool_], npt.NDArray[np.intp]] | None:
        """Return the frames a record read keeps, their presence and members.

        The result is the kept frames of `batch`, their rows of `presence`,
        and each one's member as an index into the union's members; `None`
        when none is kept. `presence` and `survived` are the core's report
        for `batch`, and `reports` its deviations per requested frame.

        Under `"warn"` the core keeps a frame whose field of the wrong kind
        it can fill; a record read drops it. A frame no member admits raises
        `SchemaError` under `"strict"` and `"required"`, and under `"warn"`
        warns and is dropped.
        """
        if conformance == "warn":
            keep = np.array(
                [
                    not mismatched(deviations)
                    for index, deviations in reports
                    if index in survived
                ],
                dtype=np.bool_,
            )
        else:
            keep = np.ones(batch.n_frames, dtype=np.bool_)
        if len(self._builds) == 1:
            tags = np.zeros(batch.n_frames, dtype=np.intp)
        else:
            tags = self._member_tags(batch, presence, keep, warns=conformance == "warn")
        if keep.all():
            return batch, presence, tags
        if not keep.any():
            return None
        picks = np.flatnonzero(keep)
        return batch[picks], presence[picks], tags[picks]

    def _member_tags(
        self,
        batch: Batch,
        presence: npt.NDArray[np.bool_],
        keep: npt.NDArray[np.bool_],
        *,
        warns: bool,
    ) -> npt.NDArray[np.intp]:
        """Return each frame's member, from its matching keys' presence.

        A kept frame no member admits is reported, in batch order, and
        under `warns` is cleared from `keep`.
        """
        union = self.union
        matching = union.matching
        columns = [self.reported[union.names[key]] for key in matching]
        rows, inverse = np.unique(presence[:, columns], axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        signatures = [
            frozenset(
                key for key, carried in zip(matching, row, strict=True) if carried
            )
            for row in rows
        ]
        found = [union.member_for(signature) for signature in signatures]
        admitted = np.array([member is not None for member in found], dtype=np.bool_)
        unadmitted = np.flatnonzero(keep & ~admitted[inverse])
        for position in unadmitted:
            signature = signatures[inverse[position]]
            _reject(union, signature, int(batch.frame_indices[position]), warns=warns)
        keep[unadmitted] = False
        members = np.array([0 if m is None else m for m in found], dtype=np.intp)
        return members[inverse]

    def empty_batch(self) -> RecordBatch[Any]:
        """Return a `target` batch of no frames, its fields of their batch dtypes."""
        target = cast("type[RecordBatch[Any]]", self.target)
        return target._trusted(  # noqa: SLF001 — the reader is trusted
            present={name: np.zeros(0, dtype=np.bool_) for name, _ in self.optional},
            tags=np.zeros(0, dtype=np.intp),
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

    `target` is a `Record` subclass or a union of sibling ones for a frame
    read, or a `RecordBatch` subclass or `RecordBatch[R]` for a batch read;
    `batch` says which the reader takes, and `None` accepts either.
    `RecordBatch[R]` binds as the class derived for it, and every spelling
    of one union binds as its members in order.

    Raises `TypeError` for any other `target`, naming the form to use, or
    for a union `union_plan` rejects, and `SchemaError` where `schema`
    disagrees with one of the fields, or its frame rule requires `Lattice`
    and no field reads that key.
    """
    if _names_batch(target):
        generic = get_origin(target) is RecordBatch
        target = batch_class(target)
        if batch is False:
            record = target.__batch_layout__.union.name
            named = f"RecordBatch[{record}]" if generic else target.__name__
            raise TypeError(
                f"record= on a frame read takes a Record subclass, not the batch "
                f"{named}; pass record={record}, or read batches with read_batch"
            )
    elif batch:
        raise TypeError(_not_a_batch(target))
    else:
        target = _members(target)
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
    try:
        members = members_of(target)
    except TypeError:
        return (
            "record= on a batch read takes a RecordBatch subclass or RecordBatch[R], "
            f"got {target!r}"
        )
    name = " | ".join(member.__name__ for member in members)
    kind = "record" if len(members) == 1 else "union"
    return (
        f"record= on a batch read takes RecordBatch[{name}] or a subclass of "
        f"it, not the {kind} {name}"
    )


def _members(target: object) -> tuple[type[Record], ...]:
    union = get_origin(target) in {Union, types.UnionType}
    if union or isinstance(target, TypeAliasType):
        return members_of(target)
    return (_record_class(target),)


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
def _bind(
    target: tuple[type[Record], ...] | type, schema: SchemaSpec | None
) -> Binding:
    if isinstance(target, tuple):
        members, built = target, target[0] if len(target) == 1 else None
        union = union_plan(members)
    else:
        union = cast("type[RecordBatch[Any]]", target).__batch_layout__.union
        members, built = union.members, target
    fields, name, strict = union.fields, union.name, union.strict
    spec = spec_of(fields, union)
    if schema is not None:
        _check_agrees(union, schema, strict=strict)
        _check_lattice_read(name, fields, schema)
        if schema.frame is not None:
            spec = dataclasses.replace(spec, frame=schema.frame)
    reported = _reported(fields, union)
    matching = frozenset(union.matching)
    by_name = {entry[0]: entry for entry in fields}
    member_fields = tuple(
        tuple(by_name[field] for field in names) for names in union.member_fields
    )
    return Binding(
        target=built,
        record=members[0] if len(members) == 1 else None,
        union=union,
        name=name,
        spec=spec,
        frame_plan=_plan(fields, union, promote=not strict, batch=False),
        fields=fields,
        member_fields=member_fields,
        reported=reported,
        optional=tuple(
            (field, reported[field])
            for field, _, field_spec in fields
            if (field_spec.axis, _key(field_spec)) in matching
        ),
        _builds=tuple(
            _builder(member, own, strict=strict)
            for member, own in zip(members, member_fields, strict=True)
        ),
    )


def _reported(fields: _Fields, union: UnionPlan) -> dict[str, int]:
    """Each field with a column in the core's presence report, and that column.

    The report has a column per optional plan entry, columns then metadata.
    """
    entries = [
        name
        for axis in ("column", "metadata")
        for name, field, spec in fields
        if field.axis == axis and (axis, _key(spec)) not in union.required
    ]
    return {name: column for column, name in enumerate(entries)}


def _key(spec: FieldSpec[Any]) -> str:
    assert spec.key is not None  # noqa: S101 — type-narrowing only
    return spec.key


def _field_name(union: UnionPlan, field: str) -> str:
    """Name `field` by the first member declaring it, then the union, if several."""
    if len(union.members) == 1:
        return f"{union.name}.{field}"
    owner = next(
        member
        for member, names in zip(union.members, union.member_fields, strict=True)
        if field in names
    )
    return f"{owner.__name__}.{field} (in {union.name})"


def _check_agrees(union: UnionPlan, schema: SchemaSpec, *, strict: bool) -> None:
    """Raise `SchemaError` where `schema`'s rule for a field's key disagrees with it."""
    for field_name, field, spec in union.fields:
        key = _key(spec)
        if field.axis == "column":
            column = schema.rule_for(key, "column")
            if column is None:
                continue
            kind = column.kind
            found = column_sig(column.kind, column.width)
            agrees = column.width == field.width
        else:
            metadata = schema.rule_for(key, "metadata")
            if metadata is None:
                continue
            kind, shape = metadata.kind, tuple(metadata.shape)
            found = metadata_sig(kind, shape)
            # A 2-D value is read as its flattened length, so its rule is too.
            agrees = shape == field.shape or (
                len(shape) > 1 and field.shape == (math.prod(shape),)
            )
        if agrees and field.takes(kind, strict=strict):
            continue
        note = field.strict_note(kind) if agrees else ""
        raise SchemaError(
            f"{_field_name(union, field_name)}: schema= gives {field.axis} {key!r} as "
            f"{found}, the record reads {field.signature()}{note}",
            name=key,
        )


def _check_lattice_read(name: str, fields: _Fields, schema: SchemaSpec) -> None:
    """Raise `SchemaError` if `schema` requires `Lattice` and no field reads it."""
    if schema.frame is None or not schema.frame.lattice_required:
        return
    if any(
        field.axis == "metadata" and spec.key == "Lattice" for _, field, spec in fields
    ):
        return
    raise SchemaError(
        f"{name}: schema= requires 'Lattice' in every frame, but no "
        "field reads it; declare a Cell field",
        name="Lattice",
    )


def _plan(
    fields: _Fields, union: UnionPlan, *, promote: bool, batch: bool
) -> ProjectionPlan:
    columns: list[tuple] = []
    metadata: list[tuple] = []
    for name, field, spec in fields:
        required, fill = _presence(union, name, field, spec, batch=batch)
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
    union: UnionPlan,
    name: str,
    field: FieldType,
    spec: FieldSpec[Any],
    *,
    batch: bool,
) -> tuple[bool, Fill | None]:
    """Return the plan entry's `(required, fill)`; a required entry has no fill.

    A matching key is left absent on a frame read, so that its presence
    picks the member, and filled on a batch read: with its `fill=`, or NaN
    for a float field without one.
    """
    key = (field.axis, _key(spec))
    if key in union.required:
        return True, None
    if key in union.core_filled:
        return False, spec.fill
    if not batch:
        return False, None
    fill = spec.fill
    if fill is None and field.kind == Kind.REAL:
        fill = math.nan
    if fill is not None:
        return False, fill
    reason = (
        field.no_null()
        if field.optional
        else f"a {KIND_TO_LETTER[field.kind]} field some member lacks has no "
        "natural null; give it a fill="
    )
    raise SchemaError(
        f"{_field_name(union, name)}: {reason} to read it into a batch",
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


def _matcher(
    binding: Binding, conformance: Conformance
) -> Callable[..., Record | None]:
    """Return `match(frame, frame_index, *, owned=True)` for a union's binding.

    It builds the member the frame's matching keys pick.
    """
    union, builds = binding.union, binding._builds  # noqa: SLF001 — the binding's own builders
    probes = tuple((key, key[0] == "column") for key in union.matching)
    member_for = union.member_for
    warns = conformance == "warn"

    def match(
        frame: Frame, frame_index: int | None, *, owned: bool = True
    ) -> Record | None:
        columns, metadata = frame.columns, frame.metadata
        signature = frozenset(
            key
            for key, is_column in probes
            if key[1] in (columns if is_column else metadata)
        )
        member = member_for(signature)
        if member is not None:
            return builds[member](frame, owned=owned)
        _reject(union, signature, frame_index, warns=warns)
        return None

    return match


def _reject(
    union: UnionPlan,
    signature: frozenset[Key],
    frame_index: int | None,
    *,
    warns: bool,
) -> None:
    """Report a frame no member admits: warn under `warns`, else raise."""
    keys = ", ".join(sorted(key for _, key in signature))
    where = "" if frame_index is None else f"frame {frame_index}: "
    message = (
        f"{where}no member of {union.name} admits keys {{{keys}}}; {union.describe()}"
    )
    if not warns:
        raise SchemaError(message, frame_index=frame_index)
    warnings.warn(message, SchemaWarning, stacklevel=3)
