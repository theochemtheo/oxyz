"""Typed record batches: `RecordBatch[R]`, generic or declared."""

from __future__ import annotations

import functools
import operator
import types
import typing
from dataclasses import InitVar, dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Self,
    cast,
    dataclass_transform,
    get_args,
    get_origin,
    overload,
)

import numpy as np
import numpy.typing as npt

from oxyz import _html, _summary
from oxyz._batch import (
    FrameLayout,
    frame_rows,
    gather_rows,
    position,
    positions,
    slice_rows,
)
from oxyz._record import (
    FieldSpec,
    Record,
    _namespace_annotations,
    resolve_hints,
    slotted_dataclass,
    spec_of,
)
from oxyz._record_union import members_of, union_plan

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from oxyz._batch import Batch
    from oxyz._record import Column, Meta
    from oxyz._record_fields import FieldType
    from oxyz._record_union import UnionPlan
    from oxyz._schema_spec import SchemaSpec

_BASE_FIELDS = ("offsets", "frame_indices", "_present", "_tags")


@dataclass(frozen=True, eq=False)
class _Layout:
    """What indexing a batch of a record or union needs, resolved once per class.

    `fields` are the union's merged fields in batch form; `member_fields`
    hold each member's own column and metadata field names. `optional`
    names the fields that carry a mask, `nullable` those of them that are
    `X | None`.
    """

    union: UnionPlan
    members: tuple[type[Record], ...]
    record: type[Record] | None
    constructs: tuple[Callable[..., Record], ...]
    member_fields: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]
    fields: dict[str, FieldType]
    columns: tuple[str, ...]
    metadata: tuple[str, ...]
    optional: tuple[str, ...]
    nullable: tuple[str, ...]
    keys: dict[str, tuple[FieldSpec[Any], ...]]

    def unbox(self, name: str, row: object) -> object:
        return self.fields[name].unbox(row)


def _layout(union: UnionPlan) -> _Layout:
    members = union.members
    fields = {name: ft.batched() for name, ft, _ in union.fields}
    keys: dict[str, tuple[FieldSpec[Any], ...]] = {}
    for member in members:
        for name, key in member.__record_keys__.items():
            found = keys.get(name, ())
            if not any(k is key for k in found):
                keys[name] = (*found, key)
    optional = tuple(union.names[key] for key in union.matching)
    return _Layout(
        union=union,
        members=members,
        record=members[0] if len(members) == 1 else None,
        constructs=tuple(member._trusted for member in members),  # noqa: SLF001 — each member's trusted constructor
        member_fields=tuple(
            (
                tuple(n for n in names if fields[n].axis == "column"),
                tuple(n for n in names if fields[n].axis == "metadata"),
            )
            for names in union.member_fields
        ),
        fields=fields,
        columns=tuple(name for name, ft in fields.items() if ft.axis == "column"),
        metadata=tuple(name for name, ft in fields.items() if ft.axis == "metadata"),
        optional=optional,
        nullable=tuple(name for name in optional if fields[name].optional),
        keys=keys,
    )


# `union_plan`'s cache would hold the members alive; a batch class keeps its own plan.
_analyse = union_plan.__wrapped__


def _members(name: str, form: object) -> tuple[type[Record], ...]:
    try:
        return members_of(form)
    except TypeError as error:
        raise TypeError(
            f"{name}: RecordBatch[R] takes a Record subclass R, or a union of "
            f"them, got {form!r}: {error}"
        ) from None


def _union_of(
    name: str, bases: tuple[type, ...], namespace: dict[str, Any]
) -> UnionPlan:
    for base in namespace.get("__orig_bases__", ()):
        if isinstance(get_origin(base), RecordBatchMeta):
            (form,) = get_args(base)
            return _analyse(_members(name, form))
    for base in bases:
        layout = getattr(base, "__batch_layout__", None)
        if layout is not None:
            return layout.union
    raise TypeError(
        f"{name}: subclass RecordBatch[R], naming the record R, not bare RecordBatch"
    )


def _owner(union: UnionPlan, field: str) -> str:
    """Return the name of the first member that declares `field`."""
    return next(
        member.__name__
        for member, names in zip(union.members, union.member_fields, strict=True)
        if field in names
    )


def _check_names(
    name: str,
    union: UnionPlan,
    bases: tuple[type, ...],
    annotations: dict[str, Any],
    namespace: dict[str, Any],
) -> None:
    for field in annotations:
        if field in namespace:
            default = namespace[field]
            reason = (
                "a field specifier; a batch field takes its key from the record"
                if isinstance(default, FieldSpec)
                else f"default {default!r}; batch fields take no default"
            )
            raise TypeError(f"{name}.{field}: {reason}")
    inherited = {
        field for base in bases for field in getattr(base, "__dataclass_fields__", {})
    }
    declared = (inherited | set(annotations)) - set(_BASE_FIELDS)
    wanted = [field for field, _, _ in union.fields]
    for field in wanted:
        if field not in declared:
            owner = _owner(union, field)
            raise TypeError(f"{name}.{field}: missing; {owner} declares it")
    for field in annotations:
        if field not in wanted:
            raise TypeError(f"{name}.{field}: not a field of {union.name}")


def _check_forms(cls: type, union: UnionPlan, annotations: dict[str, Any]) -> None:
    if not annotations:
        return
    hints = resolve_hints(cls, annotations)
    for field, ft, _ in union.fields:
        if field not in annotations:
            continue
        batched = ft.batched()
        if not batched.accepts(hints[field]):
            raise TypeError(
                f"{cls.__name__}.{field}: annotated {hints[field]!r}, but "
                f"{_owner(union, field)}.{field} batches to "
                f"{batched.annotation()!r}"
            )


def _taking_present(init: Callable[..., None]) -> Callable[..., None]:
    """Wrap a dataclass `__init__` so that `present=` and `tags=` give the slots.

    They win over `_present` and `_tags`, which `dataclasses.replace` forwards.
    """

    @functools.wraps(init)
    def wrapped(
        self: RecordBatch[Any],
        /,
        *,
        present: Mapping[str, npt.NDArray[np.bool_]] | None = None,
        tags: npt.NDArray[np.integer] | None = None,
        **values: object,
    ) -> None:
        if present is not None:
            values["_present"] = present
        if tags is not None:
            values["_tags"] = tags
        init(self, **values)

    return wrapped


@dataclass_transform(kw_only_default=True, frozen_default=True)
class RecordBatchMeta(type):
    """Metaclass of `RecordBatch`: each batch class is a frozen, slotted dataclass."""

    __record__: type[Record] | None
    __batch_layout__: _Layout
    members: tuple[type[Record], ...]

    def __new__(
        mcls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: object,
    ) -> RecordBatchMeta:
        namespace = dict(namespace)
        annotations = _namespace_annotations(namespace)
        if not any(isinstance(base, RecordBatchMeta) for base in bases):
            defaults = {slot: namespace.pop(slot) for slot in ("_present", "_tags")}
            cls = slotted_dataclass(
                mcls, name, bases, namespace, tuple(annotations), defaults, kwargs
            )
        else:
            union = _union_of(name, bases, namespace)
            _check_names(name, union, bases, annotations, namespace)
            cls = slotted_dataclass(
                mcls, name, bases, namespace, tuple(annotations), {}, kwargs
            )
            _check_forms(cls, union, annotations)
            layout = _layout(union)
            cls.__record__ = layout.record
            cls.members = layout.members
            cls.__batch_layout__ = layout
        cls.__init__ = _taking_present(cls.__init__)  # ty: ignore[invalid-assignment]
        return cls


class RecordBatch[R: Record](FrameLayout, metaclass=RecordBatchMeta):
    """Many records of one class or union, concatenated atom-major, CSR-style.

    A batch holds each field of its record `R` in batch form: a per-atom
    field concatenated over all atoms, a per-frame field with a leading
    frame axis, a string scalar as a list. Frame `i` occupies rows
    `offsets[i]:offsets[i + 1]` of every per-atom field. A batch never holds
    `None`: an `X | None` field is a dense array, filled where a frame did
    not carry the field, and a mask records which frames did.

    It has two forms with the same behaviour:

    - **Generic**, `RecordBatch[MaceSample]`: nothing is declared. Static
      checkers type `batch[i]` as `MaceSample` and `batch[MaceSample.energy]`
      by the field's batch form; field attributes are not part of this
      form's static interface.
    - **Declared**, `class MaceBatch(RecordBatch[MaceSample])`: the subclass
      declares every field of the record and no other, each annotated with
      its batch form (`NDArray[np.float64]` for a `Meta[float]`, `list[str]`
      for a `Meta[str]` or a width-1 `Column[list[str]]`, and so on), with
      no default. Fields are then typed attributes as well as keys.

    A batch is a sequence of records, indexed as numpy indexes:

    - `len(batch)` is `n_frames`, and iteration yields `R`.
    - `batch[i]` is an `R`: per-atom fields and array metadata are row
      views, scalar metadata is a Python value, and an `X | None` field the
      frame did not carry is `None`. Negative positions count from the end.
    - `batch[a:b]` is a batch of views, with `offsets` rebased to start at
      0. A step slice, a sequence of positions or a boolean mask gathers a
      new batch, a copy.
    - `batch[key]`, with a field key such as `MaceSample.energy`, is the
      field's batch value, the same object as the attribute. The key must
      be one of `R`'s keys, inherited fields' included; a key from another
      record raises `KeyError`. `batch["energy"]` looks a field up by name.

    `present(key)` gives the mask. Slices and gathers carry the masks with
    them, and so does `dataclasses.replace` unless it is given `present=`.

    `R` may be a union of sibling records, `RecordBatch[Molecule | Crystal]`
    or a `type` alias of one. The batch then holds the union's merged
    fields: a field some member lacks is filled, and masked, in that
    member's frames. `members` is the tuple of member classes, in order,
    and `tags` gives each frame's member as an index into it; `where(*ms)`
    marks the frames whose member is one of `ms`. `batch[i]` is an instance
    of `members[tags[i]]`, built from that member's fields alone. A batch of
    one record has that record as its sole member and all-zero `tags`.

    `oxyz.read_batch(path, record=MaceBatch)` and `oxyz.iread_batch` read a
    file into batches, `from_batch` builds one from an `oxyz.Batch`, and
    `spec` gives the `SchemaSpec` that describes the record's fields.

    Batches are frozen and slotted. They compare equal when they are of the
    same class and hold equal layout arrays, fields, masks and tags, NaN
    equal to NaN; they are unhashable.

    Parameters
    ----------
    offsets : numpy.ndarray
        Integer CSR row boundaries, length `n_frames + 1`, starting at 0
        and non-decreasing.
    frame_indices : numpy.ndarray
        Integer file frame index of each entry, length `n_frames`.
    **fields
        Every field of the record, in batch form. Arrays are kept as given:
        not copied, converted or made read-only.
    present : mapping of str to numpy.ndarray, optional
        Keyword only, keyed by field name: for an `X | None` field, or one
        some member of a union lacks, a bool mask of length `n_frames`
        marking the frames that carry it. A field left out is present in
        every frame. Type checkers do not see this keyword on the
        constructor, since `present` is also a method, and report it as an
        unknown argument; `dataclasses.replace(batch, present=...)` is the
        form they accept.
    tags : numpy.ndarray, optional
        Keyword only: an integer array of length `n_frames`, each frame's
        member as an index into `members`. All zeros, the first member, by
        default. Type checkers do not see it, as with `present`.

    Raises
    ------
    TypeError
        At class definition, naming the field: a missing or extra field, an
        annotation that is not the field's batch form, or a default. Also an
        unparameterised subclass, or a type argument that is not a `Record`
        subclass. From `__init__`, naming the field: a wrong type or dtype,
        or a mask for a field that carries none.
    ValueError
        From `__init__`, naming the field: offsets that do not start at 0
        or that decrease, a wrong row count, width or shape, or a tag that
        does not index a member.
    """

    offsets: npt.NDArray[np.int64]
    frame_indices: npt.NDArray[np.int64]

    __hash__ = None

    members: ClassVar[tuple[type[Record], ...]]

    # Always a dict once built; `None` only as the absent-argument default.
    _present: dict[str, npt.NDArray[np.bool_]] = None  # ty: ignore[invalid-assignment]
    # Init-only, so `dataclasses.fields` leaves it out while
    # `dataclasses.replace` still forwards the slot.
    _tags: InitVar[npt.NDArray[np.int64]] = None  # ty: ignore[invalid-assignment]

    if not TYPE_CHECKING:

        def __class_getitem__(cls, params: object) -> object:
            alias = super().__class_getitem__(params)
            if cls is not RecordBatch or _ALIAS is None:
                return alias
            # typing caches one alias for every order of a union's members; the
            # order given here is the members' order.
            union = get_origin(params) in {typing.Union, types.UnionType}
            return _ALIAS(alias.__origin__, (params,) if union else alias.__args__)

    def __post_init__(self, _tags: object) -> None:
        cls = type(self)
        name = cls.__name__
        layout = cls._record_layout()
        n_frames = _check_layout(name, self.offsets, self.frame_indices)
        total_atoms = int(self.offsets[-1])
        for field, ft in layout.fields.items():
            rows = total_atoms if ft.axis == "column" else n_frames
            try:
                ft.check(getattr(self, field), rows=rows)
            except (TypeError, ValueError) as error:
                raise type(error)(f"{name}.{field}: {error}") from None
        masks = dict(self._present or {})
        _check_masks(name, layout, masks, n_frames)
        tags = _check_tags(name, _tags, n_frames, len(layout.members))
        for field in layout.optional:
            masks.setdefault(field, np.ones(n_frames, dtype=np.bool_))
        object.__setattr__(self, "_present", masks)
        object.__setattr__(self, "_tags", tags)

    @classmethod
    def _trusted(
        cls,
        /,
        *,
        present: Mapping[str, npt.NDArray[np.bool_]],
        tags: npt.NDArray[np.integer],
        **values: object,
    ) -> Self:
        """Build a batch from values, masks and tags already known to fit, unchecked.

        `present` must hold a mask for every field that carries one.
        """
        batch = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(batch, name, value)
        object.__setattr__(batch, "_present", dict(present))
        object.__setattr__(batch, "_tags", tags)
        return batch

    @classmethod
    def _record_layout(cls) -> _Layout:
        layout = cast("_Layout | None", getattr(cls, "__batch_layout__", None))
        if layout is None:
            if _ALIAS is None:
                raise TypeError(
                    f"{cls.__name__} holds no record, and on this Python "
                    "RecordBatch[R] forwards to it; declare a batch class, "
                    "class SampleBatch(RecordBatch[R]), and use that"
                )
            raise TypeError(
                f"{cls.__name__} holds no record; use RecordBatch[R] for a Record "
                "subclass R, or declare a subclass of it"
            )
        return layout

    @classmethod
    def spec(cls) -> SchemaSpec:
        """Return the project-mode `SchemaSpec` of the batch's record, as `R.spec()`.

        Call it on `RecordBatch[R]` or a declared batch class. For a union,
        a key every member requires is required, and every other key is
        optional.
        """
        union = cls._record_layout().union
        return spec_of(union.fields, union)

    @classmethod
    def from_batch(cls, batch: Batch) -> Self:
        """Build a record batch from a `Batch`, checking it against the record's fields.

        Call it on `RecordBatch[R]`, which gives an instance of the class
        derived for `R`, or on a declared batch class. A plain `Batch`
        cannot say which member each frame is, so a union batch class
        refuses it: read a union with `read_batch(record=...)`.

        Each field's key is looked up in the batch's columns or metadata. An
        integer value is accepted for a float field, and converted, unless
        the record is `strict`. A field declared with `fill=` and absent
        from the batch takes the fill in every frame.

        A plain `Batch` records no presence, so an `X | None` field is
        present in every frame when the batch has its key, and absent from
        every frame when it does not: its value is then the fill (NaN for a
        float field declared without one) and `present` is all-false.

        Arrays in the result are read-only. Those taken unchanged from the
        batch, `offsets` and `frame_indices` included, are views of its
        arrays, so a later write to the batch shows through; call `.copy()`
        on a field for an independent array. String lists are copied.

        Raises
        ------
        oxyz.SchemaError
            A required key is absent, an optional key of a kind other than
            float is absent and has no `fill=`, or a value has the wrong
            kind, dtype, row count, width or shape. `name` is the key and
            `frame_index` is `None`.
        TypeError, ValueError
            The batch's `offsets` or `frame_indices` do not describe a batch,
            or, `TypeError`, the class is a union's.
        """
        from oxyz._record_fields import batch_value
        from oxyz._schema_match import SchemaError

        name = cls.__name__
        record = cls._record_layout().record
        if record is None:
            raise TypeError(
                f"{name} is a union batch; from_batch takes a single record's "
                "batch — read with read_batch(record=…)"
            )
        strict = record.__record_strict__
        n_frames = _check_layout(name, batch.offsets, batch.frame_indices)
        total_atoms = batch.total_atoms
        values: dict[str, object] = {}
        present: dict[str, npt.NDArray[np.bool_]] = {}
        for field, ft, key in record.__record_fields__:
            on_atoms = ft.axis == "column"
            try:
                values[field], found = batch_value(
                    ft,
                    key,
                    batch.columns if on_atoms else batch.metadata,
                    rows=total_atoms if on_atoms else n_frames,
                    strict=strict,
                )
            except SchemaError as error:
                raise SchemaError(f"{name}: {error}", name=error.name) from None
            if ft.optional:
                present[field] = np.full(n_frames, found, dtype=np.bool_)
        return cls._trusted(
            present=present,
            tags=np.zeros(n_frames, dtype=np.intp),
            offsets=_read_only(batch.offsets),
            frame_indices=_read_only(batch.frame_indices),
            **values,
        )

    def __iter__(self) -> Iterator[R]:
        """Yield each frame's record in turn, as `batch[i]` would."""
        for i in range(self.n_frames):
            yield self._row(i)

    # mypy and pyright report the overloads marked below as overlapping later
    # ones; invariance in the value type keeps each declared key to one row.
    @overload
    def __getitem__(self, key: Meta[bool | None]) -> npt.NDArray[np.bool_]: ...  # type: ignore[overload-overlap]  # pyright: ignore[reportOverlappingOverload]
    @overload
    def __getitem__(self, key: Meta[bool]) -> npt.NDArray[np.bool_]: ...  # type: ignore[overload-overlap]
    @overload
    def __getitem__(  # type: ignore[overload-overlap]
        self, key: FieldSpec[npt.NDArray[np.bool_] | None]
    ) -> npt.NDArray[np.bool_]: ...
    @overload
    def __getitem__(
        self, key: FieldSpec[npt.NDArray[np.bool_]]
    ) -> npt.NDArray[np.bool_]: ...
    @overload
    def __getitem__(self, key: Meta[int | None]) -> npt.NDArray[np.int64]: ...  # type: ignore[overload-overlap]
    @overload
    def __getitem__(self, key: Meta[int]) -> npt.NDArray[np.int64]: ...
    @overload
    def __getitem__(  # type: ignore[overload-overlap]
        self, key: FieldSpec[npt.NDArray[np.int64] | None]
    ) -> npt.NDArray[np.int64]: ...
    @overload
    def __getitem__(
        self, key: FieldSpec[npt.NDArray[np.int64]]
    ) -> npt.NDArray[np.int64]: ...
    @overload
    def __getitem__(self, key: Meta[float | None]) -> npt.NDArray[np.float64]: ...  # type: ignore[overload-overlap]
    @overload
    def __getitem__(self, key: Meta[float]) -> npt.NDArray[np.float64]: ...
    @overload
    def __getitem__(  # type: ignore[overload-overlap]
        self, key: FieldSpec[npt.NDArray[np.float64] | None]
    ) -> npt.NDArray[np.float64]: ...
    @overload
    def __getitem__(
        self, key: FieldSpec[npt.NDArray[np.float64]]
    ) -> npt.NDArray[np.float64]: ...
    @overload
    def __getitem__(self, key: Meta[str | None]) -> list[str]: ...  # type: ignore[overload-overlap]
    @overload
    def __getitem__(self, key: Meta[str]) -> list[str]: ...
    @overload
    def __getitem__(self, key: Column[list[str] | None]) -> list[str]: ...  # type: ignore[overload-overlap]  # pyright: ignore[reportOverlappingOverload]
    @overload
    def __getitem__(self, key: Column[list[str]]) -> list[str]: ...
    @overload
    def __getitem__(self, key: Meta[list[str] | None]) -> list[list[str]]: ...
    @overload
    def __getitem__(self, key: Meta[list[str]]) -> list[list[str]]: ...
    @overload
    def __getitem__(self, key: Column[list[list[str]] | None]) -> list[list[str]]: ...
    @overload
    def __getitem__(self, key: Column[list[list[str]]]) -> list[list[str]]: ...
    @overload
    def __getitem__(self, key: FieldSpec[Any]) -> Any: ...  # noqa: ANN401 — a value type outside the batch-form table
    @overload
    def __getitem__(self, key: str) -> Any: ...  # noqa: ANN401 — a name carries no type
    @overload
    def __getitem__(self, key: int | np.integer) -> R: ...
    @overload
    def __getitem__(self, key: slice | Sequence[int] | np.ndarray) -> Self: ...
    def __getitem__(self, key: object) -> object:
        """Look up a field by key or name, a record by position, or frames by selection.

        Negative positions count from the end; out of range is `IndexError`.
        A key or name that is not one of the record's fields is `KeyError`.
        """
        if isinstance(key, (FieldSpec, str)):
            return getattr(self, self._field(key))
        if isinstance(key, slice):
            start, stop, step = key.indices(self.n_frames)
            if step == 1:
                return self._slice(start, max(start, stop))
            return self._gather(np.arange(start, stop, step, dtype=np.intp))
        if isinstance(key, (int, np.integer)) and not isinstance(key, (bool, np.bool_)):
            return self._row(position(int(key), self.n_frames))
        if isinstance(key, (list, tuple, np.ndarray)):
            return self._gather(positions(key, self.n_frames))
        raise TypeError(
            "RecordBatch keys are field keys, names (str), frame positions (int), "
            f"slices, position sequences or boolean masks, not {type(key).__name__}"
        )

    def present(self, key: FieldSpec[Any] | str) -> npt.NDArray[np.bool_]:
        """Return which frames carried a field, a bool array of length `n_frames`.

        `key` is a field key or name. A field that is not `X | None` is
        present in every frame. The array is read-only: the batch's rows
        follow the mask, so to change it, pass `present=` to
        `dataclasses.replace`. A key or name that is not one of the
        record's fields is `KeyError`.
        """
        mask = self._present.get(self._field(key))
        if mask is None:
            mask = np.ones(self.n_frames, dtype=np.bool_)
        return _read_only(mask)

    @property
    def tags(self) -> npt.NDArray[np.int64]:
        """Each frame's member, as an index into `members`; a read-only array."""
        return _read_only(self._tags)

    def where(self, *members: type[Record]) -> npt.NDArray[np.bool_]:
        """Return which frames are an instance of one of `members`, a bool array.

        A frame matches its own member class only, not a base of it; no
        argument matches no frame. A class that is not one of the batch's
        `members` is `TypeError`.
        """
        own = self.members
        for member in members:
            if member not in own:
                names = ", ".join(m.__name__ for m in own)
                raise TypeError(
                    f"{getattr(member, '__name__', member)!s} is not a member of "
                    f"{type(self).__name__}; its members are {names}"
                )
        return np.isin(self._tags, [own.index(member) for member in members])

    def _field(self, key: FieldSpec[Any] | str) -> str:
        layout = type(self).__batch_layout__
        record = layout.union.name
        if isinstance(key, str):
            if key in layout.fields:
                return key
            raise KeyError(f"{key!r} is not a field of {record}")
        if key.name is not None and any(
            found is key for found in layout.keys.get(key.name, ())
        ):
            return key.name
        if key.owner is None:
            raise KeyError(f"an unbound specifier is not a field of {record}")
        raise KeyError(f"{key.owner.__name__}.{key.name} is not a field of {record}")

    def _sides(self, layout: _Layout) -> tuple[dict[str, Any], dict[str, Any]]:
        return (
            {name: getattr(self, name) for name in layout.columns},
            {name: getattr(self, name) for name in layout.metadata},
        )

    def _row(self, i: int) -> R:
        layout = type(self).__batch_layout__
        tag = int(self._tags[i])
        columns, metadata = layout.member_fields[tag]
        n_atoms, columns, metadata = frame_rows(
            {name: getattr(self, name) for name in columns},
            {name: getattr(self, name) for name in metadata},
            self.offsets,
            i,
            layout.unbox,
        )
        values = columns | metadata
        present = self._present
        for name in layout.nullable:
            if name in values and not present[name][i]:
                values[name] = None
        return cast("R", layout.constructs[tag](n_atoms=n_atoms, **values))

    def _slice(self, start: int, stop: int) -> Self:
        layout = type(self).__batch_layout__
        columns, metadata, offsets, frame_indices = slice_rows(
            *self._sides(layout), self.offsets, self.frame_indices, start, stop
        )
        return self._trusted(
            present={name: mask[start:stop] for name, mask in self._present.items()},
            tags=self._tags[start:stop],
            offsets=offsets,
            frame_indices=frame_indices,
            **columns,
            **metadata,
        )

    def _gather(self, picks: np.ndarray) -> Self:
        layout = type(self).__batch_layout__
        columns, metadata, offsets, frame_indices = gather_rows(
            *self._sides(layout), self.offsets, self.frame_indices, picks
        )
        return self._trusted(
            present={name: mask[picks] for name, mask in self._present.items()},
            tags=self._tags[picks],
            offsets=offsets,
            frame_indices=frame_indices,
            **columns,
            **metadata,
        )

    def __eq__(self, other: object) -> bool:
        """Compare class, layout arrays, fields, masks and tags, NaN equal to NaN."""
        if not isinstance(other, RecordBatch):
            return NotImplemented
        cls = type(self)
        if type(other) is not cls:
            return False
        layout = cls.__batch_layout__
        equal = _summary.values_equal
        return (
            equal(self.offsets, other.offsets)
            and equal(self.frame_indices, other.frame_indices)
            and equal(self._tags, other._tags)
            and all(
                field.equal(getattr(self, name), getattr(other, name))
                for name, field in layout.fields.items()
            )
            and all(equal(self.present(n), other.present(n)) for n in layout.optional)
        )

    def __repr__(self) -> str:
        """Summarise the layout and each field by dtype and shape, never by data."""
        cls = type(self)
        fields = list(cls.__batch_layout__.fields.items())
        limit = _summary._MAX_ENTRIES  # noqa: SLF001 — shared repr limit
        atoms = _summary.atom_range(self.n_atoms)
        shown = [f"n_frames={self.n_frames}, total_atoms={self.total_atoms}{atoms}"]
        shown += [
            f"{name}={field.summarise(getattr(self, name))}"
            for name, field in fields[:limit]
        ]
        if len(fields) > limit:
            shown.append(f"+{len(fields) - limit} more")
        return f"{cls.__name__}({', '.join(shown)})"

    def _repr_html_(self) -> str:
        """Render the fields as a table for a notebook."""
        return _html.record_batch(self)

    def __reduce__(self) -> tuple[Callable[..., RecordBatch[Any]], tuple[object, ...]]:
        cls = type(self)
        layout = cls.__batch_layout__
        members = layout.members
        target: object = cls
        if _derived(members) is cls:
            target = RecordBatch[functools.reduce(operator.or_, members)]  # ty: ignore[invalid-type-form]
        state: dict[str, object] = {
            "offsets": self.offsets,
            "frame_indices": self.frame_indices,
            "present": dict(self._present),
            "tags": self._tags,
        }
        state |= {name: getattr(self, name) for name in layout.fields}
        return _rebuild, (target, state)


def _check_layout(name: str, offsets: object, frame_indices: object) -> int:
    """Raise unless `offsets` and `frame_indices` describe a batch; its `n_frames`."""
    if not isinstance(offsets, np.ndarray) or offsets.dtype.kind not in "iu":
        raise TypeError(f"{name}.offsets: expected an integer array")
    if offsets.ndim != 1 or offsets.size == 0:
        raise ValueError(
            f"{name}.offsets: expected shape (n_frames + 1,), got {offsets.shape}"
        )
    if offsets[0] != 0:
        raise ValueError(f"{name}.offsets: must start at 0, got {offsets[0]}")
    if (offsets[1:] < offsets[:-1]).any():
        raise ValueError(f"{name}.offsets: is decreasing")
    n_frames = offsets.size - 1
    if (
        not isinstance(frame_indices, np.ndarray)
        or frame_indices.dtype.kind not in "iu"
    ):
        raise TypeError(f"{name}.frame_indices: expected an integer array")
    if frame_indices.shape != (n_frames,):
        raise ValueError(
            f"{name}.frame_indices: expected shape ({n_frames},), "
            f"got {frame_indices.shape}"
        )
    return n_frames


def _alias_class() -> type | None:
    """Return the class of `RecordBatch[R]`, or `None` where `typing` refuses it.

    `None` leaves the plain alias, which forwards `from_batch` and `spec`
    to the bare `RecordBatch`.
    """
    try:

        class _RecordBatchAlias(typing._GenericAlias, _root=True):  # noqa: SLF001  # ty: ignore[missing-attribute] — the alias type `Generic` builds
            """`RecordBatch[R]`, which calls through to its batch class."""

            def __call__(self, *args: Any, **kwargs: Any) -> RecordBatch[Any]:  # noqa: ANN401 — forwarded to the batch class as given
                return batch_class(self)(*args, **kwargs)

            def from_batch(self, batch: Batch) -> RecordBatch[Any]:
                return batch_class(self).from_batch(batch)

            def spec(self) -> SchemaSpec:
                return batch_class(self).spec()

    except Exception:  # noqa: BLE001 — a private base that may change; the plain alias still reads
        return None
    return _RecordBatchAlias


_ALIAS = _alias_class()


def _read_only(array: np.ndarray) -> np.ndarray:
    view = array.view()
    view.flags.writeable = False
    return view


def _check_masks(
    name: str, layout: _Layout, present: Mapping[str, object], n_frames: int
) -> None:
    for field, mask in present.items():
        where = f"{name}.present[{field!r}]"
        if field not in layout.optional:
            raise TypeError(f"{where}: not an optional field of {layout.union.name}")
        if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
            raise TypeError(f"{where}: expected a bool array")
        if mask.shape != (n_frames,):
            raise ValueError(f"{where}: expected shape ({n_frames},), got {mask.shape}")


def _check_tags(
    name: str, tags: object, n_frames: int, n_members: int
) -> npt.NDArray[np.integer]:
    """Return `tags`, checked to index the members, or all zeros for `None`."""
    if tags is None:
        return np.zeros(n_frames, dtype=np.intp)
    if not isinstance(tags, np.ndarray) or tags.dtype.kind not in "iu":
        raise TypeError(f"{name}.tags: expected an integer array")
    if tags.shape != (n_frames,):
        raise ValueError(f"{name}.tags: expected shape ({n_frames},), got {tags.shape}")
    outside = tags[(tags < 0) | (tags >= n_members)]
    if outside.size:
        raise ValueError(
            f"{name}.tags: {outside[0]} does not index one of {n_members} members"
        )
    return tags


def _rebuild(target: object, state: dict[str, Any]) -> RecordBatch[Any]:
    return batch_class(target)._trusted(**state)  # noqa: SLF001 — same module


def batch_class(target: object) -> type[RecordBatch[Any]]:
    """Return the batch class `target` names.

    A declared batch class is returned as is. `RecordBatch[R]` gives the
    class derived for `R`, created on first use and the same object after;
    every spelling of one union, in one member order, gives one class.
    Raises `TypeError` for anything else, or for a union `union_plan`
    rejects.
    """
    if (
        isinstance(target, type)
        and issubclass(target, RecordBatch)
        and target is not RecordBatch
    ):
        return target
    if get_origin(target) is RecordBatch:
        (form,) = get_args(target)
        members = _members("RecordBatch[R]", form)
        derived = _derived(members)
        if derived is None:
            derived = _derive(members)
            first = members[0]
            if len(members) == 1:
                type.__setattr__(first, _DERIVED, derived)
            else:
                unions = vars(first).get(_DERIVED_UNIONS)
                if unions is None:
                    unions = {}
                    type.__setattr__(first, _DERIVED_UNIONS, unions)
                unions[members] = derived
        return derived
    raise TypeError(
        f"expected a RecordBatch subclass or RecordBatch[R], got {target!r}"
    )


# Kept on the record class, so a derived class lives exactly as long as its
# record; a union's is kept on its first member, keyed by the members.
_DERIVED = "__record_batch__"
_DERIVED_UNIONS = "__record_batches__"


def _derived(members: tuple[type[Record], ...]) -> type[RecordBatch[Any]] | None:
    """Return the class derived for `members` themselves, not one inherited."""
    first = members[0]
    if len(members) == 1:
        return vars(first).get(_DERIVED)
    return vars(first).get(_DERIVED_UNIONS, {}).get(members)


def _derive(members: tuple[type[Record], ...]) -> type[RecordBatch[Any]]:
    name = "Or".join(member.__name__ for member in members) + "Batch"
    form = functools.reduce(operator.or_, members)

    annotations = {
        field: ft.batched().annotation() for field, ft, _ in _analyse(members).fields
    }

    def body(namespace: dict[str, Any]) -> None:
        namespace["__annotations__"] = annotations
        namespace["__module__"] = members[0].__module__
        namespace["__qualname__"] = name

    return cast(
        "type[RecordBatch[Any]]",
        types.new_class(name, (RecordBatch[form],), exec_body=body),  # ty: ignore[invalid-type-form]
    )
