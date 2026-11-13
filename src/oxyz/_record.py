"""Typed records: the field specifiers and the `Record` base class."""

from __future__ import annotations

import copy
import dataclasses
import sys
import typing
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Self,
    dataclass_transform,
    overload,
)

import numpy as np

from oxyz import _html, _summary

if TYPE_CHECKING:
    from collections.abc import Mapping

    from oxyz._frames import Frame
    from oxyz._record_fields import FieldType
    from oxyz._schema_spec import Absent, ColumnRule, MetadataRule, SchemaSpec

type Fill = float | int | bool | str

_RESERVED = frozenset(
    {
        "n_atoms",
        "offsets",
        "frame_indices",
        "n_frames",
        "total_atoms",
        "ptr",
        "batch",
        "present",
        "members",
        "tags",
        "where",
        "spec",
        "from_frame",
        "from_batch",
    }
)


class FieldSpec[T]:
    """Base of the field specifiers `Column`, `Meta`, `Cell` and `Matrix3`.

    `T` is the field's value type, taken from the annotation the specifier is
    assigned to. Two specifiers are equal only when they are the same object.

    Attributes
    ----------
    key : str or None
        The file key the field reads. `None` until the field is bound, when
        it defaults to the field name.
    width : int
        Values per atom; 1 for a per-frame field.
    shape : tuple of int
        Shape of a per-frame value; `()` for a per-atom field or a scalar.
    fill : float, int, bool, str or None
        Value used where the key is absent from a frame.
        For an `X | None` field, used only in batches; an individual record
        gets `None`.
    name : str or None
        The field name, once bound.
    owner : type or None
        The class that declares the field, once bound.
    """

    __slots__ = ("fill", "key", "name", "owner", "shape", "width")

    axis: ClassVar[Literal["column", "metadata"]]
    layout: ClassVar[Literal["plain", "cell", "matrix3"]] = "plain"

    key: str | None
    width: int
    shape: tuple[int, ...]
    fill: Fill | None
    name: str | None
    owner: type | None

    def __init__(
        self,
        *,
        key: str | None,
        width: int = 1,
        shape: tuple[int, ...] = (),
        fill: Fill | None,
    ) -> None:
        self.key = key
        self.width = width
        self.shape = shape
        self.fill = fill
        self.name = None
        self.owner = None

    if TYPE_CHECKING:

        @overload
        def __get__(self, instance: None, owner: type, /) -> Self: ...
        @overload
        def __get__(self, instance: object, owner: type, /) -> T: ...
        def __get__(self, instance: object, owner: type, /) -> Self | T: ...
        def __set__(self, instance: object, value: T, /) -> None: ...

    def _bound(self, name: str, owner: type) -> Self:
        bound = copy.copy(self)
        if bound.key is None:
            bound.key = name
        bound.name = name
        bound.owner = owner
        return bound


class Column[T](FieldSpec[T]):
    """Declare a per-atom field, read from one column.

    The annotation is `Column[V]`, where `V` is `NDArray[np.float64]`,
    `NDArray[np.int64]`, `NDArray[np.bool_]`, `list[str]` (width 1) or
    `list[list[str]]` (width above 1), optionally `V | None`.

    Parameters
    ----------
    key : str, optional
        Column name in the file. Defaults to the field name.
    width : int, default 1
        Values per atom. Above 1, an array value has shape `(n_atoms, width)`.
    fill : float, int, bool or str, optional
        Value used where the column is absent. Must match the field's kind.
        For an `X | None` field, used only in batches; an individual record
        gets `None`.
    """

    __slots__ = ()
    axis = "column"

    def __init__(
        self, key: str | None = None, *, width: int = 1, fill: Fill | None = None
    ) -> None:
        super().__init__(key=key, width=width, fill=fill)


class Meta[T](FieldSpec[T]):
    """Declare a per-frame field, read from one comment-line entry.

    The annotation is `Meta[V]`: `float`, `int`, `bool` or `str` with the
    default `shape=()`; `NDArray[np.float64]`, `NDArray[np.int64]`,
    `NDArray[np.bool_]` or `list[str]` with `shape=(k,)`. Either may be
    `V | None`.

    Parameters
    ----------
    key : str, optional
        Entry key in the comment line. Defaults to the field name.
    shape : tuple of int, default ()
        `()` for a scalar, `(k,)` for an array of `k` values.
    fill : float, int, bool or str, optional
        Value used where the entry is absent. Must match the field's kind.
        For an `X | None` field, used only in batches; an individual record
        gets `None`.
    """

    __slots__ = ()
    axis = "metadata"

    def __init__(
        self,
        key: str | None = None,
        *,
        shape: tuple[int, ...] = (),
        fill: Fill | None = None,
    ) -> None:
        super().__init__(key=key, shape=shape, fill=fill)


class Cell[T](FieldSpec[T]):
    """Declare the cell: the 3x3 matrix in `Lattice`, one lattice vector per row.

    The annotation is `Cell[NDArray[np.float64]]`, optionally `| None`. The
    nine values are read in C order, as ASE reads `Lattice`; `Matrix3` reads
    other 9-value entries in Fortran order.

    Parameters
    ----------
    fill : float, optional
        Value given to all nine entries where `Lattice` is absent; `0.0`
        matches `Frame.cell`.
        For an optional field, used only in batches; an individual record
        gets `None`.
    """

    __slots__ = ()
    axis = "metadata"
    layout = "cell"

    def __init__(self, *, fill: float | None = None) -> None:
        super().__init__(key="Lattice", shape=(9,), fill=fill)


class Matrix3[T](FieldSpec[T]):
    """Declare a per-frame 3x3 matrix, read from a 9-value entry.

    The annotation is `Matrix3[NDArray[np.float64]]`, optionally `| None`.
    Nine flat values are read in Fortran order, as ASE reads `stress` and
    `virial`; `Cell` reads `Lattice` in C order. A value written as a nested
    3x3 is read as written. `from_batch` cannot tell the two apart, as a
    `Batch` flattens a nested 3x3 row by row, and reads it as nine flat
    values.

    Parameters
    ----------
    key : str, optional
        Entry key in the comment line. Defaults to the field name.
    fill : float, optional
        Value given to all nine entries where the entry is absent.
        For an optional field, used only in batches; an individual record
        gets `None`.
    """

    __slots__ = ()
    axis = "metadata"
    layout = "matrix3"

    def __init__(self, key: str | None = None, *, fill: float | None = None) -> None:
        super().__init__(key=key, shape=(9,), fill=fill)


def _namespace_annotations(namespace: dict[str, Any]) -> dict[str, Any]:
    """Return the class body's annotations, unresolved names as forward references."""
    if "__annotations__" in namespace:
        return dict(namespace["__annotations__"])
    if sys.version_info >= (3, 14):
        import annotationlib

        annotate = annotationlib.get_annotate_from_class_namespace(namespace)
        if annotate is not None:
            return annotationlib.call_annotate_function(
                annotate, annotationlib.Format.FORWARDREF
            )
    return {}


def slotted_dataclass[M: type](
    mcls: type[M],
    name: str,
    bases: tuple[type, ...],
    namespace: dict[str, Any],
    slots: tuple[str, ...],
    defaults: Mapping[str, object],
    keywords: Mapping[str, Any],
) -> M:
    """Create a frozen, keyword-only dataclass of `mcls` whose fields are slots.

    `namespace` must hold no class attribute named in `slots`; a field's
    default is given in `defaults` instead. `keywords` are the class keywords.
    """
    namespace = {**namespace, "__slots__": slots}
    cls = type.__new__(mcls, name, bases, namespace, **keywords)
    descriptors = {field: cls.__dict__[field] for field in defaults}
    # `dataclass()` reads a default from the class attribute, where the slot is.
    for field, default in defaults.items():
        setattr(cls, field, default)
    dataclasses.dataclass(frozen=True, kw_only=True, eq=False, repr=False)(cls)
    for field, descriptor in descriptors.items():
        setattr(cls, field, descriptor)
    return cls


def _check_n_atoms(name: str, n_atoms: object) -> None:
    if isinstance(n_atoms, (bool, np.bool_)) or not isinstance(
        n_atoms, (int, np.integer)
    ):
        raise TypeError(f"{name}.n_atoms: expected int, got {type(n_atoms).__name__}")
    if n_atoms < 0:
        raise ValueError(f"{name}.n_atoms: is negative ({n_atoms})")


def _check_names(
    name: str, annotations: dict[str, Any], namespace: dict[str, Any]
) -> None:
    for field in annotations:
        if field in _RESERVED:
            raise TypeError(f"{name}.{field}: the name is reserved")
        if field.startswith("_"):
            raise TypeError(
                f"{name}.{field}: a field name cannot start with an underscore"
            )
        default = namespace.get(field)
        if field in namespace and not isinstance(default, FieldSpec):
            raise TypeError(
                f"{name}.{field}: default {default!r} is not a field specifier"
            )
    for attribute, value in namespace.items():
        if isinstance(value, FieldSpec) and attribute not in annotations:
            kind = type(value).__name__
            raise TypeError(
                f"{name}.{attribute}: a {kind} with no annotation is not a field; "
                f"annotate it as {attribute}: {kind}[...]"
            )


def _bind_fields(
    cls: RecordMeta,
    annotations: dict[str, Any],
    defaults: dict[str, FieldSpec[Any]],
) -> tuple[tuple[str, FieldType, FieldSpec[Any]], ...]:
    from oxyz._record_fields import field_type, unwrap
    from oxyz._schema_match import _is_pattern

    hints = resolve_hints(cls, annotations)
    fields: dict[str, tuple[FieldType, FieldSpec[Any]]] = {}
    for base in reversed(cls.__mro__[1:]):
        if isinstance(base, RecordMeta):
            fields.update(
                (field, (ft, key)) for field, ft, key in base.__record_fields__
            )
    for field in annotations:
        bare, metadata = unwrap(hints[field])
        spec = _specifier(cls, field, defaults.get(field), metadata)
        try:
            ft = field_type(bare, spec)
        except TypeError as error:
            raise TypeError(f"{cls.__name__}.{field}: {error}") from None
        key = spec._bound(field, cls)  # noqa: SLF001 — same module
        if key.key is not None and _is_pattern(key.key):
            raise TypeError(
                f"{cls.__name__}.{field}: key {key.key!r} would be read as a "
                f"pattern, not a literal name"
            )
        fields[field] = (ft, key)
    claimed: dict[tuple[str, str | None], str] = {}
    for field, (_, key) in fields.items():
        other = claimed.setdefault((key.axis, key.key), field)
        if other != field:
            raise TypeError(
                f"{cls.__name__}.{field}: key {key.key!r} is also read by {other}"
            )
    return tuple((field, ft, key) for field, (ft, key) in fields.items())


def resolve_hints(cls: type, annotations: dict[str, Any]) -> dict[str, Any]:
    """Resolve `cls`'s annotations; a `TypeError` names the first that cannot."""
    try:
        return typing.get_type_hints(cls, include_extras=True)
    except (NameError, TypeError) as error:
        for field, annotation in annotations.items():
            if not _resolves(cls, annotation):
                raise TypeError(f"{cls.__name__}.{field}: {error}") from error
        raise TypeError(f"{cls.__name__}: {error}") from error


def _resolves(cls: type, annotation: object) -> bool:
    """Whether one annotation resolves where the class's annotations do."""
    probe = type(
        cls.__name__,
        (),
        {"__module__": cls.__module__, "__annotations__": {"probe": annotation}},
    )
    try:
        typing.get_type_hints(probe, localns=dict(vars(cls)))
    except (NameError, TypeError):
        return False
    return True


def _specifier(
    cls: RecordMeta,
    field: str,
    default: FieldSpec[Any] | None,
    metadata: tuple[object, ...],
) -> FieldSpec[Any]:
    annotated = [item for item in metadata if isinstance(item, FieldSpec)]
    if default is not None and annotated:
        reason = "a specifier both as the default and in Annotated"
    elif len(annotated) > 1:
        reason = "more than one specifier in Annotated"
    elif default is None and not annotated:
        reason = "no field specifier; give one as the default or in Annotated"
    else:
        return default if default is not None else annotated[0]
    raise TypeError(f"{cls.__name__}.{field}: {reason}")


@dataclass_transform(
    kw_only_default=True,
    frozen_default=True,
    field_specifiers=(Column, Meta, Cell, Matrix3),
)
class RecordMeta(type):
    """Metaclass of `Record`: each record class is a frozen, slotted dataclass."""

    __record_fields__: tuple[tuple[str, FieldType, FieldSpec[Any]], ...]
    __record_keys__: dict[str, FieldSpec[Any]]
    __record_strict__: bool

    def __new__(
        mcls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        *,
        strict: bool | None = None,
        **kwargs: object,
    ) -> RecordMeta:
        if strict is not None and not isinstance(strict, bool):
            raise TypeError(f"{name}: strict must be a bool, got {strict!r}")
        namespace = dict(namespace)
        annotations = _namespace_annotations(namespace)
        root = not any(isinstance(base, RecordMeta) for base in bases)
        if not root:
            _check_names(name, annotations, namespace)
        defaults = {
            field: namespace.pop(field) for field in annotations if field in namespace
        }
        # Shadows an inherited map, so `dataclass()` finds each slot, not a base's key.
        namespace["__record_keys__"] = {}
        cls = slotted_dataclass(
            mcls, name, bases, namespace, tuple(annotations), {}, kwargs
        )
        fields = () if root else _bind_fields(cls, annotations, defaults)
        cls.__record_fields__ = fields
        if strict is None:
            strict = getattr(cls, "__record_strict__", False)
        cls.__record_strict__ = strict
        cls.__record_keys__ = {field: key for field, _, key in fields}
        return cls

    # Hidden from type checkers, which would otherwise accept any class attribute.
    if not TYPE_CHECKING:

        def __getattribute__(cls, name: str) -> object:
            key = type.__getattribute__(cls, "__record_keys__").get(name)
            return type.__getattribute__(cls, name) if key is None else key


class Record(metaclass=RecordMeta):
    """Base class of a typed record: one frame's data as named, typed fields.

    Subclass it and declare each field with a specifier (`Column`, `Meta`,
    `Cell` or `Matrix3`). The annotation is the specifier's class
    subscripted with the field's value type, and the specifier is given
    either as the default or in `Annotated`::

        class MaceSample(Record):
            positions: Column[NDArray[np.float64]] = Column("pos", width=3)
            forces: Annotated[Column[NDArray[np.float64]], Column(width=3)]
            energy: Meta[float] = Meta("REF_energy")

    Keep aliases used by postponed annotations at module scope;
    function-local names cannot be resolved when the class is defined.

    The `Annotated` form allows an alias shared across records. Every
    field, and `n_atoms`, is a required keyword argument of `__init__`,
    which checks each value against its field: Python type or dtype,
    width or shape, `None` only where the annotation allows it, and
    `n_atoms` rows for a per-atom field. Values are kept as given, neither
    copied nor converted nor made read-only. A numpy scalar is accepted for
    a scalar field, and an `int` for a float field. Instances are frozen and
    slotted: they have no `__dict__`, and assigning to a field raises
    `dataclasses.FrozenInstanceError`.

    `oxyz.read(path, record=MaceSample)` and `oxyz.iread` read a file
    into records, `from_frame` builds one from an `oxyz.Frame`, and `spec`
    gives the `SchemaSpec` that describes the record's fields. Records
    compare equal when they are of the same class and hold equal values,
    NaN equal to NaN; they are unhashable.

    On the class, a field name gives the field's key, such as
    `MaceSample.energy`: a copy of its specifier with `name` and `owner`
    set. Its `key` is the file key the field reads: the one given to the
    specifier, else the field name (`Lattice` for `Cell`). A subclass
    shares its base's keys.

    Parameters
    ----------
    n_atoms : int
        Number of atoms in the frame.

    Other Parameters
    ----------------
    strict : bool, optional
        Class keyword, as in `class MaceSample(Record, strict=True)`. By
        default a float field accepts an integer value in the file or frame
        and converts it, as for an integer `Lattice`. Under `strict` it
        does not: a read treats the value as the wrong kind, and
        `from_frame` raises `oxyz.SchemaError`. Inherited from the base
        record.

    Raises
    ------
    TypeError
        At class definition, naming the field: an unsupported annotation,
        a specifier whose class differs from the annotation's, no
        specifier or two, a width, shape or fill inconsistent with the
        annotation, two fields with one key on one axis, a key that a
        `SchemaSpec` would read as a pattern (`*`, `?`, `[` or a `re:`
        prefix), a reserved field name or one starting with an underscore,
        or a specifier with no annotation. From `__init__`, naming the
        field: a wrong type or dtype.
    ValueError
        From `__init__`, naming the field: a wrong width, shape or row
        count, or a negative `n_atoms`.

    Examples
    --------
    >>> import oxyz
    >>> from oxyz import Column, Meta, Record
    >>> class Sample(Record):
    ...     species: Column[list[str]] = Column()
    ...     energy: Meta[float] = Meta()
    ...     stress: Meta[float | None] = Meta()
    >>> samples = oxyz.read(DATA / "mixed.extxyz", record=Sample)
    >>> samples[0].species
    ['O', 'H', 'H']
    >>> [s.energy for s in samples]
    [-14.5, -14.48, -14.52]
    >>> samples[0].stress is None
    True
    >>> Sample.energy.key
    'energy'
    """

    n_atoms: int

    __hash__ = None

    def __post_init__(self) -> None:
        cls = type(self)
        n_atoms = self.n_atoms
        _check_n_atoms(cls.__name__, n_atoms)
        for name, field, _ in cls.__record_fields__:
            rows = n_atoms if field.axis == "column" else None
            try:
                field.check(getattr(self, name), rows=rows)
            except (TypeError, ValueError) as error:
                raise type(error)(f"{cls.__name__}.{name}: {error}") from None

    @classmethod
    def _trusted(cls, /, **values: object) -> Self:
        """Build an instance from values already known to fit, unchecked."""
        record = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(record, name, value)
        return record

    @classmethod
    def spec(cls) -> SchemaSpec:
        """Return the project-mode `SchemaSpec` that describes this record's fields.

        One rule per field, in declaration order: a field that is neither
        `X | None` nor given a fill is required, and any other field is
        optional and filled as a record batch fills it: with its `fill=`,
        or NaN for a float. An `X | None` field of another kind declared
        without a fill has `absent="omit"`. The record's `strict` setting is
        not part of the spec.

        A plain read with this spec is not a record read. It fills where a
        record has `None`; under `conformance="warn"` it keeps a frame that
        lacks a required float field, filled with NaN, where a record read
        drops it; and a batch read with it raises when frames differ in an
        omitted field.
        """
        from oxyz._schema import Kind
        from oxyz._schema_spec import ColumnRule, MetadataRule, SchemaSpec

        columns: list[ColumnRule] = []
        metadata: list[MetadataRule] = []
        for _, field, key in cls.__record_fields__:
            assert key.key is not None  # noqa: S101 — type-narrowing only
            required = not field.optional and key.fill is None
            omits = field.optional and key.fill is None and field.kind != Kind.REAL
            absent: Absent = "omit" if omits else "fill"
            if field.axis == "column":
                rule = ColumnRule(
                    key.key,
                    field.kind,
                    width=field.width,
                    required=required,
                    fill=key.fill,
                    absent=absent,
                )
                columns.append(rule)
            else:
                rule = MetadataRule(
                    key.key,
                    field.kind,
                    shape=field.shape,
                    required=required,
                    fill=key.fill,
                    absent=absent,
                )
                metadata.append(rule)
        return SchemaSpec(
            columns=tuple(columns), metadata=tuple(metadata), mode="project"
        )

    @classmethod
    def from_frame(cls, frame: Frame) -> Self:
        """Build a record from a frame, checking it against the record's fields.

        Each field's key is looked up in the frame's columns or metadata. An
        absent key gives `None` for an `X | None` field and the fill for a
        field declared with `fill=`. An integer value is accepted for a
        float field, and converted, unless the record is `strict`.

        Arrays in the record are read-only. Those taken unchanged from the
        frame are views of its arrays, so a later write to the frame shows
        through the record; call `.copy()` on a field for an independent
        array. String lists are copied.

        Raises
        ------
        oxyz.SchemaError
            A required key is absent, or a value has the wrong kind, dtype,
            width or shape. `name` is the key and `frame_index` is `None`.
        TypeError, ValueError
            The frame's `n_atoms` is not a non-negative integer.
        """
        from oxyz._record_fields import field_value
        from oxyz._schema_match import SchemaError

        strict = cls.__record_strict__
        _check_n_atoms(cls.__name__, frame.n_atoms)
        values: dict[str, object] = {"n_atoms": frame.n_atoms}
        for name, field, key in cls.__record_fields__:
            stored = frame.columns if field.axis == "column" else frame.metadata
            try:
                values[name] = field_value(
                    field,
                    key,
                    stored,
                    rows=frame.n_atoms,
                    strict=strict,
                    owned=False,
                    checked=True,
                )
            except SchemaError as error:
                raise SchemaError(f"{cls.__name__}: {error}", name=error.name) from None
        return cls._trusted(**values)

    def __eq__(self, other: object) -> bool:
        """Compare class, atom count and every field, NaN equal to NaN."""
        if not isinstance(other, Record):
            return NotImplemented
        cls = type(self)
        if type(other) is not cls or self.n_atoms != other.n_atoms:
            return False
        return all(
            field.equal(getattr(self, name), getattr(other, name))
            for name, field, _ in cls.__record_fields__
        )

    def __repr__(self) -> str:
        """Summarise the fields: scalars by value, arrays by dtype and shape."""
        cls = type(self)
        fields = cls.__record_fields__
        shown = [f"n_atoms={self.n_atoms}"]
        shown += [
            f"{name}={field.summarise(getattr(self, name))}"
            for name, field, _ in fields[: _summary._MAX_ENTRIES]  # noqa: SLF001 — shared repr limit
        ]
        elided = len(fields) - _summary._MAX_ENTRIES  # noqa: SLF001 — shared repr limit
        if elided > 0:
            shown.append(f"+{elided} more")
        return f"{cls.__name__}({', '.join(shown)})"

    def _repr_html_(self) -> str:
        """Render the fields as a table for a notebook."""
        return _html.record(self)

    def __getstate__(self) -> dict[str, object]:
        return {
            field.name: getattr(self, field.name) for field in dataclasses.fields(self)
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        for name, value in state.items():
            object.__setattr__(self, name, value)
