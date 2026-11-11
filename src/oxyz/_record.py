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

if TYPE_CHECKING:
    from oxyz._record_fields import FieldType

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
    the other 3x3 entries in Fortran order.

    Parameters
    ----------
    fill : float, optional
        Value given to all nine entries where `Lattice` is absent; `0.0`
        matches `Frame.cell`.
    """

    __slots__ = ()
    axis = "metadata"
    layout = "cell"

    def __init__(self, *, fill: float | None = None) -> None:
        super().__init__(key="Lattice", shape=(9,), fill=fill)


class Matrix3[T](FieldSpec[T]):
    """Declare a per-frame 3x3 matrix, read from a 9-value entry.

    The annotation is `Matrix3[NDArray[np.float64]]`, optionally `| None`.
    The nine values are read in Fortran order, as ASE reads `stress` and
    `virial`; `Cell` reads `Lattice` in C order.

    Parameters
    ----------
    key : str, optional
        Entry key in the comment line. Defaults to the field name.
    fill : float, optional
        Value given to all nine entries where the entry is absent.
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


def _check_names(
    name: str, annotations: dict[str, Any], namespace: dict[str, Any]
) -> None:
    for field in annotations:
        if field in _RESERVED:
            raise TypeError(f"{name}.{field}: the name is reserved")
        default = namespace.get(field)
        if field in namespace and not isinstance(default, FieldSpec):
            raise TypeError(
                f"{name}.{field}: default {default!r} is not a field specifier"
            )


def _bind_fields(
    cls: RecordMeta,
    annotations: dict[str, Any],
    defaults: dict[str, FieldSpec[Any]],
) -> tuple[tuple[str, FieldType, FieldSpec[Any]], ...]:
    from oxyz._record_fields import field_type, unwrap

    hints = _resolve(cls, annotations)
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
        fields[field] = (ft, spec._bound(field, cls))  # noqa: SLF001 — same module
    claimed: dict[tuple[str, str | None], str] = {}
    for field, (_, key) in fields.items():
        other = claimed.setdefault((key.axis, key.key), field)
        if other != field:
            raise TypeError(
                f"{cls.__name__}.{field}: key {key.key!r} is also read by {other}"
            )
    return tuple((field, ft, key) for field, (ft, key) in fields.items())


def _resolve(cls: RecordMeta, annotations: dict[str, Any]) -> dict[str, Any]:
    try:
        return typing.get_type_hints(cls, include_extras=True)
    except (NameError, TypeError) as error:
        for field, annotation in annotations.items():
            if not _resolves(cls, annotation):
                raise TypeError(f"{cls.__name__}.{field}: {error}") from error
        raise TypeError(f"{cls.__name__}: {error}") from error


def _resolves(cls: RecordMeta, annotation: object) -> bool:
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
        namespace = dict(namespace)
        annotations = _namespace_annotations(namespace)
        root = not any(isinstance(base, RecordMeta) for base in bases)
        if not root:
            _check_names(name, annotations, namespace)
        defaults = {
            field: namespace.pop(field) for field in annotations if field in namespace
        }
        namespace["__slots__"] = tuple(annotations)
        # Shadows an inherited map, so `dataclass()` finds each slot, not a base's key.
        namespace["__record_keys__"] = {}
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        dataclasses.dataclass(frozen=True, kw_only=True, eq=False)(cls)
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

    The `Annotated` form allows an alias shared across records. Every
    field, and `n_atoms`, is a required keyword argument of `__init__`.
    Instances are frozen and slotted: they have no `__dict__`, and
    assigning to a field raises `dataclasses.FrozenInstanceError`.

    On the class, a field name gives the field's key, such as
    `MaceSample.energy`: a copy of its specifier with `name` and `owner`
    set and `key` defaulting to the field name. A subclass shares its
    base's keys.

    Parameters
    ----------
    n_atoms : int
        Number of atoms in the frame.

    Other Parameters
    ----------------
    strict : bool, optional
        Class keyword, as in `class MaceSample(Record, strict=True)`. Under
        `strict`, a float field does not accept an integer value from the
        file. Inherited from the base record; `False` by default.

    Raises
    ------
    TypeError
        At class definition, naming the field: an unsupported annotation,
        a specifier whose class differs from the annotation's, no
        specifier or two, a width, shape or fill inconsistent with the
        annotation, two fields with one key on one axis, or a reserved
        field name.
    """

    n_atoms: int

    def __getstate__(self) -> dict[str, object]:
        return {
            field.name: getattr(self, field.name) for field in dataclasses.fields(self)
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        for name, value in state.items():
            object.__setattr__(self, name, value)
