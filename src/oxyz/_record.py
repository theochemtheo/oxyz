"""Typed records: the field specifiers."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, ClassVar, Literal, Self, overload

type Fill = float | int | bool | str


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
