"""The annotation parser and the closed set of record field types."""

from __future__ import annotations

import dataclasses
import importlib
import types
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    TypeAliasType,
    Union,
    get_args,
    get_origin,
)

import numpy as np
import numpy.typing as npt

from oxyz import _summary
from oxyz._project import _fill_matches_kind
from oxyz._schema import Kind, column_sig, metadata_sig
from oxyz._schema_match import SchemaError, column_signature, metadata_signature

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from oxyz._record import FieldSpec, Fill

type _Form = Literal["scalar", "array", "list", "rows"]

_SCALAR_TYPES: dict[Kind, type] = {
    Kind.REAL: float,
    Kind.INT: int,
    Kind.BOOL: bool,
    Kind.STR: str,
}
_DTYPES: dict[Kind, np.dtype] = {
    Kind.REAL: np.dtype(np.float64),
    Kind.INT: np.dtype(np.int64),
    Kind.BOOL: np.dtype(np.bool_),
}
_ARRAY_ANNOTATIONS: dict[Kind, Any] = {
    Kind.REAL: npt.NDArray[np.float64],
    Kind.INT: npt.NDArray[np.int64],
    Kind.BOOL: npt.NDArray[np.bool_],
}
_STRING_ANNOTATIONS: tuple[Any, ...] = (str, list[str], list[list[str]])


@dataclass(frozen=True)
class FieldType:
    """How one record field is typed, checked, cast and batched.

    The frame form describes a record's value; the batch form (`batch_form`)
    the value a batch holds, with a leading frame axis on per-frame fields.
    """

    kind: Kind
    axis: Literal["column", "metadata"]
    width: int = 1
    shape: tuple[int, ...] = ()
    optional: bool = False
    layout: Literal["plain", "cell", "matrix3"] = "plain"
    batch_form: bool = False

    def check(self, value: object, *, rows: int | None) -> None:
        """Raise unless `value` has this form's type and shape.

        `rows` is the expected leading length, or `None` to leave it
        unchecked. A wrong Python type or dtype is a `TypeError`; a wrong
        width, shape or row count a `ValueError`.
        """
        if value is None:
            if not self.optional:
                raise TypeError("is None but the field is not optional")
            if self.batch_form:
                raise TypeError("is None, which a batch never holds")
            return
        leading, trailing = self._value_shape()
        if self.kind == Kind.STR:
            dims = [(k, "items") for k in trailing]
            _check_strings(value, [(rows, "rows"), *dims] if leading else dims)
        elif leading or trailing:
            self._check_array(value, rows=rows, leading=leading, trailing=trailing)
        elif not _scalar_matches(value, self.kind):
            expected = _SCALAR_TYPES[self.kind].__name__
            raise TypeError(f"expected {expected}, got {type(value).__name__}")

    def cast(self, raw: object, *, owned: bool) -> object:
        """Turn a stored value into the field value.

        Unless `owned`, an array comes back read-only and a list as a copy.
        """
        if isinstance(raw, np.ndarray):
            return self._cast_array(raw, owned=owned)
        if isinstance(raw, list):
            if owned:
                return raw
            return [list(row) if isinstance(row, list) else row for row in raw]
        if (
            self.kind == Kind.REAL
            and isinstance(raw, int)
            and not isinstance(raw, bool)
        ):
            return float(raw)
        return raw

    def unbox(self, row: object) -> object:
        """Turn one batch row into the frame value."""
        if isinstance(row, np.generic):
            return row.item()
        if isinstance(row, list):
            return list(row)
        return row

    def batched(self) -> FieldType:
        """Return the batch form of this field type."""
        return self if self.batch_form else dataclasses.replace(self, batch_form=True)

    def annotation(self) -> object:
        """Return the canonical annotation of this form's value."""
        leading, trailing = self._value_shape()
        depth = int(leading) + len(trailing)
        if self.kind == Kind.STR:
            bare = _STRING_ANNOTATIONS[depth]
        elif depth:
            bare = _ARRAY_ANNOTATIONS[self.kind]
        else:
            bare = _SCALAR_TYPES[self.kind]
        return bare | None if self.optional and not self.batch_form else bare

    def accepts(self, annotation: object) -> bool:
        """Report whether `annotation` declares this form's value."""
        value, optional = _strip_none(_bare(annotation))
        own_value, own_optional = _strip_none(_bare(self.annotation()))
        found = _describe(value)
        return (
            found is not None
            and found == _describe(own_value)
            and optional == own_optional
        )

    def filled(self, fill: Fill, *, rows: int) -> object:
        """Return `fill` as a value of this form; `rows` sizes a leading row axis."""
        if self.axis == "column":
            dims = (rows,) if self.width == 1 else (rows, self.width)
        else:
            dims = (rows, *self.shape) if self.batch_form else self.shape
        if self.kind == Kind.STR:
            return _repeated(fill, dims)
        if not dims:
            return _SCALAR_TYPES[self.kind](fill)
        full = np.full(dims, fill, dtype=_DTYPES[self.kind])
        return self._cast_array(full, owned=True)

    def empty(self) -> object:
        """Return a zero-row value of the batch form."""
        if self.kind == Kind.STR:
            return []
        _, trailing = self._value_shape()
        return np.empty((0, *trailing), dtype=_DTYPES[self.kind])

    def equal(self, a: object, b: object) -> bool:
        """Report whether two field values hold the same data, NaN equal to NaN."""
        return _summary.values_equal(a, b)

    def summarise(self, value: object) -> str:
        """Describe a field value: a scalar by its repr, an array by dtype and shape."""
        if isinstance(value, (np.ndarray, list)):
            return _summary.summarise(value)
        return repr(value)

    def _value_shape(self) -> tuple[bool, tuple[int, ...]]:
        """Whether the value has a leading row axis, and its shape after it."""
        if self.axis == "column":
            return True, (() if self.width == 1 else (self.width,))
        trailing = self.shape if self.layout == "plain" else (3, 3)
        return self.batch_form, trailing

    def _check_array(
        self,
        value: object,
        *,
        rows: int | None,
        leading: bool,
        trailing: tuple[int, ...],
    ) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"expected a numpy array, got {type(value).__name__}")
        dtype = _DTYPES[self.kind]
        if value.dtype != dtype:
            raise TypeError(f"expected dtype {dtype}, got {value.dtype}")
        found = value.shape[1:] if leading else value.shape
        if value.ndim != int(leading) + len(trailing) or found != trailing:
            expected = ("n", *trailing) if leading else trailing
            raise ValueError(
                f"expected shape {_shape(expected)}, got {_shape(value.shape)}"
            )
        if leading and rows is not None and value.shape[0] != rows:
            raise ValueError(f"expected {rows} rows, got {value.shape[0]}")

    def _cast_array(self, raw: np.ndarray, *, owned: bool) -> np.ndarray:
        out = raw
        if self.axis == "metadata" and not self.batch_form and raw.ndim > 1:
            out = out.reshape(-1)
        if self.kind == Kind.REAL and out.dtype.kind in "iu":
            out = out.astype(np.float64)
        if self.layout == "cell":
            out = out.reshape(-1, 3, 3) if self.batch_form else out.reshape(3, 3)
        elif self.layout == "matrix3":
            out = (
                out.reshape(-1, 3, 3).transpose(0, 2, 1)
                if self.batch_form
                else out.reshape(3, 3, order="F")
            )
        if not owned:
            if out is raw:
                out = raw.view()
            out.flags.writeable = False
        return out


def field_value(
    field: FieldType,
    spec: FieldSpec[Any],
    stored: Mapping[str, object],
    *,
    rows: int,
    strict: bool,
    owned: bool,
) -> object:
    """Look a field's key up in `stored`, check its signature and cast it.

    An absent key gives `None` for an optional field, else the declared fill.
    An integer value satisfies a REAL field unless `strict`. Unless `owned`,
    arrays come back read-only. Raises `SchemaError` naming the key.
    """
    key = spec.key
    assert key is not None  # noqa: S101 — type-narrowing only
    if key not in stored:
        if field.optional:
            return None
        if spec.fill is None:
            raise SchemaError(f"{field.axis} {key!r}: missing (required)", name=key)
        value = field.filled(spec.fill, rows=rows)
        if isinstance(value, np.ndarray) and not owned:
            value.flags.writeable = False
        return value
    raw = stored[key]
    if field.axis == "column":
        kind, width = column_signature(raw)
        found, expected = column_sig(kind, width), column_sig(field.kind, field.width)
        agrees = width == field.width
    else:
        kind, shape = _metadata_signature(raw, key)
        found, expected = (
            metadata_sig(kind, shape),
            metadata_sig(field.kind, field.shape),
        )
        agrees = shape == field.shape
    promotes = kind == Kind.INT and field.kind == Kind.REAL
    if not agrees or (kind != field.kind and not promotes) or (promotes and strict):
        reason = "; the record is strict" if agrees and promotes else ""
        raise SchemaError(
            f"{field.axis} {key!r}: expected {expected}, found {found}{reason}",
            name=key,
        )
    return field.cast(raw, owned=owned)


def _metadata_signature(raw: object, key: str) -> tuple[Kind, tuple[int, ...]]:
    """`metadata_signature`, counting a 2-D array as its flattened length."""
    if isinstance(raw, np.ndarray) and raw.ndim > 1:
        raw = raw.reshape(-1)
    try:
        return metadata_signature(raw)
    except TypeError as error:
        raise SchemaError(f"metadata {key!r}: {error}", name=key) from None


def unwrap(annotation: object) -> tuple[object, tuple[object, ...]]:
    """Strip type aliases and `Annotated`; return the bare annotation and metadata."""
    metadata: list[object] = []
    while True:
        origin = get_origin(annotation)
        if isinstance(annotation, TypeAliasType):
            annotation = annotation.__value__
        elif isinstance(origin, TypeAliasType):
            annotation = origin.__value__[get_args(annotation)]
        elif origin is Annotated:
            annotation, *extra = get_args(annotation)
            metadata.extend(extra)
        else:
            return annotation, tuple(metadata)


def field_type(annotation: object, spec: FieldSpec[Any]) -> FieldType:
    """Parse a field's unwrapped annotation against its specifier.

    Raises `TypeError` for an annotation outside the supported set.
    """
    if get_origin(annotation) is not type(spec):
        raise TypeError(
            f"annotation {annotation!r} is not {type(spec).__name__}[...], "
            f"the class of its specifier"
        )
    (value,) = get_args(annotation)
    value, optional = _strip_none(_bare(value))
    found = _describe(value)
    if found is None and _is_quantype(value):
        try:
            quantype_fields = importlib.import_module("oxyz._record_quantype")
        except ImportError:
            raise _unsupported(annotation, spec) from None
        return quantype_fields.field_type(annotation, spec)
    if found is None or found not in _allowed_forms(spec):
        raise _unsupported(annotation, spec)
    kind, _ = found
    if spec.fill is not None and not _fill_matches_kind(spec.fill, kind):
        raise TypeError(
            f"fill {spec.fill!r} is not a valid {kind} value for {annotation!r}"
        )
    return FieldType(
        kind,
        spec.axis,
        width=spec.width,
        shape=spec.shape,
        optional=optional,
        layout=spec.layout,
    )


def _allowed_forms(spec: FieldSpec[Any]) -> set[tuple[Kind, _Form]]:
    if spec.layout != "plain":
        return {(Kind.REAL, "array")}
    arrays: set[tuple[Kind, _Form]] = {(kind, "array") for kind in _DTYPES}
    if spec.axis == "column":
        if spec.width < 1 or spec.shape:
            return set()
        return arrays | {(Kind.STR, "list" if spec.width == 1 else "rows")}
    if spec.width != 1 or len(spec.shape) > 1 or min(spec.shape, default=1) < 1:
        return set()
    if spec.shape:
        return arrays | {(Kind.STR, "list")}
    return {(kind, "scalar") for kind in _SCALAR_TYPES}


def _describe(value: object) -> tuple[Kind, _Form] | None:
    """Classify a bare value type by kind and form, or `None` if unsupported."""
    for kind, scalar in _SCALAR_TYPES.items():
        if value is scalar:
            return kind, "scalar"
    origin, args = get_origin(value), get_args(value)
    if origin is np.ndarray and len(args) == 2:  # noqa: PLR2004 — shape and dtype
        dtype = _bare(args[1])
        if get_origin(dtype) is np.dtype:
            (scalar,) = get_args(dtype)
            for kind, expected in _DTYPES.items():
                if scalar is expected.type:
                    return kind, "array"
        return None
    if origin is list and len(args) == 1:
        item = _bare(args[0])
        if item is str:
            return Kind.STR, "list"
        if get_origin(item) is list and tuple(map(_bare, get_args(item))) == (str,):
            return Kind.STR, "rows"
    return None


def _bare(annotation: object) -> object:
    return unwrap(annotation)[0]


def _strip_none(value: object) -> tuple[object, bool]:
    if get_origin(value) not in {Union, types.UnionType}:
        return value, False
    members = [_bare(member) for member in get_args(value)]
    rest = [member for member in members if member is not type(None)]
    if len(rest) == 1 and len(members) == 2:  # noqa: PLR2004 — `X | None`
        return rest[0], True
    return value, False


def _is_quantype(value: object) -> bool:
    module = type(value).__module__
    return module == "quantype" or module.startswith("quantype.")


def _unsupported(annotation: object, spec: FieldSpec[Any]) -> TypeError:
    if spec.axis == "column":
        parameters = f"width={spec.width}"
    else:
        parameters = f"shape={spec.shape}"
    return TypeError(f"{annotation!r} with {parameters} is not a supported field type")


def _scalar_matches(value: object, kind: Kind) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return kind == Kind.BOOL
    if kind == Kind.INT:
        return isinstance(value, (int, np.integer))
    if kind == Kind.REAL:
        return isinstance(value, (int, float, np.integer, np.floating))
    if kind == Kind.STR:
        return isinstance(value, str)
    return False


def _repeated(fill: Fill, dims: tuple[int, ...]) -> object:
    if not dims:
        return fill
    return [_repeated(fill, dims[1:]) for _ in range(dims[0])]


def _check_strings(value: object, dims: Sequence[tuple[int | None, str]]) -> None:
    if not dims:
        if not isinstance(value, str):
            raise TypeError(f"expected str, got {type(value).__name__}")
        return
    if not isinstance(value, list):
        raise TypeError(f"expected a list, got {type(value).__name__}")
    (expected, unit), *inner = dims
    if expected is not None and len(value) != expected:
        raise ValueError(f"expected {expected} {unit}, got {len(value)}")
    for item in value:
        _check_strings(item, inner)


def _shape(shape: tuple[object, ...]) -> str:
    return "(" + ", ".join(map(str, shape)) + ("," if len(shape) == 1 else "") + ")"
