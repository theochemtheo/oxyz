from __future__ import annotations

import types
from typing import TYPE_CHECKING, Annotated, Any, assert_type

import numpy as np
import numpy.typing as npt
import pytest

from oxyz import Cell, Column, FieldSpec, Matrix3, Meta, Record
from oxyz._record_fields import FieldType, field_type, unwrap
from oxyz._schema import Kind

type Array = npt.NDArray[np.float64]
FORCES_SPEC = Column("REF_forces", width=3)
type Forces = Annotated[Column[Array], FORCES_SPEC]

CELL = FieldType(Kind.REAL, "metadata", shape=(9,), layout="cell")
MATRIX3 = FieldType(Kind.REAL, "metadata", shape=(9,), layout="matrix3")
REAL_COLUMN = FieldType(Kind.REAL, "column")
REAL_META = FieldType(Kind.REAL, "metadata")


class _Declared:
    energy: Meta[float] = Meta("REF_energy")
    forces: Column[Array] = Column("REF_forces", width=3)


def test_specifier_is_typed_by_its_annotation():
    if TYPE_CHECKING:
        assert_type(_Declared.energy, Meta[float])
        assert_type(_Declared().energy, float)
        assert_type(_Declared().forces, Array)
        _mismatched: Meta[float] = Column()  # ty: ignore[invalid-assignment]
    assert _Declared.__dict__["energy"].key == "REF_energy"


def test_specifiers_carry_uniform_parameters():
    column, meta, cell, matrix = (
        Column(width=3, fill=0.0),
        Meta(shape=(6,)),
        Cell(),
        Matrix3("virial"),
    )
    assert (column.key, column.width, column.shape, column.fill) == (None, 3, (), 0.0)
    assert (meta.key, meta.width, meta.shape, meta.fill) == (None, 1, (6,), None)
    assert (cell.key, cell.width, cell.shape) == ("Lattice", 1, (9,))
    assert (matrix.key, matrix.width, matrix.shape) == ("virial", 1, (9,))
    assert [s.axis for s in (column, meta, cell, matrix)] == [
        "column",
        "metadata",
        "metadata",
        "metadata",
    ]
    assert [s.layout for s in (column, meta, cell, matrix)] == [
        "plain",
        "plain",
        "cell",
        "matrix3",
    ]
    assert all(s.name is None and s.owner is None for s in (column, meta, cell, matrix))
    assert all(isinstance(s, FieldSpec) for s in (column, meta, cell, matrix))


def test_specifiers_are_slotted():
    for spec in (Column(), Meta(), Cell(), Matrix3()):
        assert not hasattr(spec, "__dict__")


def test_bound_is_a_copy_whose_key_defaults_to_the_name():
    class Owner:
        pass

    spec = Meta(shape=(6,))
    bound = spec._bound("stress", Owner)
    assert bound is not spec
    assert (bound.key, bound.name, bound.owner, bound.shape) == (
        "stress",
        "stress",
        Owner,
        (6,),
    )
    assert (spec.key, spec.name, spec.owner) == (None, None, None)
    assert Meta("REF_energy")._bound("energy", Owner).key == "REF_energy"
    assert type(Cell()._bound("cell", Owner)) is Cell


def test_keys_compare_by_identity():
    assert Meta("energy") != Meta("energy")


@pytest.mark.parametrize(
    ("annotation", "spec", "expected"),
    [
        (Column[Array], Column(width=3), FieldType(Kind.REAL, "column", width=3)),
        (Column[npt.NDArray[np.int64]], Column(), FieldType(Kind.INT, "column")),
        (Column[npt.NDArray[np.bool_]], Column(), FieldType(Kind.BOOL, "column")),
        (Column[list[str]], Column(), FieldType(Kind.STR, "column")),
        (
            Column[list[list[str]]],
            Column(width=2),
            FieldType(Kind.STR, "column", width=2),
        ),
        (Meta[float], Meta(), FieldType(Kind.REAL, "metadata")),
        (Meta[int], Meta(), FieldType(Kind.INT, "metadata")),
        (Meta[bool], Meta(), FieldType(Kind.BOOL, "metadata")),
        (Meta[str], Meta(), FieldType(Kind.STR, "metadata")),
        (
            Meta[Array | None],
            Meta(shape=(6,)),
            FieldType(Kind.REAL, "metadata", shape=(6,), optional=True),
        ),
        (
            Meta[list[str]],
            Meta(shape=(2,)),
            FieldType(Kind.STR, "metadata", shape=(2,)),
        ),
        (
            Cell[Array],
            Cell(),
            FieldType(Kind.REAL, "metadata", shape=(9,), layout="cell"),
        ),
        (
            Matrix3[Array | None],
            Matrix3(),
            FieldType(
                Kind.REAL, "metadata", shape=(9,), optional=True, layout="matrix3"
            ),
        ),
        (
            Meta[float | None],
            Meta(fill=0),
            FieldType(Kind.REAL, "metadata", optional=True),
        ),
    ],
)
def test_field_type_table(annotation, spec, expected):
    assert field_type(annotation, spec) == expected


UNSUPPORTED = "is not a supported field type"


@pytest.mark.parametrize(
    ("annotation", "spec", "match"),
    [
        (Meta[float], Column(), r"is not Column\[\.\.\.\]"),
        (float, Meta(), r"is not Meta\[\.\.\.\]"),
        (Meta, Meta(), r"is not Meta\[\.\.\.\]"),
        (Column[npt.NDArray[np.float32]], Column(), rf"float32.*{UNSUPPORTED}"),
        (Meta[float], Meta(shape=(3,)), rf"shape=\(3,\) {UNSUPPORTED}"),
        (Meta[Array], Meta(), rf"shape=\(\) {UNSUPPORTED}"),
        (Column[list[str]], Column(width=2), f"width=2 {UNSUPPORTED}"),
        (Column[list[list[str]]], Column(), f"width=1 {UNSUPPORTED}"),
        (Meta[Array], Meta(shape=(3, 3)), rf"shape=\(3, 3\) {UNSUPPORTED}"),
        (Meta[Array], Meta(shape=(0,)), rf"shape=\(0,\) {UNSUPPORTED}"),
        (
            Meta[list[list[str]]],
            Meta(shape=(2,)),
            rf"list\[list\[str\]\]\].*{UNSUPPORTED}",
        ),
        (Meta[float | str], Meta(), rf"float \| str.*{UNSUPPORTED}"),
        (Cell[npt.NDArray[np.int64]], Cell(), rf"int64.*{UNSUPPORTED}"),
        (Matrix3[float], Matrix3(), rf"Matrix3\[float\].*{UNSUPPORTED}"),
        (Meta[int], Meta(fill=1.5), "fill 1.5 is not a valid Int value"),
        (Meta[bool], Meta(fill=1), "fill 1 is not a valid Bool value"),
        (Column[Array], Column(width=0), f"width=0 {UNSUPPORTED}"),
    ],
)
def test_field_type_rejects(annotation, spec, match):
    with pytest.raises(TypeError, match=match):
        field_type(annotation, spec)


@pytest.mark.parametrize(
    ("annotation", "specifier"),
    [
        pytest.param(Column[np.ndarray[Any, Any]], Column(), id="untyped-ndarray"),
        pytest.param(Column[list[int]], Column(), id="integer-list-column"),
        pytest.param(Column[list[list[int]]], Column(width=2), id="integer-list-rows"),
        pytest.param(Meta[list[int]], Meta(shape=(2,)), id="integer-list-metadata"),
    ],
)
def test_record_declarations_reject_unsupported_array_and_list_annotations(
    annotation, specifier
):
    with pytest.raises(
        TypeError, match=r"^Invalid\.values: .*not a supported field type"
    ):
        type(
            "Invalid",
            (Record,),
            {
                "__module__": __name__,
                "__annotations__": {"values": annotation},
                "values": specifier,
            },
        )


def test_unwrap_strips_pep695_alias_and_annotated():
    assert unwrap(Forces) == (Column[Array], (FORCES_SPEC,))
    assert unwrap(Annotated[Forces, "note"]) == (Column[Array], ("note", FORCES_SPEC))
    assert unwrap(Meta[float]) == (Meta[float], ())


class _QuantityAlias:
    """Stands in for a quantype annotation without importing quantype."""

    __module__ = "quantype._units"


QUANTITY_FIELD = types.GenericAlias(Meta, (_QuantityAlias(),))


def test_quantype_value_without_its_module_is_a_type_error():
    with pytest.raises(TypeError, match="not a supported field type"):
        field_type(QUANTITY_FIELD, Meta())


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        (REAL_META, npt.NDArray[np.float64]),
        (FieldType(Kind.INT, "metadata"), npt.NDArray[np.int64]),
        (FieldType(Kind.BOOL, "metadata"), npt.NDArray[np.bool_]),
        (FieldType(Kind.STR, "metadata"), list[str]),
        (FieldType(Kind.STR, "metadata", shape=(2,)), list[list[str]]),
        (
            FieldType(Kind.REAL, "metadata", shape=(6,), optional=True),
            npt.NDArray[np.float64],
        ),
        (FieldType(Kind.STR, "column"), list[str]),
        (FieldType(Kind.STR, "column", width=2), list[list[str]]),
        (FieldType(Kind.INT, "column", width=3), npt.NDArray[np.int64]),
        (CELL, npt.NDArray[np.float64]),
    ],
)
def test_batched_forms(field, expected):
    batched = field.batched()
    assert batched.batch_form
    assert batched.batched() is batched
    assert batched.annotation() == expected


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        (REAL_META, float),
        (FieldType(Kind.STR, "metadata"), str),
        (FieldType(Kind.STR, "metadata", shape=(2,)), list[str]),
        (FieldType(Kind.REAL, "metadata", optional=True), float | None),
        (FieldType(Kind.STR, "column", width=2), list[list[str]]),
        (MATRIX3, npt.NDArray[np.float64]),
    ],
)
def test_frame_form_annotations(field, expected):
    assert field.annotation() == expected


def test_accepts_sees_through_aliases():
    assert CELL.batched().accepts(Array)
    assert REAL_META.batched().accepts(Array)
    assert not REAL_META.batched().accepts(float)
    assert not REAL_META.batched().accepts(npt.NDArray[np.float32])
    assert not REAL_META.batched().accepts(Array | None)
    assert FieldType(Kind.STR, "metadata").batched().accepts(list[str])
    assert not FieldType(Kind.STR, "metadata").batched().accepts(list[list[str]])


@pytest.mark.parametrize(
    "field",
    [
        REAL_META,
        FieldType(Kind.REAL, "metadata", optional=True),
        FieldType(Kind.STR, "column", width=2, optional=True),
        FieldType(Kind.REAL, "metadata", shape=(6,), optional=True).batched(),
        CELL,
    ],
)
def test_accepts_its_own_annotation(field):
    assert field.accepts(field.annotation())


def test_accepts_tells_optional_from_required():
    optional = FieldType(Kind.REAL, "metadata", optional=True)
    assert optional.accepts(float | None)
    assert not optional.accepts(float)
    assert not REAL_META.accepts(float | None)


def test_cell_and_matrix3_frame_casts_are_views_in_ase_layout():
    flat = np.arange(9.0)
    cell = CELL.cast(flat, owned=True)
    m3 = MATRIX3.cast(flat, owned=True)
    assert isinstance(cell, np.ndarray)
    assert isinstance(m3, np.ndarray)
    assert np.shares_memory(cell, flat)
    assert np.shares_memory(m3, flat)
    assert cell.tolist() == flat.reshape(3, 3).tolist()
    assert m3.tolist() == flat.reshape(3, 3, order="F").tolist()


def test_cell_and_matrix3_batch_casts_are_views_in_ase_layout():
    rows = np.arange(18.0).reshape(2, 9)
    cells = CELL.batched().cast(rows, owned=True)
    m3s = MATRIX3.batched().cast(rows, owned=True)
    assert isinstance(cells, np.ndarray)
    assert isinstance(m3s, np.ndarray)
    assert np.shares_memory(cells, rows)
    assert np.shares_memory(m3s, rows)
    assert cells[1].tolist() == rows[1].reshape(3, 3).tolist()
    assert m3s[1].tolist() == rows[1].reshape(3, 3, order="F").tolist()


def test_two_dimensional_metadata_is_flattened_and_a_matrix3_kept_as_written():
    square = np.arange(9.0).reshape(3, 3)
    flat = FieldType(Kind.REAL, "metadata", shape=(9,)).cast(square, owned=True)
    assert isinstance(flat, np.ndarray)
    assert flat.tolist() == list(range(9))
    cell = CELL.cast(square, owned=True)
    m3 = MATRIX3.cast(square, owned=True)
    assert isinstance(cell, np.ndarray)
    assert isinstance(m3, np.ndarray)
    assert cell.tolist() == square.tolist()
    assert m3.tolist() == square.tolist()


def test_unowned_cast_is_read_only_and_promotes():
    out = REAL_COLUMN.cast(np.array([1, 2]), owned=False)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    assert not out.flags.writeable


def test_unowned_cast_is_a_read_only_view_that_leaves_the_source_writable():
    raw = np.array([1.0, 2.0])
    out = REAL_COLUMN.cast(raw, owned=False)
    assert isinstance(out, np.ndarray)
    assert np.shares_memory(out, raw)
    assert not out.flags.writeable
    assert raw.flags.writeable


def test_owned_cast_keeps_the_array():
    raw = np.array([1.0, 2.0])
    assert REAL_COLUMN.cast(raw, owned=True) is raw


def test_unowned_cast_copies_lists_down_to_the_rows():
    raw = [["a", "b"], ["c", "d"]]
    out = FieldType(Kind.STR, "column", width=2).cast(raw, owned=False)
    assert out == raw
    assert isinstance(out, list)
    assert out is not raw
    assert out[0] is not raw[0]


def test_cast_promotes_an_integer_scalar_on_a_real_field():
    out = REAL_META.cast(3, owned=True)
    assert type(out) is float
    assert out == 3.0
    assert FieldType(Kind.INT, "metadata").cast(3, owned=True) == 3
    assert type(FieldType(Kind.BOOL, "metadata").cast(True, owned=True)) is bool


def test_unbox_gives_python_scalars_views_and_list_copies():
    energies = np.array([1.5, 2.5])
    assert type(REAL_META.unbox(energies[1])) is float
    stresses = np.zeros((2, 6))
    row = FieldType(Kind.REAL, "metadata", shape=(6,)).unbox(stresses[1])
    assert isinstance(row, np.ndarray)
    assert np.shares_memory(row, stresses)
    labels = [["a", "b"], ["c", "d"]]
    unboxed = FieldType(Kind.STR, "metadata", shape=(2,)).unbox(labels[0])
    assert unboxed == ["a", "b"]
    assert unboxed is not labels[0]
    assert FieldType(Kind.STR, "metadata").unbox("x") == "x"


@pytest.mark.parametrize(
    ("field", "value", "rows", "error", "match"),
    [
        (
            FieldType(Kind.REAL, "column", width=3),
            np.zeros((4, 3), np.int64),
            4,
            TypeError,
            "dtype",
        ),
        (
            FieldType(Kind.REAL, "column", width=3),
            np.zeros((4, 2)),
            4,
            ValueError,
            "shape",
        ),
        (
            FieldType(Kind.REAL, "column", width=3),
            np.zeros((4, 3)),
            3,
            ValueError,
            "rows",
        ),
        (REAL_COLUMN, [1.0, 2.0], 2, TypeError, "numpy array"),
        (FieldType(Kind.STR, "column"), ["a", 1], 2, TypeError, "str"),
        (FieldType(Kind.STR, "column"), ["a", "b"], 3, ValueError, "rows"),
        (
            FieldType(Kind.STR, "column", width=2),
            [["a", "b"], ["c"]],
            2,
            ValueError,
            "2 items",
        ),
        (REAL_META, "1.0", None, TypeError, "float"),
        (REAL_META, True, None, TypeError, "float"),
        (FieldType(Kind.INT, "metadata"), 1.0, None, TypeError, "int"),
        (FieldType(Kind.INT, "metadata"), True, None, TypeError, "int"),
        (FieldType(Kind.INT, "metadata"), np.True_, None, TypeError, "int"),
        (FieldType(Kind.INT, "metadata"), np.float64(1.0), None, TypeError, "int"),
        (REAL_META, np.True_, None, TypeError, "float"),
        (FieldType(Kind.BOOL, "metadata"), np.int64(1), None, TypeError, "bool"),
        (FieldType(Kind.BOOL, "metadata"), 1, None, TypeError, "bool"),
        (REAL_META, None, None, TypeError, "not optional"),
        (
            FieldType(Kind.REAL, "metadata", optional=True).batched(),
            None,
            2,
            TypeError,
            "None",
        ),
        (CELL, np.zeros(9), None, ValueError, "shape"),
        (
            FieldType(Kind.STR, "metadata", shape=(2,)),
            ["a", "b", "c"],
            None,
            ValueError,
            "2 items",
        ),
        (REAL_META.batched(), np.zeros(3), 2, ValueError, "rows"),
        (CELL.batched(), np.zeros((2, 9)), 2, ValueError, "shape"),
    ],
)
def test_check_names_type_and_shape_faults(field, value, rows, error, match):
    with pytest.raises(error, match=match):
        field.check(value, rows=rows)


@pytest.mark.parametrize(
    ("field", "value", "rows"),
    [
        (FieldType(Kind.REAL, "column", width=3), np.zeros((4, 3)), 4),
        (FieldType(Kind.REAL, "column", width=3), np.zeros((4, 3)), None),
        (FieldType(Kind.STR, "column", width=2), [["a", "b"]], 1),
        (REAL_META, 1.5, None),
        (REAL_META, 1, None),
        (REAL_META, np.float64(1.5), None),
        (REAL_META, np.float32(1.5), None),
        (REAL_META, np.int64(1), None),
        (FieldType(Kind.INT, "metadata"), np.int64(1), None),
        (FieldType(Kind.INT, "metadata"), np.arange(3).sum(), None),
        (FieldType(Kind.BOOL, "metadata"), np.True_, None),
        (FieldType(Kind.BOOL, "metadata"), True, None),
        (FieldType(Kind.REAL, "metadata", optional=True), None, None),
        (FieldType(Kind.STR, "metadata"), "x", None),
        (CELL, np.zeros((3, 3)), None),
        (CELL.batched(), np.zeros((2, 3, 3)), 2),
        (FieldType(Kind.STR, "metadata").batched(), ["a", "b"], 2),
        (FieldType(Kind.STR, "metadata", shape=(2,)).batched(), [["a", "b"]], 1),
    ],
)
def test_check_accepts_well_formed_values(field, value, rows):
    field.check(value, rows=rows)


def test_empty_has_the_batch_dtype_and_trailing_shape():
    column = FieldType(Kind.REAL, "column", width=3).batched().empty()
    assert isinstance(column, np.ndarray)
    assert (column.shape, column.dtype) == ((0, 3), np.float64)
    cells = CELL.batched().empty()
    assert isinstance(cells, np.ndarray)
    assert cells.shape == (0, 3, 3)
    counts = FieldType(Kind.INT, "metadata").batched().empty()
    assert isinstance(counts, np.ndarray)
    assert (counts.shape, counts.dtype) == ((0,), np.int64)
    assert FieldType(Kind.STR, "metadata").batched().empty() == []


def test_equal_and_summarise():
    assert REAL_META.equal(np.array([np.nan, 1.0]), np.array([np.nan, 1.0]))
    assert not REAL_META.equal(None, np.zeros(1))
    assert REAL_META.summarise(1.5) == "1.5"
    assert REAL_META.summarise(None) == "None"
    assert FieldType(Kind.STR, "metadata").summarise("x") == "'x'"
    assert CELL.summarise(np.zeros((3, 3))) == "float64[3, 3]"


def _assert_same_value(out: object, expected: object) -> None:
    assert type(out) is type(expected)
    if isinstance(expected, np.ndarray):
        assert isinstance(out, np.ndarray)
        assert (out.dtype, out.shape) == (expected.dtype, expected.shape)
        np.testing.assert_array_equal(out, expected)
    else:
        assert out == expected


@pytest.mark.parametrize(
    ("field", "fill", "rows", "expected"),
    [
        (REAL_META, 0, 4, 0.0),
        (FieldType(Kind.INT, "metadata"), 3, 4, 3),
        (FieldType(Kind.BOOL, "metadata"), True, 4, True),
        (FieldType(Kind.STR, "metadata"), "x", 4, "x"),
        (FieldType(Kind.REAL, "metadata", shape=(6,)), 1.5, 4, np.full(6, 1.5)),
        (FieldType(Kind.INT, "metadata", shape=(3,)), 1, 4, np.ones(3, np.int64)),
        (FieldType(Kind.STR, "metadata", shape=(2,)), "a", 4, ["a", "a"]),
        (CELL, 0.0, 4, np.zeros((3, 3))),
        (MATRIX3, 2, 4, np.full((3, 3), 2.0)),
        (REAL_COLUMN, 0, 2, np.zeros(2)),
        (FieldType(Kind.REAL, "column", width=3), 1.0, 2, np.ones((2, 3))),
        (FieldType(Kind.BOOL, "column"), False, 2, np.zeros(2, np.bool_)),
        (FieldType(Kind.STR, "column"), "X", 2, ["X", "X"]),
        (FieldType(Kind.STR, "column", width=2), "X", 2, [["X", "X"], ["X", "X"]]),
        (REAL_META.batched(), 1, 2, np.ones(2)),
        (
            FieldType(Kind.INT, "metadata", shape=(3,)).batched(),
            0,
            2,
            np.zeros((2, 3), np.int64),
        ),
        (CELL.batched(), 0.0, 2, np.zeros((2, 3, 3))),
        (FieldType(Kind.STR, "metadata").batched(), "a", 2, ["a", "a"]),
        (
            FieldType(Kind.STR, "metadata", shape=(2,)).batched(),
            "a",
            2,
            [["a", "a"], ["a", "a"]],
        ),
    ],
)
def test_filled_is_the_fill_as_a_well_formed_value(field, fill, rows, expected):
    out = field.filled(fill, rows=rows)
    _assert_same_value(out, expected)
    field.check(out, rows=rows)


def test_filled_string_rows_are_distinct_lists():
    out = FieldType(Kind.STR, "column", width=2).filled("X", rows=2)
    assert isinstance(out, list)
    assert out[0] is not out[1]
