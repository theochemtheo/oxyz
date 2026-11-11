from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING, Annotated, assert_type

import numpy as np
import numpy.typing as npt
import pytest

from oxyz import Cell, Column, FieldSpec, Matrix3, Meta
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


@pytest.mark.parametrize(
    ("annotation", "spec"),
    [
        (Meta[float], Column()),
        (float, Meta()),
        (Meta, Meta()),
        (Column[npt.NDArray[np.float32]], Column()),
        (Meta[float], Meta(shape=(3,))),
        (Meta[Array], Meta()),
        (Column[list[str]], Column(width=2)),
        (Column[list[list[str]]], Column()),
        (Meta[Array], Meta(shape=(3, 3))),
        (Meta[Array], Meta(shape=(0,))),
        (Meta[list[list[str]]], Meta(shape=(2,))),
        (Meta[float | str], Meta()),
        (Cell[npt.NDArray[np.int64]], Cell()),
        (Matrix3[float], Matrix3()),
        (Meta[int], Meta(fill=1.5)),
        (Meta[bool], Meta(fill=1)),
        (Column[Array], Column(width=0)),
    ],
)
def test_field_type_rejects(annotation, spec):
    with pytest.raises(TypeError):
        field_type(annotation, spec)


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


def test_quantype_value_is_delegated_to_the_quantype_module(monkeypatch):
    sentinel = FieldType(Kind.REAL, "metadata")
    module = types.SimpleNamespace(field_type=lambda annotation, spec: sentinel)
    monkeypatch.setitem(sys.modules, "oxyz._record_quantype", module)
    assert field_type(QUANTITY_FIELD, Meta()) is sentinel
    with pytest.raises(TypeError, match="not a supported field type"):
        field_type(Meta[complex], Meta())


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


def test_two_dimensional_metadata_is_flattened_row_major_before_the_cast():
    square = np.arange(9.0).reshape(3, 3)
    flat = FieldType(Kind.REAL, "metadata", shape=(9,)).cast(square, owned=True)
    assert isinstance(flat, np.ndarray)
    assert flat.tolist() == list(range(9))
    cell = CELL.cast(square, owned=True)
    m3 = MATRIX3.cast(square, owned=True)
    assert isinstance(cell, np.ndarray)
    assert isinstance(m3, np.ndarray)
    assert cell.tolist() == square.tolist()
    assert m3.tolist() == square.T.tolist()


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
        (FieldType(Kind.BOOL, "metadata"), np.True_, None, TypeError, "bool"),
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
