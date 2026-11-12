from __future__ import annotations

import copy
import dataclasses
import inspect
import pickle
import re
from typing import Annotated, Any

import numpy as np
import numpy.typing as npt
import pytest

from oxyz import Cell, Column, Matrix3, Meta, Record, RecordBatch
from oxyz._record_batch import batch_class
from oxyz._record_fields import FieldType
from oxyz._schema import Kind

type Array = npt.NDArray[np.float64]
type Forces = Annotated[Column[Array], Column("REF_forces", width=3)]


class MaceSample(Record):
    positions: Column[Array] = Column("pos", width=3)
    forces: Forces
    energy: Meta[float] = Meta("REF_energy")
    stress: Meta[Array | None] = Meta("REF_stress", shape=(6,))
    cell: Cell[Array] = Cell()
    config_type: Meta[str] = Meta()


class MadSample(Record):
    species: Column[list[str]] = Column()
    positions: Column[Array] = Column("pos", width=3)
    forces: Column[Array] = Column(width=3)
    energy: Meta[float] = Meta()
    cell: Cell[Array | None] = Cell()
    stress: Matrix3[Array | None] = Matrix3()


class MaceBatch(RecordBatch[MaceSample]):
    positions: Array
    forces: Array
    energy: Array
    stress: Array
    cell: Array
    config_type: list[str]


MACE_BATCH_BODY = """\
positions: Array
forces: Array
energy: Array
stress: Array
cell: Array
config_type: list[str]
"""


def make_batch(cls: type[Any], **overrides: Any) -> Any:
    """A valid two-frame, three-atoms-per-frame batch, built through `__init__`."""
    values: dict[str, Any] = {
        "offsets": np.array([0, 3, 6]),
        "frame_indices": np.array([4, 7]),
        "positions": np.arange(18.0).reshape(6, 3),
        "forces": np.ones((6, 3)),
        "energy": np.array([-1.0, -2.0]),
        "stress": np.arange(12.0).reshape(2, 6),
        "cell": np.stack([np.eye(3), 2 * np.eye(3)]),
        "config_type": ["a", "b"],
    }
    return cls(**(values | overrides))


def define_batch(body: str, base: str = "RecordBatch[MaceSample]") -> type[Any]:
    """Exec `class B(<base>):` with `body` here; return `B`."""
    source = f"class B({base}):\n" + "".join(
        f"    {line}\n" for line in body.splitlines()
    )
    namespace = dict(globals())
    exec(compile(source, "<batch>", "exec"), namespace)
    return namespace["B"]


@pytest.mark.parametrize(
    ("base", "body", "match"),
    [
        (
            "RecordBatch[MaceSample]",
            MACE_BATCH_BODY.replace("config_type: list[str]\n", ""),
            r"B\.config_type: missing",
        ),
        (
            "RecordBatch[MaceSample]",
            MACE_BATCH_BODY + "extra: Array\n",
            r"B\.extra: not a field",
        ),
        (
            "RecordBatch[MaceSample]",
            MACE_BATCH_BODY + "offsets: Array\n",
            r"B\.offsets: not a field",
        ),
        (
            "RecordBatch[MaceSample]",
            MACE_BATCH_BODY.replace("energy: Array", "energy: float"),
            r"B\.energy: .*float",
        ),
        (
            "RecordBatch[MaceSample]",
            MACE_BATCH_BODY.replace("stress: Array", "stress: Array | None"),
            r"B\.stress: ",
        ),
        (
            "RecordBatch[MaceSample]",
            MACE_BATCH_BODY.replace(
                "config_type: list[str]", "config_type: list[list[str]]"
            ),
            r"B\.config_type: ",
        ),
        (
            "RecordBatch[MaceSample]",
            MACE_BATCH_BODY.replace("energy: Array", "energy: Array = Meta()"),
            r"B\.energy: .*specifier",
        ),
        (
            "RecordBatch[MaceSample]",
            MACE_BATCH_BODY.replace("energy: Array", "energy: Array = None"),
            r"B\.energy: .*default",
        ),
        ("RecordBatch", MACE_BATCH_BODY, r"B: .*RecordBatch\[R\]"),
        ("RecordBatch[int]", MACE_BATCH_BODY, r"B: .*int"),
        ("RecordBatch[Record]", MACE_BATCH_BODY, r"B: .*Record"),
    ],
    ids=[
        "missing",
        "extra",
        "base-field",
        "wrong-form",
        "optional-form",
        "wrong-string-form",
        "specifier",
        "default",
        "unparameterised",
        "not-a-record",
        "bare-record",
    ],
)
def test_batch_form_errors(base: str, body: str, match: str):
    with pytest.raises(TypeError, match=match):
        define_batch(body, base)


def test_declared_batch_accepts_the_batch_form_through_aliases():
    alias = define_batch(
        MACE_BATCH_BODY.replace("energy: Array", "energy: npt.NDArray[np.float64]")
    )
    assert alias.__record__ is MaceSample
    assert [f.name for f in dataclasses.fields(alias)] == [
        "offsets",
        "frame_indices",
        "positions",
        "forces",
        "energy",
        "stress",
        "cell",
        "config_type",
    ]


def test_subclass_of_a_declared_batch_inherits_its_record_and_fields():
    class Sub(MaceBatch):
        pass

    assert Sub.__record__ is MaceSample
    assert type(make_batch(Sub)[0]) is MaceSample


def test_generic_form_derives_one_cached_subclass():
    derived = batch_class(RecordBatch[MaceSample])
    assert derived is batch_class(RecordBatch[MaceSample])
    assert issubclass(derived, RecordBatch)
    assert (derived.__name__, derived.__module__) == ("MaceSampleBatch", __name__)
    assert derived.__record__ is MaceSample
    assert batch_class(MaceBatch) is MaceBatch


@pytest.mark.parametrize(
    "target",
    [MaceSample, RecordBatch, RecordBatch[int], 1],  # ty: ignore[invalid-type-arguments]
)
def test_batch_class_rejects_what_is_not_a_batch(target: object):
    with pytest.raises(TypeError):
        batch_class(target)


@pytest.mark.parametrize("cls", [MaceBatch, batch_class(RecordBatch[MaceSample])])
def test_batch_attribute_is_the_slot_not_a_key(cls: type[Any]):
    assert type(inspect.getattr_static(cls, "energy")).__name__ == "member_descriptor"
    assert not hasattr(make_batch(cls), "__dict__")


def test_init_keeps_arrays_as_given_and_instances_are_frozen():
    positions = np.arange(18.0).reshape(6, 3)
    offsets = np.array([0, 3, 6], dtype=np.int32)
    b = make_batch(MaceBatch, positions=positions, offsets=offsets)
    assert b.positions is positions
    assert b.offsets is offsets
    with pytest.raises(dataclasses.FrozenInstanceError):
        b.energy = np.zeros(2)


def test_the_base_class_is_not_a_batch_of_anything():
    with pytest.raises(TypeError, match=r"RecordBatch\[R\]"):
        RecordBatch(offsets=np.array([0]), frame_indices=np.array([], dtype=int))


@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    [
        (
            {"offsets": np.array([0.0, 3.0, 6.0])},
            TypeError,
            r"MaceBatch\.offsets: .*integer",
        ),
        ({"offsets": [0, 3, 6]}, TypeError, r"MaceBatch\.offsets: .*integer"),
        (
            {"offsets": np.array([], dtype=int)},
            ValueError,
            r"MaceBatch\.offsets: .*shape",
        ),
        (
            {"offsets": np.array([1, 3, 6])},
            ValueError,
            r"MaceBatch\.offsets: .*start at 0",
        ),
        (
            {
                "offsets": np.array([0, 4, 3]),
                "positions": np.zeros((3, 3)),
                "forces": np.zeros((3, 3)),
            },
            ValueError,
            r"MaceBatch\.offsets: .*decreas",
        ),
        (
            {"frame_indices": np.array([0.0, 1.0])},
            TypeError,
            r"MaceBatch\.frame_indices: .*integer",
        ),
        (
            {"frame_indices": np.array([0, 1, 2])},
            ValueError,
            r"MaceBatch\.frame_indices: .*2",
        ),
        (
            {"positions": np.zeros((5, 3))},
            ValueError,
            r"MaceBatch\.positions: expected 6 rows",
        ),
        ({"energy": np.zeros(3)}, ValueError, r"MaceBatch\.energy: expected 2 rows"),
        (
            {"config_type": ["a"]},
            ValueError,
            r"MaceBatch\.config_type: expected 2 rows",
        ),
        (
            {"energy": np.zeros(2, dtype=np.float32)},
            TypeError,
            r"MaceBatch\.energy: .*dtype",
        ),
        (
            {"present": {"stress": np.array([True])}},
            ValueError,
            r"MaceBatch\.present\['stress'\]: .*2",
        ),
        (
            {"present": {"stress": np.array([1, 0])}},
            TypeError,
            r"MaceBatch\.present\['stress'\]: .*bool",
        ),
        (
            {"present": {"energy": np.array([True, True])}},
            TypeError,
            r"MaceBatch\.present\['energy'\]: .*not an optional",
        ),
    ],
    ids=[
        "float-offsets",
        "list-offsets",
        "empty-offsets",
        "offsets-not-at-zero",
        "decreasing-offsets",
        "float-frame-indices",
        "frame-indices-length",
        "per-atom-rows",
        "per-frame-rows",
        "string-rows",
        "field-dtype",
        "mask-length",
        "mask-dtype",
        "mask-of-required-field",
    ],
)
def test_init_checks(overrides: dict[str, Any], error: type[Exception], match: str):
    with pytest.raises(error, match=match):
        make_batch(MaceBatch, **overrides)


def test_lookup_by_key_and_name_is_the_attribute():
    b = make_batch(MaceBatch)
    assert b[MaceSample.energy] is b.energy
    assert b["energy"] is b.energy
    assert b[MaceSample.forces] is b.forces
    assert b["config_type"] is b.config_type


def test_foreign_key_and_unknown_name_raise_key_error():
    b = make_batch(MaceBatch)
    with pytest.raises(
        KeyError, match=r"MadSample\.energy is not a field of MaceSample"
    ):
        b[MadSample.energy]
    with pytest.raises(KeyError, match="nope"):
        b["nope"]
    with pytest.raises(KeyError, match="offsets"):
        b["offsets"]
    with pytest.raises(KeyError):
        b[Meta()]
    with pytest.raises(KeyError, match=r"MadSample\.stress"):
        b.present(MadSample.stress)
    with pytest.raises(KeyError, match="nope"):
        b.present("nope")


def test_base_key_works_on_a_subclass_batch():
    class Sub(MaceSample):
        extra: Meta[int] = Meta()

    b = make_batch(batch_class(RecordBatch[Sub]), extra=np.array([1, 2]))
    assert b[MaceSample.energy] is b[Sub.energy] is b.energy
    assert b[Sub.extra] is b.extra
    with pytest.raises(KeyError, match=r"Sub\.extra is not a field of MaceSample"):
        make_batch(MaceBatch)[Sub.extra]


def test_indexing_follows_numpy():
    b = make_batch(MaceBatch)
    assert isinstance(b[0], MaceSample)
    assert len(b) == 2
    assert [type(r) for r in b] == [MaceSample] * 2
    s = b[0:1]
    assert type(s) is MaceBatch
    assert np.shares_memory(s.positions, b.positions)
    g = b[[1, 0]]
    assert type(g) is MaceBatch
    assert not np.shares_memory(g.positions, b.positions)
    assert g.offsets.tolist() == [0, 3, 6]
    assert g.frame_indices.tolist() == [7, 4]
    assert g.config_type == ["b", "a"]
    assert g.energy.tolist() == [-2.0, -1.0]
    assert b[np.array([False, True])].n_frames == 1
    assert b[::-1] == g
    assert b[[]].n_frames == 0
    assert b[[]].positions.shape == (0, 3)
    assert b[-1] == b[1]
    assert b[np.int64(1)] == b[1]
    with pytest.raises(IndexError):
        b[2]
    with pytest.raises(TypeError):
        b[True]
    with pytest.raises(TypeError):
        b[1.0]


def test_slice_rebases_offsets_onto_its_own_rows():
    b = make_batch(MaceBatch)
    s = b[1:2]
    assert s.offsets.tolist() == [0, 3]
    assert s.frame_indices.tolist() == [7]
    np.testing.assert_array_equal(s.positions, b.positions[3:6])
    assert s[0] == b[1]


def test_a_row_is_a_record_of_views_and_unboxed_scalars():
    b = make_batch(MaceBatch)
    r = b[1]
    assert r.n_atoms == 3
    assert np.shares_memory(r.positions, b.positions)
    np.testing.assert_array_equal(r.positions, b.positions[3:6])
    assert type(r.energy) is float
    assert r.energy == -2.0
    assert r.config_type == "b"
    np.testing.assert_array_equal(r.cell, 2 * np.eye(3))
    np.testing.assert_array_equal(r.stress, np.arange(6.0, 12.0))
    assert r == MaceSample(
        n_atoms=3,
        positions=b.positions[3:6],
        forces=np.ones((3, 3)),
        energy=-2.0,
        stress=np.arange(6.0, 12.0),
        cell=2 * np.eye(3),
        config_type="b",
    )


def test_each_metadata_row_is_unboxed_by_its_own_field(monkeypatch):
    monkeypatch.setattr(FieldType, "unbox", lambda self, row: (self.kind, row))
    r = make_batch(MaceBatch)[0]
    assert r.energy == (Kind.REAL, -1.0)
    assert r.config_type == (Kind.STR, "a")


def test_rows_of_string_columns_are_fresh_lists():
    class Labelled(Record):
        species: Column[list[str]] = Column()
        labels: Meta[list[str]] = Meta(shape=(2,))

    class LabelledBatch(RecordBatch[Labelled]):
        species: list[str]
        labels: list[list[str]]

    b = LabelledBatch(
        offsets=np.array([0, 2]),
        frame_indices=np.array([0]),
        species=["H", "O"],
        labels=[["x", "y"]],
    )
    r = b[0]
    assert (r.species, r.labels) == (["H", "O"], ["x", "y"])
    assert r.species is not b.species
    assert r.labels is not b.labels[0]


def test_present_defaults_to_all_true_and_travels_with_slices_and_gathers():
    b = make_batch(MaceBatch, present={"stress": np.array([True, False])})
    assert (
        b.present("stress").tolist()
        == [True, False]
        == b.present(MaceSample.stress).tolist()
    )
    assert b.present("energy").tolist() == [True, True]
    assert make_batch(MaceBatch).present("stress").tolist() == [True, True]
    assert b[1].stress is None
    assert b[0].stress is not None
    assert b[[1, 0]].present("stress").tolist() == [False, True]
    assert b[::-1].present("stress").tolist() == [False, True]
    assert b[1:2].present("stress").tolist() == [False]
    assert b[1:2][0].stress is None


def test_ptr_n_atoms_and_batch_vector():
    b = make_batch(
        MaceBatch,
        offsets=np.array([0, 2, 5]),
        positions=np.zeros((5, 3)),
        forces=np.zeros((5, 3)),
    )
    assert (b.n_frames, b.total_atoms, len(b)) == (2, 5, 2)
    assert b.n_atoms.tolist() == [2, 3]
    assert b.ptr is b.offsets
    assert b.batch.tolist() == [0, 0, 1, 1, 1]
    assert [r.n_atoms for r in b] == [2, 3]


def test_equality_compares_type_layout_fields_and_masks():
    b = make_batch(MaceBatch)
    nan = make_batch(MaceBatch, energy=np.array([np.nan, 1.0]))
    assert nan == make_batch(MaceBatch, energy=np.array([np.nan, 1.0]))
    assert b == make_batch(MaceBatch, present={"stress": np.array([True, True])})
    assert b != make_batch(MaceBatch, present={"stress": np.array([True, False])})
    assert b != make_batch(MaceBatch, frame_indices=np.array([0, 1]))
    assert b != make_batch(MaceBatch, config_type=["a", "c"])
    assert b != make_batch(batch_class(RecordBatch[MaceSample]))
    assert b.__eq__(1) is NotImplemented
    with pytest.raises(TypeError, match="unhashable"):
        hash(b)


@pytest.mark.parametrize("target", [MaceBatch, RecordBatch[MaceSample]])
def test_batches_pickle_and_copy_in_both_forms(target: object):
    b = make_batch(batch_class(target), present={"stress": np.array([True, False])})
    for again in (pickle.loads(pickle.dumps(b)), copy.copy(b), copy.deepcopy(b)):
        assert type(again) is type(b)
        assert again == b
        assert again.present("stress").tolist() == [True, False]
    assert copy.copy(b).positions is b.positions
    assert not np.shares_memory(copy.deepcopy(b).positions, b.positions)


def test_repr_summarises_without_data():
    b = make_batch(batch_class(RecordBatch[MaceSample]))
    assert repr(b[0:1]) == (
        "MaceSampleBatch(n_frames=1, total_atoms=3, n_atoms=3..3, "
        "positions=float64[3, 3], forces=float64[3, 3], energy=float64[1], "
        "stress=float64[1, 6], cell=float64[1, 3, 3], config_type=str[1])"
    )
    assert repr(make_batch(MaceBatch)[[]]).startswith(
        "MaceBatch(n_frames=0, total_atoms=0, positions=float64[0, 3]"
    )


def html_rows(markup: str) -> dict[str, list[str]]:
    rows = re.findall(r"<tr>(.*?)</tr>", markup)
    cells = [re.findall(r"<td>(.*?)</td>", row) for row in rows if "<td>" in row]
    return {row[0]: row[1:] for row in cells}


def test_repr_html_lists_every_field():
    markup = make_batch(MaceBatch)._repr_html_()
    assert "<strong>MaceBatch</strong> · 2 frames · 6 atoms (3..3 per frame)" in markup
    rows = html_rows(markup)
    assert list(rows) == [
        "positions",
        "forces",
        "energy",
        "stress",
        "cell",
        "config_type",
    ]
    assert rows["energy"] == ["float64", "(2,)", "-1.0, -2.0"]
    assert rows["config_type"] == ["str", "(2,)", "&#x27;a&#x27;, &#x27;b&#x27;"]
