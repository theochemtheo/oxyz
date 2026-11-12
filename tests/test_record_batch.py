from __future__ import annotations

import copy
import dataclasses
import gzip
import inspect
import pickle
import re
import warnings
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import numpy.typing as npt
import pytest

import oxyz
from oxyz import (
    Batch,
    Cell,
    Column,
    FrameRule,
    Matrix3,
    Meta,
    Record,
    RecordBatch,
    SchemaError,
    SchemaSpec,
    _record_batch,
    _summary,
)
from oxyz._record_batch import batch_class
from oxyz._record_bind import bind
from oxyz._record_fields import FieldType
from oxyz._schema import Kind

DATA = Path(__file__).parent / "data"
MACE = DATA / "mace_ref_energy_forces_stress.xyz"
MAD = DATA / "mad_r2scan_sample.extxyz"

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
        "_present",
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
    assert np.shares_memory(r.stress, b.stress)
    assert np.shares_memory(r.cell, b.cell)
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


REPLACERS = [dataclasses.replace] + ([copy.replace] if hasattr(copy, "replace") else [])


@pytest.mark.parametrize("replace", REPLACERS, ids=lambda f: f"{f.__module__}.replace")
def test_replace_carries_the_masks(replace: Any):
    b = make_batch(MaceBatch, present={"stress": np.array([True, False])})
    replaced = replace(b, energy=np.array([5.0, 6.0]))
    assert replaced.energy.tolist() == [5.0, 6.0]
    assert replaced.present("stress").tolist() == [True, False]
    assert replaced[1].stress is None
    assert replaced[0].stress is not None


@pytest.mark.parametrize("replace", REPLACERS, ids=lambda f: f"{f.__module__}.replace")
def test_replace_checks_the_values_it_is_given(replace: Any):
    b = make_batch(MaceBatch, present={"stress": np.array([True, False])})
    with pytest.raises(ValueError, match=r"MaceBatch\.energy: expected 2 rows"):
        replace(b, energy=np.zeros(3))
    one_frame = {
        "offsets": np.array([0, 3]),
        "frame_indices": np.array([4]),
        "positions": np.zeros((3, 3)),
        "forces": np.zeros((3, 3)),
        "energy": np.zeros(1),
        "stress": np.zeros((1, 6)),
        "cell": np.zeros((1, 3, 3)),
        "config_type": ["a"],
    }
    with pytest.raises(ValueError, match=r"MaceBatch\.present\['stress'\]: .*\(1,\)"):
        replace(b, **one_frame)


@pytest.mark.parametrize("replace", REPLACERS, ids=lambda f: f"{f.__module__}.replace")
def test_present_given_to_replace_wins(replace: Any):
    b = make_batch(MaceBatch, present={"stress": np.array([True, False])})
    replaced = replace(b, present={"stress": np.array([False, True])})
    assert replaced.present("stress").tolist() == [False, True]
    assert replaced[0].stress is None


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


def test_generic_batch_unpickles_where_its_class_is_not_yet_derived(monkeypatch):
    b = make_batch(
        batch_class(RecordBatch[MaceSample]),
        present={"stress": np.array([True, False])},
    )
    data = pickle.dumps(b)
    monkeypatch.setattr(_record_batch, "_derived", {})
    again = pickle.loads(data)
    assert type(again) is batch_class(RecordBatch[MaceSample])
    assert type(again) is not type(b)
    assert type(again).__name__ == "MaceSampleBatch"
    assert again.present("stress").tolist() == [True, False]
    assert again[1].stress is None
    np.testing.assert_array_equal(again.positions, b.positions)


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


MACE_FIELDS = ["positions", "forces", "energy", "stress", "cell", "config_type"]
SEQUENTIAL_AND_PLANNED: list[dict[str, Any]] = [{"threads": 1}, {}]
MAD_BATCH: Any = RecordBatch[MadSample]


def frames_file(path: Path, *comments: str) -> Path:
    """Write one single-hydrogen frame per comment-line tail to `path`."""
    path.write_text(
        "".join(
            f"1\nProperties=species:S:1:pos:R:3 {comment}\nH 0.0 0.0 0.0\n"
            for comment in comments
        )
    )
    return path


def unparseable(tmp_path: Path) -> Path:
    """A file whose first line is not an atom count, so reading it fails."""
    path = tmp_path / "unparseable.xyz"
    path.write_text("not an atom count\n")
    return path


@pytest.fixture(scope="module")
def whole() -> Any:
    return oxyz.read_batch(MAD, record=MAD_BATCH)


def test_generic_and_declared_forms_read_identical_values():
    g = oxyz.read_batch(MACE, record=RecordBatch[MaceSample])
    d = oxyz.read_batch(MACE, record=MaceBatch)
    assert type(g) is batch_class(RecordBatch[MaceSample])
    assert type(d) is MaceBatch
    for name in MACE_FIELDS:
        assert _summary.values_equal(g[name], d[name])
    assert d.cell.shape == (1, 3, 3)
    assert d.energy.shape == (1,)
    assert d.energy.tolist() == [-76.123]
    np.testing.assert_array_equal(d.cell[0], 5 * np.eye(3))
    assert d.config_type == ["Default"]
    assert d[0] == oxyz.read(MACE, 0, record=MaceSample)


def test_batch_rows_equal_the_frame_read(whole: Any):
    recs = oxyz.read(MAD, record=MadSample)
    assert len(whole) == len(recs) == 201
    assert all(whole[i] == recs[i] for i in range(201))
    assert whole.frame_indices.tolist() == list(range(201))
    assert oxyz.read_batch(MAD, "140:150", record=MAD_BATCH) == whole[140:150]


def test_mixed_presence_is_nan_filled_and_masked(whole: Any):
    plain = oxyz.read(MAD)
    for name, key in (("stress", "stress"), ("cell", "Lattice")):
        has = np.array([key in f.metadata for f in plain])
        assert 0 < has.sum() < 201
        assert whole.present(name).tolist() == has.tolist()
        value = whole[name]
        assert value.shape == (201, 3, 3)
        assert np.isnan(value[~has]).all()
        assert not np.isnan(value[has]).any()
        assert [getattr(whole[i], name) is None for i in range(201)] == (~has).tolist()
    assert whole.present(MadSample.stress).tolist() == whole.present("stress").tolist()


class Nanful(Record):
    pair: Meta[Array | None] = Meta(shape=(2,))
    charges: Column[Array | None] = Column("q")


def test_a_nan_in_the_file_is_present(tmp_path: Path):
    path = tmp_path / "nan.xyz"
    path.write_text(
        '1\nProperties=species:S:1:pos:R:3:q:R:1 pair="nan nan"\nH 0 0 0 nan\n'
        "1\nProperties=species:S:1:pos:R:3 step=1\nH 0 0 0\n"
    )
    b = oxyz.read_batch(path, record=RecordBatch[Nanful])
    assert b.present("pair").tolist() == [True, False]
    assert b.present("charges").tolist() == [True, False]
    assert np.isnan(b[Nanful.pair]).all()
    assert np.isnan(b[Nanful.charges]).all()
    first, second = b
    assert first.pair is not None
    assert np.isnan(first.pair).all()
    assert first.charges is not None
    assert (second.pair, second.charges) == (None, None)
    assert [first, second] == oxyz.read(path, record=Nanful)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"frames_per_batch": 50, "threads": 1}, id="sequential"),
        pytest.param({"frames_per_batch": 50}, id="planned"),
        pytest.param(
            {"frames_per_batch": 50, "shuffle": True, "seed": 0}, id="shuffled"
        ),
        pytest.param({"atoms_per_batch": 400}, id="atom-budget"),
        pytest.param({"memory_scales_with": "n_atoms", "max_scaler": 400}, id="binned"),
    ],
)
def test_iread_batch_matches_read_batch(whole: Any, kwargs: dict[str, Any]):
    seen: list[int] = []
    for b in oxyz.iread_batch(MAD, record=MAD_BATCH, **kwargs):
        assert type(b) is type(whole)
        rows = whole[b.frame_indices]
        for name in ("stress", "cell"):
            assert b.present(name).tolist() == rows.present(name).tolist()
        assert b == rows
        seen += b.frame_indices.tolist()
    assert sorted(seen) == list(range(201))


def test_compressed_source_and_repeated_selection(tmp_path: Path, whole: Any):
    gz = tmp_path / "mad.extxyz.gz"
    gz.write_bytes(gzip.compress(MAD.read_bytes()))
    assert oxyz.read_batch(gz, record=MAD_BATCH) == whole
    streamed = list(oxyz.iread_batch(gz, record=MAD_BATCH, frames_per_batch=50))
    assert [b.frame_indices[0] for b in streamed] == [0, 50, 100, 150, 200]
    assert all(b == whole[b.frame_indices] for b in streamed)
    picked = oxyz.read_batch(MAD, [150, 150, 3], record=MAD_BATCH)
    assert picked.frame_indices.tolist() == [150, 150, 3]
    assert picked.present("stress").tolist() == [True, True, False]
    assert picked == whole[[150, 150, 3]]
    assert oxyz.read_batch(gz, [150, 150, 3], record=MAD_BATCH) == picked


def test_non_real_optional_needs_a_fill_to_batch_but_not_to_read_frames(
    tmp_path: Path,
):
    class Numbered(Record):
        frame_id: Meta[int | None] = Meta()

    class Filled(Record):
        frame_id: Meta[int | None] = Meta(fill=-1)

    recs = oxyz.read(MAD, record=Numbered)
    assert all(type(r.frame_id) is int for r in recs)
    bad = unparseable(tmp_path)
    with pytest.raises(SchemaError, match=r"Numbered\.frame_id") as e:
        oxyz.read_batch(bad, record=RecordBatch[Numbered])
    assert (e.value.name, e.value.frame_index) == ("frame_id", None)
    with pytest.raises(SchemaError, match=r"Numbered\.frame_id"):
        oxyz.iread_batch(bad, record=RecordBatch[Numbered], frames_per_batch=2)
    b = oxyz.read_batch(MAD, record=RecordBatch[Filled])
    assert b[Filled.frame_id].dtype == np.int64
    assert [r.frame_id for r in b] == [r.frame_id for r in recs]
    gappy = frames_file(tmp_path / "gappy.xyz", "frame_id=7", "step=1")
    g = oxyz.read_batch(gappy, record=RecordBatch[Filled])
    assert g[Filled.frame_id].tolist() == [7, -1]
    assert g.present("frame_id").tolist() == [True, False]
    assert [r.frame_id for r in g] == [7, None]


class MaybeStressed(Record):
    step: Meta[int] = Meta()
    stress: Matrix3[Array | None] = Matrix3()


def test_warn_keeps_the_same_frames_as_the_frame_read(tmp_path: Path):
    path = frames_file(
        tmp_path / "kinds.xyz",
        'step=0 stress="1 2 3 4 5 6 7 8 9"',
        "step=1 stress=abc",
        "step=2",
    )
    target: Any = RecordBatch[MaybeStressed]
    with pytest.warns(oxyz.SchemaWarning, match="stress"):
        recs = oxyz.read(path, record=MaybeStressed, conformance="warn")
    with pytest.warns(oxyz.SchemaWarning, match="stress"):
        b = oxyz.read_batch(path, record=target, conformance="warn")
    assert b.frame_indices.tolist() == [0, 2]
    assert b.present("stress").tolist() == [True, False]
    assert list(b) == recs
    for kwargs in SEQUENTIAL_AND_PLANNED:
        with pytest.warns(oxyz.SchemaWarning, match="stress"):
            windows = list(
                oxyz.iread_batch(
                    path,
                    record=target,
                    frames_per_batch=1,
                    conformance="warn",
                    **kwargs,
                )
            )
        assert [w.frame_indices.tolist() for w in windows] == [[0], [2]]
        assert [w[0] for w in windows] == recs
    with pytest.raises(SchemaError) as e:
        oxyz.read_batch(path, record=target)
    assert (e.value.frame_index, e.value.name) == (1, "stress")


def assert_empty_mace(b: Any, cls: type[Any]) -> None:
    assert type(b) is cls
    assert len(b) == 0
    assert b.offsets.tolist() == [0]
    assert b.frame_indices.shape == (0,)
    assert (b.positions.shape, b.positions.dtype) == ((0, 3), np.float64)
    assert b.forces.shape == (0, 3)
    assert (b.energy.shape, b.energy.dtype) == ((0,), np.float64)
    assert b.stress.shape == (0, 6)
    assert b.cell.shape == (0, 3, 3)
    assert b.config_type == []
    assert b.present("stress").shape == (0,)
    assert list(b) == []
    assert dataclasses.replace(b) == b


def test_everything_dropped_is_an_empty_typed_batch(tmp_path: Path):
    path = tmp_path / "no_energy.xyz"
    path.write_text(MACE.read_text().replace("REF_energy=-76.123 ", "") * 2)
    with pytest.warns(oxyz.SchemaWarning, match="REF_energy"):
        b = oxyz.read_batch(path, record=MaceBatch, conformance="warn")
    assert_empty_mace(b, MaceBatch)
    for kwargs in SEQUENTIAL_AND_PLANNED:
        batches = oxyz.iread_batch(
            path, record=MaceBatch, frames_per_batch=2, conformance="warn", **kwargs
        )
        with pytest.warns(oxyz.SchemaWarning, match="REF_energy"):
            assert list(batches) == []
    assert_empty_mace(oxyz.read_batch(MACE, [], record=MaceBatch), MaceBatch)
    generic = oxyz.read_batch(MAD, "5:5", record=MAD_BATCH)
    assert type(generic) is batch_class(MAD_BATCH)
    assert (len(generic), generic.species, generic.cell.shape) == (0, [], (0, 3, 3))
    assert generic.present("cell").shape == (0,)


class Squares(Record):
    cell: Cell[Array] = Cell()
    virial: Matrix3[Array] = Matrix3()
    moments: Meta[Array] = Meta(shape=(9,))


def test_two_d_metadata_batch_rows_equal_the_frame_read(tmp_path: Path):
    path = frames_file(
        tmp_path / "square.xyz",
        "Lattice=[[5,1,0],[0,5,2],[0,0,5]] virial=[[1.5,2,3],[4,5,6],[7,8,9]] "
        "moments=[[1,2,3],[4,5,6],[7,8,9]]",
        'Lattice="6 1 0 0 6 2 0 0 6" virial="2.5 2 3 4 5 6 7 8 9" '
        'moments="1 2 3 4 5 6 7 8 9"',
    )
    target: Any = RecordBatch[Squares]
    b = oxyz.read_batch(path, record=target)
    recs = oxyz.read(path, record=Squares)
    assert list(b) == recs
    np.testing.assert_array_equal(b.cell[0], [[5, 1, 0], [0, 5, 2], [0, 0, 5]])
    np.testing.assert_array_equal(
        b.virial[0], [[1.5, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]]
    )
    np.testing.assert_array_equal(b.moments[0], np.arange(1.0, 10.0))
    for kwargs in SEQUENTIAL_AND_PLANNED:
        windows = oxyz.iread_batch(path, record=target, frames_per_batch=1, **kwargs)
        assert [w[0] for w in windows] == recs


def test_from_batch_equals_read_batch():
    plain = oxyz.read_batch(MACE, schema=MaceSample.spec())
    built = MaceBatch.from_batch(plain)
    assert built == oxyz.read_batch(MACE, record=MaceBatch)
    generic = batch_class(RecordBatch[MaceSample])
    assert generic.from_batch(plain) == oxyz.read_batch(
        MACE, record=RecordBatch[MaceSample]
    )
    for name in ("positions", "forces", "energy", "stress", "cell"):
        assert not built[name].flags.writeable, name
    assert not built.offsets.flags.writeable
    assert not built.frame_indices.flags.writeable
    assert np.shares_memory(built.positions, plain.columns["pos"])
    pos = plain.columns["pos"]
    assert isinstance(pos, np.ndarray)
    pos[0, 0] = 99.0
    assert built.positions[0, 0] == 99.0


def test_from_batch_promotes_integers_unless_strict():
    plain = oxyz.read_batch(MACE)
    lattice = plain.metadata["Lattice"]
    assert isinstance(lattice, np.ndarray)
    assert lattice.dtype == np.int64
    built = MaceBatch.from_batch(plain)
    assert built.cell.dtype == np.float64
    assert not built.cell.flags.writeable
    np.testing.assert_array_equal(built.cell[0], 5 * np.eye(3))

    class StrictMace(MaceSample, strict=True):
        pass

    with pytest.raises(SchemaError, match="strict") as e:
        batch_class(RecordBatch[StrictMace]).from_batch(plain)
    assert (e.value.name, e.value.frame_index) == ("Lattice", None)


def without(batch: Batch, key: str) -> Batch:
    return dataclasses.replace(
        batch, metadata={k: v for k, v in batch.metadata.items() if k != key}
    )


def test_from_batch_without_an_optional_key_is_filled_and_absent():
    plain = oxyz.read_batch(MACE)
    assert MaceBatch.from_batch(plain).present("stress").tolist() == [True]
    b = MaceBatch.from_batch(without(plain, "REF_stress"))
    assert b.present("stress").tolist() == [False]
    assert b.stress.shape == (1, 6)
    assert np.isnan(b.stress).all()
    assert not b.stress.flags.writeable
    assert b[0].stress is None

    class Counted(Record):
        count: Meta[int | None] = Meta()

    class Filled(Record):
        count: Meta[int | None] = Meta(fill=-1)
        charge: Meta[int] = Meta(fill=0)

    with pytest.raises(SchemaError, match="count") as e:
        batch_class(RecordBatch[Counted]).from_batch(plain)
    assert (e.value.name, e.value.frame_index) == ("count", None)
    filled = batch_class(RecordBatch[Filled]).from_batch(plain)
    assert filled[Filled.count].tolist() == [-1]
    assert filled[Filled.charge].tolist() == [0]
    assert filled.present("count").tolist() == [False]
    assert (filled[0].count, filled[0].charge) == (None, 0)
    counted = dataclasses.replace(
        plain, metadata={"count": np.array([4]), "charge": np.array([2])}
    )
    filled = batch_class(RecordBatch[Filled]).from_batch(counted)
    assert (filled[0].count, filled[0].charge) == (4, 2)
    assert filled.present("count").tolist() == [True]


def test_from_batch_without_a_required_key_names_it():
    with pytest.raises(SchemaError, match=r"^MaceBatch: metadata 'REF_energy'") as e:
        MaceBatch.from_batch(without(oxyz.read_batch(MACE), "REF_energy"))
    assert (e.value.name, e.value.frame_index) == ("REF_energy", None)


FROM_BATCH_ERRORS = [
    ("float32-column", "columns", "pos", lambda v: v.astype(np.float32)),
    ("float32-metadata", "metadata", "REF_energy", lambda v: v.astype(np.float32)),
    ("column-rows", "columns", "REF_forces", lambda v: v[:2]),
    ("column-width", "columns", "REF_forces", lambda v: v[:, :2]),
    ("metadata-rows", "metadata", "REF_energy", lambda v: np.concatenate([v, v])),
    ("metadata-shape", "metadata", "REF_stress", lambda v: v[:, :5]),
    ("cell-rows", "metadata", "Lattice", lambda v: v.reshape(3, 3)),
    ("wrong-kind", "metadata", "REF_energy", lambda v: v.astype(np.bool_)),
    ("string-rows", "metadata", "config_type", lambda v: v * 2),
]


@pytest.mark.parametrize(
    ("side", "key", "change"),
    [pytest.param(*case, id=case_id) for case_id, *case in FROM_BATCH_ERRORS],
)
def test_from_batch_rejects_a_value_that_does_not_fit(side: str, key: str, change: Any):
    plain = oxyz.read_batch(MACE)
    stored = dict(getattr(plain, side))
    stored[key] = change(stored[key])
    bad = dataclasses.replace(plain, **{side: stored})
    with pytest.raises(SchemaError, match=rf"^MaceBatch: \w+ '{key}': ") as e:
        MaceBatch.from_batch(bad)
    assert (e.value.name, e.value.frame_index) == (key, None)


def test_spec_and_from_batch_belong_to_a_batch_of_a_record():
    assert MaceBatch.spec() == MaceSample.spec()
    assert batch_class(MAD_BATCH).spec() == MadSample.spec()
    with pytest.raises(TypeError, match=r"RecordBatch\[R\]"):
        RecordBatch.spec()
    with pytest.raises(TypeError, match=r"RecordBatch\[R\]"):
        RecordBatch.from_batch(oxyz.read_batch(MACE))


def test_read_batch_fields_are_writable_and_unshared():
    batches = [
        oxyz.read_batch(MACE, record=MaceBatch),
        *oxyz.iread_batch(MACE, record=MaceBatch, frames_per_batch=1, threads=1),
        *oxyz.iread_batch(MACE, record=MaceBatch, frames_per_batch=1),
    ]
    assert len(batches) == 3
    for b in batches:
        arrays = [b.positions, b.forces, b.energy, b.stress, b.cell]
        assert all(a.flags.writeable for a in arrays)
        assert not any(
            np.shares_memory(a, c)
            for i, a in enumerate(arrays)
            for c in arrays[i + 1 :]
        )
    assert not np.shares_memory(batches[0].positions, batches[1].positions)


def test_wrong_target_kind_is_a_type_error():
    form = re.escape("RecordBatch[MaceSample]")
    with pytest.raises(TypeError, match=form):
        oxyz.read_batch(MACE, record=MaceSample)  # ty: ignore[no-matching-overload]
    with pytest.raises(TypeError, match=form):
        oxyz.iread_batch(MACE, record=MaceSample, frames_per_batch=1)  # ty: ignore[no-matching-overload]
    for target in (MaceBatch, RecordBatch[MaceSample]):
        with pytest.raises(TypeError, match="record=MaceSample"):
            oxyz.read(MACE, record=target)  # ty: ignore[no-matching-overload]
        with pytest.raises(TypeError, match="record=MaceSample"):
            oxyz.iread(MACE, record=target)  # ty: ignore[no-matching-overload]
    for target in (int, RecordBatch, Record, make_batch(MaceBatch)):
        with pytest.raises(TypeError, match="record= on a batch read takes"):
            oxyz.read_batch(MACE, record=target)  # ty: ignore[no-matching-overload]


def test_the_generic_form_and_its_class_share_a_binding():
    derived = batch_class(RecordBatch[MaceSample])
    assert bind(RecordBatch[MaceSample]) is bind(derived)
    assert (bind(derived).target, bind(derived).record) == (derived, MaceSample)
    assert (bind(MaceBatch).target, bind(MaceBatch).record) == (MaceBatch, MaceSample)
    assert bind(MaceBatch).spec == MaceSample.spec()


def test_mode_with_a_batch_record_is_a_value_error():
    message = "mode= cannot be combined with record=: a record read always projects"
    for mode in ("project", "validate"):
        with pytest.raises(ValueError, match=re.escape(message)):
            oxyz.read_batch(MACE, record=MaceBatch, mode=mode)  # ty: ignore[invalid-argument-type]
        with pytest.raises(ValueError, match=re.escape(message)):
            oxyz.iread_batch(MACE, record=MaceBatch, mode=mode, frames_per_batch=1)  # ty: ignore[invalid-argument-type]


def test_a_disagreeing_schema_fails_before_the_file_is_opened(tmp_path: Path):
    schema = SchemaSpec(metadata=(oxyz.MetadataRule("REF_energy", Kind.STR),))
    bad = unparseable(tmp_path)
    with pytest.raises(SchemaError, match=r"^MaceSample\.energy: ") as e:
        oxyz.read_batch(bad, record=MaceBatch, schema=schema)
    assert (e.value.name, e.value.frame_index) == ("REF_energy", None)
    with pytest.raises(SchemaError, match=r"^MaceSample\.energy: "):
        oxyz.iread_batch(bad, record=MaceBatch, schema=schema, frames_per_batch=1)


class Padded(Record):
    energy: Meta[float] = Meta()
    padding: Meta[float] = Meta("nope", fill=0.0)
    stress: Matrix3[Array | None] = Matrix3()


def test_presence_follows_each_optional_fields_report_column(whole: Any):
    b = oxyz.read_batch(MAD, record=RecordBatch[Padded])
    assert b.present("stress").tolist() == whole.present("stress").tolist()
    assert b[Padded.padding].tolist() == [0.0] * 201
    assert [r.stress is None for r in b] == [r.stress is None for r in whole]


class Energetic(Record):
    energy: Meta[float] = Meta()


LATTICE_REQUIRED = SchemaSpec(frame=FrameRule(lattice_required=True))


def messages(caught: list[warnings.WarningMessage]) -> list[str]:
    return [str(w.message) for w in caught]


@pytest.mark.parametrize(("record", "missing"), [(MadSample, 49), (Energetic, 201)])
def test_a_lattice_frame_rule_agrees_with_the_frame_read(
    record: type[Record], missing: int
):
    target: Any = RecordBatch[record]  # ty: ignore[invalid-type-form]
    with pytest.raises(SchemaError) as framed:
        oxyz.read(MAD, record=record, schema=LATTICE_REQUIRED)
    with pytest.raises(SchemaError) as batched:
        oxyz.read_batch(MAD, record=target, schema=LATTICE_REQUIRED)
    assert (batched.value.frame_index, batched.value.name) == (0, "Lattice")
    assert str(batched.value) == str(framed.value)
    with warnings.catch_warnings(record=True) as frame_warnings:
        warnings.simplefilter("always")
        recs = oxyz.read(
            MAD, record=record, schema=LATTICE_REQUIRED, conformance="warn"
        )
    assert len(frame_warnings) == missing
    with warnings.catch_warnings(record=True) as batch_warnings:
        warnings.simplefilter("always")
        b = oxyz.read_batch(
            MAD, record=target, schema=LATTICE_REQUIRED, conformance="warn"
        )
    assert messages(batch_warnings) == messages(frame_warnings)
    assert list(b) == recs
    for kwargs in SEQUENTIAL_AND_PLANNED:
        with warnings.catch_warnings(record=True) as window_warnings:
            warnings.simplefilter("always")
            windows = list(
                oxyz.iread_batch(
                    MAD,
                    record=target,
                    schema=LATTICE_REQUIRED,
                    conformance="warn",
                    frames_per_batch=50,
                    **kwargs,
                )
            )
        assert messages(window_warnings) == messages(frame_warnings)
        assert [r for w in windows for r in w] == recs


def test_an_atom_count_frame_rule_on_a_batch_read_raises_with_context():
    one_atom = SchemaSpec(frame=FrameRule(n_atoms_max=1))
    with pytest.raises(SchemaError, match="n_atoms") as e:
        oxyz.read_batch(MAD, record=MAD_BATCH, schema=one_atom)
    assert (e.value.frame_index, e.value.name) == (1, "n_atoms")
    with pytest.raises(SchemaError) as e:
        list(
            oxyz.iread_batch(MAD, record=MAD_BATCH, schema=one_atom, frames_per_batch=7)
        )
    assert (e.value.frame_index, e.value.name) == (1, "n_atoms")


def test_a_remote_source_reads_what_the_local_file_gives(s3_store: Any, whole: Any):
    s3_store.put("mad.extxyz", MAD.read_bytes())
    url, options = s3_store.url("mad.extxyz"), s3_store.options
    assert oxyz.read_batch(url, record=MAD_BATCH, storage_options=options) == whole
    streamed = list(
        oxyz.iread_batch(
            url, record=MAD_BATCH, frames_per_batch=50, storage_options=options
        )
    )
    assert [b.frame_indices[0] for b in streamed] == [0, 50, 100, 150, 200]
    assert all(b == whole[b.frame_indices] for b in streamed)
