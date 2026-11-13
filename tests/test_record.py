from __future__ import annotations

import copy
import dataclasses
import inspect
import math
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
    Cell,
    Column,
    ColumnRule,
    Frame,
    FrameRule,
    Matrix3,
    Meta,
    MetadataRule,
    Record,
    SchemaError,
    SchemaSpec,
)
from oxyz._record_bind import bind
from oxyz._record_fields import FieldType
from oxyz._schema import Kind

DATA = Path(__file__).parent / "data"
MACE = DATA / "mace_ref_energy_forces_stress.xyz"
MAD = DATA / "mad_r2scan_sample.extxyz"

type Array = npt.NDArray[np.float64]
type Forces = Annotated[Column[Array], Column("REF_forces", width=3)]
type Energy = Annotated[Meta[float | None], Meta("E", fill=0.0)]


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


MACE_FIELDS = ["positions", "forces", "energy", "stress", "cell", "config_type"]
RESERVED = [
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
]


def make_mace(**overrides: Any) -> MaceSample:
    values: dict[str, Any] = {
        "n_atoms": 3,
        "positions": np.arange(9.0).reshape(3, 3),
        "forces": np.ones((3, 3)),
        "energy": -76.123,
        "stress": None,
        "cell": np.eye(3),
        "config_type": "Default",
    }
    return MaceSample(**(values | overrides))


def define(body: str, *, postponed: bool = True) -> type[Any]:
    """Exec `class Bad(Record):` with `body` here; return `Bad`.

    Annotations resolve against this module's globals. Without `postponed`
    the body is compiled without `from __future__ import annotations`.
    """
    source = "class Bad(Record):\n" + "".join(
        f"    {line}\n" for line in body.splitlines()
    )
    namespace = dict(globals())
    code = compile(source, "<record>", "exec", dont_inherit=not postponed)
    exec(code, namespace)
    return namespace["Bad"]


def test_field_key_is_bound_to_name_and_record():
    key = MaceSample.energy
    assert isinstance(key, Meta)
    assert (key.name, key.owner, key.key) == ("energy", MaceSample, "REF_energy")
    assert MaceSample.config_type.key == "config_type"
    assert (MaceSample.forces.key, MaceSample.forces.width) == ("REF_forces", 3)
    assert MadSample.cell.key == "Lattice"


def test_declared_specifier_is_left_unbound():
    spec = Meta()

    class One(Record):
        energy: Meta[float] = spec

    assert One.energy is not spec
    assert (spec.name, spec.owner, spec.key) == (None, None, None)


def test_inherited_key_is_the_base_object():
    class Sub(MaceSample):
        extra: Meta[int] = Meta()

    assert Sub.energy is MaceSample.energy
    assert (Sub.extra.name, Sub.extra.owner) == ("extra", Sub)


def test_redeclared_field_stays_a_required_slot_with_its_own_key():
    class Sub(MaceSample):
        energy: Meta[float] = Meta("other")

    assert (Sub.energy.key, MaceSample.energy.key) == ("other", "REF_energy")
    assert type(inspect.getattr_static(Sub, "energy")).__name__ == "member_descriptor"
    energy = inspect.signature(Sub).parameters["energy"]
    assert energy.default is inspect.Parameter.empty


def test_records_sharing_an_alias_get_distinct_keys():
    class Other(Record):
        forces: Forces

    assert Other.forces is not MaceSample.forces
    assert Other.forces.key == "REF_forces"
    assert (Other.forces.owner, MaceSample.forces.owner) == (Other, MaceSample)


def test_record_fields_list_every_field_in_declaration_order():
    class Sub(MaceSample):
        extra: Meta[int] = Meta()

    names = [name for name, _, _ in Sub.__record_fields__]
    assert names == [*MACE_FIELDS, "extra"]
    fields = {name: (ft, key) for name, ft, key in Sub.__record_fields__}
    assert fields["energy"] == (FieldType(Kind.REAL, "metadata"), MaceSample.energy)
    assert fields["forces"][0] == FieldType(Kind.REAL, "column", width=3)
    assert fields["stress"][0] == FieldType(
        Kind.REAL, "metadata", shape=(6,), optional=True
    )
    assert fields["extra"][1] is Sub.extra
    assert set(Sub.__record_keys__) == {*MACE_FIELDS, "extra"}
    assert (Record.__record_fields__, Record.__record_keys__) == ((), {})


def test_class_attribute_is_the_slot():
    for name in MACE_FIELDS:
        found = inspect.getattr_static(MaceSample, name)
        assert type(found).__name__ == "member_descriptor"


def test_every_field_is_a_required_keyword():
    parameters = inspect.signature(MaceSample).parameters.values()
    assert [p.name for p in parameters] == ["n_atoms", *MACE_FIELDS]
    assert {p.kind for p in parameters} == {inspect.Parameter.KEYWORD_ONLY}
    assert {p.default for p in parameters} == {inspect.Parameter.empty}
    with pytest.raises(TypeError, match="energy"):
        MaceSample(
            n_atoms=3,
            positions=np.zeros((3, 3)),
            forces=np.zeros((3, 3)),
            stress=None,
            cell=np.eye(3),
            config_type="x",
        )


def test_init_stores_every_field():
    r = make_mace()
    assert r.n_atoms == 3
    assert r.energy == -76.123
    assert r.config_type == "Default"
    assert r.stress is None
    np.testing.assert_array_equal(r.positions, np.arange(9.0).reshape(3, 3))


def test_instances_are_slotted_and_frozen():
    r = make_mace()
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.energy = 0.0  # ty: ignore[invalid-assignment]
    assert not hasattr(r, "__dict__")
    with pytest.raises(AttributeError):
        object.__setattr__(r, "stray", 1)


def test_pickle_copy_and_replace_round_trip():
    r = make_mace()
    restored = pickle.loads(pickle.dumps(r))
    assert type(restored) is MaceSample
    assert (restored.n_atoms, restored.energy, restored.config_type) == (
        3,
        -76.123,
        "Default",
    )
    np.testing.assert_array_equal(restored.positions, r.positions)
    shallow = copy.copy(r)
    assert shallow.positions is r.positions
    assert shallow.energy == r.energy
    replaced = dataclasses.replace(r, energy=1.0)
    assert (replaced.energy, replaced.config_type) == (1.0, "Default")


def test_zero_argument_super_works_in_a_record_method():
    class Base(Record):
        energy: Meta[float] = Meta()

        def describe(self) -> str:
            return f"energy {self.energy}"

    class Derived(Base):
        def describe(self) -> str:
            return "derived " + super().describe()

    assert Derived(n_atoms=1, energy=2.0).describe() == "derived energy 2.0"


def test_init_subclass_runs_once_per_class():
    calls: list[str] = []

    class Base(Record):
        def __init_subclass__(cls, **kwargs: Any) -> None:
            super().__init_subclass__(**kwargs)
            calls.append(cls.__name__)

    class Sub(Base):
        energy: Meta[float] = Meta()

    assert calls == ["Sub"]
    assert Sub.energy.owner is Sub


def test_strict_is_inherited():
    class S(Record, strict=True):
        e: Meta[float] = Meta()

    class T(S):
        pass

    class U(S, strict=False):
        pass

    assert (S.__record_strict__, T.__record_strict__, U.__record_strict__) == (
        True,
        True,
        False,
    )
    assert (MaceSample.__record_strict__, Record.__record_strict__) == (False, False)


def test_unknown_class_keyword_is_rejected():
    with pytest.raises(TypeError, match="takes no keyword arguments"):
        type("X", (Record,), {}, extra="forbid")


def test_same_key_on_two_axes_is_allowed():
    cls = define('a: Column[Array] = Column("x")\nb: Meta[float] = Meta("x")')
    assert (cls.a.key, cls.b.key) == ("x", "x")


def test_record_declared_without_postponed_annotations():
    cls = define(
        'energy: Meta[float] = Meta("E")\nforces: Forces',
        postponed=False,
    )
    assert [name for name, _, _ in cls.__record_fields__] == ["energy", "forces"]
    assert cls.energy.key == "E"
    assert cls(n_atoms=1, energy=1.0, forces=np.zeros((1, 3))).energy == 1.0


def test_local_alias_is_a_definition_error_naming_the_field():
    def define_local() -> None:
        type Local = npt.NDArray[np.float64]

        class Bad(Record):
            pos: Column[Local] = Column(width=3)

    with pytest.raises(TypeError, match=r"^Bad\.pos: .*Local"):
        define_local()


DEFINITION_ERRORS = [
    ("unsupported", "energy: Meta[complex] = Meta()", "energy", "not a supported"),
    ("plain-annotation", "energy: float = Meta()", "energy", r"is not Meta\["),
    ("mismatched", "energy: Meta[float] = Column()", "energy", r"is not Column\["),
    ("unevaluable", "energy: Meta[float, int] = Meta()", "energy", "arguments"),
    ("undefined", "energy: Meta[Undefined] = Meta()", "energy", "Undefined"),
    ("no-specifier", "energy: Meta[float]", "energy", "no field specifier"),
    ("plain-default", "energy: Meta[float] = 1.0", "energy", "not a field specifier"),
    ("both", "energy: Annotated[Meta[float], Meta()] = Meta()", "energy", "both"),
    ("two", "energy: Annotated[Meta[float], Meta(), Meta()]", "energy", "more than"),
    ("width", "labels: Column[list[str]] = Column(width=3)", "labels", "width=3"),
    ("fill", 'energy: Meta[float] = Meta(fill="x")', "energy", "fill 'x'"),
    ("glob-key", 'energy: Meta[float] = Meta("E*")', "energy", "'E\\*' would be"),
    ("regex-key", 'pos: Column[Array] = Column("re:p.s")', "pos", "'re:p.s' would be"),
    (
        "duplicate-key",
        'a: Meta[float] = Meta("e")\nb: Meta[float] = Meta("e")',
        "b",
        "'e' is also read by a",
    ),
    ("unannotated", "step: Meta[int] = Meta()\nenergy = Meta()", "energy", "no annot"),
    ("underscore", "_trusted: Meta[float] = Meta()", "_trusted", "underscore"),
    ("private", "_energy: Meta[float] = Meta()", "_energy", "underscore"),
    *(
        (f"reserved-{name}", f"{name}: Meta[float] = Meta()", name, "reserved")
        for name in RESERVED
    ),
]


@pytest.mark.parametrize(
    ("body", "field", "reason"),
    [pytest.param(*case, id=case_id) for case_id, *case in DEFINITION_ERRORS],
)
def test_definition_errors(body: str, field: str, reason: str):
    with pytest.raises(TypeError, match=rf"^Bad\.{field}: .*{reason}"):
        define(body)


def test_strict_must_be_a_bool():
    with pytest.raises(TypeError, match=r"^S: strict must be a bool, got 'yes'"):
        type("S", (Record,), {}, strict="yes")


def test_init_keeps_the_arrays_it_is_given():
    pos = np.zeros((3, 3))
    r = make_mace(positions=pos)
    assert r.positions is pos
    assert pos.flags.writeable


@pytest.mark.parametrize(
    ("override", "error", "reason"),
    [
        ({"positions": np.zeros((3, 3), dtype=np.float32)}, TypeError, "dtype"),
        ({"positions": np.zeros((3, 2))}, ValueError, "shape"),
        ({"positions": np.zeros((4, 3))}, ValueError, "expected 3 rows"),
        ({"forces": [[0.0] * 3] * 3}, TypeError, "numpy array"),
        ({"energy": None}, TypeError, "not optional"),
        ({"energy": "x"}, TypeError, "expected float"),
        ({"energy": np.True_}, TypeError, "expected float"),
        ({"stress": np.zeros(3)}, ValueError, "shape"),
        ({"cell": np.zeros(9)}, ValueError, "shape"),
        ({"config_type": 1}, TypeError, "expected str"),
        ({"n_atoms": "3"}, TypeError, "expected int"),
        ({"n_atoms": True}, TypeError, "expected int"),
        ({"n_atoms": -1}, ValueError, "negative"),
    ],
)
def test_init_checks_name_the_field(override, error, reason):
    (field,) = override
    with pytest.raises(error, match=rf"^MaceSample\.{field}: .*{reason}"):
        make_mace(**override)


def test_init_accepts_numpy_scalars_and_keeps_them():
    class Counts(Record):
        total: Meta[int] = Meta()
        mean: Meta[float] = Meta()

    total: Any = np.arange(3).sum()
    r = Counts(n_atoms=total, total=total, mean=total)
    assert r.total is total
    assert r.mean is total


def test_dataclass_replace_is_checked():
    with pytest.raises(TypeError, match=r"^MaceSample\.energy: "):
        dataclasses.replace(make_mace(), energy="x")


def test_trusted_skips_the_checks():
    assert MaceSample._trusted(n_atoms=3, energy="not a float").energy == "not a float"


def test_trusted_sets_every_value_given():
    r = make_mace()
    values = {name: getattr(r, name) for name in ["n_atoms", *MACE_FIELDS]}
    trusted = MaceSample._trusted(**values)
    assert type(trusted) is MaceSample
    assert all(getattr(trusted, name) is value for name, value in values.items())


def test_spec_is_the_project_mode_schema_and_round_trips():
    spec = MaceSample.spec()
    assert spec == SchemaSpec(
        mode="project",
        columns=(
            ColumnRule("pos", Kind.REAL, width=3),
            ColumnRule("REF_forces", Kind.REAL, width=3),
        ),
        metadata=(
            MetadataRule("REF_energy", Kind.REAL),
            MetadataRule("REF_stress", Kind.REAL, shape=(6,), required=False),
            MetadataRule("Lattice", Kind.REAL, shape=(9,)),
            MetadataRule("config_type", Kind.STR),
        ),
    )
    assert SchemaSpec.from_yaml(spec.to_yaml()) == spec


def test_spec_carries_fills():
    class Filled(Record):
        labels: Column[list[str]] = Column(fill="X")
        charge: Meta[int | None] = Meta(fill=0)
        cell: Cell[Array] = Cell(fill=0.0)
        virial: Matrix3[Array | None] = Matrix3()
        label: Meta[str | None] = Meta()

    spec = Filled.spec()
    assert spec == SchemaSpec(
        mode="project",
        columns=(ColumnRule("labels", Kind.STR, required=False, fill="X"),),
        metadata=(
            MetadataRule("charge", Kind.INT, required=False, fill=0),
            MetadataRule("Lattice", Kind.REAL, shape=(9,), required=False, fill=0.0),
            MetadataRule("virial", Kind.REAL, shape=(9,), required=False),
            MetadataRule("label", Kind.STR, required=False, absent="omit"),
        ),
    )
    assert SchemaSpec.from_yaml(spec.to_yaml()) == spec


def test_a_plain_batch_read_with_the_spec_fills_as_the_record_batch_does():
    plain = oxyz.read_batch(MAD, schema=MadSample.spec())
    typed = oxyz.read_batch(MAD, record=oxyz.RecordBatch[MadSample])
    lattice, stress = (np.asarray(plain.metadata[k]) for k in ("Lattice", "stress"))
    np.testing.assert_array_equal(plain.metadata["energy"], typed[MadSample.energy])
    np.testing.assert_array_equal(lattice.reshape(-1, 3, 3), typed[MadSample.cell])
    np.testing.assert_array_equal(
        stress.reshape(-1, 3, 3).transpose(0, 2, 1), typed[MadSample.stress]
    )


def test_declaration_forms_give_identical_specs():
    class Default(Record):
        forces: Column[Array] = Column("REF_forces", width=3)
        energy: Meta[float | None] = Meta("E", fill=0.0)

    class InAnnotated(Record):
        forces: Annotated[Column[Array], Column("REF_forces", width=3)]
        energy: Annotated[Meta[float | None], Meta("E", fill=0.0)]

    class ViaAlias(Record):
        forces: Forces
        energy: Energy

    class AliasAgain(Record):
        forces: Forces
        energy: Energy

    specs = [cls.spec() for cls in (Default, InAnnotated, ViaAlias, AliasAgain)]
    assert specs[0].metadata == (
        MetadataRule("E", Kind.REAL, required=False, fill=0.0),
    )
    assert all(spec == specs[0] for spec in specs)


def mace_frame(
    columns: dict[str, Any] | None = None, metadata: dict[str, Any] | None = None
) -> Frame:
    """The MACE fixture's first frame with entries replaced, or removed by `None`."""
    frame = oxyz.read(MACE, 0)

    def merged(stored: dict[str, Any], changes: dict[str, Any] | None) -> dict:
        both = {**stored, **(changes or {})}
        return {key: value for key, value in both.items() if value is not None}

    return Frame(
        n_atoms=frame.n_atoms,
        columns=merged(frame.columns, columns),
        metadata=merged(frame.metadata, metadata),
    )


def test_from_frame_matches_the_frame():
    frame = oxyz.read(MACE, 0)
    r = MaceSample.from_frame(frame)
    assert type(r) is MaceSample
    assert (r.energy, r.n_atoms, r.config_type) == (-76.123, 3, "Default")
    np.testing.assert_array_equal(r.cell, frame.cell)
    assert isinstance(r.cell, np.ndarray)
    assert r.cell.dtype == np.float64
    np.testing.assert_array_equal(r.forces, frame.columns["REF_forces"])
    np.testing.assert_array_equal(r.stress, frame.metadata["REF_stress"])
    assert np.shares_memory(r.positions, frame.columns["pos"])
    assert not r.positions.flags.writeable
    assert not r.cell.flags.writeable
    assert r.stress is not None
    assert not r.stress.flags.writeable
    frame.positions[0, 0] = 9.0
    assert r.positions[0, 0] == 9.0


def test_from_frame_lax_promotes_integers_read_only():
    class Charged(Record):
        energy: Meta[float] = Meta("REF_energy")
        charges: Column[Array] = Column()

    frame = Frame(
        n_atoms=2,
        columns={"charges": np.array([1, -1])},
        metadata={"REF_energy": 0},
    )
    r = Charged.from_frame(frame)
    assert type(r.energy) is float
    assert r.energy == 0.0
    assert r.charges.dtype == np.float64
    assert not r.charges.flags.writeable


def test_from_frame_reads_two_dimensional_metadata_flattened():
    square = np.arange(9.0).reshape(3, 3)
    r = MaceSample.from_frame(mace_frame(metadata={"Lattice": square}))
    np.testing.assert_array_equal(r.cell, square)


class Named(Record):
    names: Meta[list[str]] = Meta(shape=(4,))


def test_two_d_string_metadata_reads_flattened_row_major(tmp_path: Path):
    path = frames_file(tmp_path / "names.xyz", 'names=[["a","b"],["c","d"]]')
    r = oxyz.read(path, 0, record=Named)
    assert r.names == ["a", "b", "c", "d"]
    assert Named(n_atoms=1, names=r.names) == r
    assert Named.from_frame(oxyz.read(path, 0)) == r
    assert list(oxyz.iread(path, record=Named)) == [r]


class StrictMace(MaceSample, strict=True):
    pass


class IntCounts(Record):
    counts: Column[npt.NDArray[np.int64]] = Column()


class Charges(Record):
    charges: Column[Array] = Column()


FROM_FRAME_ERRORS = [
    ("missing", MaceSample, None, {"REF_energy": None}, "REF_energy", "missing"),
    ("bool-real", MaceSample, None, {"REF_energy": True}, "REF_energy", "found L"),
    ("int-str", MaceSample, None, {"config_type": 1}, "config_type", "found I"),
    ("shape", MaceSample, None, {"REF_stress": np.zeros(3)}, "REF_stress", r"R\[3\]"),
    (
        "flat-length",
        MaceSample,
        None,
        {"Lattice": np.zeros((2, 2))},
        "Lattice",
        r"R\[4\]",
    ),
    ("strict-array", StrictMace, None, None, "Lattice", r"found I\[9\].*strict"),
    ("strict-scalar", StrictMace, None, {"REF_energy": 0}, "REF_energy", "I.*strict"),
    ("zero-d", MaceSample, None, {"REF_energy": np.array(1.0)}, "REF_energy", "0-d"),
    ("width", MaceSample, {"REF_forces": np.zeros(3)}, None, "REF_forces", "R:1"),
    ("absent-column", MaceSample, {"REF_forces": None}, None, "REF_forces", "missing"),
    (
        "float32",
        MaceSample,
        {"pos": np.zeros((3, 3), np.float32)},
        None,
        "pos",
        "dtype float64, got float32",
    ),
    (
        "uint8",
        IntCounts,
        {"counts": np.zeros(3, np.uint8)},
        None,
        "counts",
        "dtype int64, got uint8",
    ),
    ("rows", MaceSample, {"pos": np.zeros((5, 3))}, None, "pos", "expected 3 rows"),
    ("ndim", Charges, {"charges": np.zeros((3, 3, 3))}, None, "charges", "shape"),
]


@pytest.mark.parametrize(
    ("record", "columns", "metadata", "key", "reason"),
    [pytest.param(*case, id=case_id) for case_id, *case in FROM_FRAME_ERRORS],
)
def test_from_frame_errors_carry_the_key_and_no_frame_index(
    record, columns, metadata, key, reason
):
    frame = mace_frame(columns, metadata)
    pattern = rf"^{record.__name__}: \w+ '{key}': .*{reason}"
    with pytest.raises(SchemaError, match=pattern) as e:
        record.from_frame(frame)
    assert (e.value.name, e.value.frame_index) == (key, None)


@pytest.mark.parametrize(
    ("columns", "metadata", "key"),
    [
        ({"REF_forces": np.empty((3, 3), dtype=object)}, None, "REF_forces"),
        (None, {"REF_stress": np.empty(6, dtype=object)}, "REF_stress"),
    ],
    ids=["column", "metadata"],
)
def test_from_frame_rejects_an_object_array_naming_its_key(columns, metadata, key):
    frame = mace_frame(columns, metadata)
    with pytest.raises(SchemaError, match=rf"^MaceSample: \w+ '{key}': .*object"):
        MaceSample.from_frame(frame)


@pytest.mark.parametrize(
    ("n_atoms", "error", "reason"),
    [
        (1.0, TypeError, "expected int, got float"),
        (True, TypeError, "got bool"),
        (-1, ValueError, "negative"),
    ],
)
def test_from_frame_checks_the_atom_count(
    n_atoms: Any, error: type[Exception], reason: str
):
    frame = Frame(n_atoms=n_atoms, columns={}, metadata={"step": 0, "energy": 1.0})
    with pytest.raises(error, match=rf"^Stepped\.n_atoms: .*{reason}"):
        Stepped.from_frame(frame)


def test_from_frame_copies_string_columns():
    class Species(Record):
        species: Column[list[str]] = Column()

    frame = oxyz.read(MACE, 0)
    r = Species.from_frame(frame)
    assert r.species == frame.columns["species"] == ["O", "H", "H"]
    assert r.species is not frame.columns["species"]


def test_from_frame_optional_and_fill():
    class Filled(Record):
        labels: Column[list[str]] = Column(fill="X")
        charge: Meta[int | None] = Meta(fill=0)
        cell: Cell[Array] = Cell(fill=0.0)

    class MaybeCell(Record):
        cell: Cell[Array | None] = Cell(fill=0.0)

    assert (
        MaceSample.from_frame(mace_frame(metadata={"REF_stress": None})).stress is None
    )
    empty = Frame(n_atoms=2, columns={}, metadata={})
    assert MaybeCell.from_frame(empty).cell is None
    r = Filled.from_frame(empty)
    assert r.labels == ["X", "X"]
    assert r.charge is None
    assert isinstance(r.cell, np.ndarray)
    np.testing.assert_array_equal(r.cell, np.zeros((3, 3)))
    assert not r.cell.flags.writeable


def test_repr_summarises_without_data():
    r = MaceSample.from_frame(oxyz.read(MACE, 0))
    assert repr(r) == (
        "MaceSample(n_atoms=3, positions=float64[3, 3], forces=float64[3, 3], "
        "energy=-76.123, stress=float64[6], cell=float64[3, 3], "
        "config_type='Default')"
    )
    assert "stress=None" in repr(make_mace())


def test_repr_elides_after_the_entry_limit():
    body = "\n".join(f"f{i}: Meta[int] = Meta()" for i in range(10))
    cls = define(body)
    r = cls(n_atoms=0, **{f"f{i}": i for i in range(10)})
    assert repr(r) == (
        "Bad(n_atoms=0, f0=0, f1=1, f2=2, f3=3, f4=4, f5=5, f6=6, f7=7, +2 more)"
    )


def test_equality_treats_nan_as_equal_and_records_are_unhashable():
    nan_stress = np.array([np.nan, 1.0, 2.0, 3.0, 4.0, 5.0])
    a = make_mace(stress=nan_stress, energy=float("nan"))
    b = make_mace(stress=nan_stress.copy(), energy=float("nan"))
    assert a == b
    assert a != make_mace(stress=nan_stress, energy=1.0)
    assert a != make_mace(energy=float("nan"))
    first, again = (MaceSample.from_frame(oxyz.read(MACE, 0)) for _ in range(2))
    assert first == again
    with pytest.raises(TypeError, match="unhashable"):
        hash(a)


def test_equality_compares_type_and_atom_count():
    class Other(MaceSample):
        pass

    r = make_mace()
    other = Other(**{name: getattr(r, name) for name in ["n_atoms", *MACE_FIELDS]})
    assert r != other
    assert r.__eq__(1) is NotImplemented

    class One(Record):
        e: Meta[float] = Meta()

    assert One(n_atoms=1, e=1.0) == One(n_atoms=1, e=1.0)
    assert One(n_atoms=1, e=1.0) != One(n_atoms=2, e=1.0)


def html_rows(markup: str) -> list[list[str]]:
    rows = re.findall(r"<tr>(.*?)</tr>", markup)
    return [re.findall(r"<td>(.*?)</td>", row) for row in rows if "<td>" in row]


def test_repr_html_lists_every_field():
    markup = make_mace()._repr_html_()
    assert "<strong>MaceSample</strong> · 3 atoms" in markup
    rows = {row[0]: row[1:] for row in html_rows(markup)}
    assert list(rows) == MACE_FIELDS
    assert rows["energy"] == ["float", "()", "-76.123"]
    assert rows["stress"] == ["—", "—", "None"]
    assert rows["positions"][:2] == ["float64", "(3, 3)"]
    assert rows["config_type"] == ["str", "()", "&#x27;Default&#x27;"]


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


class Stepped(Record):
    step: Meta[int] = Meta()
    energy: Meta[float] = Meta()


class TwoFrame(Record):
    species: Column[list[str]] = Column()
    positions: Column[Array] = Column("pos", width=3)
    forces: Column[Array] = Column(width=3)
    energy: Meta[float] = Meta()
    cell: Cell[Array] = Cell()


def test_every_index_form():
    whole = oxyz.read(MAD, record=MadSample)
    assert len(whole) == 201
    assert all(type(r) is MadSample for r in whole)
    assert oxyz.read(MAD, 0, record=MadSample) == whole[0]
    assert oxyz.read(MAD, -1, record=MadSample) == whole[-1]
    assert oxyz.read(MAD, "3", record=MadSample) == whole[3]
    assert oxyz.read(MAD, "0:10:2", record=MadSample) == whole[0:10:2]
    assert len(whole[0:10:2]) == 5
    assert oxyz.read(MAD, "::-40", record=MadSample) == whole[::-40]
    assert oxyz.read(MAD, slice(140, 160), record=MadSample) == whole[140:160]
    picked = oxyz.read(MAD, [120, 40, 120], record=MadSample)
    assert picked == [whole[120], whole[40], whole[120]]
    plain = oxyz.read(MAD, [120, 40, 120])
    assert [r.n_atoms for r in picked] == [f.n_atoms for f in plain] == [44, 28, 44]


def test_iread_yields_what_read_returns():
    whole = oxyz.read(MAD, record=MadSample)
    assert list(oxyz.iread(MAD, record=MadSample)) == whole
    assert list(oxyz.iread(MAD, 7, record=MadSample)) == [whole[7]]
    assert list(oxyz.iread(MAD, "140:150", record=MadSample)) == whole[140:150]
    assert list(oxyz.iread(MAD, "::-50", record=MadSample)) == whole[::-50]
    assert list(oxyz.iread(MAD, [150, 3], record=MadSample)) == [whole[150], whole[3]]


def test_read_equals_from_frame():
    assert oxyz.read(MAD, record=MadSample) == [
        MadSample.from_frame(f) for f in oxyz.read(MAD)
    ]


def test_threads_and_compression_parity():
    whole = oxyz.read(MAD, record=MadSample)
    assert oxyz.read(MAD, record=MadSample, threads=1) == whole
    plain = oxyz.read(DATA / "two_frame_same_schema.xyz", record=TwoFrame)
    assert [r.energy for r in plain] == [-6.45, -6.44]
    gz = DATA / "compressed" / "two_frame.xyz.gz"
    assert oxyz.read(gz, record=TwoFrame) == plain
    assert list(oxyz.iread(gz, record=TwoFrame)) == plain


def test_optional_fields_are_none_exactly_where_the_file_lacks_them():
    plain = oxyz.read(MAD)
    recs = oxyz.read(MAD, record=MadSample)
    stressed = ["stress" in f.metadata for f in plain]
    latticed = ["Lattice" in f.metadata for f in plain]
    assert (sum(stressed), sum(latticed)) == (26, 152)
    assert [r.stress is not None for r in recs] == stressed
    assert [r.cell is not None for r in recs] == latticed


def test_lax_reads_integer_lattice_and_matches_frame_cell(tmp_path: Path):
    r = oxyz.read(MACE, 0, record=MaceSample)
    assert r.cell.dtype == np.float64
    np.testing.assert_array_equal(r.cell, oxyz.read(MACE, 0).cell)
    zero = tmp_path / "zero.xyz"
    zero.write_text(MACE.read_text().replace("REF_energy=-76.123", "REF_energy=0"))
    r = oxyz.read(zero, 0, record=MaceSample)
    assert type(r.energy) is float
    assert r.energy == 0.0


def test_strict_rejects_an_integer_lattice_with_context():
    with pytest.raises(SchemaError, match="Lattice") as e:
        oxyz.read(MACE, record=StrictMace)
    assert (e.value.frame_index, e.value.name) == (0, "Lattice")
    with pytest.raises(SchemaError) as e:
        list(oxyz.iread(MACE, record=StrictMace))
    assert (e.value.frame_index, e.value.name) == (0, "Lattice")


def test_missing_required_key_has_frame_index_and_name(tmp_path: Path):
    class Nope(MadSample):
        nope: Meta[float] = Meta()

    with pytest.raises(SchemaError) as e:
        oxyz.read(MAD, record=Nope)
    assert (e.value.frame_index, e.value.name) == (0, "nope")
    path = frames_file(
        tmp_path / "third.xyz", "step=0 energy=1.0", "step=1 energy=2.0", "step=2"
    )
    with pytest.raises(SchemaError) as e:
        oxyz.read(path, record=Stepped)
    assert (e.value.frame_index, e.value.name) == (2, "energy")
    with pytest.raises(SchemaError) as e:
        list(oxyz.iread(path, record=Stepped))
    assert (e.value.frame_index, e.value.name) == (2, "energy")


def test_cell_requires_lattice_unless_filled():
    class Molecule(Record):
        positions: Column[Array] = Column("pos", width=3)
        cell: Cell[Array] = Cell()

    class FilledMolecule(Record):
        positions: Column[Array] = Column("pos", width=3)
        cell: Cell[Array] = Cell(fill=0.0)

    path = DATA / "no_lattice_molecule.xyz"
    with pytest.raises(SchemaError) as e:
        oxyz.read(path, record=Molecule)
    assert (e.value.frame_index, e.value.name) == (0, "Lattice")
    r = oxyz.read(path, 0, record=FilledMolecule)
    assert r.cell.dtype == np.float64
    np.testing.assert_array_equal(r.cell, np.zeros((3, 3)))
    np.testing.assert_array_equal(r.cell, oxyz.read(path, 0).cell)


class Stressed(Record):
    stress: Matrix3[Array] = Matrix3()


@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array:DeprecationWarning"
)
def test_matrix3_matches_ase():
    pytest.importorskip("ase")
    import oxyz.ase

    path = DATA / "stress_matrix9.extxyz"
    r = oxyz.read(path, 0, record=Stressed)
    expected = oxyz.ase.to_atoms(oxyz.read(path, 0)).get_stress(voigt=False)
    np.testing.assert_array_equal(r.stress, expected)


def test_matrix3_reads_in_fortran_order(tmp_path: Path):
    path = frames_file(tmp_path / "asym.xyz", 'stress="1 2 3 4 5 6 7 8 9"')
    r = oxyz.read(path, 0, record=Stressed)
    np.testing.assert_array_equal(r.stress, np.arange(1.0, 10.0).reshape(3, 3).T)


def test_matrix3_reads_a_nested_3x3_as_written(tmp_path: Path):
    path = frames_file(
        tmp_path / "nested.xyz",
        "stress=[[1,2,3],[4,5,6],[7,8,9]]",
        'stress="1 4 7 2 5 8 3 6 9"',
    )
    written = np.arange(1.0, 10.0).reshape(3, 3)
    recs = oxyz.read(path, record=Stressed)
    for r in [
        *recs,
        *oxyz.iread(path, record=Stressed),
        oxyz.read(path, 0, record=Stressed),
    ]:
        np.testing.assert_array_equal(r.stress, written)
    np.testing.assert_array_equal(
        Stressed.from_frame(oxyz.read(path, 0)).stress, written
    )
    batch = oxyz.read_batch(path, record=oxyz.RecordBatch[Stressed])
    np.testing.assert_array_equal(batch[Stressed.stress], [written, written])
    assert list(batch) == recs


def test_warn_drops_a_frame_missing_a_required_real_field(tmp_path: Path):
    path = frames_file(tmp_path / "gap.xyz", "step=0 energy=-1.5", "step=1")
    with pytest.warns(oxyz.SchemaWarning, match="energy"):
        recs = oxyz.read(path, record=Stepped, conformance="warn")
    assert [(r.step, r.energy) for r in recs] == [(0, -1.5)]
    with pytest.warns(oxyz.SchemaWarning, match="energy"):
        assert list(oxyz.iread(path, record=Stepped, conformance="warn")) == recs


def test_warn_drops_a_wrong_kind_optional_field(tmp_path: Path):
    class MaybeStressed(Record):
        step: Meta[int] = Meta()
        stress: Matrix3[Array | None] = Matrix3()

    path = frames_file(
        tmp_path / "kinds.xyz",
        'step=0 stress="1 2 3 4 5 6 7 8 9"',
        "step=1 stress=abc",
        "step=2",
    )
    with pytest.warns(oxyz.SchemaWarning, match="stress"):
        recs = oxyz.read(path, record=MaybeStressed, conformance="warn")
    assert [r.step for r in recs] == [0, 2]
    assert recs[1].stress is None
    with pytest.raises(SchemaError) as e:
        oxyz.read(path, record=MaybeStressed)
    assert (e.value.frame_index, e.value.name) == (1, "stress")


def test_warn_drops_a_wrong_kind_field_that_has_a_fill(tmp_path: Path):
    class FilledEnergy(Record):
        step: Meta[int] = Meta()
        energy: Meta[float] = Meta(fill=0.0)

    path = frames_file(
        tmp_path / "filled.xyz", "step=0 energy=-1.5", "step=1 energy=abc", "step=2"
    )
    with pytest.warns(oxyz.SchemaWarning, match="energy"):
        recs = oxyz.read(path, record=FilledEnergy, conformance="warn")
    assert [(r.step, r.energy) for r in recs] == [(0, -1.5), (2, 0.0)]
    with pytest.warns(oxyz.SchemaWarning, match="energy"):
        assert list(oxyz.iread(path, record=FilledEnergy, conformance="warn")) == recs
    with pytest.warns(oxyz.SchemaWarning, match="energy"):
        batch = oxyz.read_batch(
            path, record=oxyz.RecordBatch[FilledEnergy], conformance="warn"
        )
    assert list(batch) == recs
    with pytest.raises(SchemaError) as e:
        FilledEnergy.from_frame(oxyz.read(path, 1))
    assert e.value.name == "energy"
    with pytest.raises(SchemaError) as e:
        oxyz.read(path, record=FilledEnergy)
    assert (e.value.frame_index, e.value.name) == (1, "energy")


def test_everything_dropped_is_an_empty_list(tmp_path: Path):
    path = frames_file(tmp_path / "none.xyz", "step=0", "step=1")
    with pytest.warns(oxyz.SchemaWarning):
        assert oxyz.read(path, record=Stepped, conformance="warn") == []
    with pytest.warns(oxyz.SchemaWarning):
        assert list(oxyz.iread(path, record=Stepped, conformance="warn")) == []
    with pytest.warns(oxyz.SchemaWarning), pytest.raises(IndexError):
        oxyz.read(path, 0, record=Stepped, conformance="warn")
    assert oxyz.read(MAD, "5:5", record=MadSample) == []
    assert list(oxyz.iread(MAD, "5:5", record=MadSample)) == []


def test_an_int_index_counts_file_frames_and_a_dropped_one_raises(tmp_path: Path):
    path = frames_file(
        tmp_path / "last_dropped.xyz",
        "step=0 energy=0.0",
        "step=1 energy=1.0",
        "step=2 energy=2.0",
        "step=3",
    )
    for index, step in ((0, 0), (2, 2), (-2, 2), (-4, 0)):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert (
                oxyz.read(path, index, record=Stepped, conformance="warn").step == step
            )
            (streamed,) = oxyz.iread(path, index, record=Stepped, conformance="warn")
        assert streamed.step == step
    for index in (3, -1):
        with (
            pytest.warns(oxyz.SchemaWarning),
            pytest.raises(IndexError, match=f"frame {index} was dropped"),
        ):
            oxyz.read(path, index, record=Stepped, conformance="warn")
        with (
            pytest.warns(oxyz.SchemaWarning),
            pytest.raises(IndexError, match=f"frame {index} was dropped"),
        ):
            oxyz.iread(path, index, record=Stepped, conformance="warn")
    for index in (4, -5):
        with pytest.raises(
            IndexError, match=r"frame -?\d out of range: file has 4 frames"
        ):
            oxyz.read(path, index, record=Stepped, conformance="warn")
    with pytest.raises(IndexError, match="range"):
        oxyz.read(path, [0, 4], record=Stepped, conformance="warn")


def test_a_record_read_checks_only_the_frames_it_selects(tmp_path: Path):
    path = frames_file(
        tmp_path / "bad_last.xyz", "step=0 energy=0.0", "step=1 energy=1.0", "step=2"
    )
    batch_record: Any = oxyz.RecordBatch[Stepped]
    for index, steps in ((0, [0]), (-2, [1]), ("0:2", [0, 1]), ("1::-1", [1, 0])):
        recs = oxyz.read(path, index, record=Stepped)
        assert [r.step for r in (recs if isinstance(recs, list) else [recs])] == steps
        assert [r.step for r in oxyz.iread(path, index, record=Stepped)] == steps
        batch = oxyz.read_batch(path, index, record=batch_record)
        assert batch[Stepped.step].tolist() == steps
    assert [r.step for r in oxyz.read(path, [1, 0], record=Stepped)] == [1, 0]
    for index in (2, -1, "1:", [2]):
        with pytest.raises(SchemaError) as e:
            oxyz.read(path, index, record=Stepped)
        assert (e.value.frame_index, e.value.name) == (2, "energy")
        with pytest.raises(SchemaError) as e:
            list(oxyz.iread(path, index, record=Stepped))
        assert (e.value.frame_index, e.value.name) == (2, "energy")


def test_an_int_or_bounded_forward_slice_stops_after_its_last_frame(tmp_path: Path):
    path = frames_file(tmp_path / "tail.xyz", "step=0 energy=0.0", "step=1")
    with path.open("a") as f:
        f.write("not an atom count\n")
    assert oxyz.read(path, 0, record=Stepped).step == 0
    (streamed,) = oxyz.iread(path, 0, record=Stepped)
    assert streamed.step == 0
    with pytest.warns(oxyz.SchemaWarning, match="energy"):
        recs = oxyz.read(path, slice(0, 2), record=Stepped, conformance="warn")
    assert [r.step for r in recs] == [0]
    with pytest.warns(oxyz.SchemaWarning, match="energy"):
        assert list(oxyz.iread(path, "0:2", record=Stepped, conformance="warn")) == recs
    for index in (-1, [0], "0:"):
        with pytest.raises(oxyz.ParseError):
            oxyz.read(path, index, record=Stepped, conformance="warn")


def test_a_bad_index_string_fails_before_the_file_is_read(tmp_path: Path):
    bad = unparseable(tmp_path)
    for read in (oxyz.read, oxyz.iread):
        with pytest.raises(ValueError, match="abc") as e:
            read(bad, "abc", record=Stepped)
        assert not isinstance(e.value, oxyz.ParseError)


def test_a_repeated_index_gives_records_that_share_nothing():
    for picked in (
        oxyz.read(MACE, [0, 0], record=MaceSample),
        list(oxyz.iread(MACE, [0, 0], record=MaceSample)),
    ):
        first, second = picked
        assert first == second
        assert first is not second
        for name in ("positions", "forces", "cell"):
            assert not np.shares_memory(getattr(first, name), getattr(second, name))


def test_read_fields_are_writable_and_unshared():
    r = oxyz.read(MACE, 0, record=MaceSample)
    (streamed,) = oxyz.iread(MACE, record=MaceSample)
    for record in (r, streamed):
        assert record.stress is not None
        arrays = [record.positions, record.forces, record.stress, record.cell]
        assert all(a.flags.writeable for a in arrays)
        assert not any(
            np.shares_memory(a, b)
            for i, a in enumerate(arrays)
            for b in arrays[i + 1 :]
        )
    assert not np.shares_memory(r.positions, streamed.positions)


def test_readers_do_not_run_the_checks(monkeypatch: pytest.MonkeyPatch):
    def refuse(self: Record) -> None:
        raise AssertionError("__post_init__ ran")

    monkeypatch.setattr(Record, "__post_init__", refuse)
    with pytest.raises(AssertionError, match="ran"):
        make_mace()
    assert len(oxyz.read(MAD, record=MadSample)) == 201
    assert len(list(oxyz.iread(MAD, record=MadSample))) == 201


def test_schema_frame_rule_applies_to_a_record_read(tmp_path: Path):
    ok = SchemaSpec(
        metadata=(MetadataRule("energy", Kind.REAL),), frame=FrameRule(n_atoms_max=1)
    )
    with pytest.raises(SchemaError) as e:
        oxyz.read(MAD, record=MadSample, schema=ok)
    assert (e.value.frame_index, e.value.name) == (1, "n_atoms")
    saved = tmp_path / "ok.yaml"
    saved.write_text(ok.to_yaml())
    with pytest.raises(SchemaError) as e:
        list(oxyz.iread(MAD, record=MadSample, schema=saved))
    assert (e.value.frame_index, e.value.name) == (1, "n_atoms")
    assert bind(MadSample, ok).spec == dataclasses.replace(
        MadSample.spec(), frame=ok.frame
    )


def test_schema_rules_for_other_keys_are_not_applied():
    other = SchemaSpec(
        metadata=(MetadataRule("energy", Kind.REAL), MetadataRule("subset", Kind.INT))
    )
    assert oxyz.read(MAD, record=MadSample, schema=other) == oxyz.read(
        MAD, record=MadSample
    )


SCHEMA_DISAGREEMENTS = [
    ("kind", MadSample, "energy", MetadataRule("energy", Kind.STR)),
    ("pattern", MadSample, "energy", MetadataRule("ener*", Kind.STR)),
    ("width", MadSample, "pos", ColumnRule("pos", Kind.REAL, width=2)),
    ("shape", MadSample, "stress", MetadataRule("stress", Kind.REAL, shape=(6,))),
    ("strict", StrictMace, "Lattice", MetadataRule("Lattice", Kind.INT, shape=(9,))),
]


@pytest.mark.parametrize(
    ("record", "key", "rule"),
    [pytest.param(*case, id=case_id) for case_id, *case in SCHEMA_DISAGREEMENTS],
)
def test_a_disagreeing_schema_fails_before_the_file_is_opened(
    tmp_path: Path, record: type[Record], key: str, rule: ColumnRule | MetadataRule
):
    if isinstance(rule, ColumnRule):
        schema = SchemaSpec(columns=(rule,))
    else:
        schema = SchemaSpec(metadata=(rule,))
    with pytest.raises(SchemaError, match=rf"^{record.__name__}\.\w+: ") as e:
        oxyz.read(unparseable(tmp_path), record=record, schema=schema)
    assert (e.value.name, e.value.frame_index) == (key, None)
    with pytest.raises(SchemaError):
        oxyz.iread(unparseable(tmp_path), record=record, schema=schema)


def test_an_int_schema_rule_binds_to_a_lax_float_field():
    inferred = oxyz.infer_schema(MACE).to_spec()
    lattice = inferred.rule_for("Lattice", "metadata")
    assert lattice is not None
    assert (lattice.kind, lattice.shape) == (Kind.INT, (9,))
    assert oxyz.read(MACE, record=MaceSample, schema=inferred) == oxyz.read(
        MACE, record=MaceSample
    )
    with pytest.raises(SchemaError) as e:
        bind(StrictMace, inferred)
    assert e.value.name == "Lattice"


def test_mode_with_record_is_a_value_error():
    message = "mode= cannot be combined with record=: a record read always projects"
    for mode in ("project", "validate"):
        with pytest.raises(ValueError, match=re.escape(message)):
            oxyz.read(MACE, record=MaceSample, mode=mode)  # ty: ignore[invalid-argument-type]
        with pytest.raises(ValueError, match=re.escape(message)):
            oxyz.iread(MACE, record=MaceSample, mode=mode)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("target", [int, make_mace, Frame, "MaceSample"])
def test_record_must_be_a_record_class(target: Any):
    with pytest.raises(TypeError, match=r"record= takes a Record subclass, got "):
        oxyz.read(MACE, record=target)
    with pytest.raises(TypeError, match=re.escape(repr(target))):
        oxyz.iread(MACE, record=target)


def test_the_record_base_is_not_a_record_class():
    with pytest.raises(TypeError, match="Record itself, which declares no fields"):
        bind(Record)
    with pytest.raises(TypeError, match="Record itself"):
        oxyz.read(MACE, record=Record)


def test_record_instance_is_not_a_record_class():
    with pytest.raises(TypeError, match=r"got MaceSample\(n_atoms=3"):
        oxyz.read(MACE, record=make_mace())  # ty: ignore[no-matching-overload]


class Squares(Record):
    cell: Cell[Array] = Cell()
    virial: Matrix3[Array] = Matrix3()
    moments: Meta[Array] = Meta(shape=(9,))


def flat_and_square(tmp_path: Path) -> tuple[Path, Path]:
    """One frame of 9-value metadata, written flat and as 3x3 nested lists."""
    flat = frames_file(
        tmp_path / "flat.xyz",
        'Lattice="5 1 0 0 5 2 0 0 5" virial="1.5 2 3 4 5 6 7 8 9" '
        'moments="1 2 3 4 5 6 7 8 9"',
    )
    square = frames_file(
        tmp_path / "square.xyz",
        "Lattice=[[5,1,0],[0,5,2],[0,0,5]] virial=[[1.5,2,3],[4,5,6],[7,8,9]] "
        "moments=[[1,2,3],[4,5,6],[7,8,9]]",
    )
    return flat, square


def test_two_d_metadata_reads_like_the_flat_one_but_a_matrix3_as_written(
    tmp_path: Path,
):
    flat, square = flat_and_square(tmp_path)
    from_flat = oxyz.read(flat, 0, record=Squares)
    np.testing.assert_array_equal(
        from_flat.cell, [[5.0, 1.0, 0.0], [0.0, 5.0, 2.0], [0.0, 0.0, 5.0]]
    )
    np.testing.assert_array_equal(
        from_flat.virial, [[1.5, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]]
    )
    np.testing.assert_array_equal(from_flat.moments, np.arange(1.0, 10.0))
    expected = dataclasses.replace(
        from_flat, virial=np.array([[1.5, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    )
    assert oxyz.read(square, 0, record=Squares) == expected
    assert list(oxyz.iread(square, record=Squares)) == [expected]
    inferred = oxyz.infer_schema(square).to_spec()
    assert {rule.shape for rule in inferred.metadata} == {(3, 3)}
    assert oxyz.read(square, record=Squares, schema=inferred) == [expected]


def test_two_d_metadata_with_a_frame_rule_reads_as_without_one(tmp_path: Path):
    _, square = flat_and_square(tmp_path)
    rule = SchemaSpec(frame=FrameRule(n_atoms_max=5))
    expected = oxyz.read(square, record=Squares)
    assert len(expected) == 1
    assert oxyz.read(square, record=Squares, schema=rule) == expected
    assert list(oxyz.iread(square, record=Squares, schema=rule)) == expected


ONE_ATOM = SchemaSpec(frame=FrameRule(n_atoms_max=1))


@pytest.mark.parametrize("conformance", ["required", "strict"])
def test_frame_rule_on_a_record_read_raises_with_context(conformance: Any):
    with pytest.raises(SchemaError, match="n_atoms") as e:
        oxyz.read(MAD, record=MadSample, schema=ONE_ATOM, conformance=conformance)
    assert (e.value.frame_index, e.value.name) == (1, "n_atoms")
    with pytest.raises(SchemaError) as e:
        list(
            oxyz.iread(MAD, record=MadSample, schema=ONE_ATOM, conformance=conformance)
        )
    assert (e.value.frame_index, e.value.name) == (1, "n_atoms")


def test_frame_rule_on_a_record_read_warns_and_keeps_the_frame():
    # As in a plain project-mode read, a frame-rule violation does not drop the frame.
    with pytest.warns(oxyz.SchemaWarning, match="frame 'n_atoms'") as caught:
        recs = oxyz.read(MAD, record=MadSample, schema=ONE_ATOM, conformance="warn")
    assert str(caught[0].message).startswith("frame 1: ")
    assert recs == oxyz.read(MAD, record=MadSample)
    plain_spec = dataclasses.replace(MadSample.spec(), frame=ONE_ATOM.frame)
    with pytest.warns(oxyz.SchemaWarning, match="frame 'n_atoms'") as plain_caught:
        plain = oxyz.read(MAD, schema=plain_spec, conformance="warn")
    assert len(plain) == len(recs) == 201
    assert [str(w.message) for w in plain_caught] == [str(w.message) for w in caught]


def test_a_satisfied_frame_rule_under_strict_reports_nothing(
    recwarn: pytest.WarningsRecorder,
):
    roomy = SchemaSpec(frame=FrameRule(n_atoms_max=1000))
    recs = oxyz.read(MAD, record=MadSample, schema=roomy, conformance="strict")
    assert recs == oxyz.read(MAD, record=MadSample)
    assert (
        list(oxyz.iread(MAD, record=MadSample, schema=roomy, conformance="strict"))
        == recs
    )
    assert not recwarn.list


def test_a_schema_built_with_lists_binds(tmp_path: Path):
    rules: Any = [MetadataRule("energy", Kind.REAL)]
    spec = SchemaSpec(metadata=rules, frame=FrameRule(n_atoms_max=1))
    path = frames_file(tmp_path / "one.xyz", "step=0 energy=1.0")
    assert oxyz.read(path, 0, record=Stepped, schema=spec).energy == 1.0
    batch_record: Any = oxyz.RecordBatch[Stepped]
    assert len(oxyz.read_batch(path, record=batch_record, schema=spec)) == 1


def test_bind_is_cached(tmp_path: Path):
    assert bind(MadSample) is bind(MadSample)
    assert hash(bind(MadSample)) == hash(bind(MadSample))
    spec = SchemaSpec(metadata=(MetadataRule("energy", Kind.REAL),))
    assert bind(MadSample, spec) is bind(MadSample, dataclasses.replace(spec))
    saved = tmp_path / "spec.yaml"
    saved.write_text(spec.to_yaml())
    assert bind(MadSample, saved) is bind(MadSample, spec)
    assert bind(MadSample, spec) is not bind(MadSample)


class Table(Record):
    positions: Column[Array] = Column("pos", width=3)
    labels: Column[list[str] | None] = Column(fill="X")
    charges: Column[Array | None] = Column()
    energy: Meta[float] = Meta()
    charge: Meta[int] = Meta(fill=0)
    spin: Meta[int | None] = Meta(fill=1)
    stress: Matrix3[Array | None] = Matrix3()
    cell: Cell[Array] = Cell(fill=0.0)


def without_fills(plan: Any) -> tuple[list[tuple], list[tuple], bool]:
    columns, metadata, promote = plan
    return [e[:4] for e in columns], [e[:4] for e in metadata], promote


def test_binding_plans_follow_the_field_table():
    binding = bind(Table)
    assert (binding.target, binding.record) == (Table, Table)
    assert binding.fields == Table.__record_fields__
    assert binding.spec == Table.spec()
    shape = (
        [
            ("pos", "R", 3, True),
            ("labels", "S", 1, False),
            ("charges", "R", 1, False),
        ],
        [
            ("energy", "R", (), True),
            ("charge", "I", (), False),
            ("spin", "I", (), False),
            ("stress", "R", (9,), False),
            ("Lattice", "R", (9,), False),
        ],
    )
    assert without_fills(binding.frame_plan) == (*shape, True)
    columns, metadata, _ = binding.frame_plan
    assert [e[4] for e in columns] == [None, None, None]
    assert [e[4] for e in metadata] == [None, 0, None, None, 0.0]
    assert [e[5] for e in metadata] == [False, False, False, True, False]
    assert without_fills(binding.batch_plan) == (*shape, True)
    columns, metadata, _ = binding.batch_plan
    assert [e[4] for e in columns[:2]] == [None, "X"]
    assert math.isnan(columns[2][4])
    assert [e[4] for e in metadata[:3]] == [None, 0, 1]
    assert math.isnan(metadata[3][4])
    assert metadata[4][4] == 0.0
    assert binding.optional == (
        ("labels", 0),
        ("charges", 1),
        ("spin", 3),
        ("stress", 4),
    )
    assert binding.reported == {
        "labels": 0,
        "charges": 1,
        "charge": 2,
        "spin": 3,
        "stress": 4,
        "cell": 5,
    }


def test_strict_record_plans_do_not_promote():
    assert bind(StrictMace).frame_plan[2] is False
    assert bind(StrictMace).batch_plan[2] is False
    assert bind(MaceSample).frame_plan[2] is True


def test_batch_plan_needs_a_fill_for_an_optional_non_real_field():
    class Counted(Record):
        count: Meta[int | None] = Meta()

    binding = bind(Counted)
    assert binding.frame_plan[1] == [("count", "I", (), False, None, False)]
    with pytest.raises(SchemaError, match=r"Counted\.count") as e:
        _ = binding.batch_plan
    assert e.value.name == "count"


def test_cast_frame_owned_or_read_only():
    binding = bind(MaceSample)
    frame = oxyz.read(MACE, 0, schema=MaceSample.spec())
    shared: Any = binding.cast_frame(frame, owned=False)
    assert shared == oxyz.read(MACE, 0, record=MaceSample)
    assert not shared.positions.flags.writeable
    assert np.shares_memory(shared.positions, frame.columns["pos"])
    owned: Any = binding.cast_frame(frame, owned=True)
    assert owned.positions.flags.writeable
    assert binding.caster("required")(frame, 0) == owned
