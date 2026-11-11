from __future__ import annotations

import copy
import dataclasses
import inspect
import pickle
import re
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
    Matrix3,
    Meta,
    MetadataRule,
    Record,
    SchemaError,
    SchemaSpec,
)
from oxyz._record_fields import FieldType
from oxyz._schema import Kind

MACE = Path(__file__).parent / "data" / "mace_ref_energy_forces_stress.xyz"

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
    (
        "duplicate-key",
        'a: Meta[float] = Meta("e")\nb: Meta[float] = Meta("e")',
        "b",
        "'e' is also read by a",
    ),
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
            MetadataRule(
                "REF_stress", Kind.REAL, shape=(6,), required=False, absent="omit"
            ),
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

    spec = Filled.spec()
    assert spec == SchemaSpec(
        mode="project",
        columns=(ColumnRule("labels", Kind.STR, required=False, fill="X"),),
        metadata=(
            MetadataRule("charge", Kind.INT, required=False, fill=0, absent="omit"),
            MetadataRule("Lattice", Kind.REAL, shape=(9,), required=False, fill=0.0),
            MetadataRule(
                "virial", Kind.REAL, shape=(9,), required=False, absent="omit"
            ),
        ),
    )
    assert SchemaSpec.from_yaml(spec.to_yaml()) == spec


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
        MetadataRule("E", Kind.REAL, required=False, fill=0.0, absent="omit"),
    )
    assert all(spec == specs[0] for spec in specs)


def mace_frame(**metadata: Any) -> Frame:
    """The MACE fixture's first frame, with metadata entries replaced or removed."""
    frame = oxyz.read(MACE, 0)
    merged = {**frame.metadata, **metadata}
    return Frame(
        n_atoms=frame.n_atoms,
        columns=frame.columns,
        metadata={key: value for key, value in merged.items() if value is not None},
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
    r = MaceSample.from_frame(mace_frame(Lattice=square))
    np.testing.assert_array_equal(r.cell, square)


class StrictMace(MaceSample, strict=True):
    pass


@pytest.mark.parametrize(
    ("record", "frame", "key", "reason"),
    [
        (MaceSample, mace_frame(REF_energy=None), "REF_energy", "missing"),
        (MaceSample, mace_frame(REF_energy=True), "REF_energy", "expected R, found L"),
        (MaceSample, mace_frame(config_type=1), "config_type", "expected S, found I"),
        (MaceSample, mace_frame(REF_stress=np.zeros(3)), "REF_stress", r"R\[3\]"),
        (MaceSample, mace_frame(Lattice=np.zeros((2, 2))), "Lattice", r"R\[4\]"),
        (StrictMace, mace_frame(), "Lattice", r"found I\[9\].*strict"),
        (StrictMace, mace_frame(REF_energy=0), "REF_energy", "found I.*strict"),
    ],
)
def test_from_frame_errors_carry_the_key_and_no_frame_index(record, frame, key, reason):
    with pytest.raises(
        SchemaError, match=rf"^{record.__name__}: .*{key}.*{reason}"
    ) as e:
        record.from_frame(frame)
    assert (e.value.name, e.value.frame_index) == (key, None)


def test_from_frame_column_errors_carry_the_key():
    frame = oxyz.read(MACE, 0)
    columns = {**frame.columns, "REF_forces": np.zeros(3)}
    narrow = Frame(n_atoms=3, columns=columns, metadata=frame.metadata)
    with pytest.raises(SchemaError, match=r"REF_forces.*expected R:3, found R:1") as e:
        MaceSample.from_frame(narrow)
    assert (e.value.name, e.value.frame_index) == ("REF_forces", None)
    del columns["REF_forces"]
    with pytest.raises(SchemaError, match=r"REF_forces.*missing") as e:
        MaceSample.from_frame(narrow)
    assert e.value.name == "REF_forces"


def test_from_frame_optional_and_fill():
    class Filled(Record):
        labels: Column[list[str]] = Column(fill="X")
        charge: Meta[int | None] = Meta(fill=0)
        cell: Cell[Array] = Cell(fill=0.0)

    class MaybeCell(Record):
        cell: Cell[Array | None] = Cell(fill=0.0)

    assert MaceSample.from_frame(mace_frame(REF_stress=None)).stress is None
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
