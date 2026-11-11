from __future__ import annotations

import copy
import dataclasses
import inspect
import pickle
from typing import Annotated, Any

import numpy as np
import numpy.typing as npt
import pytest

from oxyz import Cell, Column, Matrix3, Meta, Record
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
