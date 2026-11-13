from __future__ import annotations

import dataclasses
import itertools
import math
import pickle
import re
import typing
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import oxyz
from oxyz import (
    Cell,
    Column,
    FrameRule,
    Matrix3,
    Meta,
    Record,
    RecordBatch,
    SchemaSpec,
    _summary,
)
from oxyz._record_batch import batch_class
from oxyz._record_bind import bind
from oxyz._record_union import Key, members_of, union_plan

if typing.TYPE_CHECKING:
    from collections.abc import Callable

type Array = npt.NDArray[np.float64]

MAD = Path(__file__).parent / "data" / "mad_r2scan_sample.extxyz"

LATTICE: Key = ("metadata", "Lattice")
STRESS: Key = ("metadata", "stress")


class MadFrame(Record):
    species: Column[list[str]] = Column()
    positions: Column[Array] = Column("pos", width=3)
    forces: Column[Array] = Column(width=3)
    energy: Meta[float] = Meta()
    subset: Meta[str] = Meta()


class Molecule(MadFrame):
    pass


class Crystal(MadFrame):
    cell: Cell[Array] = Cell()


class Strained(MadFrame):
    cell: Cell[Array] = Cell()
    stress: Matrix3[Array] = Matrix3()


class Other(MadFrame):
    tag: Meta[str] = Meta()


type MadSample = Molecule | Crystal | Strained


class MadBatch(RecordBatch[MadSample]):
    species: list[str]
    positions: Array
    forces: Array
    energy: Array
    subset: list[str]
    cell: Array
    stress: Array


def expected_member(
    members: tuple[type[Record], ...], signature: frozenset[Key]
) -> int | None:
    """Brute-force oracle: the members whose declared keys admit the signature."""
    core = set.intersection(
        *({(s.axis, s.key) for _, _, s in m.__record_fields__} for m in members)
    )
    found = []
    for index, member in enumerate(members):
        required = {
            (s.axis, s.key)
            for _, f, s in member.__record_fields__
            if not f.optional and s.fill is None
        }
        declared = {(s.axis, s.key) for _, _, s in member.__record_fields__}
        if (required - core) <= signature <= (declared - core):
            found.append(index)
    assert len(found) <= 1
    return found[0] if found else None


def test_members_of_every_spelling():
    assert (
        members_of(Molecule | Crystal)
        == members_of(typing.Union[Molecule, Crystal])  # noqa: UP007 — the spelling under test
        == (Molecule, Crystal)
    )
    assert members_of(MadSample) == (Molecule, Crystal, Strained)
    assert members_of(Molecule) == (Molecule,)
    assert members_of(Molecule | Crystal | Molecule) == (Molecule, Crystal)


def test_non_record_member_is_a_type_error():
    with pytest.raises(TypeError, match="int"):
        members_of(Molecule | int)
    with pytest.raises(TypeError, match="int"):
        members_of(int)
    with pytest.raises(TypeError, match="Record"):
        members_of(Record)


def test_mad_union_plan():
    plan = union_plan(members_of(MadSample))
    assert plan.matching == (LATTICE, STRESS)
    assert [n for n, _, _ in plan.fields] == [
        "species",
        "positions",
        "forces",
        "energy",
        "subset",
        "cell",
        "stress",
    ]
    assert plan.member_fields[0] == (
        "species",
        "positions",
        "forces",
        "energy",
        "subset",
    )
    assert plan.member_fields[2][-2:] == ("cell", "stress")
    assert plan.names[LATTICE] == "cell"
    assert plan.names[("column", "pos")] == "positions"
    assert plan.core_filled == frozenset()
    assert plan.required == {
        ("column", "species"),
        ("column", "pos"),
        ("column", "forces"),
        ("metadata", "energy"),
        ("metadata", "subset"),
    }
    assert plan.name == "Molecule | Crystal | Strained"
    assert union_plan((Molecule,)).name == "Molecule"
    signatures = ((), (LATTICE,), (LATTICE, STRESS), (STRESS,))
    assert [plan.member_for(frozenset(s)) for s in signatures] == [0, 1, 2, None]
    assert plan.member_for(frozenset({LATTICE, ("metadata", "other")})) is None


def test_describe_lists_each_member():
    assert union_plan(members_of(MadSample)).describe().splitlines() == [
        "Molecule: requires {}; may carry {}",
        "Crystal: requires {Lattice}; may carry {}",
        "Strained: requires {Lattice, stress}; may carry {}",
    ]


@pytest.mark.parametrize(
    "members",
    [
        (Molecule, Crystal, Strained, Other),
        (Strained, Molecule, Crystal),
        (Other, Strained, Crystal, Molecule),
    ],
)
def test_member_order_does_not_change_matching(members):
    plan = union_plan(members)
    keys = [*plan.matching]
    for size in range(len(keys) + 1):
        for subset in itertools.combinations(keys, size):
            signature = frozenset(subset)
            found = plan.member_for(signature)
            wanted = expected_member(members, signature)
            assert found == wanted
    reference = union_plan((Molecule, Crystal, Strained))
    for signature in (frozenset(), frozenset({LATTICE}), frozenset({LATTICE, STRESS})):
        mine = plan.member_for(signature)
        assert mine is not None
        theirs = reference.member_for(signature)
        assert theirs is not None
        assert plan.members[mine] is reference.members[theirs]


def _subclass() -> tuple[type[Record], ...]:
    class Sub(Molecule):
        extra: Meta[int] = Meta()

    return (Molecule, Sub)


def _strict_mismatch() -> tuple[type[Record], ...]:
    class Strict(MadFrame, strict=True):
        pass

    return (Molecule, Strict)


def _one_key_two_names() -> tuple[type[Record], ...]:
    class Renamed(MadFrame):
        lattice: Cell[Array] = Cell()

    return (Crystal, Renamed)


def _one_name_two_keys() -> tuple[type[Record], ...]:
    class Boxed(MadFrame):
        cell: Meta[Array] = Meta("Box", shape=(9,))

    return (Crystal, Boxed)


def _width() -> tuple[type[Record], ...]:
    class Narrow(MadFrame):
        positions: Column[Array] = Column("pos", width=1)

    return (Molecule, Narrow)


def _fill() -> tuple[type[Record], ...]:
    class Filled(MadFrame):
        energy: Meta[float] = Meta(fill=0.0)

    return (Molecule, Filled)


def _ambiguous() -> tuple[type[Record], ...]:
    class Zeroed(MadFrame):
        cell: Cell[Array] = Cell(fill=0.0)

    return (Molecule, Zeroed)


@pytest.mark.parametrize(
    ("build", "pattern"),
    [
        (_subclass, "Sub is a subclass of Molecule; union members must be siblings"),
        (_strict_mismatch, "Molecule and Strict.*strict"),
        (_one_key_two_names, r"Crystal\.cell and Renamed\.lattice.*Lattice"),
        (_one_name_two_keys, r"Crystal and Boxed.*cell.*Lattice.*Box"),
        (_width, "Molecule and Narrow.*positions"),
        (_fill, "Molecule and Filled.*energy"),
        (_ambiguous, "Molecule and Zeroed.*Lattice"),
    ],
    ids=[
        "subclass",
        "strict",
        "key-two-names",
        "name-two-keys",
        "width",
        "fill",
        "ambiguous",
    ],
)
def test_union_errors(build: Callable[[], tuple[type[Record], ...]], pattern: str):
    with pytest.raises(TypeError, match=pattern):
        union_plan(build())


def test_a_shared_nan_fill_binds_and_reads():
    class NanBase(MadFrame):
        charge: Meta[float] = Meta(fill=math.nan)

    class Bare(NanBase):
        pass

    class Celled(NanBase):
        cell: Cell[Array] = Cell()

    recs = oxyz.read(MAD, record=Bare | Celled)
    assert Counter(type(r) for r in recs) == {Bare: 49, Celled: 152}
    assert all(math.isnan(r.charge) for r in recs)
    batch = oxyz.read_batch(MAD, record=RecordBatch[Bare | Celled])
    assert np.isnan(batch[NanBase.charge]).all()


def test_optional_member_admits_both_signatures():
    class Maybe(MadFrame):
        cell: Cell[Array | None] = Cell()

    class Tagged(MadFrame):
        tag: Meta[str] = Meta()

    plan = union_plan((Maybe, Tagged))
    assert plan.member_for(frozenset()) == 0
    assert plan.member_for(frozenset({LATTICE})) == 0
    assert plan.member_for(frozenset({("metadata", "tag")})) == 1
    with pytest.raises(TypeError, match="Molecule"):
        union_plan((Maybe, Molecule))


def test_a_key_every_member_fills_is_not_a_matching_key():
    class A(Record):
        q: Meta[float] = Meta(fill=0.0)
        a: Meta[int] = Meta()

    class B(Record):
        q: Meta[float] = Meta(fill=0.0)
        b: Meta[int] = Meta()

    plan = union_plan((A, B))
    assert plan.core_filled == {("metadata", "q")}
    assert ("metadata", "q") not in plan.matching
    assert plan.member_for(frozenset({("metadata", "a")})) == 0
    assert plan.member_for(frozenset({("metadata", "b")})) == 1


def test_a_single_record_is_a_union_of_one():
    plan = union_plan((MadFrame,))
    assert plan.required == {
        ("column", "species"),
        ("column", "pos"),
        ("column", "forces"),
        ("metadata", "energy"),
        ("metadata", "subset"),
    }
    assert plan.matching == ()
    assert plan.member_for(frozenset()) == 0

    class Loose(MadFrame):
        cell: Cell[Array | None] = Cell()
        note: Meta[str] = Meta(fill="")

    loose = union_plan((Loose,))
    assert loose.matching == (LATTICE,)
    assert loose.core_filled == {("metadata", "note")}
    assert loose.member_for(frozenset({LATTICE})) == 0
    assert loose.member_for(frozenset({("metadata", "unknown")})) == 0


def test_the_plan_is_cached_and_hashable():
    plan = union_plan((Molecule, Crystal))
    assert union_plan((Molecule, Crystal)) is plan
    assert hash(plan) == hash(union_plan((Molecule, Crystal)))


def member_of(frame: oxyz.Frame) -> type[Record]:
    members = members_of(MadSample)
    signature = frozenset(key for key in (LATTICE, STRESS) if key[1] in frame.metadata)
    index = expected_member(members, signature)
    assert index is not None
    return members[index]


def mad_file(path: Path, *tails: str) -> Path:
    """Write one single-hydrogen MAD-shaped frame per comment-line tail to `path`."""
    path.write_text(
        "".join(
            "1\nProperties=species:S:1:pos:R:3:forces:R:3 energy=-1.0 subset=x "
            f"{tail}\nH 0.0 0.0 0.0 0.0 0.0 0.0\n"
            for tail in tails
        )
    )
    return path


LATTICE_TAIL = 'Lattice="10 0 0 0 10 0 0 0 10"'
STRESS_TAIL = 'stress="1 2 3 4 5 6 7 8 9"'


def unadmitted(tmp_path: Path) -> Path:
    """Three frames; frame 1 carries `stress` but no `Lattice`."""
    return mad_file(tmp_path / "unadmitted.xyz", "", STRESS_TAIL, LATTICE_TAIL)


def test_every_frame_becomes_the_member_its_keys_imply():
    recs = oxyz.read(MAD, record=MadSample)
    plain = oxyz.read(MAD)
    assert [type(r) for r in recs] == [member_of(f) for f in plain]
    assert Counter(type(r) for r in recs) == {Molecule: 49, Crystal: 126, Strained: 26}
    strained = next(r for r in recs if type(r) is Strained)
    molecule = next(r for r in recs if type(r) is Molecule)
    assert strained.stress.shape == (3, 3)
    assert not hasattr(molecule, "cell")


def test_spellings_and_order_read_the_same():
    base = oxyz.read(MAD, record=MadSample)
    assert oxyz.read(MAD, record=Molecule | Crystal | Strained) == base
    assert oxyz.read(MAD, record=typing.Union[Strained, Molecule, Crystal]) == base  # noqa: UP007 — the spelling under test


def test_union_of_one_reads_as_the_record():
    type Only = MadFrame
    assert oxyz.read(MAD, record=Only) == oxyz.read(MAD, record=MadFrame)


def test_iread_threads_and_index_parity():
    base = oxyz.read(MAD, record=MadSample)
    assert list(oxyz.iread(MAD, record=MadSample)) == base
    assert oxyz.read(MAD, record=MadSample, threads=1) == base
    assert oxyz.read(MAD, 60, record=MadSample) == base[60]
    assert oxyz.read(MAD, "0:10:3", record=MadSample) == base[0:10:3]


def test_unadmitted_frame_raises_with_index_and_signature(tmp_path: Path):
    path = unadmitted(tmp_path)
    with pytest.raises(oxyz.SchemaError, match="stress") as e:
        oxyz.read(path, record=MadSample)
    assert e.value.frame_index == 1
    assert "no member of Molecule | Crystal | Strained admits keys {stress}" in str(
        e.value
    )


def test_unadmitted_frame_is_dropped_under_warn(tmp_path: Path):
    path = unadmitted(tmp_path)
    with pytest.warns(oxyz.SchemaWarning, match="frame 1"):
        recs = oxyz.read(path, record=MadSample, conformance="warn")
    assert [type(r) for r in recs] == [Molecule, Crystal]
    with pytest.warns(oxyz.SchemaWarning):
        assert list(oxyz.iread(path, record=MadSample, conformance="warn")) == recs
    with (
        pytest.warns(oxyz.SchemaWarning),
        pytest.raises(IndexError, match=re.escape("Molecule | Crystal | Strained")),
    ):
        oxyz.read(path, 1, record=MadSample, conformance="warn")


def test_wrong_kind_matching_key_fails_as_that_member(tmp_path: Path):
    path = mad_file(tmp_path / "wrong.xyz", f"{LATTICE_TAIL} stress=abc")
    with pytest.raises(oxyz.SchemaError, match="stress") as e:
        oxyz.read(path, record=MadSample)
    assert (e.value.frame_index, e.value.name) == (0, "stress")


def test_missing_all_required_key_is_the_projection_error(tmp_path: Path):
    path = tmp_path / "no_energy.xyz"
    path.write_text(
        "1\nProperties=species:S:1:pos:R:3:forces:R:3 subset=x\n"
        "H 0.0 0.0 0.0 0.0 0.0 0.0\n"
    )
    with pytest.raises(oxyz.SchemaError, match="energy") as e:
        oxyz.read(path, record=MadSample)
    assert (e.value.frame_index, e.value.name) == (0, "energy")


def test_an_optional_member_field_is_none_where_its_key_is_absent(tmp_path: Path):
    class Charged(MadFrame):
        cell: Cell[Array] = Cell()
        charge: Meta[float | None] = Meta()

    path = mad_file(
        tmp_path / "charged.xyz", "", LATTICE_TAIL, f"{LATTICE_TAIL} charge=0.5"
    )
    recs = oxyz.read(path, record=Molecule | Charged)
    assert [type(r) for r in recs] == [Molecule, Charged, Charged]
    assert [getattr(r, "charge", "lacks") for r in recs] == ["lacks", None, 0.5]
    batch = oxyz.read_batch(path, record=RecordBatch[Molecule | Charged])
    assert [batch[i] for i in range(3)] == recs
    assert batch.present(Charged.charge).tolist() == [False, False, True]


def test_a_fill_field_one_member_lacks_is_filled_on_a_frame_read():
    class WithQ(MadFrame):
        cell: Cell[Array] = Cell()
        charge: Meta[float] = Meta(fill=0.0)

    recs = oxyz.read(MAD, record=Molecule | WithQ)
    assert Counter(type(r) for r in recs) == {Molecule: 49, WithQ: 152}
    assert all(r.charge == 0.0 for r in recs if isinstance(r, WithQ))


def test_binding_errors_happen_before_the_file_is_opened(tmp_path: Path):
    class Maybe(MadFrame):
        cell: Cell[Array | None] = Cell()

    missing = tmp_path / "missing.xyz"
    with pytest.raises(TypeError, match="Molecule and Maybe both admit"):
        oxyz.read(missing, record=Molecule | Maybe)
    with pytest.raises(TypeError, match="int is not a subclass of Record"):
        oxyz.read(missing, record=Molecule | int)  # ty: ignore[no-matching-overload]


def test_a_union_on_a_batch_read_names_its_batch_form(tmp_path: Path):
    missing = tmp_path / "missing.xyz"
    with pytest.raises(
        TypeError, match=re.escape("takes RecordBatch[Molecule | Crystal] or a")
    ):
        oxyz.read_batch(missing, record=Molecule | Crystal)  # ty: ignore[no-matching-overload]
    with pytest.raises(
        TypeError, match=re.escape("RecordBatch[Molecule | Crystal | Strained]")
    ):
        oxyz.read_batch(missing, record=MadSample)  # ty: ignore[no-matching-overload]
    with pytest.raises(TypeError, match="RecordBatch subclass or RecordBatch"):
        oxyz.read_batch(missing, record=Molecule | int)  # ty: ignore[no-matching-overload]


def test_schema_check_runs_over_the_union_keys(tmp_path: Path):
    schema = oxyz.SchemaSpec(metadata=(oxyz.MetadataRule("stress", oxyz.Kind.STR),))
    with pytest.raises(oxyz.SchemaError, match="stress") as e:
        oxyz.read(tmp_path / "missing.xyz", record=MadSample, schema=schema)
    assert e.value.name == "stress"
    assert str(e.value).startswith(
        "Strained.stress (in Molecule | Crystal | Strained): schema= gives"
    )


def test_binding_spec_is_the_merged_project_spec():
    spec = bind(MadSample).spec
    assert [r.name for r in spec.columns] == ["species", "pos", "forces"]
    assert [r.key for r in spec.metadata] == ["energy", "subset", "Lattice", "stress"]
    assert [r.required for r in spec.metadata] == [True, True, False, False]
    lattice = spec.rule_for("Lattice", "metadata")
    assert lattice is not None
    assert lattice.required is False


def test_bind_caches_one_binding_per_union():
    assert (
        bind(MadSample)
        is bind(Molecule | Crystal | Strained)
        is bind(typing.Union[Molecule, Crystal, Strained])  # noqa: UP007 — the spelling under test
    )
    assert bind(MadSample).record is None
    assert bind(MadFrame).record is MadFrame


@pytest.fixture(scope="module")
def mad() -> RecordBatch[MadSample]:
    return oxyz.read_batch(MAD, record=RecordBatch[MadSample])


@pytest.fixture(scope="module")
def recs() -> list[MadSample]:
    return oxyz.read(MAD, record=MadSample)


def test_batch_rows_equal_the_frame_read(
    mad: RecordBatch[MadSample], recs: list[MadSample]
):
    assert len(mad) == len(recs) == 201
    assert all(mad[i] == recs[i] and type(mad[i]) is type(recs[i]) for i in range(201))
    assert [type(r) for r in mad] == [member_of(f) for f in oxyz.read(MAD)]


def test_tags_members_and_where_agree_with_the_frame_read(
    mad: RecordBatch[MadSample], recs: list[MadSample]
):
    assert mad.members == (Molecule, Crystal, Strained)
    assert [mad.members[t] for t in mad.tags] == [type(r) for r in recs]
    assert mad.where(Strained).sum() == 26
    assert mad.where(Crystal).sum() == 126
    assert (
        mad.where(Crystal, Strained) == (mad.where(Crystal) | mad.where(Strained))
    ).all()
    assert not mad.where().any()
    assert not mad.tags.flags.writeable


def test_where_rejects_a_non_member(mad: RecordBatch[MadSample]):
    with pytest.raises(TypeError, match="Molecule, Crystal, Strained"):
        mad.where(MadFrame)


def test_fields_some_members_lack_are_nan_exactly_there(mad: RecordBatch[MadSample]):
    lacks = ~mad.where(Strained)
    assert np.isnan(mad[Strained.stress][lacks]).all()
    assert not np.isnan(mad[Strained.stress][~lacks]).any()
    assert mad.present(Strained.stress).tolist() == mad.where(Strained).tolist()
    assert mad.present("cell").tolist() == mad.where(Crystal, Strained).tolist()
    assert mad.present(MadFrame.energy).all()


def test_keys_of_every_member_reach_the_one_field(mad: RecordBatch[MadSample]):
    class Elsewhere(Record):
        energy: Meta[float] = Meta()

    assert mad[Crystal.cell] is mad[Strained.cell] is mad["cell"]
    assert mad[MadFrame.energy] is mad[Molecule.energy]
    with pytest.raises(KeyError, match=r"Other\.tag .*Molecule \| Crystal"):
        mad[Other.tag]
    with pytest.raises(KeyError, match=r"Elsewhere\.energy"):
        mad[Elsewhere.energy]


def test_slices_and_gathers_carry_tags(mad: RecordBatch[MadSample]):
    strained = mad[mad.where(Strained)]
    assert type(strained) is type(mad)
    assert len(strained) == 26
    assert (strained.tags == 2).all()
    assert all(type(r) is Strained for r in strained)
    assert mad[10:20].tags.tolist() == mad.tags[10:20].tolist()
    assert mad[::7].tags.tolist() == mad.tags[::7].tolist()


def test_every_spelling_is_one_class():
    derived = batch_class(RecordBatch[MadSample])
    assert derived is batch_class(RecordBatch[Molecule | Crystal | Strained])
    assert derived is batch_class(
        RecordBatch[typing.Union[Molecule, Crystal, Strained]]  # noqa: UP007 — the spelling under test
    )
    assert batch_class(RecordBatch[Crystal | Molecule]) is not batch_class(
        RecordBatch[Molecule | Crystal]
    )
    assert derived.__name__ == "MoleculeOrCrystalOrStrainedBatch"
    assert batch_class(RecordBatch[Molecule]).__name__ == "MoleculeBatch"


def test_declared_union_batch_reads_the_same(mad: RecordBatch[MadSample]):
    declared = oxyz.read_batch(MAD, record=MadBatch)
    assert all(
        _summary.values_equal(declared[n], mad[n])
        for n in ("positions", "energy", "cell", "stress")
    )
    assert declared.tags.tolist() == mad.tags.tolist()
    assert declared.members == mad.members
    assert [type(r) for r in declared] == [type(r) for r in mad]


def one_frame(**overrides: Any) -> MadBatch:
    """A one-frame `MadBatch` built through `__init__`."""
    nan = np.full((1, 3, 3), np.nan)
    values: dict[str, Any] = {
        "offsets": np.array([0, 1]),
        "frame_indices": np.array([0]),
        "species": ["H"],
        "positions": np.zeros((1, 3)),
        "forces": np.zeros((1, 3)),
        "energy": np.array([-1.0]),
        "subset": ["x"],
        "cell": nan,
        "stress": nan,
    }
    return MadBatch(**(values | overrides))


def test_hand_built_union_batch_checks_tags():
    with pytest.raises(ValueError, match=r"MadBatch\.tags: .*3"):
        one_frame(tags=np.array([3]))
    with pytest.raises(ValueError, match=r"MadBatch\.tags: .*-1"):
        one_frame(tags=np.array([-1]))
    with pytest.raises(ValueError, match=r"MadBatch\.tags: .*\(1,\)"):
        one_frame(tags=np.array([0, 0]))
    with pytest.raises(TypeError, match=r"MadBatch\.tags: .*integer"):
        one_frame(tags=np.array([0.0]))
    assert type(one_frame()[0]) is Molecule
    assert type(one_frame(tags=np.array([2]))[0]) is Strained
    assert one_frame(tags=np.array([2], dtype=np.uint8)).tags.dtype == np.intp


def test_a_member_the_file_never_contains(tmp_path: Path, recs: list[MadSample]):
    path = tmp_path / "molecules.xyz"
    oxyz.write(path, [f for f in oxyz.read(MAD) if "Lattice" not in f.metadata])
    molecules = oxyz.read_batch(path, record=RecordBatch[MadSample])
    assert len(molecules) == sum(type(r) is Molecule for r in recs)
    assert molecules[Crystal.cell].shape == (len(molecules), 3, 3)
    assert np.isnan(molecules[Strained.stress]).all()
    assert not molecules.where(Strained).any()
    assert type(molecules[0]) is Molecule


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
def test_iread_batch_and_threads_parity(
    mad: RecordBatch[MadSample], kwargs: dict[str, Any]
):
    assert oxyz.read_batch(MAD, record=RecordBatch[MadSample], threads=1) == mad
    seen: list[int] = []
    for b in oxyz.iread_batch(MAD, record=RecordBatch[MadSample], **kwargs):
        rows = mad[b.frame_indices]
        assert type(b) is type(mad)
        assert b.tags.tolist() == rows.tags.tolist()
        for name in ("stress", "cell"):
            assert b.present(name).tolist() == rows.present(name).tolist()
        assert b == rows
        seen += b.frame_indices.tolist()
    assert sorted(seen) == list(range(201))


def test_non_real_field_some_member_lacks_needs_a_fill(tmp_path: Path):
    class Labelled(MadFrame):
        cell: Cell[Array] = Cell()
        label: Meta[str] = Meta()

    with pytest.warns(oxyz.SchemaWarning, match="Labelled"):
        recs = oxyz.read(MAD, record=Molecule | Labelled, conformance="warn")
    assert {type(r) for r in recs} == {Molecule}
    with pytest.raises(oxyz.SchemaError, match="label") as e:
        oxyz.read_batch(
            tmp_path / "missing.xyz", record=RecordBatch[Molecule | Labelled]
        )
    assert e.value.name == "label"
    assert str(e.value) == (
        "Labelled.label (in Molecule | Labelled): a S field some member lacks has "
        "no natural null; give it a fill= to read it into a batch"
    )


def warned_frames(caught: pytest.WarningsRecorder) -> list[int]:
    """The frame index each caught warning's message starts with, in order."""
    return [int(re.findall(r"^frame (\d+)", str(w.message))[0]) for w in caught]


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"frames_per_batch": 1}, id="planned"),
        pytest.param({"frames_per_batch": 1, "threads": 1}, id="sequential"),
        pytest.param({"atoms_per_batch": 2}, id="atom-budget"),
    ],
)
def test_warn_keeps_the_same_frames_on_every_path(
    tmp_path: Path, kwargs: dict[str, Any]
):
    path = mad_file(
        tmp_path / "five.xyz",
        "",
        STRESS_TAIL,
        LATTICE_TAIL,
        f"{LATTICE_TAIL} stress=abc",
        f"{LATTICE_TAIL} {STRESS_TAIL}",
    )
    with pytest.warns(oxyz.SchemaWarning) as caught:
        recs = oxyz.read(path, record=MadSample, conformance="warn")
    assert sorted(warned_frames(caught)) == [1, 3]
    with pytest.warns(oxyz.SchemaWarning) as caught:
        b = oxyz.read_batch(path, record=RecordBatch[MadSample], conformance="warn")
    assert sorted(warned_frames(caught)) == [1, 3]
    assert b.frame_indices.tolist() == [0, 2, 4]
    assert [b[i] for i in range(3)] == recs
    assert [type(r) for r in b] == [Molecule, Crystal, Strained]
    with pytest.warns(oxyz.SchemaWarning) as caught:
        windows = list(
            oxyz.iread_batch(
                path, record=RecordBatch[MadSample], conformance="warn", **kwargs
            )
        )
    assert sorted(warned_frames(caught)) == [1, 3]
    rows = sorted(
        (int(index), record)
        for w in windows
        for index, record in zip(w.frame_indices, w, strict=True)
    )
    assert [index for index, _ in rows] == [0, 2, 4]
    assert [record for _, record in rows] == recs


def test_an_all_dropped_union_batch_is_empty_under_warn(tmp_path: Path):
    path = mad_file(tmp_path / "dropped.xyz", STRESS_TAIL, STRESS_TAIL)
    with pytest.warns(oxyz.SchemaWarning):
        batch = oxyz.read_batch(path, record=RecordBatch[MadSample], conformance="warn")
    assert type(batch) is batch_class(RecordBatch[MadSample])
    assert len(batch) == 0
    assert batch.tags.shape == (0,)
    assert batch.where(Molecule).shape == (0,)
    assert batch == batch[:]
    with pytest.warns(oxyz.SchemaWarning):
        windows = list(
            oxyz.iread_batch(
                path,
                record=RecordBatch[MadSample],
                frames_per_batch=1,
                conformance="warn",
            )
        )
    assert windows == []


def test_unadmitted_frame_in_a_batch_read_raises_with_its_file_index(tmp_path: Path):
    path = unadmitted(tmp_path)
    for index in (":", [2, 1], [2, 0, 1]):
        with pytest.raises(oxyz.SchemaError, match="Strained") as e:
            oxyz.read_batch(path, index, record=RecordBatch[MadSample])
        assert e.value.frame_index == 1
    with pytest.raises(oxyz.SchemaError) as e:
        oxyz.read_batch(path, record=RecordBatch[MadSample], conformance="strict")
    assert e.value.frame_index == 1


def test_lattice_frame_rule_sees_a_filled_cell_as_absent():
    first_molecule = next(
        i for i, f in enumerate(oxyz.read(MAD)) if "Lattice" not in f.metadata
    )
    rule = SchemaSpec(frame=FrameRule(lattice_required=True))
    with pytest.raises(oxyz.SchemaError) as e:
        oxyz.read_batch(MAD, record=RecordBatch[Molecule | Crystal], schema=rule)
    assert e.value.name == "Lattice"
    assert e.value.frame_index == first_molecule


@pytest.mark.parametrize("target", [RecordBatch[MadSample], MadBatch])
def test_union_batches_pickle(mad: RecordBatch[MadSample], target: Any):
    batch = oxyz.read_batch(MAD, record=target)
    again = pickle.loads(pickle.dumps(batch))
    assert type(again) is type(batch)
    assert again == batch
    assert again.tags.tolist() == mad.tags.tolist()


def test_single_record_batches_have_one_member_and_zero_tags():
    single = oxyz.read_batch(MAD, record=RecordBatch[MadFrame])
    assert single.members == (MadFrame,)
    assert not single.tags.any()
    assert single.where(MadFrame).all()
    assert type(single[0]) is MadFrame


def test_a_single_record_batch_resizes_through_replace_without_tags():
    single = oxyz.read_batch(MAD, [0, 1, 2], record=RecordBatch[MadFrame])
    first = single[:1]
    names = ("species", "positions", "forces", "energy", "subset")
    fields = {name: getattr(first, name) for name in names}
    resized = dataclasses.replace(
        single, offsets=first.offsets, frame_indices=first.frame_indices, **fields
    )
    assert len(resized) == 1
    assert resized.tags.tolist() == [0]
    assert resized[0] == single[0]


def test_a_union_batch_resizing_through_replace_needs_tags(
    mad: RecordBatch[MadSample],
):
    first = mad[:1]
    fields = {
        name: getattr(first, name) for name in type(first).__batch_layout__.fields
    }
    with pytest.raises(ValueError, match=r"tags: expected shape \(1,\)"):
        dataclasses.replace(
            mad,
            offsets=first.offsets,
            frame_indices=first.frame_indices,
            present={name: first.present(name) for name in ("cell", "stress")},
            **fields,
        )
    resized = dataclasses.replace(
        mad,
        offsets=first.offsets,
        frame_indices=first.frame_indices,
        present={name: first.present(name) for name in ("cell", "stress")},
        tags=first.tags.copy(),
        **fields,
    )
    assert resized == first


def test_union_batch_spec_and_from_batch():
    assert RecordBatch[MadSample].spec() == bind(MadSample).spec
    plain = oxyz.read_batch(MAD, schema=bind(MadSample).spec)
    with pytest.raises(TypeError, match="read_batch"):
        RecordBatch[MadSample].from_batch(plain)
