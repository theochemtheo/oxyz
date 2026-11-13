from __future__ import annotations

import itertools
import re
import typing
from collections import Counter
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

import oxyz
from oxyz import Cell, Column, Matrix3, Meta, Record
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
    assert "Strained" in str(e.value)


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


def test_schema_check_runs_over_the_union_keys(tmp_path: Path):
    schema = oxyz.SchemaSpec(metadata=(oxyz.MetadataRule("stress", oxyz.Kind.STR),))
    with pytest.raises(oxyz.SchemaError, match="stress") as e:
        oxyz.read(tmp_path / "missing.xyz", record=MadSample, schema=schema)
    assert e.value.name == "stress"


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
