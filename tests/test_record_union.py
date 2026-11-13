from __future__ import annotations

import itertools
import typing

import numpy as np
import numpy.typing as npt
import pytest

from oxyz import Cell, Column, Matrix3, Meta, Record
from oxyz._record_union import Key, members_of, union_plan

if typing.TYPE_CHECKING:
    from collections.abc import Callable

type Array = npt.NDArray[np.float64]

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
