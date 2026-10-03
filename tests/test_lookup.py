"""Mapping-like lookup by name on `Frame`, `Batch`, `Schema`, and `SchemaSpec`.

`obj[name]` searches columns then metadata, so a caller need not know the two
live apart. A name on both sides is an `AmbiguousNameError` — never a silent
pick, and never a `KeyError` that `get` or `except KeyError` could swallow.
None of the four is a `Mapping`: a `Mapping` needs `__len__` and `__iter__`,
and numpy then reads a list of frames as a list of key names.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest

import oxyz
from oxyz import (
    AmbiguousNameError,
    ColumnRule,
    ColumnSchema,
    Frame,
    Kind,
    MetadataRule,
    MetadataSchema,
    SchemaSpec,
)
from oxyz._schema import column_sig
from oxyz._schema_match import compile_spec, validate_frame

DATA_DIR = Path(__file__).parent / "data"
TWO_FRAME = DATA_DIR / "two_frame_same_schema.xyz"  # species/pos/forces; energy, Time

# `charge` is both a per-atom column and a comment-line key.
CLASHING = """\
2
Properties=species:S:1:pos:R:3:charge:R:1 charge=0.0
H 0.0 0.0 0.0 0.4
H 1.0 0.0 0.0 -0.4
"""


@pytest.fixture
def clashing(tmp_path: Path) -> Path:
    path = tmp_path / "clashing.extxyz"
    path.write_text(CLASHING)
    return path


@pytest.fixture
def frame() -> Frame:
    return oxyz.read(TWO_FRAME, 0)


def test_frame_lookup_finds_columns_and_metadata_as_stored(frame: Frame) -> None:
    assert frame["pos"] is frame.columns["pos"]
    assert frame["energy"] is frame.metadata["energy"]


def test_frame_lookup_of_an_absent_name_is_a_key_error(frame: Frame) -> None:
    with pytest.raises(KeyError, match="'forcez'"):
        frame["forcez"]


def test_frame_lookup_of_a_clashing_name_is_ambiguous(clashing: Path) -> None:
    clash = oxyz.read(clashing, 0)

    with pytest.raises(AmbiguousNameError, match=r"'charge'.*columns.*metadata") as err:
        clash["charge"]
    assert err.value.name == "charge"
    assert not isinstance(err.value, KeyError)
    assert isinstance(err.value, oxyz.OxyzError)


def test_frame_get_defaults_only_when_absent(frame: Frame, clashing: Path) -> None:
    assert frame.get("forces") is frame.columns["forces"]
    assert frame.get("forcez") is None
    assert frame.get("forcez", 0.0) == 0.0
    with pytest.raises(AmbiguousNameError):
        oxyz.read(clashing, 0).get("charge", 0.0)


def test_frame_contains_either_side_without_raising(clashing: Path) -> None:
    clash = oxyz.read(clashing, 0)

    assert "pos" in clash
    assert "charge" in clash
    assert "forces" not in clash


def test_frame_keys_list_columns_then_metadata_once(clashing: Path) -> None:
    assert oxyz.read(TWO_FRAME, 0).keys() == [
        "species",
        "pos",
        "forces",
        "Lattice",
        "energy",
        "Time",
    ]
    assert oxyz.read(clashing, 0).keys() == ["species", "pos", "charge"]


def test_frame_unpacks_into_a_dict(frame: Frame) -> None:
    flat = dict(frame)

    assert list(flat) == frame.keys()
    assert flat["energy"] is frame.metadata["energy"]


def test_frame_rejects_a_non_string_key(frame: Frame) -> None:
    with pytest.raises(TypeError, match="names"):
        frame[0]  # ty: ignore[invalid-argument-type]


def test_frame_has_no_len_and_is_not_iterable(frame: Frame) -> None:
    # len(frame) would collide with the lookup: see the numpy test below.
    assert not hasattr(frame, "__len__")
    assert not isinstance(frame, Iterable)
    with pytest.raises(TypeError):
        iter(frame)  # ty: ignore[no-matching-overload]


def test_numpy_keeps_a_list_of_frames_as_frames() -> None:
    # With __getitem__ and __len__ both defined, numpy reads a frame as a
    # sequence: these raised, or — for a Mapping — returned key names.
    frames = oxyz.read(TWO_FRAME)
    rng = np.random.default_rng(0)

    # numpy's stubs type these as ArrayLike-only; at runtime any list works.
    as_objects = np.array(frames, dtype=object)
    assert as_objects.shape == (2,)
    assert all(isinstance(f, Frame) for f in as_objects)
    assert all(isinstance(f, Frame) for f in rng.choice(frames, 2))  # ty: ignore[no-matching-overload]
    assert all(isinstance(f, Frame) for f in rng.permutation(frames))  # ty: ignore[no-matching-overload]


def test_batch_lookup_matches_frame_lookup() -> None:
    batch = oxyz.read_batch(TWO_FRAME)

    assert batch["pos"] is batch.columns["pos"]
    assert batch["energy"] is batch.metadata["energy"]
    assert "Time" in batch
    assert batch.get("forcez") is None
    assert batch.keys() == ["species", "pos", "forces", "Lattice", "energy", "Time"]
    with pytest.raises(KeyError):
        batch["forcez"]


def test_batch_lookup_of_a_clashing_name_is_ambiguous(clashing: Path) -> None:
    with pytest.raises(AmbiguousNameError, match="'charge'"):
        oxyz.read_batch(clashing)["charge"]


def test_schema_lookup_returns_the_entry_for_a_name() -> None:
    schema = oxyz.infer_schema(TWO_FRAME)

    pos = schema["pos"]
    energy = schema["energy"]
    assert isinstance(pos, ColumnSchema)
    assert pos is schema.columns[1]
    assert isinstance(energy, MetadataSchema)
    assert energy.key == "energy"
    assert "forces" in schema
    assert schema.get("forcez") is None
    assert schema.keys() == ["species", "pos", "forces", "Lattice", "energy", "Time"]
    with pytest.raises(KeyError):
        schema["forcez"]
    assert not isinstance(schema, Iterable)


def test_schema_lookup_of_a_clashing_name_is_ambiguous(clashing: Path) -> None:
    schema = oxyz.infer_schema(clashing)

    assert "charge" in schema
    with pytest.raises(AmbiguousNameError, match="'charge'"):
        schema["charge"]


SPEC = SchemaSpec(
    columns=(
        ColumnRule("pos", Kind.REAL, width=3),
        ColumnRule("REF_1", Kind.REAL, width=1),
        ColumnRule("REF_*", Kind.REAL, width=3),
        ColumnRule("REF_?", Kind.INT, width=2),
        ColumnRule("re:desc_[0-9]+", Kind.REAL, width=4),
    ),
    metadata=(
        MetadataRule("energy", Kind.REAL),
        MetadataRule("REF_*", Kind.REAL, shape=(6,)),
    ),
)


def test_spec_lookup_is_by_exact_rule_name() -> None:
    assert SPEC["pos"] is SPEC.columns[0]
    assert SPEC["energy"] is SPEC.metadata[0]
    assert SPEC["re:desc_[0-9]+"] is SPEC.columns[4]
    assert "REF_2" not in SPEC
    assert SPEC.get("REF_2") is None
    with pytest.raises(KeyError):
        SPEC["REF_2"]
    assert SPEC.keys() == ["pos", "REF_1", "REF_*", "REF_?", "re:desc_[0-9]+", "energy"]
    assert not isinstance(SPEC, Iterable)


def test_spec_lookup_of_a_rule_on_both_axes_is_ambiguous() -> None:
    with pytest.raises(AmbiguousNameError, match="'REF_\\*'"):
        SPEC["REF_*"]


@pytest.mark.parametrize(
    ("name", "governing"),
    [
        ("pos", SPEC.columns[0]),  # a literal
        ("REF_1", SPEC.columns[1]),  # a literal beats the patterns that match it
        ("REF_2", SPEC.columns[2]),  # the first matching pattern wins
        ("REF_long", SPEC.columns[2]),
        ("desc_12", SPEC.columns[4]),  # a regex
        ("desc_x", None),
        ("x_desc_12", None),  # a regex anchors at the start, as validation does
        ("forces", None),
    ],
)
def test_rule_for_a_column_is_the_rule_validation_applies(
    name: str, governing: ColumnRule | None
) -> None:
    assert SPEC.rule_for(name, axis="column") is governing


def test_rule_for_without_an_axis_searches_both() -> None:
    assert SPEC.rule_for("pos") is SPEC.columns[0]
    assert SPEC.rule_for("energy") is SPEC.metadata[0]
    assert SPEC.rule_for("forces") is None
    assert SPEC.rule_for("REF_x", axis="metadata") is SPEC.metadata[1]
    with pytest.raises(AmbiguousNameError, match=r"'REF_x'.*axis"):
        SPEC.rule_for("REF_x")


@pytest.mark.parametrize("name", ["pos", "REF_1", "REF_2", "REF_long", "desc_12"])
def test_rule_for_agrees_with_validation(name: str) -> None:
    # Give the column a width no rule declares, so validation reports a
    # mismatch whose `expected` names the rule that claimed it.
    frame = Frame(
        n_atoms=1,
        columns={"pos": np.zeros((1, 3)), name: np.zeros((1, 7))},
        metadata={},
    )
    violations = validate_frame(frame, compile_spec(SPEC), "required")
    [mismatch] = [v for v in violations if v.name == name]
    rule = SPEC.rule_for(name, axis="column")

    assert isinstance(rule, ColumnRule)
    assert mismatch.expected == column_sig(rule.kind, rule.width)
