"""The `Frame`/`Batch` surface that is not the read path: repr, equality,
`len`, and the derived well-known-field accessors.

The two dataclasses hold dicts of numpy arrays, so the generated `__repr__`
would print the data and the generated `__eq__` would raise on an array. Both
are replaced; these tests pin the replacements and the accessors that resolve
`pos`/`Z`/`species`/`Lattice`/`pbc` without touching what the frame stores.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import oxyz
from oxyz import ColumnValues, Frame, MetadataValue

DATA_DIR = Path(__file__).parent / "data"
PERIODIC = DATA_DIR / "minimal_periodic.extxyz"  # species + pos, Lattice, no pbc
MOLECULE = DATA_DIR / "no_lattice_molecule.xyz"  # no Lattice, no pbc
Z_ONLY = DATA_DIR / "atomic_numbers_z.extxyz"  # a Z column, no species
PBC_TTF = DATA_DIR / "periodic_pbc_ttf.extxyz"  # an explicit mixed pbc
TWO_FRAME = DATA_DIR / "two_frame_same_schema.xyz"


def frame(**metadata: MetadataValue) -> Frame:
    """A one-atom frame carrying the given metadata."""
    return Frame(
        n_atoms=1,
        columns={"species": ["H"], "pos": np.zeros((1, 3))},
        metadata=dict(metadata),
    )


def test_repr_summarises_shape_and_prints_no_data() -> None:
    text = repr(oxyz.read(PERIODIC, 0))

    assert text.startswith("Frame(n_atoms=2, ")
    assert "'species': str[2]" in text
    assert "'pos': float64[2, 3]" in text
    assert "'Lattice': float64[9]" in text
    # The fixture's coordinates and cell lengths; none may appear in the repr.
    assert "1.5" not in text
    assert "3.0" not in text


def test_repr_is_bounded_for_a_huge_frame() -> None:
    n_atoms = 100_000
    huge = Frame(
        n_atoms=n_atoms,
        columns={"species": ["Si"] * n_atoms, "pos": np.zeros((n_atoms, 3))},
        metadata={"Lattice": np.zeros(9)},
    )

    # A dataclass repr of the same frame is ~500 KB; the summary form is ~120
    # chars, so this bound also fails anything that starts printing elements.
    assert len(repr(huge)) < 200


def test_repr_elides_past_eight_entries() -> None:
    crowded = Frame(
        n_atoms=1,
        columns={"pos": np.zeros((1, 3))},
        metadata={f"key{i}": float(i) for i in range(12)},
    )

    text = repr(crowded)
    assert "'key7': float" in text
    assert "'key8'" not in text
    assert "+4 more" in text


def test_repr_reports_the_shape_of_a_two_dimensional_string_column() -> None:
    # A `species:S:2`-style column crosses as list[list[str]].
    paired = Frame(
        n_atoms=2,
        columns={"labels": [["a", "b"], ["c", "d"]]},
        metadata={},
    )

    assert "'labels': str[2, 2]" in repr(paired)
    assert "'a'" not in repr(paired)


def test_batch_repr_names_the_layout_and_prints_no_data() -> None:
    text = repr(oxyz.read_batch(TWO_FRAME))

    assert text.startswith("Batch(n_frames=2, total_atoms=4, n_atoms=2..2, ")
    assert "'pos': float64[4, 3]" in text
    assert "'energy': float64[2]" in text
    assert "-6.45" not in text


def test_batch_repr_spans_uneven_frames_and_omits_an_empty_range(
    tmp_path: Path,
) -> None:
    uneven = DATA_DIR / "varying_atom_counts.xyz"  # 3, 1, 2 atoms
    empty = tmp_path / "empty.xyz"
    empty.touch()

    assert repr(oxyz.read_batch(uneven)).startswith(
        "Batch(n_frames=3, total_atoms=6, n_atoms=1..3, "
    )
    assert repr(oxyz.read_batch(empty)) == (
        "Batch(n_frames=0, total_atoms=0, columns={}, metadata={})"
    )


def test_reading_the_same_file_twice_gives_equal_frames() -> None:
    assert oxyz.read(PERIODIC, 0) == oxyz.read(PERIODIC, 0)
    assert oxyz.read(PERIODIC) == oxyz.read(PERIODIC)  # list-wise, element by element


def test_frames_differing_anywhere_are_unequal() -> None:
    original = oxyz.read(PERIODIC, 0)

    moved = dict(original.columns)
    moved["pos"] = original.positions + 1.0
    assert original != Frame(original.n_atoms, moved, original.metadata)

    extra = {**original.metadata, "energy": -1.0}
    assert original != Frame(original.n_atoms, original.columns, extra)

    assert original != Frame(original.n_atoms + 1, original.columns, original.metadata)


def test_equality_against_a_non_frame_defers_to_python() -> None:
    one = oxyz.read(PERIODIC, 0)

    assert one.__eq__(42) is NotImplemented  # so Python falls back to identity
    assert one != 42


def test_nan_columns_equal_themselves() -> None:
    with_nan = Frame(
        n_atoms=1,
        columns={"species": ["H"], "forces": np.array([[np.nan, 0.0, 0.0]])},
        metadata={},
    )
    twin = Frame(
        n_atoms=1,
        columns={"species": ["H"], "forces": np.array([[np.nan, 0.0, 0.0]])},
        metadata={},
    )

    assert with_nan == twin


def test_equality_ignores_key_order() -> None:
    forward = Frame(
        n_atoms=1,
        columns={"species": ["H"], "pos": np.zeros((1, 3))},
        metadata={},
    )
    reversed_ = Frame(
        n_atoms=1,
        columns={"pos": np.zeros((1, 3)), "species": ["H"]},
        metadata={},
    )

    assert forward == reversed_


def test_equality_is_false_rather_than_raising_on_hostile_values() -> None:
    """`__eq__` must return a bool for anything a hand-built frame can hold."""
    # Same key on both sides, so the comparison reaches the value itself: numpy
    # cannot coerce a ragged list at all, and it must not escape as an error.
    ragged = Frame(
        n_atoms=2, columns={}, metadata={"odd": cast("MetadataValue", [[1, 2], [3]])}
    )
    square = Frame(
        n_atoms=2, columns={}, metadata={"odd": cast("MetadataValue", [[1, 2], [3, 4]])}
    )

    assert ragged != square
    assert square != ragged  # symmetric: the ragged side may be either operand
    assert ragged == ragged  # noqa: PLR0124  reflexivity is the contract here

    # An object array of arrays: elementwise comparison has no unambiguous truth.
    boxed = np.empty(2, dtype=object)
    boxed[0], boxed[1] = np.array([1, 2]), np.array([3, 4])
    left = Frame(n_atoms=2, columns={"boxed": boxed}, metadata={})
    right = Frame(n_atoms=2, columns={"boxed": boxed.copy()}, metadata={})

    assert left == left  # noqa: PLR0124  reflexivity is the contract here
    assert isinstance(left == right, bool)  # not demonstrably equal, but no raise


def test_columns_of_differing_shape_are_unequal() -> None:
    flat = Frame(n_atoms=2, columns={"pos": np.zeros(6)}, metadata={})
    shaped = Frame(n_atoms=2, columns={"pos": np.zeros((2, 3))}, metadata={})

    assert flat != shaped


def test_frames_are_unhashable() -> None:
    with pytest.raises(TypeError, match="unhashable type: 'Frame'"):
        hash(oxyz.read(PERIODIC, 0))


def test_batch_equality_covers_the_layout_arrays() -> None:
    batch = oxyz.read_batch(TWO_FRAME)

    assert batch == oxyz.read_batch(TWO_FRAME)
    shifted = oxyz.Batch(
        columns=batch.columns,
        metadata=batch.metadata,
        offsets=np.array([0, 1, 4]),
        frame_indices=batch.frame_indices,
    )
    assert batch != shifted
    regathered = oxyz.Batch(
        columns=batch.columns,
        metadata=batch.metadata,
        offsets=batch.offsets,
        frame_indices=np.array([1, 0]),
    )
    assert batch != regathered
    retagged = oxyz.Batch(
        columns=batch.columns,
        metadata={**batch.metadata, "energy": np.zeros(batch.n_frames)},
        offsets=batch.offsets,
        frame_indices=batch.frame_indices,
    )
    assert batch != retagged
    assert batch != 42


def test_len_is_the_atom_count_and_does_not_make_a_frame_iterable() -> None:
    # MOLECULE has 3 atoms but 2 columns, so this cannot pass on a len() that
    # returns the column count.
    molecule = oxyz.read(MOLECULE, 0)

    assert len(molecule) == molecule.n_atoms == 3
    assert len(molecule.columns) == 2
    # oxyz.write dispatches on Iterable; a Frame must not look like a sequence.
    assert not isinstance(molecule, Iterable)


def test_accessors_on_a_periodic_frame() -> None:
    one = oxyz.read(PERIODIC, 0)

    assert one.positions is one.columns["pos"]
    assert one.numbers.dtype == np.int32
    assert_array_equal(one.numbers, [14, 14])
    assert one.symbols is one.columns["species"] == ["Si", "Si"]
    assert one.cell.shape == (3, 3)
    lattice = np.asarray(one.metadata["Lattice"])
    assert_array_equal(one.cell, lattice.reshape((3, 3), order="F").T)
    assert one.pbc.tolist() == [True, True, True]  # Lattice present, pbc absent


def test_accessors_leave_the_stored_dicts_alone() -> None:
    one = oxyz.read(PERIODIC, 0)
    metadata_before = {key: np.asarray(value) for key, value in one.metadata.items()}
    columns_before = set(one.columns)

    _ = one.cell, one.pbc, one.numbers, one.symbols, one.positions

    assert set(one.columns) == columns_before  # no derived key written back
    assert set(one.metadata) == set(metadata_before)
    for key, value in metadata_before.items():
        assert_array_equal(np.asarray(one.metadata[key]), value)
    assert_array_equal(one.columns["pos"], [[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]])
    assert one.columns["species"] == ["Si", "Si"]


def test_explicit_pbc_wins() -> None:
    assert oxyz.read(PBC_TTF, 0).pbc.tolist() == [True, True, False]


def test_scalar_pbc_broadcasts_to_three_axes() -> None:
    assert frame(pbc=True).pbc.tolist() == [True, True, True]


def test_symbols_round_trip_from_a_z_only_frame() -> None:
    z_only = oxyz.read(Z_ONLY, 0)

    assert "species" not in z_only.columns
    assert z_only.symbols == ["Si", "Si"]


def test_cell_transposes_a_triclinic_lattice_into_row_vectors() -> None:
    # A symmetric Lattice cannot tell the two conventions apart; this one can.
    # extxyz stores the ASE cell flattened in Fortran order, so the row vectors
    # come back as consecutive triples of the flat array.
    triclinic = frame(Lattice=np.arange(9.0))

    assert_array_equal(triclinic.cell, [[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    assert not np.array_equal(triclinic.cell, triclinic.cell.T)  # not self-inverse


def test_an_explicit_z_column_outranks_species() -> None:
    # `species` is often a force-field label; an explicit Z is authoritative.
    disagreeing = Frame(
        n_atoms=1,
        columns={"species": ["H"], "Z": np.array([14], dtype=np.int32)},
        metadata={},
    )

    assert_array_equal(disagreeing.numbers, [14])
    assert disagreeing.symbols == ["H"]  # symbols still prefer the stored column


def test_a_float_z_column_rounds_rather_than_truncates() -> None:
    assert_array_equal(
        Frame(n_atoms=2, columns={"Z": np.array([5.6, 7.4])}, metadata={}).numbers,
        [6, 7],
    )


def test_a_non_numeric_z_column_raises_field_error() -> None:
    # Reachable from a valid file: `Properties=...:Z:S:1` gives a string column.
    z_strings = Frame(n_atoms=1, columns={"Z": ["x"]}, metadata={})

    with pytest.raises(oxyz.FieldError, match="'Z' holds no atomic numbers"):
        _ = z_strings.numbers


def test_a_scalar_species_column_raises_field_error() -> None:
    scalar = Frame(
        n_atoms=1, columns=cast("dict[str, ColumnValues]", {"species": 42}), metadata={}
    )

    with pytest.raises(oxyz.FieldError, match="not a per-atom column"):
        _ = scalar.symbols


def test_a_non_numeric_lattice_raises_field_error() -> None:
    with pytest.raises(oxyz.FieldError, match="no numeric cell"):
        _ = frame(Lattice=["a"] * 9).cell


def test_an_atomic_number_with_no_symbol_raises_field_error() -> None:
    beyond = Frame(n_atoms=2, columns={"Z": np.array([999, -1])}, metadata={})

    with pytest.raises(oxyz.FieldError, match=r"no chemical symbol: \[-1, 999\]"):
        _ = beyond.symbols


def test_ragged_lattice_and_pbc_raise_field_error() -> None:
    # numpy cannot coerce these at all; the failure must still be an oxyz error.
    ragged = cast("MetadataValue", [[1, 2], [3]])

    with pytest.raises(oxyz.FieldError, match="Lattice is not an array"):
        _ = frame(Lattice=ragged).cell
    with pytest.raises(oxyz.FieldError, match="pbc is not a boolean value"):
        _ = frame(pbc=ragged).pbc


def test_a_multi_component_species_column_has_no_symbols() -> None:
    # A `species:S:2` column crosses as list[list[str]]: no one symbol per atom.
    paired = Frame(n_atoms=1, columns={"species": [["Si", "a"]]}, metadata={})

    with pytest.raises(oxyz.FieldError, match="multi-component"):
        _ = paired.symbols


def test_a_frame_without_a_lattice_has_a_zero_cell_and_no_pbc() -> None:
    molecule = oxyz.read(MOLECULE, 0)

    assert_array_equal(molecule.cell, np.zeros((3, 3)))
    assert molecule.pbc.tolist() == [False, False, False]


def test_a_malformed_lattice_raises_field_error() -> None:
    with pytest.raises(oxyz.FieldError, match="9 components"):
        _ = frame(Lattice=np.zeros(6)).cell


def test_a_malformed_pbc_raises_field_error() -> None:
    with pytest.raises(oxyz.FieldError, match="pbc must be a scalar or 3"):
        _ = frame(pbc=np.array([True, False])).pbc


def test_a_frame_without_positions_raises_field_error() -> None:
    bare = Frame(n_atoms=1, columns={"species": ["H"]}, metadata={})

    with pytest.raises(oxyz.FieldError, match="no 'pos' column"):
        _ = bare.positions


def test_a_frame_without_species_or_numbers_raises_field_error() -> None:
    bare = Frame(n_atoms=1, columns={"pos": np.zeros((1, 3))}, metadata={})

    with pytest.raises(oxyz.FieldError, match="no 'species', 'Z', or 'numbers'"):
        _ = bare.numbers
    with pytest.raises(oxyz.FieldError, match="no 'species', 'Z', or 'numbers'"):
        _ = bare.symbols


def test_field_error_is_an_oxyz_error() -> None:
    assert issubclass(oxyz.FieldError, oxyz.OxyzError)
    assert issubclass(oxyz.FieldError, ValueError)


def test_writing_a_short_column_is_refused(tmp_path: Path) -> None:
    # Shorter than n_atoms: the Rust encoder indexes row * width and panics.
    short = Frame(
        n_atoms=3,
        columns={"species": ["H"], "pos": np.zeros((1, 3))},
        metadata={},
    )

    with pytest.raises(oxyz.FieldError, match=r"'species' has 1 rows.*n_atoms=3"):
        oxyz.write(tmp_path / "out.extxyz", short)


def test_writing_a_long_column_is_refused(tmp_path: Path) -> None:
    # Longer than n_atoms: silently truncated to the first row otherwise.
    long = Frame(
        n_atoms=1,
        columns={"species": ["H"] * 3, "pos": np.zeros((3, 3))},
        metadata={},
    )

    with pytest.raises(oxyz.FieldError, match=r"'species' has 3 rows.*n_atoms=1"):
        oxyz.write(tmp_path / "out.extxyz", long)


def test_a_zero_dimensional_column_is_one_row(tmp_path: Path) -> None:
    # `_rust.pyi` documents scalar-width-1 arrays as a legal column, and the
    # encoder writes one from a 0-D value, so it is legal at n_atoms == 1.
    out = tmp_path / "out.extxyz"
    single = Frame(
        n_atoms=1,
        columns={"species": ["H"], "pos": np.zeros((1, 3)), "energy": np.array(2.5)},
        metadata={},
    )

    oxyz.write(out, single)
    assert_array_equal(oxyz.read(out, 0).columns["energy"], [2.5])

    # ...and one row is not enough for a two-atom frame.
    pair = Frame(
        n_atoms=2,
        columns={
            "species": ["H", "H"],
            "pos": np.zeros((2, 3)),
            "energy": np.array(2.5),
        },
        metadata={},
    )
    with pytest.raises(oxyz.FieldError, match=r"'energy' has 1 rows.*n_atoms=2"):
        oxyz.write(out, pair)


def test_writing_a_column_that_is_not_a_sequence_is_refused(tmp_path: Path) -> None:
    # Deliberately off-contract: a bare float is not a ColumnValues.
    bad = Frame(
        n_atoms=1,
        columns=cast(
            "dict[str, ColumnValues]",
            {"species": ["H"], "pos": np.zeros((1, 3)), "energy": 2.5},
        ),
        metadata={},
    )

    with pytest.raises(oxyz.FieldError, match=r"'energy' is not a per-atom sequence"):
        oxyz.write(tmp_path / "out.extxyz", bad)
