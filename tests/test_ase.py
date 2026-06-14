"""Golden tests: oxyz.ase.read must agree with ase.io.read.

The fixture corpus is read by both readers and compared field by field;
documented divergences (Voigt stress) get their own explicit tests.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("ase") is None, reason="ase not installed"
)

DATA_DIR = Path(__file__).parent / "data"

# Documented divergences from ase.io.read, each asserted explicitly below:
# Voigt stress (ASE rejects it), new-style string arrays (ASE leaves them as
# one raw string), and single-quoted values (ASE strips the quotes; oxyz keeps
# them per the grammar). See the "Divergences from ASE" section of the README.
VOIGT_STRESS = {"simple.extxyz", "nonorthogonal.extxyz", "stress_voigt6.extxyz"}
DIVERGENT = VOIGT_STRESS | {
    "newstyle_array_metadata.extxyz",
    "singlequote_metadata.extxyz",
}

GOLDEN = sorted(
    path.name
    for path in list(DATA_DIR.glob("*.xyz")) + list(DATA_DIR.glob("*.extxyz"))
    if path.name not in DIVERGENT
)


def assert_values_equal(ours, theirs, context: str) -> None:
    ours, theirs = np.asarray(ours), np.asarray(theirs)
    assert ours.shape == theirs.shape, context
    if theirs.dtype.kind in "fc":
        assert np.allclose(ours.astype(float), theirs.astype(float)), context
    else:
        assert (ours == theirs).all(), context


def assert_atoms_match(ours, theirs) -> None:
    assert np.array_equal(ours.numbers, theirs.numbers)
    assert np.allclose(ours.positions, theirs.positions)
    assert np.allclose(ours.cell[:], theirs.cell[:])
    assert (ours.pbc == theirs.pbc).all()

    assert set(ours.arrays) == set(theirs.arrays)
    for key, value in theirs.arrays.items():
        assert_values_equal(ours.arrays[key], value, f"arrays[{key!r}]")

    assert set(ours.info) == set(theirs.info)
    for key, value in theirs.info.items():
        if isinstance(value, np.ndarray):
            assert_values_equal(ours.info[key], value, f"info[{key!r}]")
        else:
            assert ours.info[key] == value, f"info[{key!r}]"

    assert (ours.calc is None) == (theirs.calc is None)
    if theirs.calc is not None:
        assert set(ours.calc.results) == set(theirs.calc.results)
        for key, value in theirs.calc.results.items():
            assert_values_equal(ours.calc.results[key], value, f"results[{key!r}]")

    assert len(ours.constraints) == len(theirs.constraints)
    for c_ours, c_theirs in zip(ours.constraints, theirs.constraints, strict=False):
        assert repr(c_ours) == repr(c_theirs)


@pytest.mark.parametrize("name", GOLDEN)
def test_matches_ase_reader(name: str) -> None:
    import ase.io

    import oxyz.ase

    ours = oxyz.ase.read(DATA_DIR / name, index=":")
    theirs = ase.io.read(DATA_DIR / name, index=":", format="extxyz")
    assert len(ours) == len(theirs)
    for frame_ours, frame_theirs in zip(ours, theirs, strict=False):
        assert_atoms_match(frame_ours, frame_theirs)


def test_trailing_blank_line_matches_ase(tmp_path: Path) -> None:
    """A trailing blank line is end of input for both readers."""
    import ase.io

    import oxyz.ase

    # Built at runtime: a committed trailing blank would be stripped by the
    # end-of-file hook, silently voiding the test (a clean file reads the
    # same one frame).
    clean = (DATA_DIR / "minimal_periodic.extxyz").read_text()
    path = tmp_path / "trailing.extxyz"
    path.write_text(clean + "\n")

    ours = oxyz.ase.read(path, index=":")
    theirs = ase.io.read(path, index=":", format="extxyz")
    assert len(ours) == len(theirs) == 1
    assert_atoms_match(ours[0], theirs[0])


@pytest.mark.parametrize("name", sorted(VOIGT_STRESS))
def test_voigt_stress_diverges_from_ase(name: str) -> None:
    """ase.io.read rejects 6-component stress; we accept it as already-Voigt."""
    import ase.io

    import oxyz.ase

    path = DATA_DIR / name
    with pytest.raises(ValueError, match="3x3"):
        ase.io.read(path, format="extxyz")

    atoms = oxyz.ase.read(path, index=0)
    assert atoms.calc is not None
    assert atoms.calc.results["stress"].shape == (6,)


def test_newstyle_string_array_diverges_from_ase() -> None:
    """ASE keeps `tags=["a","b"]` as one raw string; we type it as a list."""
    import ase.io

    import oxyz.ase

    path = DATA_DIR / "newstyle_array_metadata.extxyz"
    ours = oxyz.ase.read(path, index=0)
    theirs = ase.io.read(path, index=0, format="extxyz")
    # ase.io.read types as Atoms | list[Atoms] even for an int index; narrow.
    assert not isinstance(theirs, list)

    assert ours.info["tags"] == ["slab", "relaxed"]
    assert theirs.info["tags"] == '"slab","relaxed"'
    for key in ("kpoints", "cutoffs"):
        assert_values_equal(ours.info[key], theirs.info[key], f"info[{key!r}]")


def test_singlequote_values_diverge_from_ase() -> None:
    """`"` is the only quote character in the grammar, so single quotes are
    ordinary bare characters: oxyz keeps them, while ASE treats `'` as a quote,
    stripping it and reading `it's` as `its`."""
    import ase.io

    import oxyz.ase

    path = DATA_DIR / "singlequote_metadata.extxyz"
    ours = oxyz.ase.read(path, index=0)
    theirs = ase.io.read(path, index=0, format="extxyz")
    assert not isinstance(theirs, list)

    assert ours.info["label"] == "'hello'"
    assert theirs.info["label"] == "hello"
    assert ours.info["note"] == "it's"
    assert theirs.info["note"] == "its"


def test_iread_matches_ase_iread() -> None:
    import ase.io

    import oxyz.ase

    path = DATA_DIR / "varying_atom_counts.xyz"
    ours = oxyz.ase.iread(path)
    theirs = ase.io.iread(path, index=":", format="extxyz")
    for atoms_ours, atoms_theirs in zip(ours, theirs, strict=True):
        assert_atoms_match(atoms_ours, atoms_theirs)


def test_reads_are_lazy_past_the_requested_frames(tmp_path: Path) -> None:
    """Like ase.io.read, a malformed later frame goes unnoticed."""
    import oxyz.ase

    text = (DATA_DIR / "varying_atom_counts.xyz").read_text()
    broken = tmp_path / "broken.xyz"
    broken.write_text(text + "not-a-count\n")

    assert oxyz.ase.read(broken, index=0).get_global_number_of_atoms() == 3
    assert len(oxyz.ase.read(broken, index="0:2")) == 2
    iterator = oxyz.ase.iread(broken)
    assert next(iterator) is not None
    with pytest.raises(ValueError, match="frame 3"):
        list(iterator)


def test_read_default_index_is_last_frame() -> None:
    import oxyz.ase

    path = DATA_DIR / "varying_atom_counts.xyz"
    all_frames = oxyz.ase.read(path, index=":")
    assert_atoms_match(oxyz.ase.read(path), all_frames[-1])


def test_read_negative_and_out_of_range_indices() -> None:
    import oxyz.ase

    path = DATA_DIR / "varying_atom_counts.xyz"
    all_frames = oxyz.ase.read(path, index=":")
    assert_atoms_match(oxyz.ase.read(path, index=-3), all_frames[0])
    with pytest.raises(IndexError, match="3 frames"):
        oxyz.ase.read(path, index=-4)
    with pytest.raises(IndexError):
        oxyz.ase.read(path, index=5)


# The full grammar a drop-in must accept: ints, int strings, slice strings
# (with negatives and steps), and slice objects.
@pytest.mark.parametrize(
    "index",
    [0, 2, -1, "1", "1:", ":2", ":-1", "-2:", "::2", "1::2", slice(-2, None)],
)
def test_index_grammar_matches_ase(index) -> None:
    import ase.io

    import oxyz.ase

    path = DATA_DIR / "varying_atom_counts.xyz"
    ours = oxyz.ase.read(path, index=index)
    theirs = ase.io.read(path, index=index, format="extxyz")

    if isinstance(theirs, list):
        assert isinstance(ours, list)
        assert len(ours) == len(theirs)
        for atoms_ours, atoms_theirs in zip(ours, theirs, strict=True):
            assert_atoms_match(atoms_ours, atoms_theirs)
    else:
        assert not isinstance(ours, list)
        assert_atoms_match(ours, theirs)


def test_reverse_slice_reads_backwards() -> None:
    import oxyz.ase

    path = DATA_DIR / "varying_atom_counts.xyz"
    forward = oxyz.ase.read(path, index=":")
    for atoms_reversed, atoms in zip(
        oxyz.ase.read(path, index="::-1"), reversed(forward), strict=True
    ):
        assert_atoms_match(atoms_reversed, atoms)


def test_non_chemical_species_is_strict_error(tmp_path: Path) -> None:
    import oxyz.ase

    path = tmp_path / "bad_species.extxyz"
    path.write_text("1\nProperties=species:S:1:pos:R:3\nQq 0 0 0\n")
    with pytest.raises(oxyz.ase.ToAseError, match="Qq"):
        oxyz.ase.read(path, index=0)


def test_frame_to_ase_method() -> None:
    import ase.io

    import oxyz

    path = DATA_DIR / "minimal_periodic.extxyz"
    atoms = oxyz.read_first_frame(path).to_ase()
    assert_atoms_match(atoms, ase.io.read(path, index=0, format="extxyz"))


def test_read_rejects_other_formats() -> None:
    import oxyz.ase

    with pytest.raises(ValueError, match="extxyz"):
        oxyz.ase.read(DATA_DIR / "simple.extxyz", format="vasp")
