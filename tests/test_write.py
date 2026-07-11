"""Writing frames out: lossless round-trips, the imposed order, append, stdout,
the codecs, and ASE equivalence through `oxyz.ase.from_atoms`.

The corpus round-trip is the central promise: every frame that has both a
`species` and a `pos` column survives `write` then `read` bit for bit;
those without are rejected, since the result would not be valid extxyz.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import cast

import numpy as np
import pytest

import oxyz
from oxyz import ColumnValues, Frame

DATA_DIR = Path(__file__).parent / "data"

FIXTURES = sorted(
    path.name for path in [*DATA_DIR.glob("*.xyz"), *DATA_DIR.glob("*.extxyz")]
)

has_ase = importlib.util.find_spec("ase") is not None


def arrays_equal(ours: object, theirs: object) -> bool:
    a, b = np.asarray(ours), np.asarray(theirs)
    if a.shape != b.shape:
        return False
    if a.dtype.kind == "f":
        return np.array_equal(a, b, equal_nan=True)  # bit-exact, NaN included
    return bool(np.array_equal(a, b))


def assert_frames_equivalent(originals: list[Frame], rewritten: list[Frame]) -> None:
    """Columns and metadata match as (unordered) name->value maps; write imposes
    its own order, so order is not compared, but every value must be exact."""
    assert len(originals) == len(rewritten)
    for original, frame in zip(originals, rewritten, strict=True):
        assert original.n_atoms == frame.n_atoms
        assert set(original.columns) == set(frame.columns)
        for name, values in original.columns.items():
            assert arrays_equal(values, frame.columns[name]), f"column {name!r}"
        assert set(original.metadata) == set(frame.metadata)
        for key, value in original.metadata.items():
            assert arrays_equal(value, frame.metadata[key]), f"metadata {key!r}"


def has_species_and_pos(frames: list[Frame]) -> bool:
    return all("species" in f.columns and "pos" in f.columns for f in frames)


@pytest.mark.parametrize("name", FIXTURES)
def test_corpus_round_trips_losslessly(name: str, tmp_path: Path) -> None:
    frames = oxyz.read(DATA_DIR / name)
    out = tmp_path / "out.extxyz"

    if not has_species_and_pos(frames):
        # extxyz needs species and pos; a frame without them must be refused,
        # not written into something unreadable.
        with pytest.raises(ValueError, match=r"species|pos"):
            oxyz.write(out, frames)
        return

    oxyz.write(out, frames)
    assert_frames_equivalent(frames, oxyz.read(out))


def test_reals_round_trip_bit_exact(tmp_path: Path) -> None:
    awkward = [0.1, 1.0 / 3.0, 1e-300, 1e300, -2.5e-8, np.nextafter(1.0, 2.0)]
    frame = Frame(
        n_atoms=len(awkward),
        columns={
            "species": ["H"] * len(awkward),
            "pos": np.array([[x, x, x] for x in awkward]),
        },
        metadata={"energy": awkward[1]},
    )
    out = tmp_path / "reals.extxyz"
    oxyz.write(out, frame)
    back = oxyz.read(out)[0]
    assert np.array_equal(back.columns["pos"], frame.columns["pos"])
    assert back.metadata["energy"] == frame.metadata["energy"]


def test_imposed_order_is_species_pos_then_lattice_pbc_properties(
    tmp_path: Path,
) -> None:
    frame = Frame(
        n_atoms=1,
        columns={
            "forces": np.array([[1.0, 2.0, 3.0]]),
            "pos": np.array([[0.0, 0.0, 0.0]]),
            "species": ["Fe"],
        },
        metadata={
            "config_type": "bulk",
            "pbc": np.array([True, True, True]),
            "Lattice": np.arange(9.0),
        },
    )
    out = tmp_path / "order.extxyz"
    oxyz.write(out, frame)
    lines = out.read_text().splitlines()
    assert lines[1] == (
        'Lattice="0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0" pbc="T T T" '
        "Properties=species:S:1:pos:R:3:forces:R:3 config_type=bulk"
    )
    assert lines[2] == "Fe 0.0 0.0 0.0 1.0 2.0 3.0"


@pytest.mark.parametrize("missing", ["species", "pos"])
def test_missing_required_column_is_rejected(missing: str, tmp_path: Path) -> None:
    columns: dict[str, ColumnValues] = {
        "species": ["H"],
        "pos": np.array([[0.0, 0.0, 0.0]]),
    }
    del columns[missing]
    frame = Frame(n_atoms=1, columns=columns, metadata={})
    with pytest.raises(ValueError, match=missing):
        oxyz.write(tmp_path / "x.extxyz", frame)


def test_unsupported_object_is_rejected(tmp_path: Path) -> None:
    # cast past the type checker: the point is the runtime TypeError.
    with pytest.raises(TypeError, match="cannot write"):
        oxyz.write(tmp_path / "x.extxyz", cast("Frame", 42))


def test_string_array_column_is_written(tmp_path: Path) -> None:
    # A species column handed in as a numpy string array (not a list) must cross
    # as a list and read back unchanged.
    frame = Frame(
        n_atoms=2,
        columns={"species": np.array(["O", "H"]), "pos": np.zeros((2, 3))},
        metadata={},
    )
    out = tmp_path / "strcol.extxyz"
    oxyz.write(out, frame)
    assert list(oxyz.read(out)[0].columns["species"]) == ["O", "H"]


@pytest.mark.parametrize(
    "suffix", ["xyz", "extxyz", "xyz.gz", "xyz.zip", "tar", "tar.gz"]
)
def test_every_codec_round_trips(suffix: str, tmp_path: Path) -> None:
    frames = oxyz.read(DATA_DIR / "two_frame_same_schema.xyz")
    out = tmp_path / f"frames.{suffix}"
    oxyz.write(out, frames)
    assert_frames_equivalent(frames, oxyz.read(out))


def test_zstd_write_is_rejected(tmp_path: Path) -> None:
    frames = oxyz.read(DATA_DIR / "two_frame_same_schema.xyz")
    with pytest.raises(ValueError, match="zstd"):
        oxyz.write(tmp_path / "x.xyz.zst", frames)
    with pytest.raises(ValueError, match="zstd"):
        oxyz.write(tmp_path / "x.xyz", frames, compression="zstd")


def test_level_out_of_range_is_rejected(tmp_path: Path) -> None:
    frames = oxyz.read(DATA_DIR / "two_frame_same_schema.xyz")
    with pytest.raises(ValueError, match="level"):
        oxyz.write(tmp_path / "x.xyz.gz", frames, level=12)


@pytest.mark.parametrize("suffix", ["xyz", "xyz.gz"])
def test_append_concatenates(suffix: str, tmp_path: Path) -> None:
    frames = oxyz.read(DATA_DIR / "two_frame_same_schema.xyz")
    out = tmp_path / f"a.{suffix}"
    oxyz.write(out, frames)
    oxyz.write(out, frames, append=True)
    assert len(oxyz.read(out)) == 2 * len(frames)


@pytest.mark.parametrize("suffix", ["zip", "tar", "tar.gz"])
def test_append_rejected_for_archives(suffix: str, tmp_path: Path) -> None:
    frames = oxyz.read(DATA_DIR / "two_frame_same_schema.xyz")
    with pytest.raises(ValueError, match="append"):
        oxyz.write(tmp_path / f"a.{suffix}", frames, append=True)


def test_stdout_target(capfd: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    frames = oxyz.read(DATA_DIR / "two_frame_same_schema.xyz")
    oxyz.write("-", frames)
    captured = capfd.readouterr().out
    # Round-trip the captured text through a file to confirm it is valid extxyz.
    echo = tmp_path / "echo.extxyz"
    echo.write_text(captured)
    assert_frames_equivalent(frames, oxyz.read(echo))


def test_writer_matches_one_shot(tmp_path: Path) -> None:
    frames = oxyz.read(DATA_DIR / "two_frame_same_schema.xyz")
    one_shot = tmp_path / "one.extxyz"
    incremental = tmp_path / "inc.extxyz"

    oxyz.write(one_shot, frames)
    with oxyz.Writer(incremental) as writer:
        for frame in frames:
            writer.write(frame)
    assert incremental.read_text() == one_shot.read_text()


@pytest.mark.parametrize("suffix", ["xyz", "xyz.gz", "xyz.zip", "tar.gz"])
def test_threads_produce_identical_bytes(suffix: str, tmp_path: Path) -> None:
    # The parity promise at the Python surface: output is independent of the
    # thread count, byte for byte, for every codec.
    frames = oxyz.read(DATA_DIR / "two_frame_same_schema.xyz") * 60
    # One path, reused: the archive codecs embed the file name in their header,
    # so byte-identity is only meaningful for the same path.
    out = tmp_path / f"frames.{suffix}"
    oxyz.write(out, frames, threads=1)
    reference = out.read_bytes()
    for threads in (None, 2, 8):
        oxyz.write(out, frames, threads=threads)
        assert out.read_bytes() == reference, f"threads={threads}"
    assert_frames_equivalent(frames, oxyz.read(out))


@pytest.mark.parametrize("batch", [1, 7, 1000])
def test_writer_batch_matches_streaming(batch: int, tmp_path: Path) -> None:
    frames = oxyz.read(DATA_DIR / "two_frame_same_schema.xyz") * 40
    streamed = tmp_path / "stream.extxyz"
    batched = tmp_path / "batch.extxyz"

    with oxyz.Writer(streamed) as writer:
        for frame in frames:
            writer.write(frame)
    with oxyz.Writer(batched, batch=batch) as writer:
        for frame in frames:
            writer.write(frame)
    assert batched.read_bytes() == streamed.read_bytes()


def test_writer_rejects_zero_batch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="batch"):
        oxyz.Writer(tmp_path / "x.extxyz", batch=0)


# --- ASE equivalence -------------------------------------------------------

ase_only = pytest.mark.skipif(not has_ase, reason="ase not installed")


@ase_only
def test_mixed_frame_and_atoms_iterable(tmp_path: Path) -> None:
    import oxyz.ase

    frames = oxyz.read(DATA_DIR / "two_frame_same_schema.xyz")
    atoms = oxyz.ase.read(DATA_DIR / "two_frame_same_schema.xyz", index=0)
    out = tmp_path / "mixed.extxyz"
    oxyz.write(out, [frames[0], atoms])
    assert len(oxyz.read(out)) == 2


@ase_only
def test_atoms_round_trip_matches_ase(tmp_path: Path) -> None:
    import ase.io
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator

    # ase.io.read types as Atoms | list[Atoms]; a single-frame file yields one.

    atoms = Atoms(
        "H2O",
        positions=[[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [0.0, 0.95, 0.0]],
        cell=[5.0, 6.0, 7.0],
        pbc=True,
    )
    atoms.info["config_type"] = "water"
    atoms.info["weight"] = 0.5
    atoms.calc = SinglePointCalculator(
        atoms, energy=-12.3, forces=np.arange(9.0).reshape(3, 3)
    )

    out = tmp_path / "atoms.extxyz"
    oxyz.write(out, atoms)
    back = ase.io.read(out, format="extxyz")
    assert isinstance(back, Atoms)

    assert np.array_equal(back.numbers, atoms.numbers)
    assert np.allclose(back.positions, atoms.positions)
    assert np.allclose(back.cell[:], atoms.cell[:])
    assert (back.pbc == atoms.pbc).all()
    assert back.info["config_type"] == "water"
    assert np.allclose(back.info["weight"], 0.5)
    assert np.allclose(back.get_potential_energy(), -12.3)
    assert np.allclose(back.get_forces(), atoms.get_forces())


@ase_only
@pytest.mark.parametrize(
    "name", ["minimal_periodic.extxyz", "two_frame_same_schema.xyz"]
)
def test_atoms_corpus_round_trips_through_ase(name: str, tmp_path: Path) -> None:
    """Read with ASE, write through oxyz, read with ASE again: the two ASE reads
    must agree, so `from_atoms` mirrors ASE's own write mapping."""
    import ase.io

    originals = ase.io.read(DATA_DIR / name, index=":", format="extxyz")
    out = tmp_path / "viaoxyz.extxyz"
    oxyz.write(out, originals)
    rewritten = ase.io.read(out, index=":", format="extxyz")

    assert len(originals) == len(rewritten)
    for original, frame in zip(originals, rewritten, strict=True):
        assert np.array_equal(original.numbers, frame.numbers)
        assert np.allclose(original.positions, frame.positions)
        assert np.allclose(original.cell[:], frame.cell[:])
        assert (original.pbc == frame.pbc).all()


@ase_only
def test_from_atoms_isolated_molecule_has_no_lattice() -> None:
    from ase import Atoms

    import oxyz.ase

    molecule = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    frame = oxyz.ase.from_atoms(molecule)
    assert "Lattice" not in frame.metadata
    assert "pbc" not in frame.metadata
    assert frame.columns["species"] == ["H", "H"]


@ase_only
def test_from_atoms_flattens_3x3_info_key() -> None:
    from ase import Atoms

    import oxyz.ase

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[3.0, 3.0, 3.0], pbc=True)
    atoms.info["virial"] = np.arange(9.0).reshape(3, 3)
    frame = oxyz.ase.from_atoms(atoms)
    # Stored flat (9,), and to_atoms restores the 3x3 — a round trip of the key.
    assert np.asarray(frame.metadata["virial"]).shape == (9,)
    assert oxyz.ase.to_atoms(frame).info["virial"].shape == (3, 3)


@ase_only
def test_from_atoms_fixatoms_becomes_move_mask() -> None:
    from ase import Atoms
    from ase.constraints import FixAtoms

    import oxyz.ase

    atoms = Atoms("H3", positions=[[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    atoms.set_constraint(FixAtoms(indices=[0, 2]))
    mask = np.asarray(oxyz.ase.from_atoms(atoms).columns["move_mask"])
    # move_mask is True where free, so the two fixed atoms are False.
    assert mask.tolist() == [False, True, False]


@ase_only
def test_from_atoms_rejects_unsupported_constraint() -> None:
    from ase import Atoms
    from ase.constraints import FixBondLength

    import oxyz.ase

    atoms = Atoms("H2", positions=[[0.0, 0, 0], [1.0, 0, 0]])
    atoms.set_constraint(FixBondLength(0, 1))
    with pytest.raises(ValueError, match="constraint"):
        oxyz.ase.from_atoms(atoms)
