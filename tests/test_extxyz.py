from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import oxyz

DATA_DIR = Path(__file__).parent / "data"

CORPUS = sorted(path for ext in ("*.xyz", "*.extxyz") for path in DATA_DIR.glob(ext))


def as_array(value: object) -> np.ndarray:
    """Assert that ``value`` is an ndarray and re-type it for ty.

    Exists only because of a current ty limitation: isinstance-narrowing
    ``np.ndarray`` out of a union yields a type that fails numpy's
    ``assert_allclose`` overloads, even though a plain ``np.ndarray`` passes.
    Delete this helper (plain ``assert isinstance`` is enough) once
    test_ty_canary.py fails.
    """
    assert isinstance(value, np.ndarray)
    return value


@pytest.mark.parametrize("path", CORPUS, ids=lambda path: path.name)
def test_every_fixture_converts_to_python(path: Path) -> None:
    frame = oxyz.read(path, 0)

    assert frame.n_atoms > 0
    assert frame.columns
    for values in frame.columns.values():
        assert len(values) == frame.n_atoms


def test_read_frames_trajectory() -> None:
    frames = oxyz.read(DATA_DIR / "varying_atom_counts.xyz")

    assert [frame.n_atoms for frame in frames] == [3, 1, 2]
    assert [frame.metadata["energy"] for frame in frames] == [-76.3, -13.6, -31.8]

    last_pos = as_array(frames[2].columns["pos"])
    assert last_pos.shape == (2, 3)


def test_read_frames_error_carries_frame_index(tmp_path: Path) -> None:
    text = (DATA_DIR / "varying_atom_counts.xyz").read_text()
    broken = tmp_path / "broken.xyz"
    broken.write_text(text + "not-a-count\n")

    with pytest.raises(oxyz.ParseError, match="frame 3") as excinfo:
        oxyz.read(broken)
    assert excinfo.value.frame_index == 3


def test_parse_error_is_a_value_error() -> None:
    assert issubclass(oxyz.ParseError, ValueError)


def test_parse_error_locates_a_short_atom_line(tmp_path: Path) -> None:
    good = "1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n"
    short = "1\nProperties=species:S:1:pos:R:3\nH 0 0\n"
    path = tmp_path / "short.xyz"
    path.write_text(good + short)

    with pytest.raises(oxyz.ParseError) as excinfo:
        oxyz.read(path)
    error = excinfo.value
    assert error.frame_index == 1
    assert error.line == 6  # 1-based file line of the short atom row
    assert error.column is None


def test_parse_error_locates_the_offending_column(tmp_path: Path) -> None:
    path = tmp_path / "badvalue.xyz"
    path.write_text("1\nProperties=species:S:1:pos:R:3\nH 0 zzz 0\n")

    with pytest.raises(oxyz.ParseError) as excinfo:
        oxyz.read(path)
    error = excinfo.value
    assert error.frame_index == 0
    assert error.line == 3
    assert error.column == 3  # 1-based char column where the "pos" column starts
    assert "pos" in str(error)


def test_parse_error_location_attributes(tmp_path: Path) -> None:
    # A count-line error pins the frame and the line, but has no single token
    # to point a column at.
    path = tmp_path / "badcount.xyz"
    path.write_text("not-a-count\ncomment\n")

    with pytest.raises(oxyz.ParseError) as excinfo:
        oxyz.read(path)
    error = excinfo.value
    assert error.frame_index == 0
    assert error.line == 1
    assert error.column is None

    # A directly constructed instance carries None on every axis.
    bare = oxyz.ParseError("boom")
    assert bare.frame_index is None
    assert bare.line is None
    assert bare.column is None


def test_out_of_range_is_index_error_not_parse_error(tmp_path: Path) -> None:
    path = tmp_path / "one.xyz"
    path.write_text("1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n")

    with pytest.raises(IndexError) as excinfo:
        oxyz.read_batch(path, [5])
    assert not isinstance(excinfo.value, oxyz.ParseError)


def test_missing_file_is_os_error(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        oxyz.read(tmp_path / "does-not-exist.xyz")


@pytest.mark.parametrize("path", CORPUS, ids=lambda path: path.name)
def test_parallel_read_frames_matches_serial(path: Path) -> None:
    serial = oxyz.read(path, threads=1)
    parallel = oxyz.read(path, threads=4)

    assert len(parallel) == len(serial)
    for left, right in zip(parallel, serial, strict=True):
        assert left.n_atoms == right.n_atoms
        assert list(left.columns) == list(right.columns)
        for name, values in right.columns.items():
            if isinstance(values, np.ndarray):
                assert_array_equal(as_array(left.columns[name]), values)
            else:
                assert left.columns[name] == values
        assert list(left.metadata) == list(right.metadata)


def test_iter_frames_streams_the_trajectory() -> None:
    frames = oxyz.iread(DATA_DIR / "varying_atom_counts.xyz")

    assert [frame.n_atoms for frame in frames] == [3, 1, 2]


def test_iter_frames_yields_good_frames_then_raises_then_fuses(tmp_path: Path) -> None:
    text = (DATA_DIR / "varying_atom_counts.xyz").read_text()
    broken = tmp_path / "broken.xyz"
    broken.write_text(text + "not-a-count\n")

    frames = oxyz.iread(broken)
    assert [next(frames).n_atoms for _ in range(3)] == [3, 1, 2]
    with pytest.raises(ValueError, match="frame 3"):
        next(frames)
    with pytest.raises(StopIteration):
        next(frames)


def test_two_iterators_are_independent() -> None:
    path = DATA_DIR / "varying_atom_counts.xyz"
    first, second = oxyz.iread(path), oxyz.iread(path)

    next(first)
    assert next(first).n_atoms == 1
    assert next(second).n_atoms == 3


def test_scan_reports_structure_and_statistics() -> None:
    index = oxyz.scan(DATA_DIR / "varying_atom_counts.xyz")

    assert index.n_frames == 3
    assert index.total_atoms == 6
    assert list(index.n_atoms) == [3, 1, 2]
    assert (index.min_atoms, index.max_atoms) == (1, 3)
    assert index.mean_atoms == 2.0
    assert index.median_atoms == 2.0
    assert index.std_atoms == pytest.approx((2 / 3) ** 0.5)


def test_scan_volumes_default_off_and_opt_in() -> None:
    plain = oxyz.scan(DATA_DIR / "varying_density.extxyz")
    assert plain.volumes is None

    index = oxyz.scan(DATA_DIR / "varying_density.extxyz", with_volume=True)
    assert index.volumes is not None
    assert list(index.volumes) == pytest.approx([1.0, 1000.0, 1000.0])


def test_scan_volume_is_nan_without_lattice() -> None:
    index = oxyz.scan(DATA_DIR / "no_lattice_molecule.xyz", with_volume=True)
    assert index.volumes is not None
    assert np.isnan(index.volumes).all()


def test_scan_rejects_structural_garbage(tmp_path: Path) -> None:
    text = (DATA_DIR / "varying_atom_counts.xyz").read_text()
    broken = tmp_path / "broken.xyz"
    broken.write_text(text + "not-a-count\n")

    with pytest.raises(ValueError, match="frame 3"):
        oxyz.scan(broken)


# A single valid frame; the building block for the blank-line cases below.
_FRAME = "1\nProperties=species:S:1:pos:R:3\nH 0.0 0.0 0.0\n"


def test_trailing_blank_line_is_tolerated(tmp_path: Path) -> None:
    # ASE tolerates a trailing blank line where a count is expected; oxyz
    # reads the file's one frame across every entry point rather than raising.
    for suffix in ("\n", "\n\n", "   \n", "\t\n"):
        path = tmp_path / "trailing.xyz"
        path.write_text(_FRAME + suffix)
        assert len(oxyz.read(path)) == 1, repr(suffix)
        assert oxyz.scan(path).n_frames == 1, repr(suffix)
        assert oxyz.infer_schema(path).n_frames == 1, repr(suffix)


def test_blank_line_between_frames_stops_the_read(tmp_path: Path) -> None:
    # ASE truncates at the blank; oxyz matches, reading only the first frame.
    path = tmp_path / "interspersed.xyz"
    path.write_text(_FRAME + "\n" + _FRAME)
    assert len(oxyz.read(path)) == 1
    assert oxyz.scan(path).n_frames == 1


def test_leading_blank_line_yields_no_frames(tmp_path: Path) -> None:
    path = tmp_path / "leading.xyz"
    path.write_text("\n" + _FRAME)
    assert oxyz.read(path) == []
    assert oxyz.scan(path).n_frames == 0


def test_infer_schema_report() -> None:
    report = oxyz.infer_schema(DATA_DIR / "varying_atom_counts.xyz").report()

    assert "3 frames, 6 atoms (min 1, max 3)" in report
    assert "pos: R:3 (3/3 frames)" in report
    assert "energy: Real (3/3 frames)" in report


def test_read_first_simple_extxyz() -> None:
    frame = oxyz.read(DATA_DIR / "simple.extxyz", 0)

    assert frame.n_atoms == 1
    assert list(frame.columns) == ["species", "pos", "forces"]

    assert frame.columns["species"] == ["H"]

    pos = as_array(frame.columns["pos"])
    assert pos.dtype == np.float64
    assert pos.shape == (1, 3)
    assert pos.flags.c_contiguous
    assert_allclose(pos, np.array([[0.0, 0.0, 0.0]]))

    forces = as_array(frame.columns["forces"])
    assert forces.shape == (1, 3)
    assert_allclose(forces, np.array([[0.0, 0.0, 0.0]]))

    assert frame.metadata["energy"] == -1.0
    assert isinstance(frame.metadata["energy"], float)

    # Lattice arrives flat, in as-written order; reshaping and reordering are
    # the normalisation layer's job.
    lattice = as_array(frame.metadata["Lattice"])
    assert lattice.shape == (9,)
    assert_allclose(lattice, np.array([15.0, 0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 15.0]))

    stress = as_array(frame.metadata["stress"])
    assert stress.shape == (6,)
    assert_allclose(stress, np.zeros(6))

    pbc = as_array(frame.metadata["pbc"])
    assert pbc.dtype == np.bool_
    assert_array_equal(pbc, np.array([True, True, True]))

    # Properties is consumed into columns, not duplicated in metadata.
    assert "Properties" not in frame.metadata


def test_nonorthogonal_lattice_preserved_as_written() -> None:
    frame = oxyz.read(DATA_DIR / "nonorthogonal.extxyz", 0)

    lattice = as_array(frame.metadata["Lattice"])
    assert_allclose(lattice, np.array([10.0, 1.0, 2.0, 0.0, 11.0, 3.0, 0.0, 0.0, 12.0]))

    pos = as_array(frame.columns["pos"])
    assert pos.shape == (2, 3)
    assert_allclose(pos, np.array([[0.0, 0.1, 0.2], [3.0, 3.1, 3.2]]))


def test_integer_and_string_columns() -> None:
    frame = oxyz.read(DATA_DIR / "id_and_selection.extxyz", 0)

    assert list(frame.columns) == ["id", "species", "pos", "selection"]

    ids = as_array(frame.columns["id"])
    assert ids.dtype == np.int64
    assert_array_equal(ids, np.array([10, 11, 12]))

    assert frame.columns["species"] == ["Si", "Si", "O"]

    selection = as_array(frame.columns["selection"])
    assert_array_equal(selection, np.array([1, 0, 1]))


def test_metadata_value_typing() -> None:
    frame = oxyz.read(DATA_DIR / "quoted_strings_booleans_scalars.extxyz", 0)

    assert frame.metadata["source"] == "generated for parser study"
    assert frame.metadata["split"] == "train"
    assert frame.metadata["converged"] is True
    assert frame.metadata["frozen"] is False
    assert frame.metadata["temperature"] == 298.15

    # bool is a subclass of int in Python, so check step isn't a bool too.
    step = frame.metadata["step"]
    assert step == 12
    assert isinstance(step, int)
    assert not isinstance(step, bool)


def test_bracket_array_metadata() -> None:
    frame = oxyz.read(DATA_DIR / "newstyle_array_metadata.extxyz", 0)

    kpoints = as_array(frame.metadata["kpoints"])
    assert kpoints.dtype == np.int64
    assert_array_equal(kpoints, np.array([2, 2, 1]))

    cutoffs = as_array(frame.metadata["cutoffs"])
    assert cutoffs.dtype == np.float64
    assert_allclose(cutoffs, np.array([4.5, 5.0]))

    assert frame.metadata["tags"] == ["slab", "relaxed"]


def test_mace_training_schema_names_preserved() -> None:
    frame = oxyz.read(DATA_DIR / "mace_ref_energy_forces_stress.xyz", 0)

    ref_forces = as_array(frame.columns["REF_forces"])
    assert ref_forces.shape == (3, 3)

    assert frame.metadata["REF_energy"] == -76.123
    assert frame.metadata["config_type"] == "Default"

    ref_stress = as_array(frame.metadata["REF_stress"])
    assert ref_stress.shape == (6,)
