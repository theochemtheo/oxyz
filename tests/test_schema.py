from __future__ import annotations

from pathlib import Path

import pytest

import oxyz
from oxyz import ColumnVariant, Kind, MetadataVariant

DATA_DIR = Path(__file__).parent / "data"

# MACE-style drift: an isolated-atom frame with integer forces and energy and
# no cell, then a bulk frame with the Real equivalents, a Lattice, and a
# differently-sized stress.
MIXED = (
    "1\n"
    "Properties=species:S:1:pos:R:3:forces:I:3 energy=-158"
    ' stress="1.0 2.0 3.0 4.0 5.0 6.0" config_type=IsolatedAtom\n'
    "Si 0 0 0 0 0 0\n"
    "2\n"
    'Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0"'
    " Properties=species:S:1:pos:R:3:forces:R:3 energy=-412.08"
    ' stress="1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0"\n'
    "Si 0 0 0 0.1 0.2 0.3\n"
    "Si 1 1 1 -0.1 -0.2 -0.3\n"
)


@pytest.fixture
def mixed_schema(tmp_path: Path) -> oxyz.Schema:
    path = tmp_path / "mixed.xyz"
    path.write_text(MIXED)
    return oxyz.infer_schema(path)


def test_schema_counts(mixed_schema: oxyz.Schema) -> None:
    assert mixed_schema.n_frames == 2
    assert mixed_schema.total_atoms == 3
    assert mixed_schema.min_atoms == 1
    assert mixed_schema.max_atoms == 2
    assert not mixed_schema.is_consistent


def test_schema_distribution_stats_match_scan(tmp_path: Path) -> None:
    # The single schema pass keeps the per-frame counts, so its distribution
    # statistics agree with a separate scan -- no second read needed.
    path = tmp_path / "mixed.xyz"
    path.write_text(MIXED)
    schema = oxyz.infer_schema(path)
    index = oxyz.scan(path)

    assert list(schema.n_atoms) == list(index.n_atoms)
    assert schema.mean_atoms == index.mean_atoms
    assert schema.median_atoms == index.median_atoms
    assert schema.std_atoms == index.std_atoms


def test_column_variants_and_unification(mixed_schema: oxyz.Schema) -> None:
    columns = {column.name: column for column in mixed_schema.columns}
    assert list(columns) == ["species", "pos", "forces"]

    # Stable column: one variant, unifies to itself.
    pos = columns["pos"]
    assert pos.frames_present == 2
    assert pos.variants == (ColumnVariant(Kind.REAL, 3, 2),)
    assert pos.unified == (Kind.REAL, 3)

    # Int/Real drift at equal width: both variants kept, promoted to Real.
    forces = columns["forces"]
    assert forces.variants == (
        ColumnVariant(Kind.INT, 3, 1),
        ColumnVariant(Kind.REAL, 3, 1),
    )
    assert forces.unified == (Kind.REAL, 3)


def test_metadata_presence_shapes_and_conflicts(mixed_schema: oxyz.Schema) -> None:
    metadata = {entry.key: entry for entry in mixed_schema.metadata}

    # Scalar Int/Real drift promotes to a Real scalar.
    energy = metadata["energy"]
    assert energy.variants == (
        MetadataVariant(Kind.INT, (), 1),
        MetadataVariant(Kind.REAL, (), 1),
    )
    assert energy.unified == (Kind.REAL, ())

    # Length change is a genuine conflict: no unified reading.
    stress = metadata["stress"]
    assert stress.variants == (
        MetadataVariant(Kind.REAL, (6,), 1),
        MetadataVariant(Kind.REAL, (9,), 1),
    )
    assert stress.unified is None

    # Presence gaps live in frames_present, not in the variants.
    lattice = metadata["Lattice"]
    assert lattice.frames_present == 1
    assert lattice.variants == (MetadataVariant(Kind.REAL, (9,), 1),)
    assert lattice.unified == (Kind.REAL, (9,))

    config_type = metadata["config_type"]
    assert config_type.frames_present == 1
    assert config_type.variants == (MetadataVariant(Kind.STR, (), 1),)


def test_report_matches_str(mixed_schema: oxyz.Schema) -> None:
    report = mixed_schema.report()
    assert str(mixed_schema) == report
    assert "2 frames, 3 atoms (min 1, max 2)" in report
    assert "forces: I:3 (1/2 frames), R:3 (1/2 frames) (unifies to R:3)" in report
    assert (
        "stress: RealArray[6] (1/2 frames), RealArray[9] (1/2 frames) [inconsistent]"
        in report
    )


def test_stable_file_is_consistent() -> None:
    schema = oxyz.infer_schema(DATA_DIR / "two_frame_same_schema.xyz")

    assert schema.is_consistent
    for column in schema.columns:
        assert len(column.variants) == 1
        assert column.frames_present == schema.n_frames
        assert column.unified == (column.variants[0].kind, column.variants[0].width)
