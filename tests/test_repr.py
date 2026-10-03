"""Every public object's `repr`: one line, data-free, bounded, constructor-shaped.

Objects that hold data (`Frame`, `Batch`, the converter sources) describe values
in numpy terms; objects that describe a file (`Schema`, `SchemaSpec`) use the
extxyz letters written in `Properties=` and in YAML specs. `Frame`/`Batch` are
pinned in `test_frame_api.py`; the rest are pinned here as exact strings.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import cast

import numpy as np
import pytest

import oxyz
from oxyz import (
    ColumnRule,
    ColumnSchema,
    ColumnVariant,
    FrameIndex,
    FrameRule,
    Kind,
    MetadataRule,
    MetadataSchema,
    MetadataVariant,
    SchemaSpec,
    Violation,
)

DATA_DIR = Path(__file__).parent / "data"
TWO_FRAME = DATA_DIR / "two_frame_same_schema.xyz"
VARYING = DATA_DIR / "varying_atom_counts.xyz"  # 3, 1, 2 atoms; cubic Lattice
DRIFT = DATA_DIR / "schema_drift_type.extxyz"  # magmom R:3 then R:1
OPTIONAL = DATA_DIR / "mixed_schema_optional_column.xyz"  # charge in frame 0 only


@pytest.fixture
def empty(tmp_path: Path) -> Path:
    path = tmp_path / "empty.xyz"
    path.touch()
    return path


def test_kind_repr_names_the_member() -> None:
    assert repr(Kind.REAL) == "Kind.REAL"


def test_kind_is_still_its_value_as_text() -> None:
    # A StrEnum: callers format and compare it as text.
    assert str(Kind.INT) == "Int"
    assert f"{Kind.BOOL}" == "Bool"
    assert Kind.STR == "Str"


def test_variant_reprs_carry_the_short_kind() -> None:
    assert repr(ColumnVariant(Kind.REAL, 3, 2)) == (
        "ColumnVariant(kind=Kind.REAL, width=3, frames=2)"
    )
    assert repr(MetadataVariant(Kind.INT, (9,), 1)) == (
        "MetadataVariant(kind=Kind.INT, shape=(9,), frames=1)"
    )


def test_frame_index_repr_summarises_counts_and_volumes() -> None:
    assert repr(oxyz.scan(VARYING, with_volume=True)) == (
        "FrameIndex(n_frames=3, total_atoms=6, n_atoms=1..3, volumes=float64[3])"
    )


def test_frame_index_repr_omits_unscanned_volumes() -> None:
    assert repr(oxyz.scan(VARYING)) == (
        "FrameIndex(n_frames=3, total_atoms=6, n_atoms=1..3)"
    )


def test_frame_index_repr_omits_the_range_of_an_empty_file(empty: Path) -> None:
    assert repr(oxyz.scan(empty)) == "FrameIndex(n_frames=0, total_atoms=0)"


def test_frame_index_repr_is_bounded_for_a_million_frames() -> None:
    n = 10**6
    index = FrameIndex(offsets=np.arange(n, dtype=np.uint64), n_atoms=np.full(n, 7))

    assert repr(index) == (
        "FrameIndex(n_frames=1000000, total_atoms=7000000, n_atoms=7..7)"
    )


def test_schema_repr_lists_each_entry_in_extxyz_letters() -> None:
    assert repr(oxyz.infer_schema(TWO_FRAME)) == (
        "Schema(n_frames=2, total_atoms=4, n_atoms=2..2, "
        "columns={'species': S:1, 'pos': R:3, 'forces': R:3}, "
        "metadata={'Lattice': I[9], 'energy': R, 'Time': R}, is_consistent=True)"
    )


def test_schema_repr_joins_drifting_variants() -> None:
    assert repr(oxyz.infer_schema(DRIFT)) == (
        "Schema(n_frames=2, total_atoms=4, n_atoms=2..2, "
        "columns={'species': S:1, 'pos': R:3, 'magmom': R:3|R:1}, "
        "metadata={'energy': R}, is_consistent=False)"
    )


def test_schema_repr_marks_a_partially_present_column() -> None:
    assert repr(oxyz.infer_schema(OPTIONAL)) == (
        "Schema(n_frames=2, total_atoms=4, n_atoms=2..2, "
        "columns={'species': S:1, 'pos': R:3, 'charge': R:1?}, "
        "metadata={}, is_consistent=False)"
    )


def test_schema_repr_renders_metadata_shapes_drift_and_absence() -> None:
    schema = dataclasses.replace(
        oxyz.infer_schema(TWO_FRAME),
        columns=(),
        metadata=(
            MetadataSchema(
                "stress",
                (MetadataVariant(Kind.REAL, (3, 3), 2),),
                frames_present=2,
                unified=(Kind.REAL, (3, 3)),
            ),
            MetadataSchema(
                "cell",
                (
                    MetadataVariant(Kind.INT, (9,), 1),
                    MetadataVariant(Kind.REAL, (9,), 0),
                ),
                frames_present=1,
                unified=(Kind.REAL, (9,)),
            ),
        ),
    )

    assert "metadata={'stress': R[3, 3], 'cell': I[9]|R[9]?}" in repr(schema)


def test_schema_repr_omits_the_range_of_an_empty_file(empty: Path) -> None:
    assert repr(oxyz.infer_schema(empty)) == (
        "Schema(n_frames=0, total_atoms=0, columns={}, metadata={}, is_consistent=True)"
    )


def test_schema_repr_elides_a_wide_schema() -> None:
    wide = tuple(
        ColumnSchema(f"c{i}", (ColumnVariant(Kind.REAL, 1, 2),), 2, (Kind.REAL, 1))
        for i in range(50)
    )
    text = repr(dataclasses.replace(oxyz.infer_schema(TWO_FRAME), columns=wide))

    assert "'c7': R:1, +42 more}" in text
    assert "'c8'" not in text
    assert len(text) < 300


def test_schema_str_is_still_the_report() -> None:
    schema = oxyz.infer_schema(TWO_FRAME)

    assert str(schema) == schema.report()


def test_spec_repr_shows_rules_and_non_default_fields() -> None:
    spec = SchemaSpec(
        columns=(
            ColumnRule("pos", Kind.REAL, width=3),
            ColumnRule("REF_*", Kind.REAL, width=3, required=False),
        ),
        metadata=(
            MetadataRule("energy", Kind.REAL),
            MetadataRule("Lattice", Kind.REAL, shape=(9,), required=False),
        ),
        frame=FrameRule(n_atoms_max=500),
        mode="project",
    )

    assert repr(spec) == (
        "SchemaSpec(columns={'pos': R:3, 'REF_*': R:3?}, "
        "metadata={'energy': R, 'Lattice': R[9]?}, "
        "frame=FrameRule(n_atoms_max=500), mode='project')"
    )


def test_default_spec_repr_is_two_empty_maps() -> None:
    assert repr(SchemaSpec()) == "SchemaSpec(columns={}, metadata={})"


def test_inferred_spec_repr_round_trips_the_schema_notation() -> None:
    assert repr(oxyz.infer_schema(OPTIONAL).to_spec()) == (
        "SchemaSpec(columns={'species': S:1, 'pos': R:3, 'charge': R:1?}, metadata={})"
    )


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        (
            ColumnRule("pos", Kind.REAL, width=3),
            "ColumnRule(name='pos', kind=Kind.REAL, width=3)",
        ),
        (
            ColumnRule("charge", Kind.REAL, required=False),
            "ColumnRule(name='charge', kind=Kind.REAL, required=False)",
        ),
        (
            ColumnRule("descriptor_*", Kind.REAL, count=5),
            "ColumnRule(name='descriptor_*', kind=Kind.REAL, count=5)",
        ),
        (
            ColumnRule("tag", Kind.INT, required=False, fill=0),
            "ColumnRule(name='tag', kind=Kind.INT, required=False, fill=0)",
        ),
        (
            MetadataRule("stress", Kind.REAL, shape=(3, 3)),
            "MetadataRule(key='stress', kind=Kind.REAL, shape=(3, 3))",
        ),
        (
            MetadataRule("energy", Kind.REAL),
            "MetadataRule(key='energy', kind=Kind.REAL)",
        ),
        (FrameRule(), "FrameRule()"),
        (FrameRule(lattice_required=True), "FrameRule(lattice_required=True)"),
    ],
)
def test_rule_repr_drops_default_fields(rule: object, expected: str) -> None:
    assert repr(rule) == expected


def test_violation_repr_drops_unset_location() -> None:
    violation = Violation("column", "pos", "missing", expected="R:3", found=None)

    assert repr(violation) == (
        "Violation(axis='column', name='pos', deviation='missing', "
        "expected='R:3', found=None)"
    )
    assert repr(dataclasses.replace(violation, frame_index=4)).endswith(
        "found=None, frame_index=4)"
    )


def test_writer_repr_counts_frames_and_reports_closing(tmp_path: Path) -> None:
    path = tmp_path / "out.xyz"
    frames = oxyz.read(TWO_FRAME)
    writer = oxyz.Writer(path)

    assert repr(writer) == f"Writer({str(path)!r}, frames_written=0)"
    writer.write(frames)
    assert repr(writer) == f"Writer({str(path)!r}, frames_written=2)"
    writer.close()
    assert repr(writer) == f"Writer({str(path)!r}, frames_written=2, closed=True)"


def test_writer_repr_shows_non_default_options(tmp_path: Path) -> None:
    path = tmp_path / "out.xyz.gz"
    with oxyz.Writer(path, compression="gzip", level=3, batch=4) as writer:
        assert repr(writer) == (
            f"Writer({str(path)!r}, compression='gzip', level=3, batch=4, "
            "frames_written=0)"
        )


def test_writer_repr_does_not_count_a_rejected_item(tmp_path: Path) -> None:
    path = tmp_path / "out.xyz"
    with oxyz.Writer(path) as writer:
        with pytest.raises(TypeError):
            writer.write(cast("oxyz.Frame", object()))
        assert repr(writer) == f"Writer({str(path)!r}, frames_written=0)"
