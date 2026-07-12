from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from oxyz import (
    Kind,
    iread,
    read,
)
from oxyz._schema_match import SchemaError, SchemaWarning
from oxyz._schema_spec import ColumnRule, MetadataRule, SchemaSpec

DATA = Path(__file__).parent / "data"

SPEC = SchemaSpec(
    columns=(
        ColumnRule("species", Kind.STR),
        ColumnRule("pos", Kind.REAL, width=3),
        # Optional so the conformant/extra-column files (which lack it) pass;
        # a present-but-mismatched magmom still fires (drift fixture, frame 1).
        ColumnRule("magmom", Kind.REAL, width=3, required=False),
    ),
    metadata=(MetadataRule("energy", Kind.REAL),),
)


def test_conformant_file_reads_all_frames():
    frames = read(DATA / "schema_conformant.extxyz", schema=SPEC)
    assert len(frames) == 2


def test_no_schema_is_unchanged():
    assert len(read(DATA / "schema_extra_column.extxyz")) == 2


def test_extra_column_strict_raises_with_frame_index():
    with pytest.raises(SchemaError) as excinfo:
        read(DATA / "schema_extra_column.extxyz", schema=SPEC, conformance="strict")
    assert excinfo.value.frame_index == 1
    assert excinfo.value.name == "charge"


def test_extra_column_required_allowed():
    frames = read(
        DATA / "schema_extra_column.extxyz", schema=SPEC, conformance="required"
    )
    assert "charge" in frames[1].columns


def test_drift_type_required_raises_at_frame_1():
    with pytest.raises(SchemaError) as excinfo:
        read(DATA / "schema_drift_type.extxyz", schema=SPEC, conformance="required")
    assert excinfo.value.frame_index == 1
    assert excinfo.value.name == "magmom"


def test_warn_conformance_warns_not_raises():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        frames = read(
            DATA / "schema_drift_type.extxyz", schema=SPEC, conformance="warn"
        )
    assert len(frames) == 2
    assert any(issubclass(w.category, SchemaWarning) for w in caught)


def test_schema_from_yaml_path(tmp_path: Path):
    (tmp_path / "s.yaml").write_text(
        "columns:\n  species: {kind: S}\n  pos: {kind: R, width: 3}\n"
        "metadata:\n  energy: {kind: R}\n"
    )
    frames = read(DATA / "schema_conformant.extxyz", schema=tmp_path / "s.yaml")
    assert len(frames) == 2


def test_read_index_zero_with_schema_validates_whole_file():
    # read(path, 0, schema) inherits nth_frame's whole-file validation: a drift
    # at frame 1 is raised even when selecting frame 0, so an indexed read
    # validates consistently with ase.read whatever the index.
    with pytest.raises(SchemaError) as excinfo:
        read(DATA / "schema_drift_type.extxyz", 0, schema=SPEC, conformance="required")
    assert excinfo.value.frame_index == 1


def test_iread_raises_mid_stream_at_drift():
    it = iread(DATA / "schema_drift_type.extxyz", schema=SPEC, conformance="required")
    assert next(it) is not None
    with pytest.raises(SchemaError) as excinfo:
        next(it)
    assert excinfo.value.frame_index == 1


def test_read_slice_reports_original_index():
    with pytest.raises(SchemaError) as excinfo:
        read(
            DATA / "schema_drift_type.extxyz",
            slice(1, None),
            schema=SPEC,
            conformance="required",
        )
    assert excinfo.value.frame_index == 1


def test_projected_binding_entries_exist():
    import oxyz._rust as _rust

    for name in (
        "read_frames_projected",
        "read_first_frame_projected",
        "read_frames_projected_reader",
        "read_first_frame_projected_reader",
        "FrameIterProjected",
    ):
        assert hasattr(_rust, name), name


def _mixed_file(tmp_path):
    f = tmp_path / "mixed.xyz"
    f.write_text(
        "1\nProperties=species:S:1:pos:R:3:charge:R:1\nH 0 0 0 0.5\n"
        "1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n"
    )
    return f


def test_project_drops_extra_and_fills_absent(tmp_path):
    import math

    import numpy as np

    import oxyz
    from oxyz._schema import Kind
    from oxyz._schema_spec import ColumnRule, SchemaSpec

    f = _mixed_file(tmp_path)
    spec = SchemaSpec(
        columns=(
            ColumnRule("species", Kind.STR),
            ColumnRule("pos", Kind.REAL, width=3),
            ColumnRule("charge", Kind.REAL, required=False),
        ),
        mode="project",
    )
    frames = oxyz.read(f, schema=spec)
    assert [set(fr.columns) for fr in frames] == [
        {"species", "pos", "charge"},
        {"species", "pos", "charge"},
    ]
    assert np.asarray(frames[0].columns["charge"]).tolist() == [0.5]
    assert math.isnan(np.asarray(frames[1].columns["charge"])[0])  # filled


def test_mode_override_beats_spec(tmp_path):
    import oxyz
    from oxyz._schema import Kind
    from oxyz._schema_spec import ColumnRule, SchemaSpec

    f = _mixed_file(tmp_path)
    spec = SchemaSpec(columns=(ColumnRule("pos", Kind.REAL, width=3),), mode="project")
    # override back to validate: the extra 'charge' is allowed, nothing reshaped
    frames = oxyz.read(f, schema=spec, mode="validate", conformance="required")
    assert "charge" in frames[0].columns


def test_mode_without_schema_errors(tmp_path):
    import oxyz

    f = _mixed_file(tmp_path)
    with pytest.raises(ValueError, match="mode"):
        oxyz.read(f, mode="project")


def test_read_index_zero_projects(tmp_path):
    import oxyz
    from oxyz._schema import Kind
    from oxyz._schema_spec import ColumnRule, SchemaSpec

    f = _mixed_file(tmp_path)
    spec = SchemaSpec(
        columns=(
            ColumnRule("species", Kind.STR),
            ColumnRule("pos", Kind.REAL, width=3),
        ),
        mode="project",
    )
    fr = oxyz.read(f, 0, schema=spec)
    assert set(fr.columns) == {"species", "pos"}  # charge dropped


def test_iread_projects_and_drops(tmp_path):
    import oxyz
    from oxyz._schema import Kind
    from oxyz._schema_spec import ColumnRule, SchemaSpec

    f = _mixed_file(tmp_path)
    # require an int id with no fill: neither frame has it -> both dropped under warn
    spec = SchemaSpec(
        columns=(ColumnRule("pos", Kind.REAL, width=3), ColumnRule("id", Kind.INT)),
        mode="project",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frames = list(oxyz.iread(f, schema=spec, conformance="warn"))
    assert frames == []


def test_strict_and_required_collapse_under_project(tmp_path):
    import oxyz

    f = _mixed_file(tmp_path)  # frame 0 has an undeclared 'charge'
    spec = SchemaSpec(
        columns=(
            ColumnRule("species", Kind.STR),
            ColumnRule("pos", Kind.REAL, width=3),
        ),
        mode="project",
    )
    strict = oxyz.read(f, schema=spec, conformance="strict")
    required = oxyz.read(f, schema=spec, conformance="required")
    # Under project, an undeclared field is dropped, never a violation — so
    # strict and required behave identically (neither raises on 'charge').
    assert [set(fr.columns) for fr in strict] == [set(fr.columns) for fr in required]
    assert all(set(fr.columns) == {"species", "pos"} for fr in strict)


def test_metadata_projection_through_reader(tmp_path):
    import numpy as np

    import oxyz

    f = tmp_path / "meta.xyz"
    f.write_text(
        "1\nProperties=species:S:1:pos:R:3 "
        'energy=-1.0 stress="1.0 2.0 3.0 4.0 5.0 6.0"\n'
        "H 0 0 0\n"
        "1\nProperties=species:S:1:pos:R:3\nH 1 0 0\n"
    )
    spec = SchemaSpec(
        columns=(
            ColumnRule("species", Kind.STR),
            ColumnRule("pos", Kind.REAL, width=3),
        ),
        metadata=(
            MetadataRule("energy", Kind.REAL, required=False),
            MetadataRule("stress", Kind.REAL, shape=(6,), required=False),
        ),
        mode="project",
    )
    frames = oxyz.read(f, schema=spec)
    assert frames[0].metadata["energy"] == -1.0
    assert np.asarray(frames[0].metadata["stress"]).tolist() == [1, 2, 3, 4, 5, 6]
    # frame 1 lacks both -> filled (scalar NaN, and a 6-length NaN array)
    assert np.isnan(np.asarray(frames[1].metadata["energy"]))
    assert np.isnan(np.asarray(frames[1].metadata["stress"])).all()
    assert np.asarray(frames[1].metadata["stress"]).shape == (6,)


def test_wrong_kind_under_warn_fills_nan_and_warns(tmp_path):
    import numpy as np

    import oxyz

    f = tmp_path / "w.xyz"
    # 'val' is Int in the file but the schema declares it Real -> wrong kind
    f.write_text("1\nProperties=species:S:1:pos:R:3:val:I:1\nH 0 0 0 7\n")
    spec = SchemaSpec(
        columns=(
            ColumnRule("species", Kind.STR),
            ColumnRule("pos", Kind.REAL, width=3),
            ColumnRule("val", Kind.REAL),
        ),
        mode="project",
    )
    with pytest.warns(SchemaWarning, match="val"):
        frames = oxyz.read(f, schema=spec, conformance="warn")
    # the real int 7 is discarded and replaced with a NaN fill (data loss)
    assert np.isnan(np.asarray(frames[0].columns["val"])[0])


def test_falsy_fills_survive_projection(tmp_path):
    import numpy as np

    import oxyz

    f = tmp_path / "m.xyz"
    f.write_text("1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n")  # no id / tag
    spec = SchemaSpec(
        columns=(
            ColumnRule("species", Kind.STR),
            ColumnRule("pos", Kind.REAL, width=3),
            ColumnRule("id", Kind.INT, required=False, fill=0),
            ColumnRule("tag", Kind.STR, required=False, fill=""),
        ),
        mode="project",
    )
    fr = oxyz.read(f, schema=spec)[0]
    assert np.asarray(fr.columns["id"]).tolist() == [0]  # 0 is a real fill, not absence
    assert list(fr.columns["tag"]) == [""]


def test_warn_drop_emits_warning_naming_field(tmp_path):
    import oxyz

    f = _mixed_file(tmp_path)  # neither frame carries 'id'
    spec = SchemaSpec(
        columns=(ColumnRule("pos", Kind.REAL, width=3), ColumnRule("id", Kind.INT)),
        mode="project",
    )
    with pytest.warns(SchemaWarning, match="id"):
        frames = list(oxyz.iread(f, schema=spec, conformance="warn"))
    assert frames == []  # every frame dropped (unfillable required id)


def _proj_spec():
    return SchemaSpec(
        columns=(
            ColumnRule("species", Kind.STR),
            ColumnRule("pos", Kind.REAL, width=3),
            ColumnRule("charge", Kind.REAL, required=False),
        ),
        mode="project",
    )


def test_read_frames_sliced_projects_kept_frames(tmp_path):
    import math

    import numpy as np

    f = _mixed_file(tmp_path)
    # Slice off the first frame: the projected branch reshapes only what the
    # slice keeps, indexed by original position.
    frames = read(f, slice(1, None), schema=_proj_spec())
    assert len(frames) == 1
    assert set(frames[0].columns) == {"species", "pos", "charge"}
    assert math.isnan(np.asarray(frames[0].columns["charge"])[0])  # filled


def test_reverse_slice_with_schema_projects(tmp_path):
    pytest.importorskip("ase")
    import oxyz.ase

    f = _mixed_file(tmp_path)
    # A reverse slice with a schema takes sliced_frames' read-and-project path
    # (the streaming/index shortcut is skipped so the sought frames are shaped).
    atoms = list(oxyz.ase.iread(f, "::-1", schema=_proj_spec()))
    assert len(atoms) == 2
    assert all(len(a) == 1 for a in atoms)
