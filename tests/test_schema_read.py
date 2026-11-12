from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from oxyz import (
    Kind,
    iread,
    iread_batch,
    read,
    read_batch,
)
from oxyz._schema_match import SchemaError, SchemaWarning
from oxyz._schema_spec import ColumnRule, MetadataRule, SchemaSpec

DATA = Path(__file__).parent / "data"

SPEC = SchemaSpec(
    columns=(
        ColumnRule("species", Kind.STR),
        ColumnRule("pos", Kind.REAL, width=3),
        # Optional, so only the drift fixture's mismatched magmom fires.
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
    # The whole file is validated whatever the index.
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
    from oxyz import _rust

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
    # Projection drops undeclared fields, so strict and required agree.
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


def test_2d_metadata_shape_rejected_by_projection(tmp_path):
    import oxyz

    f = tmp_path / "meta2d.xyz"
    f.write_text("1\nProperties=species:S:1:pos:R:3 key=[[1,2],[3,4]]\nH 0 0 0\n")
    spec = SchemaSpec(
        columns=(
            ColumnRule("species", Kind.STR),
            ColumnRule("pos", Kind.REAL, width=3),
        ),
        metadata=(MetadataRule("key", Kind.INT, shape=(2, 2), required=True),),
        mode="project",
    )
    # 2-D metadata projection is unsupported: an error, not a panic or flattening.
    with pytest.raises(ValueError, match="more than one dimension"):
        oxyz.read(f, schema=spec)


def test_wrong_kind_under_warn_fills_nan_and_warns(tmp_path):
    import numpy as np

    import oxyz

    f = tmp_path / "w.xyz"
    # 'val' is a string in the file but the schema declares it Real -> wrong kind
    f.write_text("1\nProperties=species:S:1:pos:R:3:val:S:1\nH 0 0 0 seven\n")
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
    # the string is discarded and replaced with a NaN fill (data loss)
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
    frames = read(f, slice(1, None), schema=_proj_spec())
    assert len(frames) == 1
    assert set(frames[0].columns) == {"species", "pos", "charge"}
    assert math.isnan(np.asarray(frames[0].columns["charge"])[0])  # filled


def test_reverse_slice_with_schema_projects(tmp_path):
    pytest.importorskip("ase")
    import oxyz.ase

    f = _mixed_file(tmp_path)
    atoms = list(oxyz.ase.iread(f, "::-1", schema=_proj_spec()))
    assert len(atoms) == 2
    assert all(len(a) == 1 for a in atoms)


def _charge_omit_spec(width=1, kind=Kind.REAL, name="charge"):
    return SchemaSpec(
        columns=(
            ColumnRule("species", Kind.STR),
            ColumnRule("pos", Kind.REAL, width=3),
            ColumnRule(name, kind, width=width, required=False, absent="omit"),
        ),
        mode="project",
    )


def test_omit_rule_leaves_the_key_out_of_frames_that_lack_it():
    frames = read(DATA / "mixed_schema_optional_column.xyz", schema=_charge_omit_spec())
    assert ["charge" in f.columns for f in frames] == [True, False]


def test_omit_rule_of_the_wrong_kind_raises_with_frame_and_name():
    spec = _charge_omit_spec(width=3, name="magmom")
    with pytest.raises(SchemaError) as excinfo:
        read(DATA / "schema_drift_type.extxyz", schema=spec)
    assert (excinfo.value.frame_index, excinfo.value.name) == (1, "magmom")


def test_omit_rule_with_mixed_presence_cannot_batch():
    import oxyz

    with pytest.raises(oxyz.ParseError, match="charge"):
        oxyz.read_batch(
            DATA / "mixed_schema_optional_column.xyz", schema=_charge_omit_spec()
        )


def test_omit_rule_on_metadata_with_mixed_presence_cannot_batch(tmp_path):
    import oxyz

    path = _stepped_file(tmp_path, "energy=-1.5", "step=1", "energy=-2.5")
    spec = SchemaSpec(
        metadata=(MetadataRule("energy", Kind.REAL, required=False, absent="omit"),),
        mode="project",
    )
    with pytest.raises(
        oxyz.ParseError, match=r"frame 1 is missing metadata \"energy\""
    ):
        oxyz.read_batch(path, schema=spec)


LATTICE = SchemaSpec(
    metadata=(MetadataRule("Lattice", Kind.REAL, shape=(9,)),), mode="project"
)
INTEGER_LATTICE_FILE = DATA / "mace_ref_energy_forces_stress.xyz"


def test_project_mode_promotes_an_integer_lattice():
    frame = read(INTEGER_LATTICE_FILE, 0, schema=LATTICE)
    lattice = np.asarray(frame.metadata["Lattice"])
    assert lattice.dtype == np.float64
    assert lattice.tolist() == [5, 0, 0, 0, 5, 0, 0, 0, 5]


def test_validate_mode_still_reports_an_integer_lattice():
    with pytest.raises(SchemaError) as e:
        read(INTEGER_LATTICE_FILE, schema=replace(LATTICE, mode="validate"))
    assert (e.value.frame_index, e.value.name) == (0, "Lattice")


def test_project_batch_promotes_too():
    batch = read_batch(INTEGER_LATTICE_FILE, schema=LATTICE)
    lattice = np.asarray(batch.metadata["Lattice"])
    assert lattice.dtype == np.float64
    assert lattice.shape == (1, 9)
    assert lattice[0].tolist() == [5, 0, 0, 0, 5, 0, 0, 0, 5]


def _stepped_file(tmp_path: Path, *comments: str) -> Path:
    """One single-hydrogen frame per comment-line tail."""
    path = tmp_path / "stepped.xyz"
    path.write_text(
        "".join(
            f"1\nProperties=species:S:1:pos:R:3 {comment}\nH 0 0 0\n"
            for comment in comments
        )
    )
    return path


# An absent INT has no NaN to fill with, so a frame without `step` is dropped.
STEP_PROJECTION = SchemaSpec(
    metadata=(MetadataRule("step", Kind.INT), MetadataRule("energy", Kind.REAL)),
    mode="project",
)


def test_a_project_mode_warn_read_indexes_the_frames_it_kept(tmp_path):
    path = _stepped_file(
        tmp_path,
        "energy=0.0",
        "step=1 energy=1.0",
        "step=2 energy=2.0",
        "step=3 energy=3.0",
    )
    with pytest.warns(SchemaWarning, match="step"):
        frame = read(path, 1, schema=STEP_PROJECTION, conformance="warn")
    assert frame.metadata["step"] == 2
    with pytest.warns(SchemaWarning, match="step"):
        picked = read(path, [0, 2], schema=STEP_PROJECTION, conformance="warn")
    assert [f.metadata["step"] for f in picked] == [1, 3]
    with pytest.warns(SchemaWarning, match="step"):
        sliced = read(path, slice(1, 3), schema=STEP_PROJECTION, conformance="warn")
    assert [f.metadata["step"] for f in sliced] == [2, 3]
    with pytest.warns(SchemaWarning, match="step"):
        last = read(path, -1, schema=STEP_PROJECTION, conformance="warn")
    assert last.metadata["step"] == 3
    with pytest.warns(SchemaWarning, match="step"):
        streamed = list(iread(path, "1:", schema=STEP_PROJECTION, conformance="warn"))
    assert [f.metadata["step"] for f in streamed] == [2, 3]


WRONG_SCHEMA_READS = {
    "read": lambda p, s: read(p, schema=s),
    "read-int": lambda p, s: read(p, 0, schema=s),
    "read-list": lambda p, s: read(p, [0], schema=s),
    "iread": lambda p, s: list(iread(p, schema=s)),
    "read_batch": lambda p, s: read_batch(p, schema=s),
    "iread_batch": lambda p, s: iread_batch(p, schema=s, frames_per_batch=1),
}


@pytest.mark.parametrize("reader", WRONG_SCHEMA_READS.values(), ids=WRONG_SCHEMA_READS)
def test_a_schema_that_is_neither_a_spec_nor_a_path_is_a_type_error(reader):
    with pytest.raises(TypeError, match="not 'dict'"):
        reader(DATA / "schema_conformant.extxyz", {"mode": "project"})
