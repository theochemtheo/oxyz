from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from oxyz import Kind
from oxyz._schema_spec import (
    ColumnRule,
    FrameRule,
    MetadataRule,
    SchemaSpec,
    render_yaml,
)

if TYPE_CHECKING:
    from pathlib import Path

SPEC_DICT = {
    "columns": {
        "species": {"kind": "S"},
        "pos": {"kind": "R", "width": 3},
        "forces": {"kind": "R", "width": 3, "required": False},
        "descriptor_*": {"kind": "R", "count": 5},
    },
    "metadata": {
        "energy": {"kind": "R"},
        "stress": {"kind": "R", "shape": [9], "required": False},
        "re:^md_.*$": {"kind": "R"},
    },
    "frame": {"n_atoms": {"min": 1, "max": 512}, "lattice": "required"},
}


def test_from_dict_builds_typed_rules():
    spec = SchemaSpec.from_dict(SPEC_DICT)
    assert spec.columns[0] == ColumnRule(name="species", kind=Kind.STR)
    assert spec.columns[1] == ColumnRule(name="pos", kind=Kind.REAL, width=3)
    assert spec.columns[2] == ColumnRule(
        name="forces", kind=Kind.REAL, width=3, required=False
    )
    assert spec.columns[3] == ColumnRule(name="descriptor_*", kind=Kind.REAL, count=5)
    assert spec.metadata[0] == MetadataRule(name="energy", kind=Kind.REAL)
    assert spec.metadata[1] == MetadataRule(
        name="stress", kind=Kind.REAL, shape=(9,), required=False
    )
    assert spec.frame == FrameRule(
        n_atoms_min=1, n_atoms_max=512, lattice_required=True
    )


def test_unknown_kind_letter_is_rejected():
    with pytest.raises(ValueError, match="kind"):
        SchemaSpec.from_dict({"columns": {"x": {"kind": "Q"}}})


def test_json_round_trips_through_dict():
    spec = SchemaSpec.from_dict(SPEC_DICT)
    assert SchemaSpec.from_dict(json.loads(spec.to_json())) == spec


def test_from_file_reads_yaml_json_toml_identically(tmp_path: Path):
    yaml_text = (
        "columns:\n"
        "  species: {kind: S}\n"
        "  pos: {kind: R, width: 3}\n"
        "metadata:\n"
        "  energy: {kind: R}\n"
    )
    (tmp_path / "s.yaml").write_text(yaml_text)
    (tmp_path / "s.json").write_text(
        json.dumps(
            {
                "columns": {"species": {"kind": "S"}, "pos": {"kind": "R", "width": 3}},
                "metadata": {"energy": {"kind": "R"}},
            }
        )
    )
    (tmp_path / "s.toml").write_text(
        'columns.species.kind = "S"\n'
        'columns.pos.kind = "R"\n'
        "columns.pos.width = 3\n"
        'metadata.energy.kind = "R"\n'
    )
    specs = [
        SchemaSpec.from_file(tmp_path / f"s.{ext}") for ext in ("yaml", "json", "toml")
    ]
    assert specs[0] == specs[1] == specs[2]


def test_from_file_rejects_unknown_extension(tmp_path: Path):
    (tmp_path / "s.txt").write_text("nope")
    with pytest.raises(ValueError, match="extension"):
        SchemaSpec.from_file(tmp_path / "s.txt")


def test_render_yaml_omits_defaults_and_quotes_patterns():
    spec = SchemaSpec.from_dict(SPEC_DICT)
    text = render_yaml(spec)
    assert "species: {kind: S}" in text
    assert "pos: {kind: R, width: 3}" in text
    assert "required: false" in text  # forces
    assert '"descriptor_*": {kind: R, count: 5}' in text
    # rendered text is itself valid and reloads equal
    assert SchemaSpec.from_yaml_text(text) == spec


def test_render_yaml_notes_render_as_trailing_comments():
    spec = SchemaSpec(metadata=(MetadataRule(name="charge", kind=Kind.REAL),))
    text = render_yaml(
        spec, notes={"charge": "drift: R:1 in 3/5, I:1 in 2/5 — using R"}
    )
    assert "charge: {kind: R}  # drift:" in text


def test_to_yaml_method_matches_render_yaml():
    spec = SchemaSpec.from_dict(SPEC_DICT)
    assert spec.to_yaml() == render_yaml(spec)


def test_to_dict_omits_absent_sections():
    cols_only = SchemaSpec(columns=(ColumnRule("pos", Kind.REAL, width=3),))
    assert cols_only.to_dict() == {"columns": {"pos": {"kind": "R", "width": 3}}}

    meta_only = SchemaSpec(metadata=(MetadataRule("energy", Kind.REAL),))
    assert meta_only.to_dict() == {"metadata": {"energy": {"kind": "R"}}}

    assert SchemaSpec().to_dict() == {}


def test_metadata_count_is_serialised():
    spec = SchemaSpec(metadata=(MetadataRule("md_*", Kind.REAL, count=3),))
    assert spec.to_dict()["metadata"]["md_*"] == {"kind": "R", "count": 3}


def test_frame_attrs_partial_bounds_and_lattice():
    assert SchemaSpec(frame=FrameRule(n_atoms_min=1)).to_dict()["frame"] == {
        "n_atoms": {"min": 1}
    }
    assert SchemaSpec(frame=FrameRule(n_atoms_max=5)).to_dict()["frame"] == {
        "n_atoms": {"max": 5}
    }
    assert SchemaSpec(frame=FrameRule(lattice_required=True)).to_dict()["frame"] == {
        "lattice": "required"
    }


def test_empty_frame_rule_renders_no_frame_section():
    empty = SchemaSpec(frame=FrameRule())
    assert empty.to_dict()["frame"] == {}
    assert "frame:" not in render_yaml(empty)


def test_mode_defaults_to_validate_and_roundtrips():
    spec = SchemaSpec.from_dict({"columns": {"pos": {"kind": "R", "width": 3}}})
    assert spec.mode == "validate"
    assert "mode" not in spec.to_dict()  # validate is the default, not emitted


def test_project_mode_roundtrips_through_dict_and_yaml():
    spec = SchemaSpec.from_dict(
        {"mode": "project", "columns": {"pos": {"kind": "R", "width": 3}}}
    )
    assert spec.mode == "project"
    assert spec.to_dict()["mode"] == "project"
    assert "mode: project" in spec.to_yaml()
    assert SchemaSpec.from_yaml_text(spec.to_yaml()).mode == "project"


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="mode"):
        SchemaSpec.from_dict({"mode": "reshape"})


def test_fill_roundtrips_on_column_and_metadata():
    spec = SchemaSpec.from_dict(
        {
            "columns": {"id": {"kind": "I", "required": False, "fill": -1}},
            "metadata": {"tag": {"kind": "S", "required": False, "fill": "none"}},
        }
    )
    assert spec.columns[0].fill == -1
    assert spec.metadata[0].fill == "none"
    reloaded = SchemaSpec.from_yaml_text(spec.to_yaml())
    assert reloaded.columns[0].fill == -1
    assert reloaded.metadata[0].fill == "none"
    assert spec.to_dict()["columns"]["id"]["fill"] == -1


def test_freeze_expands_patterns_and_marks_optional(tmp_path):
    f = tmp_path / "mixed.xyz"
    # frame 1 has d_0,d_1; frame 2 has only d_0 -> d_1 present in some frames
    f.write_text(
        "1\nProperties=species:S:1:pos:R:3:d_0:R:1:d_1:R:1\nH 0 0 0 0.1 0.2\n"
        "1\nProperties=species:S:1:pos:R:3:d_0:R:1\nH 0 0 0 0.3\n"
    )
    spec = SchemaSpec(columns=(ColumnRule(name="d_*", kind=Kind.REAL),), mode="project")
    frozen = spec.freeze(f)
    assert frozen.mode == "project"
    names = {c.name: c for c in frozen.columns}
    assert names.keys() == {"d_0", "d_1"}
    assert names["d_0"].required is True  # in every frame
    assert names["d_1"].required is False  # only some frames


def test_freeze_expands_metadata_patterns(tmp_path):
    from oxyz._schema_spec import MetadataRule

    f = tmp_path / "meta.xyz"
    # frame 1 has e_a and e_b metadata; frame 2 only e_a -> e_b optional
    f.write_text(
        "1\ne_a=1.0 e_b=2.0 Properties=species:S:1:pos:R:3\nH 0 0 0\n"
        "1\ne_a=3.0 Properties=species:S:1:pos:R:3\nH 0 0 0\n"
    )
    spec = SchemaSpec(
        metadata=(MetadataRule(name="e_*", kind=Kind.REAL),), mode="project"
    )
    frozen = spec.freeze(f)
    names = {m.name: m for m in frozen.metadata}
    assert names.keys() == {"e_a", "e_b"}
    assert names["e_a"].required is True
    assert names["e_b"].required is False  # only in some frames, fills NaN


def test_freeze_raises_on_kind_conflict(tmp_path):
    from oxyz import SchemaError

    f = tmp_path / "conflict.xyz"
    f.write_text(
        "1\nProperties=species:S:1:pos:R:3:x:R:1\nH 0 0 0 0.1\n"
        "1\nProperties=species:S:1:pos:R:3:x:S:1\nH 0 0 0 tag\n"
    )
    spec = SchemaSpec(columns=(ColumnRule(name="x*", kind=Kind.REAL),), mode="project")
    with pytest.raises(SchemaError, match="x"):
        spec.freeze(f)


def test_string_fill_with_quotes_roundtrips_through_yaml():
    spec = SchemaSpec.from_dict(
        {"columns": {"tag": {"kind": "S", "required": False, "fill": 'a"b\\c'}}}
    )
    reloaded = SchemaSpec.from_yaml_text(spec.to_yaml())
    assert reloaded.columns[0].fill == 'a"b\\c'


def test_freeze_optional_non_real_without_fill_raises(tmp_path):
    import pytest

    from oxyz._schema_match import SchemaError

    f = tmp_path / "mixed.xyz"
    # label:S present only in frame 1 -> optional STR, no natural null, no fill
    f.write_text(
        "1\nProperties=species:S:1:pos:R:3:label:S:1\nH 0 0 0 a\n"
        "1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n"
    )
    spec = SchemaSpec(columns=(ColumnRule("label*", Kind.STR),), mode="project")
    with pytest.raises(SchemaError, match="fill"):
        spec.freeze(f)
