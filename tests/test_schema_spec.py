from __future__ import annotations

import json
from pathlib import Path

import pytest

from oxyz import Kind
from oxyz._schema_spec import (
    ColumnRule,
    FrameRule,
    MetadataRule,
    SchemaSpec,
    render_yaml,
)

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
