from __future__ import annotations

from pathlib import Path

from oxyz import Kind, infer_schema, read

DATA = Path(__file__).parent / "data"


def test_to_spec_of_consistent_file_revalidates_under_strict():
    schema = infer_schema(DATA / "schema_conformant.extxyz")
    spec = schema.to_spec()
    # round-trip: emitted spec validates its own source under strict
    assert (
        len(read(DATA / "schema_conformant.extxyz", schema=spec, conformance="strict"))
        == 2
    )


def test_partial_presence_marked_optional():
    # schema_extra_column: `charge` appears in 1 of 2 frames
    spec = infer_schema(DATA / "schema_extra_column.extxyz").to_spec()
    charge = next(rule for rule in spec.columns if rule.name == "charge")
    assert charge.required is False


def test_descriptor_family_collapses_to_glob(tmp_path: Path):
    lines = [
        "2",
        "Properties=species:S:1:pos:R:3:"
        + ":".join(f"descriptor_{i}:R:1" for i in range(5)),
    ]
    lines += ["H 0.0 0.0 0.0 " + " ".join("0.0" for _ in range(5))]
    lines += ["O 1.0 0.0 0.0 " + " ".join("0.0" for _ in range(5))]
    path = tmp_path / "desc.extxyz"
    path.write_text("\n".join(lines) + "\n")
    spec = infer_schema(path).to_spec()
    names = {rule.name for rule in spec.columns}
    assert "descriptor_*" in names
    glob = next(rule for rule in spec.columns if rule.name == "descriptor_*")
    assert glob.count == 5
    assert glob.kind is Kind.REAL


def test_no_frame_section_emitted():
    assert infer_schema(DATA / "schema_conformant.extxyz").to_spec().frame is None


def test_partial_member_of_glob_family_not_dropped_or_duplicated(tmp_path):
    f0_props = "species:S:1:pos:R:3:" + ":".join(
        f"descriptor_{i}:R:1" for i in range(5)
    )
    f1_props = "species:S:1:pos:R:3:" + ":".join(
        f"descriptor_{i}:R:1" for i in (0, 1, 3, 4)
    )
    f0 = " ".join(["0.0"] * (3 + 5))
    f1 = " ".join(["0.0"] * (3 + 4))
    text = (
        f"2\nProperties={f0_props}\nH {f0}\nO {f0}\n"
        f"2\nProperties={f1_props}\nH {f1}\nO {f1}\n"
    )
    path = tmp_path / "partial.extxyz"
    path.write_text(text)
    spec = infer_schema(path).to_spec()
    glob_rules = [r for r in spec.columns if r.name == "descriptor_*"]
    assert len(glob_rules) == 1  # not duplicated
    assert glob_rules[0].count == 4
    literal = [r for r in spec.columns if r.name == "descriptor_2"]
    assert len(literal) == 1  # not dropped
    assert literal[0].required is False
    # round-trip: the emitted spec validates its own source under required
    assert len(read(path, schema=spec, conformance="required")) == 2


def test_same_stem_families_with_different_width_not_globbed(tmp_path):
    lo = [f"d_{i}:R:1" for i in range(3)]
    hi = [f"d_{i}:R:3" for i in range(3, 6)]
    props = "species:S:1:pos:R:3:" + ":".join(lo + hi)
    vals = " ".join(["0.0"] * (3 + 3 * 1 + 3 * 3))  # pos(3) + 3xR:1 + 3xR:3
    frame = f"2\nProperties={props}\nH {vals}\nO {vals}\n"
    text = frame + frame
    path = tmp_path / "two_families.extxyz"
    path.write_text(text)
    spec = infer_schema(path).to_spec()
    assert not any(r.name == "d_*" for r in spec.columns)  # not globbed
    assert sum(r.name.startswith("d_") for r in spec.columns) == 6  # all literal
    assert len(read(path, schema=spec, conformance="strict")) == 2  # round-trips


def test_to_spec_emits_drift_note_for_conflicting_column():
    # schema_drift_type has magmom R:3 (frame 0) then R:1 (frame 1): the widths
    # conflict, so `unified` is None and emission falls back to the dominant
    # variant with a drift note rather than a clean rule.
    from oxyz._schema_emit import spec_and_notes

    _, notes = spec_and_notes(infer_schema(DATA / "schema_drift_type.extxyz"))
    assert "magmom" in notes
    assert "drift" in notes["magmom"]


def test_to_spec_emits_drift_note_for_conflicting_metadata(tmp_path):
    # `label` is an Int in frame 0 and a Str in frame 1 — no unifying type, so
    # the metadata entry carries a drift note.
    from oxyz._schema_emit import spec_and_notes

    text = (
        "1\nProperties=species:S:1:pos:R:3 label=1\nH 0.0 0.0 0.0\n"
        "1\nProperties=species:S:1:pos:R:3 label=foo\nH 0.0 0.0 0.0\n"
    )
    path = tmp_path / "meta_drift.extxyz"
    path.write_text(text)
    _, notes = spec_and_notes(infer_schema(path))
    assert "label" in notes
    assert "drift" in notes["label"]
