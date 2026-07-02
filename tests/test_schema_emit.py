from __future__ import annotations

from pathlib import Path

from oxyz import Kind, infer_schema, read_frames

DATA = Path(__file__).parent / "data"


def test_to_spec_of_consistent_file_revalidates_under_strict():
    schema = infer_schema(DATA / "schema_conformant.extxyz")
    spec = schema.to_spec()
    # round-trip: emitted spec validates its own source under strict
    assert (
        len(
            read_frames(
                DATA / "schema_conformant.extxyz", schema=spec, conformance="strict"
            )
        )
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
    assert glob.count == 5 and glob.kind is Kind.REAL


def test_no_frame_section_emitted():
    assert infer_schema(DATA / "schema_conformant.extxyz").to_spec().frame is None
