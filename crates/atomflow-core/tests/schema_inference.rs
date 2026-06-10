use std::{io::Cursor, path::PathBuf};

use atomflow_core::schema::{Schema, ValueType};
use atomflow_core::{ColumnKind, FrameIter, infer_schema};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/data")
        .join(name)
}

#[test]
fn infers_schema_of_a_varying_trajectory() {
    let schema = infer_schema(fixture("varying_atom_counts.xyz")).unwrap();

    assert_eq!(schema.n_frames, 3);
    assert_eq!(schema.total_atoms, 6);
    assert_eq!(schema.min_atoms, Some(1));
    assert_eq!(schema.max_atoms, Some(3));

    let names: Vec<&str> = schema
        .columns
        .iter()
        .map(|column| column.name.as_str())
        .collect();
    assert_eq!(names, ["species", "pos", "forces"]);

    // Stable schema: every column has one variant, present in every frame.
    for column in &schema.columns {
        assert_eq!(column.frames_present, 3);
        assert_eq!(column.variants.len(), 1);
    }

    let pos = &schema.columns[1];
    assert_eq!(pos.variants[0].kind, ColumnKind::Real);
    assert_eq!(pos.variants[0].width, 3);

    let energy = schema
        .metadata
        .iter()
        .find(|entry| entry.key == "energy")
        .unwrap();
    assert_eq!(energy.variants, [(ValueType::Real, 3)]);
}

#[test]
fn records_presence_type_and_shape_conflicts() {
    // Frame 0: Int energy, 6-value stress, extra `tag`. Frame 1: Real energy,
    // 9-value stress, no `tag`.
    let text = "\
1
Properties=species:S:1:pos:R:3 energy=-1 stress=\"1 2 3 4 5 6\" tag=reference
H 0 0 0
1
Properties=species:S:1:pos:R:3 energy=-1.5 stress=\"1 2 3 4 5 6 7 8 9\"
H 0 0 0
";

    let mut schema = Schema::default();
    for frame in FrameIter::new(Cursor::new(text)) {
        schema.observe(&frame.unwrap());
    }

    let energy = schema
        .metadata
        .iter()
        .find(|entry| entry.key == "energy")
        .unwrap();
    assert_eq!(energy.variants, [(ValueType::Int, 1), (ValueType::Real, 1)]);

    let stress = schema
        .metadata
        .iter()
        .find(|entry| entry.key == "stress")
        .unwrap();
    assert_eq!(
        stress.variants,
        [(ValueType::IntArray(6), 1), (ValueType::IntArray(9), 1)]
    );

    let tag = schema
        .metadata
        .iter()
        .find(|entry| entry.key == "tag")
        .unwrap();
    assert_eq!(tag.frames_present, 1);

    let report = schema.to_string();
    assert!(report.contains("energy: Int (1/2 frames), Real (1/2 frames) (unifies to Real)"));
    assert!(report.contains("[inconsistent]"));
    assert!(report.contains("tag: Str (1/2 frames)"));
}

#[test]
fn report_summarises_a_stable_file() {
    let schema = infer_schema(fixture("two_frame_same_schema.xyz")).unwrap();
    let report = schema.to_string();

    assert!(report.contains("2 frames, 4 atoms (min 2, max 2)"));
    assert!(report.contains("pos: R:3 (2/2 frames)"));
    assert!(report.contains("forces: R:3 (2/2 frames)"));
    assert!(report.contains("energy: Real (2/2 frames)"));
}
