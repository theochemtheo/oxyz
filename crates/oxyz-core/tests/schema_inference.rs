use std::{io::Cursor, path::PathBuf};

use oxyz_core::schema::{Schema, ValueType};
use oxyz_core::{ColumnKind, FrameIter, infer_schema};

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
fn unified_promotes_int_real_and_flags_conflicts() {
    // forces drifts Int/Real at equal width (unifiable); stress changes
    // length (genuine conflict); pos is stable (unifies to itself).
    let text = "\
1
Properties=species:S:1:pos:R:3:forces:I:3 energy=-158 stress=\"1.0 2.0 3.0 4.0 5.0 6.0\"
H 0 0 0 0 0 0
1
Properties=species:S:1:pos:R:3:forces:R:3 energy=-1.5 stress=\"1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0\"
H 0 0 0 0.1 0.2 0.3
";

    let mut schema = Schema::default();
    for frame in FrameIter::new(Cursor::new(text)) {
        schema.observe(&frame.unwrap());
    }

    let column = |name: &str| schema.columns.iter().find(|c| c.name == name).unwrap();
    assert_eq!(column("pos").unified(), Some((ColumnKind::Real, 3)));
    assert_eq!(column("forces").unified(), Some((ColumnKind::Real, 3)));

    let entry = |key: &str| schema.metadata.iter().find(|m| m.key == key).unwrap();
    assert_eq!(entry("energy").unified(), Some(ValueType::Real));
    assert_eq!(entry("stress").unified(), None);

    assert!(!schema.is_consistent());
}

#[test]
fn consistency_is_strict() {
    // Stable file: consistent.
    let schema = infer_schema(fixture("two_frame_same_schema.xyz")).unwrap();
    assert!(schema.is_consistent());

    // A key missing from one frame breaks consistency even though every
    // observed variant is stable.
    let text = "\
1
Properties=species:S:1:pos:R:3 energy=-1.0 tag=reference
H 0 0 0
1
Properties=species:S:1:pos:R:3 energy=-1.5
H 0 0 0
";
    let mut schema = Schema::default();
    for frame in FrameIter::new(Cursor::new(text)) {
        schema.observe(&frame.unwrap());
    }
    assert!(!schema.is_consistent());

    // Empty file: vacuously consistent.
    assert!(Schema::default().is_consistent());
}

#[test]
fn duplicate_metadata_keys_collapse_last_wins() {
    // One frame repeating `energy` with a different type each time. The dict
    // view of `Frame` keeps the last value, so the schema must count one
    // occurrence of the last type, not two occurrences of two types.
    let text = "\
1
Properties=species:S:1:pos:R:3 energy=-1 energy=-2.5
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
    assert_eq!(energy.frames_present, 1);
    assert_eq!(energy.variants, [(ValueType::Real, 1)]);
    assert!(schema.is_consistent());
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
