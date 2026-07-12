use std::{io::Cursor, path::PathBuf};

use oxyz_core::{
    BatchBuilder, ColumnData, FrameIter, IndexedFrames, iter_batches, read_all_batch, read_frames,
};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/data")
        .join(name)
}

fn batch_from(text: &str) -> Result<oxyz_core::Batch, oxyz_core::ExtxyzError> {
    let mut builder = BatchBuilder::new();
    for frame in FrameIter::new(Cursor::new(text)) {
        builder.push(frame?)?;
    }
    Ok(builder.finish()?)
}

#[test]
fn concatenates_a_varying_trajectory() {
    let frames = read_frames(fixture("varying_atom_counts.xyz")).unwrap();
    let mut builder = BatchBuilder::new();
    for frame in frames.clone() {
        builder.push(frame).unwrap();
    }
    let batch = builder.finish().unwrap();

    assert_eq!(batch.offsets, [0, 3, 4, 6]);
    assert_eq!(batch.n_frames(), 3);
    assert_eq!(batch.total_atoms(), 6);

    // Atom-major concatenation: each column's rows line up with offsets.
    let pos = batch.columns.iter().find(|c| c.name == "pos").unwrap();
    assert_eq!(pos.width, 3);
    let ColumnData::Real(values) = &pos.data else {
        panic!("pos should be Real");
    };
    assert_eq!(values.len(), 6 * 3);
    let ColumnData::Real(first_frame_pos) = &frames[0].columns[1].data else {
        panic!();
    };
    assert_eq!(&values[..9], &first_frame_pos[..]);

    // Frame-major metadata: one energy per frame.
    let energy = batch.metadata.iter().find(|c| c.name == "energy").unwrap();
    let ColumnData::Real(values) = &energy.data else {
        panic!("energy should be Real");
    };
    assert_eq!(values, &[-76.3, -13.6, -31.8]);
}

#[test]
fn int_real_pairs_promote_to_real() {
    // energy flips Int -> Real between frames; tag column flips Real -> Int.
    let text = "\
1
Properties=species:S:1:pos:R:3:tag:R:1 energy=-1
H 0 0 0 0.5
1
Properties=species:S:1:pos:R:3:tag:I:1 energy=-1.5
H 0 0 0 2
";
    let batch = batch_from(text).unwrap();

    let energy = batch.metadata.iter().find(|c| c.name == "energy").unwrap();
    assert_eq!(energy.data, ColumnData::Real(vec![-1.0, -1.5]));

    let tag = batch.columns.iter().find(|c| c.name == "tag").unwrap();
    assert_eq!(tag.data, ColumnData::Real(vec![0.5, 2.0]));
}

#[test]
fn kind_conflicts_and_drifting_keys_are_errors() {
    let kind_conflict = "\
1
Properties=species:S:1:pos:R:3 tag=1
H 0 0 0
1
Properties=species:S:1:pos:R:3 tag=reference
H 0 0 0
";
    let error = batch_from(kind_conflict).unwrap_err();
    assert!(error.to_string().contains("metadata \"tag\""), "{error}");
    assert!(
        error.to_string().contains("expected I:1, found S:1"),
        "{error}"
    );

    let missing_key = "\
1
Properties=species:S:1:pos:R:3 energy=-1
H 0 0 0
1
Properties=species:S:1:pos:R:3
H 0 0 0
";
    let error = batch_from(missing_key).unwrap_err();
    assert_eq!(error.to_string(), "frame 1 is missing metadata \"energy\"");

    let extra_column = "\
1
Properties=species:S:1:pos:R:3
H 0 0 0
1
Properties=species:S:1:pos:R:3:forces:R:3
H 0 0 0 0 0 0
";
    let error = batch_from(extra_column).unwrap_err();
    assert_eq!(
        error.to_string(),
        "frame 1 has unexpected column \"forces\""
    );

    let width_conflict = "\
1
Properties=species:S:1:pos:R:3 stress=\"1 2 3 4 5 6\"
H 0 0 0
1
Properties=species:S:1:pos:R:3 stress=\"1 2 3 4 5 6 7 8 9\"
H 0 0 0
";
    let error = batch_from(width_conflict).unwrap_err();
    assert!(
        error.to_string().contains("expected I:6, found I:9"),
        "{error}"
    );
}

#[test]
fn read_all_batch_matches_manual_concatenation() {
    let path = fixture("varying_atom_counts.xyz");
    let mut builder = BatchBuilder::new();
    for frame in read_frames(&path).unwrap() {
        builder.push(frame).unwrap();
    }
    assert_eq!(read_all_batch(&path).unwrap(), builder.finish().unwrap());
}

#[test]
fn read_all_batch_of_empty_file_is_the_empty_batch() {
    let path = std::env::temp_dir().join("oxyz_read_all_batch_empty.xyz");
    std::fs::write(&path, "").unwrap();
    let batch = read_all_batch(&path).unwrap();
    let _ = std::fs::remove_file(&path);

    assert_eq!(batch.offsets, [0]);
    assert_eq!(batch.n_frames(), 0);
    assert!(batch.columns.is_empty());
    assert!(batch.metadata.is_empty());
}

#[cfg(feature = "parallel")]
#[test]
fn read_all_batch_parallel_matches_serial() {
    let path = fixture("varying_atom_counts.xyz");
    assert_eq!(
        oxyz_core::read_all_batch_parallel(&path, None).unwrap(),
        read_all_batch(&path).unwrap()
    );
}

#[test]
fn empty_builder_refuses_to_finish() {
    let error = BatchBuilder::new().finish().unwrap_err();
    assert_eq!(error.to_string(), "batch is empty");
}

#[test]
fn iter_batches_chunks_with_smaller_tail() {
    let batches: Vec<_> = iter_batches(fixture("varying_atom_counts.xyz"), 2)
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();

    assert_eq!(batches.len(), 2);
    assert_eq!(batches[0].offsets, [0, 3, 4]);
    assert_eq!(batches[1].offsets, [0, 2]);
}

#[test]
fn iter_batches_rejects_zero() {
    let Err(error) = iter_batches(fixture("varying_atom_counts.xyz"), 0) else {
        panic!("zero frames_per_batch must be rejected");
    };
    assert_eq!(error.to_string(), "frames per batch must be at least 1");
}

#[test]
fn get_batch_gathers_in_requested_order() {
    let frames = read_frames(fixture("varying_atom_counts.xyz")).unwrap();
    let mut indexed = IndexedFrames::open(fixture("varying_atom_counts.xyz")).unwrap();

    let batch = indexed.get_batch(&[2, 0]).unwrap();
    assert_eq!(batch.offsets, [0, 2, 5]);

    let energy = batch.metadata.iter().find(|c| c.name == "energy").unwrap();
    assert_eq!(energy.data, ColumnData::Real(vec![-31.8, -76.3]));

    let pos = batch.columns.iter().find(|c| c.name == "pos").unwrap();
    let ColumnData::Real(values) = &pos.data else {
        panic!();
    };
    let ColumnData::Real(frame2_pos) = &frames[2].columns[1].data else {
        panic!();
    };
    assert_eq!(&values[..6], &frame2_pos[..]);
}

#[test]
fn batch_reports_missing_column_unexpected_metadata_and_column_mismatch() {
    // A column present in the first frame but absent later.
    let missing_column = "\
1
Properties=species:S:1:pos:R:3:forces:R:3
H 0 0 0 0 0 0
1
Properties=species:S:1:pos:R:3
H 0 0 0
";
    assert_eq!(
        batch_from(missing_column).unwrap_err().to_string(),
        "frame 1 is missing column \"forces\""
    );

    // A metadata key that only a later frame carries.
    let extra_metadata = "\
1
Properties=species:S:1:pos:R:3
H 0 0 0
1
Properties=species:S:1:pos:R:3 extra=5
H 0 0 0
";
    assert_eq!(
        batch_from(extra_metadata).unwrap_err().to_string(),
        "frame 1 has unexpected metadata \"extra\""
    );

    // A per-atom column whose width changes between frames (R:1 then R:2).
    let column_mismatch = "\
1
Properties=species:S:1:pos:R:3:q:R:1
H 0 0 0 0.5
2
Properties=species:S:1:pos:R:3:q:R:2
H 0 0 0 0.1 0.2
O 0 0 0 0.3 0.4
";
    let error = batch_from(column_mismatch).unwrap_err().to_string();
    assert!(error.contains("column \"q\""), "{error}");
    assert!(error.contains("expected R:1, found R:2"), "{error}");
}
