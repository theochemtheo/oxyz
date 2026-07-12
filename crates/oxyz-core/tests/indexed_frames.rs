//! Random-access reads through `IndexedFrames`: the batch, projected-batch,
//! volume-scan, and error paths beyond the streamed-parity check in
//! `index_scan.rs`.

use std::path::PathBuf;

use oxyz_core::model::ColumnKind;
use oxyz_core::project::{Fill, PlanColumn, ProjectionPlan};
use oxyz_core::{BatchBuilder, ExtxyzError, IndexedFrames, read_frames};

// Three frames, atom counts 2/1/3, cubic cells of volume 8/27/64, and a
// `charge` column that rides frames 0 and 2 but not 1 — so a plain batch of
// {0,2} is well-formed while a projection over all three fills frame 1.
const TRAJ: &str = "\
2
Lattice=\"2 0 0 0 2 0 0 0 2\" Properties=species:S:1:pos:R:3:charge:R:1 energy=-1.0 pbc=\"T T T\"
H 0 0 0 0.5
O 1 1 1 -0.5
1
Lattice=\"3 0 0 0 3 0 0 0 3\" Properties=species:S:1:pos:R:3 energy=-2.0 pbc=\"T T T\"
H 0 0 0
3
Lattice=\"4 0 0 0 4 0 0 0 4\" Properties=species:S:1:pos:R:3:charge:R:1 energy=-3.0 pbc=\"T T T\"
H 0 0 0 0.1
H 0 0 0 0.2
O 0 0 0 -0.3
";

fn traj_file(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(name);
    std::fs::write(&path, TRAJ).unwrap();
    path
}

fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/data")
        .join(name)
}

// A projection keeping pos and charge, charge filled with a concrete value so
// a filled column compares equal across serial/parallel (NaN != NaN).
fn charge_plan() -> ProjectionPlan {
    ProjectionPlan {
        columns: vec![
            PlanColumn {
                name: "pos".into(),
                kind: ColumnKind::Real,
                width: 3,
                required: true,
                fill: Some(Fill::Real(0.0)),
            },
            PlanColumn {
                name: "charge".into(),
                kind: ColumnKind::Real,
                width: 1,
                required: false,
                fill: Some(Fill::Real(-9.0)),
            },
        ],
        metadata: Vec::new(),
    }
}

#[test]
fn reports_length_and_emptiness() {
    let mut idx = IndexedFrames::open(traj_file("oxyz_idx_len.extxyz")).unwrap();
    assert_eq!(idx.len(), 3);
    assert!(!idx.is_empty());
    assert_eq!(idx.index().n_frames(), 3);
    assert_eq!(idx.index().total_atoms(), 6);
    // get() matches a streamed read of the same frame.
    let streamed = read_frames(traj_file("oxyz_idx_len.extxyz")).unwrap();
    assert_eq!(idx.get(2).unwrap(), streamed[2]);
}

#[test]
fn empty_file_indexes_as_empty() {
    let path = std::env::temp_dir().join("oxyz_idx_empty.extxyz");
    std::fs::write(&path, "").unwrap();
    let idx = IndexedFrames::open(&path).unwrap();
    assert_eq!(idx.len(), 0);
    assert!(idx.is_empty());
}

#[test]
fn get_out_of_range_reports_the_index_and_frame_count() {
    let mut idx = IndexedFrames::open(traj_file("oxyz_idx_oor.extxyz")).unwrap();
    assert!(matches!(
        idx.get(9),
        Err(ExtxyzError::FrameOutOfRange {
            frame_index: 9,
            n_frames: 3
        })
    ));
}

#[test]
fn get_batch_matches_a_manual_concatenation() {
    let path = traj_file("oxyz_idx_batch.extxyz");
    let mut idx = IndexedFrames::open(&path).unwrap();
    let batch = idx.get_batch(&[0, 2]).unwrap();
    assert_eq!(batch.n_frames(), 2);
    assert_eq!(batch.total_atoms(), 5);

    let streamed = read_frames(&path).unwrap();
    let mut builder = BatchBuilder::new();
    builder.push(streamed[0].clone()).unwrap();
    builder.push(streamed[2].clone()).unwrap();
    assert_eq!(batch, builder.finish().unwrap());
}

#[test]
fn empty_and_out_of_range_batches_error() {
    let mut idx = IndexedFrames::open(traj_file("oxyz_idx_batch_err.extxyz")).unwrap();
    assert!(matches!(
        idx.get_batch(&[]),
        Err(ExtxyzError::Batch(oxyz_core::BatchError::Empty))
    ));
    assert!(matches!(
        idx.get_batch_parallel(&[], None),
        Err(ExtxyzError::Batch(oxyz_core::BatchError::Empty))
    ));
    assert!(matches!(
        idx.get_batch(&[0, 9]),
        Err(ExtxyzError::FrameOutOfRange { frame_index: 9, .. })
    ));
    assert!(matches!(
        idx.get_batch_parallel(&[0, 9], Some(2)),
        Err(ExtxyzError::FrameOutOfRange { frame_index: 9, .. })
    ));
}

#[test]
fn get_batch_projected_fills_and_parallel_matches_serial() {
    let mut idx = IndexedFrames::open(traj_file("oxyz_idx_proj.extxyz")).unwrap();
    let plan = charge_plan();

    let serial = idx.get_batch_projected(&[0, 1, 2], &plan).unwrap();
    assert_eq!(serial.survivors, vec![0, 1, 2]); // all kept: charge is optional
    let charge = serial
        .batch
        .columns
        .iter()
        .find(|c| c.name == "charge")
        .unwrap();
    // Frame 1 had no charge, so its single atom takes the fill.
    assert_eq!(
        charge.data.as_real().unwrap(),
        &[0.5, -0.5, -9.0, 0.1, 0.2, -0.3]
    );

    for threads in [None, Some(2)] {
        let par = idx
            .get_batch_projected_parallel(&[0, 1, 2], threads, &plan)
            .unwrap();
        assert_eq!(par.survivors, serial.survivors, "threads={threads:?}");
        assert_eq!(par.batch, serial.batch, "threads={threads:?}");
        assert_eq!(par.reports, serial.reports, "threads={threads:?}");
    }
}

#[test]
fn projected_batch_rejects_an_empty_selection() {
    let mut idx = IndexedFrames::open(traj_file("oxyz_idx_proj_empty.extxyz")).unwrap();
    let plan = charge_plan();
    assert!(matches!(
        idx.get_batch_projected(&[], &plan),
        Err(ExtxyzError::Batch(oxyz_core::BatchError::Empty))
    ));
    assert!(matches!(
        idx.get_batch_projected_parallel(&[], None, &plan),
        Err(ExtxyzError::Batch(oxyz_core::BatchError::Empty))
    ));
}

#[test]
fn open_with_volume_records_cell_volumes() {
    let idx = IndexedFrames::open_with_volume(traj_file("oxyz_idx_vol.extxyz")).unwrap();
    let volumes = idx.index().volumes().expect("volumes recorded");
    assert_eq!(volumes.len(), 3);
    for (got, want) in volumes.iter().zip([8.0, 27.0, 64.0]) {
        assert!((got - want).abs() < 1e-9, "volume {got} != {want}");
    }
}

#[test]
fn random_access_refuses_a_compressed_file() {
    assert!(matches!(
        IndexedFrames::open(data("compressed/two_frame.xyz.gz")),
        Err(ExtxyzError::RandomAccessUnsupported)
    ));
}
