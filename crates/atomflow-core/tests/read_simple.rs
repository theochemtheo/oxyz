use std::path::PathBuf;

use atomflow_core::read_first_frame;

#[test]
fn reads_simple_extxyz_fixture() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/simple.extxyz");

    let frame = read_first_frame(path).unwrap();

    assert_eq!(frame.numbers, vec![1]);

    assert_eq!(frame.positions, vec![0.0, 0.0, 0.0]);

    assert_eq!(frame.forces, vec![0.0, 0.0, 0.0]);

    assert_eq!(frame.energy, -1.0);

    assert_eq!(frame.cell, [15.0, 0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 15.0]);

    assert_eq!(frame.stress, [0.0; 6]);

    assert_eq!(frame.pbc, [true, true, true]);
}

#[test]
fn reads_nonorthogonal_fixture_as_row_major_flat_buffers() {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/nonorthogonal.extxyz");

    let frame = read_first_frame(path).unwrap();

    assert_eq!(frame.numbers, vec![1, 1]);

    assert_eq!(frame.positions, vec![0.0, 0.1, 0.2, 3.0, 3.1, 3.2,]);

    assert_eq!(frame.forces, vec![1.0, 1.1, 1.2, -1.0, -1.1, -1.2,]);

    assert_eq!(frame.energy, -2.0);

    assert_eq!(
        frame.cell,
        [10.0, 0.0, 0.0, 1.0, 11.0, 0.0, 2.0, 3.0, 12.0,]
    );

    assert_eq!(frame.stress, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

    assert_eq!(frame.pbc, [true, false, true]);
}
