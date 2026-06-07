use std::path::PathBuf;

use atomflow_core::read_first_frame;

#[test]
fn reads_simple_extxyz_fixture() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/simple.extxyz");

    let frame = read_first_frame(path).unwrap();

    assert_eq!(frame.numbers, vec![1]);

    assert_eq!(frame.positions, vec![[0.0, 0.0, 0.0]]);

    assert_eq!(frame.forces, vec![[0.0, 0.0, 0.0]]);

    assert_eq!(frame.energy, -1.0);

    assert_eq!(
        frame.cell,
        [[15.0, 0.0, 0.0], [0.0, 15.0, 0.0], [0.0, 0.0, 15.0],]
    );

    assert_eq!(frame.stress, [0.0; 6]);

    assert_eq!(frame.pbc, [true, true, true]);
}
