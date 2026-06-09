//! Detailed spot-checks of parsed frames against the columnar model.
//!
//! `corpus_smoke.rs` asserts that every fixture parses; these tests assert
//! *what* a representative handful parse into.

use std::path::PathBuf;

use atomflow_core::{ColumnData, Value, read_first_frame};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/data")
        .join(name)
}

#[test]
fn reads_simple_extxyz_fixture() {
    let frame = read_first_frame(fixture("simple.extxyz")).unwrap();

    assert_eq!(frame.n_atoms, 1);

    let names: Vec<&str> = frame
        .columns
        .iter()
        .map(|column| column.name.as_str())
        .collect();
    assert_eq!(names, ["species", "pos", "forces"]);

    let species = frame.column("species").unwrap();
    assert_eq!(species.width, 1);
    assert_eq!(species.data, ColumnData::Str(vec!["H".to_owned()]));

    let pos = frame.column("pos").unwrap();
    assert_eq!(pos.width, 3);
    assert_eq!(pos.data.as_real().unwrap(), [0.0, 0.0, 0.0]);

    let forces = frame.column("forces").unwrap();
    assert_eq!(forces.width, 3);
    assert_eq!(forces.data.as_real().unwrap(), [0.0, 0.0, 0.0]);

    assert_eq!(frame.metadata_value("energy"), Some(&Value::Real(-1.0)));
    assert_eq!(
        frame.metadata_value("Lattice"),
        Some(&Value::RealArray(vec![
            15.0, 0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 15.0
        ]))
    );
    assert_eq!(
        frame.metadata_value("stress"),
        Some(&Value::RealArray(vec![0.0; 6]))
    );
    assert_eq!(
        frame.metadata_value("pbc"),
        Some(&Value::BoolArray(vec![true, true, true]))
    );

    // Properties is consumed into columns, not duplicated in metadata.
    assert_eq!(frame.metadata_value("Properties"), None);
}

#[test]
fn preserves_lattice_in_as_written_order() {
    let frame = read_first_frame(fixture("nonorthogonal.extxyz")).unwrap();

    // Supersedes the original spike, which reordered Lattice into row-major
    // at parse time: the nine values now stay as written, and conversion is
    // the normalisation layer's job.
    assert_eq!(
        frame.metadata_value("Lattice"),
        Some(&Value::RealArray(vec![
            10.0, 1.0, 2.0, 0.0, 11.0, 3.0, 0.0, 0.0, 12.0
        ]))
    );

    let pos = frame.column("pos").unwrap();
    assert_eq!(pos.data.as_real().unwrap(), [0.0, 0.1, 0.2, 3.0, 3.1, 3.2]);

    let forces = frame.column("forces").unwrap();
    assert_eq!(
        forces.data.as_real().unwrap(),
        [1.0, 1.1, 1.2, -1.0, -1.1, -1.2]
    );

    assert_eq!(
        frame.metadata_value("pbc"),
        Some(&Value::BoolArray(vec![true, false, true]))
    );
}

#[test]
fn reads_integer_columns_with_species_mid_row() {
    let frame = read_first_frame(fixture("id_and_selection.extxyz")).unwrap();

    let names: Vec<&str> = frame
        .columns
        .iter()
        .map(|column| column.name.as_str())
        .collect();
    assert_eq!(names, ["id", "species", "pos", "selection"]);

    let id = frame.column("id").unwrap();
    assert_eq!(id.data.as_int().unwrap(), [10, 11, 12]);

    let species = frame.column("species").unwrap();
    assert_eq!(
        species.data,
        ColumnData::Str(vec!["Si".to_owned(), "Si".to_owned(), "O".to_owned()])
    );

    let selection = frame.column("selection").unwrap();
    assert_eq!(selection.data.as_int().unwrap(), [1, 0, 1]);
}

#[test]
fn reads_per_atom_string_column_and_any_element() {
    let frame = read_first_frame(fixture("molecule_type_labels.extxyz")).unwrap();

    // Species are plain strings — Ar needs no element-table support.
    let species = frame.column("species").unwrap();
    assert_eq!(
        species.data,
        ColumnData::Str(vec![
            "O".to_owned(),
            "H".to_owned(),
            "H".to_owned(),
            "Ar".to_owned()
        ])
    );

    let labels = frame.column("molecule_type").unwrap();
    assert_eq!(labels.width, 1);
    assert_eq!(
        labels.data,
        ColumnData::Str(vec![
            "water".to_owned(),
            "water".to_owned(),
            "water".to_owned(),
            "noble_gas".to_owned()
        ])
    );
}

#[test]
fn reads_mace_training_schema_with_raw_names() {
    let frame = read_first_frame(fixture("mace_ref_energy_forces_stress.xyz")).unwrap();

    // REF_* names are preserved raw; mapping them is the normalisation
    // layer's job.
    let ref_forces = frame.column("REF_forces").unwrap();
    assert_eq!(ref_forces.width, 3);
    assert_eq!(ref_forces.data.len(), frame.n_atoms * 3);

    assert_eq!(
        frame.metadata_value("REF_energy"),
        Some(&Value::Real(-76.123))
    );
    assert_eq!(
        frame.metadata_value("REF_stress"),
        Some(&Value::RealArray(vec![0.5, 0.4, 0.3, 0.01, 0.02, 0.03]))
    );
    assert_eq!(
        frame.metadata_value("config_type"),
        Some(&Value::Str("Default".to_owned()))
    );
}

#[test]
fn types_quoted_strings_booleans_and_scalars_from_file() {
    let frame = read_first_frame(fixture("quoted_strings_booleans_scalars.extxyz")).unwrap();

    assert_eq!(
        frame.metadata_value("source"),
        Some(&Value::Str("generated for parser study".to_owned()))
    );
    assert_eq!(
        frame.metadata_value("split"),
        Some(&Value::Str("train".to_owned()))
    );
    assert_eq!(frame.metadata_value("converged"), Some(&Value::Bool(true)));
    assert_eq!(frame.metadata_value("frozen"), Some(&Value::Bool(false)));
    assert_eq!(frame.metadata_value("step"), Some(&Value::Int(12)));
    assert_eq!(
        frame.metadata_value("temperature"),
        Some(&Value::Real(298.15))
    );
}
