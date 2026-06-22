//! Corpus scoreboard: every frame of every fixture in `tests/data/` must
//! parse.
//!
//! Per-fixture error expectations will return when malformed fixtures
//! arrive, in their own `invalid/` subdirectory.

use std::{fs, path::PathBuf};

use oxyz_core::read_frames;

/// Kept explicit so adding a fixture without an entry, or deleting one the
/// suite still expects, fails loudly.
const FIXTURES: &[&str] = &[
    "atomic_numbers_z.extxyz",
    "force_singular_alias.extxyz",
    "id_and_selection.extxyz",
    "mace_isolated_atom_and_head.xyz",
    "mace_ref_energy_forces_stress.xyz",
    "mass_and_charge.extxyz",
    "minimal_periodic.extxyz",
    "molecule_type_labels.extxyz",
    "newstyle_array_metadata.extxyz",
    "no_lattice_molecule.xyz",
    "nonorthogonal.extxyz",
    "per_atom_boolean.extxyz",
    "periodic_pbc_ttf.extxyz",
    "quoted_strings_booleans_scalars.extxyz",
    "simple.extxyz",
    "singlequote_metadata.extxyz",
    "stress_matrix9.extxyz",
    "stress_voigt6.extxyz",
    "two_frame_same_schema.xyz",
    "varying_atom_counts.xyz",
    "virial_matrix9.extxyz",
];

#[test]
fn every_fixture_parses() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data");

    // Collect everything so one run shows the whole board.
    let mut mismatches = Vec::new();
    let mut seen = Vec::new();

    for entry in fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();

        // Skip README.md and anything else that isn't a fixture.
        let extension = path.extension().and_then(|e| e.to_str());
        if !matches!(extension, Some("xyz" | "extxyz")) {
            continue;
        }

        let name = path.file_name().unwrap().to_string_lossy().into_owned();

        if !FIXTURES.contains(&name.as_str()) {
            mismatches.push(format!("{name}: not listed in FIXTURES; add it"));
            continue;
        }

        if let Err(error) = read_frames(&path) {
            mismatches.push(format!("{name}: failed to parse: {error}"));
        }

        seen.push(name);
    }

    for name in FIXTURES {
        if !seen.iter().any(|s| s == name) {
            mismatches.push(format!(
                "{name}: listed in FIXTURES but missing from tests/data"
            ));
        }
    }

    assert!(
        mismatches.is_empty(),
        "corpus scoreboard mismatches:\n{}",
        mismatches.join("\n")
    );
}
