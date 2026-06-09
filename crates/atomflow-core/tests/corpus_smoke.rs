//! Corpus scoreboard: every fixture in `tests/data/` is run through the
//! parser, and the outcome is asserted against an explicit expectation.
//!
//! Most fixtures are *expected to fail* today — they document real-world
//! extxyz variation the spike parser doesn't handle yet. Refining the data
//! model means flipping entries from an error expectation to `Parses`, one
//! family at a time. `tests/data/README.md` documents what each fixture
//! exercises.

use std::{fs, path::PathBuf};

use atomflow_core::{ExtxyzError, read_first_frame};

/// What today's parser is expected to do with a fixture.
///
/// This is a small enum rather than a stored `Result<Frame, ExtxyzError>`
/// because `ExtxyzError` can't be compared with `==`: its `Io` variant wraps
/// `std::io::Error`, which implements neither `PartialEq` nor `Clone`, so the
/// derive isn't available. Pattern matching (in the test body) sidesteps that:
/// we describe the *shape* of the expected error, not a full value.
#[derive(Debug)]
enum Expected {
    /// Parses successfully today.
    Parses,
    /// Rejected: the Properties descriptor isn't the single hard-coded string
    /// the spike parser accepts.
    UnsupportedProperties,
    /// Rejected: a metadata key the spike parser requires is absent.
    MissingMetadata(&'static str),
}

/// The scoreboard. Every fixture file must have an entry.
///
/// Returning `Option` instead of panicking here lets the caller turn "fixture
/// with no entry" into a collected test failure, so a new fixture can't be
/// added without deciding what the parser should do with it.
fn expected(file_name: &str) -> Option<Expected> {
    use Expected::*;

    Some(match file_name {
        "simple.extxyz" => Parses,
        "nonorthogonal.extxyz" => Parses,
        "minimal_periodic.extxyz" => UnsupportedProperties,
        "periodic_pbc_ttf.extxyz" => UnsupportedProperties,
        "no_lattice_molecule.xyz" => UnsupportedProperties,
        "atomic_numbers_z.extxyz" => UnsupportedProperties,
        "mass_and_charge.extxyz" => UnsupportedProperties,
        "id_and_selection.extxyz" => UnsupportedProperties,
        "force_singular_alias.extxyz" => UnsupportedProperties,
        "quoted_strings_booleans_scalars.extxyz" => UnsupportedProperties,
        "newstyle_array_metadata.extxyz" => UnsupportedProperties,
        "stress_voigt6.extxyz" => UnsupportedProperties,
        "stress_matrix9.extxyz" => UnsupportedProperties,
        "virial_matrix9.extxyz" => UnsupportedProperties,
        // These two match the hard-coded Properties string and carry energy,
        // so they get further before tripping the required-stress check.
        "two_frame_same_schema.xyz" => MissingMetadata("stress"),
        "varying_atom_counts.xyz" => MissingMetadata("stress"),
        "molecule_type_labels.extxyz" => UnsupportedProperties,
        "mace_ref_energy_forces_stress.xyz" => UnsupportedProperties,
        "mace_isolated_atom_and_head.xyz" => UnsupportedProperties,
        _ => return None,
    })
}

#[test]
fn corpus_matches_scoreboard() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data");

    // Collect mismatches instead of asserting inside the loop: a scoreboard
    // is most useful when one run shows the whole board, not just the first
    // regression.
    let mut mismatches = Vec::new();
    let mut checked = 0;

    for entry in fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();

        // Skip README.md and anything else that isn't a fixture. `matches!`
        // is the expression form of a `match` that only asks "does this
        // pattern fit?" — `"xyz" | "extxyz"` is one pattern with two
        // alternatives.
        let extension = path.extension().and_then(|e| e.to_str());
        if !matches!(extension, Some("xyz" | "extxyz")) {
            continue;
        }

        let name = path.file_name().unwrap().to_string_lossy().into_owned();

        let Some(expected) = expected(&name) else {
            mismatches.push(format!(
                "{name}: no scoreboard entry; add one to expected()"
            ));
            continue;
        };
        checked += 1;

        let result = read_first_frame(&path);

        // Matching on references (`&expected`, `&result`) keeps both values
        // alive for the error message below; a match by value would move
        // them into the arms.
        let outcome_matches = match (&expected, &result) {
            (Expected::Parses, Ok(_)) => true,
            (Expected::UnsupportedProperties, Err(ExtxyzError::UnsupportedProperties { .. })) => {
                true
            }
            (Expected::MissingMetadata(key), Err(ExtxyzError::MissingMetadata { key: actual })) => {
                key == actual
            }
            _ => false,
        };

        if !outcome_matches {
            mismatches.push(format!("{name}: expected {expected:?}, got {result:?}"));
        }
    }

    assert!(checked > 0, "no fixtures found in {}", dir.display());
    assert!(
        mismatches.is_empty(),
        "corpus scoreboard mismatches:\n{}",
        mismatches.join("\n")
    );
}
