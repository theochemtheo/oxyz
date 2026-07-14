//! The `invalid/` corpus: every malformed fixture must fail to parse with the
//! leg-2 error contract — the right frame index, the file-absolute line, the
//! column where pinpointable, and actionable wording. Not "an error was
//! raised": the coordinates and wording are asserted per fixture, so the
//! deterministic regression bank also guards the error surface.

use std::path::PathBuf;

use oxyz_core::read_frames;

/// One malformed fixture and its expected error. `column: None` means the
/// failure has no single pinpointable token (e.g. a short atom row); the
/// harness then only asserts frame + line + wording.
struct InvalidCase {
    file: &'static str,
    frame_index: usize,
    line: usize,
    column: Option<usize>,
    wording: &'static str,
}

/// Explicit, like `corpus_smoke.rs`: adding a fixture without an entry (or an
/// entry without a fixture) fails loudly.
const INVALID: &[InvalidCase] = &[
    InvalidCase {
        file: "dangling_metadata_value.extxyz",
        frame_index: 0,
        line: 2,
        column: Some(36),
        wording: "expected 'key=value' pairs",
    },
    InvalidCase {
        file: "unterminated_quote.extxyz",
        frame_index: 0,
        line: 2,
        column: Some(37),
        wording: "expected 'key=value' pairs",
    },
    InvalidCase {
        file: "unknown_property_kind.extxyz",
        frame_index: 0,
        line: 2,
        column: None,
        wording: "unknown Properties kind",
    },
    InvalidCase {
        file: "short_atom_row.extxyz",
        frame_index: 0,
        line: 3,
        column: None,
        wording: "wrong column count",
    },
    InvalidCase {
        file: "bad_atom_value.extxyz",
        frame_index: 0,
        line: 3,
        column: Some(3),
        wording: "invalid real in column",
    },
    InvalidCase {
        file: "bad_atom_count.extxyz",
        frame_index: 0,
        line: 1,
        column: None,
        wording: "expected a non-negative integer",
    },
    InvalidCase {
        file: "truncated_frame.extxyz",
        frame_index: 1,
        line: 7,
        column: None,
        wording: "missing atom line",
    },
    InvalidCase {
        file: "ragged_bracket_array.extxyz",
        frame_index: 0,
        line: 2,
        column: Some(40),
        wording: "expected 'key=value' pairs",
    },
    InvalidCase {
        file: "trailing_comma_array.extxyz",
        frame_index: 0,
        line: 2,
        column: Some(42),
        wording: "expected 'key=value' pairs",
    },
    InvalidCase {
        file: "bare_string_excluded_char.extxyz",
        frame_index: 0,
        line: 2,
        column: Some(38),
        wording: "expected 'key=value' pairs",
    },
];

#[test]
fn every_invalid_fixture_fails_with_contract() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/invalid");
    let mut problems = Vec::new();

    for case in INVALID {
        let path = dir.join(case.file);
        let error = match read_frames(&path) {
            Err(error) => error,
            Ok(_) => {
                problems.push(format!("{}: parsed OK, expected an error", case.file));
                continue;
            }
        };
        if error.frame_index() != Some(case.frame_index) {
            problems.push(format!(
                "{}: frame_index {:?}, expected {}",
                case.file,
                error.frame_index(),
                case.frame_index
            ));
        }
        if error.line() != Some(case.line) {
            problems.push(format!(
                "{}: line {:?}, expected {}",
                case.file,
                error.line(),
                case.line
            ));
        }
        if let Some(column) = case.column {
            if error.column() != Some(column) {
                problems.push(format!(
                    "{}: column {:?}, expected {}",
                    case.file,
                    error.column(),
                    column
                ));
            }
        }
        let message = error.to_string();
        if !message.contains(case.wording) {
            problems.push(format!(
                "{}: message {message:?} missing wording {:?}",
                case.file, case.wording
            ));
        }
    }

    // Guard against orphan fixtures: every file in invalid/ must have an entry.
    for entry in std::fs::read_dir(&dir).unwrap() {
        let name = entry.unwrap().file_name().to_string_lossy().into_owned();
        let ext = std::path::Path::new(&name)
            .extension()
            .and_then(|e| e.to_str());
        if matches!(ext, Some("xyz" | "extxyz")) && !INVALID.iter().any(|c| c.file == name) {
            problems.push(format!("{name}: in invalid/ but not listed in INVALID"));
        }
    }

    assert!(
        problems.is_empty(),
        "invalid corpus problems:\n{}",
        problems.join("\n")
    );
}
