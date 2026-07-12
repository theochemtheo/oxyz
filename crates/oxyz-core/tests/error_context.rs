//! The parse-error location contract: every parse error reports the frame it
//! occurred in, the file-absolute line, and — where a token is pinpointable —
//! the 1-based character column, both structurally (accessors) and in the message.

use oxyz_core::ExtxyzError;

/// A hand-built `InFrame(Located(inner))` reports each coordinate and renders
/// them ahead of the inner message.
#[test]
fn located_error_exposes_and_renders_coordinates() {
    let inner = ExtxyzError::InvalidAtomValue {
        column: "forces".to_owned(),
        kind: "real",
        value: "abc".to_owned(),
    };
    let located = ExtxyzError::Located {
        line: 12,
        column: Some(5),
        source: Box::new(inner),
    };
    let error = ExtxyzError::InFrame {
        frame_index: 3,
        source: Box::new(located),
    };

    assert_eq!(error.frame_index(), Some(3));
    assert_eq!(error.line(), Some(12));
    assert_eq!(error.column(), Some(5));

    let rendered = error.to_string();
    assert!(rendered.contains("frame 3"), "{rendered}");
    assert!(rendered.contains("line 12"), "{rendered}");
    assert!(rendered.contains("column 5"), "{rendered}");
    assert!(rendered.contains("forces"), "{rendered}");
}

/// A `Located` with no column still reports the line and omits the column text.
#[test]
fn located_without_column_reports_line_only() {
    let error = ExtxyzError::Located {
        line: 7,
        column: None,
        source: Box::new(ExtxyzError::MissingLine("atom")),
    };
    assert_eq!(error.line(), Some(7));
    assert_eq!(error.column(), None);
    let rendered = error.to_string();
    assert!(rendered.contains("line 7"), "{rendered}");
    assert!(!rendered.contains("column"), "{rendered}");
}

use oxyz_core::iter_frames_from;
use std::io::Cursor;

fn first_error(input: &str) -> ExtxyzError {
    iter_frames_from(Cursor::new(input.as_bytes()))
        .unwrap()
        .find_map(Result::err)
        .expect("expected a parse error")
}

/// A non-numeric atom value reports its frame, its file-absolute line, and the
/// 1-based character column where the bad token starts.
#[test]
fn bad_atom_value_locates_line_and_column() {
    // Line 1: count, line 2: comment, line 3: atom row. "abc" starts at col 3.
    let input = "1\nProperties=species:S:1:pos:R:3\nH abc 0.0 0.0\n";
    let error = first_error(input);
    assert_eq!(error.frame_index(), Some(0));
    assert_eq!(error.line(), Some(3));
    assert_eq!(error.column(), Some(3));
}

/// A malformed comment (`=` with no value) reports the comment line and the
/// character column of the failure.
#[test]
fn bad_metadata_locates_comment_line_and_column() {
    let input = "1\nProperties=species:S:1:pos:R:3 bad=\nH 0.0 0.0 0.0\n";
    let error = first_error(input);
    assert_eq!(error.frame_index(), Some(0));
    assert_eq!(error.line(), Some(2));
    assert!(error.column().is_some());
}

/// A short atom row (too few columns) reports the atom line; no single token to
/// pin a column on.
#[test]
fn wrong_column_count_locates_line_only() {
    let input = "1\nProperties=species:S:1:pos:R:3\nH 0.0 0.0\n";
    let error = first_error(input);
    assert_eq!(error.frame_index(), Some(0));
    assert_eq!(error.line(), Some(3));
    assert_eq!(error.column(), None);
    assert!(error.to_string().contains("wrong column count"), "{error}");
}

/// A non-numeric count line reports that line.
#[test]
fn bad_atom_count_locates_line() {
    let error = first_error("notanumber\ncomment\n");
    assert_eq!(error.line(), Some(1));
}

/// The parallel intra-frame atom path locates a bad value identically to the
/// serial loop, even when the bad row falls inside a worker's split range.
#[cfg(feature = "parallel")]
#[test]
fn parallel_bad_atom_value_matches_serial_location() {
    use oxyz_core::read_frames_parallel_from;
    // A frame large enough to take the intra-frame parallel path (>256 KB),
    // with one bad value on a known line.
    let mut input = String::from("20000\nProperties=species:S:1:pos:R:3\n");
    for _ in 0..9999 {
        input.push_str("H 0.0 0.0 0.0\n");
    }
    input.push_str("H bad 0.0 0.0\n"); // line 10002 (1 count + 1 comment + 10000th row)
    for _ in 0..10000 {
        input.push_str("H 0.0 0.0 0.0\n");
    }
    let error = read_frames_parallel_from(Cursor::new(input.into_bytes()), Some(4))
        .expect_err("expected a parse error");
    assert_eq!(error.frame_index(), Some(0));
    assert_eq!(error.line(), Some(10002));
    assert_eq!(error.column(), Some(3));
}

use oxyz_core::scan_frames;

/// A bad count line surfaced by the structural scan reports its line.
#[test]
fn scan_bad_count_locates_line() {
    // Frame 0 is fine (1 atom); frame 1's count line (line 4) is malformed.
    let input = "1\ncomment\nH 0.0 0.0 0.0\nnotacount\ncomment\n";
    let error = scan_frames(Cursor::new(input.as_bytes())).expect_err("bad count");
    assert_eq!(error.frame_index(), Some(1));
    assert_eq!(error.line(), Some(4));
}

/// A truncated frame surfaced by the scan reports the missing line's number.
#[test]
fn scan_truncated_frame_locates_line() {
    let input = "3\ncomment\nH 0.0 0.0 0.0\n"; // declares 3 atoms, supplies 1
    let error = scan_frames(Cursor::new(input.as_bytes())).expect_err("truncated");
    assert_eq!(error.line(), Some(4));
}
