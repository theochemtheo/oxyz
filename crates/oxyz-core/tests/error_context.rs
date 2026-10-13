//! Every parse error reports its frame, file-absolute line, and, where a
//! token is at fault, its 1-based character column.

use oxyz_core::ExtxyzError;

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

#[test]
fn bad_atom_value_locates_line_and_column() {
    // Line 1: count, line 2: comment, line 3: atom row. "abc" starts at col 3.
    let input = "1\nProperties=species:S:1:pos:R:3\nH abc 0.0 0.0\n";
    let error = first_error(input);
    assert_eq!(error.frame_index(), Some(0));
    assert_eq!(error.line(), Some(3));
    assert_eq!(error.column(), Some(3));
}

/// A bad value in a later cell of a multi-width column reports that cell.
#[test]
fn bad_atom_value_in_non_first_cell_locates_that_cell() {
    // "abc" is the second `pos` cell, at characters 7-9.
    let input = "1\nProperties=species:S:1:pos:R:3\nH 0.0 abc 0.0\n";
    let error = first_error(input);
    assert_eq!(error.frame_index(), Some(0));
    assert_eq!(error.line(), Some(3));
    assert_eq!(error.column(), Some(7));
}

#[test]
fn bad_metadata_locates_comment_line_and_column() {
    let input = "1\nProperties=species:S:1:pos:R:3 bad=\nH 0.0 0.0 0.0\n";
    let error = first_error(input);
    assert_eq!(error.frame_index(), Some(0));
    assert_eq!(error.line(), Some(2));
    assert!(error.column().is_some());
}

/// A malformed value is located within the comment line, not at the value's
/// own offset.
#[test]
fn bad_array_value_locates_comment_line_and_column_at_the_value() {
    // "key=[1,2,]" starts at character 34; the trailing comma is the error.
    let input = "1\nProperties=species:S:1:pos:R:3 key=[1,2,]\nH 0.0 0.0 0.0\n";
    let error = first_error(input);
    let comment = "Properties=species:S:1:pos:R:3 key=[1,2,]";
    let value_start = comment.find("[1,2,]").expect("value present") + 1; // 1-based
    assert_eq!(error.frame_index(), Some(0));
    assert_eq!(error.line(), Some(2));
    let column = error.column().expect("array-value error carries a column");
    assert!(
        column >= value_start,
        "column {column} should land at/after the value's start {value_start}"
    );
}

/// A short atom row has no single token to blame, so only the line is given.
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

/// The intra-frame parallel path locates a bad value as the serial loop does.
#[cfg(feature = "parallel")]
#[test]
fn parallel_bad_atom_value_matches_serial_location() {
    use oxyz_core::read_frames_parallel_from;
    // Over 256 KB, so the intra-frame parallel path runs.
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

/// ...including a bad value in a later cell of a multi-width column.
#[cfg(feature = "parallel")]
#[test]
fn parallel_bad_atom_value_in_non_first_cell_matches_serial_location() {
    use oxyz_core::read_frames_parallel_from;
    // Over 256 KB, so the intra-frame parallel path runs.
    let mut input = String::from("20000\nProperties=species:S:1:pos:R:3\n");
    for _ in 0..9999 {
        input.push_str("H 0.0 0.0 0.0\n");
    }
    input.push_str("H 0.0 bad 0.0\n"); // line 10002 (1 count + 1 comment + 10000th row)
    for _ in 0..10000 {
        input.push_str("H 0.0 0.0 0.0\n");
    }
    let bytes = input.into_bytes();

    let parallel_error = read_frames_parallel_from(Cursor::new(bytes.clone()), Some(4))
        .expect_err("expected a parse error");
    let serial_error = iter_frames_from(Cursor::new(bytes))
        .unwrap()
        .find_map(Result::err)
        .expect("expected a parse error");

    // "H 0.0 bad 0.0": H=1, ' '=2, 0.0=3-5, ' '=6, bad=7-9.
    assert_eq!(parallel_error.frame_index(), Some(0));
    assert_eq!(parallel_error.line(), Some(10002));
    assert_eq!(parallel_error.column(), Some(7));
    assert_eq!(parallel_error.column(), serial_error.column());
    assert_eq!(parallel_error.line(), serial_error.line());
}

/// The intra-frame parallel path locates a bad comment line as the serial
/// reader does.
#[cfg(feature = "parallel")]
#[test]
fn parallel_bad_comment_line_matches_serial_location() {
    use oxyz_core::read_frames_parallel_from;
    // Over 256 KB, so the intra-frame parallel path runs.
    let mut input = String::from("20000\nProperties=species:Q:1:pos:R:3\n");
    for _ in 0..20000 {
        input.push_str("H 0.0 0.0 0.0\n");
    }
    let bytes = input.into_bytes();

    let parallel_error = read_frames_parallel_from(Cursor::new(bytes.clone()), Some(4))
        .expect_err("expected a parse error");
    let serial_error = iter_frames_from(Cursor::new(bytes))
        .unwrap()
        .find_map(Result::err)
        .expect("expected a parse error");

    assert_eq!(parallel_error.frame_index(), Some(0));
    assert_eq!(parallel_error.line(), Some(2));
    assert_eq!(parallel_error.frame_index(), serial_error.frame_index());
    assert_eq!(parallel_error.line(), serial_error.line());
    assert_eq!(parallel_error.column(), serial_error.column());
    assert_eq!(
        parallel_error.to_string(),
        serial_error.to_string(),
        "parallel: {parallel_error}\nserial:   {serial_error}"
    );
}

use oxyz_core::read_batch_from;

/// A truncated frame read through `read_batch_from` reports the missing
/// line's file-absolute number.
#[test]
fn batch_truncated_frame_locates_exact_line() {
    // Frame 1 declares 2 atoms but has 1.
    let input = "1\ncomment\nH 0.0 0.0 0.0\n2\ncomment\nH 0.0 0.0 0.0\n";
    let error =
        read_batch_from(Cursor::new(input.as_bytes()), &[0, 1]).expect_err("truncated frame");
    assert_eq!(error.line(), Some(7));
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

/// Wording names the expected thing, not just the wrong one.
#[test]
fn messages_are_actionable() {
    // Unknown Properties kind.
    let error = first_error("1\nProperties=species:Q:1:pos:R:3\nH 0.0 0.0 0.0\n");
    let message = error.to_string();
    assert!(message.contains("Properties"), "{message}");
    assert!(message.contains('Q'), "{message}");

    // Missing Properties descriptor entirely.
    let error = first_error("1\nenergy=-1.0\nH 0.0 0.0 0.0\n");
    assert!(error.to_string().contains("Properties"), "{error}");
}
