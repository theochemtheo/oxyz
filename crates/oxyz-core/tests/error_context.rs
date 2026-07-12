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
