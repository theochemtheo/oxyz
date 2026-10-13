//! Property tests: no input makes the parser or the scans panic or
//! over-allocate.

use std::io::Cursor;

use oxyz_core::model::{Column, ColumnData, Frame};
use oxyz_core::project::{Fill, PlanColumn, ProjectionPlan, project_frame};
use oxyz_core::{BatchBuilder, BatchError, FrameIter, scan_frames, scan_frames_with_volume};
use proptest::prelude::*;

fn parse_all(input: &str) {
    for result in FrameIter::new(Cursor::new(input.as_bytes())) {
        let _ = result;
    }
}

fn scan_all(input: &str) {
    let _ = scan_frames(Cursor::new(input.as_bytes()));
    let _ = scan_frames_with_volume(Cursor::new(input.as_bytes()));
}

proptest! {
    #[test]
    fn never_panics_on_arbitrary_input(input in ".*") {
        parse_all(&input);
        scan_all(&input);
    }

    /// The volume scan agrees with the plain scan on structure, and fails alike.
    #[test]
    fn volume_scan_entries_match_plain_scan(input in ".*") {
        let plain = scan_frames(Cursor::new(input.as_bytes()));
        let with_volume = scan_frames_with_volume(Cursor::new(input.as_bytes()));
        match (plain, with_volume) {
            (Ok(plain), Ok(with_volume)) => {
                prop_assert_eq!(plain.entries(), with_volume.entries());
                prop_assert_eq!(
                    with_volume.volumes().map(<[f64]>::len),
                    Some(with_volume.n_frames())
                );
            }
            (Err(_), Err(_)) => {}
            (plain, with_volume) => prop_assert!(
                false,
                "scans disagreed: plain ok={}, with_volume ok={}",
                plain.is_ok(),
                with_volume.is_ok()
            ),
        }
    }

    /// A huge declared atom count, up to usize::MAX, neither panics nor
    /// pre-allocates.
    #[test]
    fn never_panics_on_declared_atom_counts(count in any::<u64>(), body in "[ -~\n]{0,200}") {
        let input = format!("{count}\nProperties=species:S:1:pos:R:3\n{body}");
        parse_all(&input);
        scan_all(&input);
    }

    /// Arbitrary Properties descriptors, including huge declared widths.
    #[test]
    fn never_panics_on_arbitrary_descriptors(
        descriptor in "[A-Za-z0-9:._\\-]{0,64}",
        count in 0usize..4,
    ) {
        let mut input = format!("{count}\nProperties={descriptor}\n");
        for _ in 0..count {
            input.push_str("H 0.0 0.0 0.0\n");
        }
        parse_all(&input);
        scan_all(&input);
    }
}

proptest! {
    // Projection yields exactly the plan's shape: n_atoms * width cells per column.
    #[test]
    fn projection_yields_plan_shape_without_panic(
        n_atoms in 0usize..8,
        widths in proptest::collection::vec(1usize..4, 0..5),
    ) {
        let plan = ProjectionPlan {
            columns: widths
                .iter()
                .enumerate()
                .map(|(i, &w)| PlanColumn {
                    name: format!("c{i}"),
                    kind: oxyz_core::model::ColumnKind::Real,
                    width: w,
                    required: false,
                    fill: Some(Fill::Real(0.0)),
                })
                .collect(),
            metadata: Vec::new(),
        };
        // An empty frame forces every column to be filled to the plan shape.
        let frame = oxyz_core::model::Frame { n_atoms, columns: Vec::new(), metadata: Vec::new() };
        let projected = project_frame(&frame, &plan);
        prop_assert!(!projected.dropped); // all optional with fills
        prop_assert_eq!(projected.frame.columns.len(), widths.len());
        for (col, &w) in projected.frame.columns.iter().zip(&widths) {
            prop_assert_eq!(col.width, w);
            prop_assert_eq!(col.data.len(), n_atoms * w);
        }
    }
}

proptest! {
    // Any input columns project to the plan's shape. Every plan column is
    // optional and fillable, so no frame drops.
    #[test]
    fn projection_arbitrary_frame_never_panics(
        n_atoms in 0usize..6,
        plan_widths in proptest::collection::vec(1usize..4, 0..4),
        frame_cols in proptest::collection::vec((0usize..6, 0usize..4, 1usize..4), 0..6),
    ) {
        // Column, ColumnData and Frame come from the module-level import.
        use oxyz_core::model::ColumnKind;

        let plan = ProjectionPlan {
            columns: plan_widths
                .iter()
                .enumerate()
                .map(|(i, &w)| PlanColumn {
                    name: format!("c{i}"),
                    kind: ColumnKind::Real,
                    width: w,
                    required: false,
                    fill: Some(Fill::Real(0.0)),
                })
                .collect(),
            metadata: Vec::new(),
        };
        let columns = frame_cols
            .iter()
            .map(|&(name_i, kind_i, w)| {
                let count = n_atoms * w;
                let data = match kind_i {
                    0 => ColumnData::Real(vec![1.0; count]),
                    1 => ColumnData::Int(vec![1; count]),
                    2 => ColumnData::Bool(vec![true; count]),
                    _ => ColumnData::Str(vec!["x".into(); count]),
                };
                Column { name: format!("c{name_i}").into(), width: w, data }
            })
            .collect();
        let frame = Frame { n_atoms, columns, metadata: Vec::new() };

        let projected = project_frame(&frame, &plan);
        prop_assert!(!projected.dropped);
        prop_assert_eq!(projected.frame.columns.len(), plan_widths.len());
        for (col, &w) in projected.frame.columns.iter().zip(&plan_widths) {
            prop_assert_eq!(col.width, w);
            prop_assert_eq!(col.data.len(), n_atoms * w);
        }
    }
}

/// Well-formed key=value pairs with one corrupted token spliced in.
fn corrupt_comment_line() -> impl Strategy<Value = String> {
    let good_pair = prop_oneof![
        Just("energy=-1.0".to_string()),
        Just("name=hello".to_string()),
        Just("arr=\"1 2 3\"".to_string()),
        Just("tag=[a,b,c]".to_string()),
    ];
    let corruption = prop_oneof![
        Just("bad=\"unterminated".to_string()), // dangling quote
        Just("arr={1 2".to_string()),           // unbalanced brace
        Just("k=".to_string()),                 // value-less
        Just("=v".to_string()),                 // key-less
        Just("[,2,3]".to_string()),             // bracket array with a leading stray separator
        Just("lone".to_string()),               // bare token, no '='
    ];
    (proptest::collection::vec(good_pair, 0..4), corruption).prop_map(|(pairs, bad)| {
        let mut parts = pairs;
        parts.push(bad);
        format!("Properties=species:S:1:pos:R:3 {}", parts.join(" "))
    })
}

proptest! {
    /// Corrupt comment lines never panic, and errors name their frame.
    #[test]
    fn corrupt_comment_lines_never_panic(comment in corrupt_comment_line()) {
        let input = format!("1\n{comment}\nH 0.0 0.0 0.0\n");
        for result in FrameIter::new(Cursor::new(input.as_bytes())) {
            if let Err(error) = result {
                prop_assert_eq!(error.frame_index(), Some(0));
            }
        }
        // Scans read only structural lines but must not panic either.
        let _ = scan_frames(Cursor::new(input.as_bytes()));
        let _ = scan_frames_with_volume(Cursor::new(input.as_bytes()));
    }
}

/// Properties triples with injected malformations.
fn malformed_descriptor() -> impl Strategy<Value = String> {
    let triple = prop_oneof![
        Just("species:S:1".to_string()),
        Just("pos:R:3".to_string()),
        Just("bad:Q:1".to_string()), // unknown kind
        Just("w:R:x".to_string()),   // non-numeric width
        Just("miss:R".to_string()),  // missing width field
        Just("z::1".to_string()),    // empty kind
    ];
    proptest::collection::vec(triple, 1..5).prop_map(|ts| ts.join(":"))
}

proptest! {
    /// Malformed descriptors and ragged rows are errors naming frame 0. Row width
    /// is independent of the declared widths.
    #[test]
    fn malformed_descriptors_and_ragged_rows_never_panic(
        descriptor in malformed_descriptor(),
        row_cols in 0usize..8,
        n_atoms in 0usize..4,
    ) {
        let mut input = format!("{n_atoms}\nProperties={descriptor}\n");
        for _ in 0..n_atoms {
            let row = std::iter::repeat_n("1.0", row_cols).collect::<Vec<_>>().join(" ");
            input.push_str(&row);
            input.push('\n');
        }
        for result in FrameIter::new(Cursor::new(input.as_bytes())) {
            if let Err(error) = result {
                prop_assert_eq!(error.frame_index(), Some(0));
                // Line 2 is the comment; atom rows start at 3.
                if let Some(line) = error.line() {
                    prop_assert!(line >= 2);
                }
            }
        }
    }
}

/// One frame's schema: which of {a,b,c} columns it has and each one's kind.
fn frame_schema() -> impl Strategy<Value = Vec<(char, u8)>> {
    proptest::collection::vec((prop::sample::select(vec!['a', 'b', 'c']), 0u8..3), 0..3)
}

fn build_frame(n_atoms: usize, schema: &[(char, u8)]) -> Frame {
    let columns = schema
        .iter()
        .map(|&(name, kind)| {
            let data = match kind {
                0 => ColumnData::Real(vec![1.0; n_atoms]),
                1 => ColumnData::Int(vec![1; n_atoms]),
                _ => ColumnData::Bool(vec![true; n_atoms]),
            };
            Column {
                name: name.to_string().into(),
                width: 1,
                data,
            }
        })
        .collect();
    Frame {
        n_atoms,
        columns,
        metadata: Vec::new(),
    }
}

proptest! {
    /// Pushing a frame with a different schema names that frame, never panics.
    #[test]
    fn ragged_batch_push_errors_name_the_frame(
        first in frame_schema(),
        rest in proptest::collection::vec(frame_schema(), 0..4),
        n_atoms in 1usize..3,
    ) {
        let mut builder = BatchBuilder::new();
        // The first push defines the contract; it must succeed.
        prop_assume!(builder.push(build_frame(n_atoms, &first)).is_ok());

        for (i, schema) in rest.iter().enumerate() {
            match builder.push(build_frame(n_atoms, schema)) {
                Ok(()) => {}
                Err(error) => {
                    let named = match error {
                        BatchError::ColumnMismatch { frame, .. }
                        | BatchError::MissingColumn { frame, .. }
                        | BatchError::UnexpectedColumn { frame, .. }
                        | BatchError::MetadataMismatch { frame, .. }
                        | BatchError::MissingMetadata { frame, .. }
                        | BatchError::UnexpectedMetadata { frame, .. } => Some(frame),
                        _ => None,
                    };
                    prop_assert_eq!(named, Some(i + 1));
                    break; // builder state is undefined after a rejected push
                }
            }
        }
    }
}
