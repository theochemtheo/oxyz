use oxyz_core::model::{Column, ColumnData, ColumnKind, Frame, Value};
use oxyz_core::project::{
    Axis, DeviationKind, Fill, PlanColumn, PlanMetadata, Projected, ProjectionPlan,
};
use oxyz_core::{read_all_batch_projected_from, read_batch_projected_from};

fn real_col_plan(name: &str, width: usize, required: bool) -> PlanColumn {
    PlanColumn {
        name: name.into(),
        kind: ColumnKind::Real,
        width,
        required,
        fill: Some(Fill::Real(f64::NAN)),
    }
}

fn col(name: &str, data: ColumnData, width: usize) -> Column {
    Column {
        name: name.into(),
        width,
        data,
    }
}

fn frame(n_atoms: usize, columns: Vec<Column>) -> Frame {
    Frame {
        n_atoms,
        columns,
        metadata: Vec::new(),
    }
}

fn real_plan(name: &str, width: usize, required: bool, fill: Option<f64>) -> PlanColumn {
    PlanColumn {
        name: name.into(),
        kind: ColumnKind::Real,
        width,
        required,
        fill: fill.map(Fill::Real),
    }
}

fn frame_meta(n_atoms: usize, metadata: Vec<(String, Value)>) -> Frame {
    Frame {
        n_atoms,
        columns: Vec::new(),
        metadata,
    }
}

fn real_meta(name: &str, shape: Option<usize>, required: bool, fill: Option<f64>) -> PlanMetadata {
    PlanMetadata {
        name: name.into(),
        kind: ColumnKind::Real,
        shape,
        required,
        fill: fill.map(Fill::Real),
    }
}

#[test]
fn plan_types_are_constructible() {
    let plan = ProjectionPlan {
        columns: vec![PlanColumn {
            name: "pos".into(),
            kind: ColumnKind::Real,
            width: 3,
            required: true,
            fill: Some(Fill::Real(f64::NAN)),
        }],
        metadata: Vec::new(),
    };
    assert_eq!(plan.columns.len(), 1);
    assert_eq!(plan.columns[0].kind, ColumnKind::Real);
    assert!(matches!(plan.columns[0].fill, Some(Fill::Real(_))));
    // The Axis enum is part of the reported contract.
    assert_ne!(Axis::Column, Axis::Metadata);
}

#[test]
fn keeps_declared_drops_undeclared_in_plan_order() {
    let f = frame(
        2,
        vec![
            col("charge", ColumnData::Real(vec![0.0, 1.0]), 1),
            col(
                "pos",
                ColumnData::Real(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
                3,
            ),
        ],
    );
    let plan = ProjectionPlan {
        columns: vec![real_plan("pos", 3, true, Some(f64::NAN))],
        metadata: Vec::new(),
    };
    let Projected {
        frame,
        deviations,
        dropped,
    } = oxyz_core::project::project_frame(&f, &plan);
    assert!(!dropped);
    assert!(deviations.is_empty());
    assert_eq!(frame.columns.len(), 1);
    assert_eq!(frame.columns[0].name, "pos"); // charge dropped
}

#[test]
fn fills_absent_optional_real_silently() {
    let f = frame(2, vec![col("pos", ColumnData::Real(vec![0.0; 6]), 3)]);
    let plan = ProjectionPlan {
        columns: vec![
            real_plan("pos", 3, true, Some(f64::NAN)),
            real_plan("charge", 1, false, Some(f64::NAN)),
        ],
        metadata: Vec::new(),
    };
    let Projected {
        frame,
        deviations,
        dropped,
    } = oxyz_core::project::project_frame(&f, &plan);
    assert!(!dropped);
    assert!(deviations.is_empty()); // optional fill is silent
    assert_eq!(frame.columns[1].name, "charge");
    assert_eq!(frame.columns[1].width, 1);
    let filled = frame.columns[1].data.as_real().unwrap();
    assert_eq!(filled.len(), 2);
    assert!(filled.iter().all(|x| x.is_nan()));
}

#[test]
fn absent_required_without_fill_drops_frame_with_missing() {
    let f = frame(2, vec![col("pos", ColumnData::Real(vec![0.0; 6]), 3)]);
    let plan = ProjectionPlan {
        columns: vec![PlanColumn {
            name: "id".into(),
            kind: ColumnKind::Int,
            width: 1,
            required: true,
            fill: None, // integer has no natural null
        }],
        metadata: Vec::new(),
    };
    let Projected {
        deviations,
        dropped,
        ..
    } = oxyz_core::project::project_frame(&f, &plan);
    assert!(dropped);
    assert_eq!(deviations.len(), 1);
    assert_eq!(deviations[0].kind, DeviationKind::Missing);
    assert_eq!(deviations[0].name, "id");
    assert_eq!(deviations[0].expected, "I:1");
}

#[test]
fn wrong_kind_reports_mismatch_and_fills_when_possible() {
    let f = frame(1, vec![col("val", ColumnData::Int(vec![7]), 1)]);
    let plan = ProjectionPlan {
        columns: vec![real_plan("val", 1, true, Some(f64::NAN))],
        metadata: Vec::new(),
    };
    let Projected {
        frame,
        deviations,
        dropped,
    } = oxyz_core::project::project_frame(&f, &plan);
    assert!(!dropped);
    assert_eq!(deviations.len(), 1);
    assert_eq!(deviations[0].kind, DeviationKind::Mismatch);
    assert_eq!(deviations[0].expected, "R:1");
    assert_eq!(deviations[0].found.as_deref(), Some("I:1"));
    assert!(frame.columns[0].data.as_real().unwrap()[0].is_nan());
}

#[test]
fn projects_metadata_keep_drop_fill_in_order() {
    let f = frame_meta(
        1,
        vec![
            ("junk".into(), Value::Str("x".into())),
            ("energy".into(), Value::Real(-1.0)),
        ],
    );
    let plan = ProjectionPlan {
        columns: Vec::new(),
        metadata: vec![
            real_meta("energy", None, true, Some(f64::NAN)),
            real_meta("weight", None, false, Some(1.0)),
        ],
    };
    let Projected {
        frame,
        deviations,
        dropped,
    } = oxyz_core::project::project_frame(&f, &plan);
    assert!(!dropped);
    assert!(deviations.is_empty());
    assert_eq!(frame.metadata.len(), 2); // junk dropped
    assert_eq!(frame.metadata[0].0, "energy");
    assert_eq!(frame.metadata[1], ("weight".into(), Value::Real(1.0)));
}

#[test]
fn metadata_array_mismatch_reports_bracket_signature() {
    let f = frame_meta(1, vec![("stress".into(), Value::RealArray(vec![0.0; 6]))]);
    let plan = ProjectionPlan {
        columns: Vec::new(),
        metadata: vec![real_meta("stress", Some(9), true, Some(f64::NAN))],
    };
    let Projected {
        deviations,
        dropped,
        frame,
    } = oxyz_core::project::project_frame(&f, &plan);
    assert!(!dropped);
    assert_eq!(deviations[0].kind, DeviationKind::Mismatch);
    assert_eq!(deviations[0].expected, "R[9]");
    assert_eq!(deviations[0].found.as_deref(), Some("R[6]"));
    // Filled to the declared 9-length array (NaN != NaN, so check componentwise
    // rather than assert_eq! on the value).
    match &frame.metadata[0].1 {
        Value::RealArray(v) => {
            assert_eq!(v.len(), 9);
            assert!(v.iter().all(|x| x.is_nan()));
        }
        other => panic!("expected a 9-length RealArray, got {other:?}"),
    }
}

#[test]
fn projected_whole_file_batch_fills_and_reports() {
    let input = "1\nProperties=species:S:1:pos:R:3:charge:R:1\nH 0 0 0 0.5\n\
                 1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n";
    let plan = ProjectionPlan {
        columns: vec![
            real_col_plan("pos", 3, true),
            real_col_plan("charge", 1, false),
        ],
        metadata: Vec::new(),
    };
    let pb = read_all_batch_projected_from(input.as_bytes(), &plan).unwrap();
    assert_eq!(pb.survivors, vec![0, 1]); // species dropped, both frames kept
    assert_eq!(pb.batch.n_frames(), 2);
    let charge = pb
        .batch
        .columns
        .iter()
        .find(|c| c.name == "charge")
        .unwrap();
    let vals = charge.data.as_real().unwrap();
    assert_eq!(vals.len(), 2);
    assert_eq!(vals[0], 0.5);
    assert!(vals[1].is_nan()); // second frame's charge filled
    assert_eq!(pb.reports.len(), 2);
    assert!(pb.reports.iter().all(|(_, d)| d.is_empty())); // optional fill is silent
}

#[test]
fn projected_batch_drops_unfillable_and_records_survivors() {
    let input = "1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n\
                 1\nProperties=species:S:1:pos:R:3:id:I:1\nH 0 0 0 7\n";
    let plan = ProjectionPlan {
        columns: vec![
            real_col_plan("pos", 3, true),
            PlanColumn {
                name: "id".into(),
                kind: ColumnKind::Int,
                width: 1,
                required: true,
                fill: None, // no natural null -> an absent id drops the frame
            },
        ],
        metadata: Vec::new(),
    };
    let pb = read_all_batch_projected_from(input.as_bytes(), &plan).unwrap();
    assert_eq!(pb.survivors, vec![1]); // frame 0 dropped (no id), frame 1 kept
    assert_eq!(pb.batch.n_frames(), 1);
    assert_eq!(pb.reports.len(), 2);
    assert_eq!(pb.reports[0].1.len(), 1); // frame 0 reports a missing id
    assert!(pb.reports[1].1.is_empty());
}

#[test]
fn projected_selection_preserves_request_order() {
    let input = "1\nProperties=species:S:1:pos:R:3:e:R:1\nH 0 0 0 1.0\n\
                 1\nProperties=species:S:1:pos:R:3:e:R:1\nH 0 0 0 2.0\n\
                 1\nProperties=species:S:1:pos:R:3:e:R:1\nH 0 0 0 3.0\n";
    let plan = ProjectionPlan {
        columns: vec![real_col_plan("pos", 3, true), real_col_plan("e", 1, true)],
        metadata: Vec::new(),
    };
    let pb = read_batch_projected_from(input.as_bytes(), &[2, 0], &plan).unwrap();
    assert_eq!(pb.survivors, vec![2, 0]); // request order, not file order
    let e = pb.batch.columns.iter().find(|c| c.name == "e").unwrap();
    assert_eq!(e.data.as_real().unwrap(), &[3.0, 1.0]);
}

#[test]
fn duplicate_metadata_prefers_conforming_occurrence() {
    // `energy` appears twice: a wrong-kind Str, then a conforming Real. Projection
    // must pick the conforming one, not report a spurious mismatch on the first.
    let f = frame_meta(
        1,
        vec![
            ("energy".into(), Value::Str("oops".into())),
            ("energy".into(), Value::Real(-1.0)),
        ],
    );
    let plan = ProjectionPlan {
        columns: Vec::new(),
        metadata: vec![real_meta("energy", None, true, Some(f64::NAN))],
    };
    let Projected {
        frame,
        deviations,
        dropped,
    } = oxyz_core::project::project_frame(&f, &plan);
    assert!(!dropped);
    assert!(deviations.is_empty());
    assert_eq!(frame.metadata[0], ("energy".into(), Value::Real(-1.0)));
}
