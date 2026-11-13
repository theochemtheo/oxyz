use compact_str::CompactString;
use oxyz_core::model::{Column, ColumnData, ColumnKind, Frame, Value};
use oxyz_core::project::{
    Axis, DeviationKind, Fill, PlanColumn, PlanMetadata, Projected, ProjectionPlan, project_frame,
};
use oxyz_core::{
    BatchError, ExtxyzError, iter_batches_projected_from, read_all_batch_projected_from,
    read_all_batch_projected_parallel_from, read_batch_projected_from,
    read_batch_projected_parallel_from,
};

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

fn frame_meta(n_atoms: usize, metadata: Vec<(CompactString, Value)>) -> Frame {
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
        column_major: false,
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
        ..Default::default()
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
        ..Default::default()
    };
    let Projected {
        frame,
        deviations,
        dropped,
        ..
    } = project_frame(&f, &plan);
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
        ..Default::default()
    };
    let Projected {
        frame,
        deviations,
        dropped,
        ..
    } = project_frame(&f, &plan);
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
        ..Default::default()
    };
    let Projected {
        deviations,
        dropped,
        ..
    } = project_frame(&f, &plan);
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
        ..Default::default()
    };
    let Projected {
        frame,
        deviations,
        dropped,
        ..
    } = project_frame(&f, &plan);
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
        ..Default::default()
    };
    let Projected {
        frame,
        deviations,
        dropped,
        ..
    } = project_frame(&f, &plan);
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
        ..Default::default()
    };
    let Projected {
        deviations,
        dropped,
        frame,
        ..
    } = project_frame(&f, &plan);
    assert!(!dropped);
    assert_eq!(deviations[0].kind, DeviationKind::Mismatch);
    assert_eq!(deviations[0].expected, "R[9]");
    assert_eq!(deviations[0].found.as_deref(), Some("R[6]"));
    // NaN != NaN, so check componentwise.
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
        ..Default::default()
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
        ..Default::default()
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
        ..Default::default()
    };
    let pb = read_batch_projected_from(input.as_bytes(), &[2, 0], &plan).unwrap();
    assert_eq!(pb.survivors, vec![2, 0]); // request order, not file order
    let e = pb.batch.columns.iter().find(|c| c.name == "e").unwrap();
    assert_eq!(e.data.as_real().unwrap(), &[3.0, 1.0]);
}

#[test]
fn duplicate_metadata_prefers_conforming_occurrence() {
    // A wrong-kind duplicate must not hide the conforming one.
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
        ..Default::default()
    };
    let Projected {
        frame,
        deviations,
        dropped,
        ..
    } = project_frame(&f, &plan);
    assert!(!dropped);
    assert!(deviations.is_empty());
    assert_eq!(frame.metadata[0], ("energy".into(), Value::Real(-1.0)));
}

#[test]
fn materialises_non_real_fills() {
    // Absent optional Int/Bool/Str columns fill with their declared sentinel.
    let f = frame(2, Vec::new());
    let plan = ProjectionPlan {
        columns: vec![
            PlanColumn {
                name: "id".into(),
                kind: ColumnKind::Int,
                width: 1,
                required: false,
                fill: Some(Fill::Int(-1)),
            },
            PlanColumn {
                name: "ok".into(),
                kind: ColumnKind::Bool,
                width: 1,
                required: false,
                fill: Some(Fill::Bool(true)),
            },
            PlanColumn {
                name: "tag".into(),
                kind: ColumnKind::Str,
                width: 1,
                required: false,
                fill: Some(Fill::Str("none".into())),
            },
        ],
        metadata: Vec::new(),
        ..Default::default()
    };
    let p = project_frame(&f, &plan);
    assert!(!p.dropped);
    assert_eq!(p.frame.columns[0].data.as_int().unwrap(), &[-1, -1]);
    assert_eq!(p.frame.columns[1].data.as_bool().unwrap(), &[true, true]);
    assert_eq!(
        p.frame.columns[2].data.as_str().unwrap(),
        &[CompactString::from("none"), CompactString::from("none")]
    );
}

fn plain_meta(name: &str, kind: ColumnKind, shape: Option<usize>, fill: Fill) -> PlanMetadata {
    PlanMetadata {
        name: name.into(),
        kind,
        shape,
        required: false,
        fill: Some(fill),
        column_major: false,
    }
}

#[test]
fn keeps_conforming_metadata_of_every_kind_and_shape() {
    let f = frame_meta(
        1,
        vec![
            ("i".into(), Value::Int(3)),
            ("b".into(), Value::Bool(true)),
            ("ia".into(), Value::IntArray(vec![1, 2])),
            ("ba".into(), Value::BoolArray(vec![false, true])),
            ("sa".into(), Value::StrArray(vec!["x".into(), "y".into()])),
        ],
    );
    let plan = ProjectionPlan {
        columns: Vec::new(),
        metadata: vec![
            plain_meta("i", ColumnKind::Int, None, Fill::Int(0)),
            plain_meta("b", ColumnKind::Bool, None, Fill::Bool(false)),
            plain_meta("ia", ColumnKind::Int, Some(2), Fill::Int(0)),
            plain_meta("ba", ColumnKind::Bool, Some(2), Fill::Bool(false)),
            plain_meta("sa", ColumnKind::Str, Some(2), Fill::Str(String::new())),
        ],
        ..Default::default()
    };
    let p = project_frame(&f, &plan);
    assert!(!p.dropped);
    assert!(p.deviations.is_empty());
    assert_eq!(p.frame.metadata[0].1, Value::Int(3));
    assert_eq!(p.frame.metadata[3].1, Value::BoolArray(vec![false, true]));
    assert_eq!(
        p.frame.metadata[4].1,
        Value::StrArray(vec!["x".into(), "y".into()])
    );
}

#[test]
fn materialises_absent_optional_metadata_of_every_kind() {
    let f = frame_meta(1, Vec::new());
    let plan = ProjectionPlan {
        columns: Vec::new(),
        metadata: vec![
            plain_meta("i", ColumnKind::Int, None, Fill::Int(-1)),
            plain_meta("b", ColumnKind::Bool, None, Fill::Bool(true)),
            plain_meta("s", ColumnKind::Str, None, Fill::Str("none".into())),
            plain_meta("ia", ColumnKind::Int, Some(2), Fill::Int(-1)),
            plain_meta("ba", ColumnKind::Bool, Some(2), Fill::Bool(true)),
            plain_meta("sa", ColumnKind::Str, Some(2), Fill::Str("none".into())),
        ],
        ..Default::default()
    };
    let p = project_frame(&f, &plan);
    assert!(!p.dropped);
    assert!(p.deviations.is_empty()); // absent optional is silent
    assert_eq!(p.frame.metadata[0].1, Value::Int(-1));
    assert_eq!(p.frame.metadata[1].1, Value::Bool(true));
    assert_eq!(p.frame.metadata[2].1, Value::Str("none".into()));
    assert_eq!(p.frame.metadata[3].1, Value::IntArray(vec![-1, -1]));
    assert_eq!(p.frame.metadata[4].1, Value::BoolArray(vec![true, true]));
    assert_eq!(
        p.frame.metadata[5].1,
        Value::StrArray(vec!["none".into(), "none".into()])
    );
}

#[test]
fn required_metadata_absent_without_fill_drops_frame() {
    // A required key with no fill drops the frame.
    let f = frame_meta(1, Vec::new());
    let plan = ProjectionPlan {
        columns: Vec::new(),
        metadata: vec![PlanMetadata {
            name: "energy".into(),
            kind: ColumnKind::Int,
            shape: None,
            required: true,
            fill: None,
            column_major: false,
        }],
        ..Default::default()
    };
    let p = project_frame(&f, &plan);
    assert!(p.dropped);
    assert_eq!(p.deviations.len(), 1);
    assert_eq!(p.deviations[0].axis, Axis::Metadata);
    assert_eq!(p.deviations[0].kind, DeviationKind::Missing);
    assert_eq!(p.deviations[0].name, "energy");
}

#[test]
fn kind_mismatched_fill_falls_back_instead_of_panicking() {
    // A fill kind disagreeing with the field kind falls back to the declared
    // kind's zero rather than panicking. An empty frame fills every field.
    let f = frame(2, Vec::new());
    let mismatched = |name: &str, kind: ColumnKind| PlanColumn {
        name: name.into(),
        kind,
        width: 1,
        required: false,
        fill: Some(Fill::Bool(true)), // deliberately wrong for R/I/S
    };
    let plan = ProjectionPlan {
        columns: vec![
            mismatched("r", ColumnKind::Real),
            mismatched("i", ColumnKind::Int),
            mismatched("s", ColumnKind::Str),
            // Bool with a non-Bool fill exercises the remaining column arm.
            PlanColumn {
                name: "b".into(),
                kind: ColumnKind::Bool,
                width: 1,
                required: false,
                fill: Some(Fill::Int(9)),
            },
        ],
        metadata: vec![
            plain_meta("mr", ColumnKind::Real, None, Fill::Int(1)),
            plain_meta("mi", ColumnKind::Int, None, Fill::Bool(true)),
            plain_meta("mb", ColumnKind::Bool, None, Fill::Int(1)),
            plain_meta("ms", ColumnKind::Str, None, Fill::Int(1)),
            plain_meta("mra", ColumnKind::Real, Some(1), Fill::Int(1)),
            plain_meta("mia", ColumnKind::Int, Some(1), Fill::Bool(true)),
            plain_meta("mba", ColumnKind::Bool, Some(1), Fill::Int(1)),
            plain_meta("msa", ColumnKind::Str, Some(1), Fill::Int(1)),
        ],
        ..Default::default()
    };
    let p = project_frame(&f, &plan);
    assert!(!p.dropped);
    assert!(
        p.frame.columns[0]
            .data
            .as_real()
            .unwrap()
            .iter()
            .all(|x| x.is_nan())
    );
    assert_eq!(p.frame.columns[1].data.as_int().unwrap(), &[0, 0]);
    assert_eq!(
        p.frame.columns[2].data.as_str().unwrap(),
        &[CompactString::from(""), CompactString::from("")]
    );
    assert_eq!(p.frame.columns[3].data.as_bool().unwrap(), &[false, false]);
    assert_eq!(p.frame.metadata[1].1, Value::Int(0));
    assert_eq!(p.frame.metadata[2].1, Value::Bool(false));
    assert_eq!(p.frame.metadata[3].1, Value::Str("".into()));
    assert_eq!(p.frame.metadata[5].1, Value::IntArray(vec![0]));
    assert_eq!(p.frame.metadata[6].1, Value::BoolArray(vec![false]));
    assert_eq!(p.frame.metadata[7].1, Value::StrArray(vec!["".into()]));
}

#[test]
fn optional_without_fill_is_omitted_when_absent() {
    let plan = ProjectionPlan {
        columns: vec![
            real_col_plan("pos", 3, true),
            real_plan("charge", 1, false, None),
        ],
        metadata: vec![real_meta("energy", None, false, None)],
        ..Default::default()
    };
    let f = frame(1, vec![col("pos", ColumnData::Real(vec![0.0; 3]), 3)]);
    let p = project_frame(&f, &plan);
    assert!(!p.dropped && p.deviations.is_empty());
    assert_eq!(p.frame.columns.len(), 1);
    assert!(p.frame.metadata.is_empty());
}

#[test]
fn optional_column_without_fill_drops_on_wrong_kind() {
    let plan = ProjectionPlan {
        columns: vec![real_plan("charge", 1, false, None)],
        metadata: Vec::new(),
        ..Default::default()
    };
    let f = frame(1, vec![col("charge", ColumnData::Int(vec![1]), 1)]);
    let p = project_frame(&f, &plan);
    assert!(p.dropped);
    assert_eq!(p.deviations.len(), 1);
    let d = &p.deviations[0];
    assert_eq!(d.axis, Axis::Column);
    assert_eq!(d.name, "charge");
    assert_eq!(d.kind, DeviationKind::Mismatch);
    assert_eq!(d.found.as_deref(), Some("I:1"));
}

#[test]
fn optional_metadata_without_fill_drops_on_wrong_kind() {
    let plan = ProjectionPlan {
        columns: Vec::new(),
        metadata: vec![real_meta("energy", None, false, None)],
        ..Default::default()
    };
    let f = frame_meta(1, vec![("energy".into(), Value::Str("high".into()))]);
    let p = project_frame(&f, &plan);
    assert!(p.dropped);
    assert_eq!(p.deviations.len(), 1);
    let d = &p.deviations[0];
    assert_eq!(d.axis, Axis::Metadata);
    assert_eq!(d.name, "energy");
    assert_eq!(d.kind, DeviationKind::Mismatch);
    assert_eq!(d.found.as_deref(), Some("S"));
}

// --- parallel and streamed projected reads match the serial read ---------

// Charge present, absent, present (2 atoms), absent.
const DRIFT: &str = "1\nProperties=species:S:1:pos:R:3:charge:R:1\nH 0 0 0 0.5\n\
                     1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n\
                     2\nProperties=species:S:1:pos:R:3:charge:R:1\nH 0 0 0 1.0\nO 1 1 1 -1.0\n\
                     1\nProperties=species:S:1:pos:R:3\nO 0 0 0\n";

fn drift_plan() -> ProjectionPlan {
    // A concrete fill, as NaN != NaN would break the serial/parallel comparison.
    ProjectionPlan {
        columns: vec![
            real_col_plan("pos", 3, true),
            real_plan("charge", 1, false, Some(-1.5)),
        ],
        metadata: Vec::new(),
        ..Default::default()
    }
}

#[test]
fn projected_batch_with_mixed_presence_of_an_omitted_field_names_the_frame() {
    let plan = ProjectionPlan {
        columns: vec![
            real_col_plan("pos", 3, true),
            real_plan("charge", 1, false, None),
        ],
        metadata: Vec::new(),
        ..Default::default()
    };
    let Err(err) = read_all_batch_projected_from(DRIFT.as_bytes(), &plan) else {
        panic!("mixed presence must fail");
    };
    assert!(matches!(
        err,
        ExtxyzError::Batch(BatchError::MissingColumn { frame: 1, .. })
    ));
}

#[test]
fn projected_batch_errors_use_selected_file_indices() {
    let plan = ProjectionPlan {
        columns: vec![
            real_col_plan("pos", 3, true),
            real_plan("charge", 1, false, None),
        ],
        ..Default::default()
    };
    for result in [
        read_batch_projected_from(DRIFT.as_bytes(), &[2, 3], &plan),
        read_batch_projected_parallel_from(DRIFT.as_bytes(), &[2, 3], None, &plan),
    ] {
        let err = match result {
            Err(err) => err,
            Ok(_) => panic!("mixed presence must fail"),
        };
        assert_eq!(err.frame_index(), Some(3));
        assert!(matches!(
            err,
            ExtxyzError::Batch(BatchError::MissingColumn { frame: 3, name }) if name == "charge"
        ));
    }
}

#[test]
fn projected_batch_with_mixed_presence_of_an_omitted_metadata_key_names_the_frame() {
    let text = "1\nProperties=species:S:1:pos:R:3 energy=-1.5\nH 0 0 0\n\
                1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n\
                1\nProperties=species:S:1:pos:R:3 energy=-2.5\nH 0 0 0\n";
    let plan = ProjectionPlan {
        columns: vec![real_col_plan("pos", 3, true)],
        metadata: vec![real_meta("energy", None, false, None)],
        ..Default::default()
    };
    let Err(err) = read_all_batch_projected_from(text.as_bytes(), &plan) else {
        panic!("mixed presence must fail");
    };
    assert!(
        matches!(
            &err,
            ExtxyzError::Batch(BatchError::MissingMetadata { frame: 1, key }) if key == "energy"
        ),
        "{err:?}"
    );
}

fn assert_same_projection(a: &oxyz_core::ProjectedBatch, b: &oxyz_core::ProjectedBatch) {
    assert_eq!(a.survivors, b.survivors);
    assert_eq!(a.batch, b.batch);
    assert_eq!(a.reports, b.reports);
    assert_eq!(a.presence, b.presence);
    assert_eq!(a.n_optional, b.n_optional);
}

#[test]
fn projected_whole_file_parallel_matches_serial() {
    let plan = drift_plan();
    let serial = read_all_batch_projected_from(DRIFT.as_bytes(), &plan).unwrap();
    assert_eq!(serial.n_optional, 1);
    assert_eq!(serial.presence, vec![true, false, true, false]);
    for threads in [None, Some(1), Some(3)] {
        let par = read_all_batch_projected_parallel_from(DRIFT.as_bytes(), threads, &plan).unwrap();
        assert_same_projection(&par, &serial);
    }
}

#[test]
fn projected_selection_parallel_matches_serial() {
    let plan = drift_plan();
    let serial = read_batch_projected_from(DRIFT.as_bytes(), &[3, 0, 2], &plan).unwrap();
    let par =
        read_batch_projected_parallel_from(DRIFT.as_bytes(), &[3, 0, 2], Some(2), &plan).unwrap();
    assert_same_projection(&par, &serial);
}

#[test]
fn projected_streamed_batches_cover_every_surviving_frame() {
    let whole = read_all_batch_projected_from(DRIFT.as_bytes(), &drift_plan()).unwrap();
    let mut survivors = Vec::new();
    let mut presence = Vec::new();
    let mut n_frames = 0;
    for item in iter_batches_projected_from(DRIFT.as_bytes(), 2, drift_plan()).unwrap() {
        let pb = item.unwrap();
        assert_eq!(pb.n_optional, whole.n_optional);
        assert_eq!(pb.presence.len(), pb.survivors.len() * pb.n_optional);
        survivors.extend(pb.survivors);
        presence.extend(pb.presence);
        n_frames += pb.batch.n_frames();
    }
    assert_eq!(survivors, whole.survivors);
    assert_eq!(presence, whole.presence);
    assert_eq!(n_frames, whole.batch.n_frames());
}

#[test]
fn projected_reads_reject_an_empty_selection() {
    let plan = drift_plan();
    assert!(matches!(
        read_batch_projected_from(DRIFT.as_bytes(), &[], &plan),
        Err(ExtxyzError::Batch(oxyz_core::BatchError::Empty))
    ));
    assert!(matches!(
        read_batch_projected_parallel_from(DRIFT.as_bytes(), &[], None, &plan),
        Err(ExtxyzError::Batch(oxyz_core::BatchError::Empty))
    ));
}

#[test]
fn iter_batches_projected_rejects_zero_frames_per_batch() {
    assert!(matches!(
        iter_batches_projected_from(DRIFT.as_bytes(), 0, drift_plan()),
        Err(ExtxyzError::Batch(
            oxyz_core::BatchError::ZeroFramesPerBatch
        ))
    ));
}

fn plan_entry_meta(name: &str, kind: ColumnKind, shape: Option<usize>) -> PlanMetadata {
    PlanMetadata {
        name: name.into(),
        kind,
        shape,
        required: false,
        fill: None,
        column_major: false,
    }
}

fn int_frame() -> Frame {
    Frame {
        n_atoms: 2,
        columns: vec![col("charge", ColumnData::Int(vec![1, 2]), 1)],
        metadata: vec![
            ("energy".into(), Value::Int(0)),
            (
                "Lattice".into(),
                Value::IntArray(vec![5, 0, 0, 0, 5, 0, 0, 0, 5]),
            ),
        ],
    }
}

fn real_plan_over_int_frame(promote_int: bool) -> ProjectionPlan {
    ProjectionPlan {
        columns: vec![PlanColumn {
            name: "charge".into(),
            kind: ColumnKind::Real,
            width: 1,
            required: false,
            fill: None,
        }],
        metadata: vec![
            plan_entry_meta("energy", ColumnKind::Real, None),
            plan_entry_meta("Lattice", ColumnKind::Real, Some(9)),
        ],
        promote_int,
    }
}

#[test]
fn promote_int_accepts_int_column_scalar_and_array_for_real_entries() {
    let out = project_frame(&int_frame(), &real_plan_over_int_frame(true));
    assert!(out.deviations.is_empty());
    assert!(!out.dropped);
    assert_eq!(out.frame.columns[0].data, ColumnData::Real(vec![1.0, 2.0]));
    assert_eq!(out.frame.metadata[0].1, Value::Real(0.0));
    assert_eq!(
        out.frame.metadata[1].1,
        Value::RealArray(vec![5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0])
    );
}

#[test]
fn without_promote_int_an_int_value_is_a_mismatch() {
    let out = project_frame(&int_frame(), &real_plan_over_int_frame(false));
    assert_eq!(out.deviations.len(), 3);
    assert!(
        out.deviations
            .iter()
            .all(|d| d.kind == DeviationKind::Mismatch)
    );
    assert!(out.dropped);
}

#[test]
fn an_int_entry_never_accepts_real() {
    let frame = Frame {
        n_atoms: 1,
        columns: vec![col("charge", ColumnData::Real(vec![1.0]), 1)],
        metadata: vec![("energy".into(), Value::Real(0.0))],
    };
    let plan = ProjectionPlan {
        columns: vec![PlanColumn {
            name: "charge".into(),
            kind: ColumnKind::Int,
            width: 1,
            required: false,
            fill: None,
        }],
        metadata: vec![plan_entry_meta("energy", ColumnKind::Int, None)],
        promote_int: true,
    };
    let out = project_frame(&frame, &plan);
    assert_eq!(out.deviations.len(), 2);
    assert!(
        out.deviations
            .iter()
            .all(|d| d.kind == DeviationKind::Mismatch)
    );
}

#[test]
fn an_exact_real_occurrence_wins_over_a_promotable_duplicate() {
    let frame = Frame {
        n_atoms: 1,
        columns: vec![
            col("charge", ColumnData::Int(vec![7]), 1),
            col("charge", ColumnData::Real(vec![0.5]), 1),
        ],
        metadata: vec![
            ("energy".into(), Value::Int(7)),
            ("energy".into(), Value::Real(0.5)),
        ],
    };
    let plan = ProjectionPlan {
        columns: vec![PlanColumn {
            name: "charge".into(),
            kind: ColumnKind::Real,
            width: 1,
            required: false,
            fill: None,
        }],
        metadata: vec![plan_entry_meta("energy", ColumnKind::Real, None)],
        promote_int: true,
    };
    let out = project_frame(&frame, &plan);
    assert!(out.deviations.is_empty());
    assert_eq!(out.frame.columns[0].data, ColumnData::Real(vec![0.5]));
    assert_eq!(out.frame.metadata[0].1, Value::Real(0.5));
}

#[test]
fn promote_int_keeps_the_shape_of_a_2d_int_array() {
    let frame = frame_meta(
        0,
        vec![(
            "m".into(),
            Value::IntArray2D {
                rows: 2,
                cols: 3,
                data: vec![1, 2, 3, 4, 5, 6],
            },
        )],
    );
    let plan = ProjectionPlan {
        metadata: vec![plan_entry_meta("m", ColumnKind::Real, Some(6))],
        promote_int: true,
        ..Default::default()
    };
    let out = project_frame(&frame, &plan);
    assert!(out.deviations.is_empty());
    assert_eq!(
        out.frame.metadata[0].1,
        Value::RealArray2D {
            rows: 2,
            cols: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    );
}

fn real_energy_plan() -> ProjectionPlan {
    ProjectionPlan {
        metadata: vec![plan_entry_meta("energy", ColumnKind::Real, None)],
        promote_int: true,
        ..Default::default()
    }
}

fn real_charge_plan(width: usize) -> ProjectionPlan {
    ProjectionPlan {
        columns: vec![PlanColumn {
            name: "charge".into(),
            kind: ColumnKind::Real,
            width,
            required: false,
            fill: None,
        }],
        promote_int: true,
        ..Default::default()
    }
}

#[test]
fn a_promotable_metadata_duplicate_wins_over_an_earlier_mismatch() {
    let frame = frame_meta(
        0,
        vec![
            ("energy".into(), Value::Str("x".into())),
            ("energy".into(), Value::Int(3)),
        ],
    );
    let out = project_frame(&frame, &real_energy_plan());
    assert!(out.deviations.is_empty());
    assert!(!out.dropped);
    assert_eq!(out.frame.metadata[0].1, Value::Real(3.0));
}

#[test]
fn a_promotable_column_duplicate_wins_over_an_earlier_mismatch() {
    let frame = frame(
        1,
        vec![
            col("charge", ColumnData::Str(vec!["x".into()]), 1),
            col("charge", ColumnData::Int(vec![3]), 1),
        ],
    );
    let out = project_frame(&frame, &real_charge_plan(1));
    assert!(out.deviations.is_empty());
    assert!(!out.dropped);
    assert_eq!(out.frame.columns[0].data, ColumnData::Real(vec![3.0]));
}

#[test]
fn promote_int_does_not_widen_an_int_column() {
    let frame = frame(1, vec![col("charge", ColumnData::Int(vec![1, 2, 3]), 3)]);
    let out = project_frame(&frame, &real_charge_plan(1));
    assert_eq!(out.deviations.len(), 1);
    assert_eq!(out.deviations[0].kind, DeviationKind::Mismatch);
    assert_eq!(out.deviations[0].found.as_deref(), Some("I:3"));
}

#[test]
fn promote_int_does_not_reshape_an_int_array() {
    let frame = frame_meta(0, vec![("Lattice".into(), Value::IntArray(vec![1, 2, 3]))]);
    let plan = ProjectionPlan {
        metadata: vec![plan_entry_meta("Lattice", ColumnKind::Real, Some(9))],
        promote_int: true,
        ..Default::default()
    };
    let out = project_frame(&frame, &plan);
    assert_eq!(out.deviations.len(), 1);
    assert_eq!(out.deviations[0].kind, DeviationKind::Mismatch);
    assert_eq!(out.deviations[0].found.as_deref(), Some("I[3]"));
}

#[test]
fn presence_flags_found_filled_and_mismatched() {
    let plan = ProjectionPlan {
        columns: vec![
            real_col_plan("pos", 3, true),
            real_plan("charge", 1, false, Some(-1.5)),
            PlanColumn {
                name: "tag".into(),
                kind: ColumnKind::Str,
                width: 1,
                required: false,
                fill: Some(Fill::Str("x".into())),
            },
        ],
        metadata: vec![real_meta("energy", None, false, None)],
        ..Default::default()
    };
    let f = Frame {
        n_atoms: 1,
        columns: vec![
            col("pos", ColumnData::Real(vec![0.0; 3]), 3),
            col("charge", ColumnData::Int(vec![2]), 1),
        ],
        metadata: vec![("energy".into(), Value::Real(-1.0))],
    };
    let p = project_frame(&f, &plan);
    assert!(!p.dropped);
    assert_eq!(plan.n_optional(), 3);
    assert_eq!(p.present, vec![true, false, true]);
}

#[test]
fn projected_batch_presence_has_a_row_per_survivor_only() {
    let input = "1\nProperties=species:S:1:pos:R:3:charge:R:1\nH 0 0 0 0.5\n\
                 1\nProperties=species:S:1:pos:R:3:id:I:1\nH 0 0 0 7\n";
    let plan = ProjectionPlan {
        columns: vec![
            real_col_plan("pos", 3, true),
            PlanColumn {
                name: "id".into(),
                kind: ColumnKind::Int,
                width: 1,
                required: true,
                fill: None,
            },
            real_plan("charge", 1, false, Some(-1.5)),
        ],
        metadata: Vec::new(),
        ..Default::default()
    };
    let pb = read_all_batch_projected_from(input.as_bytes(), &plan).unwrap();
    assert_eq!(pb.survivors, vec![1]);
    assert_eq!(pb.n_optional, 1);
    assert_eq!(pb.presence, vec![false]);
}

#[test]
fn presence_flags_count_a_promoted_field_as_present() {
    let p = project_frame(&int_frame(), &real_plan_over_int_frame(true));
    assert!(p.deviations.is_empty());
    assert_eq!(p.present, vec![true, true, true]);
}

fn column_major_plan(promote_int: bool) -> ProjectionPlan {
    let mut entry = plan_entry_meta("m", ColumnKind::Real, Some(6));
    entry.column_major = true;
    ProjectionPlan {
        metadata: vec![entry],
        promote_int,
        ..Default::default()
    }
}

#[test]
fn a_column_major_entry_flattens_a_2d_value_column_by_column() {
    let real = frame_meta(
        0,
        vec![(
            "m".into(),
            Value::RealArray2D {
                rows: 2,
                cols: 3,
                data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            },
        )],
    );
    let int = frame_meta(
        0,
        vec![(
            "m".into(),
            Value::IntArray2D {
                rows: 2,
                cols: 3,
                data: vec![1, 2, 3, 4, 5, 6],
            },
        )],
    );
    let expected = Value::RealArray(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    for (frame, promote_int) in [(real, false), (int, true)] {
        let out = project_frame(&frame, &column_major_plan(promote_int));
        assert!(out.deviations.is_empty());
        assert_eq!(out.frame.metadata[0].1, expected);
    }
}

#[test]
fn a_column_major_entry_keeps_a_flat_value_as_written() {
    let flat = Value::RealArray(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let frame = frame_meta(0, vec![("m".into(), flat.clone())]);
    let out = project_frame(&frame, &column_major_plan(false));
    assert!(out.deviations.is_empty());
    assert_eq!(out.frame.metadata[0].1, flat);
}

#[test]
fn column_major_projection_preserves_non_real_value_kinds() {
    let cases = [
        (
            ColumnKind::Int,
            Value::IntArray2D {
                rows: 2,
                cols: 3,
                data: vec![1, 2, 3, 4, 5, 6],
            },
            Value::IntArray(vec![1, 4, 2, 5, 3, 6]),
        ),
        (
            ColumnKind::Bool,
            Value::BoolArray2D {
                rows: 2,
                cols: 3,
                data: vec![true, false, false, false, true, true],
            },
            Value::BoolArray(vec![true, false, false, true, false, true]),
        ),
        (
            ColumnKind::Str,
            Value::StrArray2D {
                rows: 2,
                cols: 3,
                data: Vec::from(["a", "b", "c", "d", "e", "f"].map(CompactString::from)),
            },
            Value::StrArray(Vec::from(
                ["a", "d", "b", "e", "c", "f"].map(CompactString::from),
            )),
        ),
    ];
    for (kind, value, expected) in cases {
        let mut entry = plan_entry_meta("m", kind, Some(6));
        entry.column_major = true;
        let plan = ProjectionPlan {
            metadata: vec![entry],
            ..Default::default()
        };
        let out = project_frame(&frame_meta(0, vec![("m".into(), value)]), &plan);
        assert!(out.deviations.is_empty());
        assert_eq!(out.frame.metadata, vec![("m".into(), expected)]);
    }
}
