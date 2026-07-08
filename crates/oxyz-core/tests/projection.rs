use oxyz_core::model::{Column, ColumnData, ColumnKind, Frame, Value};
use oxyz_core::project::{
    Axis, DeviationKind, Fill, PlanColumn, PlanMetadata, Projected, ProjectionPlan,
};

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
