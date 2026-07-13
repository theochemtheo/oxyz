//! Projecting a frame onto a fixed, declared schema.
//!
//! A validation gate (does this file match what I expect) can only report;
//! projection *transforms* — every output frame ends up with exactly the plan's
//! declared fields, in declaration order: undeclared fields dropped, absent
//! optionals filled, wrong-kind fields flagged. Policy (raise / warn / drop)
//! lives in the Python surface; this module is policy-free — it reshapes and
//! reports what it saw, and marks a frame `dropped` when a required field has no
//! fill to hold the fixed shape.

use crate::model::{Column, ColumnData, ColumnKind, Frame, Value};
use compact_str::CompactString;

/// A single-cell fill value; its kind matches the field it fills.
#[derive(Debug, Clone, PartialEq)]
pub enum Fill {
    Real(f64),
    Int(i64),
    Bool(bool),
    Str(String),
}

/// One declared per-atom column in the projected shape.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanColumn {
    pub name: String,
    pub kind: ColumnKind,
    pub width: usize,
    pub required: bool,
    /// `Some` when the field can be filled if absent/wrong; `None` marks a
    /// required field with no null (drop rather than fabricate one).
    pub fill: Option<Fill>,
}

/// One declared metadata key in the projected shape.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanMetadata {
    pub name: String,
    pub kind: ColumnKind,
    /// `None` for a scalar, `Some(n)` for an array of length n.
    pub shape: Option<usize>,
    pub required: bool,
    pub fill: Option<Fill>,
}

/// The fixed shape to project each frame onto, in declaration order.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ProjectionPlan {
    pub columns: Vec<PlanColumn>,
    pub metadata: Vec<PlanMetadata>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    Column,
    Metadata,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviationKind {
    Missing,
    Mismatch,
}

/// One reported divergence from the plan. `found` is `None` for `Missing`.
#[derive(Debug, Clone, PartialEq)]
pub struct Deviation {
    pub axis: Axis,
    pub name: String,
    pub kind: DeviationKind,
    pub expected: String,
    pub found: Option<String>,
}

/// The outcome of projecting one frame. When `dropped`, `frame` is incomplete
/// and the caller discards it (having consumed `deviations` for reporting).
#[derive(Debug, Clone, PartialEq)]
pub struct Projected {
    pub frame: Frame,
    pub deviations: Vec<Deviation>,
    pub dropped: bool,
}

/// `R:3`-style signature for a column plan field.
fn column_sig(kind: ColumnKind, width: usize) -> String {
    format!("{kind}:{width}")
}

/// Build a width-`width` fill column of `n_atoms` rows from `fill`.
fn materialise_column(kind: ColumnKind, width: usize, n_atoms: usize, fill: &Fill) -> ColumnData {
    let count = n_atoms * width;
    match (kind, fill) {
        (ColumnKind::Real, Fill::Real(x)) => ColumnData::Real(vec![*x; count]),
        (ColumnKind::Int, Fill::Int(x)) => ColumnData::Int(vec![*x; count]),
        (ColumnKind::Bool, Fill::Bool(x)) => ColumnData::Bool(vec![*x; count]),
        (ColumnKind::Str, Fill::Str(x)) => {
            ColumnData::Str(vec![CompactString::from(x.as_str()); count])
        }
        // The compiler in Python pairs kind with a like-kinded fill; a mismatch
        // here is a plan-construction bug, so fall back to the declared kind's
        // zero rather than panicking on hostile input.
        (ColumnKind::Real, _) => ColumnData::Real(vec![f64::NAN; count]),
        (ColumnKind::Int, _) => ColumnData::Int(vec![0; count]),
        (ColumnKind::Bool, _) => ColumnData::Bool(vec![false; count]),
        (ColumnKind::Str, _) => ColumnData::Str(vec![CompactString::default(); count]),
    }
}

/// `(kind, shape)` of a metadata value, where `shape` is `None` for a scalar.
fn value_kind_shape(value: &Value) -> (ColumnKind, Option<usize>) {
    match value {
        Value::Real(_) => (ColumnKind::Real, None),
        Value::Int(_) => (ColumnKind::Int, None),
        Value::Bool(_) => (ColumnKind::Bool, None),
        Value::Str(_) => (ColumnKind::Str, None),
        Value::RealArray(v) => (ColumnKind::Real, Some(v.len())),
        Value::IntArray(v) => (ColumnKind::Int, Some(v.len())),
        Value::BoolArray(v) => (ColumnKind::Bool, Some(v.len())),
        Value::StrArray(v) => (ColumnKind::Str, Some(v.len())),
        // Flattened length, matching the batch's own treatment of 2-D
        // metadata: shape is not part of this typing contract, only width.
        Value::RealArray2D { rows, cols, .. } => (ColumnKind::Real, Some(rows * cols)),
        Value::IntArray2D { rows, cols, .. } => (ColumnKind::Int, Some(rows * cols)),
        Value::BoolArray2D { rows, cols, .. } => (ColumnKind::Bool, Some(rows * cols)),
        Value::StrArray2D { rows, cols, .. } => (ColumnKind::Str, Some(rows * cols)),
    }
}

/// `R` (scalar) or `R[n]` (array) signature for a metadata plan field.
fn metadata_sig(kind: ColumnKind, shape: Option<usize>) -> String {
    match shape {
        None => format!("{kind}"),
        Some(n) => format!("{kind}[{n}]"),
    }
}

fn materialise_value(kind: ColumnKind, shape: Option<usize>, fill: &Fill) -> Value {
    match (shape, kind, fill) {
        (None, ColumnKind::Real, Fill::Real(x)) => Value::Real(*x),
        (None, ColumnKind::Int, Fill::Int(x)) => Value::Int(*x),
        (None, ColumnKind::Bool, Fill::Bool(x)) => Value::Bool(*x),
        (None, ColumnKind::Str, Fill::Str(x)) => Value::Str(CompactString::from(x.as_str())),
        (Some(n), ColumnKind::Real, Fill::Real(x)) => Value::RealArray(vec![*x; n]),
        (Some(n), ColumnKind::Int, Fill::Int(x)) => Value::IntArray(vec![*x; n]),
        (Some(n), ColumnKind::Bool, Fill::Bool(x)) => Value::BoolArray(vec![*x; n]),
        (Some(n), ColumnKind::Str, Fill::Str(x)) => {
            Value::StrArray(vec![CompactString::from(x.as_str()); n])
        }
        // Kind/fill mismatch is a plan-construction bug; produce a like-kinded
        // zero rather than panic (see materialise_column).
        (None, ColumnKind::Real, _) => Value::Real(f64::NAN),
        (None, ColumnKind::Int, _) => Value::Int(0),
        (None, ColumnKind::Bool, _) => Value::Bool(false),
        (None, ColumnKind::Str, _) => Value::Str(CompactString::default()),
        (Some(n), ColumnKind::Real, _) => Value::RealArray(vec![f64::NAN; n]),
        (Some(n), ColumnKind::Int, _) => Value::IntArray(vec![0; n]),
        (Some(n), ColumnKind::Bool, _) => Value::BoolArray(vec![false; n]),
        (Some(n), ColumnKind::Str, _) => Value::StrArray(vec![CompactString::default(); n]),
    }
}

fn project_metadata(
    frame: &Frame,
    plan: &ProjectionPlan,
    deviations: &mut Vec<Deviation>,
    dropped: &mut bool,
) -> Vec<(CompactString, Value)> {
    let mut metadata = Vec::with_capacity(plan.metadata.len());
    for pm in &plan.metadata {
        let expected = metadata_sig(pm.kind, pm.shape);
        match lookup_metadata(frame, pm) {
            Lookup::Conforming(value) => {
                metadata.push((CompactString::from(pm.name.as_str()), value.clone()));
            }
            found => {
                match found {
                    Lookup::Mismatch(value) => {
                        let (k, s) = value_kind_shape(value);
                        deviations.push(Deviation {
                            axis: Axis::Metadata,
                            name: pm.name.clone(),
                            kind: DeviationKind::Mismatch,
                            expected,
                            found: Some(metadata_sig(k, s)),
                        });
                    }
                    Lookup::Absent if pm.required => deviations.push(Deviation {
                        axis: Axis::Metadata,
                        name: pm.name.clone(),
                        kind: DeviationKind::Missing,
                        expected,
                        found: None,
                    }),
                    _ => {}
                }
                match &pm.fill {
                    Some(fill) => metadata.push((
                        CompactString::from(pm.name.as_str()),
                        materialise_value(pm.kind, pm.shape, fill),
                    )),
                    None => *dropped = true,
                }
            }
        }
    }
    metadata
}

/// Outcome of looking a plan field up in a frame that may hold the name more
/// than once (the model preserves duplicate keys): a conforming occurrence if
/// any, else the first non-conforming one, else absent. Preferring a conforming
/// occurrence stops a stray duplicate from spuriously reporting a mismatch.
enum Lookup<T> {
    Conforming(T),
    Mismatch(T),
    Absent,
}

fn lookup_column<'a>(frame: &'a Frame, pc: &PlanColumn) -> Lookup<&'a Column> {
    let mut mismatch = None;
    for column in &frame.columns {
        if column.name == pc.name {
            if column.data.kind() == pc.kind && column.width == pc.width {
                return Lookup::Conforming(column);
            }
            mismatch.get_or_insert(column);
        }
    }
    mismatch.map_or(Lookup::Absent, Lookup::Mismatch)
}

fn lookup_metadata<'a>(frame: &'a Frame, pm: &PlanMetadata) -> Lookup<&'a Value> {
    let mut mismatch = None;
    for (key, value) in &frame.metadata {
        if *key == pm.name {
            if value_kind_shape(value) == (pm.kind, pm.shape) {
                return Lookup::Conforming(value);
            }
            mismatch.get_or_insert(value);
        }
    }
    mismatch.map_or(Lookup::Absent, Lookup::Mismatch)
}

/// Project `frame` onto `plan`: every output field is exactly the plan's, in
/// declaration order. Undeclared fields dropped; absent optionals filled;
/// absent-required and wrong-kind/width fields reported (filled if the plan
/// carries a fill, else the frame is marked `dropped`).
///
/// Fill widths (`n_atoms * width`, and array `shape`) come from the plan, which
/// Python builds from a `SchemaSpec`; the core trusts them (a hostile plan with
/// an absurd width could over-allocate, but no *file* can produce one).
pub fn project_frame(frame: &Frame, plan: &ProjectionPlan) -> Projected {
    let mut deviations = Vec::new();
    let mut dropped = false;
    let n_atoms = frame.n_atoms;

    let mut columns = Vec::with_capacity(plan.columns.len());
    for pc in &plan.columns {
        let expected = column_sig(pc.kind, pc.width);
        match lookup_column(frame, pc) {
            Lookup::Conforming(existing) => columns.push(existing.clone()),
            found => {
                // Absent, or present with the wrong kind/width. Record the
                // divergence (required-absent = Missing, present-wrong =
                // Mismatch); optional-absent is silent.
                match found {
                    Lookup::Mismatch(existing) => deviations.push(Deviation {
                        axis: Axis::Column,
                        name: pc.name.clone(),
                        kind: DeviationKind::Mismatch,
                        expected,
                        found: Some(column_sig(existing.data.kind(), existing.width)),
                    }),
                    Lookup::Absent if pc.required => deviations.push(Deviation {
                        axis: Axis::Column,
                        name: pc.name.clone(),
                        kind: DeviationKind::Missing,
                        expected,
                        found: None,
                    }),
                    _ => {}
                }
                match &pc.fill {
                    Some(fill) => columns.push(Column {
                        name: CompactString::from(pc.name.as_str()),
                        width: pc.width,
                        data: materialise_column(pc.kind, pc.width, n_atoms, fill),
                    }),
                    None => dropped = true,
                }
            }
        }
    }

    let metadata = project_metadata(frame, plan, &mut deviations, &mut dropped);

    Projected {
        frame: Frame {
            n_atoms,
            columns,
            metadata,
        },
        deviations,
        dropped,
    }
}
