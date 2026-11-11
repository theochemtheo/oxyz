//! Projecting a frame onto a fixed, declared schema.
//!
//! Each output frame has exactly the plan's fields, in declaration order. The
//! module reshapes and reports; policy (raise, warn, drop) is the Python
//! layer's. A frame is marked `dropped` when a required field has no fill, or
//! when any field with no fill is present with the wrong kind or width. An
//! absent optional field with no fill is omitted from the output.

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
    /// `None` drops the frame when the field is required or mismatched, and
    /// omits the field when it is optional and absent.
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
    /// As for [`PlanColumn::fill`].
    pub fill: Option<Fill>,
}

/// The fixed shape to project each frame onto, in declaration order.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ProjectionPlan {
    pub columns: Vec<PlanColumn>,
    pub metadata: Vec<PlanMetadata>,
    /// Let a REAL entry accept integer data, converted to `f64`.
    pub promote_int: bool,
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

/// The outcome of projecting one frame. When `dropped`, `frame` is incomplete.
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
        // A kind/fill mismatch is a plan bug; fall back rather than panic.
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
        // Flattened, as in the batch: only width is typed, not shape.
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
        // A kind/fill mismatch is a plan bug; fall back rather than panic.
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
        match lookup_metadata(frame, pm, plan.promote_int) {
            Lookup::Conforming(value) => {
                metadata.push((CompactString::from(pm.name.as_str()), value.clone()));
            }
            Lookup::Promotable(value) => {
                metadata.push((CompactString::from(pm.name.as_str()), promote_value(value)));
            }
            found => {
                let mismatched = matches!(found, Lookup::Mismatch(_));
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
                    None => *dropped |= mismatched || pm.required,
                }
            }
        }
    }
    metadata
}

/// A plan field looked up in a frame that may repeat the name: a conforming
/// occurrence if any, so a stray duplicate does not report a mismatch, else
/// one that promotes to the plan's REAL kind.
enum Lookup<T> {
    Conforming(T),
    Promotable(T),
    Mismatch(T),
    Absent,
}

fn lookup_column<'a>(frame: &'a Frame, pc: &PlanColumn, promote_int: bool) -> Lookup<&'a Column> {
    let mut promotable = None;
    let mut mismatch = None;
    for column in &frame.columns {
        if column.name == pc.name {
            let kind = column.data.kind();
            if kind == pc.kind && column.width == pc.width {
                return Lookup::Conforming(column);
            }
            if promote_int
                && pc.kind == ColumnKind::Real
                && kind == ColumnKind::Int
                && column.width == pc.width
            {
                promotable.get_or_insert(column);
            } else {
                mismatch.get_or_insert(column);
            }
        }
    }
    promotable.map_or_else(
        || mismatch.map_or(Lookup::Absent, Lookup::Mismatch),
        Lookup::Promotable,
    )
}

fn lookup_metadata<'a>(
    frame: &'a Frame,
    pm: &PlanMetadata,
    promote_int: bool,
) -> Lookup<&'a Value> {
    let mut promotable = None;
    let mut mismatch = None;
    for (key, value) in &frame.metadata {
        if *key == pm.name {
            let (kind, shape) = value_kind_shape(value);
            if kind == pm.kind && shape == pm.shape {
                return Lookup::Conforming(value);
            }
            if promote_int
                && pm.kind == ColumnKind::Real
                && kind == ColumnKind::Int
                && shape == pm.shape
            {
                promotable.get_or_insert(value);
            } else {
                mismatch.get_or_insert(value);
            }
        }
    }
    promotable.map_or_else(
        || mismatch.map_or(Lookup::Absent, Lookup::Mismatch),
        Lookup::Promotable,
    )
}

fn ints_to_reals(data: &[i64]) -> Vec<f64> {
    #[allow(clippy::cast_precision_loss)]
    data.iter().map(|&x| x as f64).collect()
}

/// The `f64` form of an integer column of the width the plan asked for.
fn promote_column(column: &Column) -> Column {
    let data = match &column.data {
        ColumnData::Int(v) => ColumnData::Real(ints_to_reals(v)),
        other => other.clone(),
    };
    Column {
        name: column.name.clone(),
        width: column.width,
        data,
    }
}

/// The `f64` form of an integer scalar or array, keeping 2-D shape.
fn promote_value(value: &Value) -> Value {
    match value {
        #[allow(clippy::cast_precision_loss)]
        Value::Int(x) => Value::Real(*x as f64),
        Value::IntArray(v) => Value::RealArray(ints_to_reals(v)),
        Value::IntArray2D { rows, cols, data } => Value::RealArray2D {
            rows: *rows,
            cols: *cols,
            data: ints_to_reals(data),
        },
        other => other.clone(),
    }
}

/// Project `frame` onto `plan`: undeclared fields dropped, absent optionals
/// filled, and missing or wrong-kind fields reported. Fill widths come from the
/// plan and are trusted.
pub fn project_frame(frame: &Frame, plan: &ProjectionPlan) -> Projected {
    let mut deviations = Vec::new();
    let mut dropped = false;
    let n_atoms = frame.n_atoms;

    let mut columns = Vec::with_capacity(plan.columns.len());
    for pc in &plan.columns {
        let expected = column_sig(pc.kind, pc.width);
        match lookup_column(frame, pc, plan.promote_int) {
            Lookup::Conforming(existing) => columns.push(existing.clone()),
            Lookup::Promotable(existing) => columns.push(promote_column(existing)),
            found => {
                let mismatched = matches!(found, Lookup::Mismatch(_));
                // Optional-absent is silent.
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
                    None => dropped |= mismatched || pc.required,
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
