//! Concatenated batches of frames: the CSR-style unit for bulk and training
//! reads.
//!
//! Per-atom columns from every frame are concatenated atom-major; `offsets`
//! (PyG's `ptr`) marks frame boundaries, so frame `i` occupies rows
//! `offsets[i]..offsets[i + 1]`. Per-frame metadata becomes frame-major
//! columns: one row per frame.
//!
//! The first frame pushed defines the contract: every later frame must match
//! its column names, kinds, and widths, and its metadata keys. The one
//! sanctioned unification is Int/Real promotion (the same rule as
//! [`crate::schema`]'s unify); everything else is an error — hiding schema
//! inconsistencies is a non-goal. A pinned dtype contract via `Schema` is
//! the designed extension, not implemented yet.

use thiserror::Error;

use crate::model::{Column, ColumnData, ColumnKind, Frame, Value};

#[derive(Debug, Error)]
pub enum BatchError {
    #[error("batch is empty")]
    Empty,

    #[error("frames per batch must be at least 1")]
    ZeroFramesPerBatch,

    #[error(
        "frame {frame} column {name:?}: expected {expected_kind}:{expected_width}, \
         found {found_kind}:{found_width}"
    )]
    ColumnMismatch {
        frame: usize,
        name: String,
        expected_kind: ColumnKind,
        expected_width: usize,
        found_kind: ColumnKind,
        found_width: usize,
    },

    #[error("frame {frame} is missing column {name:?}")]
    MissingColumn { frame: usize, name: String },

    #[error("frame {frame} has unexpected column {name:?}")]
    UnexpectedColumn { frame: usize, name: String },

    #[error(
        "frame {frame} metadata {key:?}: expected {expected_kind}:{expected_width}, \
         found {found_kind}:{found_width}"
    )]
    MetadataMismatch {
        frame: usize,
        key: String,
        expected_kind: ColumnKind,
        expected_width: usize,
        found_kind: ColumnKind,
        found_width: usize,
    },

    #[error("frame {frame} is missing metadata {key:?}")]
    MissingMetadata { frame: usize, key: String },

    #[error("frame {frame} has unexpected metadata {key:?}")]
    UnexpectedMetadata { frame: usize, key: String },
}

/// Frames concatenated atom-major, with frame-major metadata columns.
#[derive(Debug, Clone, PartialEq)]
pub struct Batch {
    /// Frame boundaries, length `n_frames + 1`, starting at 0.
    pub offsets: Vec<usize>,
    /// Per-atom columns; each upholds `data.len() == total_atoms * width`.
    pub columns: Vec<Column>,
    /// Per-frame metadata; each upholds `data.len() == n_frames * width`.
    pub metadata: Vec<Column>,
}

impl Batch {
    pub fn n_frames(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn total_atoms(&self) -> usize {
        *self.offsets.last().expect("offsets always holds 0")
    }
}

/// Accumulates frames into a [`Batch`]; `push` enforces the contract.
pub struct BatchBuilder {
    offsets: Vec<usize>,
    columns: Vec<Column>,
    metadata: Vec<Column>,
}

impl Default for BatchBuilder {
    fn default() -> Self {
        BatchBuilder::new()
    }
}

impl BatchBuilder {
    pub fn new() -> Self {
        BatchBuilder {
            offsets: vec![0],
            columns: Vec::new(),
            metadata: Vec::new(),
        }
    }

    pub fn n_frames(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn push(&mut self, frame: Frame) -> Result<(), BatchError> {
        let frame_pos = self.n_frames();
        let n_atoms = frame.n_atoms;

        if frame_pos == 0 {
            self.columns = frame.columns;
            self.metadata = frame
                .metadata
                .into_iter()
                .map(|(key, value)| {
                    let (data, width) = value_to_data(value);
                    Column {
                        name: key,
                        width,
                        data,
                    }
                })
                .collect();
        } else {
            // Match by name: extxyz files may reorder columns between frames.
            let mut incoming: Vec<Option<Column>> = frame.columns.into_iter().map(Some).collect();
            for column in &mut self.columns {
                let position = incoming
                    .iter()
                    .position(|slot| slot.as_ref().is_some_and(|c| c.name == column.name))
                    .ok_or_else(|| BatchError::MissingColumn {
                        frame: frame_pos,
                        name: column.name.clone(),
                    })?;
                let found = incoming[position].take().expect("position just found");
                append_column(column, found.data, found.width, frame_pos, false)?;
            }
            if let Some(extra) = incoming.into_iter().flatten().next() {
                return Err(BatchError::UnexpectedColumn {
                    frame: frame_pos,
                    name: extra.name,
                });
            }

            let mut incoming: Vec<Option<(String, Value)>> =
                frame.metadata.into_iter().map(Some).collect();
            for column in &mut self.metadata {
                let position = incoming
                    .iter()
                    .position(|slot| slot.as_ref().is_some_and(|(key, _)| *key == column.name))
                    .ok_or_else(|| BatchError::MissingMetadata {
                        frame: frame_pos,
                        key: column.name.clone(),
                    })?;
                let (_, value) = incoming[position].take().expect("position just found");
                let (data, width) = value_to_data(value);
                append_column(column, data, width, frame_pos, true)?;
            }
            if let Some((key, _)) = incoming.into_iter().flatten().next() {
                return Err(BatchError::UnexpectedMetadata {
                    frame: frame_pos,
                    key,
                });
            }
        }

        self.offsets.push(self.offsets.last().unwrap() + n_atoms);
        Ok(())
    }

    pub fn finish(self) -> Result<Batch, BatchError> {
        if self.n_frames() == 0 {
            return Err(BatchError::Empty);
        }
        Ok(Batch {
            offsets: self.offsets,
            columns: self.columns,
            metadata: self.metadata,
        })
    }
}

/// A metadata value as a one-row column: scalars have width 1, arrays their
/// length.
fn value_to_data(value: Value) -> (ColumnData, usize) {
    match value {
        Value::Real(x) => (ColumnData::Real(vec![x]), 1),
        Value::Int(x) => (ColumnData::Int(vec![x]), 1),
        Value::Bool(x) => (ColumnData::Bool(vec![x]), 1),
        Value::Str(x) => (ColumnData::Str(vec![x]), 1),
        Value::RealArray(values) => {
            let width = values.len();
            (ColumnData::Real(values), width)
        }
        Value::IntArray(values) => {
            let width = values.len();
            (ColumnData::Int(values), width)
        }
        Value::BoolArray(values) => {
            let width = values.len();
            (ColumnData::Bool(values), width)
        }
        Value::StrArray(values) => {
            let width = values.len();
            (ColumnData::Str(values), width)
        }
    }
}

fn append_column(
    existing: &mut Column,
    incoming: ColumnData,
    incoming_width: usize,
    frame: usize,
    is_metadata: bool,
) -> Result<(), BatchError> {
    let mismatch = |existing: &Column, incoming: &ColumnData| {
        let (expected_kind, expected_width) = (existing.data.kind(), existing.width);
        let (found_kind, found_width) = (incoming.kind(), incoming_width);
        if is_metadata {
            BatchError::MetadataMismatch {
                frame,
                key: existing.name.clone(),
                expected_kind,
                expected_width,
                found_kind,
                found_width,
            }
        } else {
            BatchError::ColumnMismatch {
                frame,
                name: existing.name.clone(),
                expected_kind,
                expected_width,
                found_kind,
                found_width,
            }
        }
    };

    if existing.width != incoming_width {
        return Err(mismatch(existing, &incoming));
    }

    // Take the buffer out so the Int -> Real promotion can replace it.
    let current = std::mem::replace(&mut existing.data, ColumnData::Real(Vec::new()));
    match merge_data(current, incoming) {
        Ok(merged) => {
            existing.data = merged;
            Ok(())
        }
        Err((current, incoming)) => {
            existing.data = current;
            Err(mismatch(existing, &incoming))
        }
    }
}

/// Concatenate two buffers of the same kind; Int/Real pairs promote to Real
/// (the schema's one sanctioned unification). Anything else is returned for
/// error reporting.
fn merge_data(
    current: ColumnData,
    incoming: ColumnData,
) -> Result<ColumnData, (ColumnData, ColumnData)> {
    use ColumnData::{Bool, Int, Real, Str};

    match (current, incoming) {
        (Real(mut a), Real(b)) => {
            a.extend(b);
            Ok(Real(a))
        }
        (Int(mut a), Int(b)) => {
            a.extend(b);
            Ok(Int(a))
        }
        (Bool(mut a), Bool(b)) => {
            a.extend(b);
            Ok(Bool(a))
        }
        (Str(mut a), Str(b)) => {
            a.extend(b);
            Ok(Str(a))
        }
        (Real(mut a), Int(b)) => {
            a.extend(b.into_iter().map(|x| x as f64));
            Ok(Real(a))
        }
        (Int(a), Real(b)) => {
            let mut promoted: Vec<f64> = a.into_iter().map(|x| x as f64).collect();
            promoted.extend(b);
            Ok(Real(promoted))
        }
        (current, incoming) => Err((current, incoming)),
    }
}
