//! Schema inference: a fold over frames recording which columns and metadata
//! keys appear, with what types and shapes, and how consistently.
//!
//! Role: dataset diagnostics today, and eventually the dtype/presence
//! contract for assembling a `Batch` across drifting frames. Deliberately
//! *not* a parsing fast path (supersedes the original motivation): extxyz
//! headers are self-describing per frame, so parse-speed work belongs in
//! indexing/descriptor caching, not here.
//!
//! The accumulator records observed variants verbatim; unification (e.g.
//! Int/Real promotion) happens only when reporting, so no observation is
//! lost. Strictness policy will hook in at report/unify time, not during
//! observation. This module is format-agnostic — the extxyz driver lives in
//! [`crate::extxyz::infer_schema`].

use std::fmt;

use crate::model::{ColumnKind, Frame, Value};
use compact_str::CompactString;

/// The type and shape of a metadata value, without its data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    Real,
    Int,
    Bool,
    Str,
    RealArray(usize),
    IntArray(usize),
    BoolArray(usize),
    StrArray(usize),
}

impl ValueType {
    pub fn of(value: &Value) -> ValueType {
        match value {
            Value::Real(_) => ValueType::Real,
            Value::Int(_) => ValueType::Int,
            Value::Bool(_) => ValueType::Bool,
            Value::Str(_) => ValueType::Str,
            Value::RealArray(values) => ValueType::RealArray(values.len()),
            Value::IntArray(values) => ValueType::IntArray(values.len()),
            Value::BoolArray(values) => ValueType::BoolArray(values.len()),
            Value::StrArray(values) => ValueType::StrArray(values.len()),
        }
    }
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValueType::Real => write!(f, "Real"),
            ValueType::Int => write!(f, "Int"),
            ValueType::Bool => write!(f, "Bool"),
            ValueType::Str => write!(f, "Str"),
            ValueType::RealArray(n) => write!(f, "RealArray[{n}]"),
            ValueType::IntArray(n) => write!(f, "IntArray[{n}]"),
            ValueType::BoolArray(n) => write!(f, "BoolArray[{n}]"),
            ValueType::StrArray(n) => write!(f, "StrArray[{n}]"),
        }
    }
}

/// One observed `(kind, width)` combination for a column.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnVariant {
    pub kind: ColumnKind,
    pub width: usize,
    pub frames: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ColumnSchema {
    pub name: String,
    /// Observed variants in first-seen order; more than one means the column
    /// changed kind or width between frames.
    pub variants: Vec<ColumnVariant>,
    pub frames_present: usize,
}

impl ColumnSchema {
    /// The single `(kind, width)` every frame's column can be read as: the
    /// sole observed variant, or the Real that an Int/Real pair of equal
    /// width promotes to. `None` is a genuine conflict.
    pub fn unified(&self) -> Option<(ColumnKind, usize)> {
        match self.variants.as_slice() {
            [only] => Some((only.kind, only.width)),
            [a, b] if a.width == b.width => match (a.kind, b.kind) {
                (ColumnKind::Int, ColumnKind::Real) | (ColumnKind::Real, ColumnKind::Int) => {
                    Some((ColumnKind::Real, a.width))
                }
                _ => None,
            },
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetadataSchema {
    pub key: String,
    /// Observed `(type, frame count)` variants in first-seen order.
    pub variants: Vec<(ValueType, usize)>,
    pub frames_present: usize,
}

impl MetadataSchema {
    /// The single type every frame's value can be read as; same promotion
    /// rule as [`ColumnSchema::unified`].
    pub fn unified(&self) -> Option<ValueType> {
        match self.variants.as_slice() {
            [(only, _)] => Some(*only),
            [(a, _), (b, _)] => match (a, b) {
                (ValueType::Int, ValueType::Real) | (ValueType::Real, ValueType::Int) => {
                    Some(ValueType::Real)
                }
                (ValueType::IntArray(n), ValueType::RealArray(m))
                | (ValueType::RealArray(m), ValueType::IntArray(n))
                    if n == m =>
                {
                    Some(ValueType::RealArray(*m))
                }
                _ => None,
            },
            _ => None,
        }
    }
}

/// Accumulated structure of a dataset; build with repeated [`Schema::observe`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Schema {
    pub n_frames: usize,
    pub total_atoms: usize,
    pub min_atoms: Option<usize>,
    pub max_atoms: Option<usize>,
    /// Declared atom count per frame, in file order. Kept as the sample the
    /// distribution statistics (mean/median/std) are derived from, so the
    /// schema pass alone answers what `scan` answers — `min`/`max`/`total`
    /// stay accumulated to leave the rest of this type untouched.
    pub n_atoms: Vec<usize>,
    /// Per-atom columns, in first-seen order.
    pub columns: Vec<ColumnSchema>,
    /// Metadata keys, in first-seen order.
    pub metadata: Vec<MetadataSchema>,
}

impl Schema {
    pub fn observe(&mut self, frame: &Frame) {
        self.n_frames += 1;
        self.total_atoms += frame.n_atoms;
        self.min_atoms = Some(
            self.min_atoms
                .map_or(frame.n_atoms, |m| m.min(frame.n_atoms)),
        );
        self.max_atoms = Some(
            self.max_atoms
                .map_or(frame.n_atoms, |m| m.max(frame.n_atoms)),
        );
        self.n_atoms.push(frame.n_atoms);

        for column in &frame.columns {
            // Look up by index rather than holding the `find` borrow: the
            // not-found arm must push into the same Vec.
            let index = match self
                .columns
                .iter()
                .position(|entry| entry.name == column.name)
            {
                Some(index) => index,
                None => {
                    self.columns.push(ColumnSchema {
                        name: column.name.to_string(),
                        variants: Vec::new(),
                        frames_present: 0,
                    });
                    self.columns.len() - 1
                }
            };

            let entry = &mut self.columns[index];
            entry.frames_present += 1;

            let kind = column.data.kind();
            match entry
                .variants
                .iter_mut()
                .find(|variant| variant.kind == kind && variant.width == column.width)
            {
                Some(variant) => variant.frames += 1,
                None => entry.variants.push(ColumnVariant {
                    kind,
                    width: column.width,
                    frames: 1,
                }),
            }
        }

        // Collapse duplicate keys last-wins within the frame, matching the
        // dict semantics of `Frame`'s Python view, so a repeated key counts
        // once rather than pushing frames_present past n_frames (which would
        // wrongly read as inconsistent).
        let mut deduped: Vec<(&CompactString, &Value)> = Vec::with_capacity(frame.metadata.len());
        for (key, value) in &frame.metadata {
            match deduped.iter_mut().find(|(existing, _)| *existing == key) {
                Some(slot) => slot.1 = value,
                None => deduped.push((key, value)),
            }
        }

        for (key, value) in deduped {
            let index = match self
                .metadata
                .iter()
                .position(|entry| entry.key.as_str() == key.as_str())
            {
                Some(index) => index,
                None => {
                    self.metadata.push(MetadataSchema {
                        key: key.to_string(),
                        variants: Vec::new(),
                        frames_present: 0,
                    });
                    self.metadata.len() - 1
                }
            };

            let entry = &mut self.metadata[index];
            entry.frames_present += 1;

            let value_type = ValueType::of(value);
            match entry
                .variants
                .iter_mut()
                .find(|(existing, _)| *existing == value_type)
            {
                Some((_, frames)) => *frames += 1,
                None => entry.variants.push((value_type, 1)),
            }
        }
    }

    /// Strict consistency: every column and metadata key has exactly one
    /// observed variant and appears in every frame. Int/Real promotion does
    /// not count — a unifiable file is still inconsistent. Vacuously true
    /// for an empty file.
    pub fn is_consistent(&self) -> bool {
        let single_and_everywhere =
            |variants: usize, present: usize| variants == 1 && present == self.n_frames;

        self.columns
            .iter()
            .all(|column| single_and_everywhere(column.variants.len(), column.frames_present))
            && self
                .metadata
                .iter()
                .all(|entry| single_and_everywhere(entry.variants.len(), entry.frames_present))
    }
}

impl fmt::Display for Schema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} frames, {} atoms", self.n_frames, self.total_atoms)?;
        if let (Some(min), Some(max)) = (self.min_atoms, self.max_atoms) {
            write!(f, " (min {min}, max {max})")?;
        }
        writeln!(f)?;

        writeln!(f, "\nper-atom columns:")?;
        for column in &self.columns {
            write!(f, "  {}: ", column.name)?;
            for (i, variant) in column.variants.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(
                    f,
                    "{}:{} ({}/{} frames)",
                    variant.kind, variant.width, variant.frames, self.n_frames
                )?;
            }
            if column.variants.len() > 1 {
                match column.unified() {
                    Some((kind, width)) => write!(f, " (unifies to {kind}:{width})")?,
                    None => write!(f, " [inconsistent]")?,
                }
            }
            writeln!(f)?;
        }

        writeln!(f, "\nmetadata:")?;
        for entry in &self.metadata {
            write!(f, "  {}: ", entry.key)?;
            for (i, (value_type, frames)) in entry.variants.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{value_type} ({frames}/{} frames)", self.n_frames)?;
            }
            if entry.variants.len() > 1 {
                match entry.unified() {
                    Some(value_type) => write!(f, " (unifies to {value_type})")?,
                    None => write!(f, " [inconsistent]")?,
                }
            }
            writeln!(f)?;
        }

        Ok(())
    }
}
