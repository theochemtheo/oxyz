//! atomflow's internal data model: typed, columnar, format-agnostic.
//!
//! Runtime-typed on purpose: which columns a file has, their types, and
//! their widths are discovered while parsing, so they cannot be compile-time
//! type parameters. extxyz's closed set of column kinds (`R`/`I`/`S`/`L`)
//! maps onto enums instead.
//!
//! Parsing lives in [`crate::extxyz`]; a future `Batch` (concatenated
//! columns across frames) is expected to reuse this vocabulary.

/// The values of one per-atom column, stored as a single dense buffer.
///
/// The enum wraps whole columns, not cells: one tag check per column access,
/// with contiguous data inside — the layout numpy and Arrow want.
#[derive(Debug, Clone, PartialEq)]
pub enum ColumnData {
    /// extxyz kind `R`.
    Real(Vec<f64>),
    /// extxyz kind `I`.
    Int(Vec<i64>),
    /// extxyz kind `L`.
    Bool(Vec<bool>),
    /// extxyz kind `S`.
    Str(Vec<String>),
}

impl ColumnData {
    pub fn len(&self) -> usize {
        match self {
            ColumnData::Real(values) => values.len(),
            ColumnData::Int(values) => values.len(),
            ColumnData::Bool(values) => values.len(),
            ColumnData::Str(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_real(&self) -> Option<&[f64]> {
        match self {
            ColumnData::Real(values) => Some(values),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<&[i64]> {
        match self {
            ColumnData::Int(values) => Some(values),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<&[bool]> {
        match self {
            ColumnData::Bool(values) => Some(values),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&[String]> {
        match self {
            ColumnData::Str(values) => Some(values),
            _ => None,
        }
    }
}

/// One per-atom column: a named, typed, width-strided buffer.
///
/// Deliberately frame-agnostic: row count lives with the owner, so a future
/// `Batch` can concatenate buffers across frames.
#[derive(Debug, Clone, PartialEq)]
pub struct Column {
    /// Name exactly as written in the Properties descriptor; aliasing
    /// (`force` vs `forces`) is a later layer's job.
    pub name: String,

    /// Values per atom (e.g. 3 for `pos:R:3`).
    pub width: usize,

    /// Flat row-major buffer; `data.len() == n_rows * width`.
    pub data: ColumnData,
}

/// One typed comment-line metadata value.
///
/// Values that fit nothing more specific fall back to `Str` with the raw
/// text preserved. Arrays stay in as-written order — `Lattice` is not
/// reordered or reshaped here.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Real(f64),
    Int(i64),
    Bool(bool),
    Str(String),
    RealArray(Vec<f64>),
    IntArray(Vec<i64>),
    BoolArray(Vec<bool>),
    StrArray(Vec<String>),
}

/// One parsed frame: per-atom columns plus frame-level metadata.
///
/// Both collections are `Vec`s, not maps: file order is preserved and
/// duplicate keys survive. Fields are public while the model is settling.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub n_atoms: usize,

    /// Per-atom columns, in file order; each upholds
    /// `data.len() == n_atoms * width`.
    pub columns: Vec<Column>,

    /// Frame-level metadata, in file order. `Properties` is consumed into
    /// `columns` and not repeated here.
    pub metadata: Vec<(String, Value)>,
}

impl Frame {
    /// Find a column by its as-written name; first match wins.
    pub fn column(&self, name: &str) -> Option<&Column> {
        self.columns.iter().find(|column| column.name == name)
    }

    /// Find a metadata value by key; first match wins.
    pub fn metadata_value(&self, key: &str) -> Option<&Value> {
        self.metadata
            .iter()
            .find(|(existing, _)| existing == key)
            .map(|(_, value)| value)
    }
}
