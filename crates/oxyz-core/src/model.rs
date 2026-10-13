//! oxyz's internal data model: typed, columnar, format-agnostic.
//!
//! Columns are discovered while parsing, so kinds are enums rather than type
//! parameters.

use std::fmt;

use compact_str::CompactString;

/// The type of a column, without its data — the `R`/`I`/`L`/`S` axis alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnKind {
    Real,
    Int,
    Bool,
    Str,
}

impl fmt::Display for ColumnKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let letter = match self {
            ColumnKind::Real => 'R',
            ColumnKind::Int => 'I',
            ColumnKind::Bool => 'L',
            ColumnKind::Str => 'S',
        };
        write!(f, "{letter}")
    }
}

/// The values of one per-atom column, as a single dense buffer.
#[derive(Debug, Clone, PartialEq)]
pub enum ColumnData {
    /// extxyz kind `R`.
    Real(Vec<f64>),
    /// extxyz kind `I`.
    Int(Vec<i64>),
    /// extxyz kind `L`.
    Bool(Vec<bool>),
    /// extxyz kind `S`.
    Str(Vec<CompactString>),
}

impl ColumnData {
    pub fn kind(&self) -> ColumnKind {
        match self {
            ColumnData::Real(_) => ColumnKind::Real,
            ColumnData::Int(_) => ColumnKind::Int,
            ColumnData::Bool(_) => ColumnKind::Bool,
            ColumnData::Str(_) => ColumnKind::Str,
        }
    }

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

    pub fn as_str(&self) -> Option<&[CompactString]> {
        match self {
            ColumnData::Str(values) => Some(values),
            _ => None,
        }
    }
}

/// One per-atom column: a named, typed, width-strided buffer. The row count
/// lives with the owner.
#[derive(Debug, Clone, PartialEq)]
pub struct Column {
    /// Name as written in the Properties descriptor.
    pub name: CompactString,

    /// Values per atom (e.g. 3 for `pos:R:3`).
    pub width: usize,

    /// Flat row-major buffer; `data.len() == n_rows * width`.
    pub data: ColumnData,
}

/// One typed comment-line metadata value. Anything unrecognised is `Str` with
/// the raw text kept.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Real(f64),
    Int(i64),
    Bool(bool),
    Str(CompactString),
    RealArray(Vec<f64>),
    IntArray(Vec<i64>),
    BoolArray(Vec<bool>),
    StrArray(Vec<CompactString>),
    /// Row-major: element `(r, c)` is `data[r * cols + c]`.
    RealArray2D {
        rows: usize,
        cols: usize,
        data: Vec<f64>,
    },
    IntArray2D {
        rows: usize,
        cols: usize,
        data: Vec<i64>,
    },
    BoolArray2D {
        rows: usize,
        cols: usize,
        data: Vec<bool>,
    },
    StrArray2D {
        rows: usize,
        cols: usize,
        data: Vec<CompactString>,
    },
}

/// One parsed frame. `Vec`s rather than maps keep file order and duplicate keys.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub n_atoms: usize,

    /// Per-atom columns in file order; `data.len() == n_atoms * width`.
    pub columns: Vec<Column>,

    /// Frame-level metadata in file order, without `Properties`.
    pub metadata: Vec<(CompactString, Value)>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_kind_displays_as_its_extxyz_letter() {
        assert_eq!(ColumnKind::Real.to_string(), "R");
        assert_eq!(ColumnKind::Int.to_string(), "I");
        assert_eq!(ColumnKind::Bool.to_string(), "L");
        assert_eq!(ColumnKind::Str.to_string(), "S");
    }

    #[test]
    fn column_data_reports_kind_len_and_emptiness() {
        let columns = [
            (ColumnData::Real(vec![1.0, 2.0]), ColumnKind::Real, 2),
            (ColumnData::Int(vec![7]), ColumnKind::Int, 1),
            (
                ColumnData::Bool(vec![true, false, true]),
                ColumnKind::Bool,
                3,
            ),
            (ColumnData::Str(vec!["a".into()]), ColumnKind::Str, 1),
        ];
        for (data, kind, len) in columns {
            assert_eq!(data.kind(), kind);
            assert_eq!(data.len(), len);
            assert!(!data.is_empty());
        }
        assert!(ColumnData::Int(Vec::new()).is_empty());
    }

    #[test]
    fn typed_accessors_match_only_their_own_variant() {
        let real = ColumnData::Real(vec![0.5]);
        assert_eq!(real.as_real(), Some(&[0.5][..]));
        assert_eq!(real.as_int(), None);
        assert_eq!(real.as_bool(), None);
        assert_eq!(real.as_str(), None);

        assert_eq!(ColumnData::Int(vec![7]).as_int(), Some(&[7][..]));
        assert_eq!(ColumnData::Bool(vec![true]).as_bool(), Some(&[true][..]));

        let text = ColumnData::Str(vec!["a".into()]);
        assert_eq!(text.as_str(), Some(&["a".into()][..]));
        assert_eq!(text.as_real(), None);
    }

    #[test]
    #[allow(clippy::identity_op)] // `1 * cols + 0` spells out the (row, col) formula being tested
    fn two_d_array_values_are_flat_row_major_with_shape() {
        let v = Value::RealArray2D {
            rows: 2,
            cols: 2,
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let Value::RealArray2D { rows, cols, data } = &v else {
            panic!("wrong variant")
        };
        assert_eq!((*rows, *cols), (2, 2));
        assert_eq!(data.len(), rows * cols);
        assert_eq!(data[1 * cols + 0], 3.0); // row 1, col 0
    }
}
