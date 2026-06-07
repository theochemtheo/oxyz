use std::{
    collections::HashMap,
    fs::File,
    io::{self, BufRead, BufReader},
    path::Path,
};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    /// Atomic numbers, shape `[n_atoms]`.
    pub numbers: Vec<u8>,

    /// Cartesian positions in row-major order, shape `[n_atoms, 3]`.
    pub positions: Vec<f64>,

    /// Cartesian forces in row-major order, shape `[n_atoms, 3]`.
    pub forces: Vec<f64>,

    pub energy: f64,

    /// Cell matrix in row-major order, shape `[3, 3]`.
    ///
    /// Rows are lattice vectors. Standard extxyz `Lattice` metadata is decoded
    /// from Fortran order into this representation.
    pub cell: [f64; 9],

    /// Stress in 6-component Voigt-style order.
    pub stress: [f64; 6],

    /// Periodic boundary conditions along the three lattice directions.
    pub pbc: [bool; 3],
}

#[derive(Debug, Error)]
pub enum ExtxyzError {
    #[error("I/O error")]
    Io(#[from] io::Error),

    #[error("missing {0} line")]
    MissingLine(&'static str),

    #[error("invalid atom count line: {line:?}")]
    InvalidAtomCount { line: String },

    #[error("invalid comment metadata near byte {index}")]
    InvalidMetadata { index: usize },

    #[error("missing metadata key {key:?}")]
    MissingMetadata { key: &'static str },

    #[error("unsupported Properties descriptor: {properties:?}")]
    UnsupportedProperties { properties: String },

    #[error("metadata field {field:?} has {actual} values; expected {expected}")]
    WrongValueCount {
        field: &'static str,
        expected: usize,
        actual: usize,
    },

    #[error("invalid float in field {field:?}: {value:?}")]
    InvalidFloat { field: &'static str, value: String },

    #[error("invalid bool in field {field:?}: {value:?}")]
    InvalidBool { field: &'static str, value: String },

    #[error("unsupported atomic symbol {symbol:?}")]
    UnsupportedElement { symbol: String },

    #[error("atom line {line_number} has {actual} columns; expected 7")]
    WrongAtomColumnCount { line_number: usize, actual: usize },
}

pub type Result<T> = std::result::Result<T, ExtxyzError>;

pub fn read_first_frame(path: impl AsRef<Path>) -> Result<Frame> {
    let file = File::open(path)?;
    let mut lines = BufReader::new(file).lines();

    let atom_count_line = next_line(&mut lines, "atom count")?;
    let atom_count =
        atom_count_line
            .trim()
            .parse::<usize>()
            .map_err(|_| ExtxyzError::InvalidAtomCount {
                line: atom_count_line,
            })?;

    let comment = next_line(&mut lines, "comment")?;
    let metadata = parse_comment_metadata(&comment)?;

    let properties = required_metadata(&metadata, "Properties")?;
    if properties != "species:S:1:pos:R:3:forces:R:3" {
        return Err(ExtxyzError::UnsupportedProperties {
            properties: properties.to_owned(),
        });
    }

    let cell = parse_lattice(required_metadata(&metadata, "Lattice")?)?;
    let energy = parse_one_f64(required_metadata(&metadata, "energy")?, "energy")?;
    let stress = parse_fixed_f64::<6>(required_metadata(&metadata, "stress")?, "stress")?;
    let pbc = parse_fixed_bool::<3>(required_metadata(&metadata, "pbc")?, "pbc")?;

    let mut numbers = Vec::with_capacity(atom_count);
    let mut positions = Vec::with_capacity(atom_count * 3);
    let mut forces = Vec::with_capacity(atom_count * 3);

    for atom_index in 0..atom_count {
        let line_number = atom_index + 3;
        let line = next_line(&mut lines, "atom")?;
        let columns: Vec<&str> = line.split_whitespace().collect();

        if columns.len() != 7 {
            return Err(ExtxyzError::WrongAtomColumnCount {
                line_number,
                actual: columns.len(),
            });
        }

        numbers.push(atomic_number(columns[0])?);

        positions.extend_from_slice(&[
            parse_f64(columns[1], "positions")?,
            parse_f64(columns[2], "positions")?,
            parse_f64(columns[3], "positions")?,
        ]);

        forces.extend_from_slice(&[
            parse_f64(columns[4], "forces")?,
            parse_f64(columns[5], "forces")?,
            parse_f64(columns[6], "forces")?,
        ]);
    }

    Ok(Frame {
        numbers,
        positions,
        forces,
        energy,
        cell,
        stress,
        pbc,
    })
}

fn next_line(
    lines: &mut impl Iterator<Item = io::Result<String>>,
    label: &'static str,
) -> Result<String> {
    lines
        .next()
        .transpose()?
        .ok_or(ExtxyzError::MissingLine(label))
}

fn parse_comment_metadata(comment: &str) -> Result<HashMap<String, String>> {
    let bytes = comment.as_bytes();
    let mut metadata = HashMap::new();
    let mut i = 0;

    while i < bytes.len() {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }

        if i == bytes.len() {
            break;
        }

        let key_start = i;

        while i < bytes.len() && bytes[i] != b'=' && !bytes[i].is_ascii_whitespace() {
            i += 1;
        }

        if i == key_start || i >= bytes.len() || bytes[i] != b'=' {
            return Err(ExtxyzError::InvalidMetadata { index: i });
        }

        let key = slice_comment(comment, key_start, i)?;
        i += 1; // skip '='

        if i >= bytes.len() {
            return Err(ExtxyzError::InvalidMetadata { index: i });
        }

        let value = if bytes[i] == b'"' {
            i += 1; // skip opening quote
            let value_start = i;

            while i < bytes.len() && bytes[i] != b'"' {
                i += 1;
            }

            if i >= bytes.len() {
                return Err(ExtxyzError::InvalidMetadata {
                    index: value_start.saturating_sub(1),
                });
            }

            let value = slice_comment(comment, value_start, i)?;
            i += 1; // skip closing quote
            value
        } else {
            let value_start = i;

            while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
                i += 1;
            }

            if i == value_start {
                return Err(ExtxyzError::InvalidMetadata { index: i });
            }

            slice_comment(comment, value_start, i)?
        };

        metadata.insert(key.to_owned(), value.to_owned());
    }

    Ok(metadata)
}

fn slice_comment(comment: &str, start: usize, end: usize) -> Result<&str> {
    comment
        .get(start..end)
        .ok_or(ExtxyzError::InvalidMetadata { index: start })
}

fn required_metadata<'a>(
    metadata: &'a HashMap<String, String>,
    key: &'static str,
) -> Result<&'a str> {
    metadata
        .get(key)
        .map(String::as_str)
        .ok_or(ExtxyzError::MissingMetadata { key })
}

fn parse_one_f64(value: &str, field: &'static str) -> Result<f64> {
    let values = parse_fixed_f64::<1>(value, field)?;
    Ok(values[0])
}

fn parse_fixed_f64<const N: usize>(value: &str, field: &'static str) -> Result<[f64; N]> {
    let parts: Vec<&str> = value.split_whitespace().collect();

    if parts.len() != N {
        return Err(ExtxyzError::WrongValueCount {
            field,
            expected: N,
            actual: parts.len(),
        });
    }

    let mut output = [0.0; N];

    for (slot, part) in output.iter_mut().zip(parts) {
        *slot = parse_f64(part, field)?;
    }

    Ok(output)
}

fn parse_fixed_bool<const N: usize>(value: &str, field: &'static str) -> Result<[bool; N]> {
    let parts: Vec<&str> = value.split_whitespace().collect();

    if parts.len() != N {
        return Err(ExtxyzError::WrongValueCount {
            field,
            expected: N,
            actual: parts.len(),
        });
    }

    let mut output = [false; N];

    for (slot, part) in output.iter_mut().zip(parts) {
        *slot = parse_bool(part, field)?;
    }

    Ok(output)
}

fn parse_f64(value: &str, field: &'static str) -> Result<f64> {
    value.parse::<f64>().map_err(|_| ExtxyzError::InvalidFloat {
        field,
        value: value.to_owned(),
    })
}

fn parse_bool(value: &str, field: &'static str) -> Result<bool> {
    match value {
        "T" | "True" | "true" | "1" => Ok(true),
        "F" | "False" | "false" | "0" => Ok(false),
        _ => Err(ExtxyzError::InvalidBool {
            field,
            value: value.to_owned(),
        }),
    }
}

fn atomic_number(symbol: &str) -> Result<u8> {
    match symbol {
        "H" => Ok(1),
        _ => Err(ExtxyzError::UnsupportedElement {
            symbol: symbol.to_owned(),
        }),
    }
}

fn parse_lattice(value: &str) -> Result<[f64; 9]> {
    let values = parse_fixed_f64::<9>(value, "Lattice")?;

    // extxyz/ASE stores Lattice in Fortran order.
    // atomflow stores and exposes cells in row-major order with lattice vectors as rows.
    Ok([
        values[0], values[3], values[6], values[1], values[4], values[7], values[2], values[5],
        values[8],
    ])
}
