use std::{fs::File, io::BufReader, path::PathBuf};

use ndarray::Array2;
use numpy::{Element, IntoPyArray, PyArray1};
use pyo3::{
    create_exception,
    exceptions::{PyIndexError, PyOSError, PyValueError},
    prelude::*,
    types::{PyDict, PyList, PyTuple},
};

use oxyz_core::schema::{ColumnSchema, MetadataSchema, Schema, ValueType};
use oxyz_core::{Batch, Column, ColumnData, ColumnKind, ExtxyzError, Frame, Value};

/// Streaming iterator: one frame parsed and converted per `__next__`.
///
/// Owns the file handle; it closes when the object is dropped. The inner
/// iterator is fused — after an error or EOF it only raises StopIteration.
#[pyclass]
struct FrameIter {
    inner: oxyz_core::FrameIter<BufReader<File>>,
}

#[pymethods]
impl FrameIter {
    #[new]
    fn new(path: PathBuf) -> PyResult<Self> {
        let inner = oxyz_core::iter_frames(path).map_err(extxyz_error_to_py)?;
        Ok(FrameIter { inner })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        // Parse with the interpreter detached (GIL released, or the
        // free-threaded equivalent) — only the numpy conversion needs it.
        match py.detach(|| self.inner.next()) {
            None => Ok(None),
            Some(Ok(frame)) => frame_to_pydict(py, frame).map(Some),
            Some(Err(error)) => Err(extxyz_error_to_py(error)),
        }
    }
}

/// Structurally scan the file: `{"offsets": ndarray, "n_atoms": ndarray}`.
#[pyfunction]
fn scan<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Bound<'py, PyDict>> {
    let index = py
        .detach(|| oxyz_core::scan_index(path))
        .map_err(extxyz_error_to_py)?;

    let offsets: Vec<u64> = index.entries().iter().map(|entry| entry.offset).collect();
    // Atom counts as isize (np.intp) so user arithmetic with them does not
    // promote to float64 the way an unsigned dtype would; byte offsets stay u64.
    let n_atoms: Vec<isize> = index
        .entries()
        .iter()
        .map(|entry| entry.n_atoms as isize)
        .collect();

    let data = PyDict::new(py);
    data.set_item("offsets", offsets.into_pyarray(py))?;
    data.set_item("n_atoms", n_atoms.into_pyarray(py))?;
    Ok(data)
}

/// Random-access reader: scans on construction, then `get(i)` seeks and
/// parses single frames in any order.
#[pyclass]
struct IndexedFrames {
    inner: oxyz_core::IndexedFrames,
}

#[pymethods]
impl IndexedFrames {
    #[new]
    fn new(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let inner = py
            .detach(|| oxyz_core::IndexedFrames::open(path))
            .map_err(extxyz_error_to_py)?;
        Ok(IndexedFrames { inner })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Declared atom count per frame, from the scan done at construction.
    /// Batch planning reads this instead of scanning the file again.
    #[getter]
    fn n_atoms<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<isize>> {
        let counts: Vec<isize> = self
            .inner
            .index()
            .entries()
            .iter()
            .map(|entry| entry.n_atoms as isize)
            .collect();
        counts.into_pyarray(py)
    }

    fn get<'py>(&mut self, py: Python<'py>, frame_index: usize) -> PyResult<Bound<'py, PyDict>> {
        let frame = py
            .detach(|| self.inner.get(frame_index))
            .map_err(extxyz_error_to_py)?;
        frame_to_pydict(py, frame)
    }

    #[pyo3(signature = (indices, threads=None))]
    fn get_batch<'py>(
        &mut self,
        py: Python<'py>,
        indices: Vec<usize>,
        threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let batch = py
            .detach(|| match threads {
                Some(1) => self.inner.get_batch(&indices),
                _ => self.inner.get_batch_parallel(&indices, threads),
            })
            .map_err(extxyz_error_to_py)?;
        batch_to_pydict(py, batch)
    }
}

/// Streaming batch iterator: `frames_per_batch` frames assembled per
/// `__next__`; the final batch may be smaller. Fused after errors.
#[pyclass]
struct BatchIter {
    inner: oxyz_core::BatchIter<BufReader<File>>,
}

#[pymethods]
impl BatchIter {
    #[new]
    fn new(path: PathBuf, frames_per_batch: usize) -> PyResult<Self> {
        let inner = oxyz_core::iter_batches(path, frames_per_batch).map_err(extxyz_error_to_py)?;
        Ok(BatchIter { inner })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        match py.detach(|| self.inner.next()) {
            None => Ok(None),
            Some(Ok(batch)) => batch_to_pydict(py, batch).map(Some),
            Some(Err(error)) => Err(extxyz_error_to_py(error)),
        }
    }
}

/// Read the first frame as `{"n_atoms": int, "columns": {...}, "metadata": {...}}`.
#[pyfunction]
fn read_first_frame<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Bound<'py, PyDict>> {
    let frame = oxyz_core::read_first_frame(path).map_err(extxyz_error_to_py)?;
    frame_to_pydict(py, frame)
}

/// Read every frame, as a list of per-frame dicts.
///
/// `threads=None` parses on every core; `threads=1` is the exact serial
/// streaming read. Either way the file is read in a single pass; output and
/// errors are identical.
#[pyfunction]
#[pyo3(signature = (path, threads=None))]
fn read_frames<'py>(
    py: Python<'py>,
    path: PathBuf,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyList>> {
    let frames = py
        .detach(|| match threads {
            Some(1) => oxyz_core::read_frames(&path),
            _ => oxyz_core::read_frames_parallel(&path, threads),
        })
        .map_err(extxyz_error_to_py)?;

    let dicts = frames
        .into_iter()
        .map(|frame| frame_to_pydict(py, frame))
        .collect::<PyResult<Vec<_>>>()?;

    PyList::new(py, dicts)
}

/// Gather the given frames (request order, repeats allowed) into one batch.
///
/// Single pass: the file is read once, and only as far as the last
/// requested frame — bytes past it are never inspected. `threads=None`
/// parses on every core; `threads=1` is fully serial. The batch and any
/// errors are identical either way.
#[pyfunction]
#[pyo3(signature = (path, indices, threads=None))]
fn read_batch<'py>(
    py: Python<'py>,
    path: PathBuf,
    indices: Vec<usize>,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let batch = py
        .detach(|| match threads {
            Some(1) => oxyz_core::read_batch(&path, &indices),
            _ => oxyz_core::read_batch_parallel(&path, &indices, threads),
        })
        .map_err(extxyz_error_to_py)?;
    batch_to_pydict(py, batch)
}

/// Infer the file's schema as one nested dict — counts, per-column and
/// per-key variant lists with unification verdicts, consistency, and the
/// rendered report — for the Python `Schema` dataclasses to wrap.
#[pyfunction]
fn infer_schema<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Bound<'py, PyDict>> {
    let schema = py
        .detach(|| oxyz_core::infer_schema(path))
        .map_err(extxyz_error_to_py)?;
    schema_to_pydict(py, &schema)
}

fn kind_name(kind: ColumnKind) -> &'static str {
    match kind {
        ColumnKind::Real => "Real",
        ColumnKind::Int => "Int",
        ColumnKind::Bool => "Bool",
        ColumnKind::Str => "Str",
    }
}

/// Split a metadata value type into the kind name and a numpy-style shape:
/// `()` for scalars, `(n,)` for arrays.
fn value_type_parts(value_type: ValueType) -> (&'static str, Vec<usize>) {
    match value_type {
        ValueType::Real => ("Real", vec![]),
        ValueType::Int => ("Int", vec![]),
        ValueType::Bool => ("Bool", vec![]),
        ValueType::Str => ("Str", vec![]),
        ValueType::RealArray(n) => ("Real", vec![n]),
        ValueType::IntArray(n) => ("Int", vec![n]),
        ValueType::BoolArray(n) => ("Bool", vec![n]),
        ValueType::StrArray(n) => ("Str", vec![n]),
    }
}

fn schema_to_pydict<'py>(py: Python<'py>, schema: &Schema) -> PyResult<Bound<'py, PyDict>> {
    let data = PyDict::new(py);
    data.set_item("n_frames", schema.n_frames)?;
    data.set_item("total_atoms", schema.total_atoms)?;
    data.set_item("min_atoms", schema.min_atoms)?;
    data.set_item("max_atoms", schema.max_atoms)?;
    // intp (isize), matching scan()'s FrameIndex.n_atoms — see the note there.
    let n_atoms: Vec<isize> = schema.n_atoms.iter().map(|&n| n as isize).collect();
    data.set_item("n_atoms", n_atoms.into_pyarray(py))?;
    data.set_item("is_consistent", schema.is_consistent())?;
    data.set_item("report", schema.to_string())?;

    let columns = PyList::empty(py);
    for column in &schema.columns {
        columns.append(column_schema_to_pydict(py, column)?)?;
    }
    data.set_item("columns", columns)?;

    let metadata = PyList::empty(py);
    for entry in &schema.metadata {
        metadata.append(metadata_schema_to_pydict(py, entry)?)?;
    }
    data.set_item("metadata", metadata)?;

    Ok(data)
}

fn column_schema_to_pydict<'py>(
    py: Python<'py>,
    column: &ColumnSchema,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("name", &column.name)?;
    dict.set_item("frames_present", column.frames_present)?;

    let variants = PyList::empty(py);
    for variant in &column.variants {
        let entry = PyDict::new(py);
        entry.set_item("kind", kind_name(variant.kind))?;
        entry.set_item("width", variant.width)?;
        entry.set_item("frames", variant.frames)?;
        variants.append(entry)?;
    }
    dict.set_item("variants", variants)?;

    let unified = column
        .unified()
        .map(|(kind, width)| (kind_name(kind), width));
    dict.set_item("unified", unified)?;

    Ok(dict)
}

fn metadata_schema_to_pydict<'py>(
    py: Python<'py>,
    entry: &MetadataSchema,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("key", &entry.key)?;
    dict.set_item("frames_present", entry.frames_present)?;

    let variants = PyList::empty(py);
    for &(value_type, frames) in &entry.variants {
        let (kind, shape) = value_type_parts(value_type);
        let variant = PyDict::new(py);
        variant.set_item("kind", kind)?;
        variant.set_item("shape", PyTuple::new(py, shape)?)?;
        variant.set_item("frames", frames)?;
        variants.append(variant)?;
    }
    dict.set_item("variants", variants)?;

    match entry.unified() {
        Some(value_type) => {
            let (kind, shape) = value_type_parts(value_type);
            dict.set_item("unified", (kind, PyTuple::new(py, shape)?))?;
        }
        None => dict.set_item("unified", py.None())?,
    }

    Ok(dict)
}

/// Convert one frame to `{"n_atoms": int, "columns": {...}, "metadata": {...}}`.
///
/// Numeric and boolean columns become numpy arrays (2-D when width > 1);
/// string columns become `list[str]`. Both inner dicts preserve file order.
fn frame_to_pydict(py: Python<'_>, frame: Frame) -> PyResult<Bound<'_, PyDict>> {
    let data = PyDict::new(py);
    data.set_item("n_atoms", frame.n_atoms)?;
    data.set_item("columns", columns_to_pydict(py, frame.columns)?)?;

    let metadata = PyDict::new(py);
    for (key, value) in frame.metadata {
        match value {
            Value::Real(x) => metadata.set_item(key, x)?,
            Value::Int(x) => metadata.set_item(key, x)?,
            Value::Bool(x) => metadata.set_item(key, x)?,
            Value::Str(x) => metadata.set_item(key, x)?,
            Value::RealArray(values) => metadata.set_item(key, values.into_pyarray(py))?,
            Value::IntArray(values) => metadata.set_item(key, values.into_pyarray(py))?,
            Value::BoolArray(values) => metadata.set_item(key, values.into_pyarray(py))?,
            Value::StrArray(values) => metadata.set_item(key, values)?,
        }
    }
    data.set_item("metadata", metadata)?;

    Ok(data)
}

/// Convert a batch to `{"offsets": ndarray, "columns": {...}, "metadata":
/// {...}}` — columns atom-major, metadata frame-major, both as dense arrays
/// (string columns as lists).
fn batch_to_pydict(py: Python<'_>, batch: Batch) -> PyResult<Bound<'_, PyDict>> {
    let data = PyDict::new(py);
    let offsets: Vec<i64> = batch.offsets.iter().map(|&offset| offset as i64).collect();
    data.set_item("offsets", offsets.into_pyarray(py))?;
    data.set_item("columns", columns_to_pydict(py, batch.columns)?)?;
    data.set_item("metadata", columns_to_pydict(py, batch.metadata)?)?;
    Ok(data)
}

/// Numeric and boolean columns become numpy arrays (2-D when width > 1);
/// string columns become `list[str]` (nested when width > 1). Preserves
/// order.
fn columns_to_pydict(py: Python<'_>, columns: Vec<Column>) -> PyResult<Bound<'_, PyDict>> {
    let dict = PyDict::new(py);
    for column in columns {
        let Column { name, width, data } = column;

        match data {
            ColumnData::Real(values) => {
                dict.set_item(name, array_from_flat(py, values, width)?)?;
            }
            ColumnData::Int(values) => {
                dict.set_item(name, array_from_flat(py, values, width)?)?;
            }
            ColumnData::Bool(values) => {
                dict.set_item(name, array_from_flat(py, values, width)?)?;
            }
            ColumnData::Str(values) => {
                if width == 1 {
                    dict.set_item(name, values)?;
                } else {
                    let rows: Vec<Vec<String>> =
                        values.chunks(width).map(<[String]>::to_vec).collect();
                    dict.set_item(name, rows)?;
                }
            }
        }
    }
    Ok(dict)
}

create_exception!(
    _rust,
    ParseError,
    PyValueError,
    "Raised when extxyz content cannot be parsed.\n\n\
     A `ValueError` subclass. Carries the location of the offending input as\n\
     attributes — `frame_index`, `line_number`, `column` — each `None` when\n\
     the parser cannot pin that dimension down, so callers can find the bad\n\
     frame without parsing the message string."
);

fn extxyz_error_to_py(error: ExtxyzError) -> PyErr {
    // Unwrap frame context to classify the underlying error.
    let mut inner = &error;
    while let ExtxyzError::InFrame { source, .. } = inner {
        inner = source;
    }

    // I/O and out-of-range keep their natural stdlib exception types; the
    // rest are content errors the caller may want to locate structurally.
    match inner {
        ExtxyzError::Io(io_error) => return PyOSError::new_err(io_error.to_string()),
        ExtxyzError::FrameOutOfRange { .. } => return PyIndexError::new_err(error.to_string()),
        _ => {}
    }

    let frame_index = error.frame_index();
    let line_number = error.line_number();
    let column = error.column().map(str::to_owned);
    let err = ParseError::new_err(error.to_string());
    Python::attach(|py| {
        // Set every field so access is uniform; instance values shadow the
        // `None` class-level defaults registered in the module init.
        let value = err.value(py);
        let _ = value.setattr("frame_index", frame_index);
        let _ = value.setattr("line_number", line_number);
        let _ = value.setattr("column", column);
    });
    err
}

/// Turn a flat width-strided buffer into a 1-D (width == 1) or 2-D numpy
/// array.
fn array_from_flat<T: Element>(
    py: Python<'_>,
    values: Vec<T>,
    width: usize,
) -> PyResult<Bound<'_, PyAny>> {
    if width == 1 {
        return Ok(values.into_pyarray(py).into_any());
    }

    // The parser upholds `values.len() == n_rows * width`; keep the error
    // path anyway rather than unwrap at the Python boundary.
    let n_rows = values.len() / width;
    let array = Array2::from_shape_vec((n_rows, width), values)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;

    Ok(array.into_pyarray(py).into_any())
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Lets Python (e.g. the benchmark harness) refuse debug builds.
    m.add(
        "__build_profile__",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
    )?;
    // Class-level `None` defaults so the location attributes always resolve,
    // even on a `ParseError` a user constructs directly.
    let parse_error = m.py().get_type::<ParseError>();
    parse_error.setattr("frame_index", m.py().None())?;
    parse_error.setattr("line_number", m.py().None())?;
    parse_error.setattr("column", m.py().None())?;
    m.add("ParseError", parse_error)?;

    m.add_function(wrap_pyfunction!(read_first_frame, m)?)?;
    m.add_function(wrap_pyfunction!(read_frames, m)?)?;
    m.add_function(wrap_pyfunction!(read_batch, m)?)?;
    m.add_function(wrap_pyfunction!(infer_schema, m)?)?;
    m.add_function(wrap_pyfunction!(scan, m)?)?;
    m.add_class::<FrameIter>()?;
    m.add_class::<IndexedFrames>()?;
    m.add_class::<BatchIter>()?;
    Ok(())
}
