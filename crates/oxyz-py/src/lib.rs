use std::{fs::File, io::BufReader, path::PathBuf};

use ndarray::Array2;
use numpy::{Element, IntoPyArray};
use pyo3::{
    exceptions::{PyIndexError, PyOSError, PyValueError},
    prelude::*,
    types::{PyDict, PyList},
};

use atomflow_core::{Batch, Column, ColumnData, ExtxyzError, Frame, Value};

/// Streaming iterator: one frame parsed and converted per `__next__`.
///
/// Owns the file handle; it closes when the object is dropped. The inner
/// iterator is fused — after an error or EOF it only raises StopIteration.
#[pyclass]
struct FrameIter {
    inner: atomflow_core::FrameIter<BufReader<File>>,
}

#[pymethods]
impl FrameIter {
    #[new]
    fn new(path: PathBuf) -> PyResult<Self> {
        let inner = atomflow_core::iter_frames(path).map_err(extxyz_error_to_py)?;
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
        .detach(|| atomflow_core::scan_index(path))
        .map_err(extxyz_error_to_py)?;

    let offsets: Vec<u64> = index.entries().iter().map(|entry| entry.offset).collect();
    let n_atoms: Vec<u64> = index
        .entries()
        .iter()
        .map(|entry| entry.n_atoms as u64)
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
    inner: atomflow_core::IndexedFrames,
}

#[pymethods]
impl IndexedFrames {
    #[new]
    fn new(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let inner = py
            .detach(|| atomflow_core::IndexedFrames::open(path))
            .map_err(extxyz_error_to_py)?;
        Ok(IndexedFrames { inner })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
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
    inner: atomflow_core::BatchIter<BufReader<File>>,
}

#[pymethods]
impl BatchIter {
    #[new]
    fn new(path: PathBuf, frames_per_batch: usize) -> PyResult<Self> {
        let inner =
            atomflow_core::iter_batches(path, frames_per_batch).map_err(extxyz_error_to_py)?;
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
    let frame = atomflow_core::read_first_frame(path).map_err(extxyz_error_to_py)?;
    frame_to_pydict(py, frame)
}

/// Read every frame, as a list of per-frame dicts.
///
/// `threads=None` parses on every core; `threads=1` is the exact serial
/// streaming read (no scan). Output and errors are identical either way.
#[pyfunction]
#[pyo3(signature = (path, threads=None))]
fn read_frames<'py>(
    py: Python<'py>,
    path: PathBuf,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyList>> {
    let frames = py
        .detach(|| match threads {
            Some(1) => atomflow_core::read_frames(&path),
            _ => atomflow_core::read_frames_parallel(&path, threads),
        })
        .map_err(extxyz_error_to_py)?;

    let dicts = frames
        .into_iter()
        .map(|frame| frame_to_pydict(py, frame))
        .collect::<PyResult<Vec<_>>>()?;

    PyList::new(py, dicts)
}

/// Infer the file's schema and return the human-readable report.
///
/// Provisional surface: text only, until the schema shape settles enough to
/// commit to structured Python access.
#[pyfunction]
fn infer_schema(path: PathBuf) -> PyResult<String> {
    let schema = atomflow_core::infer_schema(path).map_err(extxyz_error_to_py)?;
    Ok(schema.to_string())
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

fn extxyz_error_to_py(error: ExtxyzError) -> PyErr {
    let message = error.to_string();

    // Unwrap frame context to classify the underlying error.
    let mut inner = &error;
    while let ExtxyzError::InFrame { source, .. } = inner {
        inner = source;
    }

    match inner {
        ExtxyzError::Io(io_error) => PyOSError::new_err(io_error.to_string()),
        ExtxyzError::FrameOutOfRange { .. } => PyIndexError::new_err(message),
        _ => PyValueError::new_err(message),
    }
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
    m.add_function(wrap_pyfunction!(read_first_frame, m)?)?;
    m.add_function(wrap_pyfunction!(read_frames, m)?)?;
    m.add_function(wrap_pyfunction!(infer_schema, m)?)?;
    m.add_function(wrap_pyfunction!(scan, m)?)?;
    m.add_class::<FrameIter>()?;
    m.add_class::<IndexedFrames>()?;
    m.add_class::<BatchIter>()?;
    Ok(())
}
