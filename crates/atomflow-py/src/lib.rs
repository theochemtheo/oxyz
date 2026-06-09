use std::path::PathBuf;

use ndarray::Array2;
use numpy::{Element, IntoPyArray};
use pyo3::{
    exceptions::{PyOSError, PyValueError},
    prelude::*,
    types::{PyDict, PyList},
};

use atomflow_core::{Column, ColumnData, ExtxyzError, Frame, Value};

/// Read the first frame as `{"n_atoms": int, "columns": {...}, "metadata": {...}}`.
#[pyfunction]
fn read_first_frame<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Bound<'py, PyDict>> {
    let frame = atomflow_core::read_first_frame(path).map_err(extxyz_error_to_py)?;
    frame_to_pydict(py, frame)
}

/// Read every frame, as a list of per-frame dicts.
#[pyfunction]
fn read_frames<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Bound<'py, PyList>> {
    let frames = atomflow_core::read_frames(path).map_err(extxyz_error_to_py)?;

    let dicts = frames
        .into_iter()
        .map(|frame| frame_to_pydict(py, frame))
        .collect::<PyResult<Vec<_>>>()?;

    PyList::new(py, dicts)
}

/// Convert one frame to `{"n_atoms": int, "columns": {...}, "metadata": {...}}`.
///
/// Numeric and boolean columns become numpy arrays (2-D when width > 1);
/// string columns become `list[str]`. Both inner dicts preserve file order.
fn frame_to_pydict(py: Python<'_>, frame: Frame) -> PyResult<Bound<'_, PyDict>> {
    let data = PyDict::new(py);
    data.set_item("n_atoms", frame.n_atoms)?;

    let columns = PyDict::new(py);
    for column in frame.columns {
        let Column { name, width, data } = column;

        match data {
            ColumnData::Real(values) => {
                columns.set_item(name, array_from_flat(py, values, width)?)?;
            }
            ColumnData::Int(values) => {
                columns.set_item(name, array_from_flat(py, values, width)?)?;
            }
            ColumnData::Bool(values) => {
                columns.set_item(name, array_from_flat(py, values, width)?)?;
            }
            ColumnData::Str(values) => {
                if width == 1 {
                    columns.set_item(name, values)?;
                } else {
                    let rows: Vec<Vec<String>> =
                        values.chunks(width).map(<[String]>::to_vec).collect();
                    columns.set_item(name, rows)?;
                }
            }
        }
    }
    data.set_item("columns", columns)?;

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

fn extxyz_error_to_py(error: ExtxyzError) -> PyErr {
    let message = error.to_string();

    // Unwrap frame context to classify the underlying error.
    let mut inner = &error;
    while let ExtxyzError::InFrame { source, .. } = inner {
        inner = source;
    }

    match inner {
        ExtxyzError::Io(io_error) => PyOSError::new_err(io_error.to_string()),
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
    m.add_function(wrap_pyfunction!(read_first_frame, m)?)?;
    m.add_function(wrap_pyfunction!(read_frames, m)?)?;
    Ok(())
}
