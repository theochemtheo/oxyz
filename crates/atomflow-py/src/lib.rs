use std::path::PathBuf;

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::{
    exceptions::{PyOSError, PyValueError},
    prelude::*,
    types::PyDict,
};

#[pyfunction]
fn read_first_frame<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Bound<'py, PyDict>> {
    let frame = atomflow_core::read_first_frame(path).map_err(extxyz_error_to_py)?;

    let atomflow_core::Frame {
        numbers,
        positions,
        forces,
        energy,
        cell,
        stress,
        pbc,
    } = frame;

    let n_atoms = numbers.len();

    let data = PyDict::new(py);

    data.set_item("numbers", numbers.into_pyarray(py))?;
    data.set_item("positions", array2_from_flat(py, positions, n_atoms, 3)?)?;
    data.set_item("forces", array2_from_flat(py, forces, n_atoms, 3)?)?;
    data.set_item("energy", energy)?;
    data.set_item("cell", array2_from_flat(py, cell.to_vec(), 3, 3)?)?;
    data.set_item("stress", stress.to_vec().into_pyarray(py))?;
    data.set_item("pbc", pbc.to_vec().into_pyarray(py))?;

    Ok(data)
}

fn extxyz_error_to_py(error: atomflow_core::ExtxyzError) -> PyErr {
    match error {
        atomflow_core::ExtxyzError::Io(error) => PyOSError::new_err(error.to_string()),
        error => PyValueError::new_err(error.to_string()),
    }
}

fn array2_from_flat<'py>(
    py: Python<'py>,
    values: Vec<f64>,
    n_rows: usize,
    n_cols: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let array = Array2::from_shape_vec((n_rows, n_cols), values)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;

    Ok(array.into_pyarray(py))
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_first_frame, m)?)?;
    Ok(())
}
