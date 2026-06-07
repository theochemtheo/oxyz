use std::path::PathBuf;

use pyo3::{
    exceptions::{PyOSError, PyValueError},
    prelude::*,
    types::PyDict,
};

#[pyfunction]
fn read_first_frame<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Bound<'py, PyDict>> {
    let frame = atomflow_core::read_first_frame(path).map_err(extxyz_error_to_py)?;

    // PyO3 treats `Vec<u8>` as binary on python side, switch to u16 to avoid this
    let numbers: Vec<u16> = frame.numbers.into_iter().map(u16::from).collect();
    
    let data = PyDict::new(py);
    data.set_item("numbers", numbers)?;
    data.set_item("positions", vectors3_to_lists(frame.positions))?;
    data.set_item("forces", vectors3_to_lists(frame.forces))?;
    data.set_item("energy", frame.energy)?;
    data.set_item("cell", matrix3_to_lists(frame.cell))?;
    data.set_item("stress", frame.stress.to_vec())?;
    data.set_item("pbc", frame.pbc.to_vec())?;

    Ok(data)
}

fn extxyz_error_to_py(error: atomflow_core::ExtxyzError) -> PyErr {
    match error {
        atomflow_core::ExtxyzError::Io(error) => PyOSError::new_err(error.to_string()),
        error => PyValueError::new_err(error.to_string()),
    }
}

fn vectors3_to_lists(values: Vec<[f64; 3]>) -> Vec<Vec<f64>> {
    values.into_iter().map(|value| value.to_vec()).collect()
}

fn matrix3_to_lists(value: [[f64; 3]; 3]) -> Vec<Vec<f64>> {
    value.into_iter().map(|row| row.to_vec()).collect()
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_first_frame, m)?)?;
    Ok(())
}
