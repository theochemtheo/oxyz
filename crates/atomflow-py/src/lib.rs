use pyo3::prelude::*;

#[pyfunction]
fn version() -> &'static str {
    atomflow_core::version()
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
