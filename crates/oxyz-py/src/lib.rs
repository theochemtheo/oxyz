use std::io::{Cursor, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use ndarray::Array2;
use numpy::{Element, IntoPyArray, PyArray1, PyArrayDyn, PyArrayMethods};
use pyo3::{
    create_exception,
    exceptions::{PyIndexError, PyOSError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyDict, PyList, PyString, PyTuple},
};

use oxyz_core::project::{
    Axis, Deviation, DeviationKind, Fill, PlanColumn, PlanMetadata, Projected, ProjectionPlan,
    project_frame,
};
use oxyz_core::schema::{ColumnSchema, MetadataSchema, Schema, ValueType};
use oxyz_core::{
    Batch, ByteSource, Codec, Column, ColumnData, ColumnKind, Compression, DecodedReader,
    ExtxyzError, Frame, FrameSink, TarInner, Value, detect_codec_name, open_decoded, wrap_stream,
    wrap_tar, wrap_zip, write_frames, write_frames_parallel,
};

/// Map the Python `compression` string to the core selector; tar spellings
/// match `Codec::name()`.
fn parse_compression(name: &str) -> PyResult<Compression> {
    Ok(match name {
        "infer" => Compression::Infer,
        "none" => Compression::None,
        "gzip" => Compression::Gzip,
        "zstd" => Compression::Zstd,
        "zip" => Compression::Zip,
        "tar" => Compression::Tar,
        "tar.gz" => Compression::TarGzip,
        "tar.zst" => Compression::TarZstd,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown compression {other:?}; expected one of: infer, none, gzip, \
                 zstd, zip, tar, tar.gz, tar.zst"
            )));
        }
    })
}

/// Open a streaming, decoded reader for the given path and options.
fn open_reader(path: &Path, compression: &str, member: Option<&str>) -> PyResult<DecodedReader> {
    open_decoded(path, parse_compression(compression)?, member).map_err(extxyz_error_to_py)
}

/// Streaming iterator: one frame parsed and converted per `__next__`. Owns
/// the file handle; fused after an error or EOF.
#[pyclass]
struct FrameIter {
    inner: oxyz_core::FrameIter<DecodedReader>,
}

#[pymethods]
impl FrameIter {
    #[new]
    #[pyo3(signature = (path, compression="infer", member=None))]
    fn new(path: PathBuf, compression: &str, member: Option<String>) -> PyResult<Self> {
        let reader = open_reader(&path, compression, member.as_deref())?;
        let inner = oxyz_core::iter_frames_from(reader).map_err(extxyz_error_to_py)?;
        Ok(FrameIter { inner })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        // Only the numpy conversion needs the interpreter.
        match py.detach(|| self.inner.next()) {
            None => Ok(None),
            Some(Ok(frame)) => frame_to_pydict(py, frame).map(Some),
            Some(Err(error)) => Err(extxyz_error_to_py(error)),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (source, codec, member=None))]
    fn from_reader(
        source: Bound<'_, PyAny>,
        codec: &str,
        member: Option<String>,
    ) -> PyResult<Self> {
        let reader = build_decoded(&source, codec, member.as_deref())?;
        let inner = oxyz_core::iter_frames_from(reader).map_err(extxyz_error_to_py)?;
        Ok(FrameIter { inner })
    }
}

/// Structurally scan the file: `{"offsets", "n_atoms"}` arrays, plus
/// `"volumes"` (`NaN` without a `Lattice`) when `with_volume=True`.
#[pyfunction]
#[pyo3(signature = (path, with_volume=false, compression="infer", member=None))]
fn scan<'py>(
    py: Python<'py>,
    path: PathBuf,
    with_volume: bool,
    compression: &str,
    member: Option<String>,
) -> PyResult<Bound<'py, PyDict>> {
    let reader = open_reader(&path, compression, member.as_deref())?;
    let index = py
        .detach(move || {
            if with_volume {
                oxyz_core::scan_frames_with_volume(reader)
            } else {
                oxyz_core::scan_frames(reader)
            }
        })
        .map_err(extxyz_error_to_py)?;
    scan_index_to_pydict(py, &index)
}

fn scan_index_to_pydict<'py>(
    py: Python<'py>,
    index: &oxyz_core::index::FrameIndex,
) -> PyResult<Bound<'py, PyDict>> {
    let offsets: Vec<u64> = index.entries().iter().map(|entry| entry.offset).collect();
    // isize (np.intp): an unsigned dtype would promote user arithmetic to float64.
    let n_atoms: Vec<isize> = index
        .entries()
        .iter()
        .map(|entry| entry.n_atoms as isize)
        .collect();

    let data = PyDict::new(py);
    data.set_item("offsets", offsets.into_pyarray(py))?;
    data.set_item("n_atoms", n_atoms.into_pyarray(py))?;
    if let Some(volumes) = index.volumes() {
        data.set_item("volumes", volumes.to_vec().into_pyarray(py))?;
    }
    Ok(data)
}

#[pyfunction]
#[pyo3(signature = (source, codec, with_volume=false, member=None))]
fn scan_reader<'py>(
    py: Python<'py>,
    source: Bound<'py, PyAny>,
    codec: &str,
    with_volume: bool,
    member: Option<String>,
) -> PyResult<Bound<'py, PyDict>> {
    let reader = build_decoded(&source, codec, member.as_deref())?;
    let index = py
        .detach(move || {
            if with_volume {
                oxyz_core::scan_frames_with_volume(reader)
            } else {
                oxyz_core::scan_frames(reader)
            }
        })
        .map_err(extxyz_error_to_py)?;
    scan_index_to_pydict(py, &index)
}

#[pyfunction]
#[pyo3(signature = (source, codec, member=None))]
fn infer_schema_reader<'py>(
    py: Python<'py>,
    source: Bound<'py, PyAny>,
    codec: &str,
    member: Option<String>,
) -> PyResult<Bound<'py, PyDict>> {
    let reader = build_decoded(&source, codec, member.as_deref())?;
    let schema = py
        .detach(move || oxyz_core::infer_schema_from(reader))
        .map_err(extxyz_error_to_py)?;
    schema_to_pydict(py, &schema)
}

#[pyfunction]
#[pyo3(signature = (source, codec, indices=None, threads=None, member=None))]
fn read_batch_reader<'py>(
    py: Python<'py>,
    source: Bound<'py, PyAny>,
    codec: &str,
    indices: Option<Vec<usize>>,
    threads: Option<usize>,
    member: Option<String>,
) -> PyResult<Bound<'py, PyDict>> {
    let reader = build_decoded(&source, codec, member.as_deref())?;
    let batch = py
        .detach(move || match (indices, threads) {
            (None, Some(1)) => oxyz_core::read_all_batch_from(reader),
            (None, _) => oxyz_core::read_all_batch_parallel_from(reader, threads),
            (Some(indices), Some(1)) => oxyz_core::read_batch_from(reader, &indices),
            (Some(indices), _) => oxyz_core::read_batch_parallel_from(reader, &indices, threads),
        })
        .map_err(extxyz_error_to_py)?;
    batch_to_pydict(py, batch)
}

/// Random-access reader: scans on construction, then `get(i)` parses one frame.
#[pyclass]
struct IndexedFrames {
    inner: oxyz_core::IndexedFrames,
}

#[pymethods]
impl IndexedFrames {
    #[new]
    #[pyo3(signature = (path, with_volume=false))]
    fn new(py: Python<'_>, path: PathBuf, with_volume: bool) -> PyResult<Self> {
        let inner = py
            .detach(|| {
                if with_volume {
                    oxyz_core::IndexedFrames::open_with_volume(path)
                } else {
                    oxyz_core::IndexedFrames::open(path)
                }
            })
            .map_err(extxyz_error_to_py)?;
        Ok(IndexedFrames { inner })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Per-frame `|det(Lattice)|`, or `None` unless opened `with_volume`. `NaN`
    /// for a frame with no `Lattice`.
    #[getter]
    fn volumes<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .index()
            .volumes()
            .map(|volumes| volumes.to_vec().into_pyarray(py))
    }

    /// Declared atom count per frame, from the construction scan.
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

    #[pyo3(signature = (indices, plan, threads=None))]
    fn get_batch_projected<'py>(
        &mut self,
        py: Python<'py>,
        indices: Vec<usize>,
        plan: &Bound<'py, PyTuple>,
        threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let plan = build_plan(plan)?;
        let projected = py
            .detach(|| match threads {
                Some(1) => self.inner.get_batch_projected(&indices, &plan),
                _ => self
                    .inner
                    .get_batch_projected_parallel(&indices, threads, &plan),
            })
            .map_err(extxyz_error_to_py)?;
        projected_batch_to_py(py, projected)
    }
}

/// Streaming batch iterator: `frames_per_batch` frames assembled per
/// `__next__`; the final batch may be smaller. Fused after errors.
#[pyclass]
struct BatchIter {
    inner: oxyz_core::BatchIter<DecodedReader>,
}

#[pymethods]
impl BatchIter {
    #[new]
    #[pyo3(signature = (path, frames_per_batch, compression="infer", member=None))]
    fn new(
        path: PathBuf,
        frames_per_batch: usize,
        compression: &str,
        member: Option<String>,
    ) -> PyResult<Self> {
        let reader = open_reader(&path, compression, member.as_deref())?;
        let inner =
            oxyz_core::iter_batches_from(reader, frames_per_batch).map_err(extxyz_error_to_py)?;
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

    #[staticmethod]
    #[pyo3(signature = (source, frames_per_batch, codec, member=None))]
    fn from_reader(
        source: Bound<'_, PyAny>,
        frames_per_batch: usize,
        codec: &str,
        member: Option<String>,
    ) -> PyResult<Self> {
        let reader = build_decoded(&source, codec, member.as_deref())?;
        let inner =
            oxyz_core::iter_batches_from(reader, frames_per_batch).map_err(extxyz_error_to_py)?;
        Ok(BatchIter { inner })
    }
}

/// Read the first frame as `{"n_atoms": int, "columns": {...}, "metadata": {...}}`.
#[pyfunction]
#[pyo3(signature = (path, compression="infer", member=None))]
fn read_first_frame<'py>(
    py: Python<'py>,
    path: PathBuf,
    compression: &str,
    member: Option<String>,
) -> PyResult<Bound<'py, PyDict>> {
    let reader = open_reader(&path, compression, member.as_deref())?;
    let frame = py
        .detach(move || {
            oxyz_core::iter_frames_from(reader)?
                .next()
                .unwrap_or(Err(ExtxyzError::MissingLine("atom count")))
        })
        .map_err(extxyz_error_to_py)?;
    frame_to_pydict(py, frame)
}

/// Read every frame, as a list of per-frame dicts. `threads=None` parses on
/// every core, `threads=1` serially; output and errors are identical.
#[pyfunction]
#[pyo3(signature = (path, threads=None, compression="infer", member=None))]
fn read_frames<'py>(
    py: Python<'py>,
    path: PathBuf,
    threads: Option<usize>,
    compression: &str,
    member: Option<String>,
) -> PyResult<Bound<'py, PyList>> {
    let reader = open_reader(&path, compression, member.as_deref())?;
    let frames = py
        .detach(move || match threads {
            Some(1) => oxyz_core::iter_frames_from(reader)?.collect(),
            _ => oxyz_core::read_frames_parallel_from(reader, threads),
        })
        .map_err(extxyz_error_to_py)?;

    let dicts = frames
        .into_iter()
        .map(|frame| frame_to_pydict(py, frame))
        .collect::<PyResult<Vec<_>>>()?;

    PyList::new(py, dicts)
}

/// Assemble a batch from parsed frame dicts. An empty list yields the empty
/// batch; a non-uniform set raises `ParseError`, as the file readers do.
#[pyfunction]
fn build_batch<'py>(
    py: Python<'py>,
    frames: Vec<Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let core_frames = frames
        .iter()
        .map(pydict_to_frame)
        .collect::<PyResult<Vec<_>>>()?;
    let batch = py
        .detach(move || -> Result<Batch, ExtxyzError> {
            let mut builder = oxyz_core::BatchBuilder::new();
            for frame in core_frames {
                builder.push(frame)?;
            }
            match builder.finish() {
                Ok(batch) => Ok(batch),
                // An empty set is no frames, not an error, as in the whole-file readers.
                Err(oxyz_core::BatchError::Empty) => Ok(Batch {
                    offsets: vec![0],
                    columns: Vec::new(),
                    metadata: Vec::new(),
                }),
                Err(error) => Err(error.into()),
            }
        })
        .map_err(extxyz_error_to_py)?;
    batch_to_pydict(py, batch)
}

fn kind_from_letter(letter: &str) -> PyResult<ColumnKind> {
    match letter {
        "R" => Ok(ColumnKind::Real),
        "I" => Ok(ColumnKind::Int),
        "L" => Ok(ColumnKind::Bool),
        "S" => Ok(ColumnKind::Str),
        other => Err(PyValueError::new_err(format!(
            "unknown kind letter {other:?}; expected one of R, I, L, S"
        ))),
    }
}

fn fill_from_py(kind: ColumnKind, obj: &Bound<'_, PyAny>) -> PyResult<Fill> {
    Ok(match kind {
        ColumnKind::Real => Fill::Real(obj.extract()?),
        ColumnKind::Int => Fill::Int(obj.extract()?),
        ColumnKind::Bool => Fill::Bool(obj.extract()?),
        ColumnKind::Str => Fill::Str(obj.extract()?),
    })
}

fn opt_fill(kind: ColumnKind, obj: Bound<'_, PyAny>) -> PyResult<Option<Fill>> {
    if obj.is_none() {
        Ok(None)
    } else {
        Ok(Some(fill_from_py(kind, &obj)?))
    }
}

/// Read a projection plan from `(columns, metadata)`. A column is `(name,
/// letter, width, required, fill | None)`; a metadata entry is `(name, letter,
/// shape, required, fill | None)`, with shape `()` for a scalar.
fn build_plan(py_plan: &Bound<'_, PyTuple>) -> PyResult<ProjectionPlan> {
    let columns_obj = py_plan.get_item(0)?;
    let metadata_obj = py_plan.get_item(1)?;

    let mut columns = Vec::new();
    for entry in columns_obj.try_iter()? {
        let entry = entry?;
        let name: String = entry.get_item(0)?.extract()?;
        let kind = kind_from_letter(&entry.get_item(1)?.extract::<String>()?)?;
        let width: usize = entry.get_item(2)?.extract()?;
        let required: bool = entry.get_item(3)?.extract()?;
        let fill = opt_fill(kind, entry.get_item(4)?)?;
        columns.push(PlanColumn {
            name,
            kind,
            width,
            required,
            fill,
        });
    }

    let mut metadata = Vec::new();
    for entry in metadata_obj.try_iter()? {
        let entry = entry?;
        let name: String = entry.get_item(0)?.extract()?;
        let kind = kind_from_letter(&entry.get_item(1)?.extract::<String>()?)?;
        let shape_tuple: Vec<usize> = entry.get_item(2)?.extract()?;
        let shape = match shape_tuple.as_slice() {
            [] => None,      // scalar
            [n] => Some(*n), // 1-D array of length n
            _ => {
                return Err(PyValueError::new_err(format!(
                    "metadata {name:?}: shape {shape_tuple:?} has more than one \
                     dimension; projection supports scalar or 1-D metadata"
                )));
            }
        };
        let required: bool = entry.get_item(3)?.extract()?;
        let fill = opt_fill(kind, entry.get_item(4)?)?;
        metadata.push(PlanMetadata {
            name,
            kind,
            shape,
            required,
            fill,
        });
    }

    Ok(ProjectionPlan { columns, metadata })
}

/// Emit deviations as a list of dicts for the Python policy layer.
fn deviations_to_py<'py>(
    py: Python<'py>,
    deviations: &[Deviation],
) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for d in deviations {
        let dict = PyDict::new(py);
        dict.set_item(
            "axis",
            match d.axis {
                Axis::Column => "column",
                Axis::Metadata => "metadata",
            },
        )?;
        dict.set_item("name", &d.name)?;
        dict.set_item(
            "deviation",
            match d.kind {
                DeviationKind::Missing => "missing",
                DeviationKind::Mismatch => "mismatch",
            },
        )?;
        dict.set_item("expected", &d.expected)?;
        dict.set_item("found", d.found.clone())?;
        list.append(dict)?;
    }
    Ok(list)
}

/// The `(FrameData | None, deviations)` tuple for a projected frame; `None`
/// marks a dropped frame.
fn projected_to_py(py: Python<'_>, projected: Projected) -> PyResult<Bound<'_, PyTuple>> {
    let devs = deviations_to_py(py, &projected.deviations)?;
    let data: Bound<'_, PyAny> = if projected.dropped {
        py.None().into_bound(py)
    } else {
        frame_to_pydict(py, projected.frame)?.into_any()
    };
    PyTuple::new(py, [data, devs.into_any()])
}

/// Projected variant of [`read_frames`]: each element is `(FrameData | None,
/// deviations)`. `plan` is the crossing tuple.
#[pyfunction]
#[pyo3(signature = (path, threads=None, compression="infer", member=None, *, plan))]
fn read_frames_projected<'py>(
    py: Python<'py>,
    path: PathBuf,
    threads: Option<usize>,
    compression: &str,
    member: Option<String>,
    plan: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyList>> {
    let plan = build_plan(plan)?;
    let reader = open_reader(&path, compression, member.as_deref())?;
    let projected = py
        .detach(move || read_frames_projected_core(reader, threads, &plan))
        .map_err(extxyz_error_to_py)?;
    let items = projected
        .into_iter()
        .map(|p| projected_to_py(py, p))
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, items)
}

/// Parse every frame and project it, all with the interpreter detached.
fn read_frames_projected_core(
    reader: DecodedReader,
    threads: Option<usize>,
    plan: &ProjectionPlan,
) -> Result<Vec<Projected>, ExtxyzError> {
    let frames: Vec<Frame> = match threads {
        Some(1) => oxyz_core::iter_frames_from(reader)?.collect::<Result<_, _>>()?,
        _ => oxyz_core::read_frames_parallel_from(reader, threads)?,
    };
    Ok(frames.iter().map(|f| project_frame(f, plan)).collect())
}

/// Projected variant of [`read_first_frame`]: `(FrameData | None, deviations)`.
#[pyfunction]
#[pyo3(signature = (path, compression="infer", member=None, *, plan))]
fn read_first_frame_projected<'py>(
    py: Python<'py>,
    path: PathBuf,
    compression: &str,
    member: Option<String>,
    plan: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyTuple>> {
    let plan = build_plan(plan)?;
    let reader = open_reader(&path, compression, member.as_deref())?;
    let projected = py
        .detach(move || {
            let frame = oxyz_core::iter_frames_from(reader)?
                .next()
                .unwrap_or(Err(ExtxyzError::MissingLine("atom count")))?;
            Ok::<_, ExtxyzError>(project_frame(&frame, &plan))
        })
        .map_err(extxyz_error_to_py)?;
    projected_to_py(py, projected)
}

/// Reader-source variant of [`read_frames_projected`].
#[pyfunction]
#[pyo3(signature = (source, codec, member=None, threads=None, *, plan))]
fn read_frames_projected_reader<'py>(
    py: Python<'py>,
    source: Bound<'py, PyAny>,
    codec: &str,
    member: Option<String>,
    threads: Option<usize>,
    plan: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyList>> {
    let plan = build_plan(plan)?;
    let reader = build_decoded(&source, codec, member.as_deref())?;
    let projected = py
        .detach(move || read_frames_projected_core(reader, threads, &plan))
        .map_err(extxyz_error_to_py)?;
    let items = projected
        .into_iter()
        .map(|p| projected_to_py(py, p))
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, items)
}

/// Reader-source variant of [`read_first_frame_projected`].
#[pyfunction]
#[pyo3(signature = (source, codec, member=None, *, plan))]
fn read_first_frame_projected_reader<'py>(
    py: Python<'py>,
    source: Bound<'py, PyAny>,
    codec: &str,
    member: Option<String>,
    plan: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyTuple>> {
    let plan = build_plan(plan)?;
    let reader = build_decoded(&source, codec, member.as_deref())?;
    let projected = py
        .detach(move || {
            let frame = oxyz_core::iter_frames_from(reader)?
                .next()
                .unwrap_or(Err(ExtxyzError::MissingLine("atom count")))?;
            Ok::<_, ExtxyzError>(project_frame(&frame, &plan))
        })
        .map_err(extxyz_error_to_py)?;
    projected_to_py(py, projected)
}

/// Projected variant of [`FrameIter`]: `__next__` yields `(FrameData | None,
/// deviations)`.
#[pyclass]
struct FrameIterProjected {
    inner: oxyz_core::FrameIter<DecodedReader>,
    plan: ProjectionPlan,
}

#[pymethods]
impl FrameIterProjected {
    #[new]
    #[pyo3(signature = (path, plan, compression="infer", member=None))]
    fn new(
        path: PathBuf,
        plan: &Bound<'_, PyTuple>,
        compression: &str,
        member: Option<String>,
    ) -> PyResult<Self> {
        let plan = build_plan(plan)?;
        let reader = open_reader(&path, compression, member.as_deref())?;
        let inner = oxyz_core::iter_frames_from(reader).map_err(extxyz_error_to_py)?;
        Ok(FrameIterProjected { inner, plan })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyTuple>>> {
        let plan = &self.plan;
        let projected = py.detach(|| match self.inner.next() {
            None => None,
            Some(Ok(frame)) => Some(Ok(project_frame(&frame, plan))),
            Some(Err(error)) => Some(Err(error)),
        });
        match projected {
            None => Ok(None),
            Some(Ok(projected)) => projected_to_py(py, projected).map(Some),
            Some(Err(error)) => Err(extxyz_error_to_py(error)),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (source, plan, codec, member=None))]
    fn from_reader(
        source: Bound<'_, PyAny>,
        plan: &Bound<'_, PyTuple>,
        codec: &str,
        member: Option<String>,
    ) -> PyResult<Self> {
        let plan = build_plan(plan)?;
        let reader = build_decoded(&source, codec, member.as_deref())?;
        let inner = oxyz_core::iter_frames_from(reader).map_err(extxyz_error_to_py)?;
        Ok(FrameIterProjected { inner, plan })
    }
}

/// Gather frames into one batch: the whole file when `indices=None`, else
/// those frames (request order, repeats allowed).
///
/// Reads nothing past the last requested frame. `threads=None` parses on every
/// core, `threads=1` serially; the batch is identical, though a malformed
/// whole-file read may report a different frame's error.
#[pyfunction]
#[pyo3(signature = (path, indices=None, threads=None, compression="infer", member=None))]
fn read_batch<'py>(
    py: Python<'py>,
    path: PathBuf,
    indices: Option<Vec<usize>>,
    threads: Option<usize>,
    compression: &str,
    member: Option<String>,
) -> PyResult<Bound<'py, PyDict>> {
    let reader = open_reader(&path, compression, member.as_deref())?;
    let batch = py
        .detach(move || match (indices, threads) {
            (None, Some(1)) => oxyz_core::read_all_batch_from(reader),
            (None, _) => oxyz_core::read_all_batch_parallel_from(reader, threads),
            (Some(indices), Some(1)) => oxyz_core::read_batch_from(reader, &indices),
            (Some(indices), _) => oxyz_core::read_batch_parallel_from(reader, &indices, threads),
        })
        .map_err(extxyz_error_to_py)?;
    batch_to_pydict(py, batch)
}

/// `(BatchData, survivors, reports)`: the surviving frames' batch, their file
/// indices, and an `(index, deviations)` report per requested frame.
fn projected_batch_to_py<'py>(
    py: Python<'py>,
    projected: oxyz_core::ProjectedBatch,
) -> PyResult<Bound<'py, PyTuple>> {
    let oxyz_core::ProjectedBatch {
        batch,
        survivors,
        reports,
    } = projected;
    let batch_data = batch_to_pydict(py, batch)?;
    let survivors_list = PyList::new(py, survivors)?;
    let reports_list = PyList::empty(py);
    for (index, deviations) in &reports {
        let devs = deviations_to_py(py, deviations)?;
        reports_list.append((*index, devs))?;
    }
    PyTuple::new(
        py,
        [
            batch_data.into_any(),
            survivors_list.into_any(),
            reports_list.into_any(),
        ],
    )
}

/// Projected variant of [`read_batch`]: returns `(BatchData, survivors,
/// reports)` where the batch holds only frames that survived projection.
#[pyfunction]
#[pyo3(signature = (path, indices=None, threads=None, compression="infer", member=None, *, plan))]
fn read_batch_projected<'py>(
    py: Python<'py>,
    path: PathBuf,
    indices: Option<Vec<usize>>,
    threads: Option<usize>,
    compression: &str,
    member: Option<String>,
    plan: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyTuple>> {
    let plan = build_plan(plan)?;
    let reader = open_reader(&path, compression, member.as_deref())?;
    let projected = py
        .detach(move || match (indices, threads) {
            (None, Some(1)) => oxyz_core::read_all_batch_projected_from(reader, &plan),
            (None, _) => oxyz_core::read_all_batch_projected_parallel_from(reader, threads, &plan),
            (Some(indices), Some(1)) => {
                oxyz_core::read_batch_projected_from(reader, &indices, &plan)
            }
            (Some(indices), _) => {
                oxyz_core::read_batch_projected_parallel_from(reader, &indices, threads, &plan)
            }
        })
        .map_err(extxyz_error_to_py)?;
    projected_batch_to_py(py, projected)
}

/// Reader-source variant of [`read_batch_projected`].
#[pyfunction]
#[pyo3(signature = (source, codec, indices=None, threads=None, member=None, *, plan))]
fn read_batch_projected_reader<'py>(
    py: Python<'py>,
    source: Bound<'py, PyAny>,
    codec: &str,
    indices: Option<Vec<usize>>,
    threads: Option<usize>,
    member: Option<String>,
    plan: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyTuple>> {
    let plan = build_plan(plan)?;
    let reader = build_decoded(&source, codec, member.as_deref())?;
    let projected = py
        .detach(move || match (indices, threads) {
            (None, Some(1)) => oxyz_core::read_all_batch_projected_from(reader, &plan),
            (None, _) => oxyz_core::read_all_batch_projected_parallel_from(reader, threads, &plan),
            (Some(indices), Some(1)) => {
                oxyz_core::read_batch_projected_from(reader, &indices, &plan)
            }
            (Some(indices), _) => {
                oxyz_core::read_batch_projected_parallel_from(reader, &indices, threads, &plan)
            }
        })
        .map_err(extxyz_error_to_py)?;
    projected_batch_to_py(py, projected)
}

/// Projected variant of [`BatchIter`]: `__next__` yields `(BatchData,
/// survivors, reports)`.
#[pyclass]
struct BatchIterProjected {
    inner: oxyz_core::BatchIterProjected<DecodedReader>,
}

#[pymethods]
impl BatchIterProjected {
    #[new]
    #[pyo3(signature = (path, frames_per_batch, plan, compression="infer", member=None))]
    fn new(
        path: PathBuf,
        frames_per_batch: usize,
        plan: &Bound<'_, PyTuple>,
        compression: &str,
        member: Option<String>,
    ) -> PyResult<Self> {
        let plan = build_plan(plan)?;
        let reader = open_reader(&path, compression, member.as_deref())?;
        let inner = oxyz_core::iter_batches_projected_from(reader, frames_per_batch, plan)
            .map_err(extxyz_error_to_py)?;
        Ok(BatchIterProjected { inner })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyTuple>>> {
        match py.detach(|| self.inner.next()) {
            None => Ok(None),
            Some(Ok(projected)) => projected_batch_to_py(py, projected).map(Some),
            Some(Err(error)) => Err(extxyz_error_to_py(error)),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (source, frames_per_batch, plan, codec, member=None))]
    fn from_reader(
        source: Bound<'_, PyAny>,
        frames_per_batch: usize,
        plan: &Bound<'_, PyTuple>,
        codec: &str,
        member: Option<String>,
    ) -> PyResult<Self> {
        let plan = build_plan(plan)?;
        let reader = build_decoded(&source, codec, member.as_deref())?;
        let inner = oxyz_core::iter_batches_projected_from(reader, frames_per_batch, plan)
            .map_err(extxyz_error_to_py)?;
        Ok(BatchIterProjected { inner })
    }
}

/// Infer the file's schema as one nested dict for the Python `Schema`
/// dataclasses to wrap.
#[pyfunction]
#[pyo3(signature = (path, compression="infer", member=None))]
fn infer_schema<'py>(
    py: Python<'py>,
    path: PathBuf,
    compression: &str,
    member: Option<String>,
) -> PyResult<Bound<'py, PyDict>> {
    let reader = open_reader(&path, compression, member.as_deref())?;
    let schema = py
        .detach(move || oxyz_core::infer_schema_from(reader))
        .map_err(extxyz_error_to_py)?;
    schema_to_pydict(py, &schema)
}

/// Whether `path` would be read through a decompressing layer under
/// `compression`, so random-access batch strategies can refuse it.
#[pyfunction]
#[pyo3(signature = (path, compression="infer"))]
fn is_compressed(path: PathBuf, compression: &str) -> PyResult<bool> {
    oxyz_core::is_compressed(&path, parse_compression(compression)?).map_err(extxyz_error_to_py)
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
/// `()` for scalars, `(n,)` for 1-D arrays, `(rows, cols)` for 2-D arrays.
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
        ValueType::RealArray2D(rows, cols) => ("Real", vec![rows, cols]),
        ValueType::IntArray2D(rows, cols) => ("Int", vec![rows, cols]),
        ValueType::BoolArray2D(rows, cols) => ("Bool", vec![rows, cols]),
        ValueType::StrArray2D(rows, cols) => ("Str", vec![rows, cols]),
    }
}

fn schema_to_pydict<'py>(py: Python<'py>, schema: &Schema) -> PyResult<Bound<'py, PyDict>> {
    let data = PyDict::new(py);
    data.set_item("n_frames", schema.n_frames)?;
    data.set_item("total_atoms", schema.total_atoms)?;
    data.set_item("min_atoms", schema.min_atoms)?;
    data.set_item("max_atoms", schema.max_atoms)?;
    // intp, as in scan().
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
/// Numeric and boolean columns become numpy arrays (2-D when width > 1),
/// string columns `list[str]`. Both dicts keep file order.
fn frame_to_pydict(py: Python<'_>, frame: Frame) -> PyResult<Bound<'_, PyDict>> {
    let data = PyDict::new(py);
    data.set_item("n_atoms", frame.n_atoms)?;
    data.set_item("columns", columns_to_pydict(py, frame.columns)?)?;

    let metadata = PyDict::new(py);
    for (key, value) in frame.metadata {
        let key = key.as_str();
        match value {
            Value::Real(x) => metadata.set_item(key, x)?,
            Value::Int(x) => metadata.set_item(key, x)?,
            Value::Bool(x) => metadata.set_item(key, x)?,
            Value::Str(x) => metadata.set_item(key, x.as_str())?,
            Value::RealArray(values) => metadata.set_item(key, values.into_pyarray(py))?,
            Value::IntArray(values) => metadata.set_item(key, values.into_pyarray(py))?,
            Value::BoolArray(values) => metadata.set_item(key, values.into_pyarray(py))?,
            Value::StrArray(values) => {
                metadata.set_item(key, values.iter().map(|s| s.as_str()).collect::<Vec<_>>())?
            }
            // Unlike a per-atom column, 2-D metadata stays 2-D even for one row or column.
            Value::RealArray2D { rows, cols, data } => {
                metadata.set_item(key, array2d_from_flat(py, data, rows, cols)?)?
            }
            Value::IntArray2D { rows, cols, data } => {
                metadata.set_item(key, array2d_from_flat(py, data, rows, cols)?)?
            }
            Value::BoolArray2D { rows, cols, data } => {
                metadata.set_item(key, array2d_from_flat(py, data, rows, cols)?)?
            }
            Value::StrArray2D { cols, data, .. } => {
                let rows: Vec<Vec<&str>> = data
                    .chunks(cols)
                    .map(|row| row.iter().map(|s| s.as_str()).collect())
                    .collect();
                metadata.set_item(key, rows)?
            }
        }
    }
    data.set_item("metadata", metadata)?;

    Ok(data)
}

/// Convert a batch to `{"offsets", "columns", "metadata"}`: columns
/// atom-major, metadata frame-major.
fn batch_to_pydict(py: Python<'_>, batch: Batch) -> PyResult<Bound<'_, PyDict>> {
    let data = PyDict::new(py);
    let offsets: Vec<i64> = batch.offsets.iter().map(|&offset| offset as i64).collect();
    data.set_item("offsets", offsets.into_pyarray(py))?;
    data.set_item("columns", columns_to_pydict(py, batch.columns)?)?;
    data.set_item("metadata", columns_to_pydict(py, batch.metadata)?)?;
    Ok(data)
}

/// Numeric and boolean columns become numpy arrays (2-D when width > 1),
/// string columns `list[str]` (nested when width > 1).
fn columns_to_pydict(py: Python<'_>, columns: Vec<Column>) -> PyResult<Bound<'_, PyDict>> {
    let dict = PyDict::new(py);
    for column in columns {
        let Column { name, width, data } = column;
        let name = name.as_str();

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
                    dict.set_item(name, values.iter().map(|s| s.as_str()).collect::<Vec<_>>())?;
                } else {
                    let rows: Vec<Vec<&str>> = values
                        .chunks(width)
                        .map(|c| c.iter().map(|s| s.as_str()).collect())
                        .collect();
                    dict.set_item(name, rows)?;
                }
            }
        }
    }
    Ok(dict)
}

/// Write a list of `{"n_atoms", "columns", "metadata"}` dicts to `path`.
///
/// Encoding and I/O run without the GIL. `level` is `0..=9` (`None`: codec
/// default). `threads=None` uses every core, `1` is serial; the bytes are
/// identical either way.
#[pyfunction]
#[pyo3(signature = (path, frames, compression="infer", level=None, append=false, threads=None))]
fn write(
    py: Python<'_>,
    path: PathBuf,
    frames: Vec<Bound<'_, PyDict>>,
    compression: &str,
    level: Option<i32>,
    append: bool,
    threads: Option<usize>,
) -> PyResult<()> {
    let compression = parse_compression(compression)?;
    let frames = frames
        .iter()
        .map(pydict_to_frame)
        .collect::<PyResult<Vec<_>>>()?;
    py.detach(|| match threads {
        Some(1) => write_frames(&path, &frames, compression, level, append),
        _ => write_frames_parallel(&path, &frames, compression, level, append, threads),
    })
    .map_err(extxyz_error_to_py)
}

/// Incremental writer behind `oxyz.Writer`. `batch=None` streams each frame
/// in constant memory; `batch=Some(n)` buffers `n` frames and serialises each
/// full batch in parallel. The bytes are identical either way.
#[pyclass]
struct FrameWriter {
    // `None` once closed, so a write after close is caught.
    sink: Option<FrameSink>,
    batch: Option<usize>,
    buffer: Vec<Frame>,
}

#[pymethods]
impl FrameWriter {
    #[new]
    #[pyo3(signature = (path, compression="infer", level=None, append=false, batch=None))]
    fn new(
        path: PathBuf,
        compression: &str,
        level: Option<i32>,
        append: bool,
        batch: Option<usize>,
    ) -> PyResult<Self> {
        if batch == Some(0) {
            return Err(PyValueError::new_err("Writer batch must be at least 1"));
        }
        let compression = parse_compression(compression)?;
        let sink =
            FrameSink::create(&path, compression, level, append).map_err(extxyz_error_to_py)?;
        Ok(FrameWriter {
            sink: Some(sink),
            batch,
            buffer: Vec::new(),
        })
    }

    fn write(&mut self, py: Python<'_>, frame: Bound<'_, PyDict>) -> PyResult<()> {
        let frame = pydict_to_frame(&frame)?;
        match self.batch {
            None => {
                let sink = self.sink_mut()?;
                py.detach(|| sink.write(&frame)).map_err(extxyz_error_to_py)
            }
            Some(batch) => {
                self.buffer.push(frame);
                if self.buffer.len() >= batch {
                    self.flush(py)?;
                }
                Ok(())
            }
        }
    }

    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        self.flush(py)?;
        if let Some(sink) = self.sink.take() {
            py.detach(|| sink.finish()).map_err(extxyz_error_to_py)?;
        }
        Ok(())
    }
}

impl FrameWriter {
    fn sink_mut(&mut self) -> PyResult<&mut FrameSink> {
        self.sink
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("write on a closed Writer"))
    }

    /// Serialise and write any buffered frames, then clear the buffer.
    fn flush(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let buffer = std::mem::take(&mut self.buffer);
        let sink = self.sink_mut()?;
        py.detach(|| sink.write_batch_parallel(&buffer, None))
            .map_err(extxyz_error_to_py)
    }
}

/// Build a core `Frame` from a `{"n_atoms", "columns", "metadata"}` dict,
/// the inverse of [`frame_to_pydict`]. Numeric columns arrive already coerced
/// to f64/i64/bool.
fn pydict_to_frame(data: &Bound<'_, PyDict>) -> PyResult<Frame> {
    let n_atoms: usize = item(data, "n_atoms")?.extract()?;
    let columns = item(data, "columns")?;
    let metadata = item(data, "metadata")?;

    let mut core_columns = Vec::new();
    for (key, value) in columns.cast::<PyDict>()?.iter() {
        let name: String = key.extract()?;
        core_columns.push(py_to_column(name, &value)?);
    }

    let mut core_metadata = Vec::new();
    for (key, value) in metadata.cast::<PyDict>()?.iter() {
        core_metadata.push((key.extract::<String>()?.into(), py_to_value(&value)?));
    }

    Ok(Frame {
        n_atoms,
        columns: core_columns,
        metadata: core_metadata,
    })
}

fn item<'py>(data: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    data.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("frame dict is missing {key:?}")))
}

/// One per-atom column: a numpy array is numeric, its dtype picking the kind;
/// a list is a string column (nested for width > 1).
fn py_to_column(name: String, value: &Bound<'_, PyAny>) -> PyResult<Column> {
    if value.is_instance_of::<PyList>() {
        let (data, width) = str_column(value.cast::<PyList>()?)?;
        return Ok(Column {
            name: name.into(),
            width,
            data,
        });
    }
    let (data, width) = numeric_array(value).ok_or_else(|| {
        PyTypeError::new_err(format!(
            "column {name:?} must be a numpy float/int/bool array or a list of strings"
        ))
    })?;
    Ok(Column {
        name: name.into(),
        width,
        data,
    })
}

/// A string column from `list[str]` (width 1) or `list[list[str]]` (width n),
/// flattened row-major.
fn str_column(list: &Bound<'_, PyList>) -> PyResult<(ColumnData, usize)> {
    let nested = list
        .get_item(0)
        .ok()
        .is_some_and(|first| first.is_instance_of::<PyList>());

    if !nested {
        let flat: Vec<String> = list.extract()?;
        return Ok((
            ColumnData::Str(flat.into_iter().map(Into::into).collect()),
            1,
        ));
    }

    let (rows, width) = rectangular_string_rows(list, 1)?;
    Ok((
        ColumnData::Str(rows.into_iter().flatten().map(Into::into).collect()),
        width,
    ))
}

/// Extract `list[list[str]]`, raising `ValueError` if ragged. An empty list
/// has `default_width`.
fn rectangular_string_rows(
    list: &Bound<'_, PyList>,
    default_width: usize,
) -> PyResult<(Vec<Vec<String>>, usize)> {
    let rows: Vec<Vec<String>> = list.extract()?;
    let width = rows.first().map_or(default_width, Vec::len);
    if rows.iter().any(|row| row.len() != width) {
        return Err(PyValueError::new_err(
            "nested string list rows have differing widths",
        ));
    }
    Ok((rows, width))
}

/// A numeric column's flat buffer and width, or `None` if `value` is not a 1-D
/// or 2-D f64/i64/bool numpy array.
fn numeric_array(value: &Bound<'_, PyAny>) -> Option<(ColumnData, usize)> {
    flat_array::<f64>(value)
        .map(|(v, w)| (ColumnData::Real(v), w))
        .or_else(|| flat_array::<i64>(value).map(|(v, w)| (ColumnData::Int(v), w)))
        .or_else(|| flat_array::<bool>(value).map(|(v, w)| (ColumnData::Bool(v), w)))
}

/// Flatten a 1-D or 2-D numpy array of `T` into a buffer and its width (1
/// when 1-D). `None` if the dtype or rank does not match.
fn flat_array<T: Element + Clone>(value: &Bound<'_, PyAny>) -> Option<(Vec<T>, usize)> {
    let array = value.cast::<PyArrayDyn<T>>().ok()?;
    let readonly = array.readonly();
    let view = readonly.as_array();
    let width = match view.ndim() {
        0 | 1 => 1,
        2 => view.shape()[1],
        _ => return None,
    };
    Some((view.iter().cloned().collect(), width))
}

/// A 2-D numpy array of `T` as its flat buffer and `(rows, cols)`. Unlike
/// `flat_array`, a 1-D array does not match: metadata keeps 1-D and 2-D apart.
fn flat_array_2d<T: Element + Clone>(value: &Bound<'_, PyAny>) -> Option<(Vec<T>, usize, usize)> {
    let array = value.cast::<PyArrayDyn<T>>().ok()?;
    let readonly = array.readonly();
    let view = readonly.as_array();
    if view.ndim() != 2 {
        return None;
    }
    let shape = view.shape();
    Some((view.iter().cloned().collect(), shape[0], shape[1]))
}

/// A 2-D numeric metadata array, or `None` if `value` is not a 2-D
/// f64/i64/bool numpy array.
fn numeric_array_2d(value: &Bound<'_, PyAny>) -> Option<Value> {
    if let Some((data, rows, cols)) = flat_array_2d::<f64>(value) {
        return Some(Value::RealArray2D { rows, cols, data });
    }
    if let Some((data, rows, cols)) = flat_array_2d::<i64>(value) {
        return Some(Value::IntArray2D { rows, cols, data });
    }
    if let Some((data, rows, cols)) = flat_array_2d::<bool>(value) {
        return Some(Value::BoolArray2D { rows, cols, data });
    }
    None
}

/// A metadata string value from `list[str]` or `list[list[str]]`, raising
/// `ValueError` if ragged.
fn str_metadata(list: &Bound<'_, PyList>) -> PyResult<Value> {
    let nested = list
        .get_item(0)
        .ok()
        .is_some_and(|first| first.is_instance_of::<PyList>());

    if !nested {
        let flat: Vec<String> = list.extract()?;
        return Ok(Value::StrArray(flat.into_iter().map(Into::into).collect()));
    }

    let (rows, cols) = rectangular_string_rows(list, 0)?;
    Ok(Value::StrArray2D {
        rows: rows.len(),
        cols,
        data: rows.into_iter().flatten().map(Into::into).collect(),
    })
}

/// One metadata value. bool is tried before int, as a Python bool is an int.
fn py_to_value(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    if value.is_instance_of::<PyString>() {
        return Ok(Value::Str(value.extract::<String>()?.into()));
    }
    if let Some(array2d) = numeric_array_2d(value) {
        return Ok(array2d);
    }
    if let Some((data, _)) = numeric_array(value) {
        return Ok(match data {
            ColumnData::Real(v) => Value::RealArray(v),
            ColumnData::Int(v) => Value::IntArray(v),
            ColumnData::Bool(v) => Value::BoolArray(v),
            ColumnData::Str(v) => Value::StrArray(v),
        });
    }
    if value.is_instance_of::<PyList>() {
        return str_metadata(value.cast::<PyList>()?);
    }
    if let Ok(b) = value.cast::<pyo3::types::PyBool>() {
        return Ok(Value::Bool(b.is_true()));
    }
    if let Ok(i) = value.extract::<i64>() {
        return Ok(Value::Int(i));
    }
    if let Ok(x) = value.extract::<f64>() {
        return Ok(Value::Real(x));
    }
    Err(PyTypeError::new_err(format!(
        "unsupported metadata value type: {}",
        value.get_type().name()?
    )))
}

/// A Rust `Read` over a Python iterator of `bytes`, such as obstore's
/// `GetResult.stream()`. Each refill takes the GIL, which is negligible
/// against network latency.
struct PyChunkReader {
    iter: Py<PyAny>,
    current: Cursor<Vec<u8>>,
    done: bool,
}

impl PyChunkReader {
    fn new(iter: Py<PyAny>) -> Self {
        PyChunkReader {
            iter,
            current: Cursor::new(Vec::new()),
            done: false,
        }
    }
}

impl Read for PyChunkReader {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        loop {
            let read = self.current.read(out)?;
            if read > 0 {
                return Ok(read);
            }
            if self.done {
                return Ok(0);
            }
            let next = Python::attach(|py| match self.iter.bind(py).call_method0("__next__") {
                Ok(obj) => obj
                    .cast::<PyBytes>()
                    .map(|b| Some(b.as_bytes().to_vec()))
                    .map_err(|_| std::io::Error::other("remote stream yielded a non-bytes chunk")),
                Err(err) if err.is_instance_of::<pyo3::exceptions::PyStopIteration>(py) => Ok(None),
                Err(err) => Err(std::io::Error::other(err.to_string())),
            })?;
            match next {
                Some(chunk) => self.current = Cursor::new(chunk),
                None => {
                    self.done = true;
                    return Ok(0);
                }
            }
        }
    }
}

/// A Rust `Read + Seek` over a Python file-like (obstore's `ReadableFile`),
/// for `.zip`, whose central directory is at the end.
struct PySeekReader {
    file: Py<PyAny>,
}

impl Read for PySeekReader {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        Python::attach(|py| {
            let obj = self
                .file
                .bind(py)
                .call_method1("read", (out.len(),))
                .map_err(|e| std::io::Error::other(e.to_string()))?;
            let bytes = obj
                .cast::<PyBytes>()
                .map_err(|_| std::io::Error::other("read() did not return bytes"))?;
            let data = bytes.as_bytes();
            if data.len() > out.len() {
                return Err(std::io::Error::other(
                    "read() returned more bytes than requested",
                ));
            }
            out[..data.len()].copy_from_slice(data);
            Ok(data.len())
        })
    }
}

impl Seek for PySeekReader {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        // Python io whence: 0=SET, 1=CUR, 2=END; seek() returns the new abs pos.
        let (offset, whence): (i64, i64) = match pos {
            SeekFrom::Start(n) => (n as i64, 0),
            SeekFrom::Current(n) => (n, 1),
            SeekFrom::End(n) => (n, 2),
        };
        Python::attach(|py| {
            self.file
                .bind(py)
                .call_method1("seek", (offset, whence))
                .and_then(|obj| obj.extract::<u64>())
                .map_err(|e| std::io::Error::other(e.to_string()))
        })
    }
}

/// Assemble a `DecodedReader` from a Python source and a resolved codec.
/// `source` is a bytes-iterator for plain, gzip and zstd; a callable returning
/// a fresh one for the tar codecs; a seekable file-like for zip.
fn build_decoded(
    source: &Bound<'_, PyAny>,
    codec: &str,
    member: Option<&str>,
) -> PyResult<DecodedReader> {
    match codec {
        "plain" | "gzip" | "zstd" => {
            if member.is_some() {
                return Err(PyValueError::new_err(
                    "member= is only valid for an archive (.zip/.tar/.tar.gz/.tar.zst) source",
                ));
            }
            let codec = match codec {
                "plain" => Codec::Plain,
                "gzip" => Codec::Gzip,
                "zstd" => Codec::Zstd,
                _ => unreachable!("outer match limits codec to plain/gzip/zstd"),
            };
            let reader: ByteSource = Box::new(PyChunkReader::new(source.clone().unbind()));
            wrap_stream(reader, codec).map_err(extxyz_error_to_py)
        }
        "tar" | "tar.gz" | "tar.zst" => {
            let inner = match codec {
                "tar" => TarInner::Plain,
                "tar.gz" => TarInner::Gzip,
                _ => TarInner::Zstd,
            };
            let callable = source.clone().unbind();
            let factory = move || {
                Python::attach(|py| {
                    callable
                        .bind(py)
                        .call0()
                        .map(|iter| Box::new(PyChunkReader::new(iter.unbind())) as ByteSource)
                        .map_err(|e| std::io::Error::other(e.to_string()))
                })
            };
            wrap_tar(factory, member, inner).map_err(extxyz_error_to_py)
        }
        "zip" => {
            let reader = PySeekReader {
                file: source.clone().unbind(),
            };
            wrap_zip(reader, member).map_err(extxyz_error_to_py)
        }
        other => Err(PyValueError::new_err(format!(
            "unsupported remote codec {other:?}"
        ))),
    }
}

/// Read every frame from a bytes-iterator `source`, as per-frame dicts.
/// `codec` is `"plain"`, `"gzip"` or `"zstd"`.
#[pyfunction]
#[pyo3(signature = (source, codec, member=None, threads=None))]
fn read_frames_reader<'py>(
    py: Python<'py>,
    source: Bound<'py, PyAny>,
    codec: &str,
    member: Option<String>,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyList>> {
    let reader = build_decoded(&source, codec, member.as_deref())?;
    let frames = py
        .detach(move || match threads {
            Some(1) => oxyz_core::iter_frames_from(reader)?.collect(),
            _ => oxyz_core::read_frames_parallel_from(reader, threads),
        })
        .map_err(extxyz_error_to_py)?;
    let dicts = frames
        .into_iter()
        .map(|frame| frame_to_pydict(py, frame))
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, dicts)
}

/// Read the first frame from a bytes-iterator `source`. `codec` is
/// `"plain"`, `"gzip"` or `"zstd"`.
#[pyfunction]
#[pyo3(signature = (source, codec, member=None))]
fn read_first_frame_reader<'py>(
    py: Python<'py>,
    source: Bound<'py, PyAny>,
    codec: &str,
    member: Option<String>,
) -> PyResult<Bound<'py, PyDict>> {
    let reader = build_decoded(&source, codec, member.as_deref())?;
    let frame = py
        .detach(move || {
            oxyz_core::iter_frames_from(reader)?
                .next()
                .unwrap_or(Err(ExtxyzError::MissingLine("atom count")))
        })
        .map_err(extxyz_error_to_py)?;
    frame_to_pydict(py, frame)
}

/// Infer the codec name from a filename and optional header bytes.
///
/// Returns one of `"plain"`, `"gzip"`, `"zstd"`, `"tar"`, `"tar.gz"`, `"tar.zst"`, `"zip"`.
#[pyfunction]
#[pyo3(signature = (name, head=None))]
fn detect_codec(name: &str, head: Option<&[u8]>) -> String {
    detect_codec_name(name, head).to_owned()
}

create_exception!(
    _rust,
    OxyzError,
    PyValueError,
    "Base class for every error oxyz raises.\n\n\
     A `ValueError` subclass, so `except ValueError` still catches everything;\n\
     `except oxyz.OxyzError` narrows to errors this package raised."
);

create_exception!(
    _rust,
    ParseError,
    OxyzError,
    "Raised when extxyz content cannot be parsed.\n\n\
     An `OxyzError` (and so a `ValueError`) subclass. Carries the location of\n\
     the offending input as attributes — `frame_index`, `line`, and `column`\n\
     (the 1-based character column of the offending token within its line) —\n\
     each `None` when the parser cannot pin that dimension down, so callers can\n\
     find the bad frame without parsing the message string."
);

fn extxyz_error_to_py(error: ExtxyzError) -> PyErr {
    // Unwrap frame context to classify the underlying error.
    let mut inner = &error;
    while let ExtxyzError::InFrame { source, .. } = inner {
        inner = source;
    }

    // I/O and out-of-range keep their stdlib exception types.
    match inner {
        ExtxyzError::Io(io_error) => return PyOSError::new_err(io_error.to_string()),
        ExtxyzError::FrameOutOfRange { .. } => return PyIndexError::new_err(error.to_string()),
        // Request errors, not content errors: no frame location to attach.
        ExtxyzError::MemberNotFound { .. }
        | ExtxyzError::AmbiguousArchive { .. }
        | ExtxyzError::NoExtxyzMember { .. }
        | ExtxyzError::MemberOnNonArchive
        | ExtxyzError::RandomAccessUnsupported
        | ExtxyzError::MissingRequiredColumn { .. }
        | ExtxyzError::AppendUnsupported { .. }
        | ExtxyzError::ZstdWriteUnsupported
        | ExtxyzError::InvalidCompressionLevel { .. } => {
            return PyValueError::new_err(error.to_string());
        }
        _ => {}
    }

    let frame_index = error.frame_index();
    let line = error.line();
    let column = error.column();
    let err = ParseError::new_err(error.to_string());
    Python::attach(|py| {
        // Instance values shadow the class-level `None` defaults.
        let value = err.value(py);
        let _ = value.setattr("frame_index", frame_index);
        let _ = value.setattr("line", line);
        let _ = value.setattr("column", column);
    });
    err
}

/// Turn a flat width-strided buffer into a 1-D (width 1) or 2-D numpy array.
fn array_from_flat<T: Element>(
    py: Python<'_>,
    values: Vec<T>,
    width: usize,
) -> PyResult<Bound<'_, PyAny>> {
    if width == 1 {
        return Ok(values.into_pyarray(py).into_any());
    }
    // The parser never emits width 0, but it would divide by zero below.
    if width == 0 {
        return Ok(Array2::<T>::from_shape_vec((0, 0), values)
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .into_pyarray(py)
            .into_any());
    }

    let n_rows = values.len() / width;
    let array = Array2::from_shape_vec((n_rows, width), values)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;

    Ok(array.into_pyarray(py).into_any())
}

/// Turn a flat row-major buffer into a 2-D numpy array of shape `(rows, cols)`.
fn array2d_from_flat<T: Element>(
    py: Python<'_>,
    values: Vec<T>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'_, PyAny>> {
    let array = Array2::from_shape_vec((rows, cols), values)
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
    // So the location attributes resolve even on a user-built `ParseError`.
    m.add("OxyzError", m.py().get_type::<OxyzError>())?;
    let parse_error = m.py().get_type::<ParseError>();
    parse_error.setattr("frame_index", m.py().None())?;
    parse_error.setattr("line", m.py().None())?;
    parse_error.setattr("column", m.py().None())?;
    m.add("ParseError", parse_error)?;

    m.add_function(wrap_pyfunction!(read_first_frame, m)?)?;
    m.add_function(wrap_pyfunction!(read_frames, m)?)?;
    m.add_function(wrap_pyfunction!(read_batch, m)?)?;
    m.add_function(wrap_pyfunction!(infer_schema, m)?)?;
    m.add_function(wrap_pyfunction!(scan, m)?)?;
    m.add_function(wrap_pyfunction!(is_compressed, m)?)?;
    m.add_function(wrap_pyfunction!(write, m)?)?;
    m.add_function(wrap_pyfunction!(detect_codec, m)?)?;
    m.add_function(wrap_pyfunction!(read_frames_reader, m)?)?;
    m.add_function(wrap_pyfunction!(read_first_frame_reader, m)?)?;
    m.add_function(wrap_pyfunction!(scan_reader, m)?)?;
    m.add_function(wrap_pyfunction!(infer_schema_reader, m)?)?;
    m.add_function(wrap_pyfunction!(read_batch_reader, m)?)?;
    m.add_function(wrap_pyfunction!(read_frames_projected, m)?)?;
    m.add_function(wrap_pyfunction!(read_first_frame_projected, m)?)?;
    m.add_function(wrap_pyfunction!(read_frames_projected_reader, m)?)?;
    m.add_function(wrap_pyfunction!(read_first_frame_projected_reader, m)?)?;
    m.add_function(wrap_pyfunction!(read_batch_projected, m)?)?;
    m.add_function(wrap_pyfunction!(read_batch_projected_reader, m)?)?;
    m.add_function(wrap_pyfunction!(build_batch, m)?)?;
    m.add_class::<FrameIterProjected>()?;
    m.add_class::<BatchIterProjected>()?;
    m.add_class::<FrameIter>()?;
    m.add_class::<IndexedFrames>()?;
    m.add_class::<BatchIter>()?;
    m.add_class::<FrameWriter>()?;
    Ok(())
}
