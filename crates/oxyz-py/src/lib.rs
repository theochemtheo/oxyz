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

use oxyz_core::schema::{ColumnSchema, MetadataSchema, Schema, ValueType};
use oxyz_core::{
    Batch, ByteSource, Codec, Column, ColumnData, ColumnKind, Compression, DecodedReader,
    ExtxyzError, Frame, FrameSink, Value, detect_codec_name, open_decoded, wrap_stream, wrap_tar,
    wrap_zip, write_frames, write_frames_parallel,
};

/// Map the Python `compression` string to the core selector.
fn parse_compression(name: &str) -> PyResult<Compression> {
    Ok(match name {
        "infer" => Compression::Infer,
        "none" => Compression::None,
        "gzip" => Compression::Gzip,
        "zstd" => Compression::Zstd,
        "zip" => Compression::Zip,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown compression {other:?}; expected one of: infer, none, gzip, zstd, zip"
            )));
        }
    })
}

/// Open a streaming, decoded reader for the given path and options.
fn open_reader(path: &Path, compression: &str, member: Option<&str>) -> PyResult<DecodedReader> {
    open_decoded(path, parse_compression(compression)?, member).map_err(extxyz_error_to_py)
}

/// Streaming iterator: one frame parsed and converted per `__next__`.
///
/// Owns the file handle; it closes when the object is dropped. The inner
/// iterator is fused — after an error or EOF it only raises StopIteration.
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
        // Parse with the interpreter detached (GIL released, or the
        // free-threaded equivalent) — only the numpy conversion needs it.
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

/// Structurally scan the file: `{"offsets": ndarray, "n_atoms": ndarray}`.
/// With `with_volume=True` the result also carries `"volumes"`: per-frame
/// `|det(Lattice)|`, `NaN` where a frame has no `Lattice`.
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

/// Random-access reader: scans on construction, then `get(i)` seeks and
/// parses single frames in any order.
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

    /// Per-frame cell volume `|det(Lattice)|` from the scan, or `None` when the
    /// reader was opened without `with_volume`. `NaN` for a frame with no
    /// `Lattice`.
    #[getter]
    fn volumes<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .index()
            .volumes()
            .map(|volumes| volumes.to_vec().into_pyarray(py))
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

/// Read every frame, as a list of per-frame dicts.
///
/// `threads=None` parses on every core; `threads=1` is the exact serial
/// streaming read. Either way the file is read in a single pass; output and
/// errors are identical.
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

/// Gather frames into one batch. `indices=None` reads the whole file in file
/// order; a list gathers those frames (request order, repeats allowed).
///
/// Single pass: the file is read once, and (for a selection) only as far as
/// the last requested frame — bytes past it are never inspected. `threads=None`
/// parses on every core; `threads=1` is fully serial. The resulting batch is
/// identical either way (on a malformed whole-file read the two may report
/// different frames' errors).
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

/// Infer the file's schema as one nested dict — counts, per-column and
/// per-key variant lists with unification verdicts, consistency, and the
/// rendered report — for the Python `Schema` dataclasses to wrap.
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

/// Whether `path` would be read through a decompressing layer (`True`) or as a
/// plain file (`False`), under the given `compression`. The Python layer uses
/// this to refuse random-access batch strategies on a non-seekable source.
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

/// Write a list of `{"n_atoms", "columns", "metadata"}` dicts to `path`.
///
/// Frames are converted to the core model while the GIL is held, then the encode
/// and I/O run with it released. `level` is `0..=9` (codec default when `None`);
/// `append` adds to an existing file where the codec allows it. `threads`
/// spreads serialisation over workers (`None`: every core, `1`: serial); output
/// bytes are identical regardless.
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

/// Incremental writer: build it, `write` frames as they come, then `close`.
/// Backs the `oxyz.Writer` context manager.
///
/// `batch=None` streams each frame straight to the sink in constant memory.
/// `batch=Some(n)` buffers up to `n` frames and serialises each full batch in
/// parallel before writing it — bounded extra memory (one batch), output bytes
/// unchanged.
#[pyclass]
struct FrameWriter {
    // `None` once closed, so a double close or a write-after-close is caught.
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

    /// Serialise and write any buffered frames (batch mode), then clear the
    /// buffer. A no-op when nothing is buffered.
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

/// Build a core `Frame` from a `{"n_atoms", "columns", "metadata"}` dict — the
/// inverse of [`frame_to_pydict`]. Numeric columns arrive as numpy arrays
/// (already coerced to f64/i64/bool by the Python layer), string columns as
/// lists; both inner dicts keep their order.
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
        core_metadata.push((key.extract()?, py_to_value(&value)?));
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

/// One per-atom column. A numpy array is numeric (its dtype picks the kind, its
/// second dimension the width); a list is a string column (nested for width > 1).
fn py_to_column(name: String, value: &Bound<'_, PyAny>) -> PyResult<Column> {
    if value.is_instance_of::<PyList>() {
        let (data, width) = str_column(value.cast::<PyList>()?)?;
        return Ok(Column { name, width, data });
    }
    let (data, width) = numeric_array(value).ok_or_else(|| {
        PyTypeError::new_err(format!(
            "column {name:?} must be a numpy float/int/bool array or a list of strings"
        ))
    })?;
    Ok(Column { name, width, data })
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
        return Ok((ColumnData::Str(flat), 1));
    }

    let rows: Vec<Vec<String>> = list.extract()?;
    let width = rows.first().map_or(1, Vec::len);
    if rows.iter().any(|row| row.len() != width) {
        return Err(PyValueError::new_err(
            "string column rows have differing widths",
        ));
    }
    Ok((ColumnData::Str(rows.into_iter().flatten().collect()), width))
}

/// A numeric column's flat buffer and width, or `None` if `value` is not a 1-D
/// or 2-D f64/i64/bool numpy array.
fn numeric_array(value: &Bound<'_, PyAny>) -> Option<(ColumnData, usize)> {
    flat_array::<f64>(value)
        .map(|(v, w)| (ColumnData::Real(v), w))
        .or_else(|| flat_array::<i64>(value).map(|(v, w)| (ColumnData::Int(v), w)))
        .or_else(|| flat_array::<bool>(value).map(|(v, w)| (ColumnData::Bool(v), w)))
}

/// Flatten a 1-D or 2-D numpy array of `T` (row-major) into a buffer plus its
/// width (the trailing dimension, 1 when 1-D). `None` if the dtype or rank
/// does not match.
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

/// One metadata value. Strings stay strings; numpy arrays become the matching
/// array variant; Python lists become string arrays; bool is tried before int
/// (a Python bool is an int subclass).
fn py_to_value(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    if value.is_instance_of::<PyString>() {
        return Ok(Value::Str(value.extract()?));
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
        return Ok(Value::StrArray(value.extract()?));
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

/// A Rust `Read` over a Python iterator of `bytes` (e.g. obstore's
/// `GetResult.stream()`). Each refill calls `__next__` under the GIL; the parser
/// runs with the GIL released and reacquires here per chunk — negligible against
/// network latency. `Send + Sync` because `Py<PyAny>` is, and the single
/// consumer means the reacquire is never contended.
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

/// A Rust `Read + Seek` over a Python file-like with `read(n)` and
/// `seek(offset, whence)` (obstore's `ReadableFile`). Used for `.zip`, whose
/// central directory is at the end of the object.
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

/// Assemble a `DecodedReader` from a Python source object and a resolved codec.
/// `"plain"/"gzip"/"zstd"` → `source` is a bytes-iterator.
/// `"tar"/"tar.gz"` → `source` is a 0-arg callable returning a fresh bytes-iterator.
/// `"zip"` → `source` is a seekable file-like.
fn build_decoded(
    source: &Bound<'_, PyAny>,
    codec: &str,
    member: Option<&str>,
) -> PyResult<DecodedReader> {
    match codec {
        "plain" | "gzip" | "zstd" => {
            if member.is_some() {
                return Err(PyValueError::new_err(
                    "member= is only valid for an archive (.zip/.tar/.tar.gz) source",
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
        "tar" | "tar.gz" => {
            let gzip = codec == "tar.gz";
            let callable = source.clone().unbind();
            let factory = move || {
                Python::attach(|py| {
                    callable
                        .bind(py)
                        .call0()
                        .map(|iter| {
                            Box::new(PyChunkReader::new(iter.unbind())) as Box<dyn Read + Send>
                        })
                        .map_err(|e| std::io::Error::other(e.to_string()))
                })
            };
            wrap_tar(factory, member, gzip).map_err(extxyz_error_to_py)
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

/// Read every frame from a Python bytes-iterator source, as a list of per-frame dicts.
///
/// `source` must be an iterator yielding `bytes` objects. `codec` is one of
/// `"plain"`, `"gzip"`, `"zstd"`. `threads=None` parses on every core;
/// `threads=1` is the exact serial path. Output is identical either way.
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

/// Read the first frame from a Python bytes-iterator source.
///
/// `source` must be an iterator yielding `bytes` objects. `codec` is one of
/// `"plain"`, `"gzip"`, `"zstd"`.
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
/// Returns one of `"plain"`, `"gzip"`, `"zstd"`, `"tar"`, `"tar.gz"`, `"zip"`.
#[pyfunction]
#[pyo3(signature = (name, head=None))]
fn detect_codec(name: &str, head: Option<&[u8]>) -> String {
    detect_codec_name(name, head).to_owned()
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
        // Source/archive selection errors are about the request, not the
        // contents — plain ValueErrors, no frame location to attach.
        ExtxyzError::MemberNotFound { .. }
        | ExtxyzError::AmbiguousArchive { .. }
        | ExtxyzError::NoExtxyzMember { .. }
        | ExtxyzError::MemberOnNonArchive
        | ExtxyzError::RandomAccessUnsupported
        // Write-side request errors: the data or options are wrong, not the
        // file contents, so they carry no frame location to attach.
        | ExtxyzError::MissingRequiredColumn { .. }
        | ExtxyzError::AppendUnsupported { .. }
        | ExtxyzError::ZstdWriteUnsupported
        | ExtxyzError::InvalidCompressionLevel { .. } => {
            return PyValueError::new_err(error.to_string());
        }
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
    m.add_function(wrap_pyfunction!(is_compressed, m)?)?;
    m.add_function(wrap_pyfunction!(write, m)?)?;
    m.add_function(wrap_pyfunction!(detect_codec, m)?)?;
    m.add_function(wrap_pyfunction!(read_frames_reader, m)?)?;
    m.add_function(wrap_pyfunction!(read_first_frame_reader, m)?)?;
    m.add_function(wrap_pyfunction!(scan_reader, m)?)?;
    m.add_function(wrap_pyfunction!(infer_schema_reader, m)?)?;
    m.add_function(wrap_pyfunction!(read_batch_reader, m)?)?;
    m.add_class::<FrameIter>()?;
    m.add_class::<IndexedFrames>()?;
    m.add_class::<BatchIter>()?;
    m.add_class::<FrameWriter>()?;
    Ok(())
}
