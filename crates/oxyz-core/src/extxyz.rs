//! Lossless extxyz parsing into the columnar data model.
//!
//! The parser accepts and preserves; it does not interpret. Column names are
//! kept as written (no `force`/`forces` aliasing) and metadata values are
//! typed by shape but never renamed, reordered, or converted — `Lattice`
//! stays a flat 9-value array in as-written order. Normalisation is a
//! separate, later layer.

use std::{
    collections::HashMap,
    fs::File,
    io::{self, BufRead, BufReader, Seek, SeekFrom},
    path::Path,
};

use thiserror::Error;

use crate::batch::{Batch, BatchBuilder, BatchError};
use crate::index::{FrameEntry, FrameIndex};
use crate::model::{Column, ColumnData, ColumnKind, Frame, Value};
use crate::schema::Schema;

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

    #[error("invalid Properties descriptor {descriptor:?}: {reason}")]
    InvalidProperties {
        descriptor: String,
        reason: &'static str,
    },

    #[error("unknown Properties kind {kind:?} for column {name:?}")]
    UnknownPropertyKind { name: String, kind: String },

    #[error("invalid Properties width {width:?} for column {name:?}")]
    InvalidPropertyWidth { name: String, width: String },

    #[error("atom line {line_number} has {actual} columns; expected {expected}")]
    WrongAtomColumnCount {
        line_number: usize,
        expected: usize,
        actual: usize,
    },

    #[error("invalid {kind} in column {column:?}: {value:?}")]
    InvalidAtomValue {
        column: String,
        kind: &'static str,
        value: String,
    },

    /// Any parse error from [`FrameIter`], wrapped with the frame it
    /// occurred in.
    #[error("frame {frame_index}: {source}")]
    InFrame {
        frame_index: usize,
        source: Box<ExtxyzError>,
    },

    #[error("frame index {frame_index} out of range: file has {n_frames} frames")]
    FrameOutOfRange { frame_index: usize, n_frames: usize },

    #[error(transparent)]
    Batch(#[from] BatchError),
}

pub type Result<T> = std::result::Result<T, ExtxyzError>;

pub fn read_first_frame(path: impl AsRef<Path>) -> Result<Frame> {
    iter_frames(path)?
        .next()
        .unwrap_or(Err(ExtxyzError::MissingLine("atom count")))
}

pub fn read_frames(path: impl AsRef<Path>) -> Result<Vec<Frame>> {
    iter_frames(path)?.collect()
}

pub fn iter_frames(path: impl AsRef<Path>) -> Result<FrameIter<BufReader<File>>> {
    Ok(FrameIter::new(BufReader::new(File::open(path)?)))
}

/// Infer the whole file's schema. Full-parse fold for now: every frame is
/// parsed and validated, so this doubles as a structural check of the file.
pub fn infer_schema(path: impl AsRef<Path>) -> Result<Schema> {
    let mut schema = Schema::default();

    for frame in iter_frames(path)? {
        schema.observe(&frame?);
    }

    Ok(schema)
}

/// Sequential batches of `frames_per_batch` frames each, streamed in
/// constant memory; the final batch may be smaller.
pub fn iter_batches(
    path: impl AsRef<Path>,
    frames_per_batch: usize,
) -> Result<BatchIter<BufReader<File>>> {
    if frames_per_batch == 0 {
        return Err(BatchError::ZeroFramesPerBatch.into());
    }
    Ok(BatchIter {
        frames: iter_frames(path)?,
        frames_per_batch,
    })
}

/// Chunks a [`FrameIter`] into [`Batch`]es; fused like the frame iterator.
pub struct BatchIter<R: BufRead> {
    frames: FrameIter<R>,
    frames_per_batch: usize,
}

impl<R: BufRead> Iterator for BatchIter<R> {
    type Item = Result<Batch>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut builder = BatchBuilder::new();
        for _ in 0..self.frames_per_batch {
            match self.frames.next() {
                None => break,
                Some(Ok(frame)) => {
                    if let Err(error) = builder.push(frame) {
                        return Some(Err(error.into()));
                    }
                }
                Some(Err(error)) => return Some(Err(error)),
            }
        }

        match builder.finish() {
            Ok(batch) => Some(Ok(batch)),
            // An empty builder just means clean end-of-file.
            Err(BatchError::Empty) => None,
            Err(error) => Some(Err(error.into())),
        }
    }
}

/// Structural scan: record each frame's byte offset and declared atom count
/// without parsing comment or atom lines.
pub fn scan_index(path: impl AsRef<Path>) -> Result<FrameIndex> {
    scan_frames(BufReader::new(File::open(path)?))
}

/// Scan any reader. The count line is trusted, per the format spec: atom
/// lines are skipped blindly, so a lying count desyncs the scan and surfaces
/// as an invalid count line one frame late. Contents are never validated —
/// that is the parser's and [`infer_schema`]'s job.
pub fn scan_frames<R: BufRead>(mut reader: R) -> Result<FrameIndex> {
    let mut entries = Vec::new();
    let mut line = Vec::new();
    let mut offset: u64 = 0;
    let mut line_number: usize = 1;

    loop {
        line.clear();
        let n_read = reader.read_until(b'\n', &mut line)?;
        if n_read == 0 {
            return Ok(FrameIndex::new(entries));
        }

        let count_offset = offset;
        let count_line = line_number;
        offset += n_read as u64;
        line_number += 1;

        let n_atoms = std::str::from_utf8(&line)
            .ok()
            .and_then(|text| text.trim().parse::<usize>().ok())
            .ok_or_else(|| ExtxyzError::InFrame {
                frame_index: entries.len(),
                source: Box::new(ExtxyzError::InvalidAtomCount {
                    line: String::from_utf8_lossy(&line).trim().to_owned(),
                }),
            })?;

        for skipped in 0..=n_atoms {
            line.clear();
            let n_read = reader.read_until(b'\n', &mut line)?;
            if n_read == 0 {
                let label = if skipped == 0 { "comment" } else { "atom" };
                return Err(ExtxyzError::InFrame {
                    frame_index: entries.len(),
                    source: Box::new(ExtxyzError::MissingLine(label)),
                });
            }
            offset += n_read as u64;
            line_number += 1;
        }

        entries.push(FrameEntry {
            offset: count_offset,
            line: count_line,
            n_atoms,
        });
    }
}

/// One frame's bytes, cut out of the stream by a single-pass scan: the
/// count line through the last atom line, plus where it sat in the file.
struct RawFrame {
    frame_index: usize,
    /// 1-based file line number of the count line.
    line: usize,
    bytes: Vec<u8>,
}

/// Single-pass scanner: walks the stream once, yielding the bytes of each
/// selected frame. Unselected frames are skipped without copying, and
/// nothing past the last selected frame is ever read — the partial-read
/// promise. Structural trust matches [`scan_frames`]: only count lines are
/// interpreted. Fused after the first error.
struct RawFrames<R: BufRead> {
    reader: R,
    line_number: usize,
    frame_index: usize,
    /// Sorted, deduplicated selection; `None` selects every frame.
    selection: Option<Vec<usize>>,
    /// Last frame the scan needs; later bytes are never read.
    stop_after: Option<usize>,
    /// Line buffer reused while skipping unselected frames.
    scratch: Vec<u8>,
    fused: bool,
}

impl<R: BufRead> RawFrames<R> {
    /// Only the parallel full read scans without a selection.
    #[cfg(feature = "parallel")]
    fn all(reader: R) -> Self {
        RawFrames {
            reader,
            line_number: 1,
            frame_index: 0,
            selection: None,
            stop_after: None,
            scratch: Vec::new(),
            fused: false,
        }
    }

    fn selecting(reader: R, indices: &[usize]) -> Self {
        let mut selection = indices.to_vec();
        selection.sort_unstable();
        selection.dedup();
        let stop_after = selection.last().copied();
        RawFrames {
            reader,
            line_number: 1,
            frame_index: 0,
            selection: Some(selection),
            stop_after,
            scratch: Vec::new(),
            fused: false,
        }
    }

    fn selected(&self, frame_index: usize) -> bool {
        match &self.selection {
            None => true,
            Some(selection) => selection.binary_search(&frame_index).is_ok(),
        }
    }

    /// First selected frame the scan never reached, knowable only at EOF.
    fn first_unreached(&self) -> Option<usize> {
        self.selection.as_ref().and_then(|selection| {
            let position = selection.partition_point(|&index| index < self.frame_index);
            selection.get(position).copied()
        })
    }

    fn fuse(&mut self, source: ExtxyzError) -> Option<Result<RawFrame>> {
        self.fused = true;
        Some(Err(ExtxyzError::InFrame {
            frame_index: self.frame_index,
            source: Box::new(source),
        }))
    }
}

impl<R: BufRead> Iterator for RawFrames<R> {
    type Item = Result<RawFrame>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.fused {
            return None;
        }

        loop {
            if self.stop_after.is_some_and(|stop| self.frame_index > stop) {
                self.fused = true;
                return None;
            }

            self.scratch.clear();
            let n_read = match self.reader.read_until(b'\n', &mut self.scratch) {
                Ok(n_read) => n_read,
                Err(error) => return self.fuse(error.into()),
            };
            if n_read == 0 {
                self.fused = true;
                // EOF: any selected frame not reached is out of range, and
                // `frame_index` is now the file's true frame count.
                return self.first_unreached().map(|frame_index| {
                    Err(ExtxyzError::FrameOutOfRange {
                        frame_index,
                        n_frames: self.frame_index,
                    })
                });
            }

            let frame_index = self.frame_index;
            let line = self.line_number;
            self.line_number += 1;

            let Some(n_atoms) = std::str::from_utf8(&self.scratch)
                .ok()
                .and_then(|text| text.trim().parse::<usize>().ok())
            else {
                return self.fuse(ExtxyzError::InvalidAtomCount {
                    line: String::from_utf8_lossy(&self.scratch).trim().to_owned(),
                });
            };

            let keep = self.selected(frame_index);
            let mut bytes = if keep {
                std::mem::take(&mut self.scratch)
            } else {
                Vec::new()
            };

            for skipped in 0..=n_atoms {
                let buffer = if keep {
                    &mut bytes
                } else {
                    self.scratch.clear();
                    &mut self.scratch
                };
                let n_read = match self.reader.read_until(b'\n', buffer) {
                    Ok(n_read) => n_read,
                    Err(error) => return self.fuse(error.into()),
                };
                if n_read == 0 {
                    let label = if skipped == 0 { "comment" } else { "atom" };
                    return self.fuse(ExtxyzError::MissingLine(label));
                }
                self.line_number += 1;
            }

            self.frame_index += 1;
            if keep {
                return Some(Ok(RawFrame {
                    frame_index,
                    line,
                    bytes,
                }));
            }
        }
    }
}

/// Parser work-unit sizing: a few frames per unit amortise the `par_bridge`
/// handoff; the byte cap keeps large frames from clumping into one unit.
#[cfg(feature = "parallel")]
const CHUNK_FRAMES: usize = 64;
#[cfg(feature = "parallel")]
const CHUNK_BYTES: usize = 1 << 20;

/// Groups scanned frames into work units for the parallel pipeline. An
/// error ends the stream, but frames cut before it are delivered first.
#[cfg(feature = "parallel")]
struct RawChunks<R: BufRead> {
    raw: RawFrames<R>,
    pending_error: Option<ExtxyzError>,
}

#[cfg(feature = "parallel")]
impl<R: BufRead> Iterator for RawChunks<R> {
    type Item = Result<Vec<RawFrame>>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(error) = self.pending_error.take() {
            return Some(Err(error));
        }

        let mut chunk = Vec::new();
        let mut chunk_bytes = 0;
        while chunk.len() < CHUNK_FRAMES && chunk_bytes < CHUNK_BYTES {
            match self.raw.next() {
                Some(Ok(raw)) => {
                    chunk_bytes += raw.bytes.len();
                    chunk.push(raw);
                }
                Some(Err(error)) => {
                    if chunk.is_empty() {
                        return Some(Err(error));
                    }
                    self.pending_error = Some(error);
                    break;
                }
                None => break,
            }
        }

        if chunk.is_empty() {
            None
        } else {
            Some(Ok(chunk))
        }
    }
}

/// Parse results tagged with their file frame index.
#[cfg(feature = "parallel")]
type TaggedFrames = Vec<(usize, Result<Frame>)>;

/// Per-frame parse results, plus the scan error that ended the stream.
#[cfg(feature = "parallel")]
type PipelineOutcome = (TaggedFrames, Option<ExtxyzError>);

/// Drive a scanner through the parse workers in one pass: whichever worker
/// is idle pulls the next chunk from the shared scan (`par_bridge`), so
/// scanning overlaps parsing and `threads` is the total thread count — the
/// scan has no thread of its own.
///
/// The scanner is fused, so at most one scan error exists and every parsed
/// frame precedes it in the file.
#[cfg(feature = "parallel")]
fn run_pipeline<R: BufRead + Send>(
    raw: RawFrames<R>,
    threads: Option<usize>,
) -> Result<PipelineOutcome> {
    use rayon::prelude::*;

    let outcome: Vec<Result<TaggedFrames>> = with_pool(threads, || {
        RawChunks {
            raw,
            pending_error: None,
        }
        .par_bridge()
        .map(|item| {
            item.map(|chunk| {
                chunk
                    .iter()
                    .map(|raw| (raw.frame_index, parse_raw(raw)))
                    .collect()
            })
        })
        .collect()
    })?;

    let mut parsed = Vec::new();
    let mut scan_error = None;
    for item in outcome {
        match item {
            Ok(chunk) => parsed.extend(chunk),
            Err(error) => scan_error = Some(error),
        }
    }
    Ok((parsed, scan_error))
}

/// Run `op` on a pool of exactly `threads` workers (`None`: the global
/// all-core pool).
#[cfg(feature = "parallel")]
fn with_pool<T: Send>(threads: Option<usize>, op: impl FnOnce() -> T + Send) -> Result<T> {
    match threads {
        None => Ok(op()),
        Some(threads) => Ok(rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|error| ExtxyzError::Io(io::Error::other(error)))?
            .install(op)),
    }
}

/// Gather `indices` (request order, repeats allowed) into one batch, in a
/// single pass that ends at the last requested frame.
///
/// The partial-read promise: bytes past the last requested frame are never
/// read, so structural damage there goes unreported, and contents are
/// validated only for requested frames. Errors resolve in request order —
/// the earliest requested position that is out of range or fails to parse
/// is reported — except that a structural error in the scanned prefix
/// always wins, since nothing after it can be located.
pub fn read_batch(path: impl AsRef<Path>, indices: &[usize]) -> Result<Batch> {
    if indices.is_empty() {
        return Err(BatchError::Empty.into());
    }

    let reader = BufReader::new(File::open(path)?);
    let mut parsed = Vec::new();
    let mut scan_error = None;
    for item in RawFrames::selecting(reader, indices) {
        match item {
            Ok(raw) => parsed.push((raw.frame_index, parse_raw(&raw))),
            Err(error) => scan_error = Some(error),
        }
    }
    assemble_batch(indices, parsed, scan_error)
}

/// [`read_batch`] with the parses spread over `threads` workers (`None`:
/// every core). Output and errors are identical to the serial version.
#[cfg(feature = "parallel")]
pub fn read_batch_parallel(
    path: impl AsRef<Path>,
    indices: &[usize],
    threads: Option<usize>,
) -> Result<Batch> {
    if indices.is_empty() {
        return Err(BatchError::Empty.into());
    }

    let reader = BufReader::new(File::open(path)?);
    let (parsed, scan_error) = run_pipeline(RawFrames::selecting(reader, indices), threads)?;
    assemble_batch(indices, parsed, scan_error)
}

/// Resolve single-pass outcomes against the request, in request order.
fn assemble_batch(
    indices: &[usize],
    parsed: Vec<(usize, Result<Frame>)>,
    scan_error: Option<ExtxyzError>,
) -> Result<Batch> {
    // Out-of-range is only knowable at EOF; the scanner reports it carrying
    // the file's true frame count. Any other scan error is structural and
    // wins outright (the two-pass scan also failed before any parsing).
    let n_frames = match scan_error {
        Some(ExtxyzError::FrameOutOfRange { n_frames, .. }) => Some(n_frames),
        Some(structural) => return Err(structural),
        None => None,
    };

    let mut results: HashMap<usize, Result<Frame>> = parsed.into_iter().collect();
    let mut uses: HashMap<usize, usize> = HashMap::new();
    for &index in indices {
        *uses.entry(index).or_insert(0) += 1;
    }

    let mut builder = BatchBuilder::new();
    for &index in indices {
        if let Some(n_frames) = n_frames {
            if index >= n_frames {
                return Err(ExtxyzError::FrameOutOfRange {
                    frame_index: index,
                    n_frames,
                });
            }
        }

        let remaining = uses.get_mut(&index).expect("counted above");
        *remaining -= 1;

        // Repeats clone the frame; the last use takes ownership.
        let frame = match results.get(&index) {
            Some(Ok(frame)) if *remaining > 0 => frame.clone(),
            _ => results
                .remove(&index)
                .expect("scanner yields every selected in-range frame")?,
        };
        builder.push(frame)?;
    }
    Ok(builder.finish()?)
}

/// Random-access reader: a scanned [`FrameIndex`] plus the open file.
pub struct IndexedFrames {
    file: File,
    /// Kept so parallel reads can open per-worker handles.
    path: std::path::PathBuf,
    index: FrameIndex,
}

impl IndexedFrames {
    /// Scan `path`, keeping the file open for random access.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let index = scan_index(path)?;
        Ok(IndexedFrames {
            file: File::open(path)?,
            path: path.to_owned(),
            index,
        })
    }

    pub fn index(&self) -> &FrameIndex {
        &self.index
    }

    pub fn len(&self) -> usize {
        self.index.n_frames()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Seek to frame `frame_index` and parse it alone. Errors carry that
    /// index; the contents of other frames are never touched.
    pub fn get(&mut self, frame_index: usize) -> Result<Frame> {
        let entry = self
            .index
            .get(frame_index)
            .ok_or(ExtxyzError::FrameOutOfRange {
                frame_index,
                n_frames: self.index.n_frames(),
            })?;
        self.seek_and_parse(entry, frame_index)
    }

    /// Gather the given frames, in order (repeats allowed), into one batch.
    pub fn get_batch(&mut self, indices: &[usize]) -> Result<Batch> {
        let mut builder = BatchBuilder::new();
        for &frame_index in indices {
            builder.push(self.get(frame_index)?)?;
        }
        Ok(builder.finish()?)
    }

    /// `get_batch` with the frame parses spread over worker threads;
    /// `threads` of `None` uses every core. Output and errors are identical
    /// to the serial version.
    #[cfg(feature = "parallel")]
    pub fn get_batch_parallel(&self, indices: &[usize], threads: Option<usize>) -> Result<Batch> {
        let entries = indices
            .iter()
            .map(|&frame_index| {
                self.index
                    .get(frame_index)
                    .map(|entry| (frame_index, entry))
                    .ok_or(ExtxyzError::FrameOutOfRange {
                        frame_index,
                        n_frames: self.index.n_frames(),
                    })
            })
            .collect::<Result<Vec<_>>>()?;

        let frames = parse_entries_parallel(&self.path, &entries, threads)?;
        let mut builder = BatchBuilder::new();
        for frame in frames {
            builder.push(frame)?;
        }
        Ok(builder.finish()?)
    }

    fn seek_and_parse(&mut self, entry: FrameEntry, frame_index: usize) -> Result<Frame> {
        parse_frame_at(&mut self.file, entry, frame_index)
    }
}

/// Seek to one indexed frame and parse it alone; errors carry `frame_index`
/// and file-absolute line numbers, identical to a streamed read's.
fn parse_frame_at(file: &mut File, entry: FrameEntry, frame_index: usize) -> Result<Frame> {
    file.seek(SeekFrom::Start(entry.offset))?;
    let item = FrameIter::starting_at_line(BufReader::new(file), entry.line).next();
    relabel_frame(item, frame_index)
}

/// Parse a frame the scanner cut out of the stream; errors carry the real
/// frame index and file-absolute line numbers, identical to a streamed
/// read's.
fn parse_raw(raw: &RawFrame) -> Result<Frame> {
    let item = FrameIter::starting_at_line(raw.bytes.as_slice(), raw.line).next();
    relabel_frame(item, raw.frame_index)
}

/// Relabel a single-frame parse from the iterator's local index 0 to the
/// real one. `None` means the promised bytes were not there.
fn relabel_frame(item: Option<Result<Frame>>, frame_index: usize) -> Result<Frame> {
    match item {
        Some(Ok(frame)) => Ok(frame),
        Some(Err(ExtxyzError::InFrame { source, .. })) => Err(ExtxyzError::InFrame {
            frame_index,
            source,
        }),
        Some(Err(other)) => Err(other),
        None => Err(ExtxyzError::InFrame {
            frame_index,
            source: Box::new(ExtxyzError::MissingLine("atom count")),
        }),
    }
}

/// `read_frames` parallelised in a single pass over the file: the scan that
/// finds frame boundaries and the parses share the same `threads` workers
/// (see [`run_pipeline`]), so every byte is read exactly once. Output and
/// errors are identical to the serial version: the first error in frame
/// order wins. (This supersedes the two-pass behaviour, where a scan error
/// anywhere in the file preempted parse errors in earlier frames.)
#[cfg(feature = "parallel")]
pub fn read_frames_parallel(path: impl AsRef<Path>, threads: Option<usize>) -> Result<Vec<Frame>> {
    let reader = BufReader::new(File::open(path)?);
    let (mut parsed, scan_error) = run_pipeline(RawFrames::all(reader), threads)?;

    parsed.sort_unstable_by_key(|&(frame_index, _)| frame_index);
    let mut frames = Vec::with_capacity(parsed.len());
    for (_, result) in parsed {
        // First parse error in frame order; a scan error always sits past
        // every parsed frame, so it is checked after.
        frames.push(result?);
    }
    if let Some(error) = scan_error {
        return Err(error);
    }
    Ok(frames)
}

/// Parse tagged index entries on rayon workers, each chunk through its own
/// file handle. Results keep request order; the reported error is the first
/// in request order, exactly as a serial read would raise it.
#[cfg(feature = "parallel")]
fn parse_entries_parallel(
    path: &Path,
    entries: &[(usize, FrameEntry)],
    threads: Option<usize>,
) -> Result<Vec<Frame>> {
    use rayon::prelude::*;

    if entries.is_empty() {
        return Err(BatchError::Empty.into());
    }

    with_pool(threads, || {
        // A few chunks per thread: amortises the per-chunk open() while
        // leaving rayon room to balance.
        let chunk_size = entries
            .len()
            .div_ceil(rayon::current_num_threads() * 4)
            .max(1);

        let results: Vec<Result<Frame>> = entries
            .par_chunks(chunk_size)
            .map(|chunk| parse_chunk(path, chunk))
            .flatten_iter()
            .collect();

        results.into_iter().collect()
    })?
}

#[cfg(feature = "parallel")]
fn parse_chunk(path: &Path, chunk: &[(usize, FrameEntry)]) -> Vec<Result<Frame>> {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(error) => return vec![Err(error.into())],
    };
    chunk
        .iter()
        .map(|&(frame_index, entry)| parse_frame_at(&mut file, entry, frame_index))
        .collect()
}

/// Streaming frame reader: one frame is materialised at a time.
pub struct FrameIter<R: BufRead> {
    lines: io::Lines<R>,
    frame_index: usize,
    /// 1-based file line number of the next unread line, for diagnostics.
    line_number: usize,
    done: bool,
}

impl<R: BufRead> FrameIter<R> {
    pub fn new(reader: R) -> Self {
        FrameIter::starting_at_line(reader, 1)
    }

    /// For seek-based reads: keeps diagnostics in file line numbers even
    /// though the reader starts mid-file.
    pub(crate) fn starting_at_line(reader: R, line_number: usize) -> Self {
        FrameIter {
            lines: reader.lines(),
            frame_index: 0,
            line_number,
            done: false,
        }
    }

    fn try_next_line(&mut self) -> Result<Option<String>> {
        let line = self.lines.next().transpose()?;
        if line.is_some() {
            self.line_number += 1;
        }
        Ok(line)
    }

    fn next_line(&mut self, label: &'static str) -> Result<String> {
        self.try_next_line()?.ok_or(ExtxyzError::MissingLine(label))
    }

    /// Parse one frame, or `None` at clean end-of-file. Anything after a
    /// frame must be a new frame — blank lines in between are an error.
    fn parse_frame(&mut self) -> Result<Option<Frame>> {
        let Some(atom_count_line) = self.try_next_line()? else {
            return Ok(None);
        };

        // Trimmed in the message so streamed and scanned reads of the same
        // bad line raise the identical error.
        let n_atoms =
            atom_count_line
                .trim()
                .parse::<usize>()
                .map_err(|_| ExtxyzError::InvalidAtomCount {
                    line: atom_count_line.trim().to_owned(),
                })?;

        let comment = self.next_line("comment")?;
        let pairs = parse_comment_metadata(&comment)?;

        // `Properties` is consumed into typed columns; every other pair is
        // typed by shape and kept in file order.
        let mut metadata = Vec::with_capacity(pairs.len().saturating_sub(1));
        let mut specs: Option<Vec<PropertySpec>> = None;

        for (key, raw) in pairs {
            if key == "Properties" && specs.is_none() {
                specs = Some(parse_properties(&raw)?);
            } else {
                metadata.push((key, type_metadata_value(&raw)));
            }
        }

        let specs = specs.ok_or(ExtxyzError::MissingMetadata { key: "Properties" })?;

        let mut columns: Vec<Column> = specs
            .into_iter()
            .map(|spec| spec.into_column(n_atoms))
            .collect();
        let row_width: usize = columns.iter().map(|column| column.width).sum();

        for _ in 0..n_atoms {
            let line_number = self.line_number;
            let line = self.next_line("atom")?;
            let cells: Vec<&str> = line.split_whitespace().collect();

            if cells.len() != row_width {
                return Err(ExtxyzError::WrongAtomColumnCount {
                    line_number,
                    expected: row_width,
                    actual: cells.len(),
                });
            }

            let mut cursor = 0;
            for column in &mut columns {
                push_cells(column, &cells[cursor..cursor + column.width])?;
                cursor += column.width;
            }
        }

        Ok(Some(Frame {
            n_atoms,
            columns,
            metadata,
        }))
    }
}

impl<R: BufRead> Iterator for FrameIter<R> {
    type Item = Result<Frame>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        match self.parse_frame() {
            Ok(Some(frame)) => {
                self.frame_index += 1;
                Some(Ok(frame))
            }
            Ok(None) => {
                self.done = true;
                None
            }
            // Stop after the first error: the stream position is no longer
            // trustworthy.
            Err(error) => {
                self.done = true;
                Some(Err(ExtxyzError::InFrame {
                    frame_index: self.frame_index,
                    source: Box::new(error),
                }))
            }
        }
    }
}

/// One `name:kind:width` triplet from the Properties descriptor — the
/// header's promise, as distinct from the materialised [`Column`].
struct PropertySpec {
    name: String,
    kind: ColumnKind,
    width: usize,
}

/// Sanity bound on a column's declared width.
const MAX_COLUMN_WIDTH: usize = 1 << 16;

/// Cap on pre-allocation from the *declared* atom count, which is untrusted
/// input: a corrupt count must not trigger an absurd allocation before any
/// data is read. Buffers still grow normally past this if the file really is
/// that large.
const MAX_PREALLOC: usize = 1 << 20;

impl PropertySpec {
    /// Materialise an empty column, pre-sized for the whole frame.
    fn into_column(self, n_atoms: usize) -> Column {
        let capacity = n_atoms.saturating_mul(self.width).min(MAX_PREALLOC);
        let data = match self.kind {
            ColumnKind::Real => ColumnData::Real(Vec::with_capacity(capacity)),
            ColumnKind::Int => ColumnData::Int(Vec::with_capacity(capacity)),
            ColumnKind::Bool => ColumnData::Bool(Vec::with_capacity(capacity)),
            ColumnKind::Str => ColumnData::Str(Vec::with_capacity(capacity)),
        };

        Column {
            name: self.name,
            width: self.width,
            data,
        }
    }
}

fn parse_properties(descriptor: &str) -> Result<Vec<PropertySpec>> {
    let parts: Vec<&str> = descriptor.split(':').collect();

    if parts.len() % 3 != 0 {
        return Err(ExtxyzError::InvalidProperties {
            descriptor: descriptor.to_owned(),
            reason: "expected name:kind:width triplets",
        });
    }

    parts
        .chunks_exact(3)
        .map(|triplet| {
            let (name, kind, width) = (triplet[0], triplet[1], triplet[2]);

            if name.is_empty() {
                return Err(ExtxyzError::InvalidProperties {
                    descriptor: descriptor.to_owned(),
                    reason: "empty column name",
                });
            }

            let kind = match kind {
                "R" => ColumnKind::Real,
                "I" => ColumnKind::Int,
                "L" => ColumnKind::Bool,
                "S" => ColumnKind::Str,
                _ => {
                    return Err(ExtxyzError::UnknownPropertyKind {
                        name: name.to_owned(),
                        kind: kind.to_owned(),
                    });
                }
            };

            let parsed_width = match width.parse::<usize>() {
                Ok(parsed) if (1..=MAX_COLUMN_WIDTH).contains(&parsed) => parsed,
                _ => {
                    return Err(ExtxyzError::InvalidPropertyWidth {
                        name: name.to_owned(),
                        width: width.to_owned(),
                    });
                }
            };

            Ok(PropertySpec {
                name: name.to_owned(),
                kind,
                width: parsed_width,
            })
        })
        .collect()
}

/// Append one atom's cells onto the column's buffer.
fn push_cells(column: &mut Column, cells: &[&str]) -> Result<()> {
    match &mut column.data {
        ColumnData::Real(buffer) => {
            for cell in cells {
                buffer.push(
                    cell.parse::<f64>()
                        .map_err(|_| invalid_cell(&column.name, "real", cell))?,
                );
            }
        }
        ColumnData::Int(buffer) => {
            for cell in cells {
                buffer.push(
                    cell.parse::<i64>()
                        .map_err(|_| invalid_cell(&column.name, "int", cell))?,
                );
            }
        }
        ColumnData::Bool(buffer) => {
            for cell in cells {
                buffer
                    .push(bool_cell(cell).ok_or_else(|| invalid_cell(&column.name, "bool", cell))?);
            }
        }
        ColumnData::Str(buffer) => {
            for cell in cells {
                buffer.push((*cell).to_owned());
            }
        }
    }

    Ok(())
}

fn invalid_cell(column: &str, kind: &'static str, value: &str) -> ExtxyzError {
    ExtxyzError::InvalidAtomValue {
        column: column.to_owned(),
        kind,
        value: value.to_owned(),
    }
}

/// `0`/`1` are valid here because the `L` kind removes the ambiguity; on the
/// comment line a bare `1` must stay an integer (see [`bool_token`]).
fn bool_cell(cell: &str) -> Option<bool> {
    match cell {
        "T" | "TRUE" | "True" | "true" | "1" => Some(true),
        "F" | "FALSE" | "False" | "false" | "0" => Some(false),
        _ => None,
    }
}

/// Tokenize the comment line into ordered `(key, raw value)` pairs; file
/// order and duplicate keys are preserved.
fn parse_comment_metadata(comment: &str) -> Result<Vec<(String, String)>> {
    let bytes = comment.as_bytes();
    let mut pairs = Vec::new();
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

        pairs.push((key.to_owned(), value.to_owned()));
    }

    Ok(pairs)
}

fn slice_comment(comment: &str, start: usize, end: usize) -> Result<&str> {
    comment
        .get(start..end)
        .ok_or(ExtxyzError::InvalidMetadata { index: start })
}

/// Type a raw comment-line value by its shape, falling back to `Str` when
/// nothing more specific fits, so typing never rejects a file. Quoting does
/// not influence typing: `Lattice="9 0 0 ..."` must become numbers.
fn type_metadata_value(raw: &str) -> Value {
    if let Some(array) = parse_bracket_array(raw) {
        return array;
    }

    let tokens: Vec<&str> = raw.split_whitespace().collect();

    match tokens.as_slice() {
        // Empty (e.g. a quoted "") or all-whitespace value.
        [] => Value::Str(raw.to_owned()),
        [token] => scalar_value(token, raw),
        _ => whitespace_array_value(&tokens, raw),
    }
}

fn scalar_value(token: &str, raw: &str) -> Value {
    // Integers before booleans so `1` stays Int; `bool_token` excludes 0/1.
    if let Ok(int) = token.parse::<i64>() {
        return Value::Int(int);
    }

    if let Ok(real) = token.parse::<f64>() {
        return Value::Real(real);
    }

    match bool_token(token) {
        Some(boolean) => Value::Bool(boolean),
        None => Value::Str(raw.to_owned()),
    }
}

fn whitespace_array_value(tokens: &[&str], raw: &str) -> Value {
    // All-integer stays IntArray; promotion to floats is the normalisation
    // layer's call.
    if let Some(ints) = parse_all::<i64>(tokens) {
        return Value::IntArray(ints);
    }

    if let Some(reals) = parse_all::<f64>(tokens) {
        return Value::RealArray(reals);
    }

    if let Some(bools) = tokens.iter().map(|token| bool_token(token)).collect() {
        return Value::BoolArray(bools);
    }

    // Mixed tokens are a sentence, not an array.
    Value::Str(raw.to_owned())
}

/// Parse every token as `T`, or `None` on the first failure.
fn parse_all<T: std::str::FromStr>(tokens: &[&str]) -> Option<Vec<T>> {
    tokens.iter().map(|token| token.parse::<T>().ok()).collect()
}

/// Comment-line booleans; excludes `0`/`1` (contrast [`bool_cell`]).
fn bool_token(token: &str) -> Option<bool> {
    match token {
        "T" | "TRUE" | "True" | "true" => Some(true),
        "F" | "FALSE" | "False" | "false" => Some(false),
        _ => None,
    }
}

/// Parse a new-style bracket array like `[2,2,1]` or `["slab","relaxed"]`.
/// Returns `None` for anything else — including nested 2-D arrays, for now —
/// so the caller falls through to the `Str` fallback.
fn parse_bracket_array(raw: &str) -> Option<Value> {
    let inner = raw.strip_prefix('[')?.strip_suffix(']')?;

    if inner.contains('[') || inner.contains(']') {
        return None;
    }

    if inner.trim().is_empty() {
        return None;
    }

    let elements: Vec<&str> = inner.split(',').map(str::trim).collect();

    if elements.iter().any(|element| element.is_empty()) {
        return None;
    }

    if let Some(ints) = parse_all::<i64>(&elements) {
        return Some(Value::IntArray(ints));
    }

    if let Some(reals) = parse_all::<f64>(&elements) {
        return Some(Value::RealArray(reals));
    }

    if let Some(bools) = elements.iter().map(|element| bool_token(element)).collect() {
        return Some(Value::BoolArray(bools));
    }

    Some(Value::StrArray(
        elements
            .iter()
            .map(|element| strip_quotes(element).to_owned())
            .collect(),
    ))
}

fn strip_quotes(token: &str) -> &str {
    token
        .strip_prefix('"')
        .and_then(|stripped| stripped.strip_suffix('"'))
        .unwrap_or(token)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn types_scalar_metadata() {
        assert_eq!(type_metadata_value("12"), Value::Int(12));
        assert_eq!(type_metadata_value("298.15"), Value::Real(298.15));
        assert_eq!(type_metadata_value("T"), Value::Bool(true));
        assert_eq!(type_metadata_value("False"), Value::Bool(false));
        assert_eq!(type_metadata_value("train"), Value::Str("train".to_owned()));

        // "1" is an integer, not a boolean, on the comment line.
        assert_eq!(type_metadata_value("1"), Value::Int(1));
    }

    #[test]
    fn types_whitespace_separated_arrays() {
        assert_eq!(type_metadata_value("3 0 0"), Value::IntArray(vec![3, 0, 0]));
        assert_eq!(
            type_metadata_value("1.0 2.5"),
            Value::RealArray(vec![1.0, 2.5])
        );
        // Mixed int/real promotes to reals.
        assert_eq!(
            type_metadata_value("1 2.5"),
            Value::RealArray(vec![1.0, 2.5])
        );
        assert_eq!(
            type_metadata_value("T T F"),
            Value::BoolArray(vec![true, true, false])
        );
        // Mixed tokens are a sentence, kept whole.
        assert_eq!(
            type_metadata_value("water monomer"),
            Value::Str("water monomer".to_owned())
        );
    }

    #[test]
    fn types_bracket_arrays() {
        assert_eq!(
            type_metadata_value("[2,2,1]"),
            Value::IntArray(vec![2, 2, 1])
        );
        assert_eq!(
            type_metadata_value("[4.5,5.0]"),
            Value::RealArray(vec![4.5, 5.0])
        );
        assert_eq!(
            type_metadata_value(r#"["slab","relaxed"]"#),
            Value::StrArray(vec!["slab".to_owned(), "relaxed".to_owned()])
        );
        // 2-D arrays fall back to the raw string for now.
        assert_eq!(
            type_metadata_value("[[1,0],[0,1]]"),
            Value::Str("[[1,0],[0,1]]".to_owned())
        );
    }

    #[test]
    fn parses_properties_descriptor() {
        let specs = parse_properties("species:S:1:pos:R:3:selection:I:1:tagged:L:1").unwrap();

        let summary: Vec<(&str, ColumnKind, usize)> = specs
            .iter()
            .map(|spec| (spec.name.as_str(), spec.kind, spec.width))
            .collect();

        assert_eq!(
            summary,
            [
                ("species", ColumnKind::Str, 1),
                ("pos", ColumnKind::Real, 3),
                ("selection", ColumnKind::Int, 1),
                ("tagged", ColumnKind::Bool, 1),
            ]
        );
    }

    #[test]
    fn rejects_malformed_properties_descriptors() {
        assert!(matches!(
            parse_properties("species:S"),
            Err(ExtxyzError::InvalidProperties { .. })
        ));
        assert!(matches!(
            parse_properties("pos:Q:3"),
            Err(ExtxyzError::UnknownPropertyKind { .. })
        ));
        assert!(matches!(
            parse_properties("pos:R:0"),
            Err(ExtxyzError::InvalidPropertyWidth { .. })
        ));
        assert!(matches!(
            parse_properties("pos:R:three"),
            Err(ExtxyzError::InvalidPropertyWidth { .. })
        ));
    }
}
