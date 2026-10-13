//! Lossless extxyz parsing into the columnar data model.
//!
//! Names and values are kept as written: no aliasing, renaming, or
//! conversion. `Lattice` stays a flat 9-value array.

use std::{
    collections::HashMap,
    fs::File,
    io::{self, BufRead, BufReader, Seek, SeekFrom},
    path::Path,
};

use compact_str::CompactString;
use thiserror::Error;

use crate::batch::{Batch, BatchBuilder, BatchError};
use crate::decode::{Compression, DecodedReader, open_decoded};
use crate::index::{FrameEntry, FrameIndex};
use crate::model::{Column, ColumnData, ColumnKind, Frame, Value};
use crate::project::{Deviation, ProjectionPlan, project_frame};
use crate::schema::Schema;

#[derive(Debug, Error)]
pub enum ExtxyzError {
    #[error("I/O error")]
    Io(#[from] io::Error),

    #[error("missing {0} line")]
    MissingLine(&'static str),

    #[error("invalid atom count: expected a non-negative integer, found {line:?}")]
    InvalidAtomCount { line: String },

    #[error(
        "invalid comment metadata near byte {index}: expected 'key=value' pairs, quoting values that contain spaces"
    )]
    InvalidMetadata { index: usize },

    #[error("missing metadata key {key:?}")]
    MissingMetadata { key: &'static str },

    #[error("invalid Properties descriptor {descriptor:?}: {reason}")]
    InvalidProperties {
        descriptor: String,
        reason: &'static str,
    },

    #[error("unknown Properties kind {kind:?} for column {name:?}; expected one of S, I, R, L")]
    UnknownPropertyKind { name: String, kind: String },

    #[error(
        "invalid Properties width {width:?} for column {name:?}; expected an integer in 1..=65536"
    )]
    InvalidPropertyWidth { name: String, width: String },

    #[error("wrong column count: {actual} columns, expected {expected}")]
    WrongAtomColumnCount { expected: usize, actual: usize },

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

    /// A parse error at a 1-based file line and, where known, character column.
    #[error("line {line}{}: {source}", .column.map(|c| format!(", column {c}")).unwrap_or_default())]
    Located {
        line: usize,
        column: Option<usize>,
        source: Box<ExtxyzError>,
    },

    #[error("frame index {frame_index} out of range: file has {n_frames} frames")]
    FrameOutOfRange { frame_index: usize, n_frames: usize },

    #[error("member {member:?} not found in archive; available members: {available:?}")]
    MemberNotFound {
        member: String,
        available: Vec<String>,
    },

    #[error("archive has multiple extxyz members; pass member= to choose one: {members:?}")]
    AmbiguousArchive { members: Vec<String> },

    #[error("archive contains no extxyz member; members: {members:?}")]
    NoExtxyzMember { members: Vec<String> },

    #[error("member= given for a non-archive source")]
    MemberOnNonArchive,

    #[error(
        "random access is unsupported on a compressed source; decompress first or use streaming reads"
    )]
    RandomAccessUnsupported,

    #[error("frame is missing required column {name:?}; extxyz needs both 'species' and 'pos'")]
    MissingRequiredColumn { name: &'static str },

    #[error("append mode is not supported for {codec} output")]
    AppendUnsupported { codec: &'static str },

    #[error("writing zstd output (.zst, .tar.zst) is not yet supported")]
    ZstdWriteUnsupported,

    #[error("compression level {level} is out of range; expected 0..=9")]
    InvalidCompressionLevel { level: i32 },

    #[error(transparent)]
    Batch(#[from] BatchError),
}

pub type Result<T> = std::result::Result<T, ExtxyzError>;

impl ExtxyzError {
    /// Index of the frame the error occurred in, if it is tied to one.
    pub fn frame_index(&self) -> Option<usize> {
        match self {
            ExtxyzError::InFrame { frame_index, .. }
            | ExtxyzError::FrameOutOfRange { frame_index, .. } => Some(*frame_index),
            _ => None,
        }
    }

    /// File-absolute 1-based line the error pins down, if any.
    pub fn line(&self) -> Option<usize> {
        match self.located() {
            Some(ExtxyzError::Located { line, .. }) => Some(*line),
            _ => None,
        }
    }

    /// 1-based character column of the offending token within its line, if known.
    pub fn column(&self) -> Option<usize> {
        match self.located() {
            Some(ExtxyzError::Located { column, .. }) => *column,
            _ => None,
        }
    }

    /// The `Located` layer, if present, peeling any `InFrame` framing.
    fn located(&self) -> Option<&ExtxyzError> {
        let mut cursor = self;
        loop {
            match cursor {
                ExtxyzError::InFrame { source, .. } => cursor = source,
                located @ ExtxyzError::Located { .. } => return Some(located),
                _ => return None,
            }
        }
    }
}

/// Wrap `source` with its source location.
fn at(line: usize, column: Option<usize>, source: ExtxyzError) -> ExtxyzError {
    ExtxyzError::Located {
        line,
        column,
        source: Box::new(source),
    }
}

/// 1-based character column of `byte` in `line`; byte offset + 1 if not UTF-8.
fn char_column(line: &[u8], byte: usize) -> usize {
    match std::str::from_utf8(line.get(..byte).unwrap_or(line)) {
        Ok(prefix) => prefix.chars().count() + 1,
        Err(_) => byte + 1,
    }
}

/// The column a comment-line error points at; only `InvalidMetadata` has one.
fn comment_column(comment: &str, error: &ExtxyzError) -> Option<usize> {
    match error {
        ExtxyzError::InvalidMetadata { index } => Some(char_column(comment.as_bytes(), *index)),
        _ => None,
    }
}

/// Shift an `InvalidMetadata` index from value-relative to line-relative.
fn offset_value_error(error: ExtxyzError, value_offset: usize) -> ExtxyzError {
    match error {
        ExtxyzError::InvalidMetadata { index } => ExtxyzError::InvalidMetadata {
            index: index + value_offset,
        },
        other => other,
    }
}

pub fn read_first_frame(path: impl AsRef<Path>) -> Result<Frame> {
    iter_frames(path)?
        .next()
        .unwrap_or(Err(ExtxyzError::MissingLine("atom count")))
}

pub fn read_frames(path: impl AsRef<Path>) -> Result<Vec<Frame>> {
    iter_frames(path)?.collect()
}

/// Open `path` for streaming, choosing the codec from the extension.
pub fn iter_frames(path: impl AsRef<Path>) -> Result<FrameIter<DecodedReader>> {
    iter_frames_from(open_decoded(path.as_ref(), Compression::Infer, None)?)
}

/// Stream frames from an already-opened reader (e.g. a decoded source).
pub fn iter_frames_from<R: BufRead>(reader: R) -> Result<FrameIter<R>> {
    Ok(FrameIter::new(reader))
}

/// Infer the whole file's schema. Parses every frame, so it also validates.
pub fn infer_schema(path: impl AsRef<Path>) -> Result<Schema> {
    infer_schema_from(open_decoded(path.as_ref(), Compression::Infer, None)?)
}

/// [`infer_schema`] over an already-opened reader.
pub fn infer_schema_from<R: BufRead>(reader: R) -> Result<Schema> {
    let mut schema = Schema::default();

    for frame in FrameIter::new(reader) {
        schema.observe(&frame?);
    }

    Ok(schema)
}

/// Sequential batches of `frames_per_batch` frames each, streamed in
/// constant memory; the final batch may be smaller.
pub fn iter_batches(
    path: impl AsRef<Path>,
    frames_per_batch: usize,
) -> Result<BatchIter<DecodedReader>> {
    iter_batches_from(
        open_decoded(path.as_ref(), Compression::Infer, None)?,
        frames_per_batch,
    )
}

/// [`iter_batches`] over an already-opened reader.
pub fn iter_batches_from<R: BufRead>(reader: R, frames_per_batch: usize) -> Result<BatchIter<R>> {
    if frames_per_batch == 0 {
        return Err(BatchError::ZeroFramesPerBatch.into());
    }
    Ok(BatchIter {
        frames: FrameIter::new(reader),
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
    scan_frames(open_decoded(path.as_ref(), Compression::Infer, None)?)
}

/// [`scan_index`] that also records each frame's cell volume `|det(Lattice)|`.
pub fn scan_index_with_volume(path: impl AsRef<Path>) -> Result<FrameIndex> {
    scan_frames_with_volume(open_decoded(path.as_ref(), Compression::Infer, None)?)
}

/// Scan any reader, trusting count lines and skipping atom lines unread. A
/// wrong count surfaces as an invalid count line one frame later.
pub fn scan_frames<R: BufRead>(reader: R) -> Result<FrameIndex> {
    scan_inner(reader, false)
}

/// [`scan_frames`] that additionally parses each comment line's `Lattice` and
/// records `|det|` as the frame's volume (`NaN` when there is no `Lattice`).
pub fn scan_frames_with_volume<R: BufRead>(reader: R) -> Result<FrameIndex> {
    scan_inner(reader, true)
}

fn scan_inner<R: BufRead>(mut reader: R, with_volume: bool) -> Result<FrameIndex> {
    let mut entries = Vec::new();
    let mut volumes: Vec<f64> = Vec::new();
    let mut line = Vec::new();
    let mut offset: u64 = 0;
    let mut line_number: usize = 1;

    let finish = |entries, volumes| {
        if with_volume {
            Ok(FrameIndex::with_volumes(entries, volumes))
        } else {
            Ok(FrameIndex::new(entries))
        }
    };

    loop {
        line.clear();
        let n_read = reader.read_until(b'\n', &mut line)?;
        if n_read == 0 {
            return finish(entries, volumes);
        }

        let count_offset = offset;
        let count_line = line_number;
        offset += n_read as u64;
        line_number += 1;

        let text = std::str::from_utf8(&line).ok();
        // A blank count line ends the file, as in ASE.
        if text.is_some_and(|text| text.trim().is_empty()) {
            return finish(entries, volumes);
        }
        let n_atoms = text
            .and_then(|text| text.trim().parse::<usize>().ok())
            .ok_or_else(|| ExtxyzError::InFrame {
                frame_index: entries.len(),
                source: Box::new(at(
                    count_line,
                    None,
                    ExtxyzError::InvalidAtomCount {
                        line: String::from_utf8_lossy(&line).trim().to_owned(),
                    },
                )),
            })?;

        if with_volume {
            // Volume is a heuristic: a missing or malformed Lattice gives NaN.
            line.clear();
            let n_comment = reader.read_until(b'\n', &mut line)?;
            if n_comment == 0 {
                return Err(missing_line(entries.len(), line_number, "comment"));
            }
            offset += n_comment as u64;
            line_number += 1;
            let volume = std::str::from_utf8(&line)
                .ok()
                .and_then(lattice_volume)
                .unwrap_or(f64::NAN);
            volumes.push(volume);

            let (n_bytes, skipped) = skip_lines(&mut reader, n_atoms)?;
            offset += n_bytes;
            line_number += skipped;
            if skipped < n_atoms {
                return Err(missing_line(entries.len(), line_number, "atom"));
            }
        } else {
            // Saturating: an untrusted usize::MAX count must not wrap the +1.
            let (n_bytes, skipped) = skip_lines(&mut reader, n_atoms.saturating_add(1))?;
            offset += n_bytes;
            line_number += skipped;
            if skipped <= n_atoms {
                let label = if skipped == 0 { "comment" } else { "atom" };
                return Err(missing_line(entries.len(), line_number, label));
            }
        }

        entries.push(FrameEntry {
            offset: count_offset,
            line: count_line,
            n_atoms,
        });
    }
}

fn missing_line(frame_index: usize, line: usize, label: &'static str) -> ExtxyzError {
    ExtxyzError::InFrame {
        frame_index,
        source: Box::new(at(line, None, ExtxyzError::MissingLine(label))),
    }
}

/// `|det(cell)|` from a comment line's well-formed 9-component `Lattice`.
fn lattice_volume(comment: &str) -> Option<f64> {
    let pairs = parse_comment_metadata(comment).ok()?;
    // Last Lattice wins, matching the metadata map's last-key-wins semantics.
    let raw = pairs
        .iter()
        .rev()
        .find(|(key, _, _)| *key == "Lattice")
        .map(|(_, value, _)| *value)?;
    let mut cell = [0.0f64; 9];
    let mut count = 0usize;
    for token in raw.split_whitespace() {
        if count == 9 {
            return None; // more than nine components: not a 3x3 cell
        }
        cell[count] = token.parse().ok()?;
        count += 1;
    }
    if count != 9 {
        return None;
    }
    Some(det3(&cell).abs())
}

fn det3(m: &[f64; 9]) -> f64 {
    m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
}

/// Skip up to `n` lines without copying; returns bytes consumed and lines
/// skipped. An unterminated final line counts, as with `read_until`.
fn skip_lines<R: BufRead>(reader: &mut R, n: usize) -> io::Result<(u64, usize)> {
    let mut bytes: u64 = 0;
    let mut skipped = 0;
    // Bytes consumed past the last newline: a line still in progress.
    let mut partial = false;

    while skipped < n {
        let buf = reader.fill_buf()?;
        if buf.is_empty() {
            if partial {
                skipped += 1;
            }
            return Ok((bytes, skipped));
        }

        let mut consumed = 0;
        for position in memchr::memchr_iter(b'\n', buf) {
            consumed = position + 1;
            skipped += 1;
            if skipped == n {
                break;
            }
        }
        if skipped < n {
            partial = buf.last() != Some(&b'\n');
            consumed = buf.len();
        }
        reader.consume(consumed);
        bytes += consumed as u64;
    }

    Ok((bytes, skipped))
}

/// One frame's bytes, count line through last atom line.
struct RawFrame {
    frame_index: usize,
    /// 1-based file line number of the count line.
    line: usize,
    bytes: Vec<u8>,
}

/// Single-pass scanner yielding each selected frame's bytes. Nothing past the
/// last selected frame is read. Fused after the first error.
struct RawFrames<R: BufRead> {
    reader: R,
    line_number: usize,
    frame_index: usize,
    /// Sorted, deduplicated selection; `None` selects every frame.
    selection: Option<Vec<usize>>,
    /// Last frame the scan needs; later bytes are never read.
    stop_after: Option<usize>,
    /// Count-line buffer; a selected frame takes it over as its byte buffer.
    scratch: Vec<u8>,
    /// The last frame's length, reserved up front for the next selected frame.
    bytes_hint: usize,
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
            bytes_hint: 0,
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
            bytes_hint: 0,
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

    /// End of input: fuse, and report the first unreached selected frame as out
    /// of range.
    fn end_of_input(&mut self) -> Option<Result<RawFrame>> {
        self.fused = true;
        self.first_unreached().map(|frame_index| {
            Err(ExtxyzError::FrameOutOfRange {
                frame_index,
                n_frames: self.frame_index,
            })
        })
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
                return self.end_of_input();
            }
            // A blank count line ends the file, as in ASE.
            if std::str::from_utf8(&self.scratch).is_ok_and(|text| text.trim().is_empty()) {
                return self.end_of_input();
            }

            let frame_index = self.frame_index;
            let line = self.line_number;
            self.line_number += 1;

            let Some(n_atoms) = std::str::from_utf8(&self.scratch)
                .ok()
                .and_then(|text| text.trim().parse::<usize>().ok())
            else {
                return self.fuse(at(
                    line,
                    None,
                    ExtxyzError::InvalidAtomCount {
                        line: String::from_utf8_lossy(&self.scratch).trim().to_owned(),
                    },
                ));
            };

            if !self.selected(frame_index) {
                // Saturating: an untrusted usize::MAX count must not wrap the +1.
                let skipped = match skip_lines(&mut self.reader, n_atoms.saturating_add(1)) {
                    Ok((_, skipped)) => skipped,
                    Err(error) => return self.fuse(error.into()),
                };
                self.line_number += skipped;
                if skipped <= n_atoms {
                    let label = if skipped == 0 { "comment" } else { "atom" };
                    return self.fuse(at(self.line_number, None, ExtxyzError::MissingLine(label)));
                }
                self.frame_index += 1;
                continue;
            }

            let mut bytes = std::mem::take(&mut self.scratch);
            bytes.reserve(self.bytes_hint.saturating_sub(bytes.len()));
            for skipped in 0..=n_atoms {
                let n_read = match self.reader.read_until(b'\n', &mut bytes) {
                    Ok(n_read) => n_read,
                    Err(error) => return self.fuse(error.into()),
                };
                if n_read == 0 {
                    let label = if skipped == 0 { "comment" } else { "atom" };
                    return self.fuse(at(self.line_number, None, ExtxyzError::MissingLine(label)));
                }
                self.line_number += 1;
            }

            self.bytes_hint = bytes.len();
            self.frame_index += 1;
            return Some(Ok(RawFrame {
                frame_index,
                line,
                bytes,
            }));
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

/// Drive a scanner through the parse workers in one pass. Idle workers pull
/// the next chunk via `par_bridge`, so the scan has no thread of its own.
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
                    .map(|raw| (raw.frame_index, parse_raw_dispatch(raw)))
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
pub(crate) fn with_pool<T: Send>(
    threads: Option<usize>,
    op: impl FnOnce() -> T + Send,
) -> Result<T> {
    match threads {
        None => Ok(op()),
        Some(threads) => Ok(rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|error| ExtxyzError::Io(io::Error::other(error)))?
            .install(op)),
    }
}

/// Gather `indices` (request order, repeats allowed) into one batch.
///
/// Reads nothing past the last requested frame and validates only requested
/// frames. The first error in request order wins, unless a structural error
/// in the scanned prefix comes first.
pub fn read_batch(path: impl AsRef<Path>, indices: &[usize]) -> Result<Batch> {
    if indices.is_empty() {
        return Err(BatchError::Empty.into());
    }
    read_batch_from(
        open_decoded(path.as_ref(), Compression::Infer, None)?,
        indices,
    )
}

/// [`read_batch`] over an already-opened reader.
pub fn read_batch_from<R: BufRead>(reader: R, indices: &[usize]) -> Result<Batch> {
    if indices.is_empty() {
        return Err(BatchError::Empty.into());
    }

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
    read_batch_parallel_from(
        open_decoded(path.as_ref(), Compression::Infer, None)?,
        indices,
        threads,
    )
}

/// [`read_batch_parallel`] over an already-opened reader.
#[cfg(feature = "parallel")]
pub fn read_batch_parallel_from<R: BufRead + Send>(
    reader: R,
    indices: &[usize],
    threads: Option<usize>,
) -> Result<Batch> {
    if indices.is_empty() {
        return Err(BatchError::Empty.into());
    }

    let (parsed, scan_error) = run_pipeline(RawFrames::selecting(reader, indices), threads)?;
    assemble_batch(indices, parsed, scan_error)
}

/// Resolve single-pass outcomes against the request, in request order.
fn assemble_batch(
    indices: &[usize],
    parsed: Vec<(usize, Result<Frame>)>,
    scan_error: Option<ExtxyzError>,
) -> Result<Batch> {
    // Out-of-range is only knowable at EOF; any other scan error wins.
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

/// Read every frame into one [`Batch`], in file order. An empty file yields
/// the empty batch, not an error.
pub fn read_all_batch(path: impl AsRef<Path>) -> Result<Batch> {
    read_all_batch_from(open_decoded(path.as_ref(), Compression::Infer, None)?)
}

/// [`read_all_batch`] over an already-opened reader.
pub fn read_all_batch_from<R: BufRead>(reader: R) -> Result<Batch> {
    let mut builder = BatchBuilder::new();
    for frame in FrameIter::new(reader) {
        builder.push(frame?)?;
    }
    finish_or_empty(builder)
}

/// [`read_all_batch`] with the parses spread over `threads` workers (`None`:
/// every core). On a malformed file it may report a later frame's error than
/// the serial path, which stops at the first bad frame.
#[cfg(feature = "parallel")]
pub fn read_all_batch_parallel(path: impl AsRef<Path>, threads: Option<usize>) -> Result<Batch> {
    read_all_batch_parallel_from(
        open_decoded(path.as_ref(), Compression::Infer, None)?,
        threads,
    )
}

/// [`read_all_batch_parallel`] over an already-opened reader.
#[cfg(feature = "parallel")]
pub fn read_all_batch_parallel_from<R: BufRead + Send>(
    reader: R,
    threads: Option<usize>,
) -> Result<Batch> {
    let mut builder = BatchBuilder::new();
    for frame in read_frames_parallel_from(reader, threads)? {
        builder.push(frame)?;
    }
    finish_or_empty(builder)
}

/// Finish a whole-file builder, mapping an empty file to an empty batch.
fn finish_or_empty(builder: BatchBuilder) -> Result<Batch> {
    match builder.finish() {
        Ok(batch) => Ok(batch),
        Err(BatchError::Empty) => Ok(Batch {
            offsets: vec![0],
            columns: Vec::new(),
            metadata: Vec::new(),
        }),
        Err(error) => Err(error.into()),
    }
}

/// The outcome of a projecting batch read.
pub struct ProjectedBatch {
    /// The concatenation of the frames that survived projection.
    pub batch: Batch,
    /// File indices of the surviving frames, in push order.
    pub survivors: Vec<usize>,
    /// A deviation report per requested frame, in request order.
    pub reports: Vec<(usize, Vec<Deviation>)>,
}

/// Project one frame into an in-progress batch, recording its report and
/// pushing it unless it dropped.
fn project_into(
    builder: &mut BatchBuilder,
    survivors: &mut Vec<usize>,
    reports: &mut Vec<(usize, Vec<Deviation>)>,
    index: usize,
    frame: &Frame,
    plan: &ProjectionPlan,
) -> Result<()> {
    let projected = project_frame(frame, plan);
    reports.push((index, projected.deviations));
    if projected.dropped {
        return Ok(());
    }
    builder.push(projected.frame)?;
    survivors.push(index);
    Ok(())
}

/// [`assemble_batch`] for projected frames, collecting survivors and reports.
fn assemble_projected_batch(
    indices: &[usize],
    parsed: Vec<(usize, Result<Frame>)>,
    scan_error: Option<ExtxyzError>,
    plan: &ProjectionPlan,
) -> Result<ProjectedBatch> {
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
    let mut survivors = Vec::new();
    let mut reports = Vec::new();
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
        project_into(
            &mut builder,
            &mut survivors,
            &mut reports,
            index,
            &frame,
            plan,
        )?;
    }
    let batch = finish_or_empty(builder)?;
    Ok(ProjectedBatch {
        batch,
        survivors,
        reports,
    })
}

/// Projecting analogue of [`read_batch_from`]: gather the selection, project
/// each frame, and return the surviving batch with its report.
pub fn read_batch_projected_from<R: BufRead>(
    reader: R,
    indices: &[usize],
    plan: &ProjectionPlan,
) -> Result<ProjectedBatch> {
    if indices.is_empty() {
        return Err(BatchError::Empty.into());
    }
    let mut parsed = Vec::new();
    let mut scan_error = None;
    for item in RawFrames::selecting(reader, indices) {
        match item {
            Ok(raw) => parsed.push((raw.frame_index, parse_raw(&raw))),
            Err(error) => scan_error = Some(error),
        }
    }
    assemble_projected_batch(indices, parsed, scan_error, plan)
}

/// [`read_batch_projected_from`] with the parses spread over `threads` workers.
#[cfg(feature = "parallel")]
pub fn read_batch_projected_parallel_from<R: BufRead + Send>(
    reader: R,
    indices: &[usize],
    threads: Option<usize>,
    plan: &ProjectionPlan,
) -> Result<ProjectedBatch> {
    if indices.is_empty() {
        return Err(BatchError::Empty.into());
    }
    let (parsed, scan_error) = run_pipeline(RawFrames::selecting(reader, indices), threads)?;
    assemble_projected_batch(indices, parsed, scan_error, plan)
}

/// Projecting analogue of [`read_all_batch_from`]: project every frame in file
/// order into one batch.
pub fn read_all_batch_projected_from<R: BufRead>(
    reader: R,
    plan: &ProjectionPlan,
) -> Result<ProjectedBatch> {
    let mut builder = BatchBuilder::new();
    let mut survivors = Vec::new();
    let mut reports = Vec::new();
    for (index, frame) in FrameIter::new(reader).enumerate() {
        project_into(
            &mut builder,
            &mut survivors,
            &mut reports,
            index,
            &frame?,
            plan,
        )?;
    }
    let batch = finish_or_empty(builder)?;
    Ok(ProjectedBatch {
        batch,
        survivors,
        reports,
    })
}

/// [`read_all_batch_projected_from`] with the parses spread over `threads`
/// workers.
#[cfg(feature = "parallel")]
pub fn read_all_batch_projected_parallel_from<R: BufRead + Send>(
    reader: R,
    threads: Option<usize>,
    plan: &ProjectionPlan,
) -> Result<ProjectedBatch> {
    let mut builder = BatchBuilder::new();
    let mut survivors = Vec::new();
    let mut reports = Vec::new();
    for (index, frame) in read_frames_parallel_from(reader, threads)?
        .into_iter()
        .enumerate()
    {
        project_into(
            &mut builder,
            &mut survivors,
            &mut reports,
            index,
            &frame,
            plan,
        )?;
    }
    let batch = finish_or_empty(builder)?;
    Ok(ProjectedBatch {
        batch,
        survivors,
        reports,
    })
}

/// Projecting analogue of [`BatchIter`]. A window whose frames all dropped is
/// still yielded, so its reports are seen.
pub struct BatchIterProjected<R: BufRead> {
    frames: FrameIter<R>,
    frames_per_batch: usize,
    plan: ProjectionPlan,
    next_index: usize,
}

impl<R: BufRead> Iterator for BatchIterProjected<R> {
    type Item = Result<ProjectedBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut builder = BatchBuilder::new();
        let mut survivors = Vec::new();
        let mut reports = Vec::new();
        let mut pulled = 0usize;
        for _ in 0..self.frames_per_batch {
            match self.frames.next() {
                None => break,
                Some(Ok(frame)) => {
                    let index = self.next_index;
                    self.next_index += 1;
                    pulled += 1;
                    if let Err(error) = project_into(
                        &mut builder,
                        &mut survivors,
                        &mut reports,
                        index,
                        &frame,
                        &self.plan,
                    ) {
                        return Some(Err(error));
                    }
                }
                Some(Err(error)) => return Some(Err(error)),
            }
        }
        if pulled == 0 {
            return None; // clean end-of-file
        }
        match finish_or_empty(builder) {
            Ok(batch) => Some(Ok(ProjectedBatch {
                batch,
                survivors,
                reports,
            })),
            Err(error) => Some(Err(error)),
        }
    }
}

/// [`iter_batches_from`] with projection onto `plan`.
pub fn iter_batches_projected_from<R: BufRead>(
    reader: R,
    frames_per_batch: usize,
    plan: ProjectionPlan,
) -> Result<BatchIterProjected<R>> {
    if frames_per_batch == 0 {
        return Err(BatchError::ZeroFramesPerBatch.into());
    }
    Ok(BatchIterProjected {
        frames: FrameIter::new(reader),
        frames_per_batch,
        plan,
        next_index: 0,
    })
}

/// Random-access reader: a scanned [`FrameIndex`] plus the open file.
pub struct IndexedFrames {
    file: File,
    /// Kept so parallel reads can open per-worker handles.
    path: std::path::PathBuf,
    index: FrameIndex,
    /// Cached pool for parallel `get_batch`, rebuilt when the thread count changes.
    #[cfg(feature = "parallel")]
    pool: Option<(usize, rayon::ThreadPool)>,
}

impl IndexedFrames {
    /// Scan `path`, keeping the file open for random access.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_inner(path.as_ref(), false)
    }

    /// [`open`](Self::open) whose scan also records per-frame cell volumes
    /// (`index().volumes()`), for density-aware batch planning.
    pub fn open_with_volume(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_inner(path.as_ref(), true)
    }

    fn open_inner(path: &Path, with_volume: bool) -> Result<Self> {
        // Random access seeks, which a compressed source cannot do.
        if crate::decode::is_compressed(path, Compression::Infer)? {
            return Err(ExtxyzError::RandomAccessUnsupported);
        }
        let index = if with_volume {
            scan_index_with_volume(path)?
        } else {
            scan_index(path)?
        };
        Ok(IndexedFrames {
            file: File::open(path)?,
            path: path.to_owned(),
            index,
            #[cfg(feature = "parallel")]
            pool: None,
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

    /// `get_batch` with the parses spread over worker threads (`None`: every
    /// core). Output and errors match the serial version. Takes `&mut self` to
    /// reuse one pool across a streamed `iter_batches` loop.
    #[cfg(feature = "parallel")]
    pub fn get_batch_parallel(
        &mut self,
        indices: &[usize],
        threads: Option<usize>,
    ) -> Result<Batch> {
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

        if entries.is_empty() {
            return Err(BatchError::Empty.into());
        }

        let frames = match threads {
            None => parse_entries_on_pool(&self.path, &entries),
            Some(n) => {
                let path = self.path.clone();
                self.worker_pool(n)?
                    .install(|| parse_entries_on_pool(&path, &entries))
            }
        }?;

        let mut builder = BatchBuilder::new();
        for frame in frames {
            builder.push(frame)?;
        }
        Ok(builder.finish()?)
    }

    /// [`get_batch`](Self::get_batch) with each frame projected onto `plan`.
    /// Returns the surviving batch, the file indices that survived, and a
    /// per-requested-frame deviation report.
    pub fn get_batch_projected(
        &mut self,
        indices: &[usize],
        plan: &ProjectionPlan,
    ) -> Result<ProjectedBatch> {
        // An empty request is `Err(Empty)`, as in every other batch reader.
        if indices.is_empty() {
            return Err(BatchError::Empty.into());
        }
        let mut builder = BatchBuilder::new();
        let mut survivors = Vec::new();
        let mut reports = Vec::new();
        for &frame_index in indices {
            let frame = self.get(frame_index)?;
            project_into(
                &mut builder,
                &mut survivors,
                &mut reports,
                frame_index,
                &frame,
                plan,
            )?;
        }
        let batch = finish_or_empty(builder)?;
        Ok(ProjectedBatch {
            batch,
            survivors,
            reports,
        })
    }

    /// [`get_batch_projected`](Self::get_batch_projected) with the frame parses
    /// spread over worker threads; `threads` of `None` uses every core.
    #[cfg(feature = "parallel")]
    pub fn get_batch_projected_parallel(
        &mut self,
        indices: &[usize],
        threads: Option<usize>,
        plan: &ProjectionPlan,
    ) -> Result<ProjectedBatch> {
        if indices.is_empty() {
            return Err(BatchError::Empty.into());
        }
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

        let frames = match threads {
            None => parse_entries_on_pool(&self.path, &entries),
            Some(n) => {
                let path = self.path.clone();
                self.worker_pool(n)?
                    .install(|| parse_entries_on_pool(&path, &entries))
            }
        }?;

        let mut builder = BatchBuilder::new();
        let mut survivors = Vec::new();
        let mut reports = Vec::new();
        for (&(frame_index, _), frame) in entries.iter().zip(frames) {
            project_into(
                &mut builder,
                &mut survivors,
                &mut reports,
                frame_index,
                &frame,
                plan,
            )?;
        }
        let batch = finish_or_empty(builder)?;
        Ok(ProjectedBatch {
            batch,
            survivors,
            reports,
        })
    }

    /// The cached `threads`-wide worker pool, built on first use and rebuilt
    /// only when the requested thread count changes.
    #[cfg(feature = "parallel")]
    fn worker_pool(&mut self, threads: usize) -> Result<&rayon::ThreadPool> {
        if self.pool.as_ref().map(|(count, _)| *count) != Some(threads) {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|error| ExtxyzError::Io(io::Error::other(error)))?;
            self.pool = Some((threads, pool));
        }
        Ok(&self.pool.as_ref().expect("just ensured present").1)
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

/// Parse a frame the scanner cut out, keeping its frame index and file lines.
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

/// A parsed comment line: the `Properties` column specs and the remaining
/// metadata pairs, in file order.
type CommentHeader = (Vec<PropertySpec>, Vec<(CompactString, Value)>);

/// Split a comment line into `Properties` column specs and the remaining
/// metadata, in file order.
fn parse_comment_line(comment: &str) -> Result<CommentHeader> {
    let pairs = parse_comment_metadata(comment)?;

    let mut metadata = Vec::with_capacity(pairs.len().saturating_sub(1));
    let mut specs: Option<Vec<PropertySpec>> = None;
    for (key, raw, value_offset) in pairs {
        if key == "Properties" && specs.is_none() {
            specs = Some(parse_properties(raw)?);
        } else {
            let value =
                parse_metadata_value(raw).map_err(|e| offset_value_error(e, value_offset))?;
            metadata.push((key.into(), value));
        }
    }

    let specs = specs.ok_or(ExtxyzError::MissingMetadata { key: "Properties" })?;
    Ok((specs, metadata))
}

/// Frames at least this large have their atom rows split across workers.
/// Splitting a lone frame breaks even near 100 KB; 256 KB leaves a margin.
#[cfg(feature = "parallel")]
const INTRA_FRAME_BYTES: usize = 256 << 10;

/// Parse a scanned frame, splitting its atom rows across workers when it is
/// large enough to be worth it (see [`INTRA_FRAME_BYTES`]).
#[cfg(feature = "parallel")]
fn parse_raw_dispatch(raw: &RawFrame) -> Result<Frame> {
    if raw.bytes.len() >= INTRA_FRAME_BYTES {
        parse_raw_parallel(raw)
    } else {
        parse_raw(raw)
    }
}

/// Wrap a parse error with the frame it occurred in, matching the framing the
/// streaming reader applies via [`relabel_frame`].
#[cfg(feature = "parallel")]
fn in_frame(frame_index: usize, source: ExtxyzError) -> ExtxyzError {
    ExtxyzError::InFrame {
        frame_index,
        source: Box::new(source),
    }
}

/// Split off the first line of `bytes` — without its `\n` or `\r\n` ending —
/// from the remainder, mirroring [`FrameIter::fill_line`].
#[cfg(feature = "parallel")]
fn split_first_line(bytes: &[u8]) -> (&[u8], &[u8]) {
    match memchr::memchr(b'\n', bytes) {
        Some(pos) => (strip_cr(&bytes[..pos]), &bytes[pos + 1..]),
        None => (strip_cr(bytes), &[]),
    }
}

#[cfg(feature = "parallel")]
fn strip_cr(line: &[u8]) -> &[u8] {
    match line.last() {
        Some(&b'\r') => &line[..line.len() - 1],
        _ => line,
    }
}

/// Count the newline-delimited rows in `region`, matching how
/// [`parse_atom_lines`] iterates: a trailing unterminated line still counts.
#[cfg(feature = "parallel")]
fn count_lines(region: &[u8]) -> usize {
    if region.is_empty() {
        return 0;
    }
    let newlines = memchr::memchr_iter(b'\n', region).count();
    if region.last() == Some(&b'\n') {
        newlines
    } else {
        newlines + 1
    }
}

/// Split `region` at newlines into up to `parts` chunks, each tagged with the
/// index of its first row.
#[cfg(feature = "parallel")]
fn split_atom_ranges(region: &[u8], parts: usize) -> Vec<(&[u8], usize)> {
    if parts <= 1 || region.is_empty() {
        return vec![(region, 0)];
    }

    let target = region.len() / parts;
    let mut ranges = Vec::with_capacity(parts);
    let mut start = 0;
    let mut first_row = 0;
    while ranges.len() < parts - 1 && start < region.len() {
        let cut = (start + target).min(region.len());
        // Extend to the end of the line the cut landed in.
        let end = match memchr::memchr(b'\n', &region[cut..]) {
            Some(pos) => cut + pos + 1,
            None => region.len(),
        };
        let slice = &region[start..end];
        ranges.push((slice, first_row));
        first_row += count_lines(slice);
        start = end;
    }
    if start < region.len() {
        ranges.push((&region[start..], first_row));
    }
    ranges
}

/// Parse the atom rows in `region` into `columns`, as the streaming reader
/// would. `first_line` is the first row's file line number.
#[cfg(feature = "parallel")]
fn parse_atom_lines(
    region: &[u8],
    columns: &mut [Column],
    row_width: usize,
    first_line: usize,
) -> Result<()> {
    let mut cells: Vec<(usize, usize)> = Vec::with_capacity(row_width);
    let mut line_number = first_line;
    let mut start = 0;

    while start < region.len() {
        let line_end =
            memchr::memchr(b'\n', &region[start..]).map_or(region.len(), |pos| start + pos);
        let line = strip_cr(&region[start..line_end]);

        cells.clear();
        let mut i = 0;
        while i < line.len() {
            while i < line.len() && line[i].is_ascii_whitespace() {
                i += 1;
            }
            if i == line.len() {
                break;
            }
            let token_start = i;
            while i < line.len() && !line[i].is_ascii_whitespace() {
                i += 1;
            }
            cells.push((token_start, i));
        }

        if cells.len() != row_width {
            return Err(at(
                line_number,
                None,
                ExtxyzError::WrongAtomColumnCount {
                    expected: row_width,
                    actual: cells.len(),
                },
            ));
        }

        let mut cursor = 0;
        for column in columns.iter_mut() {
            let spans = &cells[cursor..cursor + column.width];
            push_cells(column, spans.iter().map(|&(s, e)| &line[s..e])).map_err(
                |(idx, error)| at(line_number, Some(char_column(line, spans[idx].0)), error),
            )?;
            cursor += column.width;
        }

        line_number += 1;
        start = if line_end < region.len() {
            line_end + 1
        } else {
            region.len()
        };
    }

    Ok(())
}

/// Append `src` onto `dst`; both come from the same specs, so kinds match.
#[cfg(feature = "parallel")]
fn extend_column_data(dst: &mut ColumnData, src: ColumnData) {
    use ColumnData::{Bool, Int, Real, Str};
    match (dst, src) {
        (Real(a), Real(b)) => a.extend(b),
        (Int(a), Int(b)) => a.extend(b),
        (Bool(a), Bool(b)) => a.extend(b),
        (Str(a), Str(b)) => a.extend(b),
        _ => unreachable!("range columns share the spec-derived kind"),
    }
}

/// Parse one frame with its atom rows split across workers. Output and errors
/// are identical to [`parse_raw`].
#[cfg(feature = "parallel")]
fn parse_raw_parallel(raw: &RawFrame) -> Result<Frame> {
    use rayon::prelude::*;

    let frame_index = raw.frame_index;

    // Peel the count and comment lines; the remainder is the atom region.
    let (count_bytes, rest) = split_first_line(&raw.bytes);
    let count_line_text = line_str(count_bytes).map_err(|e| in_frame(frame_index, e))?;
    let n_atoms = count_line_text.trim().parse::<usize>().map_err(|_| {
        in_frame(
            frame_index,
            at(
                raw.line,
                None,
                ExtxyzError::InvalidAtomCount {
                    line: count_line_text.trim().to_owned(),
                },
            ),
        )
    })?;

    if rest.is_empty() {
        return Err(in_frame(
            frame_index,
            at(raw.line + 1, None, ExtxyzError::MissingLine("comment")),
        ));
    }
    let (comment_bytes, atom_region) = split_first_line(rest);
    let comment = line_str(comment_bytes).map_err(|e| in_frame(frame_index, e))?;
    let comment_line = raw.line + 1;
    let (specs, metadata) = parse_comment_line(comment).map_err(|e| {
        in_frame(
            frame_index,
            at(comment_line, comment_column(comment, &e), e),
        )
    })?;
    let row_width: usize = specs.iter().map(|spec| spec.width).sum();

    // The first atom row sits two lines past the count line.
    let first_atom_line = raw.line + 2;
    let ranges = split_atom_ranges(atom_region, rayon::current_num_threads());

    let partials: Vec<Result<Vec<Column>>> = ranges
        .par_iter()
        .map(|&(region, first_row)| {
            let mut columns: Vec<Column> = specs
                .iter()
                .map(|spec| spec.column(count_lines(region)))
                .collect();
            parse_atom_lines(region, &mut columns, row_width, first_atom_line + first_row)
                .map_err(|e| in_frame(frame_index, e))?;
            Ok(columns)
        })
        .collect();

    // Results are ordered, so `?` reports the first error in frame order.
    let mut columns: Vec<Column> = specs.iter().map(|spec| spec.column(n_atoms)).collect();
    for partial in partials {
        for (dst, src) in columns.iter_mut().zip(partial?) {
            extend_column_data(&mut dst.data, src.data);
        }
    }

    Ok(Frame {
        n_atoms,
        columns,
        metadata,
    })
}

/// `read_frames` in one parallel pass: the scan and the parses share
/// `threads` workers. Output and errors match the serial version.
#[cfg(feature = "parallel")]
pub fn read_frames_parallel(path: impl AsRef<Path>, threads: Option<usize>) -> Result<Vec<Frame>> {
    read_frames_parallel_from(
        open_decoded(path.as_ref(), Compression::Infer, None)?,
        threads,
    )
}

/// [`read_frames_parallel`] over an already-opened reader.
#[cfg(feature = "parallel")]
pub fn read_frames_parallel_from<R: BufRead + Send>(
    reader: R,
    threads: Option<usize>,
) -> Result<Vec<Frame>> {
    let (mut parsed, scan_error) = run_pipeline(RawFrames::all(reader), threads)?;

    parsed.sort_unstable_by_key(|&(frame_index, _)| frame_index);
    let mut frames = Vec::with_capacity(parsed.len());
    for (_, result) in parsed {
        // A scan error always follows every parsed frame, so it is checked last.
        frames.push(result?);
    }
    if let Some(error) = scan_error {
        return Err(error);
    }
    Ok(frames)
}

/// Parse index entries on the current rayon pool, a file handle per chunk.
/// Results and the reported error follow request order.
#[cfg(feature = "parallel")]
fn parse_entries_on_pool(path: &Path, entries: &[(usize, FrameEntry)]) -> Result<Vec<Frame>> {
    use rayon::prelude::*;

    // A few chunks per thread amortise each open() and leave room to balance.
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
    reader: R,
    frame_index: usize,
    /// 1-based file line number of the next unread line, for diagnostics.
    line_number: usize,
    /// Line buffer reused across the whole stream.
    buffer: Vec<u8>,
    /// Cell offsets within the current line; offsets survive `buffer` refills.
    cells: Vec<(usize, usize)>,
    done: bool,
}

impl<R: BufRead> FrameIter<R> {
    pub fn new(reader: R) -> Self {
        FrameIter::starting_at_line(reader, 1)
    }

    /// For seek-based reads, so diagnostics keep file line numbers.
    pub(crate) fn starting_at_line(reader: R, line_number: usize) -> Self {
        FrameIter {
            reader,
            frame_index: 0,
            line_number,
            buffer: Vec::new(),
            cells: Vec::new(),
            done: false,
        }
    }

    /// Read the next line into the reused buffer, stripping the line ending
    /// (`\n` or `\r\n`, as [`io::Lines`] does). `false` at end-of-file.
    fn fill_line(&mut self) -> Result<bool> {
        self.buffer.clear();
        if self.reader.read_until(b'\n', &mut self.buffer)? == 0 {
            return Ok(false);
        }
        self.line_number += 1;
        if self.buffer.last() == Some(&b'\n') {
            self.buffer.pop();
            if self.buffer.last() == Some(&b'\r') {
                self.buffer.pop();
            }
        }
        Ok(true)
    }

    fn next_line(&mut self, label: &'static str) -> Result<&str> {
        if !self.fill_line()? {
            return Err(ExtxyzError::MissingLine(label));
        }
        line_str(&self.buffer)
    }

    /// Parse one frame, or `None` at clean end-of-file. Anything after a
    /// frame must be a new frame — blank lines in between are an error.
    fn parse_frame(&mut self) -> Result<Option<Frame>> {
        let count_line = self.line_number;
        if !self.fill_line()? {
            return Ok(None);
        }
        let atom_count_line = line_str(&self.buffer)?;

        // A blank count line ends the file, as in ASE.
        if atom_count_line.trim().is_empty() {
            return Ok(None);
        }

        // Trimmed so streamed and scanned reads raise the same error.
        let n_atoms = atom_count_line.trim().parse::<usize>().map_err(|_| {
            at(
                count_line,
                None,
                ExtxyzError::InvalidAtomCount {
                    line: atom_count_line.trim().to_owned(),
                },
            )
        })?;

        let comment_line = self.line_number;
        let comment = self
            .next_line("comment")
            .map_err(|error| at(comment_line, None, error))?;
        let (specs, metadata) = parse_comment_line(comment)
            .map_err(|error| at(comment_line, comment_column(comment, &error), error))?;

        let mut columns: Vec<Column> = specs
            .into_iter()
            .map(|spec| spec.into_column(n_atoms))
            .collect();
        let row_width: usize = columns.iter().map(|column| column.width).sum();

        for _ in 0..n_atoms {
            let line_number = self.line_number;
            if !self.fill_line()? {
                return Err(at(line_number, None, ExtxyzError::MissingLine("atom")));
            }
            // Atom rows are tokenised as bytes; only string cells are checked as UTF-8.
            self.cells.clear();
            let mut i = 0;
            while i < self.buffer.len() {
                while i < self.buffer.len() && self.buffer[i].is_ascii_whitespace() {
                    i += 1;
                }
                if i == self.buffer.len() {
                    break;
                }
                let start = i;
                while i < self.buffer.len() && !self.buffer[i].is_ascii_whitespace() {
                    i += 1;
                }
                self.cells.push((start, i));
            }

            if self.cells.len() != row_width {
                return Err(at(
                    line_number,
                    None,
                    ExtxyzError::WrongAtomColumnCount {
                        expected: row_width,
                        actual: self.cells.len(),
                    },
                ));
            }

            let mut cursor = 0;
            for column in &mut columns {
                let spans = &self.cells[cursor..cursor + column.width];
                push_cells(
                    column,
                    spans.iter().map(|&(start, end)| &self.buffer[start..end]),
                )
                .map_err(|(idx, error)| {
                    at(
                        line_number,
                        Some(char_column(&self.buffer, spans[idx].0)),
                        error,
                    )
                })?;
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
            // The stream position is no longer trustworthy.
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
    name: CompactString,
    kind: ColumnKind,
    width: usize,
}

/// Sanity bound on a column's declared width.
const MAX_COLUMN_WIDTH: usize = 1 << 16;

/// Pre-allocation cap: the declared atom count is untrusted input.
const MAX_PREALLOC: usize = 1 << 20;

impl PropertySpec {
    /// Materialise an empty column, pre-sized for `n_atoms` rows.
    fn into_column(self, n_atoms: usize) -> Column {
        let data = empty_column_data(self.kind, n_atoms.saturating_mul(self.width));
        Column {
            name: self.name,
            width: self.width,
            data,
        }
    }

    /// Like [`into_column`](Self::into_column) but borrowing, so one set of
    /// specs can seed a per-range column buffer in the parallel parser.
    #[cfg(feature = "parallel")]
    fn column(&self, atoms: usize) -> Column {
        Column {
            name: self.name.clone(),
            width: self.width,
            data: empty_column_data(self.kind, atoms.saturating_mul(self.width)),
        }
    }
}

/// An empty column buffer of `kind`, pre-sized to `cells` up to
/// [`MAX_PREALLOC`].
fn empty_column_data(kind: ColumnKind, cells: usize) -> ColumnData {
    let capacity = cells.min(MAX_PREALLOC);
    match kind {
        ColumnKind::Real => ColumnData::Real(Vec::with_capacity(capacity)),
        ColumnKind::Int => ColumnData::Int(Vec::with_capacity(capacity)),
        ColumnKind::Bool => ColumnData::Bool(Vec::with_capacity(capacity)),
        ColumnKind::Str => ColumnData::Str(Vec::with_capacity(capacity)),
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
                name: name.into(),
                kind,
                width: parsed_width,
            })
        })
        .collect()
}

/// View a line buffer as UTF-8, failing as [`io::Lines`] does.
fn line_str(buffer: &[u8]) -> Result<&str> {
    std::str::from_utf8(buffer).map_err(|_| {
        ExtxyzError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            "stream did not contain valid UTF-8",
        ))
    })
}

/// Append one atom's cells onto the column. Only string cells are checked as
/// UTF-8. On failure, also returns the offending cell's index.
fn push_cells<'a>(
    column: &mut Column,
    cells: impl Iterator<Item = &'a [u8]>,
) -> std::result::Result<(), (usize, ExtxyzError)> {
    match &mut column.data {
        ColumnData::Real(buffer) => {
            for (idx, cell) in cells.enumerate() {
                // fast-float2: std's grammar, faster on atom lines.
                buffer.push(
                    fast_float2::parse::<f64, _>(cell)
                        .map_err(|_| (idx, invalid_cell(&column.name, "real", cell)))?,
                );
            }
        }
        ColumnData::Int(buffer) => {
            for (idx, cell) in cells.enumerate() {
                buffer.push(
                    parse_int(cell)
                        .ok_or_else(|| (idx, invalid_cell(&column.name, "int", cell)))?,
                );
            }
        }
        ColumnData::Bool(buffer) => {
            for (idx, cell) in cells.enumerate() {
                buffer.push(
                    bool_cell(cell)
                        .ok_or_else(|| (idx, invalid_cell(&column.name, "bool", cell)))?,
                );
            }
        }
        ColumnData::Str(buffer) => {
            for (idx, cell) in cells.enumerate() {
                buffer.push(line_str(cell).map_err(|e| (idx, e))?.into());
            }
        }
    }

    Ok(())
}

/// Parse an integer cell with std's `i64` grammar, rejecting non-UTF-8.
fn parse_int(cell: &[u8]) -> Option<i64> {
    std::str::from_utf8(cell).ok()?.parse::<i64>().ok()
}

fn invalid_cell(column: &str, kind: &'static str, value: &[u8]) -> ExtxyzError {
    ExtxyzError::InvalidAtomValue {
        column: column.to_owned(),
        kind,
        value: String::from_utf8_lossy(value).into_owned(),
    }
}

/// `0`/`1` are valid here because the `L` kind removes the ambiguity; on the
/// comment line a bare `1` must stay an integer (see [`bool_token`]).
fn bool_cell(cell: &[u8]) -> Option<bool> {
    match cell {
        b"T" | b"TRUE" | b"True" | b"true" | b"1" => Some(true),
        b"F" | b"FALSE" | b"False" | b"false" | b"0" => Some(false),
        _ => None,
    }
}

/// Characters a bare (unquoted) value may not contain.
const BARE_EXCLUDED_CHARS: [char; 8] = ['=', '"', ',', '[', ']', '{', '}', '\\'];

/// Tokenise the comment line into `(key, raw value, value byte offset)`
/// triples, in file order with duplicates kept. Borrowing from `comment`
/// avoids two allocations per key.
fn parse_comment_metadata(comment: &str) -> Result<Vec<(&str, &str, usize)>> {
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
        i += 1;

        if i >= bytes.len() {
            return Err(ExtxyzError::InvalidMetadata { index: i });
        }

        let (value, value_offset) = if bytes[i] == b'"' {
            i += 1;
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
            i += 1;
            (value, value_start)
        } else if bytes[i] == b'{' || bytes[i] == b'[' {
            // A group is one token even with interior whitespace or commas.
            let value_start = i;
            let end = group_end(bytes, i)?;
            i = end;
            (slice_comment(comment, value_start, end)?, value_start)
        } else {
            let value_start = i;

            while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
                i += 1;
            }

            if i == value_start {
                return Err(ExtxyzError::InvalidMetadata { index: i });
            }

            let value = slice_comment(comment, value_start, i)?;
            if let Some(offset) = value.find(BARE_EXCLUDED_CHARS) {
                return Err(ExtxyzError::InvalidMetadata {
                    index: value_start + offset,
                });
            }

            (value, value_start)
        };

        pairs.push((key, value, value_offset));
    }

    Ok(pairs)
}

fn slice_comment(comment: &str, start: usize, end: usize) -> Result<&str> {
    comment
        .get(start..end)
        .ok_or(ExtxyzError::InvalidMetadata { index: start })
}

/// Index just past the group opened at `bytes[open]`, honouring nesting and
/// quotes. `Err(InvalidMetadata { index: open })` if it never closes.
fn group_end(bytes: &[u8], open: usize) -> Result<usize> {
    let mut depth = 0usize;
    let mut i = open;
    let mut in_quote = false;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => in_quote = !in_quote,
            b'{' | b'[' if !in_quote => depth += 1,
            b'}' | b']' if !in_quote => {
                depth -= 1;
                if depth == 0 {
                    return Ok(i + 1);
                }
            }
            _ => {}
        }
        i += 1;
    }
    Err(ExtxyzError::InvalidMetadata { index: open })
}

/// Strictly type a raw comment-line value; error offsets are relative to `raw`.
///
/// The splitter tracks bracket depth only, so a mismatched group like `{1]` is
/// rejected here.
fn parse_metadata_value(raw: &str) -> Result<Value> {
    let trimmed_start = raw.trim_start();
    // New-style bracket array (1-D or 2-D).
    if trimmed_start.starts_with('[') {
        return parse_array_value(raw);
    }
    // Brace group: a scalar `{3}` or a whitespace array `{1 2 3}`.
    if trimmed_start.starts_with('{') {
        return match brace_inner(raw) {
            Some(inner) => parse_group_inner(inner),
            None => Err(ExtxyzError::InvalidMetadata { index: 0 }),
        };
    }
    // Only a quoted value can contain spaces: one token is a scalar, several an
    // array or a sentence.
    let trimmed = raw.trim();
    let mut tokens = trimmed.split_whitespace();
    match (tokens.next(), tokens.next()) {
        (None, _) => Ok(Value::Str(raw.into())),
        (Some(tok), None) => Ok(classify_scalar(tok, raw)),
        (Some(_), Some(_)) => {
            let tokens: Vec<&str> = trimmed.split_whitespace().collect();
            Ok(whitespace_array_value(&tokens, raw))
        }
    }
}

/// Classify a token as int, float (Fortran `d` exponents included), bool, or
/// string. The string fallback keeps `raw`, so a quoted value keeps its
/// interior whitespace.
fn classify_scalar(token: &str, raw: &str) -> Value {
    // Integers before floats so a bare `1` stays Int, not Real.
    if let Ok(int) = token.parse::<i64>() {
        return Value::Int(int);
    }

    if let Some(real) = parse_float_grammar(token) {
        return Value::Real(real);
    }

    match bool_token(token) {
        Some(boolean) => Value::Bool(boolean),
        None => Value::Str(raw.into()),
    }
}

/// A float per the grammar, with Fortran `d`/`D` exponents. Requires a point
/// or exponent, so a bare integer is never read as one.
fn parse_float_grammar(token: &str) -> Option<f64> {
    let has_fortran_exponent = token.bytes().any(|b| matches!(b, b'd' | b'D'));
    if !has_fortran_exponent {
        if !token.contains('.') && !token.contains(['e', 'E']) {
            return None;
        }
        return token.parse::<f64>().ok();
    }
    let normalised = token.replace(['d', 'D'], "e");
    normalised.parse::<f64>().ok()
}

/// `Some(inner)` if `raw` is exactly one `{...}` group spanning the whole
/// value (a scalar `{3}` or a whitespace array `{1 2 3}`), else `None`.
fn brace_inner(raw: &str) -> Option<&str> {
    raw.strip_prefix('{')?.strip_suffix('}')
}

/// Type a `{...}` group's inner content: a single token is a scalar, several
/// whitespace-separated tokens are a whitespace array.
fn parse_group_inner(inner: &str) -> Result<Value> {
    let trimmed = inner.trim();
    let mut tokens = trimmed.split_whitespace();
    match (tokens.next(), tokens.next()) {
        (None, _) => Ok(Value::Str(inner.into())),
        (Some(tok), None) => Ok(classify_scalar(tok, inner)),
        (Some(_), Some(_)) => {
            let tokens: Vec<&str> = trimmed.split_whitespace().collect();
            Ok(whitespace_array_value(&tokens, inner))
        }
    }
}

/// Type a `[...]` array: 1-D, or 2-D when an element is itself a `[...]` row.
/// Splitting is quote- and bracket-aware ([`split_top_level`]).
fn parse_array_value(raw: &str) -> Result<Value> {
    let inner = raw
        .strip_prefix('[')
        .and_then(|rest| rest.strip_suffix(']'))
        .ok_or(ExtxyzError::InvalidMetadata { index: 0 })?;

    let elems = split_top_level(inner);

    // A quoted element such as `"a[b"` starts with `"`, so it stays 1-D.
    if elems.iter().any(|(_, element)| element.starts_with('[')) {
        return parse_array_2d(&elems);
    }

    if let Some((offset, _)) = elems.iter().find(|(_, element)| element.is_empty()) {
        // A leading, trailing, or doubled comma (`[1,2,]`, `[,2,3]`) leaves
        // an empty element; the grammar has no place for one.
        return Err(ExtxyzError::InvalidMetadata { index: 1 + offset });
    }

    let elements: Vec<String> = elems.into_iter().map(|(_, element)| element).collect();
    Ok(array_kind_value(classify_elements(&elements)))
}

/// Type the rows of a 2-D array. Rows must agree in length, and the widest
/// kind across all rows wins.
fn parse_array_2d(elems: &[(usize, String)]) -> Result<Value> {
    let mut rows: Vec<Vec<String>> = Vec::with_capacity(elems.len());
    for (offset, element) in elems {
        let row_inner = element
            .strip_prefix('[')
            .and_then(|rest| rest.strip_suffix(']'))
            .ok_or(ExtxyzError::InvalidMetadata { index: 1 + offset })?;

        // An unquoted `[` in a cell is a 3-D array or a missing separator
        // (`[[1,2][1,2]]`); neither is valid.
        let row_elems = split_top_level(row_inner);
        if let Some((inner_offset, _)) = row_elems
            .iter()
            .find(|(_, cell)| cell.is_empty() || contains_unquoted(cell, '['))
        {
            return Err(ExtxyzError::InvalidMetadata {
                index: 2 + offset + inner_offset,
            });
        }
        rows.push(row_elems.into_iter().map(|(_, cell)| cell).collect());
    }

    let cols = rows.first().map(Vec::len).unwrap_or(0);
    if rows.iter().any(|row| row.len() != cols) {
        return Err(ExtxyzError::InvalidMetadata { index: 0 });
    }

    let n_rows = rows.len();
    let flat: Vec<String> = rows.into_iter().flatten().collect();
    Ok(match classify_elements(&flat) {
        ArrayKind::Int(data) => Value::IntArray2D {
            rows: n_rows,
            cols,
            data,
        },
        ArrayKind::Real(data) => Value::RealArray2D {
            rows: n_rows,
            cols,
            data,
        },
        ArrayKind::Bool(data) => Value::BoolArray2D {
            rows: n_rows,
            cols,
            data,
        },
        ArrayKind::Str(data) => Value::StrArray2D {
            rows: n_rows,
            cols,
            data,
        },
    })
}

fn whitespace_array_value(tokens: &[&str], raw: &str) -> Value {
    // Promotion to floats is the normalisation layer's call.
    if let Some(ints) = parse_all::<i64>(tokens) {
        return Value::IntArray(ints);
    }

    if let Some(reals) = parse_all::<f64>(tokens) {
        return Value::RealArray(reals);
    }

    if let Some(bools) = tokens.iter().map(|token| bool_token(token)).collect() {
        return Value::BoolArray(bools);
    }

    // Only `[...]` produces `StrArray`; a multi-word value is a sentence.
    Value::Str(raw.into())
}

/// Parse every token as `T`, or `None` on the first failure.
fn parse_all<T: std::str::FromStr>(tokens: &[&str]) -> Option<Vec<T>> {
    tokens.iter().map(|token| token.parse::<T>().ok()).collect()
}

/// Comment-line booleans; excludes `0`/`1` (contrast [`bool_cell`]).
fn bool_token(token: &str) -> Option<bool> {
    match token {
        "t" | "T" | "TRUE" | "True" | "true" => Some(true),
        "f" | "F" | "FALSE" | "False" | "false" => Some(false),
        _ => None,
    }
}

/// Split `inner` at top-level commas, treating quoted spans and nested groups
/// as atomic. Each element is trimmed and paired with its byte offset.
fn split_top_level(inner: &str) -> Vec<(usize, String)> {
    let bytes = inner.as_bytes();
    let mut elems = Vec::new();
    let mut start = 0usize;
    let mut depth = 0i32;
    let mut in_quote = false;
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'"' => in_quote = !in_quote,
            b'[' | b'{' if !in_quote => depth += 1,
            b']' | b'}' if !in_quote => depth -= 1,
            b',' if !in_quote && depth == 0 => {
                elems.push(trimmed_span(inner, start, i));
                start = i + 1;
            }
            _ => {}
        }
    }
    elems.push(trimmed_span(inner, start, bytes.len()));
    elems
}

/// `inner[start..end]`, trimmed, paired with the byte offset (within
/// `inner`) of its first non-whitespace character.
fn trimmed_span(inner: &str, start: usize, end: usize) -> (usize, String) {
    let slice = &inner[start..end];
    let leading = slice.len() - slice.trim_start().len();
    (start + leading, slice.trim().to_string())
}

/// An array's element kind, narrowest first. A quoted element such as `"3"`
/// stays a string.
enum ArrayKind {
    Int(Vec<i64>),
    Real(Vec<f64>),
    Bool(Vec<bool>),
    Str(Vec<CompactString>),
}

fn classify_elements(elements: &[String]) -> ArrayKind {
    if let Some(ints) = elements
        .iter()
        .map(|element| element.parse::<i64>().ok())
        .collect::<Option<Vec<_>>>()
    {
        return ArrayKind::Int(ints);
    }

    if let Some(reals) = elements
        .iter()
        .map(|element| element.parse::<f64>().ok())
        .collect::<Option<Vec<_>>>()
    {
        return ArrayKind::Real(reals);
    }

    if let Some(bools) = elements
        .iter()
        .map(|element| bool_token(element))
        .collect::<Option<Vec<_>>>()
    {
        return ArrayKind::Bool(bools);
    }

    ArrayKind::Str(
        elements
            .iter()
            .map(|element| strip_quotes(element).into())
            .collect(),
    )
}

/// A 1-D `ArrayKind` as its matching `Value` variant.
fn array_kind_value(kind: ArrayKind) -> Value {
    match kind {
        ArrayKind::Int(data) => Value::IntArray(data),
        ArrayKind::Real(data) => Value::RealArray(data),
        ArrayKind::Bool(data) => Value::BoolArray(data),
        ArrayKind::Str(data) => Value::StrArray(data),
    }
}

fn strip_quotes(token: &str) -> &str {
    token
        .strip_prefix('"')
        .and_then(|stripped| stripped.strip_suffix('"'))
        .unwrap_or(token)
}

/// Whether `needle` occurs in `haystack` outside any `"..."` span.
fn contains_unquoted(haystack: &str, needle: char) -> bool {
    let mut in_quote = false;
    for c in haystack.chars() {
        match c {
            '"' => in_quote = !in_quote,
            c if c == needle && !in_quote => return true,
            _ => {}
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn types_scalar_metadata() {
        assert_eq!(parse_metadata_value("12").unwrap(), Value::Int(12));
        assert_eq!(parse_metadata_value("298.15").unwrap(), Value::Real(298.15));
        assert_eq!(parse_metadata_value("T").unwrap(), Value::Bool(true));
        assert_eq!(parse_metadata_value("False").unwrap(), Value::Bool(false));
        // Lowercase t/f are also valid booleans, per the grammar.
        assert_eq!(parse_metadata_value("t").unwrap(), Value::Bool(true));
        assert_eq!(parse_metadata_value("f").unwrap(), Value::Bool(false));
        assert_eq!(
            parse_metadata_value("train").unwrap(),
            Value::Str("train".into())
        );

        // "1" is an integer, not a boolean, on the comment line.
        assert_eq!(parse_metadata_value("1").unwrap(), Value::Int(1));

        // Fortran d/D exponents are floats, normalised like e/E.
        assert_eq!(parse_metadata_value("1.5d0").unwrap(), Value::Real(1.5));
        assert_eq!(parse_metadata_value("1.5D-2").unwrap(), Value::Real(0.015));
    }

    #[test]
    fn types_whitespace_separated_arrays() {
        assert_eq!(
            parse_metadata_value("3 0 0").unwrap(),
            Value::IntArray(vec![3, 0, 0])
        );
        assert_eq!(
            parse_metadata_value("1.0 2.5").unwrap(),
            Value::RealArray(vec![1.0, 2.5])
        );
        // Mixed int/real promotes to reals.
        assert_eq!(
            parse_metadata_value("1 2.5").unwrap(),
            Value::RealArray(vec![1.0, 2.5])
        );
        assert_eq!(
            parse_metadata_value("T T F").unwrap(),
            Value::BoolArray(vec![true, true, false])
        );
        // Mixed tokens are a sentence, kept whole.
        assert_eq!(
            parse_metadata_value("water monomer").unwrap(),
            Value::Str("water monomer".into())
        );
    }

    #[test]
    fn types_bracket_arrays() {
        assert_eq!(
            parse_metadata_value("[2,2,1]").unwrap(),
            Value::IntArray(vec![2, 2, 1])
        );
        assert_eq!(
            parse_metadata_value("[4.5,5.0]").unwrap(),
            Value::RealArray(vec![4.5, 5.0])
        );
        assert_eq!(
            parse_metadata_value(r#"["slab","relaxed"]"#).unwrap(),
            Value::StrArray(vec!["slab".into(), "relaxed".into()])
        );
        // A nested `[` makes it 2-D.
        assert_eq!(
            parse_metadata_value("[[1,0],[0,1]]").unwrap(),
            Value::IntArray2D {
                rows: 2,
                cols: 2,
                data: vec![1, 0, 0, 1]
            }
        );
    }

    #[test]
    fn types_2d_bracket_arrays() {
        assert_eq!(
            parse_metadata_value("[[1,2],[3,4]]").unwrap(),
            Value::IntArray2D {
                rows: 2,
                cols: 2,
                data: vec![1, 2, 3, 4]
            }
        );
        // An int row next to a real row promotes the whole array to real.
        assert_eq!(
            parse_metadata_value("[[1,2],[3.0,4]]").unwrap(),
            Value::RealArray2D {
                rows: 2,
                cols: 2,
                data: vec![1.0, 2.0, 3.0, 4.0]
            }
        );
    }

    #[test]
    fn rejects_ragged_2d_arrays() {
        // Row 1 has two elements, row 2 has three: not a rectangle.
        assert!(matches!(
            parse_metadata_value("[[1,2],[3,4,5]]"),
            Err(ExtxyzError::InvalidMetadata { .. })
        ));
    }

    #[test]
    fn rejects_empty_array_elements() {
        // A leading, trailing, or doubled comma leaves an empty element.
        for raw in ["[1,2,]", "[,2,3]", "[1,,3]"] {
            assert!(
                matches!(
                    parse_metadata_value(raw),
                    Err(ExtxyzError::InvalidMetadata { .. })
                ),
                "raw = {raw:?}"
            );
        }
    }

    #[test]
    fn types_brace_wrapped_scalars_and_arrays() {
        // The braces are punctuation, not part of the value.
        assert_eq!(parse_metadata_value("{3}").unwrap(), Value::Int(3));
        assert_eq!(parse_metadata_value("{ 3 }").unwrap(), Value::Int(3));
        assert_eq!(
            parse_metadata_value("{1 2 3}").unwrap(),
            Value::IntArray(vec![1, 2, 3])
        );
    }

    #[test]
    fn splitter_groups_braces_and_brackets() {
        // Interior spaces inside {} or [] do not split the value.
        let pairs = parse_comment_metadata("a={ 3 } b=[ \"x, y\", \"z]\" ]").unwrap();
        assert_eq!(pairs.len(), 2);
        assert_eq!((pairs[0].0, pairs[0].1), ("a", "{ 3 }"));
        assert_eq!((pairs[1].0, pairs[1].1), ("b", "[ \"x, y\", \"z]\" ]"));
    }

    #[test]
    fn splitter_rejects_unbalanced_group() {
        assert!(parse_comment_metadata("a={1 2").is_err());
        assert!(parse_comment_metadata("a=[1, 2").is_err());
    }

    #[test]
    fn typing_rejects_mismatched_bracket_kind() {
        // The splitter accepts `{1]` and `[1}`; typing rejects them.
        assert!(matches!(
            parse_metadata_value("{1]"),
            Err(ExtxyzError::InvalidMetadata { .. })
        ));
        assert!(matches!(
            parse_metadata_value("[1}"),
            Err(ExtxyzError::InvalidMetadata { .. })
        ));
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

    /// One valid frame; the building block for the blank-line cases.
    const FRAME: &str = "1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n";

    fn scan_count(text: &str) -> Result<usize> {
        scan_frames(std::io::Cursor::new(text)).map(|index| index.n_frames())
    }

    fn stream_count(text: &str) -> Result<usize> {
        let mut frames = 0;
        for frame in FrameIter::new(std::io::Cursor::new(text)) {
            frame?;
            frames += 1;
        }
        Ok(frames)
    }

    #[test]
    fn trailing_blank_line_is_end_of_input() {
        for text in [
            format!("{FRAME}\n"),
            format!("{FRAME}\n\n"),
            format!("{FRAME}   \n"),
            format!("{FRAME}\t\n"),
        ] {
            assert_eq!(scan_count(&text).unwrap(), 1, "scan: {text:?}");
            assert_eq!(stream_count(&text).unwrap(), 1, "stream: {text:?}");
        }
    }

    #[test]
    fn blank_line_between_frames_stops_the_read() {
        // ASE truncates at the blank: only the frames before it are read.
        let text = format!("{FRAME}\n{FRAME}");
        assert_eq!(scan_count(&text).unwrap(), 1);
        assert_eq!(stream_count(&text).unwrap(), 1);
    }

    #[test]
    fn leading_blank_line_yields_no_frames() {
        let text = format!("\n{FRAME}");
        assert_eq!(scan_count(&text).unwrap(), 0);
        assert_eq!(stream_count(&text).unwrap(), 0);
    }

    #[test]
    fn non_blank_bad_count_still_errors() {
        // Only blank lines end the file; anything else is an error.
        let text = format!("{FRAME}xyz\n{FRAME}");
        assert!(scan_count(&text).is_err());
        assert!(stream_count(&text).is_err());
    }

    #[test]
    fn raw_frames_treat_blank_as_end_of_input() {
        // The blank ended the file, so frame 1 is out of range.
        let text = format!("{FRAME}\n{FRAME}");
        let mut frames = RawFrames::selecting(std::io::Cursor::new(text), &[0, 1]);
        assert!(frames.next().unwrap().is_ok(), "frame 0 reads");
        // `RawFrame` is not `Debug`, so match rather than `unwrap_err`.
        let error = match frames.next().unwrap() {
            Err(error) => error,
            Ok(_) => panic!("frame past the blank should be out of range"),
        };
        assert!(error.to_string().contains("out of range"), "{error}");
        assert!(frames.next().is_none(), "fused after the error");
    }

    #[test]
    fn usize_max_count_does_not_overflow() {
        // The +1 for the comment line must not overflow on usize::MAX.
        let text = format!("{}\n", usize::MAX);
        assert!(scan_frames(std::io::Cursor::new(text.as_bytes())).is_err());

        let mut streamed = FrameIter::new(std::io::Cursor::new(text.as_bytes()));
        assert!(matches!(streamed.next(), Some(Err(_))));

        let mut batch = RawFrames::selecting(std::io::Cursor::new(text.as_bytes()), &[0]);
        assert!(matches!(batch.next(), Some(Err(_))));
    }

    #[test]
    fn non_utf8_in_a_string_cell_errors() {
        // A non-UTF-8 species cell is a clean error, not a panic or a mangled atom.
        let mut bytes = b"1\nProperties=species:S:1:pos:R:3\n".to_vec();
        bytes.extend_from_slice(&[0xFF, b' ', b'0', b' ', b'0', b' ', b'0', b'\n']);
        let mut frames = FrameIter::new(std::io::Cursor::new(bytes));
        assert!(matches!(frames.next(), Some(Err(_))));
    }

    #[test]
    fn non_utf8_in_a_numeric_cell_errors() {
        // A non-UTF-8 numeric cell fails as an invalid value.
        let mut bytes = b"1\nProperties=species:S:1:pos:R:3\nH ".to_vec();
        bytes.extend_from_slice(&[0xFF, b' ', b'0', b' ', b'0', b'\n']);
        let mut frames = FrameIter::new(std::io::Cursor::new(bytes));
        assert!(matches!(frames.next(), Some(Err(_))));
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn count_lines_counts_an_unterminated_final_row() {
        assert_eq!(count_lines(b""), 0);
        assert_eq!(count_lines(b"a\n"), 1);
        assert_eq!(count_lines(b"a\nb\n"), 2);
        assert_eq!(count_lines(b"a\nb"), 2); // a trailing line without \n counts
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn split_atom_ranges_covers_the_region_on_line_boundaries() {
        let region = b"a\nbb\nccc\ndddd\n";

        assert_eq!(split_atom_ranges(region, 1), vec![(&region[..], 0)]);

        let ranges = split_atom_ranges(region, 4);
        let mut joined = Vec::new();
        let mut first_row = 0;
        for (slice, start_row) in &ranges {
            // Cumulative first-row indices, and no row split across a range.
            assert_eq!(*start_row, first_row);
            assert!(slice.last() == Some(&b'\n'), "range must end on a line");
            first_row += count_lines(slice);
            joined.extend_from_slice(slice);
        }
        assert_eq!(joined, region, "ranges must cover the region exactly");
        assert_eq!(first_row, 4, "every row accounted for");
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// The batch reader's skip and read paths must not panic on a usize::MAX
        /// count.
        #[test]
        fn raw_frames_never_panic_on_declared_counts(
            count in any::<u64>(),
            body in "[ -~\n]{0,200}",
        ) {
            let input = format!("{count}\nProperties=species:S:1:pos:R:3\n{body}");
            // Selecting frame 0 drives the read path; selecting frame 1 forces
            // frame 0 down the skip path.
            for selection in [&[0usize][..], &[1usize][..]] {
                let reader = std::io::Cursor::new(input.as_bytes());
                for item in RawFrames::selecting(reader, selection) {
                    let _ = item;
                }
            }
        }
    }
}
