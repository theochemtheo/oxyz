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

    /// A parse error pinned to its source location: the file-absolute 1-based
    /// line, and where a token is pinpointable the 1-based character column of
    /// it within that line. The single carrier of location; `InFrame` wraps this.
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

    #[error("writing zstd (.zst) output is not yet supported")]
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

/// 1-based character column of byte offset `byte` within `line`. Falls back to
/// the byte offset + 1 when the prefix up to `byte` is not valid UTF-8 (only
/// reachable on already-broken input). Error path only.
fn char_column(line: &[u8], byte: usize) -> usize {
    match std::str::from_utf8(line.get(..byte).unwrap_or(line)) {
        Ok(prefix) => prefix.chars().count() + 1,
        Err(_) => byte + 1,
    }
}

/// The 1-based character column a comment-line error points at, if it carries a
/// byte index into `comment`. Comment errors other than `InvalidMetadata`
/// locate the line only.
fn comment_column(comment: &str, error: &ExtxyzError) -> Option<usize> {
    match error {
        ExtxyzError::InvalidMetadata { index } => Some(char_column(comment.as_bytes(), *index)),
        _ => None,
    }
}

/// Shift a value typer's `InvalidMetadata { index }` — a byte offset relative
/// to the raw value it was given — by that value's byte offset within the
/// whole comment line, so [`comment_column`] locates it correctly. Any other
/// error variant passes through unchanged.
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

/// Open `path` for streaming, detecting and decompressing per the extension.
/// For explicit control over the codec or an archive member, build the reader
/// with [`open_decoded`] and call [`iter_frames_from`].
pub fn iter_frames(path: impl AsRef<Path>) -> Result<FrameIter<DecodedReader>> {
    iter_frames_from(open_decoded(path.as_ref(), Compression::Infer, None)?)
}

/// Stream frames from an already-opened reader (e.g. a decoded source).
pub fn iter_frames_from<R: BufRead>(reader: R) -> Result<FrameIter<R>> {
    Ok(FrameIter::new(reader))
}

/// Infer the whole file's schema. Full-parse fold for now: every frame is
/// parsed and validated, so this doubles as a structural check of the file.
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

/// [`scan_index`] that also records each frame's cell volume `|det(Lattice)|`,
/// reading the one comment line per frame a plain scan skips. For the batched
/// `torch_sim` reader's density-aware binning; off by default so the structural
/// scan stays count-lines-only.
pub fn scan_index_with_volume(path: impl AsRef<Path>) -> Result<FrameIndex> {
    scan_frames_with_volume(open_decoded(path.as_ref(), Compression::Infer, None)?)
}

/// Scan any reader. The count line is trusted, per the format spec: atom
/// lines are skipped blindly, so a lying count desyncs the scan and surfaces
/// as an invalid count line one frame late. Contents are never validated —
/// that is the parser's and [`infer_schema`]'s job.
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
        // A blank line where a count is expected ends the file, as ASE's
        // reader does: trailing blank lines are tolerated and a blank line
        // between frames stops the read.
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
            // Read the comment line so its Lattice can be measured, then skip
            // only the atom lines. A malformed or Lattice-free comment yields
            // NaN: volume is a heuristic, never a reason to fail the scan.
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
            // Saturating: an untrusted count of usize::MAX must not overflow the
            // +1 for the comment line. usize::MAX lines never arrive, so the read
            // hits EOF and reports a missing line rather than wrapping to 0.
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

/// `|det(cell)|` from a comment line's `Lattice`, or `None` when there is no
/// well-formed 9-component `Lattice`. `|det|` is the volume independent of cell
/// handedness and of the row/column order the nine components are read in.
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

/// Skip up to `n` lines without copying them: newlines are counted straight
/// off the reader's internal buffer. Returns the bytes consumed and the
/// lines skipped (fewer than `n` only at end-of-file). Matches the
/// `read_until` view of lines: an unterminated final line counts.
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
    /// Count-line buffer, reused across frames; selected frames take it
    /// over as the start of their byte buffer.
    scratch: Vec<u8>,
    /// Byte length of the last frame emitted: frames in a file run to similar
    /// sizes, so reserving this up front spares a selected frame's buffer the
    /// handful of line-by-line `read_until` reallocations.
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

    /// End of input: fuse, and if a selected frame was never reached report
    /// the first such index as out of range. `frame_index` is the file's true
    /// frame count here. A blank line where a count is expected counts as end
    /// of input, as in ASE.
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
            // A blank line where a count is expected ends the file, as ASE's
            // reader does, so trailing and interspersed blank lines stop the
            // read rather than erroring. Checked before the int parse.
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
                // Skip the frame's lines without copying them anywhere.
                // Saturating: see scan_frames -- an untrusted usize::MAX count
                // must not overflow the +1 for the comment line.
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

/// Read every frame into one [`Batch`], in file order, single pass.
///
/// The whole-file analogue of [`read_batch`]: where that gathers a selection,
/// this concatenates the lot. An empty file yields the empty batch (no frames,
/// no columns), not an error — callers treat it as "no frames".
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
/// every core). Output is identical to the serial version; on a malformed file
/// the two may surface different frames' errors — this parses every frame up
/// front, where the serial path stops at the first bad frame.
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

/// Finish a whole-file builder, mapping the empty-file case to an empty batch
/// rather than [`BatchError::Empty`] (which `BatchIter` uses to mean EOF).
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

// ---- Projecting batch reads --------------------------------------------------
//
// Each frame is projected onto a fixed plan before assembly, so a mixed-schema
// file becomes batchable: undeclared fields are dropped, absent optionals
// filled, and a frame with an unfillable required field is left out entirely.
// These mirror the plain batch readers (reusing the same scan/gather/assemble
// machinery) but carry the survivors and a per-frame deviation report back to
// the caller for policy. The core stays policy-free — it reshapes and reports.

/// The outcome of a projecting batch read.
pub struct ProjectedBatch {
    /// The concatenation of the frames that survived projection.
    pub batch: Batch,
    /// File indices of the surviving frames, in push order.
    pub survivors: Vec<usize>,
    /// A deviation report per requested frame (survivors and drops alike), in
    /// request/file order, so the caller can apply policy in that order.
    pub reports: Vec<(usize, Vec<Deviation>)>,
}

/// Project one frame into an in-progress batch: record its report, and push it
/// (tracking its file index) unless it dropped. Push cannot fail on a projected
/// frame — every survivor shares the plan's shape — but the error is surfaced
/// rather than ignored.
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

/// Resolve projected single-pass outcomes against the request, in request
/// order. Mirrors [`assemble_batch`], but projects each gathered frame and
/// collects survivors and reports instead of demanding a uniform shape.
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

/// Projecting analogue of [`BatchIter`]. Yields every window that pulled at
/// least one frame — even one whose frames all dropped, so the caller still
/// sees the reports; a window that hits EOF without pulling ends iteration.
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
    /// Worker pool for parallel `get_batch`, built once and reused across
    /// calls (one `iter_batches` loop fires `get_batch` per batch). Tagged
    /// with its thread count so a differing request rebuilds it; `None`
    /// requests use rayon's global pool and never populate this.
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
        // Random access seeks into the file, so a non-seekable compressed
        // source is refused here rather than silently reading wrong bytes.
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

    /// `get_batch` with the frame parses spread over worker threads;
    /// `threads` of `None` uses every core. Output and errors are identical
    /// to the serial version.
    ///
    /// Takes `&mut self` to reuse one [`rayon::ThreadPool`] across calls: a
    /// streamed `iter_batches` loop calls this per batch, and rebuilding a
    /// pool each time would spawn and park N threads per batch — pure
    /// overhead that grows with the thread count.
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
            // None runs on rayon's global all-core pool: no per-call pool to
            // build or cache.
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
        // Empty request is `Err(Empty)`, matching every other batch reader
        // (`finish_or_empty` below is for the all-dropped-but-non-empty case).
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

/// A parsed comment line: the `Properties` column specs and the remaining
/// metadata pairs, in file order.
type CommentHeader = (Vec<PropertySpec>, Vec<(CompactString, Value)>);

/// Parse a comment line into its `Properties` column specs and the remaining
/// metadata. `Properties` is consumed into typed columns; every other pair is
/// typed by shape and kept in file order. Shared by the streaming reader and
/// the parallel single-frame parser.
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

/// A frame whose byte length reaches this is parsed with its atom rows split
/// across workers; smaller frames parse on one thread. Set from a sweep of
/// single-frame parse times: splitting an isolated frame breaks even around
/// ~100 KB, and finer-grained range tasks improve pool balance even when many
/// such frames keep every core busy — so the only frames to keep whole are the
/// small ones, where 8-way overhead would dominate. ~256 KB (~5k atoms) sits
/// safely past break-even and above typical small-molecule frames.
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

/// Split `region` into up to `parts` contiguous chunks at newline boundaries,
/// each tagged with the index of its first row (for file-absolute line
/// numbers). A row is never split across chunks.
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

/// Parse the atom rows in `region` into `columns` (appended), tokenising and
/// erroring exactly as the streaming reader does. `first_line` is the file
/// line number of the first row, for diagnostics.
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
            let start = cells[cursor].0;
            push_cells(column, spans.iter().map(|&(s, e)| &line[s..e]))
                .map_err(|error| at(line_number, Some(char_column(line, start)), error))?;
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

/// Append `src` onto `dst`. Range columns are built from the same specs, so
/// their kinds always match — no Int/Real promotion as in cross-frame batches.
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

/// Parse one frame with its atom region split into newline-aligned ranges
/// parsed in parallel, then concatenated in order. Output and errors are
/// identical to [`parse_raw`]: file-absolute line numbers, the real frame
/// index, and the first error in frame order.
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

    // Concatenate ranges in order; the earliest error is the first in frame
    // order, so `?` on the ordered results reports exactly what serial would.
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

/// `read_frames` parallelised in a single pass over the file: the scan that
/// finds frame boundaries and the parses share the same `threads` workers
/// (see [`run_pipeline`]), so every byte is read exactly once. Output and
/// errors are identical to the serial version: the first error in frame
/// order wins. (This supersedes the two-pass behaviour, where a scan error
/// anywhere in the file preempted parse errors in earlier frames.)
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
/// file handle. Runs in the current pool context (the caller installs it, so
/// `rayon::current_num_threads` reports the right width). Results keep request
/// order; the reported error is the first in request order, exactly as a
/// serial read would raise it.
#[cfg(feature = "parallel")]
fn parse_entries_on_pool(path: &Path, entries: &[(usize, FrameEntry)]) -> Result<Vec<Frame>> {
    use rayon::prelude::*;

    // A few chunks per thread: amortises the per-chunk open() while leaving
    // rayon room to balance.
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
    /// Line buffer reused across the whole stream: one allocation total
    /// instead of one `String` per line.
    buffer: Vec<u8>,
    /// Cell boundaries within the current line, reused per atom line.
    /// `(start, end)` offsets rather than `&str`s, so they can outlive
    /// refills of `buffer`.
    cells: Vec<(usize, usize)>,
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

        // A blank line where a count is expected ends the file, as ASE's
        // reader does: trailing blank lines are tolerated and a blank line
        // between frames stops the read.
        if atom_count_line.trim().is_empty() {
            return Ok(None);
        }

        // Trimmed in the message so streamed and scanned reads of the same
        // bad line raise the identical error.
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
            // Tokenise the raw bytes on ASCII whitespace. Atom rows are numbers
            // and element symbols, so the per-line UTF-8 check the count and
            // comment lines pay is needless here -- only string cells are
            // validated, when pushed.
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
                let start = self.cells[cursor].0;
                push_cells(
                    column,
                    spans.iter().map(|&(start, end)| &self.buffer[start..end]),
                )
                .map_err(|error| at(line_number, Some(char_column(&self.buffer, start)), error))?;
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
    name: CompactString,
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

/// An empty column buffer of `kind`, pre-sized to `cells` (capped at
/// [`MAX_PREALLOC`]: the count derives from an untrusted declared atom count).
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

/// View a line buffer as UTF-8, failing exactly as [`io::Lines`] does so
/// the buffer-reusing reader raises identical errors.
fn line_str(buffer: &[u8]) -> Result<&str> {
    std::str::from_utf8(buffer).map_err(|_| {
        ExtxyzError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            "stream did not contain valid UTF-8",
        ))
    })
}

/// Append one atom's cells onto the column's buffer. Cells are raw bytes:
/// numbers and booleans parse straight from ASCII, and only string columns are
/// validated as UTF-8.
fn push_cells<'a>(column: &mut Column, cells: impl Iterator<Item = &'a [u8]>) -> Result<()> {
    match &mut column.data {
        ColumnData::Real(buffer) => {
            for cell in cells {
                // fast-float2 parses straight from bytes; same accepted grammar
                // as std, measurably faster on the 6-floats-per-atom-line shape.
                buffer.push(
                    fast_float2::parse::<f64, _>(cell)
                        .map_err(|_| invalid_cell(&column.name, "real", cell))?,
                );
            }
        }
        ColumnData::Int(buffer) => {
            for cell in cells {
                buffer
                    .push(parse_int(cell).ok_or_else(|| invalid_cell(&column.name, "int", cell))?);
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
                buffer.push(line_str(cell)?.into());
            }
        }
    }

    Ok(())
}

/// Parse an integer cell, preserving std's `i64` grammar. UTF-8 is checked
/// first (trivial for a short numeric token) so a non-UTF-8 cell is rejected
/// rather than misread.
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

/// Tokenize the comment line into ordered `(key, raw value, value offset)`
/// triples; file order and duplicate keys are preserved. `value offset` is
/// the byte index of the raw value's first byte within `comment`, so a
/// caller typing the value can shift a byte-offset error in that value back
/// into a column on the whole comment line.
/// Borrows key and value slices out of `comment`: the pairs are consumed in
/// the same scope (typed into `Value`s, or scanned for `Lattice`), so only the
/// kept metadata key is owned later — never the raw value. Owning both here
/// cost two allocations per key, the parse's largest allocation source.
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
            // A grouped value is a single token even when it contains
            // interior whitespace or commas (e.g. `{ 3 }`, `["a, b", "c]"]`).
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

            (slice_comment(comment, value_start, i)?, value_start)
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

/// From `bytes[open]` (a `{` or `[`), return the index just past the matching
/// close, tracking nesting of both bracket kinds and skipping quoted spans
/// (so a `]`, `}`, or space inside a `"..."` span does not end the group).
/// `Err(InvalidMetadata { index: open })` if the group never closes.
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

/// Strictly type a raw comment-line value. Byte offsets in any error are
/// relative to `raw`; the caller ([`parse_comment_line`]) shifts them by the
/// value's offset within the comment line. The splitter already strips a
/// surrounding `"..."` quote pair and keeps `{...}`/`[...]` groups intact, so
/// `raw` here is either quote-stripped content (never starts with `"`), a
/// brace group, or a bracket group.
///
/// The splitter tracks bracket *depth* only, so it accepts a mismatched
/// group like `{1]`; typing is where that mismatch is finally rejected — a
/// group opened with `{` must close with `}`, and `[` with `]`.
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
    // Quote-stripped content. A space can only appear here if the value was
    // quoted (bare values cannot contain spaces — the splitter breaks on
    // them). Single token -> scalar; multiple tokens -> whitespace array or
    // a sentence.
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

/// Classify one bare scalar token by the grammar: int, else float (incl.
/// Fortran d/D exponent), else bool, else bare string. Never fails — an
/// unrecognised token is a valid bare string. Numeric/bool classification is
/// on the trimmed `token`, but the string fallback keeps the original `raw`
/// (untrimmed, quote-stripped): a quoted value preserves its interior
/// whitespace, so `" hello "` stays `Str(" hello ")` while `" 3 "` still
/// trims to `Int(3)`.
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

/// Float per the grammar, accepting Fortran `d`/`D` exponents by normalising
/// them to `e` before Rust's parser sees them. Only treated as a float when
/// the token actually has a point or an exponent marker, so a bare integer
/// (no `.`/`e`/`d`) is never misread as one; the `d`/`D` replacement only
/// allocates when one of those letters is actually present, keeping the
/// (far more common) plain-float and non-float paths allocation-free.
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

/// Type a `[...]` bracket array: 1-D (`[1,2,3]`, `["a","b"]`), or, when an
/// element itself nests a `[`, 2-D (`[[1,2],[3,4]]`) — flattened into the
/// matching `*Array2D` with its `rows`/`cols` shape. Top-level splitting is
/// quote- and bracket-aware ([`split_top_level`]), so a comma or a literal
/// `]` inside a `"..."` element is never mistaken for a separator or a
/// mismatched close.
fn parse_array_value(raw: &str) -> Result<Value> {
    let inner = raw
        .strip_prefix('[')
        .and_then(|rest| rest.strip_suffix(']'))
        .ok_or(ExtxyzError::InvalidMetadata { index: 0 })?;

    let elems = split_top_level(inner);

    // A nested `[` in any element makes this a 2-D array: every element
    // must itself be a complete `[...]` row.
    if elems.iter().any(|(_, element)| element.contains('[')) {
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

/// Type the rows of a 2-D bracket array. Each top-level element (already
/// split by the caller) must itself be a `[...]` row; rows are parsed as
/// 1-D element lists and must agree on length (a ragged shape is `Err`).
/// The flattened elements are classified once, so the widest kind across
/// every row wins — an int row next to a real row promotes the whole array
/// to real.
fn parse_array_2d(elems: &[(usize, String)]) -> Result<Value> {
    let mut rows: Vec<Vec<String>> = Vec::with_capacity(elems.len());
    for (offset, element) in elems {
        let row_inner = element
            .strip_prefix('[')
            .and_then(|rest| rest.strip_suffix(']'))
            .ok_or(ExtxyzError::InvalidMetadata { index: 1 + offset })?;

        let row_elems = split_top_level(row_inner);
        if let Some((inner_offset, _)) = row_elems
            .iter()
            .find(|(_, cell)| cell.is_empty() || cell.contains('['))
        {
            // Either an empty cell (stray comma) or a further nested `[`
            // (3-D+ arrays are not supported): both are malformed rows.
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

    // A multi-word non-numeric whitespace value is a sentence, not a string
    // array: only new-style `[...]` produces `StrArray`.
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

/// Split `inner` at top-level commas, treating `"..."` spans and nested
/// `[...]`/`{...}` groups as atomic, so a comma or bracket inside a quoted
/// element (or inside a nested array element) never splits it or is
/// mistaken for a mismatched close. Each element is trimmed and paired with
/// the byte offset of its first non-whitespace character within `inner`
/// (or its untrimmed start, when empty) — for pointing a later error at the
/// offending element rather than the whole value.
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

/// One classified array's elements, by increasing generality: an int array
/// stays exact; real, bool, or the string fallback is chosen only once every
/// element fails the narrower kind. Quote-stripping happens only in the
/// string fallback, so a quoted element like `"3"` never parses as a number
/// — it stays a string, matching the grammar's intent for quoted array
/// elements.
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
        // Strict grammar: a nested `[` makes it 2-D, not a raw-string
        // fallback (changed from the prior lenient pass, which had no 2-D
        // support at all).
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
        // A brace-wrapped scalar with interior whitespace types the same as
        // its bare form — the group is punctuation, not part of the value.
        assert_eq!(parse_metadata_value("{3}").unwrap(), Value::Int(3));
        assert_eq!(parse_metadata_value("{ 3 }").unwrap(), Value::Int(3));
        // A brace-wrapped whitespace array still works via the existing
        // whitespace-array typing.
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
        // The splitter only tracks bracket *depth*, so it accepts `{1]` and
        // `[1}` as balanced (depth returns to 0); typing is where the
        // mismatched close is finally rejected.
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

    // --- a blank count line is end of input, as in ASE ---

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
        // A trailing blank, several blanks, a space-only line, a tab-only
        // line: all tolerated, and the one real frame is read.
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
        // Only blank lines end the file; junk where a count is expected is an
        // error, not a silent stop.
        let text = format!("{FRAME}xyz\n{FRAME}");
        assert!(scan_count(&text).is_err());
        assert!(stream_count(&text).is_err());
    }

    #[test]
    fn raw_frames_treat_blank_as_end_of_input() {
        // The batch reader shares the rule: a frame requested after an
        // interspersed blank is out of range, since the blank ended the file.
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
        // The +1 for the comment line must not overflow on usize::MAX. Before
        // the saturating add this panicked in debug builds (and wrapped to 0
        // in release); now every reader reports a clean error instead.
        let text = format!("{}\n", usize::MAX);
        assert!(scan_frames(std::io::Cursor::new(text.as_bytes())).is_err());

        let mut streamed = FrameIter::new(std::io::Cursor::new(text.as_bytes()));
        assert!(matches!(streamed.next(), Some(Err(_))));

        let mut batch = RawFrames::selecting(std::io::Cursor::new(text.as_bytes()), &[0]);
        assert!(matches!(batch.next(), Some(Err(_))));
    }

    #[test]
    fn non_utf8_in_a_string_cell_errors() {
        // Atom rows are tokenised as bytes, but a string (species) cell is
        // still validated as UTF-8 when materialised: a stray non-UTF-8 byte
        // there is a clean error, not a panic and not a silently mangled atom.
        let mut bytes = b"1\nProperties=species:S:1:pos:R:3\n".to_vec();
        bytes.extend_from_slice(&[0xFF, b' ', b'0', b' ', b'0', b' ', b'0', b'\n']);
        let mut frames = FrameIter::new(std::io::Cursor::new(bytes));
        assert!(matches!(frames.next(), Some(Err(_))));
    }

    #[test]
    fn non_utf8_in_a_numeric_cell_errors() {
        // A non-UTF-8 byte in a numeric cell fails as an invalid value rather
        // than reaching the parser, since the cell never has to be valid UTF-8.
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
        /// The batch reader reads count lines too: its skip path (unselected
        /// frames) and read path must never panic on an untrusted count,
        /// including usize::MAX -- the `n_atoms + 1` overflow site.
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
