//! Lossless extxyz serialisation: the inverse of [`crate::extxyz`]'s parse.
//!
//! Reals are formatted shortest-round-trippable (Ryū), so `read -> write ->
//! read` reproduces every `f64` bit for bit. Output is compact, not
//! column-aligned — a deliberate departure from ASE's fixed-width look and the
//! price of compact + lossless.
//!
//! Two orderings are imposed and documented:
//! - columns: `species` then `pos` lead; the rest keep their order;
//! - comment line: `Lattice`, then `pbc`, then the generated `Properties`, then
//!   the remaining metadata in order.
//!
//! A frame missing `species` or `pos` is rejected — we never emit output that
//! breaks the grammar.

use std::{
    fs::{File, OpenOptions},
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
};

use flate2::{Compression as GzLevel, write::GzEncoder};

use crate::decode::{Codec, Compression, detect_for_write, is_extxyz};
use crate::extxyz::{ExtxyzError, Result};
use crate::model::{ColumnData, Frame, Value};
use compact_str::CompactString;

/// Write every frame to `path`, encoding per `compression`.
///
/// `level` (`0..=9`, codec default when `None`) tunes the deflate-based codecs
/// and is ignored for plain output. `append` adds to an existing file where the
/// codec allows it (plain, gzip); see [`FrameSink::create`].
pub fn write_frames(
    path: &Path,
    frames: &[Frame],
    compression: Compression,
    level: Option<i32>,
    append: bool,
) -> Result<()> {
    let mut sink = FrameSink::create(path, compression, level, append)?;
    for frame in frames {
        sink.write(frame)?;
    }
    sink.finish()
}

/// [`write_frames`] with the per-frame serialisation spread over `threads`
/// workers (`None`: every core, `Some(1)`: serial). Output bytes are identical
/// to [`write_frames`] for every codec — only serialisation parallelises; the
/// single output stream and any compression stay serial.
///
/// Errors match the serial writer: the first frame in order missing
/// `species`/`pos` is reported. Frames are serialised *before* the sink is
/// opened, so a rejected frame leaves no output file behind.
#[cfg(feature = "parallel")]
pub fn write_frames_parallel(
    path: &Path,
    frames: &[Frame],
    compression: Compression,
    level: Option<i32>,
    append: bool,
    threads: Option<usize>,
) -> Result<()> {
    let buffers = serialise_parallel(frames, threads)?;
    let mut sink = FrameSink::create(path, compression, level, append)?;
    for buf in &buffers {
        sink.write_serialized(buf)?;
    }
    sink.finish()
}

/// Serialise `frames` to per-chunk byte buffers on a rayon pool, in frame order.
/// The first frame (in order) that cannot be serialised is the reported error,
/// matching the serial writer's precedence.
#[cfg(feature = "parallel")]
fn serialise_parallel(frames: &[Frame], threads: Option<usize>) -> Result<Vec<Vec<u8>>> {
    use rayon::prelude::*;

    // A few chunks per thread amortises the per-chunk buffer while leaving rayon
    // room to balance — the same split the parallel reader uses.
    crate::extxyz::with_pool(threads, || {
        let chunk_size = frames
            .len()
            .div_ceil(rayon::current_num_threads() * 4)
            .max(1);
        frames
            .par_chunks(chunk_size)
            .map(serialise_chunk)
            .collect::<Vec<Result<Vec<u8>>>>()
    })?
    .into_iter()
    .collect()
}

#[cfg(feature = "parallel")]
fn serialise_chunk(chunk: &[Frame]) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    for frame in chunk {
        write_frame(&mut buf, frame)?;
    }
    Ok(buf)
}

/// Serialise one frame to `out` as an extxyz text block (count line, comment
/// line, then one line per atom). The pure serialiser, shared by every sink.
pub fn write_frame<W: Write>(out: &mut W, frame: &Frame) -> Result<()> {
    let order = column_order(frame)?;

    writeln!(out, "{}", frame.n_atoms)?;
    write_comment(out, frame, &order)?;

    let mut buf = ryu::Buffer::new();
    for row in 0..frame.n_atoms {
        for (col_pos, &index) in order.iter().enumerate() {
            let column = &frame.columns[index];
            for cell in 0..column.width {
                // A single space between every cell; none leads the line.
                if !(col_pos == 0 && cell == 0) {
                    out.write_all(b" ")?;
                }
                write_cell(out, &column.data, row * column.width + cell, &mut buf)?;
            }
        }
        out.write_all(b"\n")?;
    }
    Ok(())
}

/// Column write order as indices into `frame.columns`: `species`, then `pos`,
/// then the rest in their existing order. Errors if either required column is
/// absent.
fn column_order(frame: &Frame) -> Result<Vec<usize>> {
    let find = |name: &'static str| {
        frame
            .columns
            .iter()
            .position(|column| column.name == name)
            .ok_or(ExtxyzError::MissingRequiredColumn { name })
    };
    let species = find("species")?;
    let pos = find("pos")?;

    let mut order = Vec::with_capacity(frame.columns.len());
    order.push(species);
    order.push(pos);
    order.extend((0..frame.columns.len()).filter(|&i| i != species && i != pos));
    Ok(order)
}

fn write_comment<W: Write>(out: &mut W, frame: &Frame, order: &[usize]) -> io::Result<()> {
    let mut leading = false; // whether anything has been written yet

    // Lattice and pbc lead, in that order, ahead of the generated Properties.
    for hoisted in ["Lattice", "pbc"] {
        for (key, value) in &frame.metadata {
            if key == hoisted {
                write_metadata(out, key, value, &mut leading)?;
            }
        }
    }

    space(out, &mut leading)?;
    write!(out, "Properties=")?;
    write_properties(out, frame, order)?;

    // Everything else in its original order; Lattice/pbc already placed.
    for (key, value) in &frame.metadata {
        if key != "Lattice" && key != "pbc" {
            write_metadata(out, key, value, &mut leading)?;
        }
    }

    out.write_all(b"\n")
}

/// `species:S:1:pos:R:3:...` built from the resolved column order.
fn write_properties<W: Write>(out: &mut W, frame: &Frame, order: &[usize]) -> io::Result<()> {
    for (i, &index) in order.iter().enumerate() {
        let column = &frame.columns[index];
        if i > 0 {
            out.write_all(b":")?;
        }
        write!(
            out,
            "{}:{}:{}",
            column.name,
            column.data.kind(),
            column.width
        )?;
    }
    Ok(())
}

fn write_metadata<W: Write>(
    out: &mut W,
    key: &str,
    value: &Value,
    leading: &mut bool,
) -> io::Result<()> {
    space(out, leading)?;
    write!(out, "{key}=")?;
    let mut buf = ryu::Buffer::new();
    // Lattice/pbc keep the conventional quoted space-separated form ASE expects;
    // other arrays use brackets, which also round-trip a single element (a
    // bare-token array would re-parse as a scalar).
    let quoted = key == "Lattice" || key == "pbc";
    match value {
        Value::Real(x) => write!(out, "{}", buf.format(*x)),
        Value::Int(x) => write!(out, "{x}"),
        Value::Bool(b) => out.write_all(bool_token(*b)),
        Value::Str(s) => write_scalar_string(out, s),
        Value::RealArray(xs) => write_real_array(out, xs, quoted),
        Value::IntArray(xs) => write_int_array(out, xs, quoted),
        Value::BoolArray(xs) => write_bool_array(out, xs, quoted),
        Value::StrArray(xs) => write_str_array(out, xs),
    }
}

/// Quote a scalar string only when leaving it bare would break tokenisation:
/// whitespace splits the comment line, and an empty value has nothing to read.
fn write_scalar_string<W: Write>(out: &mut W, s: &str) -> io::Result<()> {
    if s.is_empty() || s.chars().any(char::is_whitespace) {
        write!(out, "\"{s}\"")
    } else {
        out.write_all(s.as_bytes())
    }
}

fn write_real_array<W: Write>(out: &mut W, xs: &[f64], quoted: bool) -> io::Result<()> {
    let mut buf = ryu::Buffer::new();
    write_array(out, xs.len(), quoted, |out, i| {
        out.write_all(buf.format(xs[i]).as_bytes())
    })
}

fn write_int_array<W: Write>(out: &mut W, xs: &[i64], quoted: bool) -> io::Result<()> {
    write_array(out, xs.len(), quoted, |out, i| write!(out, "{}", xs[i]))
}

fn write_bool_array<W: Write>(out: &mut W, xs: &[bool], quoted: bool) -> io::Result<()> {
    write_array(out, xs.len(), quoted, |out, i| {
        out.write_all(bool_token(xs[i]))
    })
}

/// String arrays always bracket-quote each element: `["slab","relaxed"]`.
fn write_str_array<W: Write>(out: &mut W, xs: &[CompactString]) -> io::Result<()> {
    out.write_all(b"[")?;
    for (i, s) in xs.iter().enumerate() {
        if i > 0 {
            out.write_all(b",")?;
        }
        write!(out, "\"{s}\"")?;
    }
    out.write_all(b"]")
}

/// Numeric arrays: quoted space-separated when `quoted` (Lattice/pbc) and length
/// permits it, else bracketed. A single element must bracket — `key="1.5"` would
/// re-parse as a scalar, not a one-element array.
fn write_array<W, F>(out: &mut W, len: usize, quoted: bool, mut element: F) -> io::Result<()>
where
    W: Write,
    F: FnMut(&mut W, usize) -> io::Result<()>,
{
    if quoted && len >= 2 {
        out.write_all(b"\"")?;
        for i in 0..len {
            if i > 0 {
                out.write_all(b" ")?;
            }
            element(out, i)?;
        }
        out.write_all(b"\"")
    } else {
        out.write_all(b"[")?;
        for i in 0..len {
            if i > 0 {
                out.write_all(b",")?;
            }
            element(out, i)?;
        }
        out.write_all(b"]")
    }
}

fn write_cell<W: Write>(
    out: &mut W,
    data: &ColumnData,
    i: usize,
    buf: &mut ryu::Buffer,
) -> io::Result<()> {
    match data {
        ColumnData::Real(xs) => out.write_all(buf.format(xs[i]).as_bytes()),
        ColumnData::Int(xs) => write!(out, "{}", xs[i]),
        ColumnData::Bool(xs) => out.write_all(bool_token(xs[i])),
        ColumnData::Str(xs) => out.write_all(xs[i].as_bytes()),
    }
}

fn bool_token(b: bool) -> &'static [u8] {
    if b { b"T" } else { b"F" }
}

/// Emit a separating space before the next token, except the first.
fn space<W: Write>(out: &mut W, leading: &mut bool) -> io::Result<()> {
    if *leading {
        out.write_all(b" ")?;
    }
    *leading = true;
    Ok(())
}

/// An incremental writer: build it, [`write`](Self::write) frames as they come,
/// then [`finish`](Self::finish). Constant memory for the streaming codecs;
/// archive codecs buffer their single member until `finish`.
pub struct FrameSink {
    inner: Inner,
}

enum Inner {
    /// Plain or gzip: each frame is encoded straight to the sink.
    Stream(Stream),
    /// zip/tar/tar.gz: one extxyz member, accumulated then written whole.
    Archive {
        buf: Vec<u8>,
        codec: Codec,
        path: PathBuf,
        member: String,
        level: Option<i32>,
    },
}

enum Stream {
    Plain(BufWriter<Box<dyn Write + Send + Sync>>),
    Gzip(GzEncoder<Box<dyn Write + Send + Sync>>),
}

impl FrameSink {
    /// Open a sink for `path`. `compression="infer"` reads the extension only
    /// (`-` and extensionless paths mean plain).
    ///
    /// `append` is honoured for plain and gzip — a gzip append is a fresh member
    /// concatenated onto the file, which the reader handles. zstd output is
    /// rejected (encode unsupported); appending to an archive or to stdout is
    /// rejected too.
    pub fn create(
        path: &Path,
        compression: Compression,
        level: Option<i32>,
        append: bool,
    ) -> Result<Self> {
        if let Some(level) = level {
            if !(0..=9).contains(&level) {
                return Err(ExtxyzError::InvalidCompressionLevel { level });
            }
        }

        let codec = detect_for_write(path, compression);
        let stdout = path == Path::new("-");

        if codec == Codec::Zstd {
            return Err(ExtxyzError::ZstdWriteUnsupported);
        }
        if append && (codec.is_archive() || stdout) {
            let what = if stdout {
                "stdout"
            } else {
                archive_label(codec)
            };
            return Err(ExtxyzError::AppendUnsupported { codec: what });
        }
        if codec.is_archive() && stdout {
            return Err(ExtxyzError::Io(io::Error::other(
                "archive output (.zip/.tar/.tar.gz) requires a file path, not stdout",
            )));
        }

        let inner = if codec.is_archive() {
            Inner::Archive {
                buf: Vec::new(),
                codec,
                path: path.to_owned(),
                member: archive_member_name(path),
                level,
            }
        } else {
            let raw = open_sink(path, stdout, append)?;
            Inner::Stream(match codec {
                Codec::Gzip => Stream::Gzip(GzEncoder::new(raw, gz_level(level))),
                _ => Stream::Plain(BufWriter::new(raw)),
            })
        };
        Ok(FrameSink { inner })
    }

    pub fn write(&mut self, frame: &Frame) -> Result<()> {
        match &mut self.inner {
            Inner::Stream(Stream::Plain(w)) => write_frame(w, frame),
            Inner::Stream(Stream::Gzip(w)) => write_frame(w, frame),
            Inner::Archive { buf, .. } => write_frame(buf, frame),
        }
    }

    /// Write an already-serialised block (the bytes [`write_frame`] produces) to
    /// the sink. The parallel paths serialise off the I/O thread, then hand the
    /// ordered bytes here.
    fn write_serialized(&mut self, bytes: &[u8]) -> Result<()> {
        match &mut self.inner {
            Inner::Stream(Stream::Plain(w)) => w.write_all(bytes)?,
            Inner::Stream(Stream::Gzip(w)) => w.write_all(bytes)?,
            Inner::Archive { buf, .. } => buf.extend_from_slice(bytes),
        }
        Ok(())
    }

    /// Serialise `frames` in parallel and write them in order. The bounded-memory
    /// batch path behind `Writer(batch=N)`: peak extra memory is the batch's
    /// serialised text, not the whole file. Output is identical to writing the
    /// frames one by one with [`write`](Self::write).
    #[cfg(feature = "parallel")]
    pub fn write_batch_parallel(&mut self, frames: &[Frame], threads: Option<usize>) -> Result<()> {
        for buf in &serialise_parallel(frames, threads)? {
            self.write_serialized(buf)?;
        }
        Ok(())
    }

    pub fn finish(self) -> Result<()> {
        match self.inner {
            Inner::Stream(Stream::Plain(mut w)) => {
                w.flush()?;
                Ok(())
            }
            Inner::Stream(Stream::Gzip(w)) => {
                w.finish()?.flush()?;
                Ok(())
            }
            Inner::Archive {
                buf,
                codec,
                path,
                member,
                level,
            } => write_archive(&path, &member, &buf, codec, level),
        }
    }
}

fn open_sink(path: &Path, stdout: bool, append: bool) -> Result<Box<dyn Write + Send + Sync>> {
    if stdout {
        return Ok(Box::new(io::stdout()));
    }
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .append(append)
        .truncate(!append)
        .open(path)?;
    Ok(Box::new(file))
}

fn gz_level(level: Option<i32>) -> GzLevel {
    level.map_or_else(GzLevel::default, |l| GzLevel::new(l as u32))
}

fn archive_label(codec: Codec) -> &'static str {
    match codec {
        Codec::Zip => "zip",
        Codec::Tar => "tar",
        Codec::TarGzip => "tar.gz",
        _ => "archive",
    }
}

/// Member name for an archive: the path stem with its archive suffix stripped,
/// given a `.extxyz` extension if it lacks an xyz one, so the reader's member
/// resolver finds it.
fn archive_member_name(path: &Path) -> String {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("data");
    let stem = [".tar.gz", ".tgz", ".tar", ".zip"]
        .into_iter()
        .find_map(|suffix| name.strip_suffix(suffix))
        .unwrap_or(name);
    if is_extxyz(stem) {
        stem.to_owned()
    } else {
        format!("{stem}.extxyz")
    }
}

fn write_archive(
    path: &Path,
    member: &str,
    buf: &[u8],
    codec: Codec,
    level: Option<i32>,
) -> Result<()> {
    let file = File::create(path)?;
    match codec {
        Codec::Zip => write_zip(file, member, buf, level),
        Codec::Tar => {
            write_tar(file, member, buf)?;
            Ok(())
        }
        Codec::TarGzip => {
            let gzip = write_tar(GzEncoder::new(file, gz_level(level)), member, buf)?;
            gzip.finish()?.flush()?;
            Ok(())
        }
        _ => unreachable!("write_archive only handles archive codecs"),
    }
}

fn write_zip<W: Write + io::Seek>(
    sink: W,
    member: &str,
    buf: &[u8],
    level: Option<i32>,
) -> Result<()> {
    use zip::write::SimpleFileOptions;

    let mut zip = zip::ZipWriter::new(sink);
    let options = SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated)
        .compression_level(level.map(i64::from));
    zip.start_file(member, options)
        .map_err(|error| ExtxyzError::Io(io::Error::other(error.to_string())))?;
    zip.write_all(buf)?;
    zip.finish()
        .map_err(|error| ExtxyzError::Io(io::Error::other(error.to_string())))?;
    Ok(())
}

/// Tar requires the member size up front, so the whole block is in `buf`
/// already. Returns the sink so a `.tar.gz` caller can finish the gzip layer.
fn write_tar<W: Write>(sink: W, member: &str, buf: &[u8]) -> Result<W> {
    let mut header = tar::Header::new_gnu();
    header.set_size(buf.len() as u64);
    header.set_mode(0o644);
    header.set_cksum();

    let mut builder = tar::Builder::new(sink);
    builder.append_data(&mut header, member, buf)?;
    let mut sink = builder.into_inner()?;
    sink.flush()?;
    Ok(sink)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extxyz::iter_frames_from;
    use crate::model::{Column, ColumnData, Value};

    fn serialise(frame: &Frame) -> String {
        let mut out = Vec::new();
        write_frame(&mut out, frame).unwrap();
        String::from_utf8(out).unwrap()
    }

    fn reparse(text: &str) -> Vec<Frame> {
        iter_frames_from(text.as_bytes())
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap()
    }

    fn col(name: &str, width: usize, data: ColumnData) -> Column {
        Column {
            name: name.into(),
            width,
            data,
        }
    }

    fn water() -> Frame {
        Frame {
            n_atoms: 2,
            columns: vec![
                col("species", 1, ColumnData::Str(vec!["O".into(), "H".into()])),
                col(
                    "pos",
                    3,
                    ColumnData::Real(vec![0.0, 0.0, 0.0, 0.9, 0.0, 0.0]),
                ),
            ],
            metadata: vec![(
                "Lattice".into(),
                Value::RealArray(vec![5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0]),
            )],
        }
    }

    #[test]
    fn golden_block_has_imposed_order_and_compact_reals() {
        let text = serialise(&water());
        assert_eq!(
            text,
            "2\n\
             Lattice=\"5.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0 5.0\" Properties=species:S:1:pos:R:3\n\
             O 0.0 0.0 0.0\n\
             H 0.9 0.0 0.0\n"
        );
    }

    #[test]
    fn lattice_pbc_then_properties_then_rest() {
        let mut frame = water();
        frame.metadata = vec![
            ("config_type".into(), Value::Str("bulk".into())),
            (
                "Lattice".into(),
                Value::RealArray(vec![5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0]),
            ),
            ("pbc".into(), Value::BoolArray(vec![true, true, false])),
        ];
        let text = serialise(&frame);
        let comment = text.lines().nth(1).unwrap();
        assert!(comment.starts_with(
            "Lattice=\"5.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0 5.0\" pbc=\"T T F\" \
             Properties=species:S:1:pos:R:3 config_type=bulk"
        ));
    }

    #[test]
    fn species_and_pos_are_hoisted_ahead_of_other_columns() {
        let frame = Frame {
            n_atoms: 1,
            columns: vec![
                col("forces", 3, ColumnData::Real(vec![1.0, 2.0, 3.0])),
                col("pos", 3, ColumnData::Real(vec![0.0, 0.0, 0.0])),
                col("species", 1, ColumnData::Str(vec!["Fe".into()])),
            ],
            metadata: vec![],
        };
        let comment = serialise(&frame);
        let line = comment.lines().nth(1).unwrap();
        assert_eq!(line, "Properties=species:S:1:pos:R:3:forces:R:3");
        assert_eq!(
            comment.lines().nth(2).unwrap(),
            "Fe 0.0 0.0 0.0 1.0 2.0 3.0"
        );
    }

    #[test]
    fn missing_species_or_pos_is_rejected() {
        let no_pos = Frame {
            n_atoms: 1,
            columns: vec![col("species", 1, ColumnData::Str(vec!["H".into()]))],
            metadata: vec![],
        };
        assert!(matches!(
            write_frame(&mut Vec::new(), &no_pos),
            Err(ExtxyzError::MissingRequiredColumn { name: "pos" })
        ));

        let no_species = Frame {
            n_atoms: 1,
            columns: vec![col("pos", 3, ColumnData::Real(vec![0.0, 0.0, 0.0]))],
            metadata: vec![],
        };
        assert!(matches!(
            write_frame(&mut Vec::new(), &no_species),
            Err(ExtxyzError::MissingRequiredColumn { name: "species" })
        ));
    }

    #[test]
    fn round_trips_a_rich_frame_through_reparse() {
        let frame = Frame {
            n_atoms: 2,
            columns: vec![
                col("species", 1, ColumnData::Str(vec!["Si".into(), "O".into()])),
                col(
                    "pos",
                    3,
                    ColumnData::Real(vec![0.0, 1.5, -2.25, 1e-30, 3.0, 1e30]),
                ),
                col("tag", 1, ColumnData::Int(vec![7, -3])),
                col("fixed", 1, ColumnData::Bool(vec![true, false])),
            ],
            metadata: vec![
                ("energy".into(), Value::Real(-1.2345e3)),
                ("count".into(), Value::Int(42)),
                ("flag".into(), Value::Bool(true)),
                ("name".into(), Value::Str("water monomer".into())),
                ("scale".into(), Value::RealArray(vec![0.5])),
                ("dims".into(), Value::IntArray(vec![2, 2, 1])),
                ("tags".into(), Value::StrArray(vec!["a".into(), "b".into()])),
            ],
        };
        let reparsed = reparse(&serialise(&frame));
        assert_eq!(reparsed.len(), 1);
        // Metadata was already in non-hoisted order with no Lattice/pbc, so the
        // model compares equal field for field, reals included.
        assert_eq!(reparsed[0], frame);
    }

    #[test]
    fn writes_multiple_frames_back_to_back() {
        let frames = vec![water(), water()];
        let mut out = Vec::new();
        for frame in &frames {
            write_frame(&mut out, frame).unwrap();
        }
        let text = String::from_utf8(out).unwrap();
        assert_eq!(reparse(&text).len(), 2);
    }
}
