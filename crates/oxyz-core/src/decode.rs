//! Opening a read source through a decompressing layer.
//!
//! Everything streams: nothing is decompressed to a temporary file or held
//! whole in memory. Compressed sources are not seekable, so random access
//! stays on raw files.

use std::{
    fs::File,
    io::{self, BufRead, BufReader, Cursor, Read, Seek},
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, SyncSender, sync_channel},
    },
    thread,
};

use flate2::read::MultiGzDecoder;
use ruzstd::decoding::{FrameDecoder, StreamingDecoder};

use crate::extxyz::{ExtxyzError, Result};

/// A decoded, streaming read source. `Sync` so a `#[pyclass]` can hold one.
pub type DecodedReader = Box<dyn BufRead + Send + Sync>;

/// A raw, undecoded byte source: a file, or a remote stream from the binding.
pub type ByteSource = Box<dyn Read + Send + Sync>;

/// How to interpret a read source. `Infer` reads the extension, then sniffs
/// magic bytes; the rest force a codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Compression {
    #[default]
    Infer,
    /// No decompression — read the bytes as-is.
    None,
    Gzip,
    Zstd,
    Zip,
    Tar,
    TarGzip,
    TarZstd,
}

/// The concrete codec, after inference. Only the archive codecs accept a
/// `member` selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    Plain,
    Gzip,
    Zstd,
    Zip,
    Tar,
    TarGzip,
    TarZstd,
}

impl Codec {
    pub(crate) fn is_archive(self) -> bool {
        matches!(
            self,
            Codec::Zip | Codec::Tar | Codec::TarGzip | Codec::TarZstd
        )
    }

    pub fn name(self) -> &'static str {
        match self {
            Codec::Plain => "plain",
            Codec::Gzip => "gzip",
            Codec::Zstd => "zstd",
            Codec::Zip => "zip",
            Codec::Tar => "tar",
            Codec::TarGzip => "tar.gz",
            Codec::TarZstd => "tar.zst",
        }
    }
}

/// The decompressor applied to a tar's byte stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TarInner {
    Plain,
    Gzip,
    Zstd,
}

/// Resolve the codec for writing: there are no bytes to sniff, so `Infer`
/// reads the extension only, defaulting to [`Codec::Plain`].
pub(crate) fn detect_for_write(path: &Path, compression: Compression) -> Codec {
    match compression {
        Compression::None => Codec::Plain,
        Compression::Gzip => Codec::Gzip,
        Compression::Zstd => Codec::Zstd,
        Compression::Zip => Codec::Zip,
        Compression::Tar => Codec::Tar,
        Compression::TarGzip => Codec::TarGzip,
        Compression::TarZstd => Codec::TarZstd,
        Compression::Infer => detect_by_extension(path).unwrap_or(Codec::Plain),
    }
}

/// Open `path` as a streaming reader, decompressing per `compression`.
///
/// `member` names an archive entry and is rejected for single streams. Without
/// it, an archive must hold exactly one `.xyz`/`.extxyz` member.
pub fn open_decoded(
    path: &Path,
    compression: Compression,
    member: Option<&str>,
) -> Result<DecodedReader> {
    let codec = detect(path, compression)?;

    if member.is_some() && !codec.is_archive() {
        return Err(ExtxyzError::MemberOnNonArchive);
    }

    match codec {
        Codec::Plain => wrap_stream(Box::new(File::open(path)?), Codec::Plain),
        Codec::Gzip => wrap_stream(Box::new(File::open(path)?), Codec::Gzip),
        Codec::Zstd => wrap_stream(Box::new(File::open(path)?), Codec::Zstd),
        Codec::Zip => wrap_zip(File::open(path)?, member),
        Codec::Tar => {
            let path = path.to_owned();
            wrap_tar(
                move || File::open(&path).map(|f| Box::new(f) as ByteSource),
                member,
                TarInner::Plain,
            )
        }
        Codec::TarGzip => {
            let path = path.to_owned();
            wrap_tar(
                move || File::open(&path).map(|f| Box::new(f) as ByteSource),
                member,
                TarInner::Gzip,
            )
        }
        Codec::TarZstd => {
            let path = path.to_owned();
            wrap_tar(
                move || File::open(&path).map(|f| Box::new(f) as ByteSource),
                member,
                TarInner::Zstd,
            )
        }
    }
}

/// Wrap a raw byte source in a single-stream codec. Archive codecs are
/// rejected; use [`wrap_zip`] or [`wrap_tar`].
pub fn wrap_stream(source: ByteSource, codec: Codec) -> Result<DecodedReader> {
    match codec {
        Codec::Plain => Ok(Box::new(BufReader::new(source))),
        Codec::Gzip => Ok(Box::new(BufReader::new(MultiGzDecoder::new(source)))),
        Codec::Zstd => Ok(Box::new(BufReader::new(MultiFrameZstd::from_source(
            source,
        )?))),
        Codec::Zip | Codec::Tar | Codec::TarGzip | Codec::TarZstd => {
            Err(ExtxyzError::MemberOnNonArchive)
        }
    }
}

/// Whether `path`, read under `compression`, needs decompression.
pub fn is_compressed(path: &Path, compression: Compression) -> Result<bool> {
    Ok(detect(path, compression)? != Codec::Plain)
}

/// Resolve the codec from an explicit choice, or infer it from the extension
/// and then the leading magic bytes.
fn detect(path: &Path, compression: Compression) -> Result<Codec> {
    match compression {
        Compression::None => Ok(Codec::Plain),
        Compression::Gzip => Ok(Codec::Gzip),
        Compression::Zstd => Ok(Codec::Zstd),
        Compression::Zip => Ok(Codec::Zip),
        Compression::Tar => Ok(Codec::Tar),
        Compression::TarGzip => Ok(Codec::TarGzip),
        Compression::TarZstd => Ok(Codec::TarZstd),
        Compression::Infer => Ok(detect_by_extension(path).map_or_else(|| sniff(path), Ok)?),
    }
}

pub(crate) fn detect_by_extension(path: &Path) -> Option<Codec> {
    let name = path.file_name()?.to_str()?.to_ascii_lowercase();
    Some(if name.ends_with(".tar.gz") || name.ends_with(".tgz") {
        Codec::TarGzip
    } else if name.ends_with(".tar.zst") || name.ends_with(".tzst") {
        Codec::TarZstd
    } else if name.ends_with(".tar") {
        Codec::Tar
    } else if name.ends_with(".gz") {
        Codec::Gzip
    } else if name.ends_with(".zst") {
        Codec::Zstd
    } else if name.ends_with(".zip") {
        Codec::Zip
    } else {
        return None;
    })
}

/// Magic-byte codec for a head slice. A tar has no magic, so none is detected.
fn sniff_bytes(head: &[u8]) -> Codec {
    if head.starts_with(&[0x1f, 0x8b]) {
        Codec::Gzip
    } else if head.starts_with(&[0x28, 0xb5, 0x2f, 0xfd]) {
        Codec::Zstd
    } else if head.starts_with(b"PK\x03\x04") {
        Codec::Zip
    } else {
        Codec::Plain
    }
}

/// Magic-byte fallback for an uninformative extension. Gzip is reported as
/// `Gzip`, not `TarGzip`: a tar inside would surface as a parse error, not
/// silent corruption.
fn sniff(path: &Path) -> Result<Codec> {
    let mut head = [0u8; 4];
    let read = File::open(path)?.read(&mut head)?;
    Ok(sniff_bytes(&head[..read]))
}

/// Codec for a filename and optional leading bytes, without I/O.
pub fn detect_codec_name(name: &str, head: Option<&[u8]>) -> &'static str {
    let path = Path::new(name);
    if let Some(codec) = detect_by_extension(path) {
        return codec.name();
    }
    head.map_or(Codec::Plain, sniff_bytes).name()
}

/// Boxed so the leftover stream can be re-wrapped with the peeked byte.
type ZstdSource = Box<dyn Read + Send + Sync>;

/// Streams every frame of a `.zst` file. ruzstd's [`StreamingDecoder`] stops
/// after one frame, so the decoder is recreated at each frame boundary.
struct MultiFrameZstd {
    decoder: Option<StreamingDecoder<ZstdSource, FrameDecoder>>,
}

impl MultiFrameZstd {
    fn from_source(source: ZstdSource) -> Result<Self> {
        Ok(Self::from_source_io(source)?)
    }

    /// The `io::Error` variant, for the tar pipe's `io::Result` channel.
    fn from_source_io(source: ZstdSource) -> io::Result<Self> {
        Ok(MultiFrameZstd {
            decoder: Some(zstd_decoder(source)?),
        })
    }
}

fn zstd_decoder(source: ZstdSource) -> io::Result<StreamingDecoder<ZstdSource, FrameDecoder>> {
    StreamingDecoder::new(source).map_err(|error| io::Error::other(error.to_string()))
}

impl Read for MultiFrameZstd {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        // Otherwise a zero-length read looks like frame exhaustion and drops a byte.
        if out.is_empty() {
            return Ok(0);
        }
        loop {
            let Some(decoder) = self.decoder.as_mut() else {
                return Ok(0);
            };
            let read = decoder.read(out)?;
            if read > 0 {
                return Ok(read);
            }
            // Frame exhausted: one more byte means another frame follows.
            let mut source = self.decoder.take().expect("checked above").into_inner();
            let mut peek = [0u8; 1];
            if source.read(&mut peek)? == 0 {
                return Ok(0);
            }
            let chained: ZstdSource = Box::new(Cursor::new([peek[0]]).chain(source));
            self.decoder = Some(zstd_decoder(chained)?);
        }
    }
}

pub(crate) fn is_extxyz(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.ends_with(".xyz") || lower.ends_with(".extxyz")
}

/// Pick the member to read: the named one, or the sole extxyz-looking one.
fn resolve_member(names: &[String], member: Option<&str>) -> Result<String> {
    if let Some(wanted) = member {
        return names
            .iter()
            .find(|name| name.as_str() == wanted)
            .cloned()
            .ok_or_else(|| ExtxyzError::MemberNotFound {
                member: wanted.to_owned(),
                available: names.to_vec(),
            });
    }

    let mut extxyz = names.iter().filter(|name| is_extxyz(name));
    match (extxyz.next(), extxyz.next()) {
        (None, _) => Err(ExtxyzError::NoExtxyzMember {
            members: names.to_vec(),
        }),
        (Some(only), None) => Ok(only.clone()),
        (Some(_), Some(_)) => Err(ExtxyzError::AmbiguousArchive {
            members: names.iter().filter(|n| is_extxyz(n)).cloned().collect(),
        }),
    }
}

/// Stream one member of a zip. The central directory is at the end, so the
/// source must seek.
pub fn wrap_zip<R>(source: R, member: Option<&str>) -> Result<DecodedReader>
where
    R: Read + Seek + Send + 'static,
{
    let mut archive = zip::ZipArchive::new(source).map_err(zip_error)?;
    let names: Vec<String> = archive
        .file_names()
        .filter(|name| !name.ends_with('/'))
        .map(str::to_owned)
        .collect();
    let target = resolve_member(&names, member)?;
    let index = (0..archive.len())
        .find(|&i| {
            archive
                .by_index_raw(i)
                .is_ok_and(|entry| entry.name() == target)
        })
        .expect("resolved member is present");

    Ok(spawn_pipe(move |tx| match archive.by_index(index) {
        Ok(mut entry) => pump(&mut entry, tx),
        Err(error) => {
            let _ = tx.send(Err(io::Error::other(error.to_string())));
        }
    }))
}

/// Stream one member of a tar, decompressing it by `inner` first. `factory`
/// yields a fresh raw stream per pass: one to enumerate, one to read.
pub fn wrap_tar<F>(factory: F, member: Option<&str>, inner: TarInner) -> Result<DecodedReader>
where
    F: Fn() -> io::Result<ByteSource> + Send + 'static,
{
    let names = tar_member_names(&factory, inner)?;
    let target = resolve_member(&names, member)?;

    Ok(spawn_pipe(move |tx| {
        let stream = match tar_stream_from(&factory, inner) {
            Ok(stream) => stream,
            Err(error) => {
                let _ = tx.send(Err(error));
                return;
            }
        };
        let mut archive = tar::Archive::new(stream);
        let entries = match archive.entries() {
            Ok(entries) => entries,
            Err(error) => {
                let _ = tx.send(Err(error));
                return;
            }
        };
        for entry in entries {
            match entry {
                Ok(mut entry) => {
                    if entry_name(&entry).as_deref() == Some(target.as_str()) {
                        pump(&mut entry, tx);
                        return;
                    }
                }
                Err(error) => {
                    let _ = tx.send(Err(error));
                    return;
                }
            }
        }
    }))
}

fn tar_member_names<F>(factory: &F, inner: TarInner) -> Result<Vec<String>>
where
    F: Fn() -> io::Result<ByteSource>,
{
    let mut archive = tar::Archive::new(tar_stream_from(factory, inner)?);
    let mut names = Vec::new();
    for entry in archive.entries()? {
        let entry = entry?;
        if entry.header().entry_type().is_dir() {
            continue;
        }
        if let Some(name) = entry_name(&entry) {
            names.push(name);
        }
    }
    Ok(names)
}

fn tar_stream_from<F>(factory: &F, inner: TarInner) -> io::Result<ByteSource>
where
    F: Fn() -> io::Result<ByteSource>,
{
    let raw = factory()?;
    Ok(match inner {
        TarInner::Plain => raw,
        TarInner::Gzip => Box::new(MultiGzDecoder::new(raw)),
        TarInner::Zstd => Box::new(MultiFrameZstd::from_source_io(raw)?),
    })
}

fn entry_name<R: Read>(entry: &tar::Entry<'_, R>) -> Option<String> {
    entry.path().ok()?.to_str().map(str::to_owned)
}

fn zip_error(error: zip::result::ZipError) -> ExtxyzError {
    ExtxyzError::Io(io::Error::other(error.to_string()))
}

const CHUNK: usize = 64 * 1024;

/// Copy `reader` to the channel in chunks, stopping if the consumer hangs up.
fn pump<R: Read>(reader: &mut R, tx: &SyncSender<io::Result<Vec<u8>>>) {
    let mut buf = vec![0u8; CHUNK];
    loop {
        match reader.read(&mut buf) {
            Ok(0) => return,
            Ok(read) => {
                if tx.send(Ok(buf[..read].to_vec())).is_err() {
                    return;
                }
            }
            Err(error) => {
                let _ = tx.send(Err(error));
                return;
            }
        }
    }
}

/// Run `producer` on its own thread, streaming bytes back over a bounded
/// channel. Archive readers borrow their archive, so the producer owns it.
fn spawn_pipe<F>(producer: F) -> DecodedReader
where
    F: FnOnce(&SyncSender<io::Result<Vec<u8>>>) + Send + 'static,
{
    let (tx, rx) = sync_channel::<io::Result<Vec<u8>>>(4);
    let finished = Arc::new(AtomicBool::new(false));
    let producer_finished = Arc::clone(&finished);
    thread::spawn(move || {
        producer(&tx);
        // Set before `tx` drops, so a panicking producer surfaces as an error
        // rather than a short read.
        producer_finished.store(true, Ordering::Release);
    });
    Box::new(BufReader::new(PipeReader {
        rx: Mutex::new(rx),
        current: Cursor::new(Vec::new()),
        finished,
        done: false,
    }))
}

/// The consumer end of [`spawn_pipe`]. A producer error surfaces here; a
/// closed channel is EOF. The `Mutex` only makes the reader `Sync`.
struct PipeReader {
    rx: Mutex<Receiver<io::Result<Vec<u8>>>>,
    current: Cursor<Vec<u8>>,
    /// Set when the producer returns normally; unset on a closed channel means
    /// it panicked.
    finished: Arc<AtomicBool>,
    done: bool,
}

impl Read for PipeReader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        loop {
            let read = self.current.read(out)?;
            if read > 0 {
                return Ok(read);
            }
            if self.done {
                return Ok(0);
            }
            let message = self.rx.lock().expect("pipe mutex poisoned").recv();
            match message {
                Ok(Ok(chunk)) => self.current = Cursor::new(chunk),
                Ok(Err(error)) => {
                    self.done = true;
                    return Err(error);
                }
                Err(_) => {
                    self.done = true;
                    if self.finished.load(Ordering::Acquire) {
                        return Ok(0);
                    }
                    return Err(io::Error::other(
                        "decompression worker stopped before completing the stream",
                    ));
                }
            }
        }
    }
}
