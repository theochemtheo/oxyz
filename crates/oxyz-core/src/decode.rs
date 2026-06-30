//! Opening a read source through a decompressing layer.
//!
//! [`open_decoded`] turns a path into a streaming [`BufRead`], detecting the
//! codec from the extension (with a magic-byte fallback) or honouring an
//! explicit [`Compression`]. Everything streams: nothing is decompressed to a
//! temporary file or held whole in memory.
//!
//! Random access is out of scope — a compressed stream is not seekable, so the
//! seek-based [`IndexedFrames`](crate::extxyz::IndexedFrames) path stays on raw
//! files. The streaming and parallel readers consume a `BufRead` and so work
//! unchanged over a decoded source.

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

/// A decoded, streaming read source. Boxed so a single type spans every codec.
/// `Sync` as well as `Send` so the binding can hold one in a `#[pyclass]`.
pub type DecodedReader = Box<dyn BufRead + Send + Sync>;

/// A raw, undecoded byte source — a file, or a remote stream from the binding.
/// `Send + Sync` so it can cross into the codec wrappers and the pyclass.
pub type ByteSource = Box<dyn Read + Send + Sync>;

/// How to interpret a read source. `Infer` reads the extension, falling back to
/// a magic-byte sniff; the rest force a codec regardless of the name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Compression {
    #[default]
    Infer,
    /// No decompression — read the bytes as-is.
    None,
    Gzip,
    Zstd,
    Zip,
}

/// The concrete codec, after inference. Archive codecs (`Zip`, `Tar`,
/// `TarGzip`) carry members and accept a `member` selector; the rest are single
/// streams and reject one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    Plain,
    Gzip,
    Zstd,
    Zip,
    Tar,
    TarGzip,
}

impl Codec {
    pub(crate) fn is_archive(self) -> bool {
        matches!(self, Codec::Zip | Codec::Tar | Codec::TarGzip)
    }
}

/// Resolve the codec for *writing*. Unlike [`detect`], there are no bytes to
/// sniff (the file may not exist yet), so `Infer` reads the extension only and
/// falls back to [`Codec::Plain`] — including for stdout (`-`), which has none.
pub(crate) fn detect_for_write(path: &Path, compression: Compression) -> Codec {
    match compression {
        Compression::None => Codec::Plain,
        Compression::Gzip => Codec::Gzip,
        Compression::Zstd => Codec::Zstd,
        Compression::Zip => Codec::Zip,
        Compression::Infer => detect_by_extension(path).unwrap_or(Codec::Plain),
    }
}

/// Open `path` as a streaming reader, decompressing per `compression`.
///
/// `member` names one entry inside an archive (`.zip`, `.tar`, `.tar.gz`); it is
/// rejected for single-stream sources. With no `member`, an archive must hold
/// exactly one extxyz-looking member (`.xyz`/`.extxyz`), else the call errors
/// and lists what it found.
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
                move || File::open(&path).map(|f| Box::new(f) as Box<dyn Read + Send>),
                member,
                false,
            )
        }
        Codec::TarGzip => {
            let path = path.to_owned();
            wrap_tar(
                move || File::open(&path).map(|f| Box::new(f) as Box<dyn Read + Send>),
                member,
                true,
            )
        }
    }
}

/// Wrap an already-opened raw byte source in a single-stream codec. Archive
/// codecs (`Zip`/`Tar`/`TarGzip`) are not streams and are rejected — use
/// [`wrap_zip`] / [`wrap_tar`].
pub fn wrap_stream(source: ByteSource, codec: Codec) -> Result<DecodedReader> {
    match codec {
        Codec::Plain => Ok(Box::new(BufReader::new(source))),
        Codec::Gzip => Ok(Box::new(BufReader::new(MultiGzDecoder::new(source)))),
        Codec::Zstd => Ok(Box::new(BufReader::new(MultiFrameZstd::from_source(
            source,
        )?))),
        Codec::Zip | Codec::Tar | Codec::TarGzip => Err(ExtxyzError::MemberOnNonArchive),
    }
}

/// Whether `path`, read under `compression`, needs decompression — i.e. is not
/// a plain seekable file. The random-access reader uses this to refuse a
/// compressed source up front.
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
        Compression::Infer => Ok(detect_by_extension(path).map_or_else(|| sniff(path), Ok)?),
    }
}

pub(crate) fn detect_by_extension(path: &Path) -> Option<Codec> {
    let name = path.file_name()?.to_str()?.to_ascii_lowercase();
    Some(if name.ends_with(".tar.gz") || name.ends_with(".tgz") {
        Codec::TarGzip
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

/// Magic-byte fallback for a file whose extension says nothing. A bare `.gz`
/// magic is reported as `Gzip`, not `TarGzip`: a tar inside cannot be told from
/// the first bytes, and a plain-gzip read of a tar would surface as a parse
/// error, not silent corruption.
fn sniff(path: &Path) -> Result<Codec> {
    let mut head = [0u8; 4];
    let read = File::open(path)?.read(&mut head)?;
    let head = &head[..read];
    Ok(if head.starts_with(&[0x1f, 0x8b]) {
        Codec::Gzip
    } else if head.starts_with(&[0x28, 0xb5, 0x2f, 0xfd]) {
        Codec::Zstd
    } else if head.starts_with(b"PK\x03\x04") {
        Codec::Zip
    } else {
        Codec::Plain
    })
}

/// Source for the zstd decoder: boxed so the leftover stream can be re-wrapped
/// (with the peeked byte prepended) to start the next frame.
type ZstdSource = Box<dyn Read + Send + Sync>;

/// Streams every frame of a `.zst` file. ruzstd's [`StreamingDecoder`] stops at
/// the first frame, but a zstd file may concatenate several (`zstd` does this,
/// as does `cat a.zst b.zst`); this re-creates the decoder at each frame
/// boundary, mirroring how [`MultiGzDecoder`] reads concatenated gzip members.
struct MultiFrameZstd {
    decoder: Option<StreamingDecoder<ZstdSource, FrameDecoder>>,
}

impl MultiFrameZstd {
    fn from_source(source: ZstdSource) -> Result<Self> {
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
        // A zero-length read returns 0 without reading; without this guard it
        // would be read as frame-exhaustion and wrongly advance to the next
        // frame, dropping a byte.
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
            // Current frame is exhausted. Peek one byte off the leftover source:
            // nothing left is clean EOF, otherwise another frame follows.
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

/// Pick the member to read: the named one if given (else `MemberNotFound`), or
/// the sole extxyz-looking member (else `NoExtxyzMember` / `AmbiguousArchive`).
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

/// Stream one member of a zip from a seekable source (the central directory is
/// at the end, so `Seek` is required — a plain stream cannot back this).
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

/// Stream one member of a tar (optionally gzip-compressed). `factory` yields a
/// fresh *raw* tar byte stream on each call; enumeration and streaming each open
/// one (two passes — a tar has no central directory).
pub fn wrap_tar<F>(factory: F, member: Option<&str>, gzip: bool) -> Result<DecodedReader>
where
    F: Fn() -> io::Result<Box<dyn Read + Send>> + Send + 'static,
{
    let names = tar_member_names(&factory, gzip)?;
    let target = resolve_member(&names, member)?;

    Ok(spawn_pipe(move |tx| {
        let stream = match tar_stream_from(&factory, gzip) {
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

fn tar_member_names<F>(factory: &F, gzip: bool) -> Result<Vec<String>>
where
    F: Fn() -> io::Result<Box<dyn Read + Send>>,
{
    let mut archive = tar::Archive::new(tar_stream_from(factory, gzip)?);
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

fn tar_stream_from<F>(factory: &F, gzip: bool) -> io::Result<Box<dyn Read + Send>>
where
    F: Fn() -> io::Result<Box<dyn Read + Send>>,
{
    let raw = factory()?;
    Ok(if gzip {
        Box::new(MultiGzDecoder::new(raw))
    } else {
        raw
    })
}

fn entry_name<R: Read>(entry: &tar::Entry<'_, R>) -> Option<String> {
    entry.path().ok()?.to_str().map(str::to_owned)
}

fn zip_error(error: zip::result::ZipError) -> ExtxyzError {
    ExtxyzError::Io(io::Error::other(error.to_string()))
}

/// 64 KiB per chunk handed across the producer/consumer boundary.
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

/// Run `producer` on its own thread, streaming its bytes back through a bounded
/// channel. Archive readers (`zip`, `tar`) borrow their archive, so they cannot
/// be returned directly; the producer owns the archive for its whole lifetime
/// and the bounded channel gives back-pressure, keeping memory flat.
fn spawn_pipe<F>(producer: F) -> DecodedReader
where
    F: FnOnce(&SyncSender<io::Result<Vec<u8>>>) + Send + 'static,
{
    let (tx, rx) = sync_channel::<io::Result<Vec<u8>>>(4);
    let finished = Arc::new(AtomicBool::new(false));
    let producer_finished = Arc::clone(&finished);
    thread::spawn(move || {
        producer(&tx);
        // Set before `tx` drops (the consumer only wakes once the channel
        // closes). A panic in `producer` skips this, turning the otherwise
        // silent truncation into a surfaced error rather than a short read.
        producer_finished.store(true, Ordering::Release);
    });
    Box::new(BufReader::new(PipeReader {
        rx: Mutex::new(rx),
        current: Cursor::new(Vec::new()),
        finished,
        done: false,
    }))
}

/// The consumer end of [`spawn_pipe`]: serves the current chunk, then blocks on
/// the next. A producer error surfaces here; a closed channel is clean EOF. The
/// `Receiver` sits behind a `Mutex` only to make the reader `Sync` (single
/// consumer, so the lock is never contended).
struct PipeReader {
    rx: Mutex<Receiver<io::Result<Vec<u8>>>>,
    current: Cursor<Vec<u8>>,
    /// Set by the producer thread once it returns normally. A closed channel
    /// with this unset means the producer aborted (panicked) mid-stream.
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
