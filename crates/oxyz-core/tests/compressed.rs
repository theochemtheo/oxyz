//! Reading from compressed sources: codec detection, archive member
//! resolution, and parity with the plain fixture they were made from.
//!
//! Fixtures under `tests/data/compressed/` are compressed twins of
//! `two_frame_same_schema.xyz` (see that directory's README).

use std::{
    io::{Cursor, Read, Write},
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

use oxyz_core::{Compression, ExtxyzError, open_decoded, read_frames};
use proptest::prelude::*;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/data")
        .join(name)
}

fn plain() -> String {
    std::fs::read_to_string(fixture("two_frame_same_schema.xyz")).unwrap()
}

fn decoded_string(name: &str, compression: Compression, member: Option<&str>) -> String {
    let mut reader = open_decoded(&fixture(name), compression, member).unwrap();
    let mut text = String::new();
    reader.read_to_string(&mut text).unwrap();
    text
}

#[test]
fn read_frames_autodetects_every_codec() {
    let expected = read_frames(fixture("two_frame_same_schema.xyz")).unwrap();
    for name in [
        "compressed/two_frame.xyz.gz",
        "compressed/two_frame.xyz.zst",
        "compressed/two_frame.xyz.zip",
        "compressed/two_frame.tar.gz",
        "compressed/two_frame.tar",
    ] {
        let got = read_frames(fixture(name)).unwrap();
        assert_eq!(got, expected, "mismatch reading {name}");
    }
}

#[test]
fn concatenated_gzip_members_are_all_read() {
    // concat.xyz.gz is the two-frame file gzipped twice and catted: a single
    // `.gz` with two members. MultiGzDecoder must read both (plain GzDecoder
    // would silently stop after the first).
    let frames = read_frames(fixture("compressed/concat.xyz.gz")).unwrap();
    assert_eq!(frames.len(), 4);
}

#[test]
fn concatenated_zstd_frames_are_all_read() {
    // concat.xyz.zst is the two-frame file zstd-compressed and catted: a single
    // `.zst` of two frames. ruzstd's StreamingDecoder stops at the first frame,
    // so the multi-frame wrapper must drive it across the boundary (4 frames).
    let frames = read_frames(fixture("compressed/concat.xyz.zst")).unwrap();
    assert_eq!(frames.len(), 4);
}

/// Encodes a plain extxyz body into one codec's bytes.
type Encoder = fn(&str) -> Vec<u8>;

/// A multi-frame body large enough to span several pipe chunks for the archive
/// routes (the producer thread streams 64 KiB at a time) and several reads for
/// the direct decoders. Each copy of the two-frame fixture adds two frames.
fn large_body(copies: usize) -> (String, usize) {
    let one = std::fs::read_to_string(fixture("two_frame_same_schema.xyz")).unwrap();
    (one.repeat(copies), copies * 2)
}

fn gzip_bytes(plain: &str) -> Vec<u8> {
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(plain.as_bytes()).unwrap();
    encoder.finish().unwrap()
}

fn zip_bytes(plain: &str) -> Vec<u8> {
    zip_member("many.xyz", plain)
}

/// A zip holding one member of the given name and body.
fn zip_member(name: &str, body: &str) -> Vec<u8> {
    let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
    writer
        .start_file(name, zip::write::SimpleFileOptions::default())
        .unwrap();
    writer.write_all(body.as_bytes()).unwrap();
    writer.finish().unwrap().into_inner()
}

fn tar_bytes(plain: &str) -> Vec<u8> {
    tar_into(plain, Vec::new())
}

fn tar_gz_bytes(plain: &str) -> Vec<u8> {
    let encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    tar_into(plain, encoder).finish().unwrap()
}

/// Write a single `many.xyz` member into a fresh tar over `sink`, returning the
/// finished sink.
fn tar_into<W: Write>(plain: &str, sink: W) -> W {
    let mut builder = tar::Builder::new(sink);
    let mut header = tar::Header::new_gnu();
    header.set_size(plain.len() as u64);
    header.set_mode(0o644);
    header.set_cksum();
    builder
        .append_data(&mut header, "many.xyz", plain.as_bytes())
        .unwrap();
    builder.into_inner().unwrap()
}

#[test]
fn every_route_reads_a_large_multi_frame_file() {
    // 1500 copies (3000 frames, ~500 KiB) clears the 64 KiB pipe chunk many
    // times over, so a chunking or truncation bug in the streaming/archive
    // paths would drop frames here. zstd has no pure-Rust encoder, so its
    // multi-frame coverage is the committed concat fixture
    // (`concatenated_zstd_frames_are_all_read`) instead.
    let (plain, expected) = large_body(1500);
    let routes: [(&str, Encoder); 4] = [
        ("gz", gzip_bytes),
        ("zip", zip_bytes),
        ("tar", tar_bytes),
        ("tar.gz", tar_gz_bytes),
    ];
    for (extension, encode) in routes {
        let path = temp_file(&encode(&plain), extension);
        let frames = read_frames(&path).unwrap();
        assert_eq!(frames.len(), expected, "frame count for .{extension}");
        let _ = std::fs::remove_file(&path);
    }
}

#[test]
fn unknown_extension_falls_back_to_magic_bytes() {
    // A `.dat` extension says nothing, so the codec is recognised from the
    // leading magic bytes. zstd magic comes from the committed fixture; the
    // rest are encoded here.
    let one = std::fs::read_to_string(fixture("two_frame_same_schema.xyz")).unwrap();
    let zstd = std::fs::read(fixture("compressed/two_frame.xyz.zst")).unwrap();
    let cases: [(Vec<u8>, usize); 4] = [
        (one.clone().into_bytes(), 2), // plain text -> read as-is
        (gzip_bytes(&one), 2),
        (zip_bytes(&one), 2),
        (zstd, 2),
    ];
    for (bytes, expected) in cases {
        let path = temp_file(&bytes, "dat");
        assert_eq!(read_frames(&path).unwrap().len(), expected);
        let _ = std::fs::remove_file(&path);
    }
}

#[test]
fn archive_with_no_extxyz_member_errors() {
    let path = temp_file(&zip_member("notes.txt", "nothing to parse here"), "zip");
    let error = open_decoded(&path, Compression::Infer, None).err().unwrap();
    assert!(
        matches!(error, ExtxyzError::NoExtxyzMember { .. }),
        "{error:?}"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn tar_skips_directory_entries_when_resolving_a_member() {
    // A directory entry alongside the sole extxyz member must be ignored during
    // enumeration, not mistaken for a second member.
    let plain = std::fs::read_to_string(fixture("two_frame_same_schema.xyz")).unwrap();
    let mut builder = tar::Builder::new(Vec::new());
    let mut dir = tar::Header::new_gnu();
    dir.set_entry_type(tar::EntryType::Directory);
    dir.set_path("sub/").unwrap();
    dir.set_size(0);
    dir.set_mode(0o755);
    dir.set_cksum();
    builder.append(&dir, std::io::empty()).unwrap();
    let mut member = tar::Header::new_gnu();
    member.set_size(plain.len() as u64);
    member.set_mode(0o644);
    member.set_cksum();
    builder
        .append_data(&mut member, "many.xyz", plain.as_bytes())
        .unwrap();
    let path = temp_file(&builder.into_inner().unwrap(), "tar");
    assert_eq!(read_frames(&path).unwrap().len(), 2);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn corrupt_zip_surfaces_an_error() {
    // Bytes with a zip extension but no valid central directory must fail at
    // open, not panic or read garbage.
    let path = temp_file(b"PK\x03\x04 not really a zip", "zip");
    assert!(open_decoded(&path, Compression::Infer, None).is_err());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_corrupt_archive_member_surfaces_an_error_not_a_silent_truncation() {
    // A stored zip member with a flipped data byte fails its CRC check while the
    // producer thread streams it; the error must reach the reader rather than
    // ending the stream early.
    let plain = std::fs::read_to_string(fixture("two_frame_same_schema.xyz")).unwrap();
    let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
    let options =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    writer.start_file("many.xyz", options).unwrap();
    writer.write_all(plain.as_bytes()).unwrap();
    let mut bytes = writer.finish().unwrap().into_inner();
    // Flip a byte inside the stored member data (past the local header, before
    // the trailing central directory) to break its CRC.
    let middle = bytes.len() / 2;
    bytes[middle] ^= 0xff;

    let path = temp_file(&bytes, "zip");
    assert!(read_frames(&path).is_err());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn plain_path_decodes_to_itself() {
    assert_eq!(
        decoded_string("two_frame_same_schema.xyz", Compression::Infer, None),
        plain()
    );
}

#[test]
fn each_codec_round_trips_to_the_plain_bytes() {
    for name in [
        "compressed/two_frame.xyz.gz",
        "compressed/two_frame.xyz.zst",
        "compressed/two_frame.xyz.zip",
        "compressed/two_frame.tar.gz",
        "compressed/two_frame.tar",
    ] {
        assert_eq!(
            decoded_string(name, Compression::Infer, None),
            plain(),
            "decoded bytes differ for {name}"
        );
    }
}

#[test]
fn ambiguous_archive_errors_and_lists_members() {
    let err = open_decoded(
        &fixture("compressed/multi_member.zip"),
        Compression::Infer,
        None,
    )
    .err()
    .unwrap();
    match err {
        ExtxyzError::AmbiguousArchive { members } => {
            assert!(members.iter().any(|m| m == "a.xyz"), "members: {members:?}");
            assert!(members.iter().any(|m| m == "b.xyz"), "members: {members:?}");
        }
        other => panic!("expected AmbiguousArchive, got {other:?}"),
    }
}

#[test]
fn ambiguous_tar_gz_errors_too() {
    let err = open_decoded(
        &fixture("compressed/multi_member.tar.gz"),
        Compression::Infer,
        None,
    )
    .err()
    .unwrap();
    assert!(
        matches!(err, ExtxyzError::AmbiguousArchive { .. }),
        "{err:?}"
    );
}

#[test]
fn member_selects_one_from_a_multi_member_archive() {
    // a.xyz is a copy of the two-frame file.
    assert_eq!(
        decoded_string(
            "compressed/multi_member.zip",
            Compression::Infer,
            Some("a.xyz")
        ),
        plain()
    );
}

#[test]
fn missing_member_errors() {
    let err = open_decoded(
        &fixture("compressed/multi_member.zip"),
        Compression::Infer,
        Some("nope.xyz"),
    )
    .err()
    .unwrap();
    assert!(matches!(err, ExtxyzError::MemberNotFound { .. }), "{err:?}");
}

#[test]
fn member_on_non_archive_errors() {
    let err = open_decoded(
        &fixture("compressed/two_frame.xyz.gz"),
        Compression::Infer,
        Some("x.xyz"),
    )
    .err()
    .unwrap();
    assert!(matches!(err, ExtxyzError::MemberOnNonArchive), "{err:?}");
}

#[test]
fn compression_none_override_skips_decoding() {
    let mut reader = open_decoded(
        &fixture("compressed/two_frame.xyz.gz"),
        Compression::None,
        None,
    )
    .unwrap();
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes).unwrap();
    assert_eq!(&bytes[..2], &[0x1f, 0x8b], "expected raw gzip magic");
}

#[test]
fn compression_can_be_forced_for_a_misnamed_file() {
    // The gzip fixture read as if it were the named codec, ignoring extension.
    assert_eq!(
        decoded_string("compressed/two_frame.xyz.gz", Compression::Gzip, None),
        plain()
    );
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Write `bytes` to a uniquely-named temp file with the given extension and
/// return its path. Random access is never used here, so a plain temp file is
/// enough; the caller removes it.
fn temp_file(bytes: &[u8], extension: &str) -> PathBuf {
    let id = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let path =
        std::env::temp_dir().join(format!("oxyz_fuzz_{}_{id}.{extension}", std::process::id()));
    std::fs::File::create(&path)
        .unwrap()
        .write_all(bytes)
        .unwrap();
    path
}

proptest! {
    /// Arbitrary bytes presented as any codec must never panic: decoding either
    /// fails cleanly or yields garbage the parser rejects. Truncated and
    /// malformed archive headers are the cases that matter.
    #[test]
    fn arbitrary_bytes_never_panic(bytes in proptest::collection::vec(any::<u8>(), 0..512)) {
        for (extension, compression) in [
            ("gz", Compression::Gzip),
            ("zst", Compression::Zstd),
            ("zip", Compression::Zip),
            ("tar.gz", Compression::Infer),
            ("tar", Compression::Infer),
            ("xyz", Compression::None),
        ] {
            let path = temp_file(&bytes, extension);
            // open_decoded may fail at the header; if it opens, draining the
            // reader must also not panic.
            if let Ok(mut reader) = open_decoded(&path, compression, None) {
                let mut sink = Vec::new();
                let _ = reader.read_to_end(&mut sink);
            }
            let _ = read_frames(&path);
            let _ = std::fs::remove_file(&path);
        }
    }
}
