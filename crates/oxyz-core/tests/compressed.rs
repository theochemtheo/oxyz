//! Reading from compressed sources: codec detection, archive member
//! resolution, and parity with the plain fixture they were made from.
//!
//! Fixtures under `tests/data/compressed/` are compressed twins of
//! `two_frame_same_schema.xyz` (see that directory's README).

use std::{
    io::{Read, Write},
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
