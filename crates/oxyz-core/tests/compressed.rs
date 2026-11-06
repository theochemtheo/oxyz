//! Reading from compressed sources. Fixtures under `tests/data/compressed/`
//! are compressed twins of `two_frame_same_schema.xyz`.

use std::{
    io::{Cursor, Read, Write},
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

use oxyz_core::{
    ALL_MEMBERS, Compression, ExtxyzError, detect_codec_name, iter_frames_from, open_decoded,
    read_frames,
};
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
        "compressed/two_frame.tar.zst",
        "compressed/two_frame.tar",
    ] {
        let got = read_frames(fixture(name)).unwrap();
        assert_eq!(got, expected, "mismatch reading {name}");
    }
}

/// `.tar.zst` also ends in `.zst`, so the tar arm must match first.
#[test]
fn tar_zstd_extensions_beat_the_bare_zstd_arm() {
    for (name, expected) in [
        ("run.tar.zst", "tar.zst"),
        ("run.tzst", "tar.zst"),
        ("run.xyz.zst", "zstd"),
        ("run.tar.gz", "tar.gz"),
        ("run.tgz", "tar.gz"),
        ("run.tar", "tar"),
    ] {
        assert_eq!(detect_codec_name(name, None), expected, "codec for {name}");
    }
}

#[test]
fn concatenated_gzip_members_are_all_read() {
    // Two gzip members in one file; plain GzDecoder stops after the first.
    let frames = read_frames(fixture("compressed/concat.xyz.gz")).unwrap();
    assert_eq!(frames.len(), 4);
}

#[test]
fn concatenated_zstd_frames_are_all_read() {
    // Two zstd frames in one file; ruzstd's StreamingDecoder stops after the first.
    let frames = read_frames(fixture("compressed/concat.xyz.zst")).unwrap();
    assert_eq!(frames.len(), 4);
}

/// Encodes a plain extxyz body into one codec's bytes.
type Encoder = fn(&str) -> Vec<u8>;

/// A body spanning several 64 KiB pipe chunks; each copy adds two frames.
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

/// Write a single `many.xyz` member into a fresh tar over `sink`.
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
    // Many pipe chunks, so a chunking or truncation bug drops frames. zstd has no
    // pure-Rust encoder; `concatenated_zstd_frames_are_all_read` covers it.
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
    // `.dat` says nothing, so the magic bytes decide.
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
    // A directory entry must not count as a second member.
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
    let path = temp_file(b"PK\x03\x04 not really a zip", "zip");
    assert!(open_decoded(&path, Compression::Infer, None).is_err());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_corrupt_archive_member_surfaces_an_error_not_a_silent_truncation() {
    // The CRC failure happens on the producer thread and must reach the reader.
    let plain = std::fs::read_to_string(fixture("two_frame_same_schema.xyz")).unwrap();
    let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
    let options =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    writer.start_file("many.xyz", options).unwrap();
    writer.write_all(plain.as_bytes()).unwrap();
    let mut bytes = writer.finish().unwrap().into_inner();
    // Past the local header, before the central directory.
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

/// A tar has no magic bytes, so only a forced codec reads one under a blind
/// `.bin` name. The original names would pass on extension inference alone.
#[test]
fn tar_codecs_can_be_forced_regardless_of_name() {
    for (name, compression) in [
        ("compressed/two_frame.tar", Compression::Tar),
        ("compressed/two_frame.tar.gz", Compression::TarGzip),
        ("compressed/two_frame.tar.zst", Compression::TarZstd),
    ] {
        let bytes = std::fs::read(fixture(name)).unwrap();
        let path = temp_file(&bytes, "bin");

        let mut reader = open_decoded(&path, compression, None)
            .unwrap_or_else(|e| panic!("failed forcing {name}: {e}"));
        let mut text = String::new();
        reader
            .read_to_string(&mut text)
            .unwrap_or_else(|e| panic!("failed forcing {name}: {e}"));
        assert_eq!(text, plain(), "mismatch forcing {name}");

        // Without forcing, the parser is handed raw tar headers.
        assert!(
            read_frames(&path).is_err(),
            "expected Infer to fail reading {name} without forcing"
        );

        let _ = std::fs::remove_file(&path);
    }
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Write `bytes` to a uniquely named temp file; the caller removes it.
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
    /// Arbitrary bytes presented as any codec fail cleanly, never panic.
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
            if let Ok(mut reader) = open_decoded(&path, compression, None) {
                let mut sink = Vec::new();
                let _ = reader.read_to_end(&mut sink);
            }
            let _ = read_frames(&path);
            let _ = std::fs::remove_file(&path);
        }
    }
}

/// A zip holding the given members, in order.
fn zip_members(members: &[(&str, &str)]) -> Vec<u8> {
    let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
    for (name, body) in members {
        writer
            .start_file(*name, zip::write::SimpleFileOptions::default())
            .unwrap();
        writer.write_all(body.as_bytes()).unwrap();
    }
    writer.finish().unwrap().into_inner()
}

/// A tar holding the given members, in order.
fn tar_members(members: &[(&str, &str)]) -> Vec<u8> {
    let mut builder = tar::Builder::new(Vec::new());
    for (name, body) in members {
        let mut header = tar::Header::new_gnu();
        header.set_size(body.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder
            .append_data(&mut header, *name, body.as_bytes())
            .unwrap();
    }
    builder.into_inner().unwrap()
}

/// The `member="*"` stream of `members`, through both archive routes.
fn all_members_text(members: &[(&str, &str)]) -> String {
    let [from_zip, from_tar] =
        [(zip_members(members), "zip"), (tar_members(members), "tar")].map(|(bytes, extension)| {
            let path = temp_file(&bytes, extension);
            let mut text = String::new();
            open_decoded(&path, Compression::Infer, Some(ALL_MEMBERS))
                .unwrap()
                .read_to_string(&mut text)
                .unwrap();
            let _ = std::fs::remove_file(&path);
            text
        });
    assert_eq!(from_zip, from_tar);
    from_zip
}

const ONE_ATOM: &str = "1\nProperties=species:S:1:pos:R:3\nH 0.0 0.0 0.0\n";
const TWO_ATOMS: &str = "2\nProperties=species:S:1:pos:R:3\nH 0.0 0.0 0.0\nO 1.0 0.0 0.0\n";

#[test]
fn all_members_concatenates_every_codec_route_in_archive_order() {
    let first = plain();
    let second = std::fs::read_to_string(fixture("varying_atom_counts.xyz")).unwrap();
    let expected = format!("{first}{second}");
    for name in [
        "compressed/multi_member.zip",
        "compressed/multi_member.tar.gz",
    ] {
        assert_eq!(
            decoded_string(name, Compression::Infer, Some(ALL_MEMBERS)),
            expected,
            "{name}"
        );
    }
}

#[test]
fn all_members_skips_other_files_and_macos_resource_forks() {
    let text = all_members_text(&[
        ("README.txt", "not a frame"),
        ("__MACOSX/._a.xyz", "\0\u{5}\u{16}\u{7}"),
        ("sub/._b.xyz", "\0\u{5}\u{16}\u{7}"),
        ("a.xyz", ONE_ATOM),
        ("sub/b.extxyz", TWO_ATOMS),
    ]);
    assert_eq!(text, format!("{ONE_ATOM}{TWO_ATOMS}"));
}

#[test]
fn a_blank_count_line_ends_its_member_not_the_stream() {
    let text = all_members_text(&[
        ("a.xyz", &format!("{ONE_ATOM}\n")),
        (
            "b.xyz",
            &format!("{ONE_ATOM}  \r\n\nignored, as in a lone file\n"),
        ),
        ("c.xyz", TWO_ATOMS),
    ]);
    assert_eq!(text, format!("{ONE_ATOM}{ONE_ATOM}{TWO_ATOMS}"));
}

#[test]
fn a_member_without_a_final_newline_does_not_merge_into_the_next() {
    let unterminated = ONE_ATOM.trim_end();
    let text = all_members_text(&[("a.xyz", unterminated), ("b.xyz", TWO_ATOMS)]);
    assert_eq!(text, format!("{ONE_ATOM}{TWO_ATOMS}"));
}

#[test]
fn an_empty_member_contributes_nothing() {
    let text = all_members_text(&[("a.xyz", ONE_ATOM), ("empty.xyz", ""), ("b.xyz", TWO_ATOMS)]);
    assert_eq!(text, format!("{ONE_ATOM}{TWO_ATOMS}"));
}

#[test]
fn an_empty_comment_is_not_mistaken_for_a_member_end() {
    let zero_atoms_blank_comment = "0\n\n";
    let blank_comment = "1\n\nH 0.0 0.0 0.0\n";
    let text = all_members_text(&[
        ("a.xyz", zero_atoms_blank_comment),
        ("b.xyz", blank_comment),
        ("c.xyz", ONE_ATOM),
    ]);
    assert_eq!(
        text,
        format!("{zero_atoms_blank_comment}{blank_comment}{ONE_ATOM}")
    );
}

#[test]
fn a_member_ending_inside_a_frame_errors_naming_the_member() {
    let truncated = "2\nProperties=species:S:1:pos:R:3\nH 0.0 0.0 0.0\n";
    let members = [
        ("a.xyz", ONE_ATOM),
        ("short.xyz", truncated),
        ("c.xyz", TWO_ATOMS),
    ];
    for (bytes, extension) in [
        (zip_members(&members), "zip"),
        (tar_members(&members), "tar"),
    ] {
        let path = temp_file(&bytes, extension);
        let mut reader = open_decoded(&path, Compression::Infer, Some(ALL_MEMBERS)).unwrap();
        let error = reader.read_to_end(&mut Vec::new()).unwrap_err().to_string();
        assert!(error.contains("short.xyz"), "{extension}: {error}");
        let _ = std::fs::remove_file(&path);
    }
}

#[test]
fn an_invalid_count_line_is_left_for_the_parser_to_report() {
    let members = [
        ("a.xyz", ONE_ATOM),
        ("bad.xyz", "not a count\n"),
        ("c.xyz", ONE_ATOM),
    ];
    let path = temp_file(&zip_members(&members), "zip");
    let reader = open_decoded(&path, Compression::Infer, Some(ALL_MEMBERS)).unwrap();
    let mut frames = iter_frames_from(reader).unwrap();
    assert!(frames.next().unwrap().is_ok());
    let error = frames.next().unwrap().unwrap_err().to_string();
    assert!(error.contains("not a count"), "{error}");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn all_members_without_an_extxyz_member_errors() {
    let path = temp_file(&zip_member("notes.txt", "nothing"), "zip");
    let error = open_decoded(&path, Compression::Infer, Some(ALL_MEMBERS))
        .err()
        .unwrap();
    assert!(
        matches!(error, ExtxyzError::NoExtxyzMember { .. }),
        "{error:?}"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn all_members_on_a_non_archive_errors() {
    let error = open_decoded(
        &fixture("two_frame_same_schema.xyz"),
        Compression::Infer,
        Some(ALL_MEMBERS),
    )
    .err()
    .unwrap();
    assert!(
        matches!(error, ExtxyzError::MemberOnNonArchive),
        "{error:?}"
    );
}

#[test]
fn many_small_members_all_arrive() {
    let names: Vec<String> = (0..5000).map(|i| format!("{i}.xyz")).collect();
    let members: Vec<(&str, &str)> = names.iter().map(|name| (name.as_str(), ONE_ATOM)).collect();
    let path = temp_file(&zip_members(&members), "zip");
    let reader = open_decoded(&path, Compression::Infer, Some(ALL_MEMBERS)).unwrap();
    assert_eq!(iter_frames_from(reader).unwrap().count(), 5000);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn errors_listing_members_show_a_bounded_preview_and_keep_the_full_list() {
    let members: Vec<String> = (0..1000).map(|i| format!("wbm-{i}.extxyz")).collect();
    let errors = [
        ExtxyzError::AmbiguousArchive {
            members: members.clone(),
        },
        ExtxyzError::NoExtxyzMember {
            members: members.clone(),
        },
        ExtxyzError::MemberNotFound {
            member: "nope.xyz".to_owned(),
            available: members.clone(),
        },
    ];
    for error in errors {
        let message = error.to_string();
        assert!(message.len() < 500, "{} bytes: {message}", message.len());
        assert!(message.contains("wbm-0.extxyz"), "{message}");
        assert!(!message.contains("wbm-999.extxyz"), "{message}");
        assert!(message.contains("990 more"), "{message}");
    }
}

#[test]
fn a_short_member_list_is_shown_in_full() {
    let message = ExtxyzError::AmbiguousArchive {
        members: vec!["a.xyz".to_owned(), "b.xyz".to_owned()],
    }
    .to_string();
    assert!(message.contains(r#"["a.xyz", "b.xyz"]"#), "{message}");
    assert!(!message.contains("more"), "{message}");
}

#[test]
fn the_ambiguous_error_names_the_all_members_selector() {
    let message = ExtxyzError::AmbiguousArchive {
        members: vec!["a.xyz".to_owned(), "b.xyz".to_owned()],
    }
    .to_string();
    assert!(message.contains(r#"member="*""#), "{message}");
}

proptest! {
    /// Any split of frames into members reads like the one concatenated file,
    /// with or without final newlines and trailing blank lines on each member.
    #[test]
    fn splitting_frames_across_members_changes_nothing(
        parts in proptest::collection::vec((1usize..4, any::<bool>(), any::<bool>()), 1..8),
    ) {
        let mut whole = String::new();
        let mut bodies = Vec::new();
        for (frames, drop_newline, blank_tail) in &parts {
            let body = ONE_ATOM.repeat(*frames);
            whole.push_str(&body);
            bodies.push(match (drop_newline, blank_tail) {
                (true, _) => body.trim_end().to_owned(),
                (false, true) => format!("{body}\n"),
                (false, false) => body,
            });
        }
        let names: Vec<String> = (0..bodies.len()).map(|i| format!("{i}.xyz")).collect();
        let members: Vec<(&str, &str)> = names
            .iter()
            .zip(&bodies)
            .map(|(name, body)| (name.as_str(), body.as_str()))
            .collect();
        prop_assert_eq!(all_members_text(&members), whole);
    }

    /// Arbitrary member bytes fail cleanly, never panic.
    #[test]
    fn arbitrary_member_bytes_never_panic(
        first in proptest::collection::vec(any::<u8>(), 0..256),
        second in proptest::collection::vec(any::<u8>(), 0..256),
    ) {
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        for (name, bytes) in [("a.xyz", &first), ("b.xyz", &second)] {
            writer.start_file(name, zip::write::SimpleFileOptions::default()).unwrap();
            writer.write_all(bytes).unwrap();
        }
        let path = temp_file(&writer.finish().unwrap().into_inner(), "zip");
        let mut reader = open_decoded(&path, Compression::Infer, Some(ALL_MEMBERS)).unwrap();
        let _ = reader.read_to_end(&mut Vec::new());
        let _ = std::fs::remove_file(&path);
    }
}
