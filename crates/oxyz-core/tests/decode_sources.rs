use std::io::{Cursor, Read};

use flate2::{Compression, write::GzEncoder};
use oxyz_core::decode::{ByteSource, Codec, wrap_stream};

fn gzip(bytes: &[u8]) -> Vec<u8> {
    use std::io::Write;
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(bytes).unwrap();
    encoder.finish().unwrap()
}

#[test]
fn wrap_stream_plain_passes_bytes_through() {
    let source: ByteSource = Box::new(Cursor::new(b"hello world".to_vec()));
    let mut reader = wrap_stream(source, Codec::Plain).unwrap();
    let mut out = String::new();
    reader.read_to_string(&mut out).unwrap();
    assert_eq!(out, "hello world");
}

#[test]
fn wrap_stream_gzip_decodes() {
    let source: ByteSource = Box::new(Cursor::new(gzip(b"hello world")));
    let mut reader = wrap_stream(source, Codec::Gzip).unwrap();
    let mut out = String::new();
    reader.read_to_string(&mut out).unwrap();
    assert_eq!(out, "hello world");
}

#[test]
fn wrap_stream_rejects_archive_codec() {
    let source: ByteSource = Box::new(Cursor::new(Vec::new()));
    assert!(wrap_stream(source, Codec::Zip).is_err());
}

fn tar_two_members() -> Vec<u8> {
    let mut builder = tar::Builder::new(Vec::new());
    for (name, body) in [("a.txt", b"alpha".as_slice()), ("b.xyz", b"beta")] {
        let mut header = tar::Header::new_gnu();
        header.set_size(body.len() as u64);
        header.set_cksum();
        builder.append_data(&mut header, name, body).unwrap();
    }
    builder.into_inner().unwrap()
}

#[test]
fn wrap_tar_selects_named_member() {
    let bytes = tar_two_members();
    let factory = move || Ok(Box::new(Cursor::new(bytes.clone())) as Box<dyn std::io::Read + Send>);
    let mut reader = oxyz_core::decode::wrap_tar(factory, Some("b.xyz"), false).unwrap();
    let mut out = String::new();
    std::io::Read::read_to_string(&mut reader, &mut out).unwrap();
    assert_eq!(out, "beta");
}

fn zip_two_members() -> Vec<u8> {
    use zip::write::SimpleFileOptions;
    let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
    let options = SimpleFileOptions::default();
    for (name, body) in [("a.txt", "alpha"), ("b.xyz", "beta")] {
        writer.start_file(name, options).unwrap();
        std::io::Write::write_all(&mut writer, body.as_bytes()).unwrap();
    }
    writer.finish().unwrap().into_inner()
}

#[test]
fn wrap_zip_selects_named_member() {
    let bytes = zip_two_members();
    let mut reader = oxyz_core::decode::wrap_zip(Cursor::new(bytes), Some("b.xyz")).unwrap();
    let mut out = String::new();
    std::io::Read::read_to_string(&mut reader, &mut out).unwrap();
    assert_eq!(out, "beta");
}
