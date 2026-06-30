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
