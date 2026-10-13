//! Writing frames: codec round-trips, append, rejections, and the lossless
//! round-trip invariant. A failing test leaves its temp file for inspection.

use std::{
    fs,
    io::BufReader,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

use oxyz_core::{
    Column, ColumnData, Compression, ExtxyzError, Frame, FrameSink, Value, iter_frames_from,
    open_decoded, read_frames, write_frames, write_frames_parallel,
};
use proptest::prelude::*;

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// A unique temp path ending in `name`, so codec inference sees the extension.
fn temp_path(name: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("oxyz_write_{}_{n}_{name}", std::process::id()))
}

fn col(name: &str, width: usize, data: ColumnData) -> Column {
    Column {
        name: name.into(),
        width,
        data,
    }
}

/// Two frames already in write order, so a round trip compares equal.
fn sample_frames() -> Vec<Frame> {
    let frame = Frame {
        n_atoms: 2,
        columns: vec![
            col("species", 1, ColumnData::Str(vec!["Si".into(), "O".into()])),
            col(
                "pos",
                3,
                ColumnData::Real(vec![0.0, 1.5, -2.25, 3.5, 4.0, 5.0]),
            ),
            col("tag", 1, ColumnData::Int(vec![1, -2])),
        ],
        metadata: vec![("energy".into(), Value::Real(-12.5))],
    };
    vec![frame.clone(), frame]
}

#[test]
fn every_codec_round_trips_through_the_reader() {
    let frames = sample_frames();
    for name in [
        "two.xyz",
        "two.extxyz",
        "two.xyz.gz",
        "two.xyz.zip",
        "two.tar",
        "two.tar.gz",
    ] {
        let path = temp_path(name);
        write_frames(&path, &frames, Compression::Infer, None, false).unwrap();
        let back = read_frames(&path).unwrap();
        assert_eq!(back, frames, "round trip mismatch for {name}");
        fs::remove_file(&path).unwrap();
    }
}

#[test]
fn explicit_compression_overrides_a_plain_extension() {
    let frames = sample_frames();
    let path = temp_path("forced_gzip.xyz");
    write_frames(&path, &frames, Compression::Gzip, Some(9), false).unwrap();
    // The name says plain, so reading must be told the codec too.
    let back = read_frames(&path).unwrap();
    // Auto-detect falls back to the magic bytes and still reads it.
    assert_eq!(back, frames);
    fs::remove_file(&path).unwrap();
}

#[test]
fn forcing_tar_codecs_on_write_produces_a_readable_archive() {
    // Unhelpful names, so the codec can only come from the forced argument.
    let frames = sample_frames();
    for (name, compression) in [
        ("forced_tar.bin", Compression::Tar),
        ("forced_tar_gz.bin", Compression::TarGzip),
    ] {
        let path = temp_path(name);
        write_frames(&path, &frames, compression, None, false).unwrap();

        let reader = open_decoded(&path, compression, None)
            .unwrap_or_else(|e| panic!("failed reading back {name}: {e}"));
        let back: Vec<Frame> = iter_frames_from(BufReader::new(reader))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap_or_else(|e| panic!("failed parsing {name}: {e}"));
        assert_eq!(back, frames, "round trip mismatch forcing {name}");
        fs::remove_file(&path).unwrap();
    }
}

#[test]
fn append_concatenates_for_plain_and_gzip() {
    let frames = sample_frames();
    for name in ["append.xyz", "append.xyz.gz"] {
        let path = temp_path(name);
        write_frames(&path, &frames, Compression::Infer, None, false).unwrap();
        write_frames(&path, &frames, Compression::Infer, None, true).unwrap();
        let back = read_frames(&path).unwrap();
        assert_eq!(back.len(), 4, "append mismatch for {name}");
        assert_eq!(back[3], frames[1]);
        fs::remove_file(&path).unwrap();
    }
}

#[test]
fn append_is_rejected_for_archive_codecs() {
    for name in ["a.xyz.zip", "a.tar", "a.tar.gz"] {
        let path = temp_path(name);
        let result = FrameSink::create(&path, Compression::Infer, None, true);
        assert!(
            matches!(result, Err(ExtxyzError::AppendUnsupported { .. })),
            "expected append refusal for {name}"
        );
    }
}

#[test]
fn zstd_write_is_refused() {
    let path = temp_path("x.xyz.zst");
    let by_extension = FrameSink::create(&path, Compression::Infer, None, false);
    assert!(matches!(
        by_extension,
        Err(ExtxyzError::ZstdWriteUnsupported)
    ));
    let by_request = FrameSink::create(&temp_path("x.xyz"), Compression::Zstd, None, false);
    assert!(matches!(by_request, Err(ExtxyzError::ZstdWriteUnsupported)));

    // Forcing the codec does not make zstd writable either.
    let tar_zstd_forced = FrameSink::create(&temp_path("x.bin"), Compression::TarZstd, None, false);
    assert!(matches!(
        tar_zstd_forced,
        Err(ExtxyzError::ZstdWriteUnsupported)
    ));
}

#[test]
fn tar_zstd_write_is_refused_not_unreachable() {
    // No zstd encoder is available, so this must fail cleanly, not panic.
    for name in ["x.tar.zst", "x.tzst"] {
        let result = FrameSink::create(&temp_path(name), Compression::Infer, None, false);
        assert!(
            matches!(result, Err(ExtxyzError::ZstdWriteUnsupported)),
            "expected refusal for {name}"
        );
    }
}

#[test]
fn out_of_range_level_is_refused() {
    let result = FrameSink::create(&temp_path("x.xyz.gz"), Compression::Infer, Some(12), false);
    assert!(matches!(
        result,
        Err(ExtxyzError::InvalidCompressionLevel { level: 12 })
    ));
}

/// `count` distinct frames in write order, varied so a reordering would show.
fn many_frames(count: usize) -> Vec<Frame> {
    (0..count)
        .map(|i| {
            let f = i as f64;
            Frame {
                n_atoms: 2,
                columns: vec![
                    col("species", 1, ColumnData::Str(vec!["Si".into(), "O".into()])),
                    col(
                        "pos",
                        3,
                        ColumnData::Real(vec![f, 1.5, -2.25, 3.5 + f, 4.0, 5.0]),
                    ),
                    col("tag", 1, ColumnData::Int(vec![i as i64, -(i as i64)])),
                ],
                metadata: vec![("energy".into(), Value::Real(-12.5 * f))],
            }
        })
        .collect()
}

#[test]
fn parallel_write_is_byte_identical_to_serial() {
    let frames = many_frames(200);
    // Archive headers embed the file name, so serial and parallel share a path.
    for name in ["p.xyz", "p.xyz.gz", "p.xyz.zip", "p.tar", "p.tar.gz"] {
        let path = temp_path(name);
        write_frames(&path, &frames, Compression::Infer, None, false).unwrap();
        let serial_bytes = fs::read(&path).unwrap();

        for threads in [None, Some(1), Some(2), Some(8)] {
            write_frames_parallel(&path, &frames, Compression::Infer, None, false, threads)
                .unwrap();
            assert_eq!(
                fs::read(&path).unwrap(),
                serial_bytes,
                "parallel bytes differ from serial for {name} at threads={threads:?}"
            );
        }
        fs::remove_file(&path).unwrap();
    }
}

#[test]
fn parallel_write_reports_the_first_invalid_frame_and_leaves_no_file() {
    // The frame missing `pos` is reported, as in the serial writer, and no file
    // is left behind.
    let mut frames = many_frames(50);
    frames[20] = Frame {
        n_atoms: 1,
        columns: vec![col("species", 1, ColumnData::Str(vec!["H".into()]))],
        metadata: vec![],
    };

    let serial = temp_path("bad_serial.xyz");
    let serial_err = write_frames(&serial, &frames, Compression::Infer, None, false).unwrap_err();
    assert!(matches!(
        serial_err,
        ExtxyzError::MissingRequiredColumn { name: "pos" }
    ));
    let _ = fs::remove_file(&serial);

    let path = temp_path("bad_parallel.xyz");
    let err =
        write_frames_parallel(&path, &frames, Compression::Infer, None, false, None).unwrap_err();
    assert!(matches!(
        err,
        ExtxyzError::MissingRequiredColumn { name: "pos" }
    ));
    assert!(!path.exists(), "a rejected write must leave no output file");
}

#[test]
fn batched_sink_writes_match_a_serial_whole_file() {
    // Writer(batch=N): batched writes equal a one-shot serial write.
    let frames = many_frames(100);
    for name in ["batch.xyz", "batch.xyz.gz"] {
        let serial = temp_path(name);
        write_frames(&serial, &frames, Compression::Infer, None, false).unwrap();
        let serial_bytes = fs::read(&serial).unwrap();
        fs::remove_file(&serial).unwrap();

        for batch in [1usize, 7, 100, 256] {
            let path = temp_path(name);
            let mut sink = FrameSink::create(&path, Compression::Infer, None, false).unwrap();
            for chunk in frames.chunks(batch) {
                sink.write_batch_parallel(chunk, None).unwrap();
            }
            sink.finish().unwrap();
            assert_eq!(
                fs::read(&path).unwrap(),
                serial_bytes,
                "batched write differs from serial for {name} at batch={batch}"
            );
            fs::remove_file(&path).unwrap();
        }
    }
}

/// A chemical symbol, so the `species` column is well-formed.
fn species_strategy() -> impl Strategy<Value = String> {
    prop::sample::select(vec!["H", "C", "N", "O", "Si", "Fe"]).prop_map(str::to_owned)
}

/// Any finite f64; the round trip must reproduce its bits.
fn finite() -> impl Strategy<Value = f64> {
    any::<f64>().prop_filter("finite", |x| x.is_finite())
}

prop_compose! {
    /// A frame in write order, with metadata that survives re-parsing unchanged
    /// and no Lattice/pbc to reorder.
    fn frame_strategy()(
        n in 0usize..6,
    )(
        species in prop::collection::vec(species_strategy(), n..=n),
        pos in prop::collection::vec(finite(), (n * 3)..=(n * 3)),
        charge in prop::collection::vec(finite(), n..=n),
        tag in prop::collection::vec(any::<i64>(), n..=n),
        energy in finite(),
        count in any::<i64>(),
        flag in any::<bool>(),
        lattice in prop::collection::vec(finite(), 9..=9),
    ) -> Frame {
        Frame {
            n_atoms: species.len(),
            columns: vec![
                col("species", 1, ColumnData::Str(species.into_iter().map(Into::into).collect())),
                col("pos", 3, ColumnData::Real(pos)),
                col("charge", 1, ColumnData::Real(charge)),
                col("tag", 1, ColumnData::Int(tag)),
            ],
            metadata: vec![
                ("energy".into(), Value::Real(energy)),
                ("count".into(), Value::Int(count)),
                ("flag".into(), Value::Bool(flag)),
                ("box".into(), Value::RealArray(lattice)),
            ],
        }
    }
}

proptest! {
    #[test]
    fn write_then_reparse_is_lossless(frame in frame_strategy()) {
        let mut bytes = Vec::new();
        oxyz_core::write_frame(&mut bytes, &frame).unwrap();
        let reparsed = oxyz_core::FrameIter::new(std::io::Cursor::new(&bytes))
            .collect::<Result<Vec<_>, _>>()
            .expect("written frame must re-parse");
        prop_assert_eq!(reparsed.len(), 1);
        prop_assert_eq!(&reparsed[0], &frame);
    }

    /// Parallel serialisation reproduces the serial bytes for any frames and
    /// thread count.
    #[test]
    fn parallel_serialisation_matches_serial(
        frames in prop::collection::vec(frame_strategy(), 0..40),
        threads in prop::option::of(1usize..6),
    ) {
        let serial = temp_path("prop_serial.xyz");
        write_frames(&serial, &frames, Compression::Infer, None, false).unwrap();
        let serial_bytes = fs::read(&serial).unwrap();
        fs::remove_file(&serial).unwrap();

        let parallel = temp_path("prop_parallel.xyz");
        write_frames_parallel(&parallel, &frames, Compression::Infer, None, false, threads)
            .unwrap();
        let parallel_bytes = fs::read(&parallel).unwrap();
        fs::remove_file(&parallel).unwrap();

        prop_assert_eq!(parallel_bytes, serial_bytes);
    }
}
