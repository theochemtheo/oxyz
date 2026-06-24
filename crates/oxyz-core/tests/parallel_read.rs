#![cfg(feature = "parallel")]

use std::path::PathBuf;

use oxyz_core::{IndexedFrames, read_frames, read_frames_parallel};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/data")
        .join(name)
}

fn corpus() -> Vec<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            matches!(
                path.extension().and_then(|ext| ext.to_str()),
                Some("xyz" | "extxyz")
            )
        })
        .collect();
    paths.sort();
    paths
}

fn write_temp(name: &str, text: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("oxyz_par_{}_{name}", std::process::id()));
    std::fs::write(&path, text).unwrap();
    path
}

/// A frame whose bytes exceed the intra-frame threshold (~2 MiB) so its atom
/// rows are split across workers. `bad_row`, if set, gets an extra column.
fn large_frame_text(n_frames: usize, n_atoms: usize, bad_row: Option<usize>) -> String {
    let mut text = String::new();
    for _ in 0..n_frames {
        text.push_str(&format!("{n_atoms}\n"));
        text.push_str(
            "Lattice=\"10 0 0 0 10 0 0 0 10\" \
             Properties=species:S:1:pos:R:3:forces:R:3 energy=-1.5\n",
        );
        for row in 0..n_atoms {
            if bad_row == Some(row) {
                text.push_str("Si 0.1 0.2 0.3 0.0 0.0 0.0 EXTRA\n");
            } else {
                text.push_str("Si 0.1 0.2 0.3 0.0 0.0 0.0\n");
            }
        }
    }
    text
}

/// ~2.6 MiB per frame engages the intra-frame parallel parser; the split and
/// concatenation must reproduce the serial frame exactly, at every thread
/// count and across a frame boundary.
#[test]
fn large_frame_parallel_matches_serial() {
    let path = write_temp("large_frame.extxyz", &large_frame_text(2, 50_000, None));

    let serial = read_frames(&path).unwrap();
    assert_eq!(serial.len(), 2);
    assert_eq!(serial[0].n_atoms, 50_000);
    for threads in [Some(1), Some(2), Some(8), None] {
        let parallel = read_frames_parallel(&path, threads).unwrap();
        assert_eq!(parallel, serial, "threads={threads:?}");
    }

    std::fs::remove_file(path).ok();
}

/// A malformed row deep inside a large frame must raise the identical error —
/// same message and file-absolute line number — as a serial read, regardless
/// of which worker's range it landed in.
#[test]
fn large_frame_error_matches_serial() {
    let path = write_temp(
        "large_frame_bad.extxyz",
        &large_frame_text(1, 50_000, Some(40_000)),
    );

    let serial = read_frames(&path).unwrap_err().to_string();
    assert!(serial.contains("atom line"), "{serial}");
    for threads in [Some(1), Some(2), Some(8), None] {
        let parallel = read_frames_parallel(&path, threads)
            .unwrap_err()
            .to_string();
        assert_eq!(parallel, serial, "threads={threads:?}");
    }

    std::fs::remove_file(path).ok();
}

/// The invariant: parallel reads are observably identical to serial ones.
#[test]
fn parallel_reads_match_serial_across_corpus() {
    for path in corpus() {
        let serial = read_frames(&path).unwrap();
        for threads in [Some(1), Some(4), None] {
            let parallel = read_frames_parallel(&path, threads).unwrap();
            assert_eq!(parallel, serial, "{path:?} threads={threads:?}");
        }
    }
}

#[test]
fn parallel_get_batch_matches_serial() {
    let path = fixture("varying_atom_counts.xyz");
    let mut indexed = IndexedFrames::open(&path).unwrap();

    let serial = indexed.get_batch(&[2, 0, 2]).unwrap();
    let parallel = indexed.get_batch_parallel(&[2, 0, 2], Some(4)).unwrap();
    assert_eq!(parallel, serial);
}

/// With several broken frames racing, the reported error must be the first
/// in request order — the same one a serial read raises.
#[test]
fn first_error_in_request_order_wins() {
    let mut text = String::new();
    for frame in 0..32 {
        // Frames 9 and 27 declare a 4-column row but provide 5 values.
        let row = if frame == 9 || frame == 27 {
            "H 0 0 0 99"
        } else {
            "H 0 0 0"
        };
        text.push_str(&format!(
            "1\nProperties=species:S:1:pos:R:3 i={frame}\n{row}\n"
        ));
    }
    let path = write_temp("first_error.xyz", &text);

    let serial_error = read_frames(&path).unwrap_err().to_string();
    assert!(serial_error.contains("frame 9"), "{serial_error}");

    for _ in 0..8 {
        let parallel_error = read_frames_parallel(&path, Some(4))
            .unwrap_err()
            .to_string();
        assert_eq!(parallel_error, serial_error);
    }

    std::fs::remove_file(path).ok();
}

#[test]
fn empty_selection_is_an_empty_batch_error() {
    let mut indexed = IndexedFrames::open(fixture("varying_atom_counts.xyz")).unwrap();
    let error = indexed.get_batch_parallel(&[], Some(2)).unwrap_err();
    assert_eq!(error.to_string(), "batch is empty");
}
