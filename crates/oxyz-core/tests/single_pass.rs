//! Single-pass reads must match the two-pass path on valid input, and the
//! partial-read promise (nothing past the last requested frame is
//! inspected) must hold where they deliberately differ.

use std::path::PathBuf;

use oxyz_core::{IndexedFrames, read_batch, read_frames};
#[cfg(feature = "parallel")]
use oxyz_core::{read_batch_parallel, read_frames_parallel};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/data")
        .join(name)
}

fn write_temp(name: &str, text: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("oxyz_sp_{}_{name}", std::process::id()));
    std::fs::write(&path, text).unwrap();
    path
}

/// `n` one-atom frames, each distinguishable by its `i` metadata and z.
fn simple_frames(n: usize) -> String {
    (0..n)
        .map(|i| format!("1\nProperties=species:S:1:pos:R:3 i={i}\nH 0 0 {i}\n"))
        .collect()
}

#[test]
fn read_batch_matches_the_two_pass_gather() {
    let path = fixture("varying_atom_counts.xyz");
    let mut indexed = IndexedFrames::open(&path).unwrap();
    let n = indexed.len();

    let requests: Vec<Vec<usize>> = vec![
        vec![0],
        vec![n - 1, 0],
        (0..n).rev().collect(),
        vec![1, 1, 0, 1],
    ];
    for request in requests {
        let expected = indexed.get_batch(&request).unwrap();
        assert_eq!(
            read_batch(&path, &request).unwrap(),
            expected,
            "{request:?}"
        );
        #[cfg(feature = "parallel")]
        for threads in [Some(4), None] {
            assert_eq!(
                read_batch_parallel(&path, &request, threads).unwrap(),
                expected,
                "{request:?} threads={threads:?}"
            );
        }
    }
}

#[test]
fn read_batch_never_reads_past_the_last_requested_frame() {
    // Three good frames, then bytes that fail even a structural scan.
    let text = format!("{}garbage\n", simple_frames(3));
    let path = write_temp("tail.xyz", &text);

    // Whole-file readers must still reject the file outright...
    assert!(IndexedFrames::open(&path).is_err());
    assert!(read_frames(&path).is_err());

    // ...but a partial request stops scanning at frame 2 and succeeds.
    let batch = read_batch(&path, &[0, 2]).unwrap();
    assert_eq!(batch.n_frames(), 2);
    #[cfg(feature = "parallel")]
    assert_eq!(read_batch_parallel(&path, &[0, 2], Some(4)).unwrap(), batch);

    std::fs::remove_file(path).ok();
}

#[test]
fn out_of_range_reports_the_file_frame_count() {
    let path = write_temp("oob.xyz", &simple_frames(3));

    let error = read_batch(&path, &[1, 5]).unwrap_err();
    assert_eq!(
        error.to_string(),
        "frame index 5 out of range: file has 3 frames"
    );
    #[cfg(feature = "parallel")]
    assert_eq!(
        read_batch_parallel(&path, &[1, 5], Some(2))
            .unwrap_err()
            .to_string(),
        error.to_string()
    );

    std::fs::remove_file(path).ok();
}

#[test]
fn batch_errors_resolve_in_request_order() {
    // Frame 1's atom line breaks its declared column count.
    let text = simple_frames(3).replace("H 0 0 1\n", "H 0 0 1 99\n");
    let path = write_temp("order.xyz", &text);

    // Out of range at request position 0 beats the parse error in frame 1...
    let error = read_batch(&path, &[9, 1]).unwrap_err();
    assert!(error.to_string().contains("out of range"), "{error}");

    // ...and the parse error wins when frame 1 is requested first.
    let error = read_batch(&path, &[1, 9]).unwrap_err();
    assert!(error.to_string().contains("frame 1"), "{error}");

    #[cfg(feature = "parallel")]
    {
        let error = read_batch_parallel(&path, &[9, 1], Some(4)).unwrap_err();
        assert!(error.to_string().contains("out of range"), "{error}");
        let error = read_batch_parallel(&path, &[1, 9], Some(4)).unwrap_err();
        assert!(error.to_string().contains("frame 1"), "{error}");
    }

    std::fs::remove_file(path).ok();
}

#[test]
fn structural_damage_in_the_needed_prefix_wins() {
    // Frame 1 has a content error; frame 2's count line is structural junk.
    // Requesting frame 3 forces the scan through the damage, and the
    // structural error must win even though frame 1 sits earlier in the
    // request — without trustworthy structure, frame 3 cannot be located.
    let frames = simple_frames(2).replace("H 0 0 1\n", "H 0 0 1 99\n");
    let text = format!("{frames}not-a-count\n");
    let path = write_temp("structural.xyz", &text);

    let error = read_batch(&path, &[1, 3]).unwrap_err();
    assert!(error.to_string().contains("invalid atom count"), "{error}");
    assert!(error.to_string().contains("frame 2"), "{error}");
    #[cfg(feature = "parallel")]
    assert_eq!(
        read_batch_parallel(&path, &[1, 3], Some(2))
            .unwrap_err()
            .to_string(),
        error.to_string()
    );

    std::fs::remove_file(path).ok();
}

/// Serial and parallel full reads must agree on which error is reported:
/// the first in frame order, even when a content error (frame 1) races a
/// structural error further on (frame 3). The two-pass path got this wrong
/// — its up-front scan reported frame 3.
#[cfg(feature = "parallel")]
#[test]
fn full_read_reports_the_first_error_in_frame_order() {
    let frames = simple_frames(3).replace("H 0 0 1\n", "H 0 0 1 99\n");
    let text = format!("{frames}junk\n");
    let path = write_temp("precedence.xyz", &text);

    let serial = read_frames(&path).unwrap_err().to_string();
    assert!(serial.contains("frame 1"), "{serial}");
    for threads in [Some(2), Some(4), None] {
        let parallel = read_frames_parallel(&path, threads)
            .unwrap_err()
            .to_string();
        assert_eq!(parallel, serial, "threads={threads:?}");
    }

    std::fs::remove_file(path).ok();
}

/// The two-pass path raised "batch is empty" here; an empty file is no
/// frames, exactly as the serial read says.
#[cfg(feature = "parallel")]
#[test]
fn empty_file_is_no_frames_at_every_thread_count() {
    let path = write_temp("empty.xyz", "");

    assert!(read_frames(&path).unwrap().is_empty());
    assert!(read_frames_parallel(&path, Some(2)).unwrap().is_empty());
    assert!(read_frames_parallel(&path, None).unwrap().is_empty());

    std::fs::remove_file(path).ok();
}

/// A bad count line must raise the identical error from a streamed parse
/// and a structural scan; the message is the trimmed line either way.
#[test]
fn count_line_errors_match_between_scan_and_parse() {
    let path = write_temp("badcount.xyz", "  abc  \n");

    let streamed = read_frames(&path).unwrap_err().to_string();
    let scanned = oxyz_core::scan_index(&path).unwrap_err().to_string();
    assert_eq!(streamed, scanned);

    std::fs::remove_file(path).ok();
}
