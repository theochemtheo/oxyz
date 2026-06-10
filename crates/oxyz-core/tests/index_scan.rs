use std::{io::Cursor, path::PathBuf};

use atomflow_core::{IndexedFrames, read_frames, scan_frames, scan_index};

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

/// The scan and the parser must agree on every corpus file: same frame
/// count, same atom counts, and `get(i)` returns the identical `Frame`.
#[test]
fn indexed_reads_match_streamed_reads_across_corpus() {
    for path in corpus() {
        let streamed = read_frames(&path).unwrap();
        let mut indexed = IndexedFrames::open(&path).unwrap();

        assert_eq!(indexed.len(), streamed.len(), "{path:?}");
        let scanned_counts: Vec<usize> = indexed
            .index()
            .entries()
            .iter()
            .map(|entry| entry.n_atoms)
            .collect();
        let parsed_counts: Vec<usize> = streamed.iter().map(|frame| frame.n_atoms).collect();
        assert_eq!(scanned_counts, parsed_counts, "{path:?}");

        // Read back-to-front to prove order independence.
        for frame_index in (0..streamed.len()).rev() {
            assert_eq!(
                indexed.get(frame_index).unwrap(),
                streamed[frame_index],
                "{path:?} frame {frame_index}"
            );
        }
    }
}

#[test]
fn scan_records_offsets_and_counts() {
    let index = scan_index(fixture("varying_atom_counts.xyz")).unwrap();
    let text = std::fs::read_to_string(fixture("varying_atom_counts.xyz")).unwrap();

    let counts: Vec<usize> = index.entries().iter().map(|entry| entry.n_atoms).collect();
    assert_eq!(counts, [3, 1, 2]);
    assert_eq!(index.total_atoms(), 6);

    // Each recorded offset must point at the frame's count line.
    for entry in index.entries() {
        let at_offset = &text[entry.offset as usize..];
        let count_line = at_offset.lines().next().unwrap();
        assert_eq!(count_line.trim().parse::<usize>().unwrap(), entry.n_atoms);
    }
}

#[test]
fn lying_count_surfaces_one_frame_late() {
    // Frame 0 declares 3 atoms but has 2; the scan desyncs and trips on the
    // next count line, blaming frame 1 — the documented trust tradeoff.
    let text = "3\ncomment\nH 0 0 0\nH 1 1 1\n1\ncomment\nH 2 2 2\n";
    let error = scan_frames(Cursor::new(text)).unwrap_err();
    assert!(error.to_string().contains("frame 1"), "{error}");
    assert!(error.to_string().contains("invalid atom count"), "{error}");
}

#[test]
fn truncated_final_frame_is_an_error() {
    let text = "2\ncomment\nH 0 0 0\n";
    let error = scan_frames(Cursor::new(text)).unwrap_err();
    assert!(error.to_string().contains("frame 0"), "{error}");
    assert!(error.to_string().contains("missing atom line"), "{error}");
}

#[test]
fn empty_input_scans_to_empty_index() {
    let index = scan_frames(Cursor::new("")).unwrap();
    assert_eq!(index.n_frames(), 0);
}

#[test]
fn out_of_range_get_reports_bounds() {
    let mut indexed = IndexedFrames::open(fixture("varying_atom_counts.xyz")).unwrap();
    let error = indexed.get(3).unwrap_err();
    assert_eq!(
        error.to_string(),
        "frame index 3 out of range: file has 3 frames"
    );
}
