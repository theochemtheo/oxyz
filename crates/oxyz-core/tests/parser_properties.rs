//! Property tests: the parser must never panic, whatever the input.
//! Malformed input must surface as `Err`, not as a crash or an absurd
//! allocation. Both the streaming parser and the structural scan are driven,
//! since they read count lines independently.

use std::io::Cursor;

use oxyz_core::{FrameIter, scan_frames};
use proptest::prelude::*;

fn parse_all(input: &str) {
    for result in FrameIter::new(Cursor::new(input.as_bytes())) {
        let _ = result;
    }
}

fn scan_all(input: &str) {
    let _ = scan_frames(Cursor::new(input.as_bytes()));
}

proptest! {
    #[test]
    fn never_panics_on_arbitrary_input(input in ".*") {
        parse_all(&input);
        scan_all(&input);
    }

    /// The declared atom count is untrusted: huge values (up to usize::MAX,
    /// the `n_atoms + 1` overflow site) must not panic or pre-allocate
    /// proportionally, in either the parser or the scan.
    #[test]
    fn never_panics_on_declared_atom_counts(count in any::<u64>(), body in "[ -~\n]{0,200}") {
        let input = format!("{count}\nProperties=species:S:1:pos:R:3\n{body}");
        parse_all(&input);
        scan_all(&input);
    }

    /// Arbitrary Properties descriptors, including huge declared widths.
    #[test]
    fn never_panics_on_arbitrary_descriptors(
        descriptor in "[A-Za-z0-9:._\\-]{0,64}",
        count in 0usize..4,
    ) {
        let mut input = format!("{count}\nProperties={descriptor}\n");
        for _ in 0..count {
            input.push_str("H 0.0 0.0 0.0\n");
        }
        parse_all(&input);
        scan_all(&input);
    }
}
