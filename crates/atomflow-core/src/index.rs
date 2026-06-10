//! Byte-offset frame index: the result of a structural scan.
//!
//! An index records where each frame starts and how many atoms it declares —
//! nothing else is parsed. It is the foundation for random access, negative
//! indexing, batch preallocation, and (later) parallel parsing. Statistics
//! are derived from the stored entries on demand, never accumulated during
//! the scan. In-memory only; an on-disk cache is a separate, future feature.

/// One frame's structural facts: where it starts, what its count line says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameEntry {
    /// Byte offset of the frame's count line.
    pub offset: u64,
    pub n_atoms: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FrameIndex {
    entries: Vec<FrameEntry>,
}

impl FrameIndex {
    pub fn new(entries: Vec<FrameEntry>) -> Self {
        FrameIndex { entries }
    }

    pub fn n_frames(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn get(&self, frame: usize) -> Option<FrameEntry> {
        self.entries.get(frame).copied()
    }

    pub fn entries(&self) -> &[FrameEntry] {
        &self.entries
    }

    pub fn total_atoms(&self) -> usize {
        self.entries.iter().map(|entry| entry.n_atoms).sum()
    }

    pub fn min_atoms(&self) -> Option<usize> {
        self.entries.iter().map(|entry| entry.n_atoms).min()
    }

    pub fn max_atoms(&self) -> Option<usize> {
        self.entries.iter().map(|entry| entry.n_atoms).max()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index(counts: &[usize]) -> FrameIndex {
        FrameIndex::new(
            counts
                .iter()
                .map(|&n_atoms| FrameEntry { offset: 0, n_atoms })
                .collect(),
        )
    }

    #[test]
    fn aggregates_over_entries() {
        let idx = index(&[3, 1, 2]);
        assert_eq!(idx.n_frames(), 3);
        assert_eq!(idx.total_atoms(), 6);
        assert_eq!(idx.min_atoms(), Some(1));
        assert_eq!(idx.max_atoms(), Some(3));
    }

    #[test]
    fn empty_index_has_no_extremes() {
        let idx = FrameIndex::default();
        assert_eq!(idx.n_frames(), 0);
        assert_eq!(idx.total_atoms(), 0);
        assert_eq!(idx.min_atoms(), None);
        assert_eq!(idx.max_atoms(), None);
    }
}
