//! Byte-offset frame index: the result of a structural scan.

/// One frame's structural facts: where it starts, what its count line says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameEntry {
    /// Byte offset of the frame's count line.
    pub offset: u64,
    /// 1-based file line of the count line, for seek-based diagnostics.
    pub line: usize,
    pub n_atoms: usize,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct FrameIndex {
    entries: Vec<FrameEntry>,
    /// Per-frame `|det(Lattice)|` when the scan measured it; `NaN` for a frame
    /// with no `Lattice`. Kept outside `FrameEntry` so that stays `Copy + Eq`.
    volumes: Option<Vec<f64>>,
}

impl FrameIndex {
    pub fn new(entries: Vec<FrameEntry>) -> Self {
        FrameIndex {
            entries,
            volumes: None,
        }
    }

    pub fn with_volumes(entries: Vec<FrameEntry>, volumes: Vec<f64>) -> Self {
        debug_assert_eq!(entries.len(), volumes.len());
        FrameIndex {
            entries,
            volumes: Some(volumes),
        }
    }

    pub fn volumes(&self) -> Option<&[f64]> {
        self.volumes.as_deref()
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
                .map(|&n_atoms| FrameEntry {
                    offset: 0,
                    line: 1,
                    n_atoms,
                })
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
