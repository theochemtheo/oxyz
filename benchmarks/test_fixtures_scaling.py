"""The sweep fixtures generate the right shapes and cache deterministically."""

from __future__ import annotations

import numpy as np

import conftest
import oxyz


def test_dataset_fixture_matches_corpus_shape():
    counts = np.array(conftest.dataset_frame_atoms(20_000))
    assert counts.min() >= 1
    assert counts.max() <= 240
    # Corpus: mean 6.9, p50 6, p90 12. Allow generous tolerance — this is a
    # statistical stand-in, not a replica.
    assert 5.0 <= counts.mean() <= 9.0
    assert np.percentile(counts, 50) <= 8
    assert np.percentile(counts, 90) <= 20


def test_dataset_fixture_scans_to_requested_frames():
    path = conftest.sweep_dataset_size_file(1_000)
    index = oxyz.scan(path)
    assert index.n_frames == 1_000
    assert index.total_atoms == sum(conftest.dataset_frame_atoms(1_000))


def test_system_fixture_has_expected_atoms():
    path = conftest.sweep_system_size_file(1_000)
    index = oxyz.scan(path)
    assert index.n_frames == conftest.SYSTEM_SIZE_FRAMES
    assert index.total_atoms == conftest.SYSTEM_SIZE_FRAMES * 1_000
