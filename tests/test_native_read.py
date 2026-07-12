"""The unified native reader surface: `oxyz.read` / `oxyz.iread` with an index.

`read` materialises, `iread` streams; the index selects (int -> one Frame,
slice/str/sequence -> a list). Replaces the old `read_frames`/`iter_frames`/
`read_first`/`read_frames_sliced` split.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

import oxyz
from oxyz import Frame

DATA = Path(__file__).parent / "data"
MULTI = DATA / "varying_atom_counts.xyz"  # 3 frames


def test_new_names_are_public_and_old_ones_gone() -> None:
    assert "read" in oxyz.__all__
    assert "iread" in oxyz.__all__
    for gone in ("read_frames", "iter_frames", "read_first", "read_frames_sliced"):
        assert gone not in oxyz.__all__, gone
        assert not hasattr(oxyz, gone), gone


def test_read_default_is_every_frame() -> None:
    frames = oxyz.read(MULTI)
    assert isinstance(frames, list)
    assert len(frames) == 3
    assert all(isinstance(f, Frame) for f in frames)


def test_read_int_index_returns_one_frame() -> None:
    every = oxyz.read(MULTI)
    first = oxyz.read(MULTI, 0)
    assert isinstance(first, Frame)
    assert first.n_atoms == every[0].n_atoms
    last = oxyz.read(MULTI, -1)
    assert isinstance(last, Frame)
    assert last.n_atoms == every[-1].n_atoms


def test_read_slice_and_str_return_lists() -> None:
    every = oxyz.read(MULTI)
    by_slice = oxyz.read(MULTI, slice(0, 2))
    assert isinstance(by_slice, list)
    assert [f.n_atoms for f in by_slice] == [every[0].n_atoms, every[1].n_atoms]
    by_str = oxyz.read(MULTI, "1:3")
    assert isinstance(by_str, list)  # a slice string returns a list
    assert [f.n_atoms for f in by_str] == [every[1].n_atoms, every[2].n_atoms]


def test_read_sequence_preserves_order_and_repeats() -> None:
    every = oxyz.read(MULTI)
    picked = oxyz.read(MULTI, [2, 0, 2])
    assert isinstance(picked, list)
    assert [f.n_atoms for f in picked] == [
        every[2].n_atoms,
        every[0].n_atoms,
        every[2].n_atoms,
    ]


def test_read_threads_agree() -> None:
    serial = oxyz.read(MULTI, threads=1)
    parallel = oxyz.read(MULTI, threads=4)
    assert [f.n_atoms for f in serial] == [f.n_atoms for f in parallel]


def test_iread_streams_every_frame_by_default() -> None:
    stream = oxyz.iread(MULTI)
    assert isinstance(stream, Iterator)
    assert len([f for f in stream]) == 3


def test_iread_index_selects() -> None:
    by_str = oxyz.read(MULTI, "0:2")
    assert isinstance(by_str, list)
    assert [f.n_atoms for f in oxyz.iread(MULTI, "0:2")] == [f.n_atoms for f in by_str]
    single = list(oxyz.iread(MULTI, 1))
    assert len(single) == 1
    assert single[0].n_atoms == oxyz.read(MULTI, 1).n_atoms


def test_read_out_of_range_raises() -> None:
    with pytest.raises(IndexError):
        oxyz.read(MULTI, 99)
