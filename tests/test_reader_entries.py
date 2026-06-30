"""The _rust *_reader entries, driven by in-memory Python sources (no S3)."""

from __future__ import annotations

import gzip
from pathlib import Path

import oxyz._rust as _rust

DATA = Path(__file__).parent / "data"


def chunks(blob: bytes, size: int = 7):
    """A bytes-iterator like obstore's GetResult.stream()."""
    for start in range(0, len(blob), size):
        yield blob[start : start + size]


def test_read_frames_reader_matches_path_plain():
    path = DATA / "minimal_periodic.extxyz"
    blob = path.read_bytes()
    via_reader = _rust.read_frames_reader(chunks(blob), "plain", None, None)
    via_path = _rust.read_frames(str(path), None, "infer", None)
    assert len(via_reader) == len(via_path)
    assert via_reader[0]["n_atoms"] == via_path[0]["n_atoms"]
    assert via_reader[0]["metadata"].keys() == via_path[0]["metadata"].keys()


def test_read_frames_reader_decodes_gzip():
    path = DATA / "minimal_periodic.extxyz"
    blob = gzip.compress(path.read_bytes())
    via_reader = _rust.read_frames_reader(chunks(blob), "gzip", None, None)
    via_path = _rust.read_frames(str(path), None, "infer", None)
    assert len(via_reader) == len(via_path)


def test_read_first_frame_reader_plain():
    path = DATA / "minimal_periodic.extxyz"
    frame = _rust.read_first_frame_reader(chunks(path.read_bytes()), "plain", None)
    assert frame["n_atoms"] > 0
