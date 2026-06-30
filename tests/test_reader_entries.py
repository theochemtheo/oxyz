"""The _rust *_reader entries, driven by in-memory Python sources (no S3)."""

from __future__ import annotations

import gzip
import io
import tarfile
import zipfile
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


def test_scan_reader_matches_path():
    path = DATA / "minimal_periodic.extxyz"
    blob = path.read_bytes()
    via_reader = _rust.scan_reader(chunks(blob), "plain", False, None)
    via_path = _rust.scan(str(path), False, "infer", None)
    assert list(via_reader["n_atoms"]) == list(via_path["n_atoms"])


def test_infer_schema_reader_matches_path():
    path = DATA / "minimal_periodic.extxyz"
    via_reader = _rust.infer_schema_reader(chunks(path.read_bytes()), "plain", None)
    via_path = _rust.infer_schema(str(path), "infer", None)
    assert via_reader["n_frames"] == via_path["n_frames"]
    assert via_reader["is_consistent"] == via_path["is_consistent"]


def test_read_batch_reader_all_frames():
    path = DATA / "minimal_periodic.extxyz"
    batch = _rust.read_batch_reader(
        chunks(path.read_bytes()), "plain", None, None, None
    )
    assert len(batch["offsets"]) >= 2


def test_frame_iter_from_reader_streams():
    path = DATA / "minimal_periodic.extxyz"
    frames = list(_rust.FrameIter.from_reader(chunks(path.read_bytes()), "plain", None))
    assert len(frames) == len(_rust.read_frames(str(path), None, "infer", None))


def test_batch_iter_from_reader_streams():
    path = DATA / "minimal_periodic.extxyz"
    batches = list(
        _rust.BatchIter.from_reader(chunks(path.read_bytes()), 1, "plain", None)
    )
    assert len(batches) >= 1


def test_read_frames_reader_tar_member():
    path = DATA / "minimal_periodic.extxyz"
    body = path.read_bytes()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo("inner.xyz")
        info.size = len(body)
        tar.addfile(info, io.BytesIO(body))
    blob = buf.getvalue()

    def factory():  # a fresh bytes-iterator per call (tar reads twice)
        return chunks(blob)

    frames = _rust.read_frames_reader(factory, "tar", None, None)
    assert len(frames) == len(_rust.read_frames(str(path), None, "infer", None))


def test_read_frames_reader_zip_member():
    path = DATA / "minimal_periodic.extxyz"
    body = path.read_bytes()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("inner.xyz", body)
    seekable = io.BytesIO(buf.getvalue())  # has read/seek/tell

    frames = _rust.read_frames_reader(seekable, "zip", None, None)
    assert len(frames) == len(_rust.read_frames(str(path), None, "infer", None))


def test_detect_codec():
    assert _rust.detect_codec("train.xyz", None) == "plain"
    assert _rust.detect_codec("train.xyz.gz", None) == "gzip"
    assert _rust.detect_codec("a.tar.gz", None) == "tar.gz"
    assert _rust.detect_codec("a.zip", None) == "zip"
    assert _rust.detect_codec("blob", b"PK\x03\x04") == "zip"
    assert _rust.detect_codec("blob", b"hello") == "plain"
