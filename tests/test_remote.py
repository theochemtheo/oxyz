"""End-to-end remote reads against a local moto S3 server.

All tests skip cleanly when moto/boto3/obstore are absent, so the base install
(no s3 extra) continues to pass CI without a running object store.
"""

from __future__ import annotations

import gzip
import io
import tarfile
import zipfile
from pathlib import Path

import pytest

import oxyz

DATA = Path(__file__).parent / "data"
KEY_DATA = (DATA / "minimal_periodic.extxyz").read_bytes()


def _url(key: str) -> str:
    return f"s3://test/{key}"


@pytest.mark.parametrize(
    ("key", "body"),
    [
        ("train.extxyz", KEY_DATA),
        ("train.extxyz.gz", gzip.compress(KEY_DATA)),
    ],
)
def test_read_frames_remote_matches_local(s3_store, key, body):
    put, options = s3_store
    put(key, body)
    remote = oxyz.read_frames(_url(key), storage_options=options)
    local = oxyz.read_frames(str(DATA / "minimal_periodic.extxyz"))
    assert len(remote) == len(local)
    assert remote[0].n_atoms == local[0].n_atoms


def test_read_frames_remote_zip(s3_store):
    put, options = s3_store
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("inner.xyz", KEY_DATA)
    put("train.zip", buf.getvalue())
    remote = oxyz.read_frames(_url("train.zip"), storage_options=options)
    local = oxyz.read_frames(str(DATA / "minimal_periodic.extxyz"))
    assert len(remote) == len(local)


def test_read_frames_remote_targz_member(s3_store):
    put, options = s3_store
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in ("a.xyz", "b.xyz"):
            info = tarfile.TarInfo(name)
            info.size = len(KEY_DATA)
            tar.addfile(info, io.BytesIO(KEY_DATA))
    put("train.tar.gz", buf.getvalue())
    remote = oxyz.read_frames(
        _url("train.tar.gz"), member="b.xyz", storage_options=options
    )
    local = oxyz.read_frames(str(DATA / "minimal_periodic.extxyz"))
    assert len(remote) == len(local)


def test_iter_scan_schema_remote(s3_store):
    put, options = s3_store
    put("train.extxyz", KEY_DATA)
    url = _url("train.extxyz")
    assert sum(1 for _ in oxyz.iter_frames(url, storage_options=options)) > 0
    assert oxyz.scan(url, storage_options=options).n_frames > 0
    assert oxyz.infer_schema(url, storage_options=options).n_frames > 0
    assert oxyz.read_batch(url, storage_options=options).n_frames > 0


def test_ase_read_remote(s3_store):
    pytest.importorskip("ase")
    import oxyz.ase

    put, options = s3_store
    put("train.extxyz", KEY_DATA)
    url = _url("train.extxyz")
    assert len(oxyz.ase.read(url, index=0, storage_options=options)) > 0
    assert len(oxyz.ase.read(url, index=-1, storage_options=options)) > 0


def test_missing_object_raises(s3_store):
    _put, options = s3_store
    # obstore surfaces a FileNotFoundError (404 Not Found from the store).
    with pytest.raises(FileNotFoundError):
        oxyz.read_frames(_url("nope.xyz"), storage_options=options)
