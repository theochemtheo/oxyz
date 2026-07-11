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


@pytest.mark.parametrize(
    ("key", "body"),
    [
        ("train.extxyz", KEY_DATA),
        ("train.extxyz.gz", gzip.compress(KEY_DATA)),
    ],
)
def test_read_frames_remote_matches_local(s3_store, key, body):
    s3_store.put(key, body)
    remote = oxyz.read(s3_store.url(key), storage_options=s3_store.options)
    local = oxyz.read(str(DATA / "minimal_periodic.extxyz"))
    assert len(remote) == len(local)
    assert remote[0].n_atoms == local[0].n_atoms


def test_read_frames_remote_zip(s3_store):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("inner.xyz", KEY_DATA)
    s3_store.put("train.zip", buf.getvalue())
    remote = oxyz.read(s3_store.url("train.zip"), storage_options=s3_store.options)
    local = oxyz.read(str(DATA / "minimal_periodic.extxyz"))
    assert len(remote) == len(local)


def test_read_frames_remote_targz_member(s3_store):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in ("a.xyz", "b.xyz"):
            info = tarfile.TarInfo(name)
            info.size = len(KEY_DATA)
            tar.addfile(info, io.BytesIO(KEY_DATA))
    s3_store.put("train.tar.gz", buf.getvalue())
    remote = oxyz.read(
        s3_store.url("train.tar.gz"), member="b.xyz", storage_options=s3_store.options
    )
    local = oxyz.read(str(DATA / "minimal_periodic.extxyz"))
    assert len(remote) == len(local)


def test_iter_scan_schema_remote(s3_store):
    s3_store.put("train.extxyz", KEY_DATA)
    url = s3_store.url("train.extxyz")
    options = s3_store.options
    assert sum(1 for _ in oxyz.iread(url, storage_options=options)) > 0
    assert oxyz.scan(url, storage_options=options).n_frames > 0
    assert oxyz.infer_schema(url, storage_options=options).n_frames > 0
    assert oxyz.read_batch(url, storage_options=options).n_frames > 0


def test_ase_read_remote(s3_store):
    pytest.importorskip("ase")
    import oxyz.ase

    s3_store.put("train.extxyz", KEY_DATA)
    url = s3_store.url("train.extxyz")
    options = s3_store.options
    assert len(oxyz.ase.read(url, index=0, storage_options=options)) > 0
    assert len(oxyz.ase.read(url, index=-1, storage_options=options)) > 0


def test_missing_object_raises(s3_store):
    # obstore surfaces a FileNotFoundError (404 Not Found from the store).
    with pytest.raises(FileNotFoundError):
        oxyz.read(s3_store.url("nope.xyz"), storage_options=s3_store.options)


# A mixed-schema body: frame 0 carries 'charge', frame 1 does not — the case
# projection reshapes into a uniform, batchable set (charge filled with NaN).
MIXED_BODY = (
    b"1\nProperties=species:S:1:pos:R:3:charge:R:1\nH 0 0 0 0.5\n"
    b"1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n"
)


def _project_spec():
    from oxyz._schema import Kind
    from oxyz._schema_spec import ColumnRule, SchemaSpec

    return SchemaSpec(
        columns=(
            ColumnRule("species", Kind.STR),
            ColumnRule("pos", Kind.REAL, width=3),
            ColumnRule("charge", Kind.REAL, required=False),
        ),
        mode="project",
    )


def test_read_frames_projected_remote(s3_store):
    """The reader-source projected frame path: read_frames_projected_reader."""
    import math

    import numpy as np

    s3_store.put("mixed.extxyz", MIXED_BODY)
    spec = _project_spec()
    frames = oxyz.read(
        s3_store.url("mixed.extxyz"), schema=spec, storage_options=s3_store.options
    )
    assert [set(fr.columns) for fr in frames] == [
        {"species", "pos", "charge"},
        {"species", "pos", "charge"},
    ]
    assert math.isnan(np.asarray(frames[1].columns["charge"])[0])  # filled


def test_read_first_and_iter_projected_remote(s3_store):
    """read_first_frame_projected_reader and FrameIterProjected.from_reader."""
    s3_store.put("mixed.extxyz", MIXED_BODY)
    url = s3_store.url("mixed.extxyz")
    spec = _project_spec()
    first = oxyz.read(url, 0, schema=spec, storage_options=s3_store.options)
    assert "charge" in first.columns
    streamed = list(oxyz.iread(url, schema=spec, storage_options=s3_store.options))
    assert len(streamed) == 2
    assert all("charge" in fr.columns for fr in streamed)


def test_read_batch_projected_remote(s3_store):
    """The reader-source projected batch path: read_batch_projected_reader."""
    s3_store.put("mixed.extxyz", MIXED_BODY)
    spec = _project_spec()
    batch = oxyz.read_batch(
        s3_store.url("mixed.extxyz"), schema=spec, storage_options=s3_store.options
    )
    assert batch.n_frames == 2
    assert "charge" in batch.columns  # mixed file made batchable by projection


def test_iread_batch_projected_remote(s3_store):
    """Streaming projected batches over a reader: BatchIterProjected.from_reader."""
    s3_store.put("mixed.extxyz", MIXED_BODY)
    spec = _project_spec()
    batches = list(
        oxyz.iread_batch(
            s3_store.url("mixed.extxyz"),
            frames_per_batch=1,
            schema=spec,
            storage_options=s3_store.options,
        )
    )
    assert len(batches) == 2
    assert all("charge" in b.columns for b in batches)


def test_ase_read_projected_remote(s3_store):
    """Output-target projection over a remote source (nth_frame + slice paths)."""
    pytest.importorskip("ase")
    import oxyz.ase

    s3_store.put("mixed.extxyz", MIXED_BODY)
    url = s3_store.url("mixed.extxyz")
    spec = _project_spec()
    options = s3_store.options
    last = oxyz.ase.read(url, index=-1, schema=spec, storage_options=options)
    assert len(last) == 1  # one-atom frame, projected then converted
    both = oxyz.ase.read(url, index=":", schema=spec, storage_options=options)
    assert len(both) == 2
