from __future__ import annotations

from pathlib import Path

import pytest

import oxyz
from oxyz import _remote


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("s3://bucket/train.xyz", True),
        ("gs://bucket/train.xyz", True),
        ("az://acct/container/train.xyz", True),
        ("/local/train.xyz", False),
        ("train.xyz", False),
        ("file:///local/train.xyz", False),  # local file URL is not "remote" here
    ],
)
def test_is_remote(path, expected):
    assert _remote.is_remote(path) is expected


def test_missing_obstore_raises_helpful_error(monkeypatch):
    monkeypatch.setattr(_remote, "_import_obstore", _remote._raise_missing)
    with pytest.raises(ImportError, match=r"oxyz\[s3\]"):
        _remote.open_source(
            "s3://bucket/train.xyz",
            compression="infer",
            member=None,
            storage_options=None,
        )


def test_read_frames_routes_remote(monkeypatch):
    path = Path("tests/data/minimal_periodic.extxyz")
    blob = path.read_bytes()

    def fake_open_source(p, *, compression, member, storage_options):
        from oxyz._remote import RemoteSource

        def chunks():
            yield blob

        return RemoteSource(obj=chunks(), codec="plain", member=None)

    monkeypatch.setattr(oxyz._remote, "is_remote", lambda p: True)
    monkeypatch.setattr(oxyz._remote, "open_source", fake_open_source)

    remote = oxyz.read_frames("s3://bucket/minimal_periodic.extxyz")
    local = oxyz.read_frames(str(path))
    assert len(remote) == len(local)
    assert remote[0].n_atoms == local[0].n_atoms
