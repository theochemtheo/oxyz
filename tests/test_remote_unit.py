from __future__ import annotations

import sys
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

    remote = oxyz.read("s3://bucket/minimal_periodic.extxyz")
    local = oxyz.read(str(path))
    assert len(remote) == len(local)
    assert remote[0].n_atoms == local[0].n_atoms


def test_scan_and_schema_route_remote(monkeypatch):
    path = Path("tests/data/minimal_periodic.extxyz")
    blob = path.read_bytes()

    # Baselines computed before patching so the local reads take the normal path.
    local_idx = oxyz.scan(str(path))
    local_sch = oxyz.infer_schema(str(path))

    def fake_open_source(p, *, compression, member, storage_options):
        from oxyz._remote import RemoteSource

        return RemoteSource(obj=iter([blob]), codec="plain", member=None)

    monkeypatch.setattr(oxyz._remote, "is_remote", lambda p: True)
    monkeypatch.setattr(oxyz._remote, "open_source", fake_open_source)

    idx = oxyz.scan("s3://bucket/minimal_periodic.extxyz")
    assert idx.n_frames == local_idx.n_frames
    sch = oxyz.infer_schema("s3://bucket/minimal_periodic.extxyz")
    assert sch.is_consistent == local_sch.is_consistent


def test_read_batch_routes_remote(monkeypatch):
    path = Path("tests/data/minimal_periodic.extxyz")
    blob = path.read_bytes()

    local_batch = oxyz.read_batch(str(path))

    def fake_open_source(p, *, compression, member, storage_options):
        from oxyz._remote import RemoteSource

        return RemoteSource(obj=iter([blob]), codec="plain", member=None)

    monkeypatch.setattr(oxyz._remote, "is_remote", lambda p: True)
    monkeypatch.setattr(oxyz._remote, "open_source", fake_open_source)

    remote_batch = oxyz.read_batch("s3://bucket/minimal_periodic.extxyz")
    assert remote_batch.n_frames == local_batch.n_frames


def test_iter_batches_streams_remote(monkeypatch):
    path = Path("tests/data/minimal_periodic.extxyz")
    blob = path.read_bytes()

    local_frames = sum(
        b.n_frames for b in oxyz.iter_batches(str(path), frames_per_batch=1)
    )

    def fake_open_source(p, *, compression, member, storage_options):
        from oxyz._remote import RemoteSource

        return RemoteSource(obj=iter([blob]), codec="plain", member=None)

    monkeypatch.setattr(oxyz._remote, "is_remote", lambda p: True)
    monkeypatch.setattr(oxyz._remote, "open_source", fake_open_source)

    batches = list(
        oxyz.iter_batches("s3://bucket/minimal_periodic.extxyz", frames_per_batch=1)
    )
    assert sum(b.n_frames for b in batches) == local_frames


def test_iter_batches_remote_rejects_random_access(monkeypatch):
    monkeypatch.setattr(oxyz._remote, "is_remote", lambda p: True)
    with pytest.raises(ValueError, match="randomly accessed"):
        list(oxyz.iter_batches("s3://bucket/x.xyz", frames_per_batch=2, shuffle=True))


def test_ase_read_routes_remote(monkeypatch):
    pytest.importorskip("ase")
    import oxyz.ase

    path = Path("tests/data/minimal_periodic.extxyz")
    blob = path.read_bytes()

    def fake_open_source(p, *, compression, member, storage_options):
        from oxyz._remote import RemoteSource

        return RemoteSource(obj=iter([blob]), codec="plain", member=None)

    monkeypatch.setattr(oxyz._remote, "is_remote", lambda p: True)
    monkeypatch.setattr(oxyz._remote, "open_source", fake_open_source)

    # index=0 (forward) and index=-1 (reverse fallback) both work remotely.
    first = oxyz.ase.read("s3://bucket/minimal_periodic.extxyz", index=0)
    last = oxyz.ase.read("s3://bucket/minimal_periodic.extxyz", index=-1)
    assert len(first) > 0
    assert len(last) > 0


def test_cli_storage_option_parsing(monkeypatch, capsys):
    import oxyz._cli as cli

    path = Path("tests/data/minimal_periodic.extxyz")
    blob = path.read_bytes()
    seen = {}

    def fake_open_source(p, *, compression, member, storage_options):
        from oxyz._remote import RemoteSource

        seen["storage_options"] = storage_options
        return RemoteSource(obj=iter([blob]), codec="plain", member=None)

    monkeypatch.setattr(oxyz._remote, "is_remote", lambda p: True)
    monkeypatch.setattr(oxyz._remote, "open_source", fake_open_source)

    rc = cli.main(
        [
            "scan",
            "s3://bucket/minimal_periodic.extxyz",
            "--no-schema",
            "--storage-option",
            "endpoint=http://localhost:9000",
            "--storage-option",
            "region=us-east-1",
        ]
    )
    assert rc == 0
    assert seen["storage_options"] == {
        "endpoint": "http://localhost:9000",
        "region": "us-east-1",
    }


def test_import_obstore_missing_raises(monkeypatch):
    # A None entry in sys.modules makes `import obstore` raise ImportError,
    # exercising the real _import_obstore body (not the monkeypatched stub).
    monkeypatch.setitem(sys.modules, "obstore", None)
    with pytest.raises(ImportError, match=r"oxyz\[s3\]"):
        _remote._import_obstore()


def test_split_url_parses_and_requires_key():
    assert _remote._split_url("s3://bucket/path/to/x.xyz") == (
        "s3",
        "s3://bucket",
        "path/to/x.xyz",
    )
    with pytest.raises(ValueError, match="no object path"):
        _remote._split_url("s3://bucket")
    with pytest.raises(ValueError, match="no object path"):
        _remote._split_url("s3://bucket/")


def test_resolve_codec_explicit_compression_skips_sniff():
    # Explicit compression returns without touching the store (obstore is None).
    assert _remote._resolve_codec(None, None, "train.xyz", "none") == "plain"
    assert _remote._resolve_codec(None, None, "train.xyz", "gzip") == "gzip"


def test_resolve_codec_infers_from_magic_bytes():
    class FakeObstore:
        @staticmethod
        def get_range(store, key, *, start, length):
            return b"\x1f\x8b\x08\x00"  # gzip magic, extension says nothing

    assert _remote._resolve_codec(FakeObstore, object(), "blob", "infer") == "gzip"


def test_readable_bytes_adapter_returns_plain_bytes():
    class FakeReader:
        def __init__(self) -> None:
            self.pos = 0

        def read(self, n: int = -1) -> object:
            return memoryview(b"abc")  # a buffer, not plain bytes

        def seek(self, pos: int, whence: int = 0) -> int:
            self.pos = pos
            return pos

        def tell(self) -> int:
            return self.pos

    adapter = _remote._ReadableBytesAdapter(FakeReader())
    out = adapter.read(3)
    assert isinstance(out, bytes)
    assert out == b"abc"
    assert adapter.seek(5) == 5
    assert adapter.tell() == 5
    assert adapter.seekable() is True


def test_readable_bytes_adapter_handles_none_eof():
    class NoneReader:
        def read(self, n: int = -1) -> None:
            return None

    assert _remote._ReadableBytesAdapter(NoneReader()).read() == b""


def test_parse_storage_options_rejects_malformed():
    import oxyz._cli as cli

    assert cli._parse_storage_options([]) is None
    assert cli._parse_storage_options(["region=us-east-1"]) == {"region": "us-east-1"}
    with pytest.raises(ValueError, match="KEY=VALUE"):
        cli._parse_storage_options(["noequals"])
