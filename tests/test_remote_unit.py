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


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("hf://datasets/owner/repo/data/train.extxyz", True),
        ("https://huggingface.co/datasets/owner/repo/blob/main/a.extxyz", True),
        ("https://hf.co/datasets/owner/repo/resolve/main/a.extxyz", True),
        # only the Hub's own hosts are claimed; http(s) in general is not
        ("https://example.com/train.extxyz", False),
        ("http://localhost:9000/bucket/train.extxyz", False),
    ],
)
def test_is_remote_huggingface(path, expected):
    assert _remote.is_remote(path) is expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        # a repo kind prefix selects the repo type; the revision defaults to main
        (
            "hf://datasets/o/r/data/a.extxyz",
            "https://huggingface.co/datasets/o/r/resolve/main/data/a.extxyz",
        ),
        (
            "hf://datasets/o/r@v1.0/data/a.extxyz",
            "https://huggingface.co/datasets/o/r/resolve/v1.0/data/a.extxyz",
        ),
        (
            "hf://spaces/o/r@abc123/a.extxyz",
            "https://huggingface.co/spaces/o/r/resolve/abc123/a.extxyz",
        ),
        # no prefix (and the explicit `models` prefix) means a model repo, which
        # the Hub serves straight off the root
        ("hf://o/r/a.extxyz", "https://huggingface.co/o/r/resolve/main/a.extxyz"),
        (
            "hf://models/o/r@dev/a.extxyz",
            "https://huggingface.co/o/r/resolve/dev/a.extxyz",
        ),
        # a URL pasted from the browser points at the HTML page, not the bytes
        (
            "https://huggingface.co/datasets/o/r/blob/main/a.extxyz",
            "https://huggingface.co/datasets/o/r/resolve/main/a.extxyz",
        ),
        # hf.co is the short host for the same Hub; /resolve/ is already correct
        (
            "https://hf.co/datasets/o/r/resolve/main/a.extxyz",
            "https://huggingface.co/datasets/o/r/resolve/main/a.extxyz",
        ),
    ],
)
def test_normalise_hf_url(url, expected):
    assert _remote._normalise_hf_url(url) == expected


def test_normalise_hf_url_passes_other_urls_through():
    assert _remote._normalise_hf_url("s3://bucket/a.xyz") == "s3://bucket/a.xyz"


@pytest.mark.parametrize(
    "url",
    [
        "hf://datasets/owner",  # no repo name
        "hf://datasets/owner/repo",  # repo but no file within it
        "hf://owner",
        "hf://",
    ],
)
def test_normalise_hf_url_rejects_incomplete(url):
    with pytest.raises(ValueError, match="hf:// URL"):
        _remote._normalise_hf_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/datasets/o/r",  # the repo page, not a file
        "https://huggingface.co/datasets/o/r/tree/main/data",  # a directory page
        "https://huggingface.co/",
        "https://huggingface.co/datasets/o/r/resolve/main",  # revision, no file
        "https://huggingface.co/datasets/o/r/blob",  # endpoint, nothing after
        "https://huggingface.co/datasets/o/resolve/main/a.xyz",  # no repo name
    ],
)
def test_normalise_hf_url_rejects_http_urls_naming_no_file(url):
    # These serve HTML, so fetching them would hand the parser markup.
    with pytest.raises(ValueError, match="does not name a file"):
        _remote._normalise_hf_url(url)


def test_normalise_hf_url_keeps_raw_endpoint():
    # /raw/ is a deliberate Hub endpoint; unlike /blob/ it is not rewritten.
    assert (
        _remote._normalise_hf_url("https://huggingface.co/datasets/o/r/raw/main/a.xyz")
        == "https://huggingface.co/datasets/o/r/raw/main/a.xyz"
    )


def test_parse_hf_scheme_url_fields():
    ref = _remote._parse_hf_scheme_url("hf://datasets/o/r@v1.0/data/a.extxyz")
    assert ref.kind is _remote._HfRepoKind.DATASET
    assert ref.owner == "o"
    assert ref.repo == "r"
    assert ref.revision == "v1.0"
    assert ref.path == "data/a.extxyz"
    assert ref.endpoint == "resolve"


def test_parse_hf_scheme_url_defaults_to_a_model_on_main():
    ref = _remote._parse_hf_scheme_url("hf://o/r/a.extxyz")
    assert ref.kind is _remote._HfRepoKind.MODEL
    assert ref.revision == "main"
    assert ref.kind.prefix == ""  # models are served off the Hub root


def test_parse_hf_http_url_fields():
    ref = _remote._parse_hf_http_url("https://hf.co/o/r/raw/dev/data/a.xyz")
    assert ref.kind is _remote._HfRepoKind.MODEL
    assert ref.owner == "o"
    assert ref.repo == "r"
    assert ref.revision == "dev"
    assert ref.path == "data/a.xyz"
    assert ref.endpoint == "raw"  # preserved; only blob is redirected


def test_hf_file_renders_its_canonical_url():
    ref = _remote._HfFile(
        kind=_remote._HfRepoKind.SPACE,
        owner="o",
        repo="r",
        revision="main",
        path="data/a.xyz",
    )
    assert ref.url == "https://huggingface.co/spaces/o/r/resolve/main/data/a.xyz"


class _FakeObstore:
    """Just enough of obstore for open_source's plain-codec path."""

    class _Get:
        @staticmethod
        def stream(min_chunk_size=0):
            return iter([b""])

    @classmethod
    def get(cls, store, key):
        return cls._Get


def _capture_open_source(monkeypatch, url, storage_options=None):
    """Run `open_source` against fakes; return what reached the store layer.

    Asserting on the captured `(bucket_url, key, storage_options)` tests the
    promise that matters — what oxyz actually asks obstore for — rather than
    the shape of a helper's return value.
    """
    seen = {}

    def fake_build_store(bucket_url, options):
        seen["bucket_url"] = bucket_url
        seen["storage_options"] = options
        return object()

    def fake_resolve_codec(obstore, store, key, compression):
        seen["key"] = key
        return "plain"

    monkeypatch.setattr(_remote, "_build_store", fake_build_store)
    monkeypatch.setattr(_remote, "_resolve_codec", fake_resolve_codec)
    monkeypatch.setattr(_remote, "_import_obstore", lambda: _FakeObstore)

    _remote.open_source(
        url, compression="none", member=None, storage_options=storage_options
    )
    return seen


HF_URL = "hf://datasets/o/r@main/data/a.extxyz"


def test_hf_token_from_env_becomes_bearer_header(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    seen = _capture_open_source(monkeypatch, HF_URL)
    assert seen["storage_options"]["default_headers"] == {
        "authorization": "Bearer hf_secret"
    }


def test_hf_token_legacy_env_var(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_legacy")
    seen = _capture_open_source(monkeypatch, HF_URL)
    assert seen["storage_options"]["default_headers"] == {
        "authorization": "Bearer hf_legacy"
    }


def test_hf_explicit_authorization_wins_over_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    given = {"default_headers": {"authorization": "Bearer explicit"}}
    seen = _capture_open_source(monkeypatch, HF_URL, given)
    assert seen["storage_options"]["default_headers"] == {
        "authorization": "Bearer explicit"
    }
    assert given == {"default_headers": {"authorization": "Bearer explicit"}}


def test_hf_no_token_leaves_options_untouched(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    seen = _capture_open_source(monkeypatch, HF_URL, {"timeout": "30s"})
    assert seen["storage_options"] == {"timeout": "30s"}


def test_hf_token_not_applied_to_other_stores(monkeypatch):
    # The Hub token must never be attached to an unrelated store's requests.
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    seen = _capture_open_source(
        monkeypatch, "s3://bucket/train.xyz", {"region": "us-east-1"}
    )
    assert seen["storage_options"] == {"region": "us-east-1"}


def test_open_source_normalises_hf_url(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    seen = _capture_open_source(monkeypatch, HF_URL)
    assert seen["bucket_url"] == "https://huggingface.co"
    assert seen["key"] == "datasets/o/r/resolve/main/data/a.extxyz"


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


def test_iread_batch_streams_remote(monkeypatch):
    path = Path("tests/data/minimal_periodic.extxyz")
    blob = path.read_bytes()

    local_frames = sum(
        b.n_frames for b in oxyz.iread_batch(str(path), frames_per_batch=1)
    )

    def fake_open_source(p, *, compression, member, storage_options):
        from oxyz._remote import RemoteSource

        return RemoteSource(obj=iter([blob]), codec="plain", member=None)

    monkeypatch.setattr(oxyz._remote, "is_remote", lambda p: True)
    monkeypatch.setattr(oxyz._remote, "open_source", fake_open_source)

    batches = list(
        oxyz.iread_batch("s3://bucket/minimal_periodic.extxyz", frames_per_batch=1)
    )
    assert sum(b.n_frames for b in batches) == local_frames


def test_iread_batch_remote_rejects_random_access(monkeypatch):
    monkeypatch.setattr(oxyz._remote, "is_remote", lambda p: True)
    with pytest.raises(ValueError, match="randomly accessed"):
        list(oxyz.iread_batch("s3://bucket/x.xyz", frames_per_batch=2, shuffle=True))


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


def test_read_frames_routes_remote_tar_zstd(monkeypatch):
    blob = Path("tests/data/compressed/two_frame.tar.zst").read_bytes()

    def fake_open_source(p, *, compression, member, storage_options):
        from oxyz._remote import RemoteSource

        return RemoteSource(obj=lambda: iter([blob]), codec="tar.zst", member=None)

    monkeypatch.setattr(oxyz._remote, "is_remote", lambda p: True)
    monkeypatch.setattr(oxyz._remote, "open_source", fake_open_source)

    remote = oxyz.read("s3://bucket/two_frame.tar.zst")
    local = oxyz.read("tests/data/two_frame_same_schema.xyz")
    assert len(remote) == len(local)


def test_open_source_dispatches_tar_zst_to_callable_factory(monkeypatch):
    # A tar has no central directory, so the tar codecs need a 0-arg callable
    # that produces a fresh bytes-iterator per call (one pass to enumerate
    # members, another to stream them) rather than a plain, single-use one.
    class FakeGetResult:
        @staticmethod
        def stream(*, min_chunk_size):
            return iter([b"blob"])

    class FakeObstore:
        @staticmethod
        def get(store, key):
            return FakeGetResult()

    monkeypatch.setattr(_remote, "_import_obstore", lambda: FakeObstore)
    monkeypatch.setattr(_remote, "_build_store", lambda *_a, **_k: object())

    src = _remote.open_source(
        "s3://bucket/train.tar.zst",
        compression="tar.zst",
        member=None,
        storage_options=None,
    )
    assert callable(src.obj)
    assert list(src.obj()) == [b"blob"]
    assert list(src.obj()) == [b"blob"]  # a second call yields a fresh iterator


def test_resolve_codec_explicit_compression_skips_sniff():
    # Explicit compression returns without touching the store (obstore is None).
    assert _remote._resolve_codec(None, None, "train.xyz", "none") == "plain"
    assert _remote._resolve_codec(None, None, "train.xyz", "gzip") == "gzip"
    assert _remote._resolve_codec(None, None, "train.tar.zst", "tar.zst") == "tar.zst"


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
