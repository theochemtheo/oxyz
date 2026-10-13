"""Reading extxyz from remote object stores through the optional `obstore`.

`hf://` and Hub http(s) URLs are rewritten onto the Hub's raw-bytes HTTPS
endpoint; see `_normalise_hf_url`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from oxyz import _rust

# obstore ClientConfig keys; the rest of storage_options is store config.
_CLIENT_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "allow_http",
        "allow_invalid_certificates",
        "connect_timeout",
        "default_content_type",
        "default_headers",
        "http1_only",
        "http2_keep_alive_interval",
        "http2_keep_alive_timeout",
        "http2_keep_alive_while_idle",
        "http2_only",
        "pool_idle_timeout",
        "pool_max_idle_per_host",
        "proxy_url",
        "proxy_ca_certificate",
        "proxy_excludes",
        "root_certificate",
        "randomize_addresses",
        "read_timeout",
        "timeout",
        "user_agent",
    }
)

if TYPE_CHECKING:
    from pathlib import Path

    from obstore.store import AzureConfig, GCSConfig, S3Config

    StorageOptions = S3Config | GCSConfig | AzureConfig | dict[str, Any]
    """The accepted shape of `storage_options`, keyed by the provider the URL
    scheme selects. obstore's TypedDicts (endpoint, region, credentials, ...)
    are the source of truth; all keys are optional (total=False), so AWS needs
    none and non-AWS stores set endpoint/region/path-style as required."""
else:
    StorageOptions = dict

# `hf` is rewritten onto the Hub's HTTPS endpoint; see `_normalise_hf_url`.
SUPPORTED_SCHEMES = frozenset({"s3", "gs", "az", "azure", "abfs", "abfss", "hf"})

# Only these http(s) hosts are claimed, so a pasted Hub browser URL works.
_HF_HOSTS = frozenset({"huggingface.co", "hf.co"})
_HF_HOST = "huggingface.co"
# `blob` is the file's HTML page; `raw` gives an LFS pointer for large files.
_HF_RESOLVE = "resolve"
_HF_BLOB = "blob"
_HF_FILE_ENDPOINTS = frozenset({_HF_RESOLVE, _HF_BLOB, "raw"})
_HF_DEFAULT_REVISION = "main"
# HUGGING_FACE_HUB_TOKEN is the older name.
_HF_TOKEN_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
_HF_REPO_PARTS = 2
_HF_REVISION_AND_PATH = 2


class _HfRepoKind(StrEnum):
    """A Hub repo type, valued as the segment naming it in a URL.

    Attributes
    ----------
    MODEL
        A model repo, served off the root with no prefix.
    DATASET
        A dataset repo, under `datasets/`.
    SPACE
        A Space, under `spaces/`.
    """

    MODEL = "models"
    DATASET = "datasets"
    SPACE = "spaces"

    @property
    def prefix(self) -> str:
        """Return the URL path prefix: empty for a model, `<kind>/` otherwise."""
        return "" if self is _HfRepoKind.MODEL else f"{self}/"


_HF_KIND_SEGMENTS = frozenset(kind.value for kind in _HfRepoKind)


@dataclass(frozen=True, slots=True)
class _HfFile:
    """One file in a Hub repo; `url` renders the URL that serves its bytes."""

    kind: _HfRepoKind
    owner: str
    repo: str
    revision: str
    path: str
    endpoint: str = _HF_RESOLVE

    @property
    def url(self) -> str:
        """The Hub URL these bytes are fetched from."""
        return (
            f"https://{_HF_HOST}/{self.kind.prefix}{self.owner}/{self.repo}"
            f"/{self.endpoint}/{self.revision}/{self.path}"
        )


_SNIFF_BYTES = 8
# Bounds the memory `iread` holds while streaming.
_CHUNK = 1024 * 1024


def _scheme(path: str | Path) -> str:
    """Return `path`'s URL scheme, lowercased (empty for a plain filesystem path)."""
    return urlsplit(str(path)).scheme.lower()


def _is_hf_http(path: str | Path) -> bool:
    """Report whether `path` is an http(s) URL on one of the Hub's hosts."""
    parts = urlsplit(str(path))
    return parts.scheme.lower() in ("http", "https") and (
        parts.netloc.lower() in _HF_HOSTS
    )


def is_remote(path: str | Path) -> bool:
    """Report whether `path` is a URL oxyz reads through obstore.

    Parameters
    ----------
    path
        The path or URL to check.

    Returns
    -------
    bool
        `True` for a `SUPPORTED_SCHEMES` URL or an http(s) URL on a
        HuggingFace Hub host.
    """
    return _scheme(path) in SUPPORTED_SCHEMES or _is_hf_http(path)


def _take_repo_kind(segments: list[str]) -> tuple[_HfRepoKind, list[str]]:
    """Split a leading repo-kind segment off `segments`, defaulting to a model.

    As elsewhere with `hf://`, a model repo owned by `datasets` is unreachable.
    """
    if segments and segments[0] in _HF_KIND_SEGMENTS:
        return _HfRepoKind(segments[0]), segments[1:]
    return _HfRepoKind.MODEL, segments


def _parse_hf_scheme_url(url: str) -> _HfFile:
    """Parse `hf://[datasets|spaces|models/]owner/repo[@revision]/path`."""
    kind, rest = _take_repo_kind([s for s in url[len("hf://") :].split("/") if s])
    if len(rest) <= _HF_REPO_PARTS:
        raise ValueError(
            f"hf:// URL needs owner, repo and a file path within the repo: {url!r}"
        )
    owner, repo = rest[0], rest[1]
    # `repo@v1.0`, as `hf://` paths are written elsewhere.
    repo, _, revision = repo.partition("@")
    return _HfFile(
        kind=kind,
        owner=owner,
        repo=repo,
        revision=revision or _HF_DEFAULT_REVISION,
        path="/".join(rest[_HF_REPO_PARTS:]),
    )


def _parse_hf_http_url(url: str) -> _HfFile:
    """Parse a Hub http(s) URL: `[kind/]owner/repo/<endpoint>/<revision>/path`.

    A `blob` endpoint is swapped for `resolve`, which serves the bytes; an
    explicit `raw` is kept.
    """
    not_a_file = (
        f"HuggingFace URL does not name a file: {url!r} — expected a "
        f"'/resolve/' or '/blob/' URL, or an hf:// path"
    )
    segments = urlsplit(url).path.strip("/").split("/")
    # Without an endpoint the URL is a repo, directory or search page.
    at = next(
        (i for i, seg in enumerate(segments) if i > 0 and seg in _HF_FILE_ENDPOINTS),
        None,
    )
    if at is None:
        raise ValueError(not_a_file)
    kind, repo = _take_repo_kind(segments[:at])
    revision_and_path = segments[at + 1 :]
    if len(repo) != _HF_REPO_PARTS or len(revision_and_path) < _HF_REVISION_AND_PATH:
        raise ValueError(not_a_file)
    endpoint = segments[at]
    return _HfFile(
        kind=kind,
        owner=repo[0],
        repo=repo[1],
        revision=revision_and_path[0],
        path="/".join(revision_and_path[1:]),
        endpoint=_HF_RESOLVE if endpoint == _HF_BLOB else endpoint,
    )


def _normalise_hf_url(url: str) -> str:
    """Rewrite a HuggingFace URL to the Hub endpoint that serves the bytes.

    Any other URL is returned unchanged.
    """
    if _is_hf_http(url):
        return _parse_hf_http_url(url).url
    if _scheme(url) != "hf":
        return url
    return _parse_hf_scheme_url(url).url


def _hf_storage_options(
    storage_options: StorageOptions | None,
) -> StorageOptions | None:
    """Add a bearer header from `HF_TOKEN` unless the caller set one.

    Only gated and private repos need it, so a missing token is not an error.
    """
    headers: dict[str, Any] = dict((storage_options or {}).get("default_headers") or {})
    if any(k.lower() == "authorization" for k in headers):
        return storage_options
    token = next((t for t in map(os.environ.get, _HF_TOKEN_ENV) if t), None)
    if token is None:
        return storage_options
    headers["authorization"] = f"Bearer {token}"
    return {**(storage_options or {}), "default_headers": headers}


@dataclass(frozen=True, slots=True)
class RemoteSource:
    """A streaming source for the `_rust.*_reader` entries.

    `obj` is a bytes-iterator (plain/gzip/zstd), a callable returning a fresh
    one (tar codecs), or a seekable file-like (zip), as `codec` says.
    """

    obj: Any
    codec: str
    member: str | None


def _raise_missing(*_args: object, **_kwargs: object) -> Any:
    """Raise the `obstore`-not-installed `ImportError`."""
    raise ImportError(
        "reading from a remote URL needs the optional 'obstore' dependency — "
        "install it with: pip install oxyz[s3]"
    )


def _import_obstore() -> Any:
    """Import and return the `obstore` module, or raise if it is not installed."""
    try:
        import obstore
    except ImportError:
        _raise_missing()
    return obstore


def _split_url(url: str) -> tuple[str, str, str]:
    """`(scheme, bucket_url, key)` from a remote URL."""
    parts = urlsplit(url)
    bucket_url = f"{parts.scheme}://{parts.netloc}"
    key = parts.path.lstrip("/")
    if not key:
        raise ValueError(f"remote URL has no object path: {url!r}")
    return parts.scheme, bucket_url, key


def _build_store(bucket_url: str, storage_options: StorageOptions | None) -> Any:
    """Build an obstore store for `bucket_url`, splitting client/provider config."""
    _import_obstore()
    from obstore.store import from_url

    raw: dict[str, Any] = dict(storage_options or {})
    # obstore's from_url panics if ClientConfig keys land in config=.
    client: dict[str, Any] = {k: raw.pop(k) for k in _CLIENT_CONFIG_KEYS if k in raw}
    # Any: obstore's overloads reject a plain dict that works at runtime.
    cfg: Any = raw
    _from_url: Any = from_url
    return _from_url(
        bucket_url,
        config=cfg or None,
        client_options=client or None,
    )


def _resolve_codec(obstore: Any, store: Any, key: str, compression: str) -> str:
    """Resolve the codec for this object.

    On `"infer"` the extension decides; only if it says nothing is one range
    request spent sniffing the magic bytes.
    """
    if compression != "infer":
        return "plain" if compression == "none" else compression
    codec = _rust.detect_codec(key, None)
    if codec == "plain":
        head = bytes(obstore.get_range(store, key, start=0, length=_SNIFF_BYTES))
        codec = _rust.detect_codec(key, head)
    return codec


class _ReadableBytesAdapter:
    """Wrap obstore's `ReadableFile` so `read()` returns `bytes`.

    The binding cannot extract the `obstore.Bytes` it returns otherwise.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def read(self, n: int = -1) -> bytes:
        chunk = self._inner.read(n)
        return bytes(chunk) if chunk is not None else b""

    def seek(self, pos: int, whence: int = 0) -> int:
        return self._inner.seek(pos, whence)

    def tell(self) -> int:
        return self._inner.tell()

    def seekable(self) -> bool:
        return True


def open_source(
    path: str | Path,
    *,
    compression: str,
    member: str | None,
    storage_options: StorageOptions | None,
) -> RemoteSource:
    """Open `path` as a `RemoteSource` for the `_rust.*_reader` entries.

    Parameters
    ----------
    path
        A URL `is_remote` accepts.
    compression
        Forces a codec, or `"infer"` to detect it from the key/magic bytes.
    member
        Selects one entry from an archive holding more than one.
    storage_options
        Endpoint/credentials for the store, falling back to `AWS_*` env vars.

    Returns
    -------
    RemoteSource
        A streaming source ready for the `_rust.*_reader` entries.
    """
    obstore = _import_obstore()
    url = str(path)
    if _scheme(url) == "hf" or _is_hf_http(url):
        url = _normalise_hf_url(url)
        storage_options = _hf_storage_options(storage_options)
    _, bucket_url, key = _split_url(url)
    store = _build_store(bucket_url, storage_options)
    codec = _resolve_codec(obstore, store, key, compression)

    if codec == "zip":
        obj: Any = _ReadableBytesAdapter(obstore.open_reader(store, key))
    elif codec in ("tar", "tar.gz", "tar.zst"):
        obj = lambda: iter(obstore.get(store, key).stream(min_chunk_size=_CHUNK))  # noqa: E731
    else:
        obj = iter(obstore.get(store, key).stream(min_chunk_size=_CHUNK))
    return RemoteSource(obj=obj, codec=codec, member=member)
