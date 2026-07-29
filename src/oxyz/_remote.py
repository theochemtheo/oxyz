"""Reading extxyz from remote object stores via the optional `obstore` dep.

Recognises S3-and-friends URLs, builds an obstore store from the URL plus typed
`storage_options` (falling back to AWS_* env vars), resolves the codec from the
URL name (with a cheap magic-byte sniff), and hands the binding a streaming
source. The only module that imports `obstore`; the import is lazy, so the base
install stays numpy-only.

HuggingFace Hub URLs are handled here too. The Hub serves a repo's files as raw
bytes over plain HTTPS, so `hf://` URLs are rewritten onto that endpoint and read
through obstore's HTTP store; see `_normalise_hf_url`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from oxyz import _rust

# Keys belonging to obstore's ClientConfig TypedDict (HTTP transport layer).
# Anything else in storage_options is treated as store config (S3Config etc.).
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

# obstore-backed schemes. `hf` is served over plain HTTPS against the Hub, so it
# is rewritten before the store is built; see `_normalise_hf_url`.
SUPPORTED_SCHEMES = frozenset({"s3", "gs", "az", "azure", "abfs", "abfss", "hf"})

# Hub hosts oxyz claims when they appear in an http(s) URL. http(s) is *not*
# supported in general — only these hosts, so pasting a browser URL works
# without oxyz laying claim to every URL on the web.
_HF_HOSTS = frozenset({"huggingface.co", "hf.co"})
_HF_HOST = "huggingface.co"
# The Hub's raw-bytes endpoint. `blob` is the HTML page for the same file, and
# `raw` serves the file itself for small files, an LFS pointer for large ones.
_HF_RESOLVE = "resolve"
_HF_BLOB = "blob"
# Segments that mark a Hub URL as naming a file rather than a page.
_HF_FILE_ENDPOINTS = frozenset({_HF_RESOLVE, _HF_BLOB, "raw"})
_HF_DEFAULT_REVISION = "main"
# Checked in order; HF_TOKEN is current, HUGGING_FACE_HUB_TOKEN the older name.
_HF_TOKEN_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
# owner + repo, before any path within the repo.
_HF_REPO_PARTS = 2
# revision + at least one path segment, after the endpoint in an http(s) URL.
_HF_REVISION_AND_PATH = 2


class _HfRepoKind(StrEnum):
    """A Hub repo type, valued as the segment naming it in a URL.

    Attributes
    ----------
    MODEL
        A model repo. The Hub serves these off the root, so unlike the other
        two this kind contributes no path prefix.
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
    """One file in a Hub repo, however it was named.

    Both accepted URL forms parse to this, and `url` renders the one the Hub
    serves bytes from, so the two forms cannot drift apart.
    """

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


# How many leading bytes to sniff when the URL extension is uninformative.
_SNIFF_BYTES = 8
# Minimum streamed chunk size: balances request count against memory held while
# iterating. 1 MiB keeps `iread`'s memory bounded.
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
        `True` if `path`'s scheme is one of `SUPPORTED_SCHEMES`, or if it is an
        http(s) URL on a HuggingFace Hub host. Other http(s) URLs are not
        claimed.
    """
    return _scheme(path) in SUPPORTED_SCHEMES or _is_hf_http(path)


def _take_repo_kind(segments: list[str]) -> tuple[_HfRepoKind, list[str]]:
    """Split a leading repo-kind segment off `segments`, defaulting to a model.

    A model repo whose owner is literally named `datasets` is unreachable this
    way — an ambiguity the wider `hf://` convention shares, so matching it beats
    inventing a divergence.
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
    # The revision rides on the repo name (`repo@v1.0`), as `hf://` paths do
    # elsewhere in the ecosystem; the Hub wants it as its own path segment.
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

    A `blob` endpoint is the file's HTML page — what the browser address bar
    holds — so it is swapped for `resolve`, which serves the bytes. `raw` is
    kept as given: it is a deliberate, documented endpoint, and rewriting it
    would override an explicit choice.
    """
    not_a_file = (
        f"HuggingFace URL does not name a file: {url!r} — expected a "
        f"'/resolve/' or '/blob/' URL, or an hf:// path"
    )
    segments = urlsplit(url).path.strip("/").split("/")
    # The endpoint always follows the repo, so it never leads. Its absence marks
    # a repo, directory or search page, all of which serve HTML rather than a
    # file.
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

    Both accepted forms — an `hf://` path and a Hub http(s) URL — parse to the
    same `_HfFile`, which renders the canonical URL, so the two cannot drift
    apart. Any other URL is returned unchanged.
    """
    if _is_hf_http(url):
        return _parse_hf_http_url(url).url
    if _scheme(url) != "hf":
        return url
    return _parse_hf_scheme_url(url).url


def _hf_storage_options(
    storage_options: StorageOptions | None,
) -> StorageOptions | None:
    """Add a bearer header from `HF_TOKEN` unless the caller set one already.

    Mirrors the `AWS_*` fallback the S3 path gets for free from obstore, which
    has no notion of a Hub token. Gated and private repos need it; public ones
    do not, so a missing token is not an error.
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

    `obj` is a bytes-iterator (plain/gzip/zstd), a 0-arg callable returning a
    fresh bytes-iterator (tar/tar.gz/tar.zst), or a seekable file-like (zip);
    `codec` says which.
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
    """`(scheme, bucket_url, key)` from a remote URL.

    The store is built from the scheme+host (an empty prefix) and the object is
    fetched by `key`, so the URL's path never leaks into the store prefix.
    """
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
    # Split transport-layer keys (ClientConfig) from provider-config keys.
    # obstore's from_url panics if ClientConfig keys land in config=.
    client: dict[str, Any] = {k: raw.pop(k) for k in _CLIENT_CONFIG_KEYS if k in raw}
    # cfg typed Any so ty doesn't reject plain dict against obstore's overloads
    # (TypedDicts are dicts at runtime; from_url accepts all three at runtime).
    cfg: Any = raw
    # Cast to Any: obstore's per-provider overloads don't express the combined
    # config + client_options case; the implementation accepts both at runtime.
    _from_url: Any = from_url
    return _from_url(
        bucket_url,
        config=cfg or None,
        client_options=client or None,
    )


def _resolve_codec(obstore: Any, store: Any, key: str, compression: str) -> str:
    """Resolve the codec the binding should use for this object.

    Explicit `compression` maps straight through (`"none"` → `"plain"`); on
    `"infer"`, the extension decides, and only if it says nothing do we pay one
    cheap range request to sniff the magic bytes.
    """
    if compression != "infer":
        return "plain" if compression == "none" else compression
    codec = _rust.detect_codec(key, None)
    if codec == "plain":
        head = bytes(obstore.get_range(store, key, start=0, length=_SNIFF_BYTES))
        codec = _rust.detect_codec(key, head)
    return codec


class _ReadableBytesAdapter:
    """Wraps obstore's ReadableFile so that read() returns plain bytes.

    obstore.ReadableFile.read() returns obstore.Bytes (a zero-copy buffer).
    The Rust binding calls .read() and extracts PyBytes — it panics on
    obstore.Bytes even though it implements the buffer protocol.  Wrapping
    ensures the return type is always plain Python bytes.
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
        Selects one entry from an archive holding more than one; carried
        through unused, for the reader to apply.
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
        # Wrap so that read() returns plain bytes; obstore.ReadableFile.read()
        # returns obstore.Bytes, which the Rust binding cannot extract.
        obj: Any = _ReadableBytesAdapter(obstore.open_reader(store, key))
    elif codec in ("tar", "tar.gz", "tar.zst"):
        obj = lambda: iter(obstore.get(store, key).stream(min_chunk_size=_CHUNK))  # noqa: E731
    else:
        obj = iter(obstore.get(store, key).stream(min_chunk_size=_CHUNK))
    return RemoteSource(obj=obj, codec=codec, member=member)
