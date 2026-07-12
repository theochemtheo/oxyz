"""Reading extxyz from remote object stores via the optional `obstore` dep.

Recognises S3-and-friends URLs, builds an obstore store from the URL plus typed
`storage_options` (falling back to AWS_* env vars), resolves the codec from the
URL name (with a cheap magic-byte sniff), and hands the binding a streaming
source. The only module that imports `obstore`; the import is lazy, so the base
install stays numpy-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import oxyz._rust as _rust

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

    # The accepted shape of `storage_options`, keyed by the provider the URL
    # scheme selects. obstore's TypedDicts are the source of truth; all keys are
    # optional (total=False), so AWS needs none and non-AWS stores set endpoint
    # /region/path-style as required.
    StorageOptions = S3Config | GCSConfig | AzureConfig | dict[str, Any]
else:
    StorageOptions = dict

# obstore-backed schemes. `hf` is deliberately excluded until obstore ships a
# HuggingFace backend (tracked separately).
SUPPORTED_SCHEMES = frozenset({"s3", "gs", "az", "azure", "abfs", "abfss"})

# How many leading bytes to sniff when the URL extension is uninformative.
_SNIFF_BYTES = 8
# Minimum streamed chunk size: balances request count against memory held while
# iterating. 1 MiB keeps `iread` near-constant memory.
_CHUNK = 1024 * 1024


def _scheme(path: str | Path) -> str:
    return urlsplit(str(path)).scheme.lower()


def is_remote(path: str | Path) -> bool:
    """True if `path` is a URL in a supported remote scheme."""
    return _scheme(path) in SUPPORTED_SCHEMES


@dataclass(frozen=True, slots=True)
class RemoteSource:
    """A streaming source for the `_rust.*_reader` entries.

    `obj` is a bytes-iterator (plain/gzip/zstd), a 0-arg callable returning a
    fresh bytes-iterator (tar/tar.gz), or a seekable file-like (zip); `codec`
    says which.
    """

    obj: Any
    codec: str
    member: str | None


def _raise_missing(*_args: object, **_kwargs: object) -> Any:
    raise ImportError(
        "reading from a remote URL needs the optional 'obstore' dependency — "
        "install it with: pip install oxyz[s3]"
    )


def _import_obstore() -> Any:
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
    """Open `path` as a `RemoteSource` for the `_rust.*_reader` entries."""
    obstore = _import_obstore()
    _, bucket_url, key = _split_url(str(path))
    store = _build_store(bucket_url, storage_options)
    codec = _resolve_codec(obstore, store, key, compression)

    if codec == "zip":
        # Wrap so that read() returns plain bytes; obstore.ReadableFile.read()
        # returns obstore.Bytes, which the Rust binding cannot extract.
        obj: Any = _ReadableBytesAdapter(obstore.open_reader(store, key))
    elif codec in ("tar", "tar.gz"):
        obj = lambda: iter(obstore.get(store, key).stream(min_chunk_size=_CHUNK))  # noqa: E731
    else:
        obj = iter(obstore.get(store, key).stream(min_chunk_size=_CHUNK))
    return RemoteSource(obj=obj, codec=codec, member=member)
