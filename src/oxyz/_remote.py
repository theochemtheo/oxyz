"""Reading extxyz from remote object stores via the optional `obstore` dep.

Recognises S3-and-friends URLs, builds an obstore store from the URL plus typed
`storage_options` (falling back to AWS_* env vars), resolves the codec from the
URL name (with a cheap magic-byte sniff), and hands the binding a streaming
source. The only module that imports `obstore`; the import is lazy, so the base
install stays numpy-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import oxyz._rust as _rust

if TYPE_CHECKING:
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
# iterating. 1 MiB keeps `iter_frames` near-constant memory.
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
    from obstore.store import from_url

    # cfg typed Any so ty doesn't reject plain dict against obstore's overloads
    # (TypedDicts are dicts at runtime; from_url accepts all three at runtime).
    cfg: Any = dict(storage_options or {})
    return from_url(bucket_url, config=cfg)


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


def open_source(
    path: str | Path,
    *,
    compression: str,
    member: str | None,
    storage_options: StorageOptions | None,
) -> RemoteSource:
    """Open `path` as a `RemoteSource` for the `_rust.*_reader` entries."""
    obstore = _import_obstore()
    _scheme_, bucket_url, key = _split_url(str(path))
    store = _build_store(bucket_url, storage_options)
    codec = _resolve_codec(obstore, store, key, compression)

    if codec == "zip":
        obj: Any = obstore.open_reader(store, key)  # seekable ReadableFile
    elif codec in ("tar", "tar.gz"):
        obj = lambda: iter(obstore.get(store, key).stream(min_chunk_size=_CHUNK))  # noqa: E731
    else:
        obj = iter(obstore.get(store, key).stream(min_chunk_size=_CHUNK))
    return RemoteSource(obj=obj, codec=codec, member=member)
