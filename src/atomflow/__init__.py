from __future__ import annotations

from . import _rust


def rust_version() -> str:
    """Return the version of the Rust extension.

    This is just a scaffold sanity check. It can disappear once real API
    functions exist.
    """
    return _rust.version()


__all__ = ["rust_version"]
