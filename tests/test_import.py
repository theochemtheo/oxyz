from __future__ import annotations

import atomflow


def test_rust_extension_imports() -> None:
    assert atomflow.rust_version()
