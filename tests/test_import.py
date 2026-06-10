from __future__ import annotations

import oxyz


def test_public_api_imports() -> None:
    assert oxyz.Frame is not None
    assert oxyz.read_first_frame is not None
