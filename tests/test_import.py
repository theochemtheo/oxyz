from __future__ import annotations

import atomflow


def test_public_api_imports() -> None:
    assert atomflow.Frame is not None
    assert atomflow.read_first_frame is not None
