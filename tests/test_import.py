from __future__ import annotations

import oxyz


def test_public_api_imports() -> None:
    assert oxyz.Frame is not None
    assert oxyz.read_first is not None


def test_schema_public_api_exported():
    import oxyz

    for name in (
        "SchemaSpec",
        "ColumnRule",
        "MetadataRule",
        "FrameRule",
        "SchemaError",
        "SchemaWarning",
        "Violation",
    ):
        assert hasattr(oxyz, name), name
    assert name in oxyz.__all__ if (name := "SchemaSpec") else True
