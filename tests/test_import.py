from __future__ import annotations

import inspect

import pytest

import oxyz


def test_public_api_imports() -> None:
    assert oxyz.Frame is not None
    assert oxyz.read is not None
    assert oxyz.iread is not None


def test_mode_and_writable_are_exported() -> None:
    # Parity with the other public type aliases (Conformance, Compression): the
    # kwarg/type names users annotate with are importable and in __all__.
    assert oxyz.Mode is not None
    assert oxyz.Writable is not None
    assert "Mode" in oxyz.__all__
    assert "Writable" in oxyz.__all__


# Every reader that takes both `threads` and the schema/IO options shares one
# canonical keyword tail, so the surface reads the same across the package.
CANONICAL_TAIL = [
    "threads",
    "schema",
    "conformance",
    "mode",
    "compression",
    "member",
    "storage_options",
]


def _reader(name: str):
    import importlib

    module_name, _, attr = name.rpartition(".")
    module = importlib.import_module(f"oxyz{'.' + module_name if module_name else ''}")
    return getattr(module, attr)


@pytest.mark.parametrize(
    "name",
    [
        "read",
        "read_batch",
        "iread_batch",
        "ase.read",
        "metatomic.read",
        "torch_sim.read",
    ],
)
def test_reader_keyword_tail_is_canonical(name: str) -> None:
    pytest.importorskip("ase") if name.startswith("ase") else None
    pytest.importorskip("metatomic.torch") if name.startswith("metatomic") else None
    pytest.importorskip("torch_sim") if name.startswith("torch_sim") else None
    params = list(inspect.signature(_reader(name)).parameters)
    tail = params[params.index("threads") :]
    assert tail == CANONICAL_TAIL, name


def test_schema_public_api_exported():
    import oxyz

    for name in (
        "Conformance",
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
