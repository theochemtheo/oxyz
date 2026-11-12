from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import pytest

import oxyz


def test_public_api_imports() -> None:
    assert oxyz.Frame is not None
    assert oxyz.read is not None
    assert oxyz.iread is not None


def test_mode_and_writable_are_exported() -> None:
    assert oxyz.Mode is not None
    assert oxyz.Writable is not None
    assert "Mode" in oxyz.__all__
    assert "Writable" in oxyz.__all__


# Readers share one keyword tail for the threads, schema and IO options.
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


def test_a_read_without_record_loads_no_record_machinery() -> None:
    path = Path(__file__).parent / "data" / "mace_ref_energy_forces_stress.xyz"
    code = f"""
import sys
import oxyz
spec = oxyz.SchemaSpec(
    metadata=(oxyz.MetadataRule("REF_energy", oxyz.Kind.REAL),), mode="project"
)
oxyz.read({str(path)!r}, schema=spec)
list(oxyz.iread({str(path)!r}, schema=spec))
oxyz.read({str(path)!r}, schema=spec, mode="validate")
print([m for m in ("oxyz._record_bind", "oxyz._record_fields") if m in sys.modules])
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
