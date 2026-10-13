"""Every name in `oxyz.__all__` has its own docstring.

ruff's `D` rules cannot see this: the defining modules are private.
"""

from __future__ import annotations

import importlib.util

import pytest

import oxyz

# Type aliases are documented by a comment at their definition, as they have
# no docstring of their own.
KNOWN_TYPE_ALIASES = frozenset(
    {
        "Compression",
        "ColumnValues",
        "MetadataValue",
        "Conformance",
        "Mode",
        "MemoryScaling",
        "Writable",
    }
)


def _has_own_docstring(obj: object) -> bool:
    doc = obj.__dict__.get("__doc__") if isinstance(obj, type) else obj.__doc__
    return isinstance(doc, str) and bool(doc.strip())


@pytest.mark.parametrize("name", sorted(oxyz.__all__))
def test_public_surface_name_is_documented(name: str) -> None:
    """Check that an `__all__` name is either documented or a listed type alias."""
    if name in KNOWN_TYPE_ALIASES:
        pytest.skip(f"{name} is a type alias, exempted via KNOWN_TYPE_ALIASES")
    obj = getattr(oxyz, name)
    assert _has_own_docstring(obj), (
        f"oxyz.{name} has no docstring of its own; document it or, if it is a "
        f"type alias, add it to KNOWN_TYPE_ALIASES in this test"
    )


def test_known_type_aliases_are_still_in_all() -> None:
    """`KNOWN_TYPE_ALIASES` must not go stale: every listed name is still exported."""
    assert set(oxyz.__all__) >= KNOWN_TYPE_ALIASES


@pytest.mark.skipif(importlib.util.find_spec("ase") is None, reason="requires ase")
@pytest.mark.parametrize("name", ["read", "iread"])
def test_ase_entry_points_are_documented(name: str) -> None:
    """`oxyz.ase.read`/`iread` carry their own docstring."""
    import oxyz.ase

    obj = getattr(oxyz.ase, name)
    assert _has_own_docstring(obj), f"oxyz.ase.{name} has no docstring"


@pytest.mark.skipif(
    importlib.util.find_spec("ase") is None
    or importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("metatomic") is None,
    reason="requires ase, torch, and metatomic-torch",
)
@pytest.mark.parametrize("name", ["read", "iread", "SystemSource"])
def test_metatomic_entry_points_are_documented(name: str) -> None:
    """`oxyz.metatomic.read`/`iread`/`SystemSource` carry their own docstring."""
    import oxyz.metatomic

    obj = getattr(oxyz.metatomic, name)
    assert _has_own_docstring(obj), f"oxyz.metatomic.{name} has no docstring"


@pytest.mark.skipif(
    importlib.util.find_spec("ase") is None
    or importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("torch_sim") is None,
    reason="requires ase, torch, and torch-sim-atomistic",
)
@pytest.mark.parametrize("name", ["read", "iread", "SimStateSource"])
def test_torch_sim_entry_points_are_documented(name: str) -> None:
    """`oxyz.torch_sim.read`/`iread`/`SimStateSource` carry their own docstring."""
    import oxyz.torch_sim

    obj = getattr(oxyz.torch_sim, name)
    assert _has_own_docstring(obj), f"oxyz.torch_sim.{name} has no docstring"
