"""The error hierarchy: a single `oxyz.OxyzError` base under `ValueError`.

Every error oxyz raises subclasses `OxyzError`, so `except oxyz.OxyzError`
catches the lot while `except ValueError` still works (OxyzError is a
ValueError).
"""

from __future__ import annotations

import importlib.util

import pytest

import oxyz


def test_oxyz_error_is_a_value_error() -> None:
    assert issubclass(oxyz.OxyzError, ValueError)
    assert "OxyzError" in oxyz.__all__


def test_top_level_errors_subclass_oxyz_error() -> None:
    assert issubclass(oxyz.ParseError, oxyz.OxyzError)
    assert issubclass(oxyz.SchemaError, oxyz.OxyzError)


@pytest.mark.skipif(importlib.util.find_spec("ase") is None, reason="ase not installed")
def test_ase_errors_subclass_oxyz_error() -> None:
    import oxyz.ase

    assert issubclass(oxyz.ase.ToAseError, oxyz.OxyzError)
    assert issubclass(oxyz.ase.FromAtomsError, oxyz.OxyzError)


@pytest.mark.skipif(
    importlib.util.find_spec("metatomic.torch") is None,
    reason="metatomic not installed",
)
def test_metatomic_error_subclasses_oxyz_error() -> None:
    import oxyz.metatomic

    assert issubclass(oxyz.metatomic.ToSystemError, oxyz.OxyzError)


@pytest.mark.skipif(
    importlib.util.find_spec("torch_sim") is None, reason="torch_sim not installed"
)
def test_torch_sim_error_subclasses_oxyz_error() -> None:
    import oxyz.torch_sim

    assert issubclass(oxyz.torch_sim.ToSimStateError, oxyz.OxyzError)


def test_parse_error_is_caught_as_oxyz_error(tmp_path) -> None:
    broken = tmp_path / "broken.xyz"
    broken.write_text("not-a-count\n")
    with pytest.raises(oxyz.OxyzError):
        oxyz.read(broken)
