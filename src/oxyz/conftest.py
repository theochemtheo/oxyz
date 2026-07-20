"""Pytest configuration for the ``src/oxyz`` doctests.

This conftest sits beside the package so it applies to the ``--doctest-modules``
run over ``src/oxyz``. It is deliberately not at the repository root: a top-level
``conftest.py`` would shadow ``benchmarks/conftest.py`` for the benchmark suite's
bare ``import conftest``. It is excluded from the built wheel (see ``[tool.maturin]``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

_EXAMPLES_DATA = Path(__file__).resolve().parent.parent.parent / "examples" / "data"


@pytest.fixture(autouse=True)
def _doctest_data(doctest_namespace: dict[str, object]) -> None:
    """Expose the committed example data to doctests as ``DATA``.

    Doctests reference ``DATA / "water.extxyz"`` rather than a repo-root-relative
    string, so they resolve the file by absolute path and pass regardless of the
    directory ``pytest`` runs from.
    """
    doctest_namespace["DATA"] = _EXAMPLES_DATA
