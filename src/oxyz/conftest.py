"""Doctest configuration for ``src/oxyz``.

Not at the repository root, where it would shadow ``benchmarks/conftest.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_EXAMPLES_DATA = Path(__file__).resolve().parent.parent.parent / "examples" / "data"


@pytest.fixture(autouse=True)
def _doctest_data(doctest_namespace: dict[str, object]) -> None:
    """Expose the example data to doctests as ``DATA``, an absolute path."""
    doctest_namespace["DATA"] = _EXAMPLES_DATA
