"""Canary: fails once ty can narrow ``np.ndarray`` out of a union.

Then delete this file and ``as_array`` in test_extxyz.py.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

REPRO = """\
import numpy as np
from numpy.testing import assert_allclose


def narrows_union(x: np.ndarray | list[str]) -> None:
    assert isinstance(x, np.ndarray)
    assert_allclose(x, np.zeros(3))
"""


def test_ty_still_rejects_isinstance_narrowed_ndarray(tmp_path: Path) -> None:
    # A type-checker canary, so a tier without ty skips rather than fails.
    ty = shutil.which("ty")
    if ty is None:
        pytest.skip("ty not on PATH; run via `uv run pytest`")

    repro = tmp_path / "repro.py"
    repro.write_text(REPRO)

    result = subprocess.run(
        [ty, "check", "--python", sys.prefix, str(repro)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr

    assert "unresolved-import" not in output, f"ty could not see numpy:\n{output}"

    assert "no-matching-overload" in output, (
        "ty no longer flags assert_allclose on an isinstance-narrowed union — "
        "the workaround is obsolete. Delete this file and the as_array helper "
        f"in test_extxyz.py.\n\nty output:\n{output}"
    )
