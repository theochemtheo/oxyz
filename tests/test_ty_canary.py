"""Canary: fails when ty fixes the limitation we work around.

The ``as_array`` helper in test_extxyz.py exists only because ty currently
mis-handles isinstance-narrowing of ``np.ndarray`` out of a union: the
narrowed type fails numpy's ``assert_allclose`` overloads even though a plain
``np.ndarray`` parameter passes (observed in ty 0.0.44 and 0.0.46).

When this test fails, ty has fixed the issue: delete this file and the
``as_array`` helper.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

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
    ty = shutil.which("ty")
    assert ty is not None, "ty not on PATH; run via `uv run pytest`"

    repro = tmp_path / "repro.py"
    repro.write_text(REPRO)

    result = subprocess.run(
        [ty, "check", "--python", sys.prefix, str(repro)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr

    # Guard against a broken run masquerading as "fixed".
    assert "unresolved-import" not in output, f"ty could not see numpy:\n{output}"

    assert "no-matching-overload" in output, (
        "ty no longer flags assert_allclose on an isinstance-narrowed union — "
        "the workaround is obsolete. Delete this file and the as_array helper "
        f"in test_extxyz.py.\n\nty output:\n{output}"
    )
