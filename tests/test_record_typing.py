"""ty sees a record's fields with their value types, and its keys as specifiers."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

DECLARATION = """\
from typing import Annotated, reveal_type

import numpy as np
import numpy.typing as npt

from oxyz import Cell, Column, Meta, Record

type Array = npt.NDArray[np.float64]
type Forces = Annotated[Column[Array], Column("REF_forces", width=3)]


class MaceSample(Record):
    positions: Column[Array] = Column("pos", width=3)
    forces: Forces
    energy: Meta[float] = Meta("REF_energy")
    stress: Meta[Array | None] = Meta("REF_stress", shape=(6,))
    cell: Cell[Array] = Cell()
    config_type: Meta[str] = Meta()


a: Array = np.zeros((3, 3))
r = MaceSample(
    n_atoms=3, positions=a, forces=a, energy=1.0, stress=None, cell=a, config_type="x"
)
"""

REVEALS = """\
reveal_type(r.n_atoms)            # T: int
reveal_type(r.energy)             # T: float
reveal_type(r.forces)             # T: ndarray[_AnyShape, dtype[float64]]
reveal_type(r.stress)             # T: ndarray[_AnyShape, dtype[float64]] | None
reveal_type(MaceSample.energy)    # T: Meta[float]
reveal_type(MaceSample.forces)    # T: Column[Array]
"""

ERRORS = """\
r.energy = 2.0                                                   # error
MaceSample(n_atoms=3)                                            # error
MaceSample(
    n_atoms=3, positions=a, forces=a, stress=None, cell=a, config_type="x",
    energy="x",                                                  # error
)
MaceSample.typo                                                  # error
class P(Record): energy: float = Meta()                          # error
class Q(Record): energy: Meta[float] = Column()                  # error
# ty applies no class field specifier, so a default-form field reads as optional.
MaceSample(n_atoms=3, positions=a, forces=a, stress=None, cell=a, config_type="x")
"""

DIAGNOSTIC = re.compile(r"^(?P<path>.+?):(?P<line>\d+):\d+: (?P<message>.+)$")


def ty_check(tmp_path: Path, source: str) -> dict[int, list[str]]:
    """Run `ty check --output-format concise`; diagnostics by 1-based line."""
    ty = shutil.which("ty")
    if ty is None:
        pytest.skip("ty not on PATH; run via `uv run pytest`")
    path = tmp_path / "fixture.py"
    path.write_text(source)
    result = subprocess.run(
        [ty, "check", "--output-format", "concise", "--python", sys.prefix, str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    assert "unresolved-import" not in output, f"ty could not see oxyz:\n{output}"
    found: dict[int, list[str]] = {}
    for line in output.splitlines():
        match = DIAGNOSTIC.match(line)
        if match and match["path"] == str(path):
            found.setdefault(int(match["line"]), []).append(match["message"])
    return found


def check_fixture(tmp_path: Path, body: str) -> None:
    source = DECLARATION + body
    found = ty_check(tmp_path, source)
    for number, line in enumerate(source.splitlines(), start=1):
        messages = found.pop(number, [])
        if "# T: " in line:
            expected = f"`{line.split('# T: ', 1)[1]}`"
            assert any(
                m.startswith("info[revealed-type]") and expected in m for m in messages
            ), f"line {number}: expected {expected}, got {messages}"
        elif line.endswith("# error"):
            assert any(m.startswith("error[") for m in messages), (
                f"line {number}: expected an error, got {messages}"
            )
        else:
            assert not messages, f"line {number}: unexpected {messages}"
    assert not found, f"diagnostics past the source: {found}"


def test_fields_have_value_types_and_keys_have_specifier_types(tmp_path: Path):
    check_fixture(tmp_path, REVEALS)


def test_misuse_of_a_record_is_a_static_error(tmp_path: Path):
    check_fixture(tmp_path, ERRORS)
