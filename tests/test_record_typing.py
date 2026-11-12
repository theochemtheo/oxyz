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

import oxyz
from oxyz import Cell, Column, Matrix3, Meta, Record, RecordBatch

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
p = "frames.xyz"
"""

REVEALS = """\
reveal_type(r.n_atoms)            # T: int
reveal_type(r.energy)             # T: float
reveal_type(r.forces)             # T: ndarray[_AnyShape, dtype[float64]]
reveal_type(r.stress)             # T: ndarray[_AnyShape, dtype[float64]] | None
reveal_type(MaceSample.energy)    # T: Meta[float]
reveal_type(MaceSample.forces)    # T: Column[Array]
reveal_type(oxyz.read(p, record=MaceSample))          # T: list[MaceSample]
reveal_type(oxyz.read(p, 0, record=MaceSample))       # T: MaceSample
reveal_type(oxyz.read(p, "0", record=MaceSample))     # T: MaceSample | list[MaceSample]
reveal_type(oxyz.read(p, [0], record=MaceSample))     # T: list[MaceSample]
reveal_type(oxyz.iread(p, record=MaceSample))         # T: Iterator[MaceSample]
reveal_type(MaceSample.from_frame(oxyz.read(p, 0)))   # T: MaceSample
reveal_type(oxyz.read(p))                             # T: list[Frame]
reveal_type(oxyz.iread(p))                            # T: Iterator[Frame]
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
oxyz.read(p, record=MaceSample, mode="project")                  # error
oxyz.iread(p, record=MaceSample, mode="project")                 # error
oxyz.read(p, record=r)                                           # error
# ty applies no class field specifier, so a default-form field reads as optional.
MaceSample(n_atoms=3, positions=a, forces=a, stress=None, cell=a, config_type="x")
"""

BATCHES = """\
class MaceBatch(RecordBatch[MaceSample]):
    positions: Array
    forces: Array
    energy: Array
    stress: Array
    cell: Array
    config_type: list[str]


type Ints = npt.NDArray[np.int64]
type Bools = npt.NDArray[np.bool_]
ints: Ints = np.zeros(1, dtype=np.int64)


class Table(Record):
    rc: Column[Array] = Column(width=3)
    rc_o: Column[Array | None] = Column()
    ic: Column[Ints] = Column()
    ic_o: Column[Ints | None] = Column()
    lc: Column[Bools] = Column()
    lc_o: Column[Bools | None] = Column()
    sc: Column[list[str]] = Column()
    sc_o: Column[list[str] | None] = Column()
    sc2: Column[list[list[str]]] = Column(width=2)
    sc2_o: Column[list[list[str]] | None] = Column(width=2)
    rm: Meta[float] = Meta()
    rm_o: Meta[float | None] = Meta()
    im: Meta[int] = Meta()
    im_o: Meta[int | None] = Meta()
    lm: Meta[bool] = Meta()
    lm_o: Meta[bool | None] = Meta()
    sm: Meta[str] = Meta()
    sm_o: Meta[str | None] = Meta()
    ra: Meta[Array] = Meta(shape=(3,))
    ra_o: Meta[Array | None] = Meta(shape=(3,))
    ia: Meta[Ints] = Meta(shape=(3,))
    ia_o: Meta[Ints | None] = Meta(shape=(3,))
    la: Meta[Bools] = Meta(shape=(3,))
    la_o: Meta[Bools | None] = Meta(shape=(3,))
    sa: Meta[list[str]] = Meta(shape=(2,))
    sa_o: Meta[list[str] | None] = Meta(shape=(2,))
    cell: Cell[Array] = Cell()
    m3: Matrix3[Array] = Matrix3()
    m3_o: Matrix3[Array | None] = Matrix3("virial")
    f32: Column[npt.NDArray[np.float32]] = Column()


class NoCell(Record):
    cell: Cell[Array | None] = Cell()


def batches(
    b: RecordBatch[MaceSample],
    m: MaceBatch,
    t: RecordBatch[Table],
    n: RecordBatch[NoCell],
) -> None:
"""

FLOAT = "ndarray[_AnyShape, dtype[float64]]"
INT = "ndarray[_AnyShape, dtype[signedinteger[_64Bit]]]"
BOOL = "ndarray[_AnyShape, dtype[numpy.bool[builtins.bool]]]"
GENERIC = "RecordBatch[MaceSample]"

BATCH_REVEALS = f"""\
    reveal_type(b[0])                          # T: MaceSample
    reveal_type(b[np.int64(0)])                # T: MaceSample
    reveal_type(b[0:1])                        # T: RecordBatch[MaceSample]
    reveal_type(b[[1, 0]])                     # T: RecordBatch[MaceSample]
    for r in b:
        reveal_type(r)                         # T: MaceSample
    reveal_type(b["x"])                        # T: Any
    reveal_type(b.present(MaceSample.stress))  # T: {BOOL}
    reveal_type(b[MaceSample.energy])          # T: {FLOAT}
    reveal_type(m[0])                          # T: MaceSample
    reveal_type(m[0:1])                        # T: MaceBatch
    reveal_type(m.energy)                      # T: {FLOAT}
    reveal_type(m.config_type)                 # T: list[str]
    reveal_type(t[Table.rc])                   # T: {FLOAT}
    reveal_type(t[Table.rc_o])                 # T: {FLOAT}
    reveal_type(t[Table.ic])                   # T: {INT}
    reveal_type(t[Table.ic_o])                 # T: {INT}
    reveal_type(t[Table.lc])                   # T: {BOOL}
    reveal_type(t[Table.lc_o])                 # T: {BOOL}
    reveal_type(t[Table.sc])                   # T: list[str]
    reveal_type(t[Table.sc_o])                 # T: list[str]
    reveal_type(t[Table.sc2])                  # T: list[list[str]]
    reveal_type(t[Table.sc2_o])                # T: list[list[str]]
    reveal_type(t[Table.rm])                   # T: {FLOAT}
    reveal_type(t[Table.rm_o])                 # T: {FLOAT}
    reveal_type(t[Table.im])                   # T: {INT}
    reveal_type(t[Table.im_o])                 # T: {INT}
    reveal_type(t[Table.lm])                   # T: {BOOL}
    reveal_type(t[Table.lm_o])                 # T: {BOOL}
    reveal_type(t[Table.sm])                   # T: list[str]
    reveal_type(t[Table.sm_o])                 # T: list[str]
    reveal_type(t[Table.ra])                   # T: {FLOAT}
    reveal_type(t[Table.ra_o])                 # T: {FLOAT}
    reveal_type(t[Table.ia])                   # T: {INT}
    reveal_type(t[Table.ia_o])                 # T: {INT}
    reveal_type(t[Table.la])                   # T: {BOOL}
    reveal_type(t[Table.la_o])                 # T: {BOOL}
    reveal_type(t[Table.sa])                   # T: list[list[str]]
    reveal_type(t[Table.sa_o])                 # T: list[list[str]]
    reveal_type(t[Table.cell])                 # T: {FLOAT}
    reveal_type(n[NoCell.cell])                # T: {FLOAT}
    reveal_type(t[Table.m3])                   # T: {FLOAT}
    reveal_type(t[Table.m3_o])                 # T: {FLOAT}
    reveal_type(t[Table.f32])                  # T: Any
    reveal_type(oxyz.read_batch(p, record={GENERIC}))  # T: {GENERIC}
    reveal_type(oxyz.read_batch(p, "0:2", record=MaceBatch))   # T: MaceBatch
    reveal_type(oxyz.iread_batch(p, record=MaceBatch))         # T: Iterator[MaceBatch]
    reveal_type(oxyz.read_batch(p))                            # T: Batch
    reveal_type(oxyz.iread_batch(p))                           # T: Iterator[Batch]
    reveal_type(MaceBatch.from_batch(oxyz.read_batch(p)))      # T: MaceBatch
    reveal_type(MaceBatch.spec())                              # T: SchemaSpec
    reveal_type(RecordBatch[MaceSample].from_batch(oxyz.read_batch(p)))  # T: {GENERIC}
    reveal_type(RecordBatch[MaceSample].spec())                # T: SchemaSpec
"""

BATCH_ERRORS = """\
    b.energy                                                     # error
    m.energy = m.energy                                          # error
    MaceBatch(offsets=ints, frame_indices=ints)                  # error
    oxyz.read_batch(p, record=MaceSample)                        # error
    oxyz.iread_batch(p, record=MaceSample, frames_per_batch=2)   # error
    oxyz.read(p, record=MaceBatch)                               # error
    oxyz.read_batch(p, record=MaceBatch, mode="project")         # error
    oxyz.iread_batch(p, record=MaceBatch, mode="project", frames_per_batch=2)  # error


class Bad(RecordBatch[int]): pass                                # error
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


def test_batch_lookups_have_batch_form_types(tmp_path: Path):
    check_fixture(tmp_path, BATCHES + BATCH_REVEALS)


def test_misuse_of_a_batch_is_a_static_error(tmp_path: Path):
    check_fixture(tmp_path, BATCHES + BATCH_ERRORS)
