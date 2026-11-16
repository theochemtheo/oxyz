"""Check local oxyz workflows; report optional skips and fail on wrong results."""

# Standalone smoke assertions and diagnostic output are intentional.
# ruff: noqa: INP001, T201, S101, PLR2004

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

import oxyz
from oxyz import Column, Record, RecordBatch

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "tests" / "data"
VARYING = DATA / "varying_atom_counts.xyz"
SIMPLE = DATA / "simple.extxyz"
CONFORMANT = DATA / "schema_conformant.extxyz"
MIXED = DATA / "mixed_schema_optional_column.xyz"

type Array = npt.NDArray[np.float64]


class Sample(Record):
    """Positions and nullable per-atom charge."""

    positions: Column[Array] = Column("pos", width=3)
    charge: Column[Array | None] = Column()


class Atom(Record):
    """Shared positions for key-discriminated record members."""

    positions: Column[Array] = Column("pos", width=3)


class Bare(Atom):
    """An atom frame without charge."""


class Charged(Atom):
    """An atom frame with required charge."""

    charge: Column[Array] = Column()


def core() -> None:
    """Check local selections, streaming, structural counts and batch layout."""
    frames = oxyz.read(VARYING)
    assert [frame.n_atoms for frame in frames] == [3, 1, 2]
    assert frames[0].columns["species"] == ["O", "H", "H"]
    assert frames[0].metadata["energy"] == -76.3
    assert oxyz.read(VARYING, 0) == frames[0]
    assert oxyz.read(VARYING, "::2") == [frames[0], frames[2]]
    assert oxyz.read(VARYING, [2, 0, 2]) == [frames[2], frames[0], frames[2]]
    assert oxyz.read(VARYING, threads=1) == frames
    assert list(oxyz.iread(VARYING)) == frames
    index = oxyz.scan(VARYING)
    assert (index.n_frames, index.total_atoms) == (3, 6)

    batch = oxyz.read_batch(VARYING)
    assert (batch.n_frames, batch.total_atoms) == (3, 6)
    assert batch.ptr.tolist() == [0, 3, 4, 6]
    assert batch.frame_indices.tolist() == [0, 1, 2]
    np.testing.assert_array_equal(
        batch.columns["pos"], np.concatenate([f.columns["pos"] for f in frames])
    )
    windows = list(oxyz.iread_batch(VARYING, frames_per_batch=2))
    assert [b.frame_indices.tolist() for b in windows] == [[0, 1], [2]]
    shuffled = list(oxyz.iread_batch(VARYING, frames_per_batch=2, shuffle=True, seed=0))
    assert sorted(i for b in shuffled for i in b.frame_indices.tolist()) == [0, 1, 2]
    print("PASS reads/indexing/iteration/batching: 3 frames, 6 atoms; ptr=[0,3,4,6]")


def compressed() -> None:
    """Compare codec results and an explicit archive member."""
    directory = DATA / "compressed"
    reference = oxyz.read(directory / "two_frame.xyz.gz")
    assert len(reference) == 2
    for name in (
        "two_frame.xyz.zst",
        "two_frame.xyz.zip",
        "two_frame.tar",
        "two_frame.tar.gz",
        "two_frame.tar.zst",
    ):
        assert oxyz.read(directory / name) == reference
    member = directory / "multi_member.zip"
    selected = oxyz.read(member, member="a.xyz")
    assert oxyz.scan(member, member="a.xyz").n_frames == len(selected)
    assert selected == reference
    print("PASS compressed reads: gzip/zstd/zip/tar/tar.gz/tar.zst and ZIP member")


def schema_tour() -> None:
    """Check inference, validation failure context and projection fills."""
    observed = oxyz.infer_schema(MIXED)
    assert not observed.is_consistent
    assert observed["charge"].frames_present == 1
    inferred = oxyz.infer_schema(CONFORMANT).to_spec()
    assert oxyz.read(CONFORMANT, schema=inferred, conformance="strict") == oxyz.read(
        CONFORMANT
    )
    spec = oxyz.SchemaSpec(
        columns=(
            oxyz.ColumnRule("species", oxyz.Kind.STR),
            oxyz.ColumnRule("pos", oxyz.Kind.REAL, width=3),
            oxyz.ColumnRule("magmom", oxyz.Kind.REAL, width=3, required=False),
        ),
        metadata=(oxyz.MetadataRule("energy", oxyz.Kind.REAL),),
    )
    try:
        oxyz.read(DATA / "schema_drift_type.extxyz", schema=spec)
    except oxyz.SchemaError as error:
        assert (error.frame_index, error.name) == (1, "magmom")  # noqa: PT017
    else:
        raise AssertionError("validation accepted a wrong-width magmom")
    project = oxyz.SchemaSpec.from_dict(
        {
            "columns": {
                "species": {"kind": "S"},
                "pos": {"kind": "R", "width": 3},
                "charge": {"kind": "R", "required": False},
            },
            "mode": "project",
        }
    )
    batch = oxyz.read_batch(MIXED, schema=project)
    assert batch.ptr.tolist() == [0, 2, 4]
    np.testing.assert_array_equal(batch.columns["charge"][:2], [0.5, -0.5])
    assert np.isnan(batch.columns["charge"][2:]).all()
    print("PASS schema inference/validation/projection: frame 1 magmom; NaN fills")


def records() -> None:
    """Check nullable records, union identity, dense fills and presence masks."""
    samples = oxyz.read(MIXED, record=Sample)
    assert samples[1].charge is None
    np.testing.assert_array_equal(samples[0].charge, [0.5, -0.5])
    assert Sample.from_frame(oxyz.read(MIXED, 0)) == samples[0]
    batch = oxyz.read_batch(MIXED, record=RecordBatch[Sample])
    assert batch.present(Sample.charge).tolist() == [True, False]
    assert np.isnan(batch[Sample.charge][2:]).all()
    assert list(batch) == samples

    members = Bare | Charged
    rows = oxyz.read(MIXED, record=members)
    assert [type(row) for row in rows] == [Charged, Bare]
    assert list(oxyz.iread(MIXED, record=members)) == rows
    union = oxyz.read_batch(MIXED, record=RecordBatch[Bare | Charged])
    assert union.members == (Bare, Charged)
    assert union.tags.tolist() == [1, 0]
    assert union.present(Charged.charge).tolist() == [True, False]
    assert union.where(Bare).tolist() == [False, True]
    np.testing.assert_array_equal(union[Charged.charge][:2], [0.5, -0.5])
    assert np.isnan(union[Charged.charge][2:]).all()
    assert list(union) == rows
    selected = union[[1, 0]]
    assert selected.frame_indices.tolist() == [1, 0]
    assert selected.tags.tolist() == [0, 1]
    assert selected.present(Charged.charge).tolist() == [False, True]
    print("PASS typed records/union batches: nullable charge, tags, masks, selection")


def write_tour(directory: Path) -> None:
    """Check plain, gzip and incremental writer round trips."""
    frames = oxyz.read(VARYING)
    for name in ("out.extxyz", "out.xyz.gz"):
        path = directory / name
        oxyz.write(path, frames)
        assert oxyz.read(path) == frames
    path = directory / "incremental.extxyz"
    with oxyz.Writer(path) as writer:
        for frame in frames:
            writer.write(frame)
    assert oxyz.read(path) == frames
    print("PASS write/Writer: full frame equality after plain/gzip/incremental writes")


def cli_tour(directory: Path) -> None:
    """Check CLI statistics and read the emitted and frozen schemas."""

    def run(*args: str) -> str:
        done = subprocess.run(  # noqa: S603 — fixed local argv, no shell
            [sys.executable, "-m", "oxyz", *args],
            capture_output=True,
            text=True,
            check=True,
            cwd=directory,
        )
        return done.stdout

    stats = json.loads(run("scan", str(VARYING), "--no-schema", "--json"))["stats"]
    assert (stats["n_frames"], stats["total_atoms"]) == (3, 6)
    schema = directory / "schema.yaml"
    run("scan", str(CONFORMANT), "--emit-schema", str(schema))
    spec = oxyz.SchemaSpec.from_file(schema)
    assert spec == oxyz.infer_schema(CONFORMANT).to_spec()
    report = json.loads(
        run("check", str(CONFORMANT), "--schema", str(schema), "--json")
    )
    assert report["n_frames"] == 2
    assert report["violations"] == []
    frozen_path = directory / "frozen.yaml"
    run("freeze", str(CONFORMANT), "--schema", str(schema), "--out", str(frozen_path))
    frozen = oxyz.SchemaSpec.from_file(frozen_path)
    assert frozen == spec.freeze(CONFORMANT)
    assert frozen.mode == "project"
    assert oxyz.read(CONFORMANT, schema=frozen) == oxyz.read(CONFORMANT)
    print("PASS CLI scan/check/freeze: JSON counts, emitted spec and projected values")


def ase_tour() -> None:
    """Check ASE structures and selected iteration."""
    import oxyz.ase

    atoms = oxyz.ase.read(SIMPLE)
    assert atoms.get_chemical_formula() == "H"
    assert atoms.get_potential_energy() == -1.0
    np.testing.assert_array_equal(atoms.positions, [[0.0, 0.0, 0.0]])
    assert [len(a) for a in oxyz.ase.read(VARYING, ":")] == [3, 1, 2]
    assert [len(a) for a in oxyz.ase.iread(VARYING, "::2")] == [3, 2]


def metatomic_tour() -> None:
    """Check metatomic structure and target extraction."""
    import oxyz.metatomic

    system = oxyz.metatomic.read(SIMPLE, index=0)
    assert system.types.tolist() == [1]
    assert system.positions.tolist() == [[0.0, 0.0, 0.0]]
    source = oxyz.metatomic.SystemSource(SIMPLE)
    assert len(source.systems()) == 1
    assert source.per_config("energy").reshape(-1).tolist() == [-1.0]


def torch_sim_tour() -> None:
    """Check torch_sim structures, batches and source access."""
    import oxyz.torch_sim as ts

    state = ts.read(VARYING)
    assert state.n_systems == 3
    assert state.positions.shape == (6, 3)
    np.testing.assert_array_equal(
        state.positions.cpu().numpy(), oxyz.read_batch(VARYING).columns["pos"]
    )
    assert [b.n_systems for b in ts.iread(VARYING, frames_per_batch=2)] == [2, 1]
    source = ts.SimStateSource(VARYING)
    assert len(source) == 3
    assert source.state().n_systems == 3


def main() -> int:
    """Run the local tour and report exactly which optional targets ran."""
    if not __debug__:
        raise RuntimeError("the smoke tour requires assertions; run without -O")
    print(f"repo: {ROOT}")
    core()
    compressed()
    schema_tour()
    records()
    with tempfile.TemporaryDirectory(prefix="oxyz-tour-") as tmp:
        directory = Path(tmp)
        write_tour(directory)
        cli_tour(directory)
    passed: list[str] = []
    skipped: list[str] = []
    tours: tuple[tuple[str, tuple[str, ...], Callable[[], None]], ...] = (
        ("ase", ("ase",), ase_tour),
        ("metatomic", ("torch", "metatomic"), metatomic_tour),
        ("torch_sim", ("torch", "torch_sim"), torch_sim_tour),
    )
    for name, dependencies, tour in tours:
        missing = [dep for dep in dependencies if find_spec(dep) is None]
        if missing:
            skipped.append(name)
            print(f"SKIP {name}: missing {', '.join(missing)}")
        else:
            tour()
            passed.append(name)
            print(f"PASS optional {name}")
    print(
        "\nOK: 6 local workflow groups; "
        f"optional passed={','.join(passed) or 'none'}; "
        f"skipped={','.join(skipped) or 'none'}"
    )
    print("NOT COVERED: live remote sources, full suites, wheel tiers, benchmarks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
