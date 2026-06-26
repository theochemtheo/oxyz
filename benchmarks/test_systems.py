"""System / target benchmarks: oxyz.metatomic vs the ASE path it replaces.

Two workloads, mirroring test_read.py's tagging and `run` helper:

- `systems/*`  builds `metatomic.torch.System`s — oxyz.metatomic.read (thread
  sweep) and iread, against systems_to_torch(ase.io.read(...)), the conversion
  metatrain uses today.
- `targets/metadata_heavy`  extracts energy + forces + stress as torch tensors.
  oxyz parses once via SystemSource; the ASE baseline re-reads the file per
  quantity (metatrain's pattern), so this row measures the repeat-read saving.

Gated on torch + metatomic-torch; the env in run.py supplies CPU wheels.
"""

from __future__ import annotations

import importlib.util

import pytest
from test_read import THREAD_SWEEP, needs_ase, row, run

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("metatomic") is None,
    reason="torch / metatomic-torch not installed",
)


def oxyz_read_systems_with(threads: int):
    @row("metatomic System", "serial" if threads == 1 else "parallel", threads=threads)
    def read(path):
        import torch

        import oxyz.metatomic as om

        return om.read(path, dtype=torch.float64, threads=threads)

    return read


@row("metatomic System", "serial")
def oxyz_iread_systems(path):
    import torch

    import oxyz.metatomic as om

    # Collected so it does the same total work as the eager rows.
    return list(om.iread(path, dtype=torch.float64))


@row("ase→System", "serial")
def ase_systems_to_torch(path):
    import torch
    from ase.io import read
    from metatomic.torch import systems_to_torch

    frames = read(path, index=":", format="extxyz")
    assert isinstance(frames, list)
    # ase.Atoms is not declared to satisfy metatomic's abstract IntoSystem.
    return systems_to_torch(frames, dtype=torch.float64)  # ty: ignore[invalid-argument-type]


SYSTEMS = [
    *(pytest.param(oxyz_read_systems_with(t), id=f"oxyz-{t}t") for t in THREAD_SWEEP),
    pytest.param(oxyz_iread_systems, id="oxyz-iter"),
    pytest.param(ase_systems_to_torch, id="ase-to-system", marks=needs_ase),
]


@pytest.mark.benchmark(group="systems/many_small_frames")
@pytest.mark.parametrize("read", SYSTEMS)
def test_systems_many_small_frames(benchmark, read, many_small_frames):
    systems = run(benchmark, read, many_small_frames)
    assert len(systems) == 2_000


@pytest.mark.benchmark(group="systems/large_frames")
@pytest.mark.parametrize("read", SYSTEMS)
def test_systems_large_frames(benchmark, read, large_frames):
    systems = run(benchmark, read, large_frames)
    assert len(systems) == 4


@pytest.mark.benchmark(group="systems/mace_mixed")
@pytest.mark.parametrize("read", SYSTEMS)
def test_systems_mace_mixed(benchmark, read, mace_mixed):
    systems = run(benchmark, read, mace_mixed)
    assert len(systems) == 1_000


@row("metatomic targets", "parallel")
def oxyz_targets(path):
    import torch

    import oxyz.metatomic as om

    source = om.SystemSource(path)
    energy = source.per_config("energy", dtype=torch.float64)
    forces, _ = source.per_atom("forces", dtype=torch.float64)
    stress = source.per_config("stress", dtype=torch.float64)
    return energy, forces, stress


@row("ase targets", "serial")
def ase_targets(path):
    import torch
    from ase.io import read

    def frames_with_results():
        # metatrain's read() wrapper: surface calculator results in info/arrays.
        frames = read(path, index=":", format="extxyz")
        assert isinstance(frames, list)
        for atoms in frames:
            if atoms.calc is not None:
                results = atoms.calc.results
                if "energy" in results:
                    atoms.info["energy"] = results["energy"]
                if "forces" in results:
                    atoms.arrays["forces"] = results["forces"]
                if "stress" in results:
                    atoms.info["stress"] = results["stress"]
        return frames

    # One file read per quantity, as metatrain's _read_*_ase functions do.
    energy = torch.tensor(
        [[a.info["energy"]] for a in frames_with_results()], dtype=torch.float64
    )
    forces = [
        torch.tensor(a.arrays["forces"], dtype=torch.float64)
        for a in frames_with_results()
    ]
    stress = [
        torch.tensor(a.info["stress"], dtype=torch.float64)
        for a in frames_with_results()
    ]
    return energy, forces, stress


@pytest.mark.benchmark(group="targets/metadata_heavy")
@pytest.mark.parametrize(
    "extract",
    [
        pytest.param(oxyz_targets, id="oxyz-one-pass"),
        pytest.param(ase_targets, id="ase-per-quantity", marks=needs_ase),
    ],
)
def test_targets_metadata_heavy(benchmark, extract, metadata_heavy):
    energy, forces, stress = run(benchmark, extract, metadata_heavy)
    # The two paths return different containers by design — oxyz's array-native
    # one-pass (stacked tensors) vs metatrain's per-frame per-quantity reads — so
    # forces (atom-major tensor vs per-frame list) can't share a length check;
    # energy and stress are per-config, so both expose one entry per frame.
    assert len(energy) == 2_000
    assert len(stress) == 2_000
    assert forces is not None
