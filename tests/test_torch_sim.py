"""Parity tests: oxyz.torch_sim must agree with atoms_to_state(ase.io.read).

A batched `SimState`'s core fields (positions/masses/cell/pbc/atomic_numbers/
system_idx) are compared against `torch_sim.io.atoms_to_state` fed by
`ase.io.read(index=':')`, the path oxyz.torch_sim replaces. The few fixtures
ASE's own reader rejects (its Voigt/stress-shape limitation) are excluded, as in
the ase/metatomic golden sets.
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("ase") is None
    or importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("torch_sim") is None,
    reason="requires ase, torch, and torch-sim-atomistic",
)

DATA_DIR = Path(__file__).parent / "data"

# Files ase.io.read rejects (a stress shape ASE's reader will not accept), plus
# the schema-conformance fixtures that vary `magmom` width or per-atom columns
# between frames on purpose — oxyz's own batch reader requires a uniform
# schema across frames, so these can't build a reference either way.
ASE_REJECTS = {
    "simple.extxyz",
    "nonorthogonal.extxyz",
    "stress_voigt6.extxyz",
    "schema_drift_type.extxyz",
    "schema_extra_column.extxyz",
    "mixed_schema_optional_column.xyz",
}

GOLDEN = sorted(
    path.name
    for path in list(DATA_DIR.glob("*.xyz")) + list(DATA_DIR.glob("*.extxyz"))
    if path.name not in ASE_REJECTS
)


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "frame.extxyz"
    path.write_text(body)
    return path


def assert_state_parity(got, ref) -> None:
    import torch

    assert torch.equal(got.atomic_numbers, ref.atomic_numbers)
    assert torch.equal(got.system_idx, ref.system_idx)
    assert torch.equal(got.pbc, ref.pbc)
    assert got.positions.dtype == ref.positions.dtype
    assert torch.allclose(got.positions, ref.positions)
    assert torch.allclose(got.masses, ref.masses)
    assert torch.allclose(got.cell, ref.cell)
    assert got.cell.shape == ref.cell.shape


@pytest.mark.parametrize("name", GOLDEN)
@pytest.mark.parametrize("dtype_name", ["default", "float64"])
def test_state_parity_with_atoms_to_state(name: str, dtype_name: str) -> None:
    import ase.io
    import torch
    from torch_sim.io import atoms_to_state

    import oxyz.torch_sim

    dtype = None if dtype_name == "default" else torch.float64
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = oxyz.torch_sim.read(DATA_DIR / name, dtype=dtype)
        atoms = ase.io.read(DATA_DIR / name, index=":")
        ref = atoms_to_state(atoms, dtype=dtype)
    assert_state_parity(got, ref)


def test_read_returns_one_batched_state_not_a_list() -> None:
    import torch
    from torch_sim import SimState

    import oxyz.torch_sim

    state = oxyz.torch_sim.read(DATA_DIR / "varying_atom_counts.xyz")
    assert isinstance(state, SimState)
    assert state.n_systems == 3
    assert state.system_idx.tolist() == [0, 0, 0, 1, 2, 2]
    assert state.cell.shape == (3, 3, 3)
    assert state.cell.dtype == torch.float64  # dtype=None infers, like atoms_to_state


def test_int_index_yields_single_system_state() -> None:
    import oxyz.torch_sim

    state = oxyz.torch_sim.read(DATA_DIR / "varying_atom_counts.xyz", 1)
    assert state.n_systems == 1
    assert state.n_atoms == 1


def test_negative_index_resolves_from_the_end() -> None:
    import oxyz.torch_sim

    state = oxyz.torch_sim.read(DATA_DIR / "varying_atom_counts.xyz", -1)
    assert state.n_systems == 1
    assert state.n_atoms == 2  # the last frame has 2 atoms


def test_slice_selects_a_subset_into_one_state() -> None:
    import oxyz.torch_sim

    state = oxyz.torch_sim.read(DATA_DIR / "varying_atom_counts.xyz", "0:2")
    assert state.n_systems == 2
    assert state.n_atoms == 4


def test_cell_is_column_convention_transpose_of_lattice(tmp_path: Path) -> None:
    import torch

    import oxyz.torch_sim

    # An asymmetric cell makes the transpose observable. extxyz Lattice is the
    # ASE (row-vector) cell in Fortran order; torch_sim stores its transpose.
    path = _write(
        tmp_path,
        '1\nLattice="1 2 3 4 5 6 7 8 9" Properties=species:S:1:pos:R:3\nH 0 0 0\n',
    )
    state = oxyz.torch_sim.read(path, dtype=torch.float64)
    ase_cell = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9]).reshape((3, 3), order="F").T
    expected = torch.tensor(ase_cell.T)  # torch_sim column convention
    assert torch.allclose(state.cell[0], expected)


def test_masses_column_wins_over_derived(tmp_path: Path) -> None:
    import torch

    import oxyz.torch_sim

    path = _write(
        tmp_path,
        "2\nProperties=species:S:1:pos:R:3:masses:R:1\nH 0 0 0 2.5\nO 1 0 0 9.0\n",
    )
    state = oxyz.torch_sim.read(path, dtype=torch.float64)
    assert state.masses.tolist() == [2.5, 9.0]


def test_masses_derived_from_atomic_numbers_when_no_column() -> None:
    import oxyz.torch_sim

    state = oxyz.torch_sim.read(DATA_DIR / "varying_atom_counts.xyz")
    # O, H, H, H, O, H -> standard weights from the ASE-parity table.
    assert state.masses[0].item() == pytest.approx(15.999)
    assert state.masses[1].item() == pytest.approx(1.008)


def test_frames_disagreeing_on_pbc_raise(tmp_path: Path) -> None:
    import oxyz.torch_sim

    path = _write(
        tmp_path,
        '1\nLattice="1 0 0 0 1 0 0 0 1" pbc="T T T" Properties=species:S:1:pos:R:3\n'
        "H 0 0 0\n"
        '1\nLattice="1 0 0 0 1 0 0 0 1" pbc="T T F" Properties=species:S:1:pos:R:3\n'
        "H 0 0 0\n",
    )
    with pytest.raises(oxyz.torch_sim.ToSimStateError, match="disagree on pbc"):
        oxyz.torch_sim.read(path)


def test_missing_pos_and_species_raise(tmp_path: Path) -> None:
    import oxyz.torch_sim

    no_pos = _write(tmp_path, "1\nProperties=species:S:1\nH\n")
    with pytest.raises(oxyz.torch_sim.ToSimStateError, match="pos"):
        oxyz.torch_sim.read(no_pos)

    bad_species = _write(tmp_path, "1\nProperties=species:S:1:pos:R:3\nZz 0 0 0\n")
    with pytest.raises(oxyz.torch_sim.ToSimStateError, match="Zz"):
        oxyz.torch_sim.read(bad_species)


def test_no_lattice_molecule_has_zero_cell_and_false_pbc() -> None:
    import torch

    import oxyz.torch_sim

    state = oxyz.torch_sim.read(DATA_DIR / "no_lattice_molecule.xyz")
    assert torch.count_nonzero(state.cell) == 0
    assert state.pbc.tolist() == [False, False, False]


def test_int_index_out_of_range_raises() -> None:
    import oxyz.torch_sim

    with pytest.raises(IndexError, match="out of range"):
        oxyz.torch_sim.read(DATA_DIR / "varying_atom_counts.xyz", 99)


def test_frame_without_species_or_z_raises(tmp_path: Path) -> None:
    import oxyz.torch_sim

    # pos but no species/Z column: nothing to derive atomic numbers from.
    path = _write(tmp_path, "1\nProperties=pos:R:3\n0.0 0.0 0.0\n")
    with pytest.raises(oxyz.torch_sim.ToSimStateError, match="'species' or 'Z'"):
        oxyz.torch_sim.read(path)


def test_masses_column_of_wrong_width_raises(tmp_path: Path) -> None:
    import oxyz.torch_sim

    # A width-2 masses column flattens to twice the atom count; masses are scalar.
    path = _write(
        tmp_path,
        "1\nProperties=species:S:1:pos:R:3:masses:R:2\nH 0 0 0 1.0 2.0\n",
    )
    with pytest.raises(oxyz.torch_sim.ToSimStateError, match="masses"):
        oxyz.torch_sim.read(path)


def test_malformed_lattice_raises(tmp_path: Path) -> None:
    import oxyz.torch_sim

    path = _write(
        tmp_path,
        '1\nLattice="1 2 3 4 5 6" Properties=species:S:1:pos:R:3\nH 0 0 0\n',
    )
    with pytest.raises(oxyz.torch_sim.ToSimStateError, match="9 components"):
        oxyz.torch_sim.read(path)


def test_scalar_pbc_broadcasts_to_three_axes(tmp_path: Path) -> None:
    import oxyz.torch_sim

    # A scalar pbc=T applies to all three axes, as ASE has it.
    path = _write(
        tmp_path,
        '1\nLattice="1 0 0 0 1 0 0 0 1" pbc=T Properties=species:S:1:pos:R:3\n'
        "H 0 0 0\n",
    )
    state = oxyz.torch_sim.read(path)
    assert state.pbc.tolist() == [True, True, True]


def test_malformed_pbc_shape_raises(tmp_path: Path) -> None:
    import oxyz.torch_sim

    path = _write(
        tmp_path,
        '1\npbc="T F" Lattice="1 0 0 0 1 0 0 0 1" Properties=species:S:1:pos:R:3\n'
        "H 0 0 0\n",
    )
    with pytest.raises(oxyz.torch_sim.ToSimStateError, match="scalar or 3"):
        oxyz.torch_sim.read(path)


def test_extras_pull_metadata_and_columns(tmp_path: Path) -> None:
    import torch

    import oxyz.torch_sim

    path = _write(
        tmp_path,
        '2\nLattice="10 0 0 0 10 0 0 0 10" energy=-1.5 '
        "Properties=species:S:1:pos:R:3:forces:R:3\n"
        "H 0 0 0 0.1 0.0 0.0\nH 1 0 0 -0.1 0.0 0.0\n"
        '1\nLattice="10 0 0 0 10 0 0 0 10" energy=-0.5 '
        "Properties=species:S:1:pos:R:3:forces:R:3\n"
        "H 0 0 0 0.0 0.0 0.0\n",
    )
    state = oxyz.torch_sim.read(
        path,
        dtype=torch.float64,
        system_extras={"energy": "energy"},
        atom_extras={"forces": "forces"},
    )
    assert state.energy.shape == (2,)  # per-system
    assert torch.allclose(state.energy, torch.tensor([-1.5, -0.5], dtype=torch.float64))
    assert state.forces.shape == (3, 3)  # per-atom


def test_missing_extra_source_raises(tmp_path: Path) -> None:
    import oxyz.torch_sim

    path = _write(
        tmp_path,
        '1\nLattice="1 0 0 0 1 0 0 0 1" Properties=species:S:1:pos:R:3\nH 0 0 0\n',
    )
    with pytest.raises(oxyz.torch_sim.ToSimStateError, match="missing"):
        oxyz.torch_sim.read(path, system_extras={"energy": "absent"})


def test_iread_streams_batches_covering_every_frame() -> None:
    import oxyz.torch_sim

    states = list(
        oxyz.torch_sim.iread(DATA_DIR / "varying_atom_counts.xyz", frames_per_batch=2)
    )
    assert [s.n_systems for s in states] == [2, 1]
    assert sum(s.n_systems for s in states) == 3


def test_iread_density_binning_matches_iter_batches() -> None:
    import oxyz.torch_sim

    states = list(
        oxyz.torch_sim.iread(
            DATA_DIR / "varying_density.extxyz",
            memory_scales_with="n_atoms_x_density",
            max_scaler=4,
        )
    )
    assert [s.n_systems for s in states] == [1, 2]


def test_iread_requires_a_strategy() -> None:
    import oxyz.torch_sim

    with pytest.raises(ValueError, match="exactly one"):
        list(oxyz.torch_sim.iread(DATA_DIR / "varying_atom_counts.xyz"))


def test_source_state_matches_read_and_serves_arrays() -> None:
    import torch

    import oxyz.torch_sim

    source = oxyz.torch_sim.SimStateSource(DATA_DIR / "varying_atom_counts.xyz")
    assert len(source) == 3
    state = source.state(dtype=torch.float64)
    direct = oxyz.torch_sim.read(
        DATA_DIR / "varying_atom_counts.xyz", dtype=torch.float64
    )
    assert torch.equal(state.system_idx, direct.system_idx)
    assert torch.allclose(state.positions, direct.positions)

    energy = source.per_config("energy", dtype=torch.float64)
    assert energy.shape == (3,)
    forces, offsets = source.per_atom("forces", dtype=torch.float64)
    assert forces.shape == (6, 3)
    assert list(offsets) == [0, 3, 4, 6]


def test_source_missing_key_raises() -> None:
    import oxyz.torch_sim

    source = oxyz.torch_sim.SimStateSource(DATA_DIR / "varying_atom_counts.xyz")
    with pytest.raises(ValueError, match="missing"):
        source.per_config("nope")
    with pytest.raises(ValueError, match="missing"):
        source.per_atom("nope")


def test_default_dtype_infers_float64_like_atoms_to_state(tmp_path: Path) -> None:
    import torch

    import oxyz.torch_sim

    # dtype=None infers from the data (float64), as atoms_to_state does, rather
    # than resolving to torch.get_default_dtype().
    path = _write(
        tmp_path,
        '1\nLattice="1 0 0 0 1 0 0 0 1" Properties=species:S:1:pos:R:3\nH 0 0 0\n',
    )
    state = oxyz.torch_sim.read(path)
    assert state.positions.dtype == torch.float64


def test_positions_requires_grad_is_honoured(tmp_path: Path) -> None:
    import oxyz.torch_sim

    path = _write(
        tmp_path,
        '1\nLattice="1 0 0 0 1 0 0 0 1" Properties=species:S:1:pos:R:3\nH 0 0 0\n',
    )
    state = oxyz.torch_sim.read(path, positions_requires_grad=True)
    assert state.positions.requires_grad
