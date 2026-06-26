"""Parity tests: oxyz.metatomic must agree with systems_to_torch(ase.io.read).

System fields (types/positions/cell/pbc) are compared against
metatomic.torch.systems_to_torch fed by ase.io.read, the path oxyz.metatomic
replaces. Array-native extraction (per_config/per_atom) is checked against ASE's
info/arrays. Voigt stress — which ASE's reader rejects — gets its own test.
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
    or importlib.util.find_spec("metatomic") is None,
    reason="requires ase, torch, and metatomic-torch",
)

DATA_DIR = Path(__file__).parent / "data"

# Files ase.io.read disagrees on (Voigt stress it rejects, plus oxyz's
# documented metadata divergences). System fields are unaffected, but we keep
# the reference set to those ASE reads cleanly, matching test_ase's GOLDEN.
DIVERGENT = {
    "simple.extxyz",
    "nonorthogonal.extxyz",
    "stress_voigt6.extxyz",
    "newstyle_array_metadata.extxyz",
    "singlequote_metadata.extxyz",
}

GOLDEN = sorted(
    path.name
    for path in list(DATA_DIR.glob("*.xyz")) + list(DATA_DIR.glob("*.extxyz"))
    if path.name not in DIVERGENT
)


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "frame.extxyz"
    path.write_text(body)
    return path


def assert_system_parity(got, ref) -> None:
    import torch

    assert torch.equal(got.types, ref.types)
    assert torch.equal(got.pbc, ref.pbc)
    assert got.positions.dtype == ref.positions.dtype
    assert torch.allclose(got.positions, ref.positions)
    assert torch.allclose(got.cell, ref.cell)


@pytest.mark.parametrize("name", GOLDEN)
@pytest.mark.parametrize("dtype_name", ["default", "float64"])
def test_system_parity_with_systems_to_torch(name: str, dtype_name: str) -> None:
    import torch
    from ase.io import read as ase_read
    from metatomic.torch import systems_to_torch

    import oxyz.metatomic as om

    dtype = None if dtype_name == "default" else torch.float64
    path = DATA_DIR / name

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # pbc/cell mismatch fires in both paths
        got = om.read(path, dtype=dtype)
        ase_frames = ase_read(path, ":", format="extxyz")
        assert isinstance(ase_frames, list)  # narrow ase.io.read's union
        # metatomic types its input as the abstract IntoSystem, which ase.Atoms
        # is not declared to satisfy, so ty cannot see the (supported) call.
        ref = systems_to_torch(ase_frames, dtype=dtype)  # ty: ignore[invalid-argument-type]
        assert isinstance(ref, list)

    assert len(got) == len(ref)
    for g, r in zip(got, ref, strict=True):
        assert_system_parity(g, r)


def test_per_config_and_per_atom_match_ase() -> None:
    import torch
    from ase.io import read as ase_read

    import oxyz.metatomic as om

    path = DATA_DIR / "mace_ref_energy_forces_stress.xyz"
    frames = ase_read(path, ":", format="extxyz")
    source = om.SystemSource(path)

    energy = source.per_config("REF_energy", dtype=torch.float64)
    assert energy.shape == (len(frames),)
    np.testing.assert_allclose(energy.numpy(), [f.info["REF_energy"] for f in frames])

    forces, offsets = source.per_atom("REF_forces", dtype=torch.float64)
    expected = np.concatenate([f.arrays["REF_forces"] for f in frames], axis=0)
    np.testing.assert_allclose(forces.numpy(), expected)
    np.testing.assert_array_equal(offsets, [0, *np.cumsum([len(f) for f in frames])])


def test_per_config_voigt_stress_keeps_width_six() -> None:
    # ASE's comment parser rejects 6-vector stress, so there is no ASE reference;
    # oxyz keeps it as-written, stacked (n_frames, 6) for the caller to reshape.
    import torch

    import oxyz.metatomic as om

    source = om.SystemSource(DATA_DIR / "stress_voigt6.extxyz")
    stress = source.per_config("stress", dtype=torch.float64)
    assert stress.shape == (len(source), 6)


def test_dtype_and_requires_grad_options() -> None:
    import torch

    import oxyz.metatomic as om

    path = DATA_DIR / "minimal_periodic.extxyz"

    (system,) = om.read(path, dtype=torch.float32, positions_requires_grad=True)
    assert system.positions.dtype == torch.float32
    assert system.positions.requires_grad
    assert system.types.dtype == torch.int32


def test_default_dtype_follows_torch_default() -> None:
    import torch

    import oxyz.metatomic as om

    path = DATA_DIR / "minimal_periodic.extxyz"
    (system,) = om.read(path)
    assert system.positions.dtype == torch.get_default_dtype()


def test_pbc_cell_mismatch_warns_like_systems_to_torch() -> None:
    import oxyz.metatomic as om

    # periodic_pbc_ttf has a non-zero third cell vector but pbc="T T F".
    with pytest.warns(UserWarning, match="non-zero cell vectors"):
        om.read(DATA_DIR / "periodic_pbc_ttf.extxyz")


def test_non_periodic_molecule_has_zero_cell() -> None:
    import torch

    import oxyz.metatomic as om

    (system,) = om.read(DATA_DIR / "no_lattice_molecule.xyz")
    assert torch.equal(system.cell, torch.zeros((3, 3), dtype=system.cell.dtype))
    assert not system.pbc.any()


def test_index_selects_single_or_list() -> None:
    import oxyz.metatomic as om

    path = DATA_DIR / "two_frame_same_schema.xyz"
    # System is a TorchScript custom class, not an isinstance-able type; check
    # the shape of the result instead — a bare System vs a list of them.
    one = om.read(path, 0)
    assert not isinstance(one, list)
    assert hasattr(one, "positions")
    several = om.read(path, ":")
    assert isinstance(several, list) and len(several) == 2


def test_iread_streams_same_systems_as_read() -> None:
    import torch

    import oxyz.metatomic as om

    path = DATA_DIR / "two_frame_same_schema.xyz"
    eager = om.read(path, dtype=torch.float64)
    streamed = list(om.iread(path, dtype=torch.float64))
    assert len(eager) == len(streamed)
    for a, b in zip(eager, streamed, strict=True):
        assert torch.equal(a.types, b.types)
        assert torch.allclose(a.positions, b.positions)


def test_missing_species_and_pos_raise_to_system_error(tmp_path: Path) -> None:
    import oxyz.metatomic as om

    no_pos = _write(tmp_path, "1\nProperties=species:S:1\nH\n")
    with pytest.raises(om.ToSystemError, match="pos"):
        om.read(no_pos)

    bad_species = _write(tmp_path, "1\nProperties=species:S:1:pos:R:3\nZz 0 0 0\n")
    with pytest.raises(om.ToSystemError, match="Zz"):
        om.read(bad_species)


def test_malformed_lattice_raises_to_system_error(tmp_path: Path) -> None:
    import oxyz.metatomic as om

    path = _write(
        tmp_path,
        '1\nLattice="1 2 3 4 5 6" Properties=species:S:1:pos:R:3\nH 0 0 0\n',
    )
    with pytest.raises(om.ToSystemError, match="9 components"):
        om.read(path)


def test_extraction_missing_key_raises() -> None:
    import oxyz.metatomic as om

    source = om.SystemSource(DATA_DIR / "minimal_periodic.extxyz")
    with pytest.raises(ValueError, match="missing from frame"):
        source.per_config("nonexistent")
    with pytest.raises(ValueError, match="missing from frame"):
        source.per_atom("nonexistent")


def test_scalar_pbc_broadcasts_to_three_axes(tmp_path: Path) -> None:
    import oxyz.metatomic as om

    # `pbc=T` (a scalar) is accepted by ASE and broadcast to all three axes;
    # reproduce that rather than crashing on a reshape.
    path = _write(
        tmp_path,
        '1\nLattice="1 0 0 0 1 0 0 0 1" pbc=T '
        "Properties=species:S:1:pos:R:3\nH 0 0 0\n",
    )
    (system,) = om.read(path)
    assert system.pbc.tolist() == [True, True, True]


def test_non_numeric_column_raises_clear_error(tmp_path: Path) -> None:
    import oxyz.metatomic as om

    path = _write(tmp_path, "1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n")
    source = om.SystemSource(path)
    with pytest.raises(ValueError, match="not numeric"):
        source.per_atom("species")


def test_extraction_on_empty_source_raises(tmp_path: Path) -> None:
    import oxyz.metatomic as om

    source = om.SystemSource(_write(tmp_path, ""))
    with pytest.raises(ValueError, match="no frames"):
        source.per_config("energy")
    with pytest.raises(ValueError, match="no frames"):
        source.per_atom("forces")


def test_per_config_inconsistent_shapes_raises(tmp_path: Path) -> None:
    import oxyz.metatomic as om

    path = _write(
        tmp_path,
        '1\nfoo="1 2 3" Properties=species:S:1:pos:R:3\nH 0 0 0\n'
        '1\nfoo="1 2" Properties=species:S:1:pos:R:3\nH 0 0 0\n',
    )
    with pytest.raises(ValueError, match="inconsistent shapes"):
        om.SystemSource(path).per_config("foo")
