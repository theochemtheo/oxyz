# Converting to atomistic libraries

Native `Frame` and `Batch` objects preserve extxyz field names and values.
Converters interpret those fields for ASE, metatomic or torch_sim. Parsing a
file successfully does not guarantee that it describes a convertible chemical
structure. Keep native data when arbitrary labels or the original field layout
matter; see [compatibility and interpretation](../reference/compatibility.md).

## Choose a target

Install only the extras your application needs:

```sh title="Install optional output targets"
pip install "oxyz[ase]"
pip install "oxyz[metatomic]"
pip install "oxyz[torch-sim]"
```

The `torch-sim` extra installs the distribution `torch-sim-atomistic`, imported
as `torch_sim`. Optional libraries are imported through their target modules;
native reads do not require them. The examples below require all three extras
and run without network access, a model or accelerator hardware.

| Entry point | Default selection | Result |
| --- | --- | --- |
| `oxyz.read(path)` | All frames | List of native `Frame` objects |
| `oxyz.ase.read(path)` | Last frame | One `ase.Atoms` |
| `oxyz.ase.read(path, ":")` | All frames explicitly | List of `Atoms` |
| `oxyz.metatomic.read(path)` | All frames | List of `metatomic.torch.System` objects |
| `oxyz.torch_sim.read(path)` | All frames | One batched `torch_sim.SimState` |

Converter `read` selectors accept integers, slices and ASE-style slice strings,
including negative indices and reverse slices. An integer returns one `Atoms`
or `System`; a slice returns a list. For torch_sim, **every** selection returns
one batched state, including an integer selection with one system. Do not assume
that converters accept native readers' explicit index-sequence API.

These APIs use runtime column and metadata keys, not `record=` declarations.
`R.from_frame` and `RecordBatch[R].from_batch` bind
[typed records](typed-records.md); they are not optional-target converters.

## Create a tiny reference trajectory

The asymmetric cell makes row/column conventions observable. The unprefixed
`energy`, `forces` and `stress` are calculator results to ASE; `REF_energy`
remains an ordinary label. Coordinates, targets and charges are deliberately
non-zero so the comparisons check more than object creation.

```python title="Create local structures and targets"
from pathlib import Path

import numpy as np
import oxyz

trajectory = Path("converter.extxyz")
header = (
    'Lattice="2 1 0 0 3 0 0 0 4" pbc="T T T" '
    'Properties=species:S:1:pos:R:3:forces:R:3:charge:R:1 '
    'stress="1 0 0 0 2 0 0 0 3" '
)
trajectory.write_text(
    "2\n" + header + "energy=-1.5 REF_energy=-2.0 config_type=pair\n"
    "H 0.25 0 0 0.25 0 0 0.5\n"
    "O 1.25 0.5 0 -0.25 0 0 -0.5\n"
    "1\n" + header + "energy=-0.5 REF_energy=-0.75 config_type=single\n"
    "H 0 0.25 0.5 0 0.5 0 0\n"
)
native = oxyz.read(trajectory)
assert [frame.n_atoms for frame in native] == [2, 1]
assert native[0].columns["species"] == ["H", "O"]
assert native[0].metadata["Lattice"].shape == (9,)
assert native[0].metadata["stress"].shape == (9,)
assert native[0].metadata["REF_energy"] == -2.0
```

oxyz does not infer units or rescale scientific values. Supply coordinates,
energies, forces, stress and masses in the units expected by your downstream
library or model. Conversion does not turn forces into energy gradients, derive
stress from virial, or change signs and volume factors.

Both torch targets require `pos` and resolve atomic numbers from `Z`, then
`numbers`, then `species`. Numeric floating atomic numbers are rounded before
conversion to int32; species must be case-normalisable chemical symbols, not
arbitrary atom-type labels. These are target conventions, not changes to the
stored native columns.

## ASE: structures, calculator routing and writing

`oxyz.ase.read` defaults to the last frame, as `ase.io.read` does; native
`oxyz.read` defaults to all frames. `oxyz.ase.iread` defaults to all frames and
always returns an iterator, even for an integer. Only `format=None`, `"extxyz"`
and `"xyz"` are accepted; `"xyz"` still requires extxyz `Properties`, not a
free-text ordinary XYZ comment.

```python title="Compare ASE structures and calculator results"
import ase.io
import oxyz.ase

reference_atoms = ase.io.read(trajectory, index=":", format="extxyz")
images = oxyz.ase.read(trajectory, ":")
assert len(images) == len(reference_atoms) == 2
assert len(oxyz.ase.read(trajectory)) == 1
assert oxyz.ase.read(trajectory).positions.tolist() == [[0.0, 0.25, 0.5]]
assert [len(atoms) for atoms in oxyz.ase.iread(trajectory)] == [2, 1]

for atoms, reference in zip(images, reference_atoms, strict=True):
    np.testing.assert_array_equal(atoms.numbers, reference.numbers)
    np.testing.assert_allclose(atoms.positions, reference.positions, rtol=0, atol=0)
    np.testing.assert_allclose(atoms.cell.array, reference.cell.array, rtol=0, atol=0)
    np.testing.assert_array_equal(atoms.pbc, reference.pbc)
    np.testing.assert_array_equal(
        atoms.get_initial_charges(), reference.get_initial_charges()
    )
    assert atoms.get_potential_energy() == reference.get_potential_energy()
    np.testing.assert_allclose(
        atoms.get_forces(), reference.get_forces(), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        atoms.get_stress(), reference.get_stress(), rtol=0, atol=0
    )
    assert atoms.info["REF_energy"] == reference.info["REF_energy"]

assert images[0].numbers.tolist() == [1, 8]
assert images[0].cell.array.tolist() == [[2.0, 1.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]
assert images[0].get_potential_energy() == -1.5
assert images[0].get_forces().tolist() == [[0.25, 0.0, 0.0], [-0.25, 0.0, 0.0]]
assert images[0].get_stress().tolist() == [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]
assert "energy" not in images[0].info
assert "forces" not in images[0].arrays
assert images[0].info["REF_energy"] == -2.0
```

The converter reuses ASE's extxyz naming and result-routing tables. `species`,
`pos`, `Z` and `charge` map to symbols, positions, atomic numbers and initial
charges. Recognised result keys enter a `SinglePointCalculator`; other metadata
and columns follow ASE's `info`/array routing. Arbitrary prefixes are not stripped
and singular `force` is not automatically renamed to `forces`.

Convert an existing native frame with `frame.to_atoms()` or
`oxyz.ase.to_atoms(frame)`. The inverse is `oxyz.ase.from_atoms(atoms)`.
There are no ASE `from_frame` or `to_frame` functions. `oxyz.write` and `Writer`
accept `Atoms` directly and use the inverse conversion; there is no separate
`oxyz.ase.write` API.

```python title="Convert existing frames and write ASE results"
from ase.constraints import FixAtoms

converted = native[0].to_atoms()
assert converted.get_potential_energy() == -1.5
np.testing.assert_array_equal(
    oxyz.ase.to_atoms(native[0]).positions, converted.positions
)
converted.set_constraint(FixAtoms(indices=[0]))
back_to_native = oxyz.ase.from_atoms(converted)
assert back_to_native.columns["move_mask"].tolist() == [False, True]
assert back_to_native.metadata["energy"] == -1.5
assert back_to_native.columns["forces"].shape == (2, 3)

oxyz.write("ase-roundtrip.extxyz", converted)
restored_atoms = oxyz.ase.read("ase-roundtrip.extxyz", 0)
assert restored_atoms.get_potential_energy() == -1.5
np.testing.assert_array_equal(restored_atoms.positions, converted.positions)
np.testing.assert_array_equal(restored_atoms.cell.array, converted.cell.array)
np.testing.assert_array_equal(
    restored_atoms.get_forces(apply_constraint=False),
    converted.get_forces(apply_constraint=False),
)
assert restored_atoms.constraints[0].get_indices().tolist() == [0]
assert oxyz.read("ase-roundtrip.extxyz", 0).metadata["stress"].shape == (6,)
np.testing.assert_array_equal(restored_atoms.get_stress(), converted.get_stress())
```

!!! warning "ASE conversion is an interpretation, not a native round trip"

    `move_mask=True` means free motion. Reading supports width-one `FixAtoms`
    and width-three `FixCartesian` masks, but inverse conversion supports only
    `FixAtoms`; other constraints raise `FromAtomsError`. Calculator results are
    folded into unprefixed native fields. Original field names and layout are
    therefore not guaranteed after conversion through ASE.

Missing positions, unknown chemical species and malformed lattices are conversion
errors, not requests to repair a structure. Explicit `Z` wins over valid species,
but species present on the ASE path are still checked. Writing the calculator
stress above produces six-component Voigt metadata: re-read it with `oxyz.ase`,
which accepts that form, rather than ASE's parser, which rejects it. See the
[full divergence list](../reference/compatibility.md#deliberate-and-grammar-driven-divergences)
for stress, species, string arrays and apostrophes, and the
[writing guide](writing.md) for codecs and incremental output.

## Metatomic: independent systems and target tensors

`oxyz.metatomic.read` returns a list by default; an integer returns one `System`.
`iread` yields systems individually. Frames convert independently, so metadata
keys may differ between frames. The public conversion paths are these readers
and `SystemSource.systems()`; there is no public `from_frame`, inverse conversion
or metatomic writer.

```python title="Compare metatomic systems with systems_to_torch"
import torch
from metatomic.torch import systems_to_torch

import oxyz.metatomic

cpu = torch.device("cpu")
systems = oxyz.metatomic.read(trajectory, dtype=torch.float64, device=cpu)
reference_systems = systems_to_torch(reference_atoms, dtype=torch.float64, device=cpu)
assert len(systems) == len(reference_systems) == 2
for system, reference in zip(systems, reference_systems, strict=True):
    torch.testing.assert_close(system.types, reference.types, rtol=0, atol=0)
    torch.testing.assert_close(system.positions, reference.positions, rtol=0, atol=0)
    torch.testing.assert_close(system.cell, reference.cell, rtol=0, atol=0)
    torch.testing.assert_close(system.pbc, reference.pbc, rtol=0, atol=0)
assert systems[0].types.tolist() == [1, 8]
assert systems[0].positions.shape == (2, 3)
assert systems[0].cell.tolist() == images[0].cell.array.tolist()
assert systems[0].pbc.tolist() == [True, True, True]
assert systems[0].types.dtype == torch.int32
assert systems[0].positions.device == cpu
assert [system.positions.shape[0] for system in oxyz.metatomic.iread(trajectory)] == [2, 1]

differentiable = oxyz.metatomic.read(
    trajectory, 0, dtype=torch.float32, device=cpu,
    positions_requires_grad=True, cell_requires_grad=True,
)
assert differentiable.positions.dtype == torch.float32
assert differentiable.positions.requires_grad
assert not differentiable.cell.requires_grad
assert oxyz.metatomic.read(trajectory, 0).positions.dtype == torch.get_default_dtype()
```

`types` are int32 atomic numbers, `pbc` is boolean, and floating positions/cell
use `torch.get_default_dtype()` when `dtype=None`. `device` places returned
tensors; `positions_requires_grad=True` enables position gradients.
`cell_requires_grad` is accepted for reference-signature parity but **not
applied**. Tensors are constructed from parsed data, not attached to an existing
autograd history.

Metatomic uses row-vector cells. A missing lattice gives a zero cell; absent
`pbc` means all periodic with a lattice and all non-periodic without one.
Scalar `pbc` broadcasts to three flags. Like `systems_to_torch`, conversion
zeros non-periodic cell vectors and warns when the non-zero cell vectors and
periodicity disagree. This differs from preserving the full native lattice.

For structures and supervised targets together, `SystemSource` parses the
whole file once and retains native frames. Its `systems()` method creates
systems; `per_config(key)` stacks metadata as `(n_frames, *value_shape)`;
`per_atom(key)` concatenates a column and returns NumPy frame offsets alongside
the tensor. Scalar metadata has shape `(n_frames,)`, not `(n_frames, 1)`.

```python title="Extract metatomic targets without renaming or scientific transforms"
source = oxyz.metatomic.SystemSource(trajectory)
assert len(source) == 2
cached_systems = source.systems(dtype=torch.float64, device=cpu)
torch.testing.assert_close(cached_systems[0].positions, systems[0].positions, rtol=0, atol=0)
energy = source.per_config("energy", dtype=torch.float64, device=cpu)
forces, offsets = source.per_atom("forces", dtype=torch.float64, device=cpu)
assert energy.shape == (2,)
assert energy.tolist() == [-1.5, -0.5]
assert forces.shape == (3, 3)
assert offsets.tolist() == [0, 2, 3]
np.testing.assert_array_equal(energy.numpy(), [a.get_potential_energy() for a in reference_atoms])
np.testing.assert_array_equal(
    forces.numpy(), np.concatenate([a.get_forces() for a in reference_atoms])
)
assert forces[offsets[1]:offsets[2]].tolist() == [[0.0, 0.5, 0.0]]
assert source.per_config("REF_energy").tolist() == [-2.0, -0.75]
stored_stress = source.per_config("stress", dtype=torch.float64)
assert stored_stress.shape == (2, 9)
assert stored_stress[0].tolist() == [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]
assert source.per_config("energy").dtype == torch.get_default_dtype()
```

Extraction requires the key in every frame and numeric or boolean values;
strings cannot become target tensors. Per-configuration shapes must agree;
per-atom widths must concatenate. Empty-source extraction raises `ValueError`.
Unlike the torch_sim source below, metatomic extraction with `dtype=None` casts
to `torch.get_default_dtype()`, including integer/boolean source fields; request
an explicit dtype when those kinds matter.

`System` carries structure, not an automatic bundle of energy/force/stress
labels. Target extraction preserves stored values and widths: six-component
stress stays six components, nine-component stress stays nine. A downstream
metatrain reader can use `systems()` for structures and `per_config`/`per_atom`
for targets, but gradient signs, volume factors and `TensorMap` construction
belong to that integration, not oxyz. oxyz does not depend on metatrain.

## Torch_sim: one state containing many systems

`oxyz.torch_sim.read` assembles a native batch, then one `SimState`. Selected
frames must have compatible batch fields; varying atom counts are supported,
but arbitrary missing/drifting fields are not. Use
[schema projection](schemas-and-projection.md) to select compatible fields
before conversion, retaining positions and atomic identity. There is no public
`from_batch`, inverse conversion or torch_sim writer.

```python title="Compare the batched state with atoms_to_state"
from torch_sim.io import atoms_to_state

import oxyz.torch_sim

state = oxyz.torch_sim.read(trajectory, dtype=torch.float64, device=cpu)
reference_state = atoms_to_state(reference_atoms, dtype=torch.float64, device=cpu)
assert state.n_systems == 2
assert state.positions.shape == (3, 3)
assert state.cell.shape == (2, 3, 3)
assert state.masses.shape == (3,)
assert state.atomic_numbers.tolist() == [1, 8, 1]
assert state.system_idx.tolist() == [0, 0, 1]
assert state.system_idx.dtype == torch.int64
assert state.atomic_numbers.dtype == torch.int32
assert state.pbc.shape == (3,)
assert state.pbc.tolist() == [True, True, True]
for field in ("positions", "cell", "masses", "atomic_numbers", "system_idx", "pbc"):
    torch.testing.assert_close(
        getattr(state, field), getattr(reference_state, field), rtol=0, atol=0
    )
np.testing.assert_array_equal(state.cell[0].numpy(), images[0].cell.array.T)
np.testing.assert_allclose(state.masses.numpy(), [1.008, 15.999, 1.008], rtol=0, atol=1e-12)
assert oxyz.torch_sim.read(trajectory).positions.dtype == torch.float64
single_state = oxyz.torch_sim.read(trajectory, -1)
assert single_state.n_systems == 1
assert single_state.positions.tolist() == [[0.0, 0.25, 0.5]]
reverse_state = oxyz.torch_sim.read(trajectory, "::-1")
assert reverse_state.system_idx.tolist() == [0, 1, 1]
assert reverse_state.positions[0].tolist() == [0.0, 0.25, 0.5]
```

Cells use **column vectors**, the transpose of ASE/metatomic cells. Missing
lattices give zero cells; absent periodic flags default as above. All systems in
a state must share one three-flag `pbc`; disagreement raises `ToSimStateError`.
Masses come from a per-atom `masses` column if supplied, otherwise the standard
atomic weights matching ASE's table. A masses column must supply one value per
atom; this is not an isotope inference step.

```python title="Override standard atomic weights explicitly"
isotopes = Path("custom-masses.extxyz")
isotopes.write_text(
    "2\nProperties=species:S:1:pos:R:3:masses:R:1\n"
    "H 0 0 0 2.5\nO 1 0 0 18\n"
)
custom_state = oxyz.torch_sim.read(isotopes)
custom_reference = atoms_to_state(ase.io.read(isotopes, index=":"))
assert custom_state.masses.tolist() == [2.5, 18.0]
torch.testing.assert_close(custom_state.masses, custom_reference.masses, rtol=0, atol=0)
assert custom_state.pbc.tolist() == [False, False, False]
assert torch.count_nonzero(custom_state.cell).item() == 0
```

Floating positions, cell and masses retain float64 with `dtype=None`, unlike
metatomic's torch-default convention. Pass `torch.float32` explicitly for models
that require it. `device` and `positions_requires_grad` work on the returned
state. Extra tensors are opt-in maps of `{state_key: native_key}`, not automatic
calculator-result routing. Keys must exist and values must be numeric/boolean;
`dtype=None` preserves their inferred numeric/boolean dtype, whereas an explicit
`dtype` also casts extras.

```python title="Attach targets and stream fixed-size states"
labelled_state = oxyz.torch_sim.read(
    trajectory, dtype=torch.float32, device=cpu, positions_requires_grad=True,
    system_extras={"energy": "REF_energy"}, atom_extras={"forces": "forces"},
)
assert labelled_state.positions.dtype == torch.float32
assert labelled_state.positions.device == cpu
assert labelled_state.positions.requires_grad
assert labelled_state.energy.shape == (2,)
assert labelled_state.energy.tolist() == [-2.0, -0.75]
assert labelled_state.forces.shape == (3, 3)
assert labelled_state.forces.tolist() == [[0.25, 0.0, 0.0], [-0.25, 0.0, 0.0], [0.0, 0.5, 0.0]]
states = list(oxyz.torch_sim.iread(trajectory, frames_per_batch=1))
assert [item.n_systems for item in states] == [1, 1]
assert [item.positions.shape for item in states] == [(2, 3), (1, 3)]
torch.testing.assert_close(
    torch.cat([item.positions for item in states]), state.positions, rtol=0, atol=0
)
```

`iread` requires exactly one batching strategy: `frames_per_batch`,
`atoms_per_batch`, or `memory_scales_with` together with `max_scaler`.
The memory-scaling choices are `"n_atoms"` and `"n_atoms_x_density"`.
`shuffle`/`seed` and source restrictions follow
[batching and streaming](batching-and-streaming.md); density-based planning is
not model-memory measurement.

For model-aware batching of an already materialised state, torch_sim's
[`BinningAutoBatcher`](https://github.com/torchsim/torch-sim/blob/main/torch_sim/autobatching.py)
accepts that state through `load_states`. It uses a model to estimate a suitable
memory-scaling limit when no explicit `max_memory_scaler` is supplied, then
packs systems into batches. This belongs to torch_sim and may require a model
and GPU memory probing; it is not exercised by this local-data guide. It also
does not avoid materialising the input state: use oxyz `iread` when that is the
constraint.

`SimStateSource` parses the whole file once into a compatible native batch.
`state()` converts it; `per_config` and `per_atom` return independently extracted
tensors with the same layouts and NumPy offset convention as `SystemSource`.
Here `dtype=None` infers from stored data rather than using torch's default.

```python title="Reuse the torch_sim source for structures and targets"
state_source = oxyz.torch_sim.SimStateSource(trajectory)
assert len(state_source) == 2
cached_state = state_source.state(dtype=torch.float64, device=cpu)
torch.testing.assert_close(cached_state.positions, state.positions, rtol=0, atol=0)
state_energy = state_source.per_config("energy")
state_forces, state_offsets = state_source.per_atom("forces")
assert state_energy.dtype == torch.float64
assert state_energy.shape == (2,)
assert state_energy.tolist() == [-1.5, -0.5]
assert state_offsets.tolist() == [0, 2, 3]
torch.testing.assert_close(state_forces, forces, rtol=0, atol=0)
assert state_source.per_config("stress").shape == (2, 9)
```

Neither source object accepts `record=` or exposes schema processing options;
converter readers accept `schema`, `conformance` and `mode`. Projection happens
before conversion and can remove required structure fields. For source URLs,
compression and archive members, see [compressed and remote sources](compressed-and-remote.md).
Writing remains limited to native frames and ASE `Atoms`, not `System` or
`SimState` objects.

## Contract sources

- [ASE implementation](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/ase.py),
  [golden and divergence tests](https://github.com/theochemtheo/oxyz/blob/main/tests/test_ase.py)
  and [inverse/write tests](https://github.com/theochemtheo/oxyz/blob/main/tests/test_write.py).
- [Metatomic implementation](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/metatomic.py)
  and [reference parity and extraction tests](https://github.com/theochemtheo/oxyz/blob/main/tests/test_metatomic.py).
- [Torch_sim implementation](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/torch_sim.py)
  and [batch/reference parity tests](https://github.com/theochemtheo/oxyz/blob/main/tests/test_torch_sim.py).
- [Shared chemistry conventions](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_convert.py),
  [tensor conversion](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_torch.py)
  and [element/mass table tests](https://github.com/theochemtheo/oxyz/blob/main/tests/test_convert.py).
