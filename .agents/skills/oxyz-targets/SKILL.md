---
name: oxyz-targets
description: Use when changing oxyz ASE, metatomic or torch_sim conversion, optional import guards, scientific field conventions, Torch device/dtype handling or reference-library parity.
---

# oxyz targets

Native frames preserve stored data; converters interpret it. Start with the
[root router](../oxyz/SKILL.md), [converter guide](../../../docs/guides/converters.md),
[compatibility](../../../docs/reference/compatibility.md) and
[optional-target decision](../../../docs/development/decisions/optional-output-targets.md).

## Owners and checks

Paths below are under `src/oxyz/`; tests are under `tests/`.

| Concern | Owner | Existing checks |
| --- | --- | --- |
| ASE naming, calculator routing, constraints and inverse conversion | `ase.py`; writer integration in `_write.py` | `test_ase.py`, `test_write.py` |
| Independent metatomic systems and target extraction | `metatomic.py` | `test_metatomic.py` |
| Batched SimState, extras, masses and streaming states | `torch_sim.py`; native assembly in `_batch.py` | `test_torch_sim.py` |
| Shared atomic identity, cell/pbc and mass conventions | `_convert.py` | `test_convert.py` and target suites |
| Tensor construction and dtype resolution | `_torch.py` | Metatomic/torch_sim dtype, extraction and gradient cases |
| Optional guards and dependency-tier isolation | Target module imports, `__init__.py` | `test_optional_deps.py`; wheel tiers in `.github/workflows/test.yml` |

## Invariants and traps

- Keep optional imports behind target guards. `_torch` imports Torch eagerly;
  only guarded target modules may reach it. Extras are `ase`, `metatomic` and
  `torch-sim` (distribution `torch-sim-atomistic`, import `torch_sim`). Missing
  dependencies raise informative `ImportError` with the correct install hint.
  Do not install all extras to make an isolated-tier failure disappear.
- Do not move scientific interpretation into parsing: arbitrary labels and
  flat native lattice/stress survive native reads. Conversion does not infer
  units, change force/gradient signs, or derive stress from virial/volume.
- ASE reuses its extxyz name/result routing; unprefixed results enter a
  single-point calculator, not arbitrary prefixed training labels. Preserve
  row-vector cells, special stress/virial matrix ordering and documented
  divergences. `move_mask=True` means free motion; inverse conversion supports
  only `FixAtoms`, not every readable constraint.
- Torch targets resolve numbers from `Z`, then `numbers`, then `species`.
  Metatomic returns independent row-cell systems, zeros non-periodic cell
  vectors and defaults floats to `torch.get_default_dtype()`.
  `cell_requires_grad` is accepted but not applied.
- Torch_sim returns one batched state, uses column-vector cells, requires
  compatible batch fields and shared pbc, and defaults floating structure
  arrays to float64. Masses prefer the explicit column; extras are opt-in maps.
- Keep `device`, dtype, shapes and extraction offsets explicit. `_torch.to_tensor`
  constructs tensors from numeric/bool arrays, not an existing autograd history;
  position gradients do not promise upstream graph preservation or backend
  support. CPU parity is not accelerator coverage. Consult the guide for the
  different extraction dtype defaults and source-object limitations.

## Targeted verification

Choose the matching field-for-field golden: `test_matches_ase_reader` compares
structures, arrays/info, calculator results and constraints with ASE;
`test_system_parity_with_systems_to_torch` compares metatomic systems;
`test_state_parity_with_atoms_to_state` compares SimState fields. Include each
target's convention/error tests when touching its boundary, not just construction.

For import guards and restoration of hidden dependencies:

```sh
uv run pytest -q tests/test_optional_deps.py
```

Use [testing](../../../docs/development/testing.md#source-and-wheel-environments)
for actual base/ASE/S3 wheel isolation and its `OXYZ_TEST_TIER` canary; the full
environment's guard tests alone do not prove those tiers. Run a target's module
with `uv run pytest -q tests/test_ase.py` (or the corresponding table entry).
Use [run-oxyz](../run-oxyz/SKILL.md) for environment mechanics and
[develop-oxyz](../develop-oxyz/SKILL.md) for public API/stub obligations.
