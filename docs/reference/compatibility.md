# Compatibility and interpretation

Native oxyz frames preserve represented extxyz data. Optional converters interpret
that data for another library. Those are different contracts: a successful parse
does not imply that a frame describes a chemical structure or can be converted.
See [the format reference](extxyz.md), [native data model](../concepts/data-model.md)
and [API reference](api.md).

## What lossless means

!!! note "Lossless does not mean byte-for-byte"

    Native columns retain their names, kinds, widths and values; metadata retains
    its inferred values and dimensions. `REF_energy` is not renamed to `energy`,
    `force` is not silently made `forces`, and species labels are not resolved to
    elements by the parser. `Lattice` stays flat; stress and virial are not
    converted or reinterpreted. Lossless does not mean preservation of whitespace,
    quote spelling, numeric spelling or byte-for-byte file layout.

Accessors such as `frame.cell`, `frame.pbc` and `frame.numbers` provide interpreted
views without changing those stored fields. For example, absent `Lattice` means
a zero interpreted cell; absent `pbc` means all periodic with a `Lattice`, and
all non-periodic without one. A scalar `pbc` broadcasts to three flags. These
are accessor defaults, not inserted metadata.

The [lossless-core decision](../development/decisions/lossless-core-and-layering.md)
records this boundary. Use native frames when preserving arbitrary training
labels or inspecting a file before choosing a target representation matters.

## ASE boundary

`oxyz.ase.read` and `oxyz.ase.iread` read extxyz into `ase.Atoms`;
`frame.to_atoms()` and `oxyz.ase.to_atoms(frame)` convert an existing frame.
ASE is optional and imported through this converter, not required by the native
reader. These are extxyz entry points, not replacements for ASE's format registry.
Only `format=None`, `"extxyz"` and `"xyz"` are accepted, and `"xyz"` still uses
the extxyz grammar with required `Properties`. Ordinary XYZ with a free-text
comment and no descriptor is not supported.

### Selection defaults and partial validation

| Entry point | Default selection | Result |
| --- | --- | --- |
| `oxyz.read(path)` | Every frame | List of native frames |
| `oxyz.ase.read(path)` | Last frame (`-1`; `None` also means `-1`) | One `Atoms` |
| `oxyz.ase.read(path, ":")` | Every frame explicitly | List of `Atoms` |
| `oxyz.ase.iread(path)` | Every frame | Iterator of `Atoms` |

Integer selections return one object; slice and slice-string selections return
lists from `read`. ASE-style indices include negative integers and reverse
slices. The native [reading guide](../guides/reading-and-selecting.md) explains
selection and [storage concepts](../concepts/storage-and-streaming.md) explain
scanning versus parsing.

A bounded forward read can finish without validating later frames. Tests show
that requesting frame 0 or `"0:2"` succeeds even when a later count line is
malformed; consuming the remaining iterator then raises. On a local seekable
source, negative or reverse selections use a structural scan and seek: skipped
atom values are not thereby validated, but malformed counts or truncation can
still prevent indexing. A structural scan is not a full grammar check.

Do not extend that partial-read guarantee to every option. For compressed or
remote sources, or when a schema is supplied, ASE negative/reverse selection
uses a whole-file read. Schema-backed ASE reads apply schema processing before
selection; the tests explicitly require a schema failure outside the selected
slice to abort the read. Projection also precedes conversion: removing `pos`
can make a previously convertible frame fail. Iterator errors and native parser
termination are described in [errors](errors.md).

### Field interpretation

The converter reuses ASE's extxyz name tables and `set_calc_and_arrays` routing.
Conventional column names such as `species`, `pos`, `Z` and `charge` become the
corresponding ASE symbols, positions, numbers and initial charges. Results known
to ASE are routed to a single-point calculator; other values become arrays or
`Atoms.info` as ASE's routing prescribes. Arbitrary prefixed labels are not
automatically promoted to calculator results.

An explicit `Z` column supplies numbers in preference to valid `species`.
Species, when present, are still checked: case-normalised chemical symbols such
as `si` and `FE` are recognised; unknown or non-scalar species raise
`ToAseError`. Missing both species/numbers and missing positions are errors;
positions are not silently replaced with zeros. A malformed `Lattice` is an
error at conversion, although its native metadata can be parsed.

For the flat nine-value lattice, ASE receives the row-vector cell. If no `pbc`
is supplied, a lattice implies three true flags. Without a lattice, ASE receives
its zero-cell/non-periodic defaults unless explicit `pbc` says otherwise.

`move_mask` has true where an atom or Cartesian component is free. Width one
becomes `FixAtoms`; width three becomes `FixCartesian`; other widths raise
`ToAseError`. The inverse conversion, `oxyz.ase.from_atoms`, is more restrictive:
only `FixAtoms` constraints can currently be represented, not `FixCartesian`
or arbitrary constraint classes. It folds calculator results into unprefixed
fields and follows ASE's writer naming conventions. An ASE round trip therefore
is not a promise to reproduce the original native field names or metadata layout.

### Stress and virial

Native stress and virial remain exactly their inferred arrays. At the ASE
boundary, nine-component values in ASE's special matrix-key table are reshaped
in Fortran order. ASE's result routing then interprets a stress matrix as Voigt
stress. Six-component `stress` is accepted directly as already-Voigt and sent to
the calculator. A `virial` matrix follows ASE's metadata treatment; oxyz does not
compute stress from virial by changing sign or dividing by cell volume.

Do not confuse lattice row-vector presentation with stress/virial matrix
ordering. The lattice is transposed after the Fortran reshape for ASE's cell;
the stress/virial special matrices are not given that cell transpose.

### Deliberate and grammar-driven divergences

The corpus comparison tests document four differences from ASE's reader:

| Case | oxyz behaviour | ASE behaviour recorded by the tests |
| --- | --- | --- |
| Six-component `stress` | Accepted as Voigt calculator stress | Comment parser rejects it |
| Non-chemical species | Conversion error | Can construct an unintended `Atoms` |
| `tags=["a","b"]` | Typed list of strings | Keeps one raw string containing the quoted elements |
| Single quotes in `label='hello'` or `note=it's` | Preserved as literal characters | Strips the apostrophes |

The first two are strictness/acceptance choices at the converter boundary; the
last two follow the native grammar and typing. Golden tests compare atomic
numbers, positions, cells, periodicity, arrays, information, calculator results
and constraints on the remaining corpus. This is evidence for those fixtures,
not a claim of compatibility with every ASE version or every input dialect.

### Check the cell and stress boundary

This non-orthogonal example verifies the native representation separately from
ASE's matrix interpretation.

```python title="Check the ASE cell and stress interpretation"
from pathlib import Path

import oxyz
import oxyz.ase

path = Path("ase-boundary.extxyz")
path.write_text(
    '1\nProperties=species:S:1:pos:R:3 '
    'Lattice="2 1 0 0 3 0 0 0 4" stress="1 2 3 4 5 6" '
    'virial="1 2 3 4 5 6 7 8 9"\nSi 0.5 0.25 0\n'
)
frame = oxyz.read(path, 0)
assert frame.metadata["Lattice"].shape == (9,)
assert frame.metadata["Lattice"].tolist() == [2, 1, 0, 0, 3, 0, 0, 0, 4]
assert frame.metadata["stress"].tolist() == [1, 2, 3, 4, 5, 6]
assert "pbc" not in frame.metadata
atoms = oxyz.ase.to_atoms(frame)
assert atoms.numbers.tolist() == [14]
assert atoms.positions.tolist() == [[0.5, 0.25, 0.0]]
assert atoms.cell.array.tolist() == [[2.0, 1.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]
assert atoms.pbc.tolist() == [True, True, True]
assert atoms.calc.results["stress"].tolist() == [1, 2, 3, 4, 5, 6]
assert atoms.info["virial"].tolist() == [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
assert frame.metadata["virial"].shape == (9,)
```

## Other optional targets: contract summary

These converters are separate interpretations, not alternate native storage
formats. Installing or importing their optional dependencies is required before
using them. See the [optional-target decision](../development/decisions/optional-output-targets.md)
and [API reference](api.md) for public entry points.

| Target | Representation and restrictions |
| --- | --- |
| `oxyz.metatomic` | Converts each frame independently to `metatomic.torch.System`; requires positions and resolvable atomic numbers. Metadata may differ between frames. Uses row-vector cells and three periodic flags. Non-periodic cell vectors are zeroed, with the parity warning when cell-vector presence and periodicity disagree. Float dtype defaults to `torch.get_default_dtype()`. `cell_requires_grad` is accepted for signature parity but is not applied. |
| `oxyz.torch_sim` | Builds one batched `SimState` using native batch assembly, so selected frames need compatible batch fields. Cells use column vectors (the row-vector cell transposed). All systems must share the same three periodic flags. Requires positions and resolvable atomic numbers; masses come from a `masses` column or the atomic-weight table. Default floating data stays float64. Additional metadata/columns enter extras only through explicit key maps and must be tensor-convertible. |

Both use shared atomic-number resolution (`Z`, then `numbers`, then `species`),
which is distinct from promising arbitrary atom labels are convertible. A missing
cell defaults to zero; a present cell must have the expected nine components.
Converter errors describe target representation failures, not parser failures.
No general target-to-native round-trip guarantee is implied for these targets.

## Contract sources

- [ASE converter and docstrings](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/ase.py)
  and [ASE golden/divergence tests](https://github.com/theochemtheo/oxyz/blob/main/tests/test_ase.py).
- [Shared chemical conversion helpers](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_convert.py)
  and [element/mass contracts](https://github.com/theochemtheo/oxyz/blob/main/tests/test_convert.py).
- [Native format tests](https://github.com/theochemtheo/oxyz/blob/main/tests/test_extxyz.py),
  including unchanged training labels and flat lattice storage.
- [Metatomic converter](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/metatomic.py)
  and [torch_sim converter](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/torch_sim.py).
