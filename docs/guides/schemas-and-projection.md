# Schemas and projection

Use `infer_schema` to inspect observed data, a `SchemaSpec` to state expectations,
and project mode to select and materialise a fixed set of fields. For the
reasoning behind these distinctions, see [schema concepts](../concepts/schemas.md).
For named Python fields and presence masks, use [typed records](typed-records.md)
and [mixed datasets](mixed-datasets.md).

## Infer, review, save and read

`infer_schema` parses every frame, unlike the structural `scan`. This small file
has a charge column in only its first frame:

```python title="Infer and save a dataset contract"
from pathlib import Path

import numpy as np
import oxyz

source = Path("charges.extxyz")
source.write_text(
    "1\nProperties=species:S:1:pos:R:3:charge:R:1 energy=-1.5\n"
    "H 0.25 0 0 0.5\n"
    "1\nProperties=species:S:1:pos:R:3 energy=-2.5\n"
    "He 1.25 0 0\n"
)
observed = oxyz.infer_schema(source)
assert observed.n_frames == 2
assert observed.n_atoms.tolist() == [1, 1]
assert observed["charge"].frames_present == 1
assert observed["charge"].unified == (oxyz.Kind.REAL, 1)
assert not observed.is_consistent

spec = observed.to_spec()
assert spec.mode == "validate"
assert spec.frame is None
assert not spec["charge"].required
spec.to_file("charges.yaml")
assert oxyz.SchemaSpec.from_file("charges.yaml") == spec

frames = oxyz.read(source, schema="charges.yaml")
assert frames[0].columns["charge"].tolist() == [0.5]
assert "charge" not in frames[1].columns
assert [frame.metadata["energy"] for frame in frames] == [-1.5, -2.5]
```

Review the generated spec before treating it as a contract. Partial presence
becomes `required=False`, not a promise that absence is scientifically valid.
`to_spec()` uses unified kinds where possible; unresolved drift uses the
most frequent variant (first seen wins ties). It can collapse at least three
compatible, fully present numbered column names into a counted glob. It sets
no atom-count or lattice requirements. A generated spec is therefore a starting
point, not proof that every observed frame validates against it.

Specs also round-trip through dictionaries, JSON and YAML. `from_file` reads
JSON, YAML and TOML; `to_file` writes JSON or YAML, not TOML.

## Validate without changing values

Validation checks exact kinds, column widths and scalar/vector metadata shapes
without casting, removing or filling values. See the
[shape limitation](../concepts/schemas.md#declared-structure-is-a-consumer-contract)
before using higher-rank metadata. The reader's default conformance is `"required"`:

| Conformance | Validation behaviour |
| --- | --- |
| `"required"` | Raise on missing required fields, mismatches, pattern cardinality or frame-rule failures; allow extras |
| `"strict"` | The same, but also reject undeclared fields |
| `"warn"` | Warn for deviations, including extras, and return the unmodified frames |

```python title="Check strict failure context and warning behaviour"
import warnings

positions_only = oxyz.SchemaSpec(
    columns=(oxyz.ColumnRule("pos", oxyz.Kind.REAL, width=3),)
)
allowed = oxyz.read(source, schema=positions_only, conformance="required")
assert allowed[0].columns["species"] == ["H"]
assert allowed[0].metadata["energy"] == -1.5

try:
    oxyz.read(source, schema=positions_only, conformance="strict")
except oxyz.SchemaError as error:
    assert (error.frame_index, error.name) == (0, "species")
else:
    raise AssertionError("strict validation accepted an undeclared column")

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", oxyz.SchemaWarning)
    unchanged = oxyz.read(source, schema=positions_only, conformance="warn")
assert any(issubclass(item.category, oxyz.SchemaWarning) for item in caught)
assert any("species" in str(item.message) for item in caught)
assert unchanged == allowed
```

`SchemaError.frame_index` and `.name` give structured schema failure context;
a spec error detected before reading may have neither. These are not parser
line/column guarantees. A plain `read(..., schema=...)` validates the whole
file even when selecting one frame; `iread` encounters failures as it advances.
Do not transfer the selected-only contract of record reads to schema validation.

## Project fields and fill absence

Project mode outputs only declared fields, in declaration order. It promotes
integer data to a declared REAL kind, but is not a general cast: strings are
not parsed into numbers, reals are not narrowed to integers, and widths or
shapes are not repaired. Metadata projection supports scalars and one-dimensional
arrays, not arbitrary multidimensional metadata.

```python title="Project a mixed file into a uniform batch"
projected_spec = oxyz.SchemaSpec(
    columns=(
        oxyz.ColumnRule("pos", oxyz.Kind.REAL, width=3),
        oxyz.ColumnRule("charge", oxyz.Kind.REAL, required=False),
        oxyz.ColumnRule("id", oxyz.Kind.INT, required=False, fill=-1),
    ),
    metadata=(oxyz.MetadataRule("energy", oxyz.Kind.REAL),),
    mode="project",
)
projected = oxyz.read(source, schema=projected_spec)
assert list(projected[0].columns) == ["pos", "charge", "id"]
assert projected[0].columns["pos"].tolist() == [[0.25, 0.0, 0.0]]
assert projected[0].columns["charge"].tolist() == [0.5]
assert np.isnan(projected[1].columns["charge"]).all()
assert [frame.columns["id"].tolist() for frame in projected] == [[-1], [-1]]
assert projected[1].metadata["energy"] == -2.5

batch = oxyz.read_batch(source, schema=projected_spec)
assert batch.offsets.tolist() == [0, 1, 2]
assert batch.columns["id"].tolist() == [-1, -1]
assert np.isnan(batch.columns["charge"][1])
assert batch.metadata["energy"].tolist() == [-1.5, -2.5]
```

Optional REAL fields default to NaN. Optional INT, BOOL and STR fields need an
explicit kind-compatible `fill` unless omitted; `0`, `False` and `""` are valid
fills. A fill broadcasts over the declared width or shape. It does not make a
required field optional: missing required data remains a deviation.

The read argument `mode=` overrides `spec.mode` and requires `schema=`. Under
projection, `"strict"` and `"required"` enforce the same deviations: extras are
removed rather than rejected.

```python title="Separate exact validation from integer-to-real promotion"
integer_source = Path("integer-energy.extxyz")
integer_source.write_text("1\nProperties=species:S:1:pos:R:3 energy=2\nH 0 0 0\n")
real_energy = oxyz.SchemaSpec(metadata=(oxyz.MetadataRule("energy", oxyz.Kind.REAL),))
try:
    oxyz.read(integer_source, schema=real_energy)
except oxyz.SchemaError as error:
    assert (error.frame_index, error.name) == (0, "energy")
else:
    raise AssertionError("validation promoted an integer")

promoted = oxyz.read(integer_source, 0, schema=real_energy, mode="project")
assert promoted.columns == {}
assert promoted.metadata["energy"] == 2.0
assert isinstance(promoted.metadata["energy"], float)
```

!!! warning "Projection with warnings can lose data"

    With `conformance="warn"`, a missing or mismatched field can be replaced by
    its fill, even when required; an unfillable frame is dropped. This is not
    validation's return-unchanged behaviour. Plain projected frame reads select
    from the surviving frames after drops; record reads instead retain source
    selection semantics. See [mixed datasets](mixed-datasets.md).

```python title="Check a warned mismatch and its replacement value"
bad_source = Path("wrong-charge.extxyz")
bad_source.write_text("1\nProperties=charge:S:1\nseven\n")
charge_spec = oxyz.SchemaSpec(
    columns=(oxyz.ColumnRule("charge", oxyz.Kind.REAL),), mode="project"
)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", oxyz.SchemaWarning)
    replaced = oxyz.read(bad_source, schema=charge_spec, conformance="warn")
assert len(replaced) == 1
assert np.isnan(replaced[0].columns["charge"]).all()
assert any(
    "frame 0" in str(item.message) and "charge" in str(item.message) for item in caught
)
assert all(issubclass(item.category, oxyz.SchemaWarning) for item in caught)
```

## Leave optional fields absent

Use `absent="omit"` when absence must remain a missing dictionary key. This
requires no fill, takes precedence over a fill, and has no effect on required
rules. Present data must still have the declared kind and width or shape.

```python title="Preserve absence rather than manufacturing a value"
omit_spec = oxyz.SchemaSpec(
    columns=(oxyz.ColumnRule("charge", oxyz.Kind.REAL, required=False, absent="omit"),),
    mode="project",
)
omitted = oxyz.read(source, schema=omit_spec)
assert omitted[0].columns["charge"].tolist() == [0.5]
assert omitted[1].columns == {}

try:
    oxyz.read_batch(source, schema=omit_spec)
except oxyz.ParseError as error:
    assert error.frame_index == 1
    assert "charge" in str(error)
else:
    raise AssertionError("a plain batch accepted mixed field presence")
```

A plain `Batch` requires uniform field presence and compatible signatures;
validation's optional rule does not relax that assembly contract. Projection
with fills can make the selected fields uniform. A fill in a plain batch does
not carry a presence mask, so a filled zero cannot be distinguished from a
source zero. `RecordBatch` has a separate filling and per-frame presence
contract; see [typed records](typed-records.md).

## Add structural requirements

`FrameRule` can bound atom counts and require `Lattice` metadata. Unset bounds
are not enforced. These checks apply in both modes; lattice presence alone
is not a promise of a particular shape or physical cell validity.

```python title="Check structural failure context while streaming"
structural = oxyz.SchemaSpec(frame=oxyz.FrameRule(n_atoms_min=2))
try:
    next(oxyz.iread(source, schema=structural))
except oxyz.SchemaError as error:
    assert (error.frame_index, error.name) == (0, "n_atoms")
else:
    raise AssertionError("the minimum atom count was not enforced")

needs_lattice = oxyz.SchemaSpec(frame=oxyz.FrameRule(lattice_required=True))
try:
    oxyz.read(source, schema=needs_lattice)
except oxyz.SchemaError as error:
    assert (error.frame_index, error.name) == (0, "Lattice")
else:
    raise AssertionError("missing lattice was accepted")
```

In project mode retain a declared `Lattice` field if the projected frame must
satisfy `lattice_required`; projection does not add undeclared metadata.

## Freeze patterns before projection

Validation accepts glob names and `re:` regular expressions. Literal rules take
precedence, then the first matching pattern in declaration order. Pattern
`count` sets an exact match count; otherwise `min`/`max` set bounds, with a
default minimum of one for required patterns and zero for optional ones.
Projection needs literal names: freeze against a representative local sample.

```python title="Expand a pattern and check the resulting projected values"
descriptors = Path("descriptors.extxyz")
descriptors.write_text(
    "1\nProperties=d_0:R:1:d_1:R:1\n0.1 0.2\n1\nProperties=d_0:R:1\n0.3\n"
)
pattern = oxyz.SchemaSpec(columns=(oxyz.ColumnRule("d_*", oxyz.Kind.REAL),))
frozen = pattern.freeze(descriptors)
assert pattern.mode == "validate"
assert frozen.mode == "project"
assert [rule.name for rule in frozen.columns] == ["d_0", "d_1"]
assert frozen["d_0"].required
assert not frozen["d_1"].required
expanded = oxyz.read(descriptors, schema=frozen)
assert expanded[0].columns["d_1"].tolist() == [0.2]
assert expanded[1].columns["d_0"].tolist() == [0.3]
assert np.isnan(expanded[1].columns["d_1"]).all()
```

Freeze parses the sample. Literal rules pass through unchanged. Expanded rules
use observed unified kind and width/shape, become required only when present
in every sample frame, and retain the pattern's fill/absence policy. Freeze is
not validation of the pattern's declared signature or cardinality. Conflicting
observed signatures cannot be frozen; optional non-REAL expansions need fills
or `absent="omit"`. The result fixes the sample's names, not those of future
files: review and save it as a versioned dataset contract.
