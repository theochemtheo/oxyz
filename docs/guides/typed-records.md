# Typed records

Use a `Record` when a workflow knows the names and types of the fields it
consumes. A record read projects onto those fields rather than retaining every
file key. See [schemas and projection](schemas-and-projection.md) for untyped
projection, [schema concepts](../concepts/schemas.md) for missingness, and the
[data model](../concepts/data-model.md) for ownership and mutability.

## Declare fields and bind file keys

The annotation contains the field specifier, not just its value type:
`Column[NDArray[np.float64]]`, not a bare array annotation. A specifier is either
the default or metadata in `Annotated`, never both. Aliases used by postponed
annotations must be resolvable at module scope, not local to a function.

```python title="Read mapped fields and optional values" linenums="1"
from pathlib import Path
from typing import Annotated

import numpy as np
from numpy.typing import NDArray
import oxyz
from oxyz import Column, Meta, Record, RecordBatch

type Array = NDArray[np.float64]
type Positions = Annotated[Column[Array], Column("pos", width=3)]


class Sample(Record):
    species: Column[list[str]] = Column()
    positions: Positions
    energy: Meta[float] = Meta("REF_energy")
    charge: Meta[int | None] = Meta(fill=0)
    forces: Column[Array | None] = Column(width=3)


source = Path("records.extxyz")
source.write_text(
    "1\nProperties=species:S:1:pos:R:3 REF_energy=-1.5\nH 0.25 0 0\n"
    "1\nProperties=species:S:1:pos:R:3:forces:R:3 REF_energy=0 charge=0\n"
    "He 0.5 0 0 0 0 0\n"
)
samples = oxyz.read(source, record=Sample)
assert [s.energy for s in samples] == [-1.5, 0.0]
assert samples[0].species == ["H"]
assert samples[0].positions.shape == (1, 3)
assert samples[0].positions[0, 0] == 0.25
assert samples[0].charge is None and samples[0].forces is None
assert samples[1].charge == 0
assert samples[1].forces is not None
assert samples[1].forces.tolist() == [[0.0, 0.0, 0.0]]
assert oxyz.read(source, 1, record=Sample) == samples[1]
assert list(oxyz.iread(source, record=Sample)) == samples
assert Sample.from_frame(oxyz.read(source, 0)) == samples[0]
assert Sample.energy.key == "REF_energy"
assert Sample.positions.key == "pos"
assert Sample.species.key == "species"
assert Sample.energy.name == "energy" and Sample.energy.owner is Sample
```

On the class, `Sample.energy` is a bound field key; on an instance it is the
float value. Binding copies the declaration's specifier and supplies its field
name, owner and default file key. Inherited keys retain their identity. Keys are
literal names, not schema glob/regular-expression patterns. Two fields cannot
read the same key on the same axis.

| Declaration | Value in one record |
| --- | --- |
| `Column[NDArray[np.float64]]` | Float array: `(n_atoms,)`, or `(n_atoms, width)` for `width > 1` |
| `Column[NDArray[np.int64]]`, `Column[NDArray[np.bool_]]` | Corresponding integer or boolean arrays |
| `Column[list[str]]` | Width-one string column |
| `Column[list[list[str]]]` | String rows, requiring `width > 1` |
| `Meta[float]`, `Meta[int]`, `Meta[bool]`, `Meta[str]` | Scalar comment-line value |
| `Meta[NDArray[np.float64]]` (also int64/bool), `Meta[list[str]]` | Fixed vector declared with `shape=(k,)` |
| `Cell[NDArray[np.float64]]` | `(3, 3)` cell from `Lattice`, lattice vectors in rows |
| `Matrix3[NDArray[np.float64]]` | `(3, 3)` matrix from a named nine-value metadata entry |

Append `| None` **inside** any specifier's value type to make it nullable.
Only the listed array dtypes are supported; this is not an arbitrary NumPy dtype
or shape-typing interface. `Cell` reads flat values in C order; `Matrix3` reads
flat values in Fortran order (for stress/virial). A nested 3×3 matrix is read as
written. Generic `Meta` vectors have one declared dimension, not arbitrary ranks.

## Choose missing-value behaviour

A field without `| None` or `fill=` is required. For an absent nullable field,
an individual record contains `None`, even when a fill was declared. For an
absent non-nullable field with `fill=`, the record contains the fill, repeated
to its declared shape. A wrong-kind value is not an absent value.

By default, reads and `from_frame` promote integer input for a float field.
Declare `class StrictSample(Sample, strict=True)` to reject that promotion;
this setting is inherited. It is distinct from `conformance="strict"`.

A direct constructor requires `n_atoms` and **every** field as keyword arguments,
including nullable and filled fields. It checks types, dtypes, widths, shapes
and atom counts, but keeps supplied values without conversion or copying.
Records are frozen, slotted and unhashable; freezing does not freeze their arrays.
Read-owned arrays are writable. `from_frame` instead returns read-only array
views where possible (changes through the original frame can show through), and
copies string lists. Copy an array explicitly when independence is needed.

## Gather records into batches

`read_batch` takes `RecordBatch[Sample]`, not `Sample`. Nullable fields become
dense batch values plus a per-frame mask; indexing a row restores `None`.

```python title="Check batch filling, presence and source indices"
batch = oxyz.read_batch(source, record=RecordBatch[Sample])
assert batch.offsets.tolist() == [0, 1, 2]
assert batch.frame_indices.tolist() == [0, 1]
assert batch[Sample.energy].tolist() == [-1.5, 0.0]
assert batch[Sample.charge].tolist() == [0, 0]
assert batch.present(Sample.charge).tolist() == [False, True]
assert batch[Sample.forces].shape == (2, 3)
assert np.isnan(batch[Sample.forces][0]).all()
assert batch[Sample.forces][1].tolist() == [0.0, 0.0, 0.0]
assert batch.present(Sample.forces).tolist() == [False, True]
assert list(batch) == samples

picked = batch[[1, 0, 1]]
assert picked.frame_indices.tolist() == [1, 0, 1]
assert picked.present("charge").tolist() == [True, False, True]
assert [s.energy for s in picked] == [0.0, -1.5, 0.0]
windows = list(oxyz.iread_batch(source, record=RecordBatch[Sample], frames_per_batch=1))
assert [w.frame_indices.tolist() for w in windows] == [[0], [1]]
assert [w[0] for w in windows] == samples
```

Columns concatenate over atoms; metadata gains a frame axis. Float fields use
NaN as the default batch fill. Nullable integer, boolean and string fields need
an explicit kind-compatible `fill=` to bind a batch read, even if a particular
file happens to contain the key everywhere. Presence is per **frame**, including
for atom columns; a zero, NaN or empty atom array alone cannot establish absence.

In a single-record batch, `present(...)` tracks nullable fields only. A
non-nullable filled field reports all-true even where its value was synthesised.
For [union batches](mixed-datasets.md), member-specific fields also carry masks.
Required fields report all-true. Masks are read-only and survive selection.

For typed attribute access, declare every batch field in its batch form, with no
defaults:

```python title="Declare typed batch attributes"
class SampleBatch(RecordBatch[Sample]):
    species: list[str]
    positions: Array
    energy: Array
    charge: NDArray[np.int64]
    forces: Array


declared = oxyz.read_batch(source, record=SampleBatch)
assert declared.energy.tolist() == [-1.5, 0.0]
assert declared.charge.tolist() == [0, 0]
assert declared[0] == samples[0]
assert declared.forces is declared[Sample.forces]
```

Unit-step slices are views; stepped slices, position sequences and boolean masks
gather copies. `frame_indices` preserves file indices through reordering,
repetition and dropped frames, whereas integer batch indexing uses batch positions.
See [batching and streaming](batching-and-streaming.md) for layout and budgets.

## Read restrictions and typing boundaries

- `read`/`iread` accept a record class or sibling-record union; batch readers
  accept its `RecordBatch[...]` form or a declared batch subclass.
- Do not pass `mode=` with `record=`: record reads always project.
- An accompanying `schema=` describes the dataset, not a second projection.
  Rules matching record keys must agree in kind and width/shape; integer rules
  may satisfy float fields unless the record is strict. Its frame rule applies;
  requiring `Lattice` needs a field that reads it. Rules for other keys are not
  applied, and schema fills/required flags do not replace record declarations.
- Under `conformance="warn"`, missing required fields and wrong-kind fields
  cause warnings and dropped records, not invalid typed values. A frame-rule
  violation warns but retains the frame. Selection still counts file frames;
  requesting a dropped frame by integer raises `IndexError`.
- `Sample.spec()` is useful for inspection but a plain read with that spec is
  not a record read: filling, warning/drop behaviour and batch missingness differ.
  A plain `Batch` has no per-frame presence history. `RecordBatch[Sample].from_batch`
  treats an existing key as present everywhere, an absent nullable key as absent
  everywhere; it cannot recover information already lost through filling.

## Typing limits

!!! warning "Static checking is narrower than runtime validation"

    Existing `ty` regression cases cover supported field values and class keys,
    `read`/`iread` return types, typed-key batch lookup, declared batch attributes
    and union rows. Generic `RecordBatch[R]` does not expose typed field
    attributes; use `batch[R.field]`. String-name lookup returns `Any`.
    NumPy shape checks remain runtime checks. `ty` may accept an omitted
    default-form constructor field which runtime still requires. Runtime batch
    constructor keywords `present=` and `tags=` are not visible to the checker;
    `dataclasses.replace(batch, present=...)` is accepted. These are not promises
    of equivalent support across type checkers or of static validation of union
    compatibility.

## Implementation and regression ownership

The [API reference](../reference/api.md) routes the public surface. Implementation
owners are [`_record.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_record.py)
(declarations, construction and frame conversion),
[`_record_fields.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_record_fields.py)
(supported types and casts),
[`_record_bind.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_record_bind.py)
(binding and read plans),
[`_record_batch.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_record_batch.py)
(batch layout, indexing and masks), and
[`_frames.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_frames.py)/
[`_batch.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_batch.py)
(reader integration).

Regression owners are
[`test_record.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_record.py),
[`test_record_fields.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_record_fields.py),
[`test_record_batch.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_record_batch.py)
and [`test_record_typing.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_record_typing.py).
Runtime examples do not establish static typing claims; retain the typing tests
when changing these contracts.
