# Mixed datasets

Use a union of sibling `Record` classes when frames have distinct field sets.
A union selects a concrete record class from **key presence**, not from a
metadata value or the order of the classes. Start with
[typed records](typed-records.md); see [schema concepts](../concepts/schemas.md),
[schemas and projection](schemas-and-projection.md) and the
[data model](../concepts/data-model.md) for the projection and presence contracts.

## Choose a representation

| Consumer needs | Use | Consequence |
| --- | --- | --- |
| Every stored key, without forcing frames into one shape | `read` / `iread` returning `Frame` | Each frame may have different fields |
| One dense untyped shape | Project-mode `SchemaSpec` and `read_batch` | Fill or discard fields; plain batches do not retain presence |
| One typed contract with genuinely optional observations | One `Record` with nullable fields, then `RecordBatch[R]` | Rows use `None`; dense batches retain optional-field masks |
| Distinct, key-identifiable frame types | A sibling-record union | Rows retain concrete member classes; batches retain masks and tags |

Use [schema projection](schemas-and-projection.md) to normalise a dataset when
type identity is irrelevant. Use a nullable record when absence does not mean a
different kind of frame. A union is useful only when its keys distinguish the
members unambiguously.

## Distinguish absent charge from present zero

Here `Bare` has no charge field; `Charged` declares one. The file deliberately
contains a present zero as well as an absent charge. The batch uses zero to fill
the absent entry, so only the presence mask tells those entries apart.

```python title="Select union members by keys, not numeric values" linenums="1" hl_lines="37-43"
from pathlib import Path

import oxyz
from oxyz import Column, Meta, Record, RecordBatch


class Atom(Record):
    species: Column[list[str]] = Column()
    energy: Meta[float] = Meta()


class Bare(Atom):
    pass


class Charged(Atom):
    charge: Meta[float] = Meta(fill=0.0)
    labelled: Meta[float] = Meta()


type Sample = Bare | Charged

source = Path("mixed-records.extxyz")
source.write_text(
    "1\nProperties=species:S:1 energy=-1.0\nH\n"
    "1\nProperties=species:S:1 energy=0.0 labelled=1.0 charge=0.0\nHe\n"
    "1\nProperties=species:S:1 energy=-2.0 labelled=1.0 charge=2.0\nLi\n"
)
records = oxyz.read(source, record=Sample)
assert [type(r) for r in records] == [Bare, Charged, Charged]
assert not hasattr(records[0], "charge")
assert isinstance(records[1], Charged) and records[1].charge == 0.0
assert isinstance(records[2], Charged) and records[2].charge == 2.0
assert [r.energy for r in records] == [-1.0, 0.0, -2.0]
assert list(oxyz.iread(source, record=Sample)) == records

batch = oxyz.read_batch(source, record=RecordBatch[Sample])
assert batch[Charged.charge].tolist() == [0.0, 0.0, 2.0]
assert batch.present(Charged.charge).tolist() == [False, True, True]
assert batch.members == (Bare, Charged)
assert batch.tags.tolist() == [0, 1, 1]
assert batch.where(Bare).tolist() == [True, False, False]
assert batch.where(Charged).tolist() == [False, True, True]
assert list(batch) == records
assert batch.frame_indices.tolist() == [0, 1, 2]
```

`labelled` is a required distinguishing key, not a value discriminator:
both `labelled=1.0` and `labelled=0.0` select `Charged`. It also makes the union
unambiguous when `Charged.charge` is filled and therefore not required. A
`Charged` frame may omit `charge`; its individual record then receives `0.0`,
while the union batch mask records that the source lacked that matching key.

## Select members and preserve provenance

`tags` contains indices into `members`, not user-defined category values.
Reversing the union order reverses that numbering, not the matching decision.
`where(*members)` combines exact member classes; a shared base is not a member,
and no arguments select no frames.

```python title="Gather a union without losing masks or member identity"
selected = batch[[2, 0, 1]]
assert selected.frame_indices.tolist() == [2, 0, 1]
assert selected.tags.tolist() == [1, 0, 1]
assert selected[Charged.charge].tolist() == [2.0, 0.0, 0.0]
assert selected.present("charge").tolist() == [True, False, True]
charged = selected[selected.where(Charged)]
assert charged.frame_indices.tolist() == [2, 1]
assert charged[Charged.charge].tolist() == [2.0, 0.0]
assert charged.present(Charged.charge).tolist() == [True, True]
assert selected.where(Bare, Charged).all()
assert not selected.where().any()
assert oxyz.read(source, [2, 0], record=Sample) == [records[2], records[0]]

windows = list(
    oxyz.iread_batch(source, record=RecordBatch[Sample], frames_per_batch=2)
)
assert [w.frame_indices.tolist() for w in windows] == [[0, 1], [2]]
assert [type(r) for w in windows for r in w] == [Bare, Charged, Charged]
assert [bool(p) for w in windows for p in w.present(Charged.charge)] == [
    False, True, True
]
```

Both typed keys and field-name strings access merged batch fields. Separate
member keys for the same compatible field reach the same batch value; a foreign
record's key raises `KeyError`. A row contains only its selected member's fields:
`batch[0]` is `Bare`, not a `Charged` whose charge is zero or `None`.
Nullable fields declared by that member still restore `None` on absent rows.

## Understand matching and presence

The reader analyses the union before opening the input:

- Members must be siblings: no member may subclass another. They may share a
  base, as above, and must agree on the inherited `strict` setting.
- Shared file keys must have the same field name, type, optionality, shape/width
  and fill. A field name cannot mean different keys in different members.
- Keys required by every member, and non-nullable keys filled by every member,
  do not distinguish members. Among the remaining matching keys, a member must
  declare every key carried by the frame and require none the frame lacks.
  Unrelated file keys outside the union projection do not select a member.
- Every possible admitted signature must identify at most one member.
  Ambiguous unions raise `TypeError`; member order is not a first-match policy.
  An optional distinguishing field may admit both its present and absent
  signatures, making a union with an otherwise bare member ambiguous.

In a **single-record** batch only nullable fields carry absence masks;
non-nullable filled fields report present everywhere. In a **union** batch,
fields not declared by every member also carry masks, even when non-nullable.
More precisely, masks track the union's matching keys: this includes optional
fields shared by all members, but excludes non-nullable fields filled by all.
Thus `present(...)` is not a general provenance report for every source key.
It reports per-frame presence, not a per-atom mask or a test for a nonzero value.
`where(Charged)` and `present(Charged.charge)` happen to agree for the input
above; they differ if a `Charged` frame lacks its filled charge.

Float fields some members lack default to NaN in batch storage; the explicit
zero fill above intentionally demonstrates why values alone are insufficient.
Integer, boolean and string fields some members lack need an explicit `fill=`
before a batch read can bind. Adding a fill also changes which keys are required
for matching, so keep another unambiguous required discriminator.

## Errors, schemas and typing

With `conformance="required"` or `"strict"`, a frame no member admits raises
`SchemaError`. Under `"warn"` it warns and is dropped; wrong-kind fields and
missing required fields are dropped too. File selections still count source
frames, not surviving records. Batches retain surviving source indices in
`frame_indices`; an integer frame read selecting a dropped frame raises
`IndexError`. Frame-rule warnings alone do not drop a frame.

The [record restrictions](typed-records.md#read-restrictions-and-typing-boundaries)
also apply to unions: no `mode=` with `record=`, and `schema=` must agree with
the merged record fields rather than replacing their projection. Use
`RecordBatch[Sample]` (or a declared subclass with all merged batch fields) for
batch reads, and `Sample` for frame reads. `RecordBatch[Sample].spec()` describes
the merged projection, not a reusable member classifier for a plain `Batch`.
Union `from_batch` is rejected because a plain batch lacks the tags and presence
history needed to select members; read with `record=` instead.

Static typing covers union return types, row iteration and key lookups in the
existing `ty` cases. Narrow a row with `isinstance` or class-pattern matching
before accessing member-specific attributes. A `where()` selection does not
change the batch's static union parameter; `members` is typed as
`tuple[type[Record], ...]`, not an exact tuple of member types. See
[typing limits](typed-records.md#typing-limits) for the wider checker boundaries.

## Implementation and regression ownership

[`_record_union.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_record_union.py)
owns sibling checks, merged fields, ambiguity rejection and signature matching.
[`_record_bind.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_record_bind.py)
owns projection/presence plans, member construction and warning/drop parity;
[`_record_batch.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_record_batch.py)
owns merged storage, rows, masks, tags and selection. Reader integration lives in
[`_frames.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_frames.py) and
[`_batch.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_batch.py).

[`test_record_union.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_record_union.py)
covers signatures, invalid unions, tags, masks, filled member fields and warning
parity across frame/batch paths.
[`test_record_batch.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_record_batch.py)
covers general batch masks and gathers;
[`test_record_typing.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_record_typing.py)
owns static union and batch assertions. See the [API reference](../reference/api.md)
for public entry points.
