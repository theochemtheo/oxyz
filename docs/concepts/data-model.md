# The data model

oxyz separates the file's stored values from an application's interpretation of
them. NumPy arrays are the native numeric output, not an intermediate form that
requires conversion to another package.

## Four containers, two representations

| Container | One entry represents | Access to data | Selection |
| --- | --- | --- | --- |
| `Frame` | One parsed frame | `columns`, `metadata`, or unambiguous string key | No frame-sequence indexing |
| `Batch` | Several compatible frames | Atom-major `columns`, frame-major `metadata`, or string key | Integer gives `Frame`; slice or gather gives `Batch` |
| `Record` subclass | One frame projected onto declared fields | Named attributes | No frame-sequence indexing |
| `RecordBatch[R]` | Several records, including supported unions | Declared field keys or field-name strings | Integer gives a record; slice or gather gives a record batch |

`Frame` and `Batch` preserve file keys: `force` and `forces` are distinct, and
there is no implicit aliasing or calculator state. Numeric and boolean per-atom
columns are arrays; string columns are lists. Metadata scalars are typed Python
values, numeric/boolean metadata arrays are NumPy arrays, and string arrays use
lists. Both dictionaries retain stored order; repeated metadata keys retain the
last value.

A `Record` instead names the fields an application needs and declares their
file keys, kinds, widths and shapes. Reading with `record=R` projects onto those
fields: it is not a lossless copy of every input key. `Cell` and `Matrix3` fields
also express matrix interpretation. Optional record fields may be `None` for an
individual frame. A `RecordBatch` holds dense filled values and presence masks
rather than `None`; selecting a row restores `None` where appropriate. A union
batch additionally carries member tags so indexing restores the matching record
class. See the [API reference](../reference/api.md) for the public declarations.

## Stored values and derived accessors

A frame's geometry accessors compute an interpretation on demand without
rewriting the dictionaries:

- `positions` returns the stored `pos` array, without copying.
- `numbers` resolves atomic numbers from `Z`, then `numbers`, then `species`.
- `symbols` prefers the stored `species` column, otherwise derives symbols.
- `cell` interprets flat Fortran-order `Lattice` as a matrix with lattice
  vectors in rows; it defaults to a zero matrix when there is no lattice.
- `pbc` reads or broadcasts the stored flags, defaulting to all true with a
  lattice and all false without one.

These accessors are not cached. They can raise `FieldError` when the required
stored data is missing or cannot be interpreted. A species token may be a
force-field label rather than a chemical symbol, for example; storing it does
not require that it can be converted to an atomic number.

```python
import numpy as np
import oxyz

stored_lattice = np.arange(1.0, 10.0)
frame = oxyz.Frame(
    n_atoms=1,
    columns={"species": ["C"], "pos": np.array([[0.5, 1.0, 1.5]])},
    metadata={"Lattice": stored_lattice, "energy": -8.0},
)
assert frame.positions is frame.columns["pos"]
assert frame.metadata["Lattice"] is stored_lattice
np.testing.assert_array_equal(frame.cell, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
np.testing.assert_array_equal(frame.numbers, [6])
np.testing.assert_array_equal(frame.pbc, [True, True, True])
assert frame.metadata["Lattice"].shape == (9,)
assert "pbc" not in frame.metadata
```

## Mapping-style access is not a mapping protocol

`Frame` and `Batch` support `obj["name"]`, `"name" in obj`, `get`, `keys` and
`dict(obj)` across columns and metadata. Returned values are the stored objects,
not copies. A missing name raises `KeyError`; `get` returns its default only for
an absent name. A key present on both sides raises `AmbiguousNameError` on lookup
or `get`, rather than silently choosing a side. Membership still returns true.
Resolve the ambiguity with `.columns[name]` or `.metadata[name]`.

A frame deliberately has no `len` and is not iterable. Use `frame.n_atoms`, not
`len(frame)`. A batch's length and iteration concern its *frames*, not its keys:
`len(batch)` is `n_frames`, `list(batch)` gives frames, and `batch.keys()` gives
names. Neither container is a `collections.abc.Mapping`.

```python
import numpy as np
import oxyz

assert frame["pos"] is frame.positions
assert frame.get("absent", "not stored") == "not stored"
assert frame.keys() == ["species", "pos", "Lattice", "energy"]
assert dict(frame)["Lattice"] is stored_lattice

collision = oxyz.Frame(
    n_atoms=1,
    columns={"score": np.array([0.25])},
    metadata={"score": 9.0},
)
assert "score" in collision
assert collision.columns["score"].tolist() == [0.25]
assert collision.metadata["score"] == 9.0
try:
    collision.get("score", 0.0)
except oxyz.AmbiguousNameError as error:
    assert error.name == "score"
else:
    raise AssertionError("a shared name must not silently select one axis")
```

Records do not acquire this two-dictionary lookup interface. Use the declared
attribute on a record. On a record batch, `batch[R.field]` is the typed field-key
form and `batch["field"]` uses the *declared field name*, not necessarily the
file key. A key belonging to an unrelated record is rejected.

## Frozen bindings, mutable contents

All four container families are frozen dataclasses: assigning a replacement to
an instance field is prohibited. This is shallow freezing, not deep immutability.
`Frame` and `Batch` dictionaries, lists and writable arrays can still be edited.
Those edits affect any object sharing the same storage.

Records distinguish construction from conversion:

| Origin | Array ownership and writeability |
| --- | --- |
| Direct record or record-batch constructor | Checks values but retains supplied arrays as given; does not copy or freeze them |
| `R.from_frame(frame)` | Unchanged arrays become read-only views; string lists are copied |
| `RecordBatch[R].from_batch(batch)` | Unchanged arrays, including layout arrays, become read-only views; string lists are copied |
| `read`/`iread` with `record=R` | Record arrays are writable and independently owned |
| `read_batch`/`iread_batch` with a record batch class | Batch arrays are writable and independently owned |

Read-only views prevent writes *through that view*. They do not prevent another
owner from changing shared storage. Take `.copy()` of an array when an
independent snapshot is required. Conversions that change kind or representation
may allocate rather than share memory.

```python
from dataclasses import FrozenInstanceError

import numpy as np
from numpy.typing import NDArray
import oxyz

class Geometry(oxyz.Record):
    coordinates: oxyz.Column[NDArray[np.float64]] = oxyz.Column("pos", width=3)
    labels: oxyz.Column[list[str]] = oxyz.Column("species")
    energy: oxyz.Meta[float] = oxyz.Meta()

record = Geometry.from_frame(frame)
assert record.coordinates.shape == (1, 3)
assert record.energy == -8.0
assert np.shares_memory(record.coordinates, frame.positions)
assert not record.coordinates.flags.writeable
assert record.labels == ["C"]
assert record.labels is not frame.columns["species"]
snapshot = record.coordinates.copy()
frame.positions[0, 0] = 2.5
assert record.coordinates[0, 0] == 2.5
assert snapshot[0, 0] == 0.5
try:
    record.energy = 0.0
except FrozenInstanceError:
    assert record.energy == -8.0
else:
    raise AssertionError("record fields must remain frozen")

supplied = np.array([[7.0, 8.0, 9.0]])
constructed = Geometry(n_atoms=1, coordinates=supplied, labels=["N"], energy=-1.0)
assert constructed.coordinates is supplied
assert constructed.coordinates.flags.writeable
```

## Batch views and gathers

Both batch families use CSR `offsets` and retain original file `frame_indices`.
Their numeric selection rules follow NumPy's distinction between basic and
advanced indexing:

- An integer selects one frame or record, with per-atom and array-metadata views;
  scalar metadata is unboxed to a Python value.
- A unit-step slice selects a batch of views and rebases offsets to start at zero.
  The rebased offsets are a new array, not a view of the original offsets.
- A stepped slice, integer sequence or boolean mask gathers a new batch and
  copies numeric values. A gather may repeat and reorder frames.
- Negative positions count from the end. Out-of-range positions and masks of
  the wrong length raise `IndexError`.

String data is stored in lists, so selections make new lists rather than NumPy
views. Each selected frame also has its own dictionaries. Sharing refers to
array storage, not identity of the frame or its dictionaries.

The following hand-built batch isolates these memory relationships from file
reading and batching strategy:

```python
import numpy as np
import oxyz

layout = oxyz.Batch(
    columns={"pos": np.array([[10.0, 0, 0], [20.0, 0, 0], [21.0, 0, 0]])},
    metadata={"energy": np.array([-1.0, -2.0])},
    offsets=np.array([0, 1, 3], dtype=np.int64),
    frame_indices=np.array([4, 9], dtype=np.int64),
)
row = layout[-1]
tail = layout[1:]
gathered = layout[[1, 0, 1]]
assert np.shares_memory(row.positions, layout["pos"])
assert np.shares_memory(tail["energy"], layout["energy"])
assert not np.shares_memory(tail.offsets, layout.offsets)
assert not np.shares_memory(gathered["pos"], layout["pos"])
assert tail.offsets.tolist() == [0, 2]
assert gathered.frame_indices.tolist() == [9, 4, 9]
assert gathered["pos"][:, 0].tolist() == [20.0, 21.0, 10.0, 20.0, 21.0]
row.positions[0, 0] = 22.0
assert tail[0].positions[0, 0] == 22.0
assert gathered[0].positions[0, 0] == 20.0
```

For file-based examples and atom-to-frame mapping, see
[batching and streaming](../guides/batching-and-streaming.md).

## Value equality and representations

Equality returns one boolean rather than NumPy's elementwise result:

- Frames compare atom count and all stored column/metadata values. Dictionary
  key order is ignored; array shapes and values matter.
- Batches additionally compare offsets and file frame indices. Identical atom
  values with different provenance are not equal batches.
- Records require the same concrete record class, atom count and field values.
- Record batches require the same concrete batch class, layout, field values,
  presence masks and member tags.

NaN compares equal to NaN in corresponding positions. These mutable-value
containers are unhashable: do not use them as dictionary keys or set members.
Value equality does not imply shared storage or object identity.

```python
import numpy as np
import oxyz

left = oxyz.Frame(
    n_atoms=1,
    columns={"q": np.array([np.nan])},
    metadata={"step": 3, "energy": -6.0},
)
right = oxyz.Frame(
    n_atoms=1,
    columns={"q": np.array([np.nan])},
    metadata={"energy": -6.0, "step": 3},
)
assert left == right
assert left is not right
assert not np.shares_memory(left["q"], right["q"])
right.columns["q"][0] = 0.0
assert left != right
```

Text representations summarise shapes and dtypes without dumping frame or batch
arrays. Record representations also show scalar field values. Notebook HTML
representations provide bounded previews. Neither representation replaces
explicit value inspection or equality checks.

## Contract sources

- [Frame implementation](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_frames.py)
  and [API tests](https://github.com/theochemtheo/oxyz/blob/main/tests/test_frame_api.py)
- [Record implementation](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_record.py)
  and [record tests](https://github.com/theochemtheo/oxyz/blob/main/tests/test_record.py)
- [RecordBatch implementation](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_record_batch.py)
- [Batch sequence tests](https://github.com/theochemtheo/oxyz/blob/main/tests/test_batch_sequence.py)
- [Representation tests](https://github.com/theochemtheo/oxyz/blob/main/tests/test_repr.py)
- [Lossless core decision](../development/decisions/lossless-core-and-layering.md)
- [Typed records and unions decision](../development/decisions/typed-records-and-unions.md)
