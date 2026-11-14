# Reading and selecting frames

Use `oxyz.read` for an eager result and `oxyz.iread` to consume frames one at
a time. Native reads preserve column and metadata names; they do not apply
ASE's interpretation of positions, cells or calculator results. See the
[data model](../concepts/data-model.md) for stored and derived values.

## Create and inspect a trajectory

This small trajectory has distinct species, coordinates and energies so that
selection order is observable. Numeric columns are NumPy arrays; string columns
are lists.

```python
from pathlib import Path

import numpy as np
import oxyz

trajectory = Path("selection.extxyz")
trajectory.write_text(
    "1\nProperties=species:S:1:pos:R:3 energy=-2.5\nHe 0.25 0 0\n"
    "2\nProperties=species:S:1:pos:R:3 energy=-4.0\nLi 1 0 0\nH 2 0 0\n"
    "1\nProperties=species:S:1:pos:R:3 energy=-7.5\nNe 3 0 0\n"
)
frames = oxyz.read(trajectory)
assert isinstance(frames, list)
assert [frame.n_atoms for frame in frames] == [1, 2, 1]
assert frames[1].columns["species"] == ["Li", "H"]
positions = frames[1].columns["pos"]
assert isinstance(positions, np.ndarray)
assert positions.shape == (2, 3)
assert positions.dtype == np.dtype("float64")
assert positions.tolist() == [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
assert frames[1].metadata["energy"] == -4.0
```

This is the local-data version of the inspection workflow in
[`examples/read_numpy.py`](https://github.com/theochemtheo/oxyz/blob/main/examples/read_numpy.py).
Use `threads=1` for serial eager parsing; the default uses available cores.
Thread count does not change results or which parse error is reported.

## Choose a return type and order

An integer (including an integer string) returns one `Frame`. A slice,
slice string or sequence returns a list, even if only one frame is selected.
The default `":"` selects all frames. Slice stops are exclusive; omitted
bounds and negative bounds have Python slice semantics. Explicit index
sequences accept non-negative indices and preserve order and repetition.

```python
import oxyz

first = oxyz.read(trajectory, 0)
assert isinstance(first, oxyz.Frame)
assert first.columns["species"] == ["He"]
assert oxyz.read(trajectory, "2").metadata["energy"] == -7.5
assert oxyz.read(trajectory, -1).columns["species"] == ["Ne"]

one = oxyz.read(trajectory, slice(0, 1))
assert isinstance(one, list) and len(one) == 1
assert one[0] == first
assert [f.metadata["energy"] for f in oxyz.read(trajectory, "0:3:2")] == [-2.5, -7.5]
assert [f.metadata["energy"] for f in oxyz.read(trajectory, "::-1")] == [-7.5, -4.0, -2.5]
picked = oxyz.read(trajectory, [2, 0, 2])
assert [f.columns["species"] for f in picked] == [["Ne"], ["He"], ["Ne"]]
assert picked[0] == picked[2]
assert picked[0] is not picked[2]
```

An out-of-range integer raises `IndexError`; a slice is clipped to the available
frames. A zero slice step is invalid. The indexed plain-file path used above
returns separate frames for repeated indices. Eager fallback paths for
compressed/remote sources or `schema=` can reuse the same mutable `Frame` for
each occurrence of an index. Do not assume repeated results are independent:

```python
import gzip

compressed = Path("selection.extxyz.gz")
with gzip.open(compressed, "wt") as output:
    output.write(trajectory.read_text())
repeated = oxyz.read(compressed, [0, 0])
assert repeated[0] is repeated[1]
repeated[0].metadata["energy"] = 0.0
assert repeated[1].metadata["energy"] == 0.0
```

## Consume an iterator

`iread` always returns an iterator, including for a single integer. Forward
slices stream through the required prefix; skipped prefix frames are parsed too.
The file stays open until the iterator is dropped. Do not retain every yielded
frame if bounded memory is the objective.

```python
import oxyz

stream = oxyz.iread(trajectory, "1:3")
assert iter(stream) is stream
assert next(stream).columns["species"] == ["Li", "H"]
assert next(stream).metadata["energy"] == -7.5
sentinel = object()
assert next(stream, sentinel) is sentinel
assert next(stream, sentinel) is sentinel
assert [f.metadata["energy"] for f in oxyz.iread(trajectory, 0)] == [-2.5]
```

A sequence selection is eager, even through `iread`. Negative and reverse
selections have source-dependent costs, described in
[storage and streaming](../concepts/storage-and-streaming.md). For packed arrays
and frame/atom budgets, see [batching and streaming](batching-and-streaming.md).

## Know what a partial read validates

Without `schema=` or `record=`, a non-negative integer or bounded forward slice
parses only the required prefix. Malformed later frames can go unnoticed.
`oxyz.ase.read` has the same tested early-stop behaviour for `index=0` and
`index="0:2"`; this does not mean every ASE-style selector stops early.
Native `read` defaults to all frames, whereas `oxyz.ase.read` defaults to the
last frame. Do not confuse `oxyz.ase` with the separate `ase.io` implementation.

```python
from pathlib import Path

import oxyz

broken_tail = Path("selection-broken-tail.extxyz")
broken_tail.write_text(trajectory.read_text() + "not-a-count\n")
assert oxyz.read(broken_tail, 0).metadata["energy"] == -2.5
assert [f.n_atoms for f in oxyz.read(broken_tail, ":2")] == [1, 2]
try:
    oxyz.infer_schema(broken_tail)
except oxyz.ParseError as error:
    assert error.frame_index == 3
else:
    raise AssertionError("whole-file inference must inspect the broken tail")
```

A structural index may still inspect the whole file before selectively parsing
frames. In particular, plain-file negative indices, reverse slices and index
lists do not promise to ignore structural damage in the tail. Eager native
`read(..., schema=...)` parses and checks the whole file before selection;
forward `iread(..., schema=...)` checks as it streams. Record reads have a
separate selected-frame checking contract in the
[`read` docstring](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_frames.py).
Use `infer_schema` for a whole-file parse rather than treating a successful
partial read as validation. See [errors](../reference/errors.md) for locations
and iterator behaviour after failure.
