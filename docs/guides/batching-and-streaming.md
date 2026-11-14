# Batching and streaming

Use `read_batch` to gather frames into one array-native `Batch`, or
`iread_batch` to consume a trajectory in batches. Numeric and boolean columns
are NumPy arrays; strings remain Python lists. No converter or optional
machine-learning framework is required.

## Inspect the atom-major layout

A batch concatenates each per-atom column. Its metadata has a leading frame
axis. Frames need not have the same number of atoms: the CSR row boundaries
record where each frame starts and ends.

This local trajectory has two, one and three atoms, with compatible columns
and metadata throughout. Its first energy is an integer in the file; batching
promotes the integer/real mixture to floating point.

```python title="Inspect atom-major arrays and frame provenance" linenums="1" hl_lines="19-23"
from pathlib import Path

import numpy as np
import oxyz

trajectory = Path("batch-input.extxyz")
trajectory.write_text(
    "2\nProperties=species:S:1:pos:R:3 energy=-2\n"
    "He 0.25 0 0\nHe 0.75 0 0\n"
    "1\nProperties=species:S:1:pos:R:3 energy=-0.5\n"
    "Ne 1.5 0 0\n"
    "3\nProperties=species:S:1:pos:R:3 energy=-4.5\n"
    "Ar 2 0 0\nAr 3 0 0\nAr 4 0 0\n"
)
batch = oxyz.read_batch(trajectory)
assert isinstance(batch, oxyz.Batch)
assert len(batch) == batch.n_frames == 3
assert batch.total_atoms == 6
assert batch.ptr is batch.offsets
np.testing.assert_array_equal(batch.ptr, [0, 2, 3, 6])
np.testing.assert_array_equal(batch.n_atoms, [2, 1, 3])
np.testing.assert_array_equal(batch.frame_indices, [0, 1, 2])
np.testing.assert_array_equal(batch.batch, [0, 0, 1, 2, 2, 2])
assert batch.columns["pos"].shape == (6, 3)
assert batch.metadata["energy"].dtype == np.float64
np.testing.assert_array_equal(batch["energy"], [-2.0, -0.5, -4.5])
np.testing.assert_array_equal(batch["pos"][:, 0], [0.25, 0.75, 1.5, 2, 3, 4])
assert batch["species"] == ["He", "He", "Ne", "Ar", "Ar", "Ar"]
assert list(batch) == oxyz.read(trajectory)
assert [frame.n_atoms for frame in batch] == [2, 1, 3]
assert batch[1].metadata["energy"] == -0.5
```

Frame `i` occupies `ptr[i]:ptr[i + 1]` of every column. `batch.batch` maps each
atom to its frame's position *within this batch*, not its file index.
`frame_indices` supplies the latter. The atom-to-frame vector is recomputed on
each access; retain it locally when reusing it in a calculation.

A selection can reorder or repeat source frames. It does not renumber their
provenance:

```python title="Reorder and repeat source frames"
import numpy as np
import oxyz

picked = oxyz.read_batch(trajectory, [2, 0, 2])
np.testing.assert_array_equal(picked.frame_indices, [2, 0, 2])
np.testing.assert_array_equal(picked.ptr, [0, 3, 5, 8])
np.testing.assert_array_equal(picked.batch, [0, 0, 0, 1, 1, 2, 2, 2])
np.testing.assert_array_equal(picked["energy"], [-4.5, -2.0, -4.5])
assert picked[0] == picked[2] == batch[2]
```

The [data model](../concepts/data-model.md) explains when indexing shares arrays
and when it copies them. File selection is covered in
[reading and selecting](reading-and-selecting.md).

## Choose a batching strategy

Give `iread_batch` exactly one of these strategies:

| Strategy | Behaviour |
| --- | --- |
| `frames_per_batch=N` | Take at most `N` frames per batch; the final batch may be smaller |
| `atoms_per_batch=N` | Greedily add whole frames in file order, or shuffled order, until the next would exceed `N` atoms |
| `memory_scales_with=...`, `max_scaler=S` | Pack balanced bins using an estimated per-frame weight with budget `S` |

!!! warning "Budgets do not split frames"

    A frame is never split. A frame over the atom or weight budget is emitted in
    a batch of its own, so these budgets are not strict upper bounds on allocation.

```python title="Compare frame-count, atom-budget and balanced packing"
import numpy as np
import oxyz

chunks = list(oxyz.iread_batch(trajectory, frames_per_batch=2, threads=1))
assert [part.frame_indices.tolist() for part in chunks] == [[0, 1], [2]]
assert [part.total_atoms for part in chunks] == [3, 3]
assert chunks[0][1]["species"] == ["Ne"]

atom_chunks = list(oxyz.iread_batch(trajectory, atoms_per_batch=3))
assert [part.frame_indices.tolist() for part in atom_chunks] == [[0, 1], [2]]
np.testing.assert_array_equal(atom_chunks[0]["energy"], [-2.0, -0.5])
oversized = list(oxyz.iread_batch(trajectory, atoms_per_batch=2))
assert [part.frame_indices.tolist() for part in oversized] == [[0], [1], [2]]
assert oversized[-1].total_atoms == 3

bins = list(oxyz.iread_batch(
    trajectory, memory_scales_with="n_atoms", max_scaler=3
))
assert [part.frame_indices.tolist() for part in bins] == [[2], [0, 1]]
np.testing.assert_array_equal(bins[1]["pos"][:, 0], [0.25, 0.75, 1.5])
```

!!! note "Balanced packing can change frame order"

    Balanced packing uses best-fit-decreasing weights, so it may reorder frames
    even without shuffling. `"n_atoms"` weighs each frame by atom count.
    `"n_atoms_x_density"` weighs it by `n_atoms**2 / volume`, a proxy for
    neighbour-graph memory. Missing, non-positive or non-finite volumes fall back
    to atom count. This is an estimate, not a measurement of a model's memory use;
    `max_scaler` is a weight budget, not bytes. Balanced packing cannot be combined
    with `shuffle`.

## Reproducible NumPy consumption

`shuffle=True` permutes frames before frame-count or greedy atom packing.
Use `seed` only with shuffling. The same input, strategy and seed produce the
same composition regardless of `threads`; `threads=1` parses serially and
`None` uses all available cores. A seed does not fix a batch's atom count.

This is the NumPy part of a training-style loop: positions are atom-major,
energies are frame-major, and provenance remains available after shuffling.

```python title="Consume seeded batches while retaining provenance"
import numpy as np
import oxyz

serial = list(oxyz.iread_batch(
    trajectory, atoms_per_batch=3, shuffle=True, seed=7, threads=1
))
parallel = list(oxyz.iread_batch(
    trajectory, atoms_per_batch=3, shuffle=True, seed=7, threads=2
))
assert serial == parallel
seen = []
energies_by_file = np.empty(3)
for part in serial:
    positions = part.columns["pos"]
    energies = part.metadata["energy"]
    atom_to_frame = part.batch
    assert positions.shape == (part.total_atoms, 3)
    assert energies.shape == (part.n_frames,)
    assert atom_to_frame.shape == (part.total_atoms,)
    np.testing.assert_array_equal(positions[:, 1:], np.zeros((part.total_atoms, 2)))
    energies_by_file[part.frame_indices] = energies
    seen.extend(part.frame_indices.tolist())
assert sorted(seen) == [0, 1, 2]
np.testing.assert_array_equal(energies_by_file, [-2.0, -0.5, -4.5])
```

## Compatible frames, not just valid frames

A plain batch requires the same column names and metadata keys across its
frames, compatible kinds, and matching stored widths. Column order may differ;
integer/real mixtures promote to real values. Varying atom counts are compatible.
Missing or additional keys, incompatible kinds and differing widths are not.
Individual frames can each be valid extxyz while being incompatible in one batch.

```python title="Detect incompatible metadata within a batch"
from pathlib import Path

import oxyz

drift = Path("batch-drift.extxyz")
drift.write_text(
    "1\nProperties=species:S:1:pos:R:3 energy=-3.0\nKr 5 0 0\n"
    "1\nProperties=species:S:1:pos:R:3\nKr 6 0 0\n"
)
try:
    oxyz.read_batch(drift)
except ValueError as error:
    assert "missing metadata" in str(error)
else:
    raise AssertionError("incompatible metadata must not form one plain batch")
singletons = list(oxyz.iread_batch(drift, frames_per_batch=1, threads=1))
assert singletons[0][0]["energy"] == -3.0
assert "energy" not in singletons[1].metadata
assert singletons[1][0]["pos"].tolist() == [[6.0, 0.0, 0.0]]
```

!!! tip "Validation is not projection"

    Compatibility is enforced within each batch, not as a whole-file schema audit.
    Changing batch boundaries can therefore expose a mismatch. A declared `schema`
    validates frames before concatenation; validation alone does not make
    incompatible frames compatible. Explicit `mode="project"` reshapes them to the
    schema. The [API reference](../reference/api.md) lists these options and typed
    record batching.

## Memory and source constraints

For an uncompressed local file, atom-budget, shuffled and balanced strategies
scan and index the file before parsing batches. Index and plan storage grows
with frame count; only the selected batch's atom arrays are materialised at a
time. Retaining yielded batches, frames or array views retains their underlying
data. The examples above collect tiny results for assertions; for a large file,
consume each batch directly in a loop instead of calling `list`.

For fixed frame-count, file-order consumption, `threads=1` uses the sequential
path without a preliminary scan. This bounds retained parsed data by the batch
window rather than the entire trajectory, but a frame-count budget alone cannot
bound atoms when frame sizes vary. Use `iread` when only one frame at a time is
needed.

Compressed and remote sources support only `frames_per_batch` without shuffle
in `iread_batch`, even when a particular underlying stream could seek. Atom
budgets and balanced packing are not available for these sources. Download
remote data and decompress it to a local uncompressed file before requesting
random-access strategies. Source buffering and archive behaviour are described
in [storage and streaming](../concepts/storage-and-streaming.md); an iterator is
not a promise that every codec uses constant internal memory.

These functions accept paths and supported URLs, not arbitrary Python file-like
objects. Do not infer random-access support from a source merely being readable:
planned strategies need the local file index, while non-seekable input requires
a sequential consumption path.

## Contract sources

- [Batch API and planning](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_batch.py)
- [Batch contract tests](https://github.com/theochemtheo/oxyz/blob/main/tests/test_batch.py)
- [Batch sequence tests](https://github.com/theochemtheo/oxyz/blob/main/tests/test_batch_sequence.py)
- [Training-loop example](https://github.com/theochemtheo/oxyz/blob/main/examples/batch_training_loop.py)
