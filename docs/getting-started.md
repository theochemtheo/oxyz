# Getting started

Read an extxyz file into native frames, inspect its values, then write it back.
This walkthrough creates all its input locally; no dataset download or optional
converter is needed.

## Install

oxyz requires Python 3.12 or newer, and supports free-threaded CPython 3.14
(`3.14t`) without re-enabling the GIL. The base package depends on NumPy and
PyYAML:

=== "Package"

    ```sh title="Install the base package"
    pip install oxyz
    ```

=== "Source checkout"

    From a cloned checkout:

    ```sh title="Prepare the full development environment"
    uv sync --extra ase
    uv run python -c "import oxyz"
    ```

    Follow [development setup](development/index.md) and use `uv run` for Python
    so Rust edits rebuild the extension.

Optional extras are `ase`, `metatomic`, `torch-sim` and `s3`. Install only the
integrations you use, for example `pip install "oxyz[ase]"`.

## Create and read a trajectory

The first line of each frame is its atom count. The comment line declares
per-atom columns and frame metadata; the following lines contain the atom rows.
This file contains water followed by a hydrogen molecule:

```python title="Create and inspect molecules.extxyz" linenums="1" hl_lines="15-17 19-21"
from pathlib import Path

import numpy as np

import oxyz

path = Path("molecules.extxyz")
path.write_text(
    "3\nProperties=species:S:1:pos:R:3 energy=-7.25\n"
    "O 0 0 0\nH 0.75 0 0.5\nH -0.75 0 0.5\n"
    "2\nProperties=species:S:1:pos:R:3 energy=-1.0\n"
    "H 0 0 0\nH 0 0 0.75\n"
)

frames = oxyz.read(path)
assert isinstance(frames, list)
assert [frame.n_atoms for frame in frames] == [3, 2]

water = oxyz.read(path, 0)
assert isinstance(water, oxyz.Frame)
assert water.columns["species"] == ["O", "H", "H"]
np.testing.assert_array_equal(
    water.columns["pos"], [[0.0, 0.0, 0.0], [0.75, 0.0, 0.5], [-0.75, 0.0, 0.5]]
)
assert water.metadata["energy"] == -7.25
```

`read(path)` defaults to every frame. An integer selects one `Frame`; a slice,
slice string or index sequence selects a list. Numeric columns are NumPy arrays,
while string columns such as `species` are lists. File names are preserved:
`pos` does not become `positions`, and metadata remains separate from columns.
The [data model](concepts/data-model.md) explains normalised convenience views,
lookup and equality.

Short values can be inspected directly:

```python {.no-copy title="Inspect values in the Python console"}
>>> water.n_atoms
3
>>> water["energy"]
-7.25
>>> np.asarray(water["pos"]).shape
(3, 3)
```

## Inspect the file before choosing a workflow

Structural scanning reads frame boundaries and declared counts. Schema inference
parses the values and reports the observed fields and their consistency:

```python title="Inspect structure and observed schema"
index = oxyz.scan(path)
assert index.n_frames == 2
assert index.total_atoms == 5
schema = oxyz.infer_schema(path)
assert schema.n_frames == 2
assert schema.is_consistent
```

!!! note "A scan is not validation"

    A scan is not content validation, and reading a selected prefix does not
    validate later frames. See [storage and streaming](concepts/storage-and-streaming.md)
    for the cost and guarantees of each operation.

## Write and check the round trip

```python title="Verify this trajectory's round trip" hl_lines="4"
output = Path("molecules-copy.extxyz")
oxyz.write(output, frames)
restored = oxyz.read(output)
assert restored == frames
assert restored[0].metadata["energy"] == -7.25
np.testing.assert_array_equal(restored[1].columns["pos"], [[0, 0, 0], [0, 0, 0.75]])
```

!!! warning "Verify before replacing source data"

    Equality compares represented values, not original file bytes or formatting.
    This example verifies its own round trip; it is not a guarantee for every
    possible parsed value. The encoder may change whitespace, quoting and field
    order, and has [string-quoting limitations](guides/writing.md#what-a-round-trip-preserves).
    The writing guide also covers incremental output, codecs and append/stdout
    restrictions.

## Choose the next step

- [Read and select](guides/reading-and-selecting.md) for subsets, repeated
  indices and frame iteration.
- [Batch and stream](guides/batching-and-streaming.md) for concatenated arrays,
  training batches and memory strategies.
- [API reference](reference/api.md) for public objects and their full docstrings.
- [Format reference](reference/extxyz.md), [compatibility](reference/compatibility.md)
  and [errors](reference/errors.md) for syntax, interpretation and failure rules.
