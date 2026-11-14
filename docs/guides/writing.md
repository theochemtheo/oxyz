# Writing trajectories

`oxyz.write` accepts a native `Frame`, an ASE `Atoms`, or an iterable mixing them.
The base-package examples here use native frames only. A writable frame must
carry `species` and `pos` columns with row counts matching `n_atoms`.

## Choose one-shot or incremental output

Use `write` for data already in memory. It materialises the input iterable;
parallel serialisation also holds encoded frame text before output. It is not
the bounded-memory way to consume an unbounded generator.

Use `Writer` as a context manager when producing frames incrementally. Its
`write` method accepts the same single objects or iterables; close the writer
before reading the completed file. With the default `batch=None`, frames are
encoded as they arrive. `batch=n` buffers up to that many frames and serialises
each batch in parallel. Streaming-codec memory is bounded by the current frame
or buffered batch, not the trajectory; archive encoders can need additional
buffering.

This example writes two distinct energy values with an incremental writer:

```python
from pathlib import Path

import numpy as np

import oxyz

frames = [
    oxyz.Frame(
        n_atoms=1,
        columns={"species": ["He"], "pos": np.array([[0.0, 0.0, distance]])},
        metadata={"energy": energy},
    )
    for distance, energy in [(0.0, -0.25), (1.5, -0.125)]
]

with oxyz.Writer("helium.extxyz", batch=2) as writer:
    for frame in frames:
        writer.write(frame)

restored = oxyz.read("helium.extxyz")
assert restored == frames
assert [frame.metadata["energy"] for frame in restored] == [-0.25, -0.125]
np.testing.assert_array_equal(restored[1].columns["pos"], [[0.0, 0.0, 1.5]])
```

`write(..., threads=None)` uses all cores for serialisation; `threads=1` uses
the serial path. The output bytes are identical for the same destination and
options. Serialisation parallelises, not the final output stream. `Writer`
has a batch-size option rather than a `threads` option.

## Select a codec

The output filename selects the codec unless `compression=` overrides it.
The override names are the same names used for reading, but not every read
codec has an encoder:

| Output | Inferred suffix | Append | Restriction |
| --- | --- | --- | --- |
| Plain extxyz | `.xyz`, `.extxyz` | Yes | Also selected with `compression="none"` |
| gzip | `.gz` | Yes | `level` from 0 to 9 |
| ZIP | `.zip` | No | Finalise the archive before reading |
| tar | `.tar` | No | Finalise the archive before reading |
| gzip-compressed tar | `.tar.gz` | No | `level` from 0 to 9 |
| zstd, zstd-compressed tar | `.zst`, `.tar.zst` | No | Writing is unsupported |

Do not infer the output codec from existing file contents. `level` controls the
supported deflate-based codecs, not numeric precision.

```python
oxyz.write("helium.extxyz.gz", frames, level=6)
assert oxyz.read("helium.extxyz.gz") == frames

oxyz.write("helium.bin", frames, compression="gzip")
assert Path("helium.bin").read_bytes()[:2] == b"\x1f\x8b"
assert oxyz.read("helium.bin", compression="gzip") == frames
```

## Append only to concatenated streams

Plain and gzip files support appending. Keep the same schema when the resulting
trajectory is intended for a native batch; individual frame reads can otherwise
accept differing schemas.

```python
oxyz.write("continued.extxyz", frames[0])
oxyz.write("continued.extxyz", frames[1], append=True)
assert oxyz.read("continued.extxyz") == frames

oxyz.write("continued.extxyz.gz", frames[0])
oxyz.write("continued.extxyz.gz", frames[1], append=True)
assert oxyz.read("continued.extxyz.gz") == frames
```

Archive append and zstd writing raise rather than silently rewriting an archive
or choosing another codec:

```python
for destination, options in [
    ("unsupported.zip", {"append": True}),
    ("unsupported.tar", {"append": True}),
    ("unsupported.tar.gz", {"append": True}),
    ("unsupported.zst", {}),
    ("unsupported.tar.zst", {}),
]:
    try:
        oxyz.write(destination, frames, **options)
    except ValueError:
        pass
    else:
        raise AssertionError(f"unsupported output was accepted: {destination}")
```

## Write to stdout

`oxyz.write("-", frames)` writes extxyz to stdout. Do not mix logs into that
stream. Appending to stdout is rejected. To verify a complete stdout round trip
without printing file contents into the guide, capture a local child process:

```python
import subprocess
import sys

completed = subprocess.run(
    [
        sys.executable,
        "-c",
        "import oxyz; oxyz.write('-', oxyz.read('helium.extxyz'))",
    ],
    check=True,
    capture_output=True,
    text=True,
)
Path("stdout.extxyz").write_text(completed.stdout)
assert oxyz.read("stdout.extxyz") == frames

try:
    oxyz.write("-", frames, append=True)
except ValueError:
    pass
else:
    raise AssertionError("stdout append was accepted")
```

## What a round trip preserves

The encoder uses shortest-round-trippable real values. It writes columns in
`species`, `pos`, then remaining-column order, and comment fields in `Lattice`,
`pbc`, `Properties`, then remaining-metadata order. This is value preservation,
not original formatting preservation. See the [data model](../concepts/data-model.md)
for equality and the [format reference](../reference/extxyz.md) for represented
kinds and values.

An invalid input object raises `TypeError`; missing required columns, invalid
options and unsupported codecs are rejected. I/O failures raise `OSError`.
Consult [errors](../reference/errors.md) and the
[writer API](../reference/api.md#writing) before catching errors broadly.
