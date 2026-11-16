# Writing trajectories

`oxyz.write` accepts a native `Frame`, an ASE `Atoms`, or an iterable mixing them.
The base-package examples here use native frames only. A writable frame must
carry `species` and `pos` columns with row counts matching `n_atoms`.

## Choose one-shot or incremental output

Use `write` for data already in memory. It materialises the input iterable;
parallel serialisation also holds encoded frame text before output. It is not
the bounded-memory way to consume an unbounded generator.

Use `Writer` as a context manager when producing frames incrementally.

!!! warning "Iterable input is materialised"

    Call `writer.write(frame)` for each frame in a loop. Passing an iterable to
    a single `Writer.write` call materialises it as a list before encoding.
    Thus `writer.write(oxyz.iread(path))` retains the trajectory, and an unbounded
    iterable never reaches encoding. `batch=n` does not bound that input
    materialisation.

Close the writer before reading the completed file. With the default
`batch=None`, individual frames are encoded as they arrive. `batch=n` buffers up
to that many frames and serialises each batch in parallel. For individual-frame
calls and streaming codecs, retained parsed data is bounded by the current
frame or buffered batch, not the trajectory. Archive encoders buffer encoded
output as well.

This example writes two distinct energy values with an incremental writer:

```python title="Write individual frames incrementally" linenums="1" hl_lines="16-18"
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

A native `Batch` is also an iterable of frames, so both writer entry points
accept it. This convenience materialises its frame views for each call; use
individual-frame calls when consuming a streaming input:

```python title="Write a native Batch through either entry point"
native_batch = oxyz.read_batch("helium.extxyz")
oxyz.write("batch-output.extxyz", native_batch)
assert oxyz.read("batch-output.extxyz") == frames

with oxyz.Writer("batch-incremental.extxyz") as writer:
    writer.write(native_batch)
assert oxyz.read("batch-incremental.extxyz") == frames
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

```python title="Choose gzip by suffix or explicit override"
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

```python title="Append to plain and gzip streams"
oxyz.write("continued.extxyz", frames[0])
oxyz.write("continued.extxyz", frames[1], append=True)
assert oxyz.read("continued.extxyz") == frames

oxyz.write("continued.extxyz.gz", frames[0])
oxyz.write("continued.extxyz.gz", frames[1], append=True)
assert oxyz.read("continued.extxyz.gz") == frames
```

Archive append and zstd writing raise rather than silently rewriting an archive
or choosing another codec:

```python title="Check unsupported archive append and zstd output"
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

!!! warning "Stdout supports streams, not archives"

    `oxyz.write("-", frames)` writes plain extxyz to stdout; `compression="gzip"`
    can produce a gzip stream instead. Do not mix logs into that stream.
    Appending to stdout is rejected. ZIP, tar and gzip-compressed tar require a
    file path: attempting archive output to stdout raises `OSError`.

To verify a complete stdout round trip without printing file contents into the
guide, capture a local child process:

```python title="Capture stdout and check its restrictions"
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

for codec in ("zip", "tar", "tar.gz"):
    try:
        oxyz.write("-", frames, compression=codec)
    except OSError:
        pass
    else:
        raise AssertionError(f"archive stdout was accepted: {codec}")
```

## What a round trip preserves

The encoder uses shortest-round-trippable real values. It writes columns in
`species`, `pos`, then remaining-column order, and comment fields in `Lattice`,
`pbc`, `Properties`, then remaining-metadata order. Equality checks represented
values rather than original formatting. The examples above verify their own
round trips; they do not establish that every parsed value is writable losslessly.

!!! warning "String quoting can prevent a round trip"

    The current encoder has string-quoting limitations. For example, input metadata
    `note="x,y"` parses as one string, but writing it can omit the necessary quotes
    and produce `note=x,y`, which fails to parse. Do not assume arbitrary string
    metadata round-trips. Re-read output and compare before replacing source data.
    See the [data model](../concepts/data-model.md) for equality and the
    [format reference](../reference/extxyz.md) for represented kinds and values.

An invalid input object raises `TypeError`; missing required columns, invalid
options and unsupported codecs are rejected. I/O failures raise `OSError`.
Consult [errors](../reference/errors.md) and the
[writer API](../reference/api.md#writing) before catching errors broadly.
