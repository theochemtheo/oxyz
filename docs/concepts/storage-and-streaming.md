# Storage and streaming

Selection has two separate costs: finding frame boundaries and parsing frame
contents. Returning a small number of frames does not by itself guarantee
little I/O, a short scan, or bounded memory.

## Structural scan versus schema inference

`scan` reads count lines and skips the declared number of atom rows. Its
`FrameIndex` contains byte offsets and declared atom counts, not parsed columns.
It detects structural failures such as invalid counts and truncated frames,
but does not validate numeric atom values or infer column types. A false atom
count can shift the apparent boundary and cause an error in a later frame.
`scan(with_volume=True)` additionally reads comment lines for cell volumes;
without that option, `volumes` is `None`.

`infer_schema` parses every frame and folds the observed column kinds/widths,
metadata kinds/shapes, presence counts and consistency into a `Schema`. It
retains per-frame atom counts, but not all parsed frame values. It is the
whole-file parse to use when a successful partial read is insufficient.

```python
from pathlib import Path

import oxyz

source = Path("scan-versus-parse.extxyz")
source.write_text(
    "1\nProperties=species:S:1:pos:R:3\nC bad 0 0\n"
    "2\nProperties=species:S:1:pos:R:3\nH 0 0 0\nH 1 0 0\n"
)
index = oxyz.scan(source)
assert index.n_frames == 2
assert index.n_atoms.tolist() == [1, 2]
assert index.total_atoms == 3
assert index.volumes is None
raw = source.read_bytes()
assert [raw[int(offset):].splitlines()[0] for offset in index.offsets] == [b"1", b"2"]
try:
    oxyz.infer_schema(source)
except oxyz.ParseError as error:
    assert (error.frame_index, error.line, error.column) == (0, 3, 3)
else:
    raise AssertionError("schema inference must parse the bad coordinate")

selected = oxyz.read(source, [1])
assert selected[0].columns["species"] == ["H", "H"]
assert selected[0].columns["pos"].tolist() == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
```

The last selection succeeds because a seekable plain-file index parses only
the selected frame after scanning the structure. A forward stream selecting
frame 1 instead parses frame 0 on its way, and would encounter the bad value.
A scan is therefore neither a weaker spelling of `infer_schema` nor a promise
that every selected read uses the same validation path.

## How native Frame selection reaches its data

These rules describe reads without `schema=` or `record=`. The
[reading guide](../guides/reading-and-selecting.md) covers the selection grammar.

| Selection | Plain local file | Compressed or remote source |
| --- | --- | --- |
| Non-negative integer | Parse the prefix through that frame | Parse the prefix through that frame |
| Forward slice with non-negative bounds | Stream the prefix, parsing skipped frames too | Same forward streaming path |
| Negative integer | Scan the structure, seek and parse the selected frame | Parse and retain the whole file, then select |
| Slice with negative bounds or reverse step | Scan the structure, seek and parse selected frames | Parse and retain the whole file, then select |
| Explicit non-negative index sequence | Scan, then fetch in requested order, including repeats | Parse and retain the whole file, then gather |

Eager `read` with an unbounded forward slice parses the whole file, with
parallel parsing available. Even a stepped slice such as `"::2"` is not a
whole-file validation shortcut: the unselected frames are parsed too.
`iread` with a forward slice streams instead. An index sequence is eager in
both APIs.

Supplying a schema changes the native eager read path: the whole file is read
and checked before selecting. Forward `iread` validates while streaming;
negative, reverse and sequence paths require eager materialisation. `record=`
has its own projection and selected-frame validation rules; consult the public
[`read` and `iread` docstrings](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_frames.py)
rather than applying the table to records.

## Seekability and buffering

Plain local files support indexed seeking. Compressed inputs are decoded as
streams, without first decompressing the entire file to a temporary file.
Offsets from a compressed `scan` refer to the decompressed stream; they cannot
accelerate a subsequent read by seeking in the compressed file.

Remote sources are treated as streaming-only for native frame selection, even
when their underlying transport offers range reads. Plain, gzip and zstd
remote inputs arrive as byte chunks. Remote ZIP access uses a seekable reader
for the archive directory; tar codecs can open fresh streams for archive
processing. These transport details do not make a remote trajectory eligible
for native indexed frame selection. Archive inspection can require additional
reads before frame iteration starts.

`iread`'s largest-frame memory bound concerns forward frame parsing, not every
possible source and selection. Codec/transport buffers add overhead; `scan`
keeps an index proportional to the number of frames, and schema inference
keeps observed schema information and per-frame counts. An eager `read`, a
negative/reverse compressed or remote selection, an index sequence on those
sources, or a consumer that retains yielded frames can hold the whole dataset.
Streaming does not mean zero buffering or zero allocation.

For workloads requiring random-access batch strategies, use a plain local
copy rather than assuming a compressed or remote source can seek. Consult
[batching and streaming](../guides/batching-and-streaming.md) for strategy
restrictions and memory ownership, and [errors](../reference/errors.md) for
failure boundaries. No network or memory-usage measurement is implied by the
local examples here.

## Contract sources

- [`scan` and `FrameIndex`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_scan.py)
  and [`infer_schema`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_schema.py)
  define public inspection results.
- [Rust structural index tests](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-core/tests/index_scan.rs)
  cover offsets, lying counts, truncation and compressed-source restrictions.
- [Reader-entry tests](https://github.com/theochemtheo/oxyz/blob/main/tests/test_reader_entries.py)
  exercise chunked plain, gzip and archive inputs without a live remote store.
