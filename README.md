# oxyz

[![test](https://github.com/theochemtheo/oxyz/actions/workflows/test.yml/badge.svg)](https://github.com/theochemtheo/oxyz/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/oxyz)](https://pypi.org/project/oxyz/)

Fast, schema-aware [extxyz](https://github.com/libAtoms/extxyz) reading for
atomistic machine learning. A Rust parser behind a small, typed Python API:
numpy arrays out, `ase.Atoms` on request, and a one-pass schema report that
tells you whether a training file is what you think it is.

```python
import oxyz

frames = oxyz.read_frames("train.extxyz")        # all cores, one pass
frames[0].columns["pos"]                         # float64 ndarray, shape (n_atoms, 3)
frames[0].metadata["energy"]                     # float

schema = oxyz.infer_schema("train.extxyz")
schema.is_consistent                             # False — now you know before training
print(schema)                                    # which keys drift, and in how many frames
```

`oxyz` exists for the gap between "extxyz is the lingua franca of atomistic
ML datasets" and "every Python extxyz reader is slow enough to matter".
Reading a dataset into numpy is 12–25× faster than `ase.io.read` on the
benchmarks below; reading it into `ase.Atoms` objects is 2.4–3.5× faster.
The same single pass can also tell you the dataset's schema — which columns
and metadata keys appear, with what types and shapes, and how consistently
— which is the part of dataset ingestion that usually goes unchecked.

Pre-1.0: minor versions may change the API.

## Install

```sh
pip install oxyz            # numpy is the only dependency
pip install "oxyz[ase]"     # adds ASE conversion (ase >=3.23,<4)
```

Wheels cover CPython ≥3.11 on Linux (x86_64, aarch64), macOS (arm64,
x86_64), and Windows (x64).

Installing puts an `oxyz` command on the path; `oxyz scan train.extxyz`
summarises a file without writing any Python. It also runs without
installing, via `uvx oxyz scan train.extxyz`.

## In place of ASE

`oxyz.ase.read` and `oxyz.ase.iread` are drop-ins for `ase.io.read` /
`ase.io.iread` on extxyz files, including ASE's full index grammar
(`-1`, `"::2"`, slices):

```python
import oxyz.ase

atoms = oxyz.ase.read("train.extxyz")             # last frame, like ase.io.read
images = oxyz.ase.read("train.extxyz", ":")       # every frame
for atoms in oxyz.ase.iread("train.extxyz", "::10"):
    ...
```

The conversion reuses `ase.io.extxyz`'s own routing tables and
`set_calc_and_arrays`, so key handling (which results go to the
calculator, which to `arrays`) agrees with ASE by construction; golden
tests hold the two readers equal on the test corpus apart from the
divergences below. Reads are lazy:
`read(path, 3)` parses four frames and stops, and negative or reverse
selections resolve through a structural scan and seek rather than a full
parse — `read(path)` on a long trajectory does not parse the whole file
to return the last frame.

## Divergences from ASE

`oxyz.ase.read` matches `ase.io.read` field for field on the test corpus
except for the cases below — two deliberate, two that follow from
honouring the extxyz grammar and oxyz's typed model where ASE's parser
does not.

**Deliberate** — an error or an acceptance, never a silently different
value:

- **Voigt stress.** 6-component `stress` is accepted and routed to the
  calculator; ASE's comment parser rejects the file.
- **Non-symbol species.** A species that is not a chemical symbol raises
  an error; ASE builds a nonsense `Atoms`.

**Grammar and typing** — a different value, no error:

- **New-style string arrays.** `tags=["a","b"]` is typed as `list[str]`;
  ASE keeps the one raw string `'"a","b"'`.
- **Single-quoted values.** The grammar makes `"` the only quote
  character, so `label='hello'` keeps its quotes and `note=it's` keeps its
  apostrophe; ASE strips the single quotes (and reads `it's` as `its`).

## What you get beyond ASE

**Array-native frames.** A `Frame` is a frozen dataclass holding the file's
columns as numpy arrays and its comment-line metadata as typed Python
values — no per-atom Python objects, no calculator indirection. Names and
values are kept exactly as written: no `force`/`forces` aliasing, no
reordering, `Lattice` stays the flat 9-value array from the file.
Normalisation is the ASE layer's job (or yours).

**Batches in the PyG layout.** `Batch` concatenates frames atom-major,
CSR-style: every per-atom column is one dense array of `total_atoms` rows,
frame `i` occupying rows `offsets[i]:offsets[i+1]`; per-frame metadata
stacks into arrays of `n_frames` rows. `batch.ptr` and `batch.batch` carry
their PyTorch Geometric names, and `torch.from_numpy(batch.columns["pos"])`
is zero-copy, so the path into a training loop is short.

```python
for batch in oxyz.iter_batches("bulk.extxyz", atoms_per_batch=4096,
                               shuffle=True, seed=0):
    batch.columns["forces"]        # (total_atoms, 3)
    batch.metadata["energy"]       # (n_frames,)
    batch.frame_indices            # which file frames these are — provenance
```

`iter_batches` packs by frame count or by a total-atom budget, in file
order or seeded-shuffled. Batch composition depends only on the file, the
knobs, and the seed — never on `threads`.

**Schema inference.** `infer_schema` folds the whole file into a `Schema`:
per-column and per-metadata-key observed variants (kind, width or shape,
and how many frames used each), presence counts, a strict `is_consistent`,
and per-entry `unified` — the single type an Int/Real drift can be
promoted to, or `None` when the conflict is genuine. The classic failure
it catches: a generator script that writes isolated-atom frames with
integer forces and no `Lattice` into an otherwise uniform bulk dataset.
The same pass keeps the per-frame atom counts, so a `Schema` also reports
the atom-count distribution (`mean_atoms`, `median_atoms`, `std_atoms`,
alongside the min/max above) without a second read of the file.

```text
>>> print(oxyz.infer_schema("train.extxyz"))
1000 frames, 63841 atoms (min 1, max 96)

per-atom columns:
  species: S:1 (1000/1000 frames)
  pos: R:3 (1000/1000 frames)
  forces: I:3 (5/1000 frames), R:3 (995/1000 frames) (unifies to R:3)

metadata:
  energy: Real (1000/1000 frames)
  Lattice: RealArray[9] (995/1000 frames)
```

**Structural scanning.** `oxyz.scan` reads only the frame skeleton — byte
offsets and declared atom counts — without parsing any contents. It is the
cheap first question to ask of an unfamiliar file (5 ms for a 22 MiB file
below) and the machinery behind random access, shuffled batching, and
lazy negative indexing. The same statistics, alongside the inferred
schema, are a terminal away with `oxyz scan` (see [Command line](#command-line)).

**Parallelism as a knob, not a mode.** Readers take `threads`: `None`
parses on every core, `1` is the exact serial streaming path. Results and
errors are identical either way — the parallel path is held to the serial
path's behaviour by parity tests, not by intention.

## Command line

Installing oxyz provides an `oxyz` command for inspecting files from the
shell; `uvx oxyz` runs it without installing anything.

```sh
oxyz scan train.extxyz
```

`scan` prints per-frame atom-count statistics followed by the inferred
schema. Unlike the `oxyz.scan` primitive, which parses nothing, the command
reads the whole file to infer the schema; `--no-schema` drops back to the
cheap structural pass and reports only the statistics. `--json` emits a
single `{"stats": ..., "schema": ...}` object for piping into other tools.

```text
$ oxyz scan train.extxyz
frames:      3
atoms total: 6
atoms/frame: min 1  max 3  mean 2.00  median 2.00  std 0.82

3 frames, 6 atoms (min 1, max 3)

per-atom columns:
  species: S:1 (3/3 frames)
  pos: R:3 (3/3 frames)
  forces: R:3 (3/3 frames)

metadata:
  Lattice: IntArray[9] (3/3 frames)
  energy: Real (3/3 frames)
```

## Performance

Timings below are means over repeated rounds — each case gets a
one-second budget, at least five rounds, median 24 in this run — on an
Apple M3 Pro under CPython 3.13. Full tables with standard deviations,
the environment, and the fixture definitions are in
[benchmarks/RESULTS.md](https://github.com/theochemtheo/oxyz/blob/main/benchmarks/RESULTS.md);
[benchmarks/run.py](https://github.com/theochemtheo/oxyz/blob/main/benchmarks/run.py)
reproduces them.

Whole-file reads to numpy (`oxyz.read_frames` vs [cextxyz], the libAtoms C
parser, via its `read_dicts`):

| workload | oxyz | oxyz `threads=1` | cextxyz |
| --- | ---: | ---: | ---: |
| 2 000 small frames | **9.7 ms** | 20.1 ms | 219 ms |
| 4 × 100 000 atoms | **37.5 ms** | 72.5 ms | 91.6 ms |
| 2 000 frames, heavy metadata | **13.8 ms** | 28.8 ms | 363 ms |
| MACE-style mixed file | **7.0 ms** | 14.8 ms | 126 ms |

Whole-file reads to `ase.Atoms` (`oxyz.ase.read` vs the [ase-extxyz] plugin
wrapping the same C parser, vs `ase.io.read`):

| workload | oxyz.ase | ase-extxyz | ase |
| --- | ---: | ---: | ---: |
| 2 000 small frames | **86 ms** | 106 ms | 206 ms |
| 4 × 100 000 atoms | 174 ms | **89 ms** | 442 ms |
| 2 000 frames, heavy metadata | **100 ms** | 246 ms | 349 ms |
| MACE-style mixed file | **54 ms** | 70 ms | 147 ms |

Losses included: the C parser's Atoms construction wins on dense
100k-atom frames, where conversion rather than parsing dominates and
oxyz pays for its untouched-raw-data model. On selective reads (every
20th frame of the small-frames file) `oxyz.read_batch` takes 1.7 ms
against 24 ms for ASE; on peak memory, streaming `iter_frames` through
the small-frames file grows RSS by 12 MiB where `ase.io.iread` grows it
by 56 MiB
([benchmarks/MEMORY.md](https://github.com/theochemtheo/oxyz/blob/main/benchmarks/MEMORY.md)).
Against
binary stores (LMDB, SQLite, mmap-backed formats) a text parser is
predictably slower; See [benchmarks/RESULTS.md](https://github.com/theochemtheo/oxyz/blob/main/benchmarks/RESULTS.md) for comparisons.

[cextxyz]: https://github.com/libAtoms/extxyz
[ase-extxyz]: https://pypi.org/project/ase-extxyz/

## API

```python
oxyz.read_frames(path, *, threads=None)      -> list[Frame]
oxyz.iter_frames(path)                       -> Iterator[Frame]   # constant memory
oxyz.read_first_frame(path)                  -> Frame
oxyz.read_batch(path, indices, *, threads=None) -> Batch
oxyz.iter_batches(path, *, frames_per_batch=None, atoms_per_batch=None,
                  shuffle=False, seed=None, threads=None) -> Iterator[Batch]
oxyz.scan(path)                              -> FrameIndex
oxyz.infer_schema(path)                      -> Schema

oxyz.ase.read(path, index=None, *, format=None)  -> Atoms | list[Atoms]  # index=None: last frame
oxyz.ase.iread(path, index=":", *, format=None)  -> Iterator[Atoms]
oxyz.ase.to_atoms(frame)                     -> Atoms              # also Frame.to_ase()
```

`Frame`, `Batch`, `FrameIndex`, `Schema` and its parts (`ColumnSchema`,
`MetadataSchema`, the variant records, the `Kind` enum) are frozen
dataclasses; everything ships with type stubs.

The command line mirrors a subset:

```text
oxyz scan <path> [--no-schema] [--json]   # stats + inferred schema
```

### The fine print

Contracts worth knowing before relying on them:

- **Mixed-schema files read per-frame, but do not batch.** `read_frames` and
  `iter_frames` handle files whose frames disagree (the MACE
  isolated-atom-plus-bulk pattern) without complaint — each `Frame` stands
  alone. `Batch` assembly currently requires every gathered frame to share
  a schema; `infer_schema` tells you in advance whether a file qualifies.
  A missing-key policy (NaN-fill plus presence mask) is planned.
- **Duplicate metadata keys collapse.** `Frame.metadata` is a dict; if a
  comment line repeats a key, the last occurrence wins.
- **`Batch.batch` is computed per access** (`np.repeat` over the atom
  counts); hoist it out of a hot loop.
- **Errors carry frame context.** Malformed input raises
  `oxyz.ParseError` (a `ValueError` subclass) with the frame index and the
  offending line or value in the message, and the same location on the
  exception as attributes — `frame_index`, `line_number`, `column`, each
  `None` where the parser cannot pin it down — so you can find the bad
  frame without parsing the message. Out-of-range frame requests raise
  `IndexError`; I/O problems raise `OSError`. After a parse error,
  streaming iterators stop rather than guess at a resynchronisation point.
- **Partial reads only promise the prefix.** `read_batch` and indexed
  reads inspect the file no further than the last requested frame; damage
  past that point goes unreported. Whole-file validation is
  `infer_schema`'s job.

### Supported extxyz

The parser accepts and preserves; it does not interpret. Accepted: the
count line; a comment line of `key=value` pairs with bare or
double-quoted values, `[1, 2.0, 3]`-style or quoted whitespace-separated
arrays, `T`/`TRUE`/`True`/`true` booleans (a bare `1` stays an integer in
metadata, but is a boolean in an `L`-kind atom column, following the
spec); a `Properties` descriptor with `S`/`R`/`I`/`L` columns of any name
and width; any species strings. Metadata values are typed by shape, and
anything that fits no narrower type falls back to a string rather than
rejecting the file. Not supported: writing (reading only, for now),
compressed input, comment lines that are not key=value metadata, and
single-quoted values.

## How it is put together

Three layers, with the boundary chosen so that each is testable on its
own:

- **`crates/oxyz-core`** — the Rust core: parser, the columnar lossless
  `Frame` model, the structural scanner and byte-offset index, batch
  assembly, and the schema fold. No Python anywhere in the crate; it
  builds and tests standalone. Errors are structured (`thiserror`) and
  wrapped with the frame they occurred in.
- **`crates/oxyz-py`** — the PyO3 binding, a cdylib named `oxyz._rust`.
  Parsing runs with the interpreter detached (the GIL released), so
  threads parse in parallel; conversion to numpy happens once at the
  boundary, column buffers passing across as whole arrays rather than
  element-wise. Built as a single abi3 wheel per platform covering
  CPython ≥3.11.
- **`src/oxyz`** — thin typed Python: frozen dataclasses over the
  binding's dicts, batch planning (the pure-Python part of
  `iter_batches`), and the index grammar. All ASE knowledge lives in
  `oxyz.ase`, which imports ASE lazily; the core never depends on it.

Testing follows the shape of the promises: Rust unit and corpus tests for
the parser; parity tests holding parallel reads byte-identical to serial,
including which error wins when several frames are bad; golden tests
holding `oxyz.ase.read` equal to `ase.io.read` frame-by-frame; and
malformed-file tests asserting the frame index in the error message, not
just that an error occurred.

## Roadmap

In rough order of intent, shaped by what removes the most reasons to fall
back to other tools:

- **Write support** — lossless `Frame` round-tripping, removing the most
  common reason to keep ASE in a read → filter → write workflow.
- **Field selection and a missing-key batching policy** — request only
  the columns and metadata you need; NaN-fill or error on absent keys, so
  mixed-schema training files batch directly.
- **Normalisation accessors** — `positions`, `cell`, `numbers`, `pbc`,
  `forces`, `energy` as conventional views over the untouched raw data,
  for training loops that want neither ASE nor the raw spelling.
- **Additional inputs and outputs** — compressed input (`.xyz.gz`,
  `.xz`), `torch.Tensor` output, and a public lazy dataset object
  (`len`, indexing, slicing over an open file).

## Licence

MIT or Apache-2.0, at your option.
