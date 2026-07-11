# Changelog

All notable changes to oxyz are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is below 1.0 the public API is not yet settled: minor
releases may make breaking changes, patch releases will not. Such changes are
recorded here.

## [Unreleased]

### Added

- Schema **projection**: set `mode="project"` on a `SchemaSpec` (or per call)
  to reshape every frame to a declared fixed schema — undeclared fields dropped,
  absent optionals filled — making a mixed-schema file batchable. Available on
  the frame readers (`read`, `iread`), the batch readers, and the
  `oxyz.ase`/`oxyz.metatomic`/`oxyz.torch_sim` output targets, all via
  `schema=`/`mode=`/`conformance=`.
  REAL columns fill `NaN` by default; other kinds take an explicit per-field
  `fill`. `SchemaSpec.freeze(path)` expands pattern rules into a project-ready
  schema, exposed on the CLI as `oxyz freeze` and `scan --emit-schema
  --project`. The batch readers (`read_batch`, `iread_batch`) now also accept
  `schema=`/`conformance=`. Validate-mode behaviour is unchanged.
- `oxyz.metatomic` and `oxyz.torch_sim` `read`/`iread` and their
  `SystemSource`/`SimStateSource` gain `storage_options=`, so reading from an
  S3-compatible URL works from every output target, as it already did for
  `oxyz.ase` and the native readers.
- `oxyz.ase.read` gains `threads=` to tune the parallel parse of an eager read
  (as the native and `oxyz.metatomic` readers already had); `None` uses all
  cores, `1` is serial. `oxyz.ase.iread` streams and takes no `threads`.
- `oxyz.OxyzError`, a common base (itself a `ValueError`) for every error the
  package raises: `ParseError`, `SchemaError`, and the converters' `ToAseError`,
  `FromAtomsError`, `ToSystemError`, `ToSimStateError` all subclass it, so
  `except oxyz.OxyzError` catches oxyz's errors as a group while
  `except ValueError` keeps working.
- Export the `Mode` (`"validate"`/`"project"`) and `Writable`
  (`Frame | ase.Atoms`, the `write` input) type aliases from `oxyz`, for parity
  with the already-exported `Conformance`, `Compression`, and `MemoryScaling`.
- `SchemaSpec` gains `from_json` and `to_file`, so every serialisation format
  now has a matching `from_`/`to_` pair (`dict`, `json`, `yaml`, `file`).
- `oxyz check --conformance` accepts `warn`, matching the Python API's
  conformance levels; like `strict` it reports extra columns/keys.

### Changed

- The native frame readers are unified under `read` and `iread`. Both take an
  `index` selection — an int (one `Frame`), a slice or slice-string like
  `"1:10:2"`, or a sequence of non-negative ints (a list, in order); the default
  `":"` reads every frame. `read_frames` becomes `read`, `iter_frames` becomes
  `iread`, and the selection that previously needed separate helpers is now a
  parameter. Selecting a single frame with a schema (`read(path, 0, schema=...)`)
  validates the whole file before indexing, consistent with `oxyz.ase.read`.
- The streaming batch reader `iter_batches` becomes `iread_batch`, so
  `read`/`iread` and `read_batch`/`iread_batch` share one rule: `read`
  materialises, `iread` streams.
- `MetadataRule`'s identifier field is renamed from `name` to `key`, matching
  `MetadataSchema.key` and the metadata `key=` used elsewhere; `ColumnRule`
  keeps `name`. The YAML/JSON schema format is unchanged (the identifier is the
  mapping key either way).
- `ParseError.line_number` is renamed to `ParseError.line`, matching
  `Violation.line`; both now pair `line` with `column`.
- `Frame.to_ase()` is renamed to `Frame.to_atoms()`, matching the
  `oxyz.ase.to_atoms` function it delegates to.
- `SchemaSpec.from_yaml_text` is renamed to `SchemaSpec.from_yaml`, pairing with
  `to_yaml` (see also the new `from_json`/`to_file` under Added).
- `read_batch`'s `indices=` parameter becomes `index=`, taking `read`'s full
  selection grammar (`":"`, an int, a slice or slice string, or a sequence);
  the default `":"` reads the whole file, as `indices=None` did.

### Removed

- `read_first` — use `read(path, 0)`. The never-public `read_frames_sliced`
  helper is absorbed into `read`'s slice handling.

### Internal

- Hardened the CI and release workflows: every GitHub Action is pinned to a
  commit SHA, jobs run with minimal per-job permissions, checkouts set
  `persist-credentials: false`, a `zizmor` gate audits the workflows, and the
  PyPI publish environment now requires approval.
- Widened the `ruff` lint selection.
- Refreshed the locked Python dependencies (numpy 2.5, ASE 3.29, and the dev
  and CI toolchain); a `tolist` call was reordered to satisfy numpy 2.5's
  stricter array stubs, with no change in behaviour.
- Refreshed the locked Rust dependencies; the `ndarray` pin moves to 0.17 to
  stay unified with the version the `numpy` crate builds against.

## [0.5.0] - 2026-07-03

### Added

- Schema-aware reading: pass `schema=` (a `SchemaSpec` or a path to a
  `.json`/`.yaml`/`.toml` file) and `conformance=` (`"strict"`, `"required"`,
  or `"warn"`) to `read_frames`, `read_first`, `read_frames_sliced`, and
  `iter_frames`. Validates each frame's columns, metadata, and structural facts,
  with frame-indexed `SchemaError`s and silenceable `SchemaWarning`s. Names may
  be literals, globs (`descriptor_*`), or regexes (`re:...`), with `count`/`min`/
  `max` on patterns.
- `oxyz check FILE --schema S`: report every schema violation in a file (with
  frame index and source line), exit non-zero when any is found; `--json` for CI.
- `oxyz scan` now prints a copy-pasteable schema, and `oxyz scan --emit-schema
  PATH` writes it to a `.yaml`/`.json` file. `Schema.to_spec()` exposes the same
  in Python.

### Changed

- `oxyz scan`'s text summary now shows the inferred schema as pasteable schema
  syntax rather than a free-form report.

### Dependencies

- Added PyYAML (`pyyaml>=6`) as a runtime dependency, for reading and writing
  YAML schemas.

## [0.4.0] - 2026-06-30

### Added

- Reading from compressed files. Every reader (`read_frames`, `iter_frames`,
  `read_batch`, `iter_batches`, `read_first`, `scan`, `infer_schema`, the
  `oxyz.ase` / `oxyz.metatomic` / `oxyz.torch_sim` converters, and the `oxyz`
  CLI) now accepts `.gz`, `.tar.gz`, `.zip`, `.zst` and `.tar` paths and decodes
  them on the fly — `read_frames("run.xyz.gz")` just works, with no separate
  decompression step. Decoding streams, so reads stay parallel without
  decompressing to a temporary file or holding the whole file in memory; a bare
  `.gz` with several concatenated members is fully read. `compression=` forces a
  codec (`"infer"` default, or `"none"`/`"gzip"`/`"zstd"`/`"zip"`) and `member=`
  selects one entry from a multi-member archive (which otherwise errors, listing
  its members). A compressed source cannot be seeked, so random-access
  strategies — `iter_batches` with `shuffle`, `atoms_per_batch`, or
  `memory_scales_with`, and reverse/negative ASE indices — either fall back to a
  full in-memory read (the ASE index path) or raise a clear error pointing at
  the limitation. Decoders are pure Rust (`flate2`, `ruzstd`, `zip`, `tar`), so
  the wheel gains no system dependencies.
- Writing extxyz. `oxyz.write(path, obj, ...)` takes a `Frame`, an `ase.Atoms`,
  or an iterable mixing them and writes (ext)xyz, removing the most common reason
  to keep ASE in a read → filter → write workflow. Reals are written
  shortest-round-trippable, so `read` then `write` reproduces every `f64` bit for
  bit; columns come out `species`, `pos`, then the rest, and the comment line
  `Lattice`, `pbc`, `Properties`, then the rest (a frame lacking `species` or
  `pos` is rejected). The codec follows the path extension — plain, `.gz`,
  `.zip`, `.tar`, `.tar.gz` — or is forced with `compression=`; `level=` tunes
  the deflate codecs, `"-"` writes to stdout, and `append=True` concatenates
  onto an existing plain or gzip file (archives and stdout reject it).
  `oxyz.Writer` is the incremental, constant-memory form (a context manager),
  and `oxyz.ase.from_atoms` is the inverse of `to_atoms`. Writing `.zst` is not
  yet supported. Serialisation runs across cores by default — `threads=` tunes
  it (`None` for every core, `1` for serial), with output bytes identical at any
  count; only serialisation parallelises, the output stream stays serial.
  `oxyz.Writer(path, batch=n)` keeps the incremental form but serialises `n`
  frames at a time in parallel, trading one batch of memory for throughput.
- Read extxyz directly from S3-compatible object stores: `read_frames`,
  `iter_frames`, `scan`, `infer_schema`, the batch readers, and
  `oxyz.ase.read`/`iread` accept `s3://`/`gs://`/`az://` URLs with the new
  `oxyz[s3]` extra. Endpoint and credentials via `storage_options=` or `AWS_*`
  env vars; all codecs and archive `member=` selection supported. `oxyz scan`
  gains `--storage-option`.

## [0.3.0] - 2026-06-27

### Removed

- Python 3.11 support. oxyz now requires Python 3.12+, following
  [SPEC 0](https://scientific-python.org/specs/spec-0000/) (a Python version is
  dropped three years after release; 3.11's window closed Q4 2025). Wheels are
  now abi3 for CPython 3.12+. Users on 3.11 can pin `oxyz<0.3`.

### Added

- `oxyz.metatomic` reads extxyz into `metatomic.torch.System`s without an ASE
  round-trip. `read`/`iread` mirror `oxyz.ase` (the same index grammar, plus
  `dtype`/`device`/`*_requires_grad` matching `systems_to_torch`); a
  `SystemSource` handle parses a file once and serves `systems()` alongside
  array-native `per_config` / `per_atom` tensor extraction for targets. New
  optional extra `oxyz[metatomic]` (torch >=2, metatomic-torch). Parity tests
  hold the result equal to `systems_to_torch(ase.io.read(...))`.
- `oxyz.torch_sim` reads extxyz into `torch_sim.SimState`, reproducing
  `torch_sim.io.atoms_to_state(ase.io.read(...))` without the ASE round-trip.
  Because `SimState` is natively batched, `read` returns a single batched state
  (the whole selection) and `iread` streams the file as batched states (with
  `oxyz.iter_batches`'s binning knobs); a `SimStateSource` serves the state plus
  array-native `per_config` / `per_atom` extraction. Cells use torch_sim's
  column convention, all systems share one pbc (frames that disagree raise), and
  masses come from a `masses` column or an ASE-parity atomic-weight table. New
  optional extra `oxyz[torch-sim]` (torch >=2, torch-sim-atomistic); parity
  tests hold the result equal to `atoms_to_state(ase.io.read(...))`.
- `oxyz.read_batch(path, indices=None)` reads the whole file into one `Batch` in
  a single pass; an empty file yields the empty batch.
- `oxyz.iter_batches(memory_scales_with=..., max_scaler=...)` packs frames into
  balanced bins (best-fit-decreasing) for roughly equal per-batch memory,
  weighting each frame by `"n_atoms"` or by `"n_atoms_x_density"`
  (`n_atoms**2 / volume`, a proxy for the neighbour-graph size that drives MLIP
  memory). A frame over the budget gets its own bin; provenance is kept in
  `frame_indices`.
- `oxyz.scan(path, with_volume=True)` additionally records each frame's cell
  volume `|det(Lattice)|` in `FrameIndex.volumes` (`NaN` where a frame has no
  `Lattice`), reading one extra line per frame; it backs the density weight.

## [0.2.0] - 2026-06-25

A performance release: faster reads across the board, no API changes. Numbers
are means on an Apple M3 Pro under CPython 3.13; full tables are in
[benchmarks/RESULTS.md](benchmarks/RESULTS.md).

### Performance

- ASE conversion (`oxyz.ase.read`) is 18–57% faster: the eager list read now
  parses on every core, and species map to atomic numbers through a cached
  lookup so ASE skips its own per-atom symbol parsing. Reading the 4 × 100 000
  atom file to `ase.Atoms` drops 167 → 71 ms — now faster than the libAtoms C
  parser's ASE plugin, which 0.1.0 was slower than on that workload.
- Files with a few very large frames now scale past four threads: a frame's
  atom rows are split across workers (4 × 100 000 atom read, all cores,
  36 → 27 ms).
- Batched reads (`read_batch`, `iter_batches`) reuse one worker pool across
  batches instead of rebuilding it per batch, restoring thread scaling
  (2 000-frame batched read, all cores, 15.8 → 12.4 ms).
- The atom-row parser tokenises raw bytes, validating UTF-8 only for string
  cells, lifting single-thread parse throughput ~15%.

### Changed

- A non-UTF-8 byte in a numeric atom cell now raises `ParseError` (an invalid
  value) rather than an `OSError`; a non-UTF-8 byte in a string cell still
  raises the I/O error, as before. Atom rows are now tokenised as bytes, so
  only string cells are validated as UTF-8.

## [0.1.0] - 2026-06-16

First public release: a Rust extxyz parser behind a small, typed Python API,
for reading atomistic-simulation datasets into numpy or ASE.

### Added

- `read_frames` (parallel) and `iter_frames` (streaming, constant memory) for
  whole-file reads, and `read_first` for the first frame alone. Column names
  and metadata are preserved as written, without aliasing or normalisation.
- `read_batch` and `iter_batches` for atom-major concatenated batches, the
  latter packing by a fixed frame count or an atom budget, with an optional
  seeded shuffle. Batch composition does not depend on the thread count.
- `scan` for a file's structure — byte offsets and atom counts, without
  parsing frame contents — and `infer_schema` for a one-pass `Schema` of the
  columns and metadata keys, their types and shapes, and how consistently they
  appear across frames.
- `Frame` and `Batch` as frozen dataclasses; `Batch` carries the CSR layout
  PyTorch Geometric expects (`offsets`/`ptr`, `batch`, `frame_indices`).
- `oxyz.ase`: `read` and `iread` as drop-ins for `ase.io.read`/`iread` over
  extxyz, with ASE's full index grammar, plus `to_atoms` and `Frame.to_ase`.
  Requires the optional `ase` extra.
- `oxyz.ParseError`, a `ValueError` carrying the `frame_index`, `line_number`,
  and `column` of the offending input.
- An `oxyz` command-line tool; `oxyz scan` summarises a file and its inferred
  schema (`--no-schema`, `--json`).
- Type stubs and `py.typed`; numpy is the only required runtime dependency.
- abi3 wheels for CPython 3.11 and newer on Linux (x86_64, aarch64), macOS
  (arm64, x86_64), and Windows (x64).

[0.5.0]: https://github.com/theochemtheo/oxyz/releases/tag/v0.5.0
[0.4.0]: https://github.com/theochemtheo/oxyz/releases/tag/v0.4.0
[0.3.0]: https://github.com/theochemtheo/oxyz/releases/tag/v0.3.0
[0.2.0]: https://github.com/theochemtheo/oxyz/releases/tag/v0.2.0
[0.1.0]: https://github.com/theochemtheo/oxyz/releases/tag/v0.1.0
