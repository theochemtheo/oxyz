# Changelog

All notable changes to oxyz are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is below 1.0 the public API is not yet settled: minor
releases may make breaking changes, patch releases will not. Such changes are
recorded here.

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

[0.3.0]: https://github.com/theochemtheo/oxyz/releases/tag/v0.3.0
[0.2.0]: https://github.com/theochemtheo/oxyz/releases/tag/v0.2.0
[0.1.0]: https://github.com/theochemtheo/oxyz/releases/tag/v0.1.0
