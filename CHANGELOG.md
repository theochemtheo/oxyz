# Changelog

All notable changes to oxyz are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is below 1.0 the public API is not yet settled: minor
releases may make breaking changes, patch releases will not. Such changes are
recorded here.

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

[0.1.0]: https://github.com/theochemtheo/oxyz/releases/tag/v0.1.0
