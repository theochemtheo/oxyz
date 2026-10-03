# Changelog

All notable changes to oxyz are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Three skip guards did not guard. `tests/test_errors.py` tested
  `find_spec("metatomic.torch")`, which imports the parent package and so raises
  `ModuleNotFoundError` when `metatomic` is absent instead of returning `None` —
  aborting collection of the module rather than skipping one test.
  `tests/test_compressed.py` used `importorskip("oxyz.ase")`, which stopped
  skipping when pytest 9.1 narrowed the default `exc_type` to
  `ModuleNotFoundError`: `oxyz.ase` imports fine and raises a plain
  `ImportError` carrying its install hint, so pytest now propagates it as a
  failure. Both are within the project's declared `pytest>=9.0.3` range. The ty
  canary asserted `ty` was on `PATH` rather than skipping without it. None of
  these were visible in CI, because every job installed every dependency.

### Added

- `Batch` is a sequence of its frames. `len(batch)` is `n_frames` and
  iterating yields each `Frame`. Indexing follows numpy: `batch[i]` is a
  `Frame` whose numeric arrays are views onto the batch, `batch[a:b]` is a
  `Batch` of views with `offsets` rebased, and a step slice, a list or array
  of positions, or a boolean mask over frames (`batch[batch["energy"] < 0]`)
  gathers a new `Batch`, a copy, keeping `frame_indices` as file provenance.
  String columns are lists and so are copied; scalar metadata comes out as the
  Python value `oxyz.read` gives, so `batch[i]` equals the frame read from the
  file, dtype for dtype. Being iterable, a batch can be passed straight to
  `oxyz.write` and `Writer.write`.
- Lookup by name across columns and metadata: `frame["pos"]` and
  `frame["energy"]` both work, as do `name in frame`, `frame.get(name,
  default)`, `frame.keys()`, and so `dict(frame)`. The value returned is the
  stored one, not a copy. `Batch`, `Schema` (returning the `ColumnSchema` or
  `MetadataSchema`) and `SchemaSpec` (returning the rule declared under that
  exact name) behave the same. A name that is both a column and a metadata key
  raises the new `oxyz.AmbiguousNameError` (an `OxyzError`, deliberately not a
  `KeyError`) rather than picking one. None of these is a `Mapping` — no
  `len()`, no iteration — so numpy keeps a list of frames as frames:
  `rng.choice(frames, k)` and `np.array(frames, dtype=object)` still work.
- `SchemaSpec.rule_for(name, axis=None)`: the rule validation would apply to a
  field of that name — its literal rule, else the first matching glob or regex
  in declaration order — or `None`.
- `Frame` derived accessors: `frame.positions`, `frame.numbers`,
  `frame.symbols`, `frame.cell`, and `frame.pbc` resolve the well-known fields
  from the untouched `columns`/`metadata` dicts on each access — `cell` is the
  3x3 ASE-convention cell from the flat Fortran-order `Lattice`, `pbc` defaults
  to `Lattice` presence. Nothing is cached and nothing stored is rewritten.
- `oxyz.FieldError` (an `OxyzError`, so still a `ValueError`): a well-known
  field is absent, or present in a shape or type it cannot hold. Raised by the
  new accessors — including when the field is present but will not coerce, so
  a `Z:S:1` string column reports a `FieldError` rather than numpy's `invalid
  literal for int()` — and by `oxyz.write` when a column's row count disagrees
  with the frame's `n_atoms`: a hand-built short column used to panic inside
  the Rust encoder and a long one silently dropped atoms. A 0-D column counts
  as the one row the encoder writes from it, so the scalar-width-1 form stays
  legal for a single-atom frame.
- Tests for the optional targets' import guards: that a missing dependency
  raises `ImportError` naming both the module and the extra to install, and
  chains the original cause. The imports are blocked in-process, so these run —
  and are measured by coverage — whether or not the optional dependency is
  installed, and the guard branches are no longer dead lines in the report.
- CI runs the test suite on **deliberately incomplete dependency
  environments**: `base` (no optional dependencies), `ase`, and `s3`, pinned to
  3.12, the abi3 floor, plus `base` on the newest supported interpreter as a
  forward-compatibility check on the single `cp312-abi3` wheel. A tier canary
  keyed on `OXYZ_TEST_TIER` asserts the environment is the one CI declared, so a
  tier that installs too much, or a guard that skips for the wrong reason, fails
  instead of passing over nothing. The `python` and `coverage` jobs declare
  `full` and are asserted the same way.

### Changed

- Every public object has a one-line `repr` that prints no data, so its length
  depends on how many names an object carries rather than how many atoms or
  frames. Objects that hold data — `Frame`, `Batch`, `SimStateSource`,
  `SystemSource` — describe values by numpy dtype and shape
  (`'pos': float64[96, 3]`); `Schema` and `SchemaSpec` describe a file in the
  extxyz letters of `Properties=` (`'pos': R:3`, `'stress': R[3, 3]`), joining
  drifting variants with `|` and marking an entry some frames lack, or an
  optional rule, with `?`. Multi-frame objects show their atom-count range,
  `n_atoms=lo..hi`. `FrameIndex` no longer prints its offset arrays; `Writer`
  shows its path, non-default options, `frames_written`, and whether it is
  closed; the schema rules and `Violation` omit fields left at their defaults;
  `Kind` is `Kind.REAL`. `str(schema)` is still the multi-line report.
- A schema violation on a 2-D metadata rule now states the expected shape in
  full (`R[3, 3]`, formerly `R[3]`).

### Internal

- The well-known-field resolution the converters each carried is now one layer:
  `oxyz._convert` owns `positions`/`numbers`/`symbols`/`cell`/`pbc` over the
  plain dicts, and `oxyz.metatomic` consumes it instead of its own
  `_cell_and_pbc` body. `numbers` and `MissingSpeciesError` move out of
  `oxyz._torch`, which imports torch eagerly and so could not back
  `Frame.numbers`. `oxyz.ase` keeps its own routing loop (ASE-table parity),
  and `oxyz.torch_sim` keeps its batched, column-vector cell.
- New `oxyz._summary`: the shape-only repr and the NaN-equal, order-insensitive
  value comparison `Frame` and `Batch` share.
- `pytest`/`pytest-cov` and `boto3`/`moto` move out of the `dev` group into new
  `test` and `s3-test` groups, which `dev` includes. A plain `uv sync` resolves
  to exactly what it did before; the split exists so CI can build a partial
  environment. The optional dependencies themselves need no group — the tiers
  install the published `ase`/`s3` extras, the way a user would.
- The tier jobs install a single wheel built once by a new `build-wheel` job
  rather than building their own, so they need no Rust toolchain and exercise the
  same abi3 artifact a user installs.
- A `test-gate` fan-in job, mirroring `wheels-gate`, is now the one status check
  the branch ruleset needs from this workflow. With matrix contexts named
  directly, every reshape had to be mirrored by hand in the ruleset — and got it
  wrong in both directions, stranding required checks that would never report
  again and adding new ones unprotected.
- Coverage is unchanged and still measured only on the full dependency set; the
  tiers never pass `--cov`, since a partial run is not a coverage figure.
- uv pinned to 0.12.17, in `pyproject.toml` and every `setup-uv` step. The
  lockfile is unchanged under it.
- Every workflow action re-pinned to its latest release, `rust-toolchain` to the
  current tip of `stable`, and zizmor to 1.30.1. The two `download-artifact`
  pins, which had drifted to different majors, now agree.

## [1.1.0] - 2026-07-29

### Added

- Read straight from the HuggingFace Hub with `hf://` URLs, on every entry point
  that already took a remote URL — `read`, `iread`, `scan`, `infer_schema`, the
  batch readers, and the `oxyz.ase`/`oxyz.metatomic`/`oxyz.torch_sim` targets:
  `oxyz.read("hf://datasets/owner/repo/data/train.extxyz")`. The path grammar is
  `hf://[datasets|spaces|models/]owner/repo[@revision]/path`, defaulting to a
  model repo on the default branch. A URL copied from the browser is accepted as
  well: the `blob` page it names is rewritten to the raw-bytes endpoint, so it
  yields the file rather than markup. Only `huggingface.co` and `hf.co` URLs are
  claimed — `http(s)` in general is still not a supported scheme. Gated and
  private repos authenticate with `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`), or
  an explicit `storage_options` header, which takes precedence. Needs the
  existing `oxyz[s3]` extra; no new dependency.
- Read zstd-compressed tars (`.tar.zst`, `.tzst`), locally and from object
  storage. Datasets shipped this way — the ELEMENTA release among them — were
  previously unreadable: the extension matched the plain `.zst` rule, so the
  tar headers reached the parser and surfaced as an atom-count error.
- All three tar codecs are now selectable through `compression=` for reading
  (`"tar"`, `"tar.gz"`, `"tar.zst"`), and through `--compression` on `oxyz scan`,
  `check`, and `freeze`; writing still accepts only `tar` and `tar.gz`. A tar
  carries no magic bytes, so one under an unrecognised name previously could not
  be read at all.

### Internal

- Bumped the developer toolchain: uv 0.12.0, ruff 0.16, ty 0.0.64, rumdl 0.2.45,
  prek 0.4.11. uv is now pinned exactly, in both `pyproject.toml` and every
  `setup-uv` step, so CI cannot float off the version the lockfile was written
  by.
- The `ruff` selection is `select = ["ALL"]` with documented opt-outs, in place
  of a hand-picked list: each exclusion now carries its reason, and a new
  release's rules arrive evaluated rather than unnoticed. No change to the
  shipped behaviour.

## [1.0.0] - 2026-07-20

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
- Comment-line metadata now types **2-D arrays** (`key=[[1,2],[3,4]]`) as
  shaped arrays, surfaced in Python as 2-D numpy arrays.
- A committed **MAD-1.5** r²SCAN sample (`tests/data/mad_r2scan_sample.extxyz`)
  in the test corpus, guarding that real, chemically diverse data — 98 elements
  across molecules, clusters, bulk, surfaces and low-dimensional structures,
  drawn from a 102-element dataset — parses.
- An [`examples/`](examples/) directory of runnable, self-contained snippets —
  scanning and schema inference, reading to numpy, atom-budgeted batching,
  schema projection, the ASE drop-in, the PyTorch targets, and a write
  round-trip — each executed in CI against a committed sample.
- `AGENTS.md`: a tool-agnostic onboarding guide for coding agents — repo map,
  the `uv` build/test commands, the TDD loop, and the pre-PR gates.

### Performance

- Faster whole-file reads, most of all on many-small-frame corpora. Profiling
  found the parse allocation-bound — a heap allocation for every species symbol,
  metadata key, and short value — which the system allocator serialised, capping
  parallel scaling. Storing those short strings inline (they are almost always
  ≤24 bytes) and no longer allocating a string for metadata values that are only
  read cuts allocations by roughly a third. The internal parser and binding
  changed; the API and output are unchanged. On the reference corpus fixture the
  core read is ~28% faster serial and ~42% faster on 12 threads (thread scaling
  2.7×→3.3×); end-to-end `oxyz.read` is ~17–19% faster. See
  [benchmarks/RESULTS.md](benchmarks/RESULTS.md).

### Dependencies

- Added `compact_str` (pure Rust, no C toolchain) for the inline short-string
  storage above.

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
- `ParseError` now locates every parse error structurally. `line_number` is
  renamed to `line` (matching `Violation.line`), and `column` is the 1-based
  character column of the offending token within its line — previously it held
  the offending data column's *name*. `frame_index`, `line`, and `column` are
  populated wherever the parser can pin each down, so a malformed file can be
  located without parsing the message string; the messages themselves now name
  the expected input, not only the fault.
- Comment-line value parsing now conforms to the libAtoms/extxyz key-value
  grammar: brace-wrapped scalars and arrays (`{3}`, `{1 2 3}`), quoted arrays
  with embedded separators (`[ "a, b", "c]" ]`), lowercase `t`/`f` booleans,
  and Fortran `d`/`D` float exponents (`-12.0d0`) are accepted; malformed
  values — trailing/leading commas in arrays (`[1,2,]`), ragged 2-D arrays,
  unbalanced brackets — are now rejected with a located error rather than
  silently kept as strings. A bare (unquoted) value containing `=`, `"`, `,`,
  `[`, `]`, `{`, `}`, or `\` is likewise rejected; quoting the value still
  permits any of them.
- `Frame.to_ase()` is renamed to `Frame.to_atoms()`, matching the
  `oxyz.ase.to_atoms` function it delegates to.
- `SchemaSpec.from_yaml_text` is renamed to `SchemaSpec.from_yaml`, pairing with
  `to_yaml` (see also the new `from_json`/`to_file` under Added).
- `read_batch`'s `indices=` parameter becomes `index=`, taking `read`'s full
  selection grammar (`":"`, an int, a slice or slice string, or a sequence);
  the default `":"` reads the whole file, as `indices=None` did.
- Restructured the README into an onboarding arc — a quickstart front door
  (install → first read → schema check), the beyond-ASE differences promoted
  ahead of the compatibility detail, and the PyTorch targets grouped. Content
  is reorganised rather than removed, bar the out-of-date Roadmap section, which
  is dropped.
- Docstrings across the public surface completed to the numpy convention and
  enforced going forward: `ruff` (pydocstyle `D`) checks formatting on every
  docstring in `src/oxyz` and presence on the public-path modules; a new
  `tests/test_public_docstrings.py` checks presence on the rest of the
  re-exported `oxyz.__all__` surface and the `oxyz.ase`/`oxyz.metatomic`/
  `oxyz.torch_sim` entry points, which ruff cannot see (their definitions
  live in underscore-prefixed modules).

### Removed

- `read_first` — use `read(path, 0)`. The never-public `read_frames_sliced`
  helper is absorbed into `read`'s slice handling.

### Fixed

- A parse error on an atom value now reports the column of the exact
  offending cell, not the first cell of its column. A bad value in the 2nd or
  3rd cell of a multi-width column (e.g. `pos:R:3`) previously pointed at the
  column's first cell instead of the cell that actually failed to parse.

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
- Benchmarks now cover the full MAD-1.5 r²SCAN training set (303.5 MiB,
  180,184 frames) alongside the generated fixtures; see
  [benchmarks/RESULTS.md](benchmarks/RESULTS.md).

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

[1.1.0]: https://github.com/theochemtheo/oxyz/releases/tag/v1.1.0
[1.0.0]: https://github.com/theochemtheo/oxyz/releases/tag/v1.0.0
[0.5.0]: https://github.com/theochemtheo/oxyz/releases/tag/v0.5.0
[0.4.0]: https://github.com/theochemtheo/oxyz/releases/tag/v0.4.0
[0.3.0]: https://github.com/theochemtheo/oxyz/releases/tag/v0.3.0
[0.2.0]: https://github.com/theochemtheo/oxyz/releases/tag/v0.2.0
[0.1.0]: https://github.com/theochemtheo/oxyz/releases/tag/v0.1.0
