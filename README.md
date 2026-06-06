# `atomflow`

**Schema-aware streaming for atomistic machine-learning datasets.**

`atomflow` is an experimental Python/Rust project for reading, validating, and streaming large atomistic datasets into machine-learning workflows.

The initial focus is on the [`extxyz`](https://github.com/libAtoms/extxyz) format, with an emphasis on fast ingestion, low memory use, schema inference, useful diagnostics, and optional interoperability with `ase.Atoms`.

> Project status: planning / early prototype.

---

## Motivation

Atomistic machine-learning datasets are often stored in flexible text-based formats such as `extxyz`. These files are convenient, human-readable, and widely supported across the materials modelling ecosystem.

However, flexibility can make large datasets difficult to use reliably in production-style ML pipelines.

Common issues include:

* inconsistent fields across frames;
* missing properties such as `forces`, `energy`, `stress`, `Lattice`, or `pbc`;
* unexpected dtype or shape changes;
* malformed metadata in comment lines;
* expensive full-file reads when only a few fields are needed;
* high memory use when working with large trajectory files;
* overhead from constructing Python objects for every frame;
* difficulty validating a dataset before using it for training.

`atomflow` aims to make these problems easier to detect and manage.

The project treats atomistic dataset ingestion as a data-engineering problem, not only a file-parsing problem.

---

## Project vision

The long-term goal is to provide a fast, reliable ingestion layer for atomistic ML data.

A typical workflow might look like:

```python
import atomflow

schema = atomflow.infer_schema("train.extxyz")
schema.report()

for batch in atomflow.iter_batches(
    "train.extxyz",
    schema=schema,
    fields=["numbers", "positions", "forces", "energy", "cell", "pbc"],
):
    train(batch)
```

The core idea is:

> Validate the structure of the data once, then stream only what is needed into downstream ML code.

`atomflow` should support ASE-style workflows where convenient, but should not require every high-throughput workflow to materialize data as `ase.Atoms` objects.

---

## Initial scope

The first version of `atomflow` will focus on `extxyz` ingestion.

Initial goals:

* read standard-compliant `extxyz` files;
* provide a Rust-backed parser with Python bindings;
* support streaming reads without loading the full file into memory;
* infer the structure of a dataset as it is read;
* detect missing, inconsistent, or unexpected fields;
* support selective reading of requested fields;
* expose data in Python-friendly array-oriented forms;
* provide optional conversion to `ase.Atoms`;
* benchmark against existing `extxyz` readers;
* keep the public API small and understandable;
* provide clear diagnostics when files are malformed or inconsistent.

---

## What `atomflow` should be good at

### Schema inference

`atomflow` should be able to inspect a dataset and answer questions such as:

* What per-atom properties are present?
* What per-frame metadata is present?
* Are fields consistent across frames?
* Which fields are required?
* Which fields are optional?
* Which frames are malformed?
* Which fields have inconsistent dtype or shape?
* Can this file be safely used for training?

Example:

```python
schema = atomflow.infer_schema("dataset.extxyz")
schema.report()
schema.to_json("schema.json")
```

---

### Dataset validation

Before training a model, it should be possible to validate the dataset against an inferred or user-provided schema.

Example:

```python
schema = atomflow.Schema.from_json("schema.json")

report = atomflow.validate(
    "dataset.extxyz",
    schema=schema,
    strictness="warn",
)

report.print_summary()
```

Validation should provide useful, actionable diagnostics rather than opaque parser errors.

For example:

```text
Validated 120000 frames.

Warnings:
- frame 8421: missing metadata key "energy"
- frame 10592: property "forces" has shape [2], expected [3]
- frame 88110: unexpected metadata key "config_type"
```

---

### Streaming reads

Large datasets should not require full-file reads.

The default high-throughput path should support iterating through frames or batches while keeping memory use low.

Example:

```python
for batch in atomflow.iter_batches(
    "dataset.extxyz",
    fields=["numbers", "positions", "forces", "energy"],
    batch_size=512,
):
    ...
```

---

### Selective reads

Many ML workflows only need a subset of the available data.

`atomflow` should make it possible to request only the fields needed for a particular task.

Example:

```python
energies = atomflow.read_arrays(
    "dataset.extxyz",
    fields=["energy"],
)
```

Where possible, the reader should avoid unnecessary parsing, allocation, and conversion work.

---

### ASE interoperability

ASE compatibility is important for inspection, debugging, and integration with existing workflows.

Example:

```python
atoms = atomflow.read_ase("dataset.extxyz", index=0)
```

```python
for atoms in atomflow.iread_ase("dataset.extxyz"):
    ...
```

However, ASE objects should not necessarily be the internal or default representation for high-throughput ingestion.

The guiding principle is:

> ASE-compatible when useful, array-native when performance matters.

---

## Possible strictness modes

`atomflow` should provide explicit control over how strictly files are interpreted.

Possible modes:

* `strict`: fail on malformed input, missing required fields, or schema drift;
* `warn`: continue where safe, while collecting diagnostics;
* `coerce`: allow controlled dtype promotion or missing-value handling.

The exact behavior of these modes should be decided before implementation.

Important questions include:

* Should `warn` be the default for exploration?
* Should `strict` be the default for training?
* What coercions are safe?
* Should missing values be represented, skipped, or treated as errors?
* Should unexpected fields be ignored, preserved, or reported?

---

## Benchmarking goals

`atomflow` should be benchmarked against existing readers on realistic datasets.

Benchmarks should measure more than raw parser speed.

Useful metrics include:

* wall-clock read time;
* peak memory usage;
* throughput in frames per second;
* throughput in atoms per second;
* cost of schema inference;
* cost of schema validation;
* cost of selective reads;
* cost of ASE object construction;
* performance on many small frames;
* performance on large individual frames;
* performance on files with extensive metadata.

Initial comparisons should include:

* ASE's default `extxyz` reader;
* `cextxyz` / `ase-extxyz`;
* `atomflow` array-oriented reads;
* `atomflow` ASE-compatible reads.

The aim is not only to be faster in every case, but to clearly identify the workloads where `atomflow` provides the most value.

---

## Testing philosophy

Correctness matters more than benchmark wins.

The test suite should give confidence that `atomflow` handles both well-formed and malformed data predictably.

Important test categories include:

* parser unit tests;
* schema inference tests;
* validation tests;
* golden-file tests;
* compatibility tests against existing readers;
* malformed-file tests;
* missing-field tests;
* schema-drift tests;
* dtype and shape consistency tests;
* large-file streaming tests;
* property-based tests;
* fuzzing for parser robustness.

The aim is high confidence, not a symbolic coverage percentage.

---

## Non-goals

`atomflow` is not intended to replace ASE.

ASE is a mature, general-purpose atomistic simulation environment. `atomflow` should complement it by focusing on fast, validated, ML-oriented dataset ingestion.

The project does not initially aim to:

* implement all ASE I/O functionality;
* support every historical `xyz` variant;
* become a simulation environment;
* silently repair malformed datasets;
* hide schema inconsistencies from the user;
* provide a large framework before the core reader is reliable.

---

## Design decisions to make before starting

Before implementation begins, several design choices should be settled or at least made explicit.

### 1. What is the minimum useful first release?

Possible first-release targets:

* a fast `extxyz` reader only;
* a schema inference tool;
* a validator and diagnostic CLI;
* a Python library for batched array reads;
* an ASE-compatible reader;
* some combination of the above.

A good first release should solve a real workflow problem without requiring the whole long-term vision to be complete.

---

### 2. What subset of `extxyz` should be supported first?

The full format is flexible. Supporting everything immediately may slow down development.

A practical first subset might include:

* atom count line;
* comment-line metadata;
* `Properties`;
* `Lattice`;
* `pbc`;
* species or atomic numbers;
* positions;
* forces;
* energy;
* stress.

Open questions:

* Which fields are required for the first work use case?
* How much non-standard syntax should be accepted?
* Should unsupported-but-valid files fail, warn, or fall back to another reader?
* Should the parser aim for exact compatibility with existing readers from the beginning?

---

### 3. What is the core data model?

The project needs a clear internal representation for parsed frames and batches.

Questions to decide:

* Is the primary representation frame-oriented or column-oriented?
* Are variable-size structures represented as ragged arrays, lists of arrays, offsets, or another structure?
* How are per-frame fields represented alongside per-atom fields?
* How are missing values represented?
* Are strings preserved as strings, encoded categorically, or converted when possible?
* How much does the internal model resemble `ase.Atoms`?
* How much does it resemble ML batch formats?

This decision will strongly affect performance, ergonomics, and future ML integration.

---

### 4. What should schema inference produce?

The schema should be useful enough to drive validation and faster reads, but not so complex that it becomes a separate project.

Questions to decide:

* What information belongs in a schema?
* Should schemas describe one file or a family of files?
* Should schema inference scan the whole file or support sampling?
* Should schemas include statistics, such as field frequency or atom-count ranges?
* Should schemas be serializable to JSON?
* Should schemas be stable enough to become part of the public API?
* How should schema versions be handled?

---

### 5. How strict should the parser be?

Real datasets often contain quirks. A useful tool needs to balance correctness and practicality.

Questions to decide:

* What counts as malformed input?
* What counts as schema drift?
* Which errors are fatal?
* Which errors can be warnings?
* Should the parser preserve unknown fields?
* Should dtype promotion be allowed?
* Should missing fields be allowed?
* Should behavior differ between exploration and training modes?

---

### 6. What should the Python API feel like?

The Python API should be small, predictable, and easy to use from notebooks and pipelines.

Questions to decide:

* Should the main entry point be `read`, `scan`, `infer_schema`, `iter_frames`, or `iter_batches`?
* Should functions return plain dictionaries, dataclasses, custom classes, NumPy arrays, or something else?
* Should ASE support live under a separate namespace?
* Should the API mirror ASE conventions or deliberately differ?
* How should errors and warnings be exposed?
* Should there be a CLI as well as a Python API?

---

### 7. What role should Rust play?

Rust should be used where it provides clear value: parsing, validation, indexing, memory safety, and parallelism.

Questions to decide:

* How much logic belongs in Rust versus Python?
* Should schema inference happen in Rust, Python, or both?
* Should the Rust core be usable independently of Python?
* How should Python bindings expose arrays efficiently?
* How much unsafe code, if any, is acceptable?
* Which dependencies are justified?

---

### 8. Should random access be part of the first version?

Frame indexing could be highly valuable for large datasets, but it may not be necessary for the first prototype.

Questions to decide:

* Should `atomflow` build byte-offset indices?
* Should indices be stored on disk?
* Should indexed reads be opt-in?
* Should random access come before or after streaming reads?
* Is parallel parsing worth implementing before the core parser is mature?

---

### 9. What is the initial ML integration target?

The long-term aim is ML-ready ingestion, but the first target should be chosen carefully.

Possible targets:

* NumPy arrays;
* PyTorch tensors;
* PyTorch Geometric-style graph data;
* TorchSim-compatible structures;
* JAX arrays;
* Arrow;
* Zarr.

A sensible first target may be NumPy, since it is widely supported and avoids coupling the project too early to one ML framework.

---

### 10. What is the compatibility policy?

The project should define what compatibility means.

Questions to decide:

* Is compatibility measured against the formal `extxyz` specification?
* Against ASE behavior?
* Against `cextxyz` behavior?
* Against real-world datasets?
* What happens when these disagree?
* Should compatibility tests include known edge cases from existing tools?

---

## Suggested early milestones

### Milestone 0: Real-file spike

Parse the smallest useful subset of `extxyz` needed for a real work dataset.

The aim is to answer:

* Can Rust parse the relevant files correctly?
* Can Python receive useful arrays?
* Is the performance promising?
* What parts of the format are most annoying in practice?

---

### Milestone 1: Schema inference prototype

Infer and report the structure of a dataset.

The aim is to answer:

* Which fields exist?
* Are they consistent?
* What breaks on real data?
* Is the schema representation useful?

---

### Milestone 2: Streaming reader

Implement low-memory iteration over frames or batches.

The aim is to answer:

* Can large files be processed without high memory usage?
* What batch representation is most ergonomic?
* Which selective reads matter most?

---

### Milestone 3: Validation and diagnostics

Turn schema inference into actionable validation.

The aim is to answer:

* Can users understand what is wrong with a dataset?
* Are errors precise enough to debug bad files?
* Which strictness modes are actually useful?

---

### Milestone 4: ASE compatibility

Add conversion to `ase.Atoms` for interoperability.

The aim is to answer:

* Does `atomflow` agree with ASE on representative files?
* How much performance is lost when constructing ASE objects?
* Which ASE conventions should be preserved?

---

### Milestone 5: Benchmarks and public release

Benchmark realistic workloads and prepare the first public release.

The aim is to answer:

* Where is `atomflow` faster?
* Where is it more memory efficient?
* Where is it more informative?
* What should be advertised as the core value proposition?

---

## Project philosophy

`atomflow` should make atomistic datasets easier to trust.

The ideal user experience is:

```python
schema = atomflow.infer_schema("dataset.extxyz")
schema.report()

for batch in atomflow.iter_batches("dataset.extxyz", schema=schema):
    train(batch)
```

The easy path should be fast.
The strict path should be safe.
The broken path should be obvious.

`atomflow` should grow from a practical solution to a real ETL problem into a reusable open-source tool for materials machine learning.
