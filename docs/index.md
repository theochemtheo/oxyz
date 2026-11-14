# oxyz

oxyz reads and writes extxyz trajectories for atomistic simulation and
machine-learning workflows. The native Python representation keeps per-atom
columns and frame metadata; optional converters produce ASE, metatomic and
torch-sim objects.

## Using oxyz

The [repository README](https://github.com/theochemtheo/oxyz/blob/main/README.md)
covers installation, reading, batches, schemas, typed records, writing and the
command-line interface. The
[examples directory](https://github.com/theochemtheo/oxyz/tree/main/examples)
contains self-contained scripts with small local datasets.

Choose native frames when you need the file's own names and values, batches
for concatenated arrays, or records for declared typed fields. Converter
modules are optional and have their own installation extras. NumPy and PyYAML
are the base runtime dependencies.

## Developing oxyz

- [Development setup](development/index.md): build from a checkout and choose
  the right verification loop.
- [Architecture](development/architecture.md): find the layer that owns a change
  and the tests that guard it.
- [Testing](development/testing.md): source examples, wheel tiers, types and
  Rust/Python checks.
- [Performance](development/performance.md): benchmark methodology and the limits
  of published comparisons.
- [Writing documentation](development/documentation.md): executable examples,
  deterministic output and the local preview.
- [Accepted decisions](development/decisions/index.md): durable contracts and
  their trade-offs.

This local documentation preview is not a published documentation service.
