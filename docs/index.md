# oxyz

oxyz reads and writes extxyz trajectories for atomistic simulation and
machine-learning workflows. The native Python representation keeps per-atom
columns and frame metadata; optional converters produce ASE, metatomic and
torch-sim objects.

## Using oxyz

Start with [getting started](getting-started.md): create a local trajectory,
inspect its actual values and verify a write/read round trip. Then choose a
guide for the task:

- [Reading and selecting](guides/reading-and-selecting.md): return shapes,
  slices, repeated indices and iteration.
- [Batching and streaming](guides/batching-and-streaming.md): array layout,
  provenance, batch budgets and schema compatibility.
- [Writing](guides/writing.md): incremental output, codecs, append and stdout.
- [Schemas and projection](guides/schemas-and-projection.md): infer a contract,
  validate input, fill fields and freeze patterns.
- [Typed records](guides/typed-records.md): declare application fields and read
  typed records or dense record batches.
- [Mixed datasets](guides/mixed-datasets.md): choose projection or record unions,
  then select by member and distinguish missing values from real zeros.
- [Compressed and remote sources](guides/compressed-and-remote.md): codecs,
  archive members, object storage and HuggingFace.
- [Converters](guides/converters.md): ASE, metatomic and torch-sim workflows
  with explicit scientific conventions.
- [Command line](guides/command-line.md): inspect, check and freeze schemas.

For the contracts behind those workflows, see the [data model](concepts/data-model.md),
[schemas](concepts/schemas.md) and [storage and streaming](concepts/storage-and-streaming.md). The
[API reference](reference/api.md) routes public names to their source docstrings;
[format](reference/extxyz.md), [compatibility](reference/compatibility.md) and
[errors](reference/errors.md) describe syntax and failure boundaries.

The [repository README](https://github.com/theochemtheo/oxyz/blob/main/README.md)
is a short installation and usage entry point. Runnable
[example scripts](https://github.com/theochemtheo/oxyz/tree/main/examples)
use small bundled datasets.

Choose native frames when you need the file's own names and values, batches
for concatenated arrays, or records for declared typed fields. Converter
modules are optional and have their own installation extras. NumPy and PyYAML
are the base runtime dependencies.

## Performance

The [performance overview](performance.md) compares full reads of a real dataset
and dataset-size scaling, with separate NumPy and ASE outputs. It includes the
measurement environment and links to complete results and reproduction commands.

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
