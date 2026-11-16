# oxyz

[![test](https://github.com/theochemtheo/oxyz/actions/workflows/test.yml/badge.svg)](https://github.com/theochemtheo/oxyz/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/oxyz)](https://pypi.org/project/oxyz/)
[![Documentation](https://img.shields.io/badge/docs-development-blue)](https://theochemtheo.github.io/oxyz/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![SPEC 0](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CB05C)](https://scientific-python.org/specs/spec-0000/)

Fast, schema-aware [extxyz](https://github.com/libAtoms/extxyz) reading and writing
for atomistic simulation and machine learning. A Rust parser behind a typed
Python API, with NumPy arrays as the native numeric output.

- **Array-native frames and batches:** stored file keys, atom-major concatenation,
  source frame indices and streaming batch budgets.
- **Observed and declared schemas:** inspect field/type drift, validate a contract,
  or project selected fields with explicit fills.
- **Typed records and unions:** named fields, dense batches, presence masks and
  concrete member types for heterogeneous datasets.
- **Optional scientific targets:** ASE, metatomic and torch-sim converters; local
  compressed files and remote object-storage/Hub reads.
- **Measured performance:** serial/parallel and reference-reader comparisons,
  with workloads and environment recorded in the [benchmark results](https://github.com/theochemtheo/oxyz/blob/main/benchmarks/RESULTS.md).

## Install

```sh
pip install oxyz
pip install "oxyz[ase]"
pip install "oxyz[metatomic]"
pip install "oxyz[torch-sim]"
pip install "oxyz[s3]"
```

The base runtime requires Python ≥3.12, NumPy ≥2 and PyYAML ≥6. Extras add
`ase.Atoms`, `metatomic.torch.System`, `torch_sim.SimState`, or remote reads
(including HuggingFace) respectively; optional libraries are imported when used.
Wheels cover Linux, macOS and Windows. The support window follows
[SPEC 0](https://scientific-python.org/specs/spec-0000/).

Installing also provides `oxyz scan`, `oxyz check` and `oxyz freeze`.
`uvx oxyz scan PATH` runs the released CLI without a persistent installation.
For checkout setup and uv-driven Rust rebuilds, see
[development](https://github.com/theochemtheo/oxyz/blob/main/docs/development/index.md).

## Read, inspect and write

This example creates its input in a temporary directory and verifies the values
after writing. It needs no downloaded dataset or optional library.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import oxyz

with TemporaryDirectory() as directory:
    source = Path(directory) / "sample.extxyz"
    source.write_text(
        "1\nProperties=species:S:1:pos:R:3 energy=-1.5\nH 0.25 0 0\n"
        "1\nProperties=species:S:1:pos:R:3 energy=-2.5\nHe 1.25 0 0\n"
    )
    frames = oxyz.read(source)
    assert [frame.n_atoms for frame in frames] == [1, 1]
    assert frames[0].columns["species"] == ["H"]
    assert frames[0].positions.tolist() == [[0.25, 0.0, 0.0]]

    batch = oxyz.read_batch(source)
    assert batch.ptr.tolist() == [0, 1, 2]
    assert batch.frame_indices.tolist() == [0, 1]
    assert batch.metadata["energy"].tolist() == [-1.5, -2.5]
    assert oxyz.infer_schema(source).is_consistent

    output = Path(directory) / "roundtrip.extxyz.gz"
    oxyz.write(output, frames)
    restored = oxyz.read(output)
    assert restored == frames
    assert restored[1].positions.tolist() == [[1.25, 0.0, 0.0]]
    assert restored[1].metadata["energy"] == -2.5
```

`read(path, 0)` returns one frame; the default returns a list. Use `iread` for
frame iteration and `iread_batch` for bounded batches. Plain batches require
compatible fields; projection or typed record batches handle optional data
under explicit contracts. Writing has codec and value-representation limits:
check the [writing guide](https://github.com/theochemtheo/oxyz/blob/main/docs/guides/writing.md)
before replacing source data.

## Documentation

The [development documentation](https://theochemtheo.github.io/oxyz/) tracks `main`.
The guides below link to repository Markdown. Local preview instructions are in
the contributor guide.

| Task | Guide |
| --- | --- |
| First read/inspect/write workflow | [Getting started](https://github.com/theochemtheo/oxyz/blob/main/docs/getting-started.md) |
| Index grammar, partial reads and iteration | [Reading and selecting](https://github.com/theochemtheo/oxyz/blob/main/docs/guides/reading-and-selecting.md) |
| Dense layout, budgets and provenance | [Batching and streaming](https://github.com/theochemtheo/oxyz/blob/main/docs/guides/batching-and-streaming.md) |
| Encoding, incremental output and restrictions | [Writing](https://github.com/theochemtheo/oxyz/blob/main/docs/guides/writing.md) |
| Infer, validate, project, omit and freeze | [Schemas and projection](https://github.com/theochemtheo/oxyz/blob/main/docs/guides/schemas-and-projection.md) |
| Field declarations, binding and typing limits | [Typed records](https://github.com/theochemtheo/oxyz/blob/main/docs/guides/typed-records.md) |
| Optional fields, unions, presence and tags | [Mixed datasets](https://github.com/theochemtheo/oxyz/blob/main/docs/guides/mixed-datasets.md) |
| Compression, archive members, S3 and HuggingFace | [Compressed and remote](https://github.com/theochemtheo/oxyz/blob/main/docs/guides/compressed-and-remote.md) |
| ASE, metatomic and torch-sim conventions | [Converters](https://github.com/theochemtheo/oxyz/blob/main/docs/guides/converters.md) |
| Inspect, check and freeze from the shell | [Command line](https://github.com/theochemtheo/oxyz/blob/main/docs/guides/command-line.md) |

For lookup, see the [API reference](https://github.com/theochemtheo/oxyz/blob/main/docs/reference/api.md),
[data model](https://github.com/theochemtheo/oxyz/blob/main/docs/concepts/data-model.md),
[schema concepts](https://github.com/theochemtheo/oxyz/blob/main/docs/concepts/schemas.md)
and [storage model](https://github.com/theochemtheo/oxyz/blob/main/docs/concepts/storage-and-streaming.md).
[Format](https://github.com/theochemtheo/oxyz/blob/main/docs/reference/extxyz.md),
[ASE compatibility](https://github.com/theochemtheo/oxyz/blob/main/docs/reference/compatibility.md)
and [errors](https://github.com/theochemtheo/oxyz/blob/main/docs/reference/errors.md)
describe grammar, deliberate divergences and failure context.
Self-contained scripts and bundled inputs live in
[`examples/`](https://github.com/theochemtheo/oxyz/tree/main/examples).

## Performance and development

The [saved results](https://github.com/theochemtheo/oxyz/blob/main/benchmarks/RESULTS.md)
contain timings, scaling figures, fixture definitions and environment details.
[Memory measurements](https://github.com/theochemtheo/oxyz/blob/main/benchmarks/MEMORY.md)
are recorded separately. These are workload-specific results, not universal
speed-up or memory guarantees; real-dataset rows require the dataset.
The [benchmark entry point](https://github.com/theochemtheo/oxyz/blob/main/benchmarks/run.py)
and [performance guide](https://github.com/theochemtheo/oxyz/blob/main/docs/development/performance.md)
explain reproduction and comparison limits.

The [architecture map](https://github.com/theochemtheo/oxyz/blob/main/docs/development/architecture.md)
routes changes across Rust core, PyO3 binding and Python policy.
[Testing](https://github.com/theochemtheo/oxyz/blob/main/docs/development/testing.md)
covers parser properties, serial/parallel parity, converter golden tests and
isolated wheel environments.
[Documentation authoring](https://github.com/theochemtheo/oxyz/blob/main/docs/development/documentation.md)
explains executable examples and local preview.

Public names and documented command-line options follow
[Semantic Versioning](https://semver.org/) under the
[release policy](https://github.com/theochemtheo/oxyz/blob/main/RELEASING.md).
Underscore-prefixed modules are implementation details.

## Licence

[MIT](https://github.com/theochemtheo/oxyz/blob/main/LICENSE-MIT) or
[Apache-2.0](https://github.com/theochemtheo/oxyz/blob/main/LICENSE-APACHE),
at your option.
