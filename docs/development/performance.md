# Performance work

Start with a measured workload. A parser optimisation, a converter change and
a batching policy change do different work: comparing their times without the
output contract can hide extra allocations or a change in semantics.

## Published evidence

[`benchmarks/RESULTS.md`](https://github.com/theochemtheo/oxyz/blob/main/benchmarks/RESULTS.md)
is the source for public performance claims. It records the saved run, machine,
interpreter, package versions, workloads and output/mode columns. Its numbers
are only comparable within one run on one machine; they are not a promise for
all datasets, platforms or later revisions.

Compare like outputs: native NumPy frames, full ASE objects and concatenated
batches have different materialisation costs. Selective reads, first/last-frame
reads, structural scans and whole-file parsing also do different work. A cheap
scan is not a full schema inference. Report thread counts and serial/parallel
mode rather than attributing every difference to parsing speed.

## Entry points

Run from the repository root:

```sh
uv run benchmarks/run.py --benchmark-disable -k test_dataset_fixture_matches_corpus_shape
uv run benchmarks/run.py --benchmark-autosave
uv run benchmarks/report.py
uv run benchmarks/plot.py
cargo bench -p oxyz-core --bench parse
```

The first command is a small fixture smoke, not a timing result. The full suite
can generate large stores and fixtures; select the relevant cases before
running it. `benchmarks/run.py` uses a separate PEP 723 environment with its own
Python constraint and comparison libraries. It currently resolves a supported
interpreter below Python 3.14 because some comparison parsers lack 3.14 wheels.
It builds this checkout in release mode. The benchmark conftest rejects a debug
extension because debug timing is misleading.

`report.py` renders results from saved pytest-benchmark JSON; `plot.py` produces
figures. Refresh generated results through these tools, not hand-edited tables.
The Rust `parse` benchmark isolates core work from PyO3/Python overhead.
[`benchmarks/memory.py`](https://github.com/theochemtheo/oxyz/blob/main/benchmarks/memory.py)
provides separate process-level memory measurements; inspect its invocation and
workload choices before using it rather than inferring peak memory from timings.

## Fixtures and limits

Deterministic generators and cache keys live in
[`benchmarks/conftest.py`](https://github.com/theochemtheo/oxyz/blob/main/benchmarks/conftest.py).
They cover many small frames, large systems, metadata-heavy and mixed training
frames, scaling sweeps and binary-store comparisons. Store fixtures have a large
first-run disk cost; generated files are cached under `benchmarks/.cache/`.
Saved timing runs are under `.benchmarks/`.

The optional real dataset is under ignored `data/`; it is not part of a fresh
checkout. Missing real-data rows skip and do not prove that dataset's coverage.
Do not download a large corpus merely to validate documentation. Base runtime
and test dependencies must not grow to satisfy a standalone benchmark script.

## Investigate before changing code

Use a profiler suited to the suspected layer: Python call overhead, native
parsing, copying at the array boundary, or allocation/peak memory. Record the
actual profiler command, workload, build profile and environment with the
measurement; there is no requirement to reproduce a developer's local profiler
setup. Then change the owning layer and remeasure the same workload.

Preserve serial/parallel parity, source error ordering and converter correctness
while optimising. Avoid added per-atom Python work and unnecessary copies across
the binding. Keep batching and seek/buffering constraints visible rather than
hiding memory growth behind a faster timing. See [architecture](architecture.md)
for owners and [testing](testing.md) for the correctness gates.
