# Performance work

Start with a measured workload. A parser optimisation, a converter change and
a batching policy change do different work: comparing their times without the
output contract can hide extra allocations or a change in semantics.

## Published evidence

The [performance overview](../performance.md) presents the real-data and
dataset-size figures; the [complete results](../reference/benchmarks.md) include
the saved run, machine, interpreter, package versions, workloads and output/mode
columns. The tables are included directly from the generated
[`benchmarks/RESULTS.md`](https://github.com/theochemtheo/oxyz/blob/main/benchmarks/RESULTS.md).
Numbers are only comparable within one run on one machine; they are not a
promise for all datasets, platforms or later revisions.

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
Python constraint and comparison libraries. Parser versions are pinned in the
script; the saved run records the interpreter and all resolved package versions.
It builds this checkout in release mode. The benchmark conftest rejects a debug
extension because debug timing is misleading.

`report.py` renders results from saved pytest-benchmark JSON; `plot.py` produces
figures. Refresh generated results through these tools, not hand-edited tables.
The Rust `parse` benchmark isolates core work from PyO3/Python overhead.
[`benchmarks/memory.py`](https://github.com/theochemtheo/oxyz/blob/main/benchmarks/memory.py)
provides separate process-level memory measurements; inspect its invocation and
workload choices before using it rather than inferring peak memory from timings.

## Reproduce the published snapshot

The selected run is archived under `docs/assets/benchmarks/` and downloadable
from the [performance overview](../performance.md).
Only the local hostname has been omitted. Timing samples, workload metadata,
package versions and the measured revision are retained. The archive is
separate from ignored ad-hoc saves under `.benchmarks/`.

These commands regenerate the tables and approved figures from that tracked
input, without benchmarking or downloading data:

```sh title="Regenerate the published benchmark views"
save=docs/assets/benchmarks/oxyz-performance-20261116T105540Z.json
uv run benchmarks/report.py "$save"
uv run benchmarks/plot.py "$save" --only real_data scaling_dataset \
    --output-dir docs/assets/benchmarks
```

The committed figures under `docs/assets/benchmarks/` are the canonical site
assets. Do not maintain copies under `benchmarks/figures/`. The plot script's
default output directory remains useful for ad-hoc figures; `--output-dir`
selects a different destination. Both generators otherwise default to the
newest local save, which may be a partial run: always supply an explicit input
when updating published evidence.

The overview includes the generated `ENVIRONMENT.md`; the complete results
include `RESULTS.md`. Both are rendered from the same input by `report.py`.
Regenerate evidence outside the docs build, and commit the selected input,
environment, tables and corresponding figures together.
The plot script loads the site's bundled fonts and exports glyph outlines;
it does not depend on fonts installed on the rendering machine.

Regeneration is not a repeat of the original measurement. To remeasure that
workload, use a separate checkout of the recorded revision with its recorded
interpreter and package versions. Real-data measurements additionally require
the cited MAD training file at `data/mad-1.5-r2scan-train.xyz`; absent input
skips those rows. Do not replace a complete published run with such a partial
save or describe old measurements as current-version performance.

The published snapshot selects the two figure workloads, including all their
comparison readers. The measurement command is:

```sh title="Measure the published workloads"
mkdir -p .benchmarks
run=.benchmarks/oxyz-performance-$(date -u +%Y%m%dT%H%M%SZ).json
uv run benchmarks/run.py -q \
    -k 'test_read_all_mad_full or test_scaling_dataset' --benchmark-only \
    --benchmark-json "$run"
```

Choose a new output filename for a new measurement rather than overwriting
an existing run. The recorded checkout had uncommitted changes, which remain
explicit in the snapshot's revision metadata.

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
