---
name: oxyz-performance
description: Use when profiling or optimising oxyz parsing, binding transfers, allocations, batching, converters, thread scaling, memory use or public benchmark claims.
---

# oxyz performance

Measure the owning layer before optimising; remeasure the same workload after.
Start with the [root router](../oxyz/SKILL.md),
[performance guide](../../../docs/development/performance.md),
[architecture](../../../docs/development/architecture.md) and
[correctness checks](../../../docs/development/testing.md).

## Owners and measurements

| Suspected cost | Owner | Relevant measurement |
| --- | --- | --- |
| Parsing, structural scan, parallel scheduling | `crates/oxyz-core/src/extxyz.rs`, `index.rs` | `crates/oxyz-core/benches/parse.rs`; `benchmarks/test_read.py` |
| Decode/encode, schema projection, dense assembly | Core `decode.rs`, `encode.rs`, `schema.rs`, `project.rs`, `batch.rs` | Read/project/batch cases in `benchmarks/test_read.py`; `test_write.py` |
| Array transfer, Python wrappers and selection | `crates/oxyz-py/src/lib.rs`; Python `_frames.py`, `_batch.py`, `_select.py` | End-to-end reads, selective reads and batches in `benchmarks/test_read.py` |
| Target conversion and tensor copies | Python `ase.py`, `metatomic.py`, `torch_sim.py`, `_torch.py` | `benchmarks/test_systems.py`; same-output converter rows in `test_read.py` |
| Thread/data/system scaling or peak memory | `benchmarks/conftest.py` fixture generators | `benchmarks/test_scaling.py`, `memory.py` |

Core paths are under `crates/oxyz-core/src/`; Python paths under `src/oxyz/`.
Benchmark filenames without a prefix are under `benchmarks/`.

## Bounded starting points

From the checkout root, isolate per-frame parser overhead or compare native
serial/four-thread end-to-end reads:

```sh
cargo bench -p oxyz-core --bench parse -- 'parse/many_small_frames'
uv run benchmarks/run.py -k 'test_read_all_many_small_frames and (oxyz-1t or oxyz-4t)' --benchmark-autosave
```

Both use deterministic 2,000-frame inputs with 16–63 atoms/frame. Rust excludes
input generation/disk I/O from this parse timing; Python includes reading and
materialising NumPy frames. The Python shared fixture also prepares the other
trajectory families, including four 100,000-atom frames. Metadata-heavy inputs
exercise comment parsing; mixed inputs exercise training schemas. Inspect
`conftest.py` before widening selection: scaling reaches a million frames/atoms,
store comparisons build 100,000 × 64-atom datasets and large derived stores.
Do not run the full suite, download real data or generate multi-GB stores by
default. Missing ignored real data is a skip, not measured coverage.

`run.py` has an independent PEP 723 environment below Python 3.14 for comparison
wheels; its conftest rejects debug extensions. A small fixture-shape smoke with
`--benchmark-disable` is not a parser timing. Use [run-oxyz](../run-oxyz/SKILL.md)
for routine builds rather than changing runtime dependencies for benchmarks.

## Boundaries and fair claims

- Profile parsing, allocation, Python overhead and peak memory separately;
  record command, revision, build, interpreter, versions, workload and threads.
- Preserve whole-array binding transfers and detached parsing. Numeric buffers
  become NumPy arrays; strings/dicts and batch offsets still need Python-side
  construction. Batch assembly, gathers, dtype/device conversion and
  `_torch.to_tensor` can allocate/copy. Do not claim the whole pipeline is zero-copy
  or add avoidable hot-path copies/per-atom Python objects.
- Match dataset, selection, output representation and work consumed. Exhaust
  iterators for full-read comparisons. Scan is not parse/schema inference;
  first/last-frame throughput is not whole-file throughput. Report serial and
  parallel modes/thread counts; retain serial value/error parity and source
  seek/buffering constraints. Use [oxyz-core](../oxyz-core/SKILL.md) and
  [oxyz-targets](../oxyz-targets/SKILL.md) for correctness, not timing assertions.
- Public claims trace to [RESULTS.md](../../../benchmarks/RESULTS.md), not copied
  historical timings or machine paths. Regenerate saved-run tables/figures with
  `uv run benchmarks/report.py` and `uv run benchmarks/plot.py`; never hand-edit
  them. Inspect `memory.py` before process-level RSS measurements; timings alone
  do not establish peak memory. Compare before/after on matching environments.
