---
name: run-oxyz
description: Use when building, running, testing or touring oxyz, exercising its Python API or CLI, or previewing its documentation.
---

# Run oxyz

Use `uv run` for Python: uv's Rust-source cache keys trigger a release maturin
rebuild after Rust edits. A bare environment interpreter can load a stale
extension. Install uv and a Rust toolchain; follow
[checkout setup](../../../docs/development/index.md#set-up-a-checkout).

## Smoke tour

From the checkout root:

```sh
uv sync --extra ase
uv run python .agents/skills/run-oxyz/driver.py
```

[driver.py](driver.py) finds committed tiny fixtures from its resolved path and
uses temporary outputs. It checks values and round trips for reads, indexing,
iteration, batches, schema inference/validation/projection, compressed sources,
typed records/union batches, writers and CLI scan/check/freeze. Errors exit
non-zero. Installed optional targets are exercised; absent dependencies print
explicit `SKIP` lines and appear in the summary. A broken installed converter
fails rather than becoming a skip.

From another working directory, substitute the checkout's absolute path:

```sh
checkout=/path/to/oxyz
uv run --project "$checkout" python "$checkout/.agents/skills/run-oxyz/driver.py"
```

This is local smoke coverage, not every API combination, the full suites,
performance evidence, isolated wheel tiers or live S3/HuggingFace coverage.
Do not describe skipped converters as tested. A minimal install need not contain
extras; the default development environment includes them.

## Focused commands

Run from the checkout root:

```sh
uv run python -c "import oxyz; print(oxyz.read('tests/data/simple.extxyz', 0).n_atoms)"
uv run oxyz scan tests/data/schema_conformant.extxyz --no-schema
uv run oxyz --help
uv run pytest -q tests/test_cli.py
cargo test -p oxyz-core -q
uv run pytest docs README.md src/oxyz
```

`read(path, 0)` returns one frame; other selections return lists. `scan` is
structural; CLI `scan` also parses for schema unless `--no-schema` is given.
Use the [root router](../oxyz/SKILL.md) for API choices,
[CLI guide](../../../docs/guides/command-line.md) for schema/check/freeze commands,
and [testing](../../../docs/development/testing.md) for all gates, stubtest and
source-versus-wheel environments. Development changes follow
[develop-oxyz](../develop-oxyz/SKILL.md).

## Local documentation preview

```sh
uv run --group docs zensical build --clean
uv run --group docs zensical serve --dev-addr 127.0.0.1:8000
```

Follow the [preview/review protocol](../../../docs/development/documentation.md#preview-and-review):
inspect actual pages in a browser, including narrow screens; present live URLs
and request explicit revision-specific approval at required gates. Passing checks
or silence is not approval. README needs separate Markdown review. Stop an owned
preview before a clean build to avoid competing writes; never stop an unrelated
server. Use another loopback port if occupied. Previewing is not publishing.
