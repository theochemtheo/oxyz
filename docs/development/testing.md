# Testing

Run from the checkout root. Use `uv run` for Python so source changes cannot
leave the extension stale. Start with the narrow test that owns the behaviour;
a documentation-only edit does not need a benchmark run.

## Choose the owning layer

| Promise | Check |
| --- | --- |
| Grammar, schema fold, projection, batch assembly, index | `cargo test -p oxyz-core` with the relevant integration-test target |
| Hostile input never panics or over-allocates | Extend `crates/oxyz-core/tests/parser_properties.rs` proptests |
| Parallel/selected reads preserve serial results and error ordering | Existing parity tests in `parallel_read.rs` and `indexed_frames.rs` |
| Python API, return shapes, errors, CLI | A focused module under `tests/` |
| Converter matches its reference library | Golden/parity tests for that target, including field values and shapes |
| Documentation examples and runner isolation | `uv run pytest docs src/oxyz` and `tests/test_doc_examples.py` |

For malformed input, assert the source frame and location, not merely an
exception type. Prefer small deterministic fixtures; the
[test data index](https://github.com/theochemtheo/oxyz/blob/main/tests/data/README.md)
describes existing cases. The
[architecture map](architecture.md#source-and-test-map) links owners to tests.

## Source and wheel environments

Full source CI runs `uv sync --extra ase` and `uv run pytest` with
`OXYZ_TEST_TIER=full`. The default development environment includes all optional
targets and S3 test tools. Missing extras must fail that tier's environment
canary, not silently reduce its coverage. Documentation converter examples also
require those extras.

Wheel CI builds once, then installs that wheel into intentionally incomplete
environments. Its sequence is:

```sh
uv sync --no-default-groups --group test --no-install-project
uv pip install --no-deps dist/*.whl
uv run --no-sync pytest tests
```

This assumes the built wheel is already in `dist/`. ASE-only adds `--extra ase`
to the sync; S3-only adds `--group s3-test --extra s3`. Base adds neither.
`--no-install-project` prevents a checkout rebuild, `--no-deps` preserves the
chosen tier, and `--no-sync` keeps the installed wheel when tests run. Do not
replace the last command with ordinary `uv run pytest`: that could resynchronise
the full environment and test different code/dependencies. Optional guards and
the `OXYZ_TEST_TIER` canary live in
[`test_optional_deps.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_optional_deps.py).

The `test` group includes Sybil without optional targets. Docs renderer tooling
is in a separate `docs` group. Minimal wheel tiers collect `tests` explicitly;
they do not execute package docstrings or optional-converter documentation.
Collection of ordinary tests and standalone benchmarks must not import Sybil or
optional converters through the root hook.

## Documentation and coverage

Sybil replaces pytest's built-in doctest plugin. Public Markdown and package
docstrings use the isolated namespace/workdir contract described in
[writing documentation](documentation.md). New examples need visible imports and
inputs; shell fences are not executable coverage. Default collection covers
`tests`, `src/oxyz`, `docs` and `README.md`.

[`scripts/coverage.sh`](https://github.com/theochemtheo/oxyz/blob/main/scripts/coverage.sh)
uses `cargo llvm-cov` and an instrumented extension to combine Rust and Python
coverage. It requires the full environment and the Rust LLVM coverage tools;
it runs default pytest collection with `--no-sync` to retain instrumentation.
Do not bypass ordinary uv rebuilding for routine tests: this is an explicit
coverage-only build workflow. Coverage outputs live under `coverage/` and
`htmlcov/`, and the script enforces its configured floors.

Coverage cleanup warnings are fatal. CI uses separate Cargo and uv caches for
coverage and initialises Cargo's cache marker before the first extension build.
An existing unmarked target directory is not retagged to bypass safe cleanup.

## Before a pull request

These commands match the repository's verification gates:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run python -m mypy.stubtest oxyz._rust --allowlist stubtest-allowlist.txt
uv run pytest
uv run rumdl check .
```

`cargo test -p oxyz-core` is the fast standalone core loop; the PR gate uses the
workspace. `ty` checks Python annotations; stubtest additionally compares the
binding stub with the compiled module, which static source checking cannot do.
The authoritative hook commands and stages are in
[`prek.toml`](https://github.com/theochemtheo/oxyz/blob/main/prek.toml), and CI
platforms/interpreters/tier invocations are in
[`test.yml`](https://github.com/theochemtheo/oxyz/blob/main/.github/workflows/test.yml).

Update the changelog for user-visible changes, keep dependency tiers isolated
and rebase onto `origin/main` before opening a PR. None of these checks publishes
a package or a website; follow the
[release procedure](https://github.com/theochemtheo/oxyz/blob/main/RELEASING.md)
when a release is actually requested.
