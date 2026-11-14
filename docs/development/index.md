# Development

oxyz has a pure Rust core, a PyO3 extension and a typed Python surface. Start
with the [architecture map](architecture.md) to locate the owner of a change.

## Set up a checkout

Install a Rust toolchain and uv 0.13.0. The project requires Python 3.12 or newer;
uv selects a compatible interpreter. Run commands from the repository root:

```sh
uv sync --extra ase
uv run python -c "import oxyz; print(oxyz.__file__)"
uv run prek install
```

The default `dev` group includes test tools and optional converter/remote
libraries. `--extra ase` also declares the ASE user extra explicitly, matching
full CI. The compiled extension is built in release mode by maturin through uv.
Use `uv run` for Python, not an interpreter invoked directly from the environment.

The uv cache keys in
[`pyproject.toml`](https://github.com/theochemtheo/oxyz/blob/main/pyproject.toml)
include that file, root Cargo manifests/lockfile, crate manifests and Rust
sources. A Rust edit therefore triggers an extension rebuild on the next uv
run. Do not bypass this path and accidentally test an old extension. A Rust-only
change can first be checked with `cargo test -p oxyz-core` without Python.

## Make a change

1. Find the owner and existing tests in the architecture map. Keep grammar and
   array assembly in Rust, Python policy in Python and interop at the boundary.
2. For a bug, reproduce it at the owning layer; add a regression when existing
   tests do not cover the path. Write a failing test before implementation.
3. Implement the smallest coherent change and run the affected checks.
4. Keep public exports, inline types, binding stubs, documentation and changelog
   consistent with the new contract. Avoid compatibility shims for private code.
5. Before a PR, run the [full verification gates](testing.md#before-a-pull-request).

Comments explain constraints the code cannot express, not the code itself.
Public docstrings document user contracts and gotchas. Use British spelling.
New runtime dependencies need a concrete justification; optional integrations
belong behind extras and lazy imports rather than base package imports.

## Choose an environment

The full source environment exercises every converter and documentation example.
A passing run there does not prove minimal installations: CI also installs a
built wheel into base, ASE-only and S3-only environments, deliberately without
other extras. See [testing](testing.md#source-and-wheel-environments) before
changing dependencies or collection.

The [documentation authoring guide](documentation.md) explains per-page example
isolation and preview/review. [Performance work](performance.md) needs measured
evidence, not an assumed speed-up. The [accepted decisions](decisions/index.md)
record current boundaries and trade-offs.

## Releases

Follow the repository's
[release procedure](https://github.com/theochemtheo/oxyz/blob/main/RELEASING.md)
for versioning, tags, provenance and publication approval. Building documentation
locally does not authorise a package release or website deployment.
