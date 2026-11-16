# Optional output targets

Status: Accepted.

## Contract

Native frames and batches need no converter. ASE, metatomic and torch-sim are
first-class optional targets, exposed through `oxyz.ase`, `oxyz.metatomic` and
`oxyz.torch_sim` with installation extras `ase`, `metatomic` and `torch-sim`.
Remote S3 access uses the separate `s3` extra. Base runtime dependencies are
NumPy and PyYAML, not these target libraries.

Root package imports must not eagerly import converter libraries. Importing a
target without its dependencies raises an informative `ImportError` naming the
extra rather than silently substituting a different output. The core remains
unaware of target-library fields, tensors, devices and conventions.

## Rationale and trade-offs

Optional extras let users keep a small environment while installing a specific
integration deliberately. A dedicated converter owns its target's field mapping,
shape restrictions and error surface; shared helpers avoid duplicating common
conversion logic without pushing that policy into Rust parsing.

Each target needs its own parity/golden tests and dependency guards. A full
development installation cannot alone prove isolation: tests must also exercise
missing dependencies and wheel installations with incomplete extras. Targets
may reject a native frame that their own representation cannot express; no fake
fallback should hide that mismatch.

## Evidence

- [`pyproject.toml`](https://github.com/theochemtheo/oxyz/blob/main/pyproject.toml)
  separates runtime dependencies, extras, testing tools and documentation tools.
- [`__init__.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/__init__.py)
  exports the native surface without eager target imports.
- [`test_optional_deps.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_optional_deps.py)
  guards install hints, import-state restoration and declared CI tier contents.
- [`test_ase.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_ase.py),
  [`test_metatomic.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_metatomic.py)
  and [`test_torch_sim.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_torch_sim.py)
  check each target's conversion contract.

See [testing](../testing.md#source-and-wheel-environments) for isolated tiers and
[lossless core and layering](lossless-core-and-layering.md) for the shared model.
