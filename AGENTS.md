# AGENTS.md

Onboarding for coding agents working in this repo. It is a distillation, not
a manual — for fuller guidance see `.claude/skills/develop-oxyz/SKILL.md`
(conventions, the loop, the gates) and `.claude/skills/run-oxyz/SKILL.md`
(build/run/test mechanics, repo tour, troubleshooting).

## What oxyz is

`oxyz` is a Rust+Python extxyz reader/writer, aiming to be:

- **The go-to extxyz reader** for atomistic simulation and ML — correct
  against the grammar and against real-world datasets.
- **Fast**, with claims backed by benchmarks, not intuition.
- **Type-safe and ergonomic in Python** — frozen dataclasses, full stubs,
  `py.typed`; numpy arrays are the native output, with `ase.Atoms`,
  `metatomic.torch.System`, and `torch_sim.SimState` as first-class
  converters over the same core data.
- **Thoroughly tested**, with tests that encode the promises being made.
- **Easy to understand** — a small, layered codebase.

## Repo map

Three layers, boundary chosen so each tests standalone:

- `crates/oxyz-core` — pure Rust: parser, lossless `Frame` model, schema
  fold, batch assembly, structural index. No Python or ASE knowledge.
- `crates/oxyz-py` — the PyO3 binding, built as `oxyz._rust`. Thin; data
  crosses as whole numpy arrays, parsing runs with the GIL released.
- `src/oxyz` — the typed Python surface: frozen dataclasses, batch
  planning, the index grammar, the CLI (`oxyz`), and the lazily-imported
  output-target converters (`oxyz.ase`, `oxyz.metatomic`, `oxyz.torch_sim`).

## Build / run / test

Always drive Python through `uv run` — it rebuilds the Rust extension
(release, via maturin) whenever the Rust sources change, so you never test
against a stale `.so`. The first run after a Rust edit takes ~10–30s to
compile; later runs are instant.

```sh
uv run python -c "import oxyz"                                   # build + import
uv run pytest -q                                                  # Python suite
cargo test -p oxyz-core -q                                        # Rust core suite
cargo clippy --workspace --all-targets --all-features -- -D warnings
uv run ruff check
uv run ty check
```

## The loop

Red/green TDD: write the failing test first, at the layer that owns the
behaviour, watch it fail, implement the minimum to pass.

- Parser/schema/batch/index behaviour → a `cargo test -p oxyz-core` test.
- API shape, UX, error surface, ASE-equivalence → `uv run pytest`.
- A feature spanning layers gets a test at each layer that owns part of it.

Match the test to the promise:

- never panics / never over-allocates on hostile input → a **proptest**.
- parallel reads match serial, byte for byte → a **parity** test.
- a converter matches its reference library field for field → a
  **golden** test.
- a malformed file → assert the frame index and location, not just that an
  error was raised (**error-context**).

## Gates before a PR

- `cargo clippy --workspace --all-targets --all-features -- -D warnings` clean.
- `uv run ruff check` and `uv run ty check` clean.
- Full Rust (`cargo test -p oxyz-core`) and Python (`uv run pytest`) suites pass.
- User-visible changes have a CHANGELOG entry.
- Rebase onto `origin/main` before opening the PR — the ruleset blocks
  branches that are behind.
