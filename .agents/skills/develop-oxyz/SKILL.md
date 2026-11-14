---
name: develop-oxyz
description: Use when changing oxyz code, fixing a bug, adding a feature or changing a public API or dependency.
---

# Develop oxyz

Keep correctness, measured speed, typed ergonomics and a small layered design
together. Start with the [root router](../oxyz/SKILL.md) and the
[architecture map](../../../docs/development/architecture.md) for owners/tests;
use [run-oxyz](../run-oxyz/SKILL.md) for setup and commands.

## Ownership and verification

| Layer | Responsibility | Owning check |
| --- | --- | --- |
| `crates/oxyz-core` | Grammar, lossless model, index, schema fold, projection, batch assembly and encoding; no Python or ASE | Relevant `cargo test -p oxyz-core` target |
| `crates/oxyz-py` | Thin PyO3 boundary, whole-array transfer, parsing with the GIL released | Core tests plus Python integration and `_rust.pyi` stubtest |
| `src/oxyz` | Typed API, selection/batch planning, schemas, records, CLI and lazy converters | Focused `uv run pytest` module and `uv run ty check` |

For a bug, use the report or existing failing test as the repro; add a regression
only where the path lacks coverage. Test at the layer owning the promise before
implementing. Hostile input needs proptests; parallel reads need serial/error
parity; converters need field-by-field goldens; malformed input needs frame and
location assertions. See [testing](../../../docs/development/testing.md).

## Change constraints

- Resolve a new public contract before implementation; restore documented
  behaviour rather than weakening checks to accommodate a bug.
- Keep hot paths free of avoidable allocations, copies and computation.
  Profile before optimising; back claims with
  [benchmark evidence](../../../docs/development/performance.md).
- Base runtime dependencies are NumPy and PyYAML. New hard dependencies need
  justification. ASE, metatomic, torch_sim and remote support remain lazy extras;
  preserve minimal wheel tiers. Tooling belongs in development/test/docs groups,
  not runtime requirements.
- Migrate callers cleanly. Update public exports/inline types and `_rust.pyi`
  alongside the implementation; check stubtest as well as static typing.
- Update user docs and the changelog for changed contracts. Follow `AGENTS.md`
  for comments/docstrings and British spelling. Assertions are not runtime
  input validation.
- Record durable rationale in the public
  [accepted decisions](../../../docs/development/decisions/index.md), using its
  status/supersession convention. Private session notes are not public authority.

Run the narrow owning checks first, then the
[PR gates](../../../docs/development/testing.md#before-a-pull-request) before
opening a PR. A smoke tour is not the full suite, a benchmark or wheel-tier proof.
