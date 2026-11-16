---
name: oxyz-core
description: Use when changing oxyz Rust parsing, encoding, lossless values, structural indexing, prefix reads, native batch assembly or serial/parallel error parity.
---

# oxyz core

Keep grammar and represented values independent of Python and target-library
conventions. Start from the [root router](../oxyz/SKILL.md),
[architecture](../../../docs/development/architecture.md) and
[lossless-core decision](../../../docs/development/decisions/lossless-core-and-layering.md).

## Owners and checks

Paths in this table are relative to `crates/oxyz-core/`; tests are Cargo
integration-test targets under `tests/`.

| Concern | Source under `src/` | Test targets |
| --- | --- | --- |
| Grammar and lossless frame values | `extxyz.rs`, `model.rs` | `grammar_conformance`, `parser_properties`, `error_context`, `read_frames` |
| Encoding and represented-value round trips | `encode.rs`, `model.rs` | `write` |
| Structural scan, indexed and prefix reads | `index.rs`, `extxyz.rs` | `index_scan`, `indexed_frames`, `single_pass` |
| Parallel values and deterministic error ordering | `extxyz.rs`, `index.rs` | `parallel_read`, `error_context`, `single_pass` |
| Observed schema fold, projection, dense assembly | `schema.rs`, `project.rs`, `batch.rs` | `schema_inference`, `projection`, `batch_build` |
| Codecs and archive decoding | `decode.rs` | `compressed`, `decode_sources` |

`crates/oxyz-py/src/lib.rs` owns PyO3 whole-array transfer and interpreter
release, not schema policy. Python `_frames.py`, `_batch.py` and `_select.py`
under `src/oxyz/` own orchestration. Use [oxyz-schema](../oxyz-schema/SKILL.md)
for declared rules and [develop-oxyz](../develop-oxyz/SKILL.md) for change discipline.

## Invariants

- Preserve file names, kinds and represented values; do not apply ASE naming,
  units or shape conventions. Lossless does not mean identical text. Consult
  [write limits](../../../docs/guides/writing.md) before promising a round trip.
- Structural scanning is not full value validation. A successful prefix read
  does not validate the suffix; preserve the selected path's scan/parse boundary.
- Hostile input must not panic or cause unbounded allocation. Extend
  `parser_properties.rs` proptests, not just one benign malformed fixture.
- Errors retain zero-based source frame indices, file-absolute one-based lines
  and character columns; unknown coordinates stay absent. Iteration reports a
  parse error once, then terminates. See [errors](../../../docs/reference/errors.md).
- Parallel results match serial results; value-parse failures follow request
  order, not worker completion. Structural damage needed to locate a requested
  frame can take precedence; preserve source coordinates after indexing.
- Avoid unnecessary hot-path allocation, copying and repeated parsing. Keep
  native arrays crossing the binding as whole arrays, not per-atom objects.

## Targeted verification

For a malformed frame's location, inspect `extxyz.rs` and
`tests/error_context.rs::bad_atom_value_locates_line_and_column`:

```sh
cargo test -p oxyz-core --test error_context bad_atom_value_locates_line_and_column
```

For parallel locations use the same target's `parallel_bad_atom_value_matches_serial_location`.
For error precedence use `parallel_read::first_error_in_request_order_wins` and
`single_pass::structural_damage_in_the_needed_prefix_wins`. Run another table
target with `cargo test -p oxyz-core --test <target>`; choose proptests for
hostility, grammar conformance for syntax, `write` for encoding and indexed/parity
targets for selection. Binding-visible changes also need Python
`tests/test_native_read.py`, `tests/test_errors.py` and `_rust.pyi` stubtest via
[run-oxyz](../run-oxyz/SKILL.md). Do not run every routed suite for a prose edit.
