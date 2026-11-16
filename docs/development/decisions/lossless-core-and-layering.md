# Lossless core and layering

Status: Accepted.

## Contract

The Rust core represents extxyz columns and metadata with their file names,
kinds and values. It owns parsing, encoding, decompression, structural indexing,
schema folding, projection execution and batch assembly. It has no Python or
ASE dependency. Lossless here means preserving represented data, not original
whitespace, quoting or byte-for-byte file layout.

PyO3 exposes core operations as `oxyz._rust`, transfers whole arrays and releases
the interpreter during native parsing. Python supplies the public dataclasses,
selection/batch planning, declared-schema policy, typed records and CLI. Library
conversion conventions live in dedicated Python converters.

## Rationale and trade-offs

One data-preserving core lets callers inspect fields without a target library
renaming or discarding them. Rust-only tests can guard the grammar, hostile
input and serial/parallel parity independently of Python installations. Whole
array transfers avoid a Python call per atom.

This boundary deliberately leaves interpretation to the caller or converter.
An ASE-shaped view may rename fields or interpret stress; it is not the native
frame representation. Keeping both layers consistent requires binding stubs,
Python types and boundary tests alongside native tests. Moving converter policy
into parsing would couple every consumer to one target's conventions.

## Evidence

- [`model.rs`](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-core/src/model.rs)
  and [`encode.rs`](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-core/src/encode.rs)
  define represented values and their encoding.
- [`lib.rs`](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-py/src/lib.rs)
  implements the binding and interpreter detachment.
- [`write.rs`](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-core/tests/write.rs),
  [`parser_properties.rs`](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-core/tests/parser_properties.rs)
  and [`test_native_read.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_native_read.py)
  guard round trips, parser invariants and native output.

See [architecture](../architecture.md) for the full ownership map and
[optional output targets](optional-output-targets.md) for converter boundaries.
