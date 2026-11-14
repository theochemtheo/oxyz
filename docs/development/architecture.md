# Architecture and ownership

The core has no Python or converter-library knowledge. The binding moves whole
arrays across the Python boundary and releases the interpreter for parsing.
The Python layer owns user-facing selection, schema policy, record declarations
and conversion conventions. Follow these boundaries rather than adding a second
implementation of an existing policy.

## Source and test map

Paths below are relative to the repository root. Links point to tracked source;
the adjacent tests are the starting point for a focused change, not an
exhaustive list of all coverage.

| Concern | Authoritative files | Existing tests |
| --- | --- | --- |
| Grammar and lossless values | [`extxyz.rs`](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-core/src/extxyz.rs), [`model.rs`](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-core/src/model.rs) | [`grammar_conformance.rs`](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-core/tests/grammar_conformance.rs), [`parser_properties.rs`](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-core/tests/parser_properties.rs), [`error_context.rs`](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-core/tests/error_context.rs) |
| Encoding, codecs and archive members | [`encode.rs`](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-core/src/encode.rs), [`decode.rs`](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-core/src/decode.rs) | [`write.rs`](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-core/tests/write.rs), [`compressed.rs`](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-core/tests/compressed.rs), [`decode_sources.rs`](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-core/tests/decode_sources.rs) |
| Structural indexing and parallel reads | [`index.rs`](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-core/src/index.rs), [`extxyz.rs`](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-core/src/extxyz.rs) | [`index_scan.rs`](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-core/tests/index_scan.rs), [`indexed_frames.rs`](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-core/tests/indexed_frames.rs), [`parallel_read.rs`](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-core/tests/parallel_read.rs) |
| Schema fold, projection and batch assembly | [`schema.rs`](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-core/src/schema.rs), [`project.rs`](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-core/src/project.rs), [`batch.rs`](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-core/src/batch.rs) | [`schema_inference.rs`](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-core/tests/schema_inference.rs), [`projection.rs`](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-core/tests/projection.rs), [`batch_build.rs`](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-core/tests/batch_build.rs) |
| Binding and stub contract | [`crates/oxyz-py/src/lib.rs`](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-py/src/lib.rs), [`_rust.pyi`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_rust.pyi) | [`test_native_read.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_native_read.py), [`test_errors.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_errors.py); stubtest in [`prek.toml`](https://github.com/theochemtheo/oxyz/blob/main/prek.toml) |
| Python reads, batches and selection | [`_frames.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_frames.py), [`_batch.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_batch.py), [`_select.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_select.py) | [`test_reader_entries.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_reader_entries.py), [`test_batch.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_batch.py), [`test_batch_sequence.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_batch_sequence.py) |
| Inferred schemas, declared rules and matching | [`_schema.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_schema.py), [`_schema_spec.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_schema_spec.py), [`_schema_match.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_schema_match.py), [`_schema_emit.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_schema_emit.py), [`_project.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_project.py) | [`test_schema_spec.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_schema_spec.py), [`test_schema_match.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_schema_match.py), [`test_schema_read.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_schema_read.py), [`test_project.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_project.py), [`test_schema_emit.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_schema_emit.py) |
| Record declarations and field types | [`_record.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_record.py), [`_record_fields.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_record_fields.py) | [`test_record.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_record.py), [`test_record_fields.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_record_fields.py), [`test_record_typing.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_record_typing.py) |
| Record binding, unions and presence | [`_record_bind.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_record_bind.py), [`_record_union.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_record_union.py), [`_record_batch.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_record_batch.py) | [`test_record_union.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_record_union.py), [`test_record_batch.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_record_batch.py) |
| Remote reads and Python writing | [`_remote.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_remote.py), [`_write.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_write.py) | [`test_remote.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_remote.py), [`test_remote_unit.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_remote_unit.py), [`test_write.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_write.py) |
| Optional output targets | [`ase.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/ase.py), [`metatomic.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/metatomic.py), [`torch_sim.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/torch_sim.py), shared [`_convert.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_convert.py) and [`_torch.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_torch.py) | [`test_ase.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_ase.py), [`test_metatomic.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_metatomic.py), [`test_torch_sim.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_torch_sim.py), [`test_optional_deps.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_optional_deps.py) |
| CLI, exports and public documentation | [`_cli.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_cli.py), [`__init__.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/__init__.py), [`conftest.py`](https://github.com/theochemtheo/oxyz/blob/main/conftest.py) | [`test_cli.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_cli.py), [`test_import.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_import.py), [`test_public_docstrings.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_public_docstrings.py), [`test_doc_examples.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_doc_examples.py) |

## Boundary contracts

The native frame preserves names, kinds and values rather than applying ASE
conventions. Numeric and boolean columns cross as NumPy arrays; string columns
use Python lists. Converter-specific renaming and shape/unit conventions belong
in converters, not the parser. See the
[lossless-core decision](decisions/lossless-core-and-layering.md).

Structural scanning discovers frame boundaries/counts without fully parsing atom
values. Schema inference parses and folds observed signatures. A declared
`SchemaSpec` is policy, not an inferred description; Python compiles and enforces
that policy while Rust applies projection and assembles arrays. See
[schemas and projection](decisions/schemas-and-projection.md).

Records bind declarations to the same reader/projection path. Optional values
on individual records differ from filled dense batch storage; presence masks and
union tags preserve the distinction. See
[typed records and unions](decisions/typed-records-and-unions.md).

## Export and verification obligations

`src/oxyz/__init__.py` defines root exports and `__all__`. Public Python APIs have
inline annotations; the compiled module is described by `_rust.pyi` and checked
against the real module by stubtest. Changes to exported binding symbols must
update the stub in the same change. `py.typed` marks the package as typed.

`pyproject.toml` owns Python/build dependencies and uv rebuild cache keys;
`Cargo.toml` and crate manifests own Rust dependencies/features. The
[test guide](testing.md) maps these boundaries to verification commands. Use
[benchmark evidence](performance.md) for performance claims, and keep optional
libraries behind their [target boundary](decisions/optional-output-targets.md).
