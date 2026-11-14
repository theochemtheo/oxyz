# API reference

Import public names from `oxyz`; their implementations live in smaller modules.
The links below open the authoritative source docstrings, including overloads,
parameter types, return values and errors. In Python, `help(oxyz.read)` or
`help(oxyz.SchemaSpec)` exposes the same contracts. This is an authored index,
not a generated dump of private helpers.

## Choose an operation

| Need | Public operation | Walkthrough |
| --- | --- | --- |
| One frame or a selected list | [`read`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_frames.py) | [Getting started](../getting-started.md), [selection](../guides/reading-and-selecting.md) |
| Iterate selected frames | [`iread`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_frames.py) | [Selection](../guides/reading-and-selecting.md) |
| One concatenated batch | [`read_batch`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_batch.py) | [Batching](../guides/batching-and-streaming.md) |
| A sequence of batches | [`iread_batch`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_batch.py) | [Batching](../guides/batching-and-streaming.md) |
| Frame offsets/counts without a full parse | [`scan`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_scan.py) | [Storage and streaming](../concepts/storage-and-streaming.md) |
| Observed fields, variants and consistency | [`infer_schema`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_schema.py) | [Schemas and projection](../guides/schemas-and-projection.md) |
| Encode objects or stream output | [`write`, `Writer`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_write.py) | [Writing](../guides/writing.md) |
| Named typed fields or typed batches | `read`/`iread(record=R)`, `read_batch`/`iread_batch(record=RecordBatch[R])` | [Typed records](../guides/typed-records.md) |
| Mixed record types and presence | `record=A | B`, `RecordBatch[A | B]` | [Mixed datasets](../guides/mixed-datasets.md) |
| Compressed files, archive members and remote sources | Reader `compression`/`member`/`storage_options` | [Compressed and remote](../guides/compressed-and-remote.md) |
| Scientific target objects | `oxyz.ase`, `oxyz.metatomic`, `oxyz.torch_sim` | [Converters](../guides/converters.md) |
| Inspect or validate from the shell | `oxyz scan`, `oxyz check`, `oxyz freeze` | [Command line](../guides/command-line.md) |

### Reader call shapes

The following is **signature notation**, not runnable Python. Return annotations
show the native outputs; `record=` changes the output type as described below.

<!-- skip: next "API signature notation uses undefined path and non-executable return annotations" -->

```python {.no-copy title="Reader signatures — not runnable Python"}
read(path, index=":", *, record=None, threads=None, schema=None,
     conformance="required", mode=None, compression="infer", member=None,
     storage_options=None) -> Frame | list[Frame]
iread(path, index=":", *, record=None, schema=None, conformance="required",
      mode=None, compression="infer", member=None,
      storage_options=None) -> Iterator[Frame]
read_batch(path, index=":", *, record=None, threads=None, schema=None,
           conformance="required", mode=None, compression="infer", member=None,
           storage_options=None) -> Batch
iread_batch(path, *, record=None, frames_per_batch=None, atoms_per_batch=None,
            memory_scales_with=None, max_scaler=None, shuffle=False, seed=None,
            threads=None, schema=None, conformance="required", mode=None,
            compression="infer", member=None, storage_options=None) -> Iterator[Batch]
scan(path, *, with_volume=False, compression="infer", member=None,
     storage_options=None) -> FrameIndex
infer_schema(path, *, compression="infer", member=None,
             storage_options=None) -> Schema
```

- `path` is a string or `Path`. Source/selection restrictions depend on the
  operation; see [storage and streaming](../concepts/storage-and-streaming.md).
  [Compressed and remote sources](../guides/compressed-and-remote.md) covers
  source codecs, archive selection, the `oxyz[s3]` extra and credential syntax.
- `index` accepts an integer, a slice, a slice string such as `"1:8:2"`, or an
  ordered sequence of non-negative integers, with repetitions allowed.
  Integer selection, including an integer string, returns one frame from `read`;
  slice/list selection returns a list. `iread` always returns an iterator.
- `threads=None` uses all cores where that parameter exists; `1` is serial.
  `iread` has no `threads` parameter.
- `schema` is a `SchemaSpec` or a path to a JSON/YAML/TOML spec. `conformance`
  controls validation handling; `mode` overrides the spec's validate/project
  choice. Projection reshapes data and can drop an unfillable frame under
  `conformance="warn"`; source frame indices remain the provenance.
- `record=R` on frame readers returns instances of the `Record` subclass `R`
  or members of a sibling-record union. Batch readers require `RecordBatch[R]`
  or a declared batch subclass. `record` cannot be combined with `mode`, even
  `mode="project"`. With `record`, `schema` checks agreement for declared fields
  and supplies frame constraints; rules for unrelated keys are not applied.
  See [typed records](../guides/typed-records.md) for binding restrictions.
- `iread_batch` requires exactly one batching strategy: `frames_per_batch`,
  `atoms_per_batch`, or `memory_scales_with` plus `max_scaler`. These are budgets,
  not a promise to split an oversized frame. Source seekability restricts which
  strategies work; see [batching](../guides/batching-and-streaming.md).
- `compression` accepts `"infer"`, `"none"`, `"gzip"`, `"zstd"`, `"zip"`,
  `"tar"`, `"tar.gz"` or `"tar.zst"`. `member` selects an archive entry;
  `"*"` combines extxyz entries in archive order. `storage_options` supplies
  store configuration rather than altering the parsed representation.
- `scan(with_volume=True)` additionally reads comment lines to report volumes.
  `infer_schema` parses the entire logical stream; it is not the cheap scan.

## Data objects

| Public object | Contract and useful members |
| --- | --- |
| [`Frame`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_frames.py) | `n_atoms`, `columns`, `metadata`; mapping-style lookup and derived `positions`, `numbers`, `symbols`, `cell`, `pbc` |
| [`Batch`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_batch.py) | `n_frames`, `total_atoms`, `n_atoms`, `offsets`/`ptr`, `frame_indices`, per-atom `batch`; integer/slice/gather access |
| [`FrameIndex`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_scan.py) | Structural `offsets`, `n_atoms`, optional `volumes` and count statistics |
| [`Schema`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_schema.py) | Observed columns/metadata, variants, counts, `is_consistent`, `report()` and `to_spec()` |
| [`ColumnSchema`, `MetadataSchema`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_schema.py) | Per-field variants, `frames_present` and a promotable `unified` signature |
| [`ColumnVariant`, `MetadataVariant`, `Kind`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_schema.py) | Observed kind/width or shape and counts; REAL/INT/BOOL/STR kinds |

See [data model](../concepts/data-model.md) for value equality, mutability and
mapping ambiguity, and [format](extxyz.md) for file syntax. Inferred observations
are not a declared dataset contract.

## Declared schemas and typed records

| Public object | Purpose |
| --- | --- |
| [`SchemaSpec`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_schema_spec.py) | Declare columns, metadata and frame rules; load/save with `from_file`/`to_file`, find rules with `rule_for`, expand patterns with `freeze` |
| [`ColumnRule`, `MetadataRule`, `FrameRule`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_schema_spec.py) | Kind/width/shape, requiredness, fills, absence handling, ranges and structural constraints |
| [`Record`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_record.py) | Typed per-frame declaration, `spec` and `from_frame` conversion |
| [`Column`, `Meta`, `Cell`, `Matrix3`, `FieldSpec`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_record.py) | Record field declarations, file-key binding and normalised cell/matrix fields |
| [`RecordBatch`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_record_batch.py) | Dense typed fields, `present`, union `members`/`tags`/`where`, sequence and gather operations |

Start with [schemas and projection](../guides/schemas-and-projection.md) or
[typed records](../guides/typed-records.md); [schema concepts](../concepts/schemas.md)
explain validation, projection and presence. For heterogeneous input, follow
[mixed datasets](../guides/mixed-datasets.md).

| Operation or member | Contract |
| --- | --- |
| `Schema.to_spec()` | Convert observations into editable rules, not a guarantee about future input |
| `SchemaSpec.freeze(sample)` | Expand patterns to observed literal keys; projection requires literal rules |
| `R.spec()` / `RecordBatch[R].spec()` | Project-mode schema; plain reads with it do not preserve record missingness or union identity |
| `R.from_frame(frame)` | Bind one frame; unchanged arrays become read-only views |
| `RecordBatch[R].from_batch(batch)` | Single-record conversion; cannot recover per-frame presence from an ordinary batch |
| `batch[R.field]` / `batch["field"]` | Dense batch-form value; strings are declared field names, not file keys |
| `batch.present(key)` | Read-only per-frame mask for an optional or member-specific field; other fields are present everywhere |
| `batch.members` / `batch.tags` | Member classes in declaration order; read-only integer tags index that tuple |
| `batch.where(A, B)` | Boolean mask selecting exact union members, not subclasses or shared bases |

The [schema decision](../development/decisions/schemas-and-projection.md) and
[record decision](../development/decisions/typed-records-and-unions.md) explain
these boundaries. Runtime shape/dtype checking is stronger than static typing:
generic batches use field keys for typed lookup, while declared batch subclasses
also expose typed attributes. See the [typing limits](../guides/typed-records.md#typing-limits)
before relying on constructor checks from an editor.

## Writing

[`write` and `Writer` docstrings](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_write.py)
are authoritative for accepted objects and encoder options. Call shapes below
are signature notation, not runnable Python:

<!-- skip: next "writer signature notation requires undefined path/obj and is not executable Python" -->

```python {.no-copy title="Writer signatures — not runnable Python"}
write(path, obj, *, append=False, compression="infer", level=None,
      threads=None) -> None
Writer(path, *, append=False, compression="infer", level=None, batch=None)
```

`Writer.write(obj)` appends objects to the open encoder, and `Writer.close()`
finalises it; use the context manager to ensure closure. Both accept native
frames or ASE objects. A native `Batch` is accepted as an iterable of frames;
records and record batches need explicit conversion to writable frames.
`Writer.write(iterable)` materialises that iterable, even with `batch=n`.
See [writing](../guides/writing.md) for codecs, append/stdout restrictions,
verified round trips, string-quoting limitations and bounded-memory output.

## Errors and keyword/value types

| Public names | Meaning |
| --- | --- |
| [`OxyzError`, `ParseError`](errors.md) | Package error base and structured parse location; binding declarations in [`_rust.pyi`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_rust.pyi) |
| [`SchemaError`, `SchemaWarning`, `Violation`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_schema_match.py) | Schema deviation errors/warnings and structured violation records |
| [`FieldError`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_convert.py) | Conversion field validation failure |
| [`AmbiguousNameError`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_lookup.py) | A lookup name exists in both columns and metadata |
| [`ColumnValues`, `MetadataValue`, `Compression`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_frames.py) | Native column/metadata value unions and codec names |
| [`Conformance`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_schema_match.py) | `"strict"`, `"required"`, `"warn"` validation handling |
| [`Mode`, `Absent`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_schema_spec.py) | `"validate"`/`"project"` and `"fill"`/`"omit"` policy names |
| [`MemoryScaling`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_batch.py) | `"n_atoms"` or `"n_atoms_x_density"` batching weights |
| [`StorageOptions`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_remote.py) | Remote store configuration dictionaries |
| [`Writable`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_write.py) | A native `Frame` or ASE `Atoms` |

Do not catch every failure as a parse error: argument errors, out-of-range
indices and I/O failures also use standard Python exceptions. See
[errors](errors.md) for handling and iterator behaviour.

## Optional modules and stability

[`oxyz.ase`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/ase.py),
[`oxyz.metatomic`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/metatomic.py)
and [`oxyz.torch_sim`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/torch_sim.py)
provide target-specific reads/conversions. Follow the
[converter guide](../guides/converters.md) for checked structures and target
extraction; the source docstrings give full signatures. ASE `read` defaults to
the last frame, while native `oxyz.read` defaults to all frames. See
[compatibility](compatibility.md) before substituting representations.

| Module | Public operations |
| --- | --- |
| `oxyz.ase` | `read`, `iread`, `to_atoms`, `from_atoms`; `ToAseError`, `FromAtomsError` |
| `oxyz.metatomic` | `read`, `iread`, `SystemSource` with `systems`/`per_config`/`per_atom`; `ToSystemError` |
| `oxyz.torch_sim` | `read`, `iread`, `SimStateSource` with `state`/`per_config`/`per_atom`; `ToSimStateError` |

There are no metatomic or torch-sim writer entry points. Native `write` accepts
frames and ASE objects under the [writing contract](../guides/writing.md).

Root exports and documented submodules follow the repository's semantic-versioning
contract. Underscore-prefixed modules, including `oxyz._rust`, are implementation
details even where the links above point to their public objects' definitions.
See the [root export list](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/__init__.py)
for the inventory and [release policy](https://github.com/theochemtheo/oxyz/blob/main/RELEASING.md)
for versioning guarantees.
