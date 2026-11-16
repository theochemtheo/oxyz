---
name: oxyz-records
description: Use when working on oxyz Record declarations, binding, sibling unions, RecordBatch views, presence masks, fills, selections or record typing.
---

# oxyz records

Records share the native reader/projection path; dense storage is not presence.
Start with the [root router](../oxyz/SKILL.md),
[typed records](../../../docs/guides/typed-records.md),
[mixed datasets](../../../docs/guides/mixed-datasets.md) and
[accepted decision](../../../docs/development/decisions/typed-records-and-unions.md).

## Owners and checks

Owners are under `src/oxyz/`; checks are under `tests/`.

| Concern | Owner | Check |
| --- | --- | --- |
| Declarations, construction, frame conversion | `_record.py` | `test_record.py` |
| Field analysis, runtime types/shapes/casts | `_record_fields.py` | `test_record_fields.py` |
| Binding, projection/presence plans, owned/view conversion | `_record_bind.py` | `test_record.py`, `test_record_batch.py` |
| Sibling compatibility, shared keys, signature ambiguity | `_record_union.py` | `test_record_union.py` |
| Dense storage, rows, masks, tags, slices/gathers | `_record_batch.py` | `test_record_batch.py`, `test_record_union.py` |
| Reader selection and warning/drop integration | `_frames.py`, `_batch.py`, `_select.py` | `test_record.py`, `test_record_union.py` |
| Consumer-visible static types | Inline annotations and reader overloads | `test_record_typing.py` |

## Invariants and common mistakes

- Unions admit siblings, never ancestor/descendant members. Shared file keys must
  agree in field name, type, optionality, width/shape and fill; shared field names
  cannot name different keys. Match key presence, not values or member order.
  Required-by-all and non-nullable-filled-by-all keys do not distinguish members.
  Reject ambiguous admitted signatures; see the mixed-datasets matching rules.
- Nullable row fields restore `None` when absent. Dense batch values may be NaN,
  zero or another fill: use `present(...)`, not value comparisons. Non-REAL
  nullable fields and non-REAL fields absent from some members need batch fills.
- Single-record masks track nullable fields only. Union masks track matching
  keys, including fields not shared by all members; non-nullable fields filled
  by all members do not track absence. Presence is per frame, not per atom or
  universal source provenance. Preserve `members`, `tags` and `where()` semantics.
- Integer batch indexing returns rows; unit-step slices are views, other
  selections gather copies. Masks, tags and source `frame_indices` travel with
  reordering/repetition. File selections count source frames even after warning
  drops; selecting a dropped frame by integer raises `IndexError`.
- With `record=`, reject `mode=`. `schema=` constrains compatible binding and
  frame rules, not a replacement projection. Frame-rule warnings retain rows;
  missing required/wrong-kind fields can drop them. Plain `Batch` cannot recover
  presence lost to fills; union `from_batch` cannot reconstruct missing tags.
- Runtime and static contracts need separate checks. Generic batches use typed
  keys; typed attributes require declared batch fields. Preserve documented
  [typing limits](../../../docs/guides/typed-records.md#typing-limits), including
  runtime-only NumPy shape validation and union compatibility checks.

## Targeted verification

For an absent union-batch field, inspect `_record_bind.py` presence planning and
`_record_batch.py` storage/row reconstruction, then:

```sh
uv run pytest -q tests/test_record_union.py::test_fields_some_members_lack_are_nan_exactly_there tests/test_record_union.py::test_batch_rows_equal_the_frame_read tests/test_record_union.py::test_slices_and_gathers_carry_tags
```

Use `test_record_batch.py` for general masks/views; `test_record_union.py` also
covers shared-key lookup, ambiguity, non-REAL fills and warning parity. Declaration
or API changes need `uv run pytest -q tests/test_record_typing.py` as well as runtime
checks. Update `src/oxyz/__init__.py` exports/`__all__` and inline types when relevant;
binding-visible changes also update `src/oxyz/_rust.pyi` and run stubtest via
[run-oxyz](../run-oxyz/SKILL.md). See [oxyz-schema](../oxyz-schema/SKILL.md)
for shared projection rules; do not relax them to make record tests pass.
