# Typed records and unions

Status: Accepted.

## Contract

A `Record` subclass declares typed fields with `Column`, `Meta` and `Cell`.
Binding translates those declarations into the existing reader/projection
pipeline; records are not a separate parser. A record may rename a file key
without changing the lossless native representation.

An optional field on an individual record can be `None` when absent. A dense
`RecordBatch` stores filled arrays and exposes per-frame presence through
`present(...)`. A fill value is data storage, not proof that the source carried
the field. Optional REAL batch fields can use NaN; kinds without an in-band null
need an explicit fill under the binding contract.

A union of record classes selects members by their distinguishing file keys,
not by positional fallback. `RecordBatch[A | B]` retains dense merged fields,
`members`, per-frame `tags` and `where()` for member selection. Union analysis
rejects declarations whose matching rules are ambiguous rather than silently
choosing whichever class was listed first.

## Rationale and trade-offs

Typed records give callers named, statically checked fields while sharing the
same source/selection and schema machinery as native frames. Dense batches suit
array consumers; presence masks and tags retain information that numerical
fills alone cannot represent. Separating declaration, field analysis, binding,
union planning and batch storage keeps the contracts independently testable.

This costs more declaration/binding policy than a dictionary lookup. Callers
must choose fills deliberately and use presence or tags rather than inferring
missingness from zero/NaN values. A union must be distinguishable from the
supported input keys; it cannot resolve arbitrary semantic ambiguity.

## Evidence

- [`_record.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_record.py)
  and [`_record_fields.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_record_fields.py)
  own declarations and annotation analysis.
- [`_record_bind.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_record_bind.py),
  [`_record_union.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_record_union.py)
  and [`_record_batch.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_record_batch.py)
  own binding, matching and dense storage.
- [`test_record_typing.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_record_typing.py),
  [`test_record_union.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_record_union.py)
  and [`test_record_batch.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_record_batch.py)
  guard static use, union selection and presence behaviour.

See [schemas and projection](schemas-and-projection.md) and the
[architecture map](../architecture.md) before changing the shared pipeline.
