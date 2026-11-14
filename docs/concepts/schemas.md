# Schemas, optionality and presence

A schema can describe what a dataset contains or prescribe what a consumer
expects. These are different jobs. Follow the executable
[schema and projection guide](../guides/schemas-and-projection.md) for inference,
saved specs, conformance, fills and pattern expansion.

## Observed structure is evidence, not policy

`infer_schema(path)` parses the whole input into a `Schema`. For each column it
records observed kind/width variants; for metadata it records kind/shape
variants. Each variant has an occurrence count, and each field has
`frames_present`. Atom-count statistics describe the same input.

`unified` answers whether those observed variants have a common supported
signature: equal-width or equal-shape integer/real pairs can unify to REAL.
It does not cast the input or fill a missing field. A conflict gives `None`.
`is_consistent` is stricter: every field must occur in every frame with just
one variant. Data that unifies through promotion can still be inconsistent.

`Schema.to_spec()` makes a best-effort declaration. It makes partially present
fields optional, may collapse compatible numbered columns to counted patterns,
and does not infer structural requirements. For conflicting variants it selects
the most frequent signature, with first-seen tie-breaking. Review it rather
than assuming it proves conformance of the dataset from which it came.

## Declared structure is a consumer contract

A `SchemaSpec` contains `ColumnRule`, `MetadataRule` and an optional `FrameRule`.
Column width is components per atom; metadata shape is independent of atom
count, with `()` meaning scalar. Kinds use extxyz's R, I, L and S categories,
not arbitrary NumPy dtypes.

!!! warning "Schema shape checks are limited to scalars and vectors"

    Metadata projection accepts only scalars and one-dimensional arrays.
    Validation reports NumPy metadata shape using only its leading dimension,
    so it does not reliably validate higher-rank shapes either. Do not rely
    on a matrix-shaped `MetadataRule` to check matrix dimensions.

A rule can name a literal field, a glob or a `re:` pattern. Literal rules win
over patterns; otherwise the first matching pattern governs. Pattern
cardinality is separate from the signature of each matching field.
`spec[name]` looks up an exactly declared name; `spec.rule_for(name, axis=...)`
finds the governing rule. A name on both axes can be ambiguous, so specify
`axis="column"` or `axis="metadata"` when needed.

`FrameRule` expresses atom-count bounds and lattice presence, not chemical or
geometrical validity. Requiring `Lattice` does not itself constrain its shape.
When projecting, that field must also survive the field declaration.

## Validation and projection have different effects

| Operation | Present values | Undeclared fields | Absent optional fields |
| --- | --- | --- | --- |
| Validate | Check exact signatures; do not cast | Allowed under `required`, errors under `strict`, warnings under `warn` | Leave absent |
| Project | Keep compatible values; allow INT to REAL promotion | Remove | Fill, or leave absent with `absent="omit"` |

Neither operation renames aliases or performs general coercion. A declared
REAL field does not make a string numeric; a width mismatch does not reshape
it. `mode=` on a read overrides the spec's mode. Under projection, strict and
required conformance coincide because extras are discarded.

With warning conformance, validation keeps the original data. Projection can
replace missing or mismatched data with fills and drop unfillable frames.
Those are lossy operations, not merely diagnostic messages. Plain projected
frame reads count surviving frames for selection after drops; typed record
reads have a distinct source-frame selection contract. Do not infer one
reader's behaviour from the other.

## Optional does not mean filled, and filled does not mean present

Three questions must remain separate:

1. **Required?** May the source omit this field without a schema deviation?
2. **Materialised?** Does the returned object contain storage for it?
3. **Present?** Did the source actually carry it?

In validation, `required=False` answers only the first question. In projection,
an optional REAL field normally gets NaN when absent; INT, BOOL and STR fields
need explicit fills because they have no natural in-band null. A scalar fill
broadcasts over the declared width or shape. An explicit zero, false or empty
string is a value, not evidence that the field was absent.

`absent="omit"` leaves an absent optional field out of a projected frame. It
requires no fill and takes precedence over a fill, but does not weaken checks
on present values or change required rules. A plain `Frame` expresses this as
a missing dictionary key; it has no independent presence mask.

A plain `Batch` concatenates frames under a uniform field contract. Optional
validation rules or omitted projection fields do not permit mixed field
presence in one batch. Compatible INT/REAL promotion is supported, but
incompatible signatures still fail assembly. A filled plain batch carries no
mask that can recover original absence.

A `Record` declares named Python fields. An optional field can be `None` on an
individual record, whereas `RecordBatch` materialises storage and tracks
per-frame presence separately. Optional non-REAL batch fields need fills;
union fields absent from some members need the same treatment. This is not a
relaxation of plain `Batch` rules. See [typed records](../guides/typed-records.md)
and [mixed datasets](../guides/mixed-datasets.md) for executable declarations,
selection and absence-versus-zero checks. With `record=`, `schema=` constrains
binding and frame structure rather than replacing the record's projection;
`mode=` cannot be combined with `record=`.

## Freezing is sample-based expansion

Project mode needs literal field names. `spec.freeze(sample)` parses a sample,
expands matching patterns, and returns a new project-mode spec. Literal rules
are unchanged. Expanded fields use observed unified signatures and become
required exactly when present in every sample frame. The pattern's fill and
absence policy are retained; pattern cardinality and declared signatures are
not validated by this expansion.

Freeze rejects conflicting observed signatures and optional non-REAL fields
without a fill unless omission is requested. A field not observed in the
sample cannot appear in the expansion. Freezing therefore establishes a fixed
consumer shape for review, not a guarantee about unseen data or a replacement
for subsequent conformance checks.

## Failure context

Schema failures use `SchemaError.frame_index` and `.name` when a frame and
field are known. Errors in a declaration before reading may have no frame
index. `SchemaWarning` messages name deviations but do not expose a structured
violation attribute. A `Violation` can represent axis, name, deviation and
expected/found signatures; it is not an attribute promised on `SchemaError`.
Batch assembly can instead raise `ParseError`, including a source frame index
for incompatible field presence. See [error reference](../reference/errors.md)
for the distinction from parser location context.
