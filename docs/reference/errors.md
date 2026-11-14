# Errors and failure boundaries

Catch structured exception attributes rather than extracting coordinates from
messages. Frame indices are zero-based; parser lines are file-absolute and
one-based, and columns are one-based character positions, not byte offsets.
An unknown coordinate is `None`, not zero.

## Exception classes

| Class | Meaning and structured context |
| --- | --- |
| `oxyz.OxyzError` | Package error base, itself a `ValueError` subclass |
| `oxyz.ParseError` | Invalid extxyz content; `frame_index`, `line`, `column` |
| `oxyz.SchemaError` | Validation, projection or record-binding failure; `frame_index` and `name` identify the frame and key or frame-rule field where known |
| `oxyz.SchemaWarning` | `UserWarning` for a schema deviation under `conformance="warn"`; not an exception subclass of `OxyzError` |
| `IndexError` | Requested frame index out of range; also an integer record selection naming a dropped frame |
| `OSError` | I/O or source failure, including a missing file |
| `ValueError`, `TypeError` | Invalid API arguments or incompatible declarations, as documented on the called function |
| `ImportError` | An optional dependency needed by the requested operation is unavailable |

`except oxyz.OxyzError` catches parser, schema and converter errors, but is not
a universal handler for I/O, indexing or Python argument errors. A parse error
is also caught by `except ValueError`. Do not convert every failure into a
parse error: an out-of-range request and an unreadable source are different
problems.

The [binding exception docstrings](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_rust.pyi)
and [schema exception docstrings](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_schema_match.py)
define the attributes. The [API reference](api.md) routes to the public readers
and schema objects.

## Locate a bad token

A short atom row has no single offending token, so its column is `None`.
A malformed numeric token can have an exact column. This example puts the
error in a second frame to distinguish file line numbers from frame-local ones.

```python title="Locate malformed input using structured attributes"
from pathlib import Path

import oxyz

bad_value = Path("error-token.extxyz")
bad_value.write_text(
    "1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n"
    "1\nProperties=species:S:1:pos:R:3\nH abc 0 0\n"
)
try:
    oxyz.read(bad_value)
except oxyz.ParseError as error:
    assert isinstance(error, oxyz.OxyzError)
    assert isinstance(error, ValueError)
    assert (error.frame_index, error.line, error.column) == (1, 6, 3)
else:
    raise AssertionError("invalid coordinate must raise ParseError")

short_row = Path("error-short-row.extxyz")
short_row.write_text("1\nProperties=species:S:1:pos:R:3\nH 0 0\n")
try:
    oxyz.read(short_row)
except oxyz.ParseError as error:
    assert (error.frame_index, error.line, error.column) == (0, 3, None)
else:
    raise AssertionError("short row must raise ParseError")

unlocated = oxyz.ParseError("caller-created error")
assert (unlocated.frame_index, unlocated.line, unlocated.column) == (None, None, None)
```

Indexed parsing preserves source frame and line context; serial and parallel
parsing report the same error location. Archive `member="*"` reads number
frames across the combined stream, not separately within each entry. An entry
ending inside a frame raises `OSError` naming that entry; do not assume all
archive failures carry `ParseError` attributes.

## Parser errors stop iteration

!!! warning "A parse error ends the stream"

    A streaming reader yields preceding valid frames, raises its parse error
    once, then terminates. It does not attempt to find a plausible next count
    line and resume. This is sometimes called a fused iterator: subsequent
    requests remain exhausted.

```python title="Report a parse error once, then remain exhausted"
import oxyz

stream = oxyz.iread(bad_value)
assert next(stream).columns["pos"].tolist() == [[0.0, 0.0, 0.0]]
try:
    next(stream)
except oxyz.ParseError as error:
    assert (error.frame_index, error.line, error.column) == (1, 6, 3)
else:
    raise AssertionError("iterator must report its malformed second frame")
sentinel = object()
assert next(stream, sentinel) is sentinel
assert next(stream, sentinel) is sentinel
```

This parser guarantee is not a promise of recovery after arbitrary converter,
user callback or argument failures. Nor does a successful partial read prove
that the unconsumed suffix is valid. See
[reading and selecting](../guides/reading-and-selecting.md) and
[storage and streaming](../concepts/storage-and-streaming.md) for which paths
parse a prefix, scan structure or materialise the whole file.

## Schema failures and warnings

`SchemaError.name` identifies a column, metadata key or frame-rule field;
`frame_index` can be `None` for declaration/binding errors before reading or
for conversion of a standalone frame. Schema failures do not have parser
line/column attributes. A syntactically valid file can still violate a schema.

```python title="Identify a missing required schema field"
from pathlib import Path

import oxyz

missing_energy = Path("error-schema.extxyz")
missing_energy.write_text("1\nProperties=species:S:1:pos:R:3\nO 0 0 0\n")
spec = oxyz.SchemaSpec(metadata=(oxyz.MetadataRule("energy", oxyz.Kind.REAL),))
try:
    oxyz.read(missing_energy, schema=spec)
except oxyz.SchemaError as error:
    assert (error.frame_index, error.name) == (0, "energy")
    assert isinstance(error, oxyz.OxyzError)
else:
    raise AssertionError("required metadata must be checked")
```

`conformance="required"` checks required fields while allowing extras;
`"strict"` also rejects extras. `"warn"` reports deviations as
`SchemaWarning`. It does not universally mean that a frame is retained:
unfillable projected frames and non-conforming record frames can be dropped.
The public [`read` docstring](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_frames.py)
describes these distinctions and the integer-selection error for a dropped
record. Native eager reads with `schema=` validate the whole file before
selection, while forward `iread` checks frames as they are consumed.

## Converter failures

Optional output targets keep their own error types under `OxyzError`. Their
docstrings define a strict, faithful representation boundary rather than
silent repairs:

- [`oxyz.ase.ToAseError` and `FromAtomsError`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/ase.py):
  a frame cannot be represented faithfully as `ase.Atoms`, or the reverse.
- [`oxyz.metatomic.ToSystemError`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/metatomic.py):
  a frame cannot be represented faithfully as a `System`.
- [`oxyz.torch_sim.ToSimStateError`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/torch_sim.py):
  frames cannot be represented faithfully as a `SimState`.

These types do not promise parser coordinates. Conversion errors concern the
target representation, not necessarily malformed extxyz syntax. Consult
[compatibility](compatibility.md) for supported interpretation boundaries.

## Contract sources

??? note "Show implementation and test references"

    [Python error tests](https://github.com/theochemtheo/oxyz/blob/main/tests/test_errors.py)
    cover the hierarchy and token/short-row locations;
    [reader tests](https://github.com/theochemtheo/oxyz/blob/main/tests/test_extxyz.py)
    cover file-absolute context and fused iteration;
    [Rust error-context tests](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-core/tests/error_context.rs)
    cover token, metadata, scan and serial/parallel locations.
