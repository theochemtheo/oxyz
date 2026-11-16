# Schemas and projection

Status: Accepted.

## Contract

An inferred `Schema` describes observed fields, variants and presence across
frames. A `SchemaSpec` declares expected columns, metadata and frame rules.
Inference is not structural scanning: a scan finds frame boundaries/counts,
while inference parses values and folds their signatures.

Validation checks conformance without reshaping the original frame. Projection
selects declared fields and applies the supported fills/conversions to produce
a consistent requested representation. Python compiles the policy; Rust
executes projection and assembles projected batches. Freezing expands pattern
rules against a sample into literal rules rather than guessing a fixed shape
from an unbounded future dataset.

## Rationale and trade-offs

Keeping observations separate from policy avoids treating a sample's contents
as a universal dataset guarantee. Users can validate a file without losing
undeclared data, or explicitly project mixed input for a dense batch. A fixed
projection plan lets native assembly avoid repeated Python work per atom.

Optionality and fills need explicit semantics: an absent field is not the same
as a present zero. `absent="omit"` can preserve omission in frames; dense batch
representations have stronger shape requirements. Integer, boolean and string
fields do not have a natural NaN sentinel, so optional filling needs a declared
value where required. Pattern freezing can fail when observed kinds/widths
conflict; it does not silently repair incompatible data.

## Evidence

- [`_schema_spec.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_schema_spec.py),
  [`_schema_match.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_schema_match.py)
  and [`_project.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_project.py)
  own declaration, matching and compilation/freezing policy.
- [`schema.rs`](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-core/src/schema.rs)
  folds observations; [`project.rs`](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-core/src/project.rs)
  applies the compiled projection.
- [`test_schema_read.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_schema_read.py),
  [`test_project.py`](https://github.com/theochemtheo/oxyz/blob/main/tests/test_project.py)
  and [`projection.rs`](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-core/tests/projection.rs)
  guard reader conformance and projection behaviour.

See [typed records and unions](typed-records-and-unions.md) for the typed
consumer of this pipeline and [architecture](../architecture.md) for owners.
