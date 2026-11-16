---
name: oxyz-schema
description: Use when working on oxyz schema inference, SchemaSpec rules, conformance, projection, optional field omission, patterns, fills or freeze.
---

# oxyz schemas

Keep observed structure, validation and value projection distinct. Start with
the [root router](../oxyz/SKILL.md), [schema guide](../../../docs/guides/schemas-and-projection.md),
[concepts](../../../docs/concepts/schemas.md) and
[accepted decision](../../../docs/development/decisions/schemas-and-projection.md).

## Owners and checks

Python paths are under `src/oxyz/`; Python checks are under `tests/`.

| Concern | Owner | Python check |
| --- | --- | --- |
| Observations, variants, promotion summaries | `_schema.py`; Rust `schema.rs` | `test_schema.py` |
| Rules, patterns, precedence, serialisation | `_schema_spec.py` | `test_schema_spec.py` |
| Inference-to-spec emission | `_schema_emit.py` | `test_schema_emit.py` |
| Conformance and structured violations | `_schema_match.py` | `test_schema_match.py` |
| Compiled projection, fills, omission, freeze | `_project.py` | `test_project.py` |
| Reader validation/projection and selections | `_frames.py`, `_batch.py`, `_select.py` | `test_schema_read.py` |

Rust files under `crates/oxyz-core/src/`: `schema.rs` folds signatures,
`project.rs` applies value projection and reports presence, `batch.rs` assembles
uniform dense fields. Their integration targets are `schema_inference`,
`projection` and `batch_build`. PyO3 transfer belongs in
`crates/oxyz-py/src/lib.rs`; do not duplicate Python policy there.

## Invariants and common mistakes

- Inference records evidence, not policy. INT/REAL variants unify only at
  compatible widths/shapes; unification does not cast data. `is_consistent`
  requires one variant and presence in every frame.
- Validation checks signatures without transforming data. Projection discards
  extras and permits INT→REAL, not general coercion or reshaping. Do not fix a
  failing contract by weakening validation.
- Optionality permits absence; fills materialise storage. Optional REAL fields
  default to NaN; other kinds need explicit fills unless omitted.
  `absent="omit"` overrides fills for absent optional fields, not checks on
  present values or required rules. Mixed omission cannot form a plain `Batch`;
  filled plain batches have no presence mask.
- Literal rules beat patterns; otherwise first matching pattern wins.
  Cardinality and signature checks are separate. Freeze expands observed names
  into a project spec using unified sample signatures and sample presence;
  it does not validate pattern signatures/cardinality or guarantee unseen data.
- `FrameRule` checks atom bounds/lattice presence, not geometry. Projected lattice
  must survive field selection. Metadata shape validation/projection has the
  documented scalar/vector limitations; do not promise full matrix validation.
- Warning-mode validation retains original data; projection may fill or drop
  unfillable frames. Preserve original error frame/name context and each reader's
  selection-after-drop contract; typed records differ from plain projected reads.

## Targeted verification

For an omitted optional projection field, follow `_project.py` to Rust
`project.rs` and reader integration. Inspect these existing regressions:

```sh
uv run pytest -q tests/test_schema_read.py::test_omit_rule_leaves_the_key_out_of_frames_that_lack_it tests/test_schema_read.py::test_omit_rule_of_the_wrong_kind_raises_with_frame_and_name tests/test_schema_read.py::test_omit_rule_with_mixed_presence_cannot_batch
cargo test -p oxyz-core --test projection optional_without_fill_is_omitted_when_absent
```

Choose other table modules with `uv run pytest -q tests/<module>`; use
`test_project.py` for fill/omit plan compilation and freeze, `test_schema_match.py`
for validation. Read the [architecture map](../../../docs/development/architecture.md#schema-and-record-change-routing)
for shared ownership and [oxyz-records](../oxyz-records/SKILL.md) for typed
presence. Run only checks affected by the change, not every route for prose.
