"""Schema projection: compile, freeze, and police a `SchemaSpec`.

Compiles a `SchemaSpec` into a fixed-shape plan, freezes its patterns against
a dataset, and translates the core's deviation report into raise / warn /
drop policy.

The Rust core reshapes each frame to the plan and reports what it saw; this
module owns everything policy-shaped that the core deliberately does not — the
effective-mode rule, the spec-error checks (patterns and un-fillable fields),
and the mapping from deviations onto `SchemaError` / `SchemaWarning`.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, cast

from oxyz._schema import Kind
from oxyz._schema_match import (
    SchemaError,
    SchemaWarning,
    Violation,
    _is_pattern,
    _matcher,
    message,
)
from oxyz._schema_spec import KIND_TO_LETTER, ColumnRule, MetadataRule, SchemaSpec

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from oxyz import _remote
    from oxyz._frames import Compression
    from oxyz._rust import DeviationData, ProjectionPlan
    from oxyz._schema import ColumnSchema, MetadataSchema
    from oxyz._schema_match import Axis, Conformance, Deviation
    from oxyz._schema_spec import Mode

# Kinds with no in-band null: an optional field of one needs an explicit fill.
_NO_NATURAL_NULL = (Kind.INT, Kind.BOOL, Kind.STR)


def freeze_spec(
    spec: SchemaSpec,
    path: str | Path,
    *,
    compression: Compression = "infer",
    member: str | None = None,
    storage_options: _remote.StorageOptions | None = None,
) -> SchemaSpec:
    """Expand pattern rules into a literal, project-ready spec (``mode='project'``).

    Rules are expanded against a representative dataset. Literal rules pass
    through untouched; each pattern rule becomes one literal rule per matched
    inferred field — required when the field is present in every frame,
    optional otherwise, so projection fills the sometimes-absent ones. A
    matched field whose kind conflicts across frames cannot be frozen and
    raises `SchemaError`.
    """
    from oxyz import infer_schema

    schema = infer_schema(
        path, compression=compression, member=member, storage_options=storage_options
    )
    columns = _freeze_columns(spec.columns, schema.columns, schema.n_frames)
    metadata = _freeze_metadata(spec.metadata, schema.metadata, schema.n_frames)
    return SchemaSpec(
        columns=tuple(columns),
        metadata=tuple(metadata),
        frame=spec.frame,
        mode="project",
    )


def _freeze_columns(
    rules: Iterable[ColumnRule],
    inferred: Iterable[ColumnSchema],
    n_frames: int,
) -> list[ColumnRule]:
    rules = tuple(rules)
    entries = {entry.name: entry for entry in inferred}
    claimed = {rule.name for rule in rules if not _is_pattern(rule.name)}
    out = [rule for rule in rules if not _is_pattern(rule.name)]
    for rule in rules:
        if not _is_pattern(rule.name):
            continue
        matcher = _matcher(rule.name)
        for name, entry in entries.items():
            if name in claimed or not matcher.match(name):
                continue
            claimed.add(name)
            if entry.unified is None:
                raise SchemaError(
                    f"column {name!r} matched by pattern {rule.name!r} has "
                    f"conflicting kinds or widths across frames and cannot be "
                    f"frozen; resolve it by hand"
                )
            kind, width = entry.unified
            required = entry.frames_present == n_frames
            _check_freezable_optional(
                name, rule.name, kind, required, rule.fill, "column"
            )
            out.append(
                ColumnRule(
                    name=name, kind=kind, width=width, required=required, fill=rule.fill
                )
            )
    return out


def _check_freezable_optional(
    name: str, pattern: str, kind: Kind, required: bool, fill: object, axis: str
) -> None:
    """Refuse to freeze an optional, un-fillable field.

    A pattern that matches a field present in only *some* frames freezes it
    to an optional rule; if that field is INT/BOOL/STR with no fill,
    projection could never materialise it, so refuse here rather than emit a
    spec that raises at read time.
    """
    if not required and kind in _NO_NATURAL_NULL and fill is None:
        raise SchemaError(
            f"{axis} {name!r} (matched by pattern {pattern!r}) is present in only "
            f"some frames and is a {KIND_TO_LETTER[kind]} field with no natural "
            f"null; give the pattern rule a 'fill' so the frozen schema can "
            f"project it"
        )


def _freeze_metadata(
    rules: Iterable[MetadataRule],
    inferred: Iterable[MetadataSchema],
    n_frames: int,
) -> list[MetadataRule]:
    rules = tuple(rules)
    entries = {entry.key: entry for entry in inferred}
    claimed = {rule.key for rule in rules if not _is_pattern(rule.key)}
    out = [rule for rule in rules if not _is_pattern(rule.key)]
    for rule in rules:
        if not _is_pattern(rule.key):
            continue
        matcher = _matcher(rule.key)
        for key, entry in entries.items():
            if key in claimed or not matcher.match(key):
                continue
            claimed.add(key)
            if entry.unified is None:
                raise SchemaError(
                    f"metadata {key!r} matched by pattern {rule.key!r} has "
                    f"conflicting kinds or shapes across frames and cannot be "
                    f"frozen; resolve it by hand"
                )
            kind, shape = entry.unified
            required = entry.frames_present == n_frames
            _check_freezable_optional(
                key, rule.key, kind, required, rule.fill, "metadata"
            )
            out.append(
                MetadataRule(
                    key=key, kind=kind, shape=shape, required=required, fill=rule.fill
                )
            )
    return out


def effective_mode(spec: SchemaSpec, override: Mode | None) -> Mode:
    """Return the mode a read runs under.

    An explicit `override` wins, else the spec's own `mode`.
    """
    return override if override is not None else spec.mode


def _fill_for(rule: ColumnRule | MetadataRule) -> float | int | bool | str | None:
    if rule.fill is not None:
        return rule.fill
    # REAL has NaN as a natural null; other kinds have none, so leave it unset
    # (the core drops an absent one rather than fabricate a value).
    return math.nan if rule.kind == Kind.REAL else None


def _fill_matches_kind(fill: object, kind: Kind) -> bool:
    # bool before int: Python's bool is an int subclass.
    if kind == Kind.BOOL:
        return isinstance(fill, bool)
    if kind == Kind.INT:
        return isinstance(fill, int) and not isinstance(fill, bool)
    if kind == Kind.REAL:
        return isinstance(fill, (int, float)) and not isinstance(fill, bool)
    return isinstance(fill, str)


def _plan_entry(rule: ColumnRule | MetadataRule, *, is_metadata: bool) -> tuple:
    letter = KIND_TO_LETTER[rule.kind]
    axis = "metadata" if is_metadata else "column"
    name = rule.key if isinstance(rule, MetadataRule) else rule.name
    fill = _fill_for(rule)
    if fill is None and not rule.required and rule.kind in _NO_NATURAL_NULL:
        raise SchemaError(
            f"{axis} {name!r} is an optional {letter} field with no natural "
            f"null; give it a 'fill' value so projection can materialise it"
        )
    if rule.fill is not None and not _fill_matches_kind(rule.fill, rule.kind):
        # Catch a mismatched fill here (e.g. a string fill on an INT column, or
        # one pattern's fill copied by freeze onto a differing-kind column) with
        # a clear message rather than a TypeError from the binding at read time.
        raise SchemaError(
            f"{axis} {name!r}: fill {rule.fill!r} is not a valid {letter} value"
        )
    if is_metadata:
        assert isinstance(rule, MetadataRule)  # noqa: S101 — type-narrowing only
        return (name, letter, tuple(rule.shape), rule.required, fill)
    assert isinstance(rule, ColumnRule)  # noqa: S101 — type-narrowing only
    return (name, letter, rule.width, rule.required, fill)


def compile_projection(spec: SchemaSpec, mode: Mode) -> ProjectionPlan | None:
    """Compile `spec` under `mode` into a crossing plan, or `None` for validate mode.

    Returns `(columns, metadata)`; `None` means nothing to project. Raises
    `SchemaError` before any read for a pattern rule (project needs a
    fixed shape — point at `freeze`) or an optional INT/BOOL/STR field with no
    fill (no null to materialise it with).
    """
    if mode == "validate":
        return None
    patterned = [r.name for r in spec.columns if _is_pattern(r.name)]
    patterned += [r.key for r in spec.metadata if _is_pattern(r.key)]
    if patterned:
        raise SchemaError(
            f"schema rule {patterned[0]!r} is a pattern; project mode needs a "
            f"fixed shape. Run SchemaSpec.freeze(sample) to expand it first"
        )
    columns = [_plan_entry(rule, is_metadata=False) for rule in spec.columns]
    metadata = [_plan_entry(rule, is_metadata=True) for rule in spec.metadata]
    return (columns, metadata)


def _to_violation(deviation: DeviationData) -> Violation:
    # The core only emits the "missing"/"mismatch" subset on the column/metadata
    # axes; the dicts type those fields as plain str, so narrow to the literals.
    return Violation(
        axis=cast("Axis", deviation["axis"]),
        name=deviation["name"],
        deviation=cast("Deviation", deviation["deviation"]),
        expected=deviation["expected"],
        found=deviation["found"],
    )


def enforce_projection(
    deviations: list[DeviationData],
    level: Conformance,
    frame_index: int,
    dropped: bool,
) -> bool:
    """Apply conformance policy to one projected frame's report.

    Returns whether to keep the frame. Under `strict`/`required`, raise
    `SchemaError` on the first deviation. Under
    `warn`, emit a `SchemaWarning` per deviation and keep the frame unless it was
    dropped (an unfillable required/wrong field). A clean, kept frame is silent.
    """
    if not deviations and not dropped:
        return True
    if level in ("strict", "required"):
        if deviations:
            first = _to_violation(deviations[0])
            raise SchemaError(
                message(first, frame_index), frame_index=frame_index, name=first.name
            )
        if dropped:
            # A drop always carries a deviation in practice; raise rather than
            # silently lose the frame under strict/required if the core ever
            # dropped one without reporting why.
            raise SchemaError(
                f"frame {frame_index}: dropped by projection", frame_index=frame_index
            )
        return True
    for deviation in deviations:
        warnings.warn(
            message(_to_violation(deviation), frame_index), SchemaWarning, stacklevel=3
        )
    return not dropped
