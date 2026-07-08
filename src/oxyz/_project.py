"""Schema projection: compile a `SchemaSpec` into a fixed-shape plan, freeze
its patterns against a dataset, and translate the core's deviation report into
raise / warn / drop policy.

The Rust core reshapes each frame to the plan and reports what it saw; this
module owns everything policy-shaped that the core deliberately does not — the
effective-mode rule, the spec-error checks (patterns and un-fillable fields),
and the mapping from deviations onto `SchemaError` / `SchemaWarning`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from oxyz._schema_match import SchemaError, _is_pattern, _matcher
from oxyz._schema_spec import ColumnRule, MetadataRule, SchemaSpec

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from oxyz import _remote
    from oxyz._frames import Compression
    from oxyz._schema import ColumnSchema, MetadataSchema


def freeze_spec(
    spec: SchemaSpec,
    path: str | Path,
    *,
    compression: Compression = "infer",
    member: str | None = None,
    storage_options: _remote.StorageOptions | None = None,
) -> SchemaSpec:
    """Expand pattern rules against a representative dataset into a literal,
    project-ready spec (``mode='project'``).

    Literal rules pass through untouched; each pattern rule becomes one literal
    rule per matched inferred field — required when the field is present in
    every frame, optional otherwise, so projection fills the sometimes-absent
    ones. A matched field whose kind conflicts across frames cannot be frozen
    and raises `SchemaError`.
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
            out.append(
                ColumnRule(
                    name=name,
                    kind=kind,
                    width=width,
                    required=entry.frames_present == n_frames,
                    fill=rule.fill,
                )
            )
    return out


def _freeze_metadata(
    rules: Iterable[MetadataRule],
    inferred: Iterable[MetadataSchema],
    n_frames: int,
) -> list[MetadataRule]:
    rules = tuple(rules)
    entries = {entry.key: entry for entry in inferred}
    claimed = {rule.name for rule in rules if not _is_pattern(rule.name)}
    out = [rule for rule in rules if not _is_pattern(rule.name)]
    for rule in rules:
        if not _is_pattern(rule.name):
            continue
        matcher = _matcher(rule.name)
        for key, entry in entries.items():
            if key in claimed or not matcher.match(key):
                continue
            claimed.add(key)
            if entry.unified is None:
                raise SchemaError(
                    f"metadata {key!r} matched by pattern {rule.name!r} has "
                    f"conflicting kinds or shapes across frames and cannot be "
                    f"frozen; resolve it by hand"
                )
            kind, shape = entry.unified
            out.append(
                MetadataRule(
                    name=key,
                    kind=kind,
                    shape=shape,
                    required=entry.frames_present == n_frames,
                    fill=rule.fill,
                )
            )
    return out
