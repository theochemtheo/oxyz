from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import TYPE_CHECKING

from oxyz._schema_spec import ColumnRule, MetadataRule, SchemaSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from oxyz._schema import ColumnSchema, ColumnVariant, Kind, MetadataVariant, Schema

_ENUMERATED = re.compile(r"^(.*?)(\d+)$")


def _dominant[V: (ColumnVariant, MetadataVariant)](variants: Sequence[V]) -> V:
    # the variant used in the most frames; first-seen wins ties
    return max(variants, key=lambda v: v.frames)


def _column_kind_width(entry: ColumnSchema) -> tuple[Kind, int, str | None]:
    if entry.unified is not None:
        return entry.unified[0], entry.unified[1], None
    top = _dominant(entry.variants)
    note = (
        "drift: "
        + ", ".join(f"{v.kind.value}:{v.width} in {v.frames}" for v in entry.variants)
        + f" — using {top.kind.value}:{top.width}"
    )
    return top.kind, top.width, note


def _collapse_columns(
    schema: Schema, glob_min_run: int
) -> tuple[list[ColumnRule], dict[str, str]]:
    # Group required, full-presence columns whose names share a stem and a
    # trailing integer, same (kind, width); collapse a run of >= glob_min_run.
    families: dict[tuple[str, Kind, int], list[str]] = defaultdict(list)
    resolved: dict[str, tuple[Kind, int, bool, str | None]] = {}
    for entry in schema.columns:
        kind, width, note = _column_kind_width(entry)
        required = entry.frames_present == schema.n_frames
        resolved[entry.name] = (kind, width, required, note)
        match = _ENUMERATED.match(entry.name)
        if match and required and note is None:
            families[(match.group(1), kind, width)].append(entry.name)

    stem_family_count = Counter(stem for (stem, _kind, _width) in families)

    globbed: set[str] = set()
    rules: list[ColumnRule] = []
    notes: dict[str, str] = {}
    for entry in schema.columns:
        if entry.name in globbed:
            continue
        kind, width, required, note = resolved[entry.name]
        match = _ENUMERATED.match(entry.name)
        if match is not None:
            stem = match.group(1)
            members = families.get((stem, kind, width), [])
            if (
                entry.name in members
                and len(members) >= glob_min_run
                and stem_family_count[stem] == 1
            ):
                globbed.update(members)
                rules.append(
                    ColumnRule(f"{stem}*", kind, width=width, count=len(members))
                )
                continue
        rules.append(ColumnRule(entry.name, kind, width=width, required=required))
        if note is not None:
            notes[entry.name] = note
    return rules, notes


def _metadata_rules(schema: Schema) -> tuple[list[MetadataRule], dict[str, str]]:
    rules: list[MetadataRule] = []
    notes: dict[str, str] = {}
    for entry in schema.metadata:
        if entry.unified is not None:
            kind, shape = entry.unified
        else:
            top = _dominant(entry.variants)
            kind, shape = top.kind, top.shape
            notes[entry.key] = "drift: " + ", ".join(
                f"{v.kind.value}{list(v.shape) or ''} in {v.frames}"
                for v in entry.variants
            )
        required = entry.frames_present == schema.n_frames
        rules.append(MetadataRule(entry.key, kind, shape=shape, required=required))
    return rules, notes


def spec_and_notes(
    schema: Schema, *, glob_min_run: int = 3
) -> tuple[SchemaSpec, dict[str, str]]:
    """Turn an observed `Schema` into a prescriptive `SchemaSpec` plus a
    name -> drift-note map for `render_yaml`. Partial-presence entries are
    `required=False`; enumerable column families collapse to a `*` glob with a
    `count`; the `frame` section is never emitted (bounds are never auto-pinned)."""

    columns, column_notes = _collapse_columns(schema, glob_min_run)
    metadata, metadata_notes = _metadata_rules(schema)
    return SchemaSpec(columns=tuple(columns), metadata=tuple(metadata)), {
        **column_notes,
        **metadata_notes,
    }


def spec_from_schema(schema: Schema, *, glob_min_run: int = 3) -> SchemaSpec:
    return spec_and_notes(schema, glob_min_run=glob_min_run)[0]
