from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from oxyz._schema import Kind
from oxyz._schema_spec import (
    KIND_TO_LETTER,
    ColumnRule,
    MetadataRule,
    SchemaSpec,
)

if TYPE_CHECKING:
    from oxyz._frames import Frame

Conformance = Literal["strict", "required", "warn"]
Axis = Literal["column", "metadata", "frame"]
Deviation = Literal["missing", "extra", "mismatch", "count"]

_NUMPY_KIND: dict[str, Kind] = {
    "f": Kind.REAL,
    "i": Kind.INT,
    "u": Kind.INT,
    "b": Kind.BOOL,
}


@dataclass(frozen=True, slots=True)
class Violation:
    axis: Axis
    name: str
    deviation: Deviation
    expected: str | None
    found: str | None
    frame_index: int | None = None
    line: int | None = None


@dataclass(frozen=True, slots=True)
class CompiledSpec:
    columns_literal: dict[str, ColumnRule]
    columns_pattern: tuple[tuple[ColumnRule, re.Pattern[str]], ...]
    metadata_literal: dict[str, MetadataRule]
    metadata_pattern: tuple[tuple[MetadataRule, re.Pattern[str]], ...]
    frame: object  # FrameRule | None; kept loose to avoid an import cycle in typing


def _is_pattern(name: str) -> bool:
    return name.startswith("re:") or any(ch in name for ch in "*?[")


def _matcher(name: str) -> re.Pattern[str]:
    if name.startswith("re:"):
        return re.compile(name[3:])
    return re.compile(fnmatch.translate(name))


def _partition(rules):
    literal: dict = {}
    patterns: list = []
    for rule in rules:
        if _is_pattern(rule.name):
            patterns.append((rule, _matcher(rule.name)))
        else:
            literal[rule.name] = rule
    return literal, tuple(patterns)


def compile_spec(spec: SchemaSpec) -> CompiledSpec:
    columns_literal, columns_pattern = _partition(spec.columns)
    metadata_literal, metadata_pattern = _partition(spec.metadata)
    return CompiledSpec(
        columns_literal=columns_literal,
        columns_pattern=columns_pattern,
        metadata_literal=metadata_literal,
        metadata_pattern=metadata_pattern,
        frame=spec.frame,
    )


def column_signature(value: object) -> tuple[Kind, int]:
    """Derive `(kind, width)` from a built column value: a numpy array (1-D =
    width 1, 2-D = its second dim) or a list of strings / list of lists."""

    if isinstance(value, np.ndarray):
        kind = _NUMPY_KIND[value.dtype.kind]
        width = value.shape[1] if value.ndim == 2 else 1
        return kind, width
    # string column: list[str] (width 1) or list[list[str]] (width n)
    if isinstance(value, list) and value and isinstance(value[0], list):
        return Kind.STR, len(value[0])
    return Kind.STR, 1


def _column_sig_str(kind: Kind, width: int) -> str:
    return f"{KIND_TO_LETTER[kind]}:{width}"


def _cardinality(rule) -> tuple[int, int | None]:
    if rule.count is not None:
        return rule.count, rule.count
    lo = rule.min if rule.min is not None else (1 if rule.required else 0)
    return lo, rule.max


def _validate_columns(
    frame: Frame, compiled: CompiledSpec, level: Conformance
) -> list[Violation]:
    out: list[Violation] = []
    claimed: set[str] = set()
    present = frame.columns

    for name, rule in compiled.columns_literal.items():
        expected = _column_sig_str(rule.kind, rule.width)
        if name not in present:
            if rule.required:
                out.append(Violation("column", name, "missing", expected, None))
            continue
        claimed.add(name)
        kind, width = column_signature(present[name])
        if (kind, width) != (rule.kind, rule.width):
            out.append(
                Violation(
                    "column", name, "mismatch", expected, _column_sig_str(kind, width)
                )
            )

    for rule, matcher in compiled.columns_pattern:
        matches = [n for n in present if n not in claimed and matcher.match(n)]
        expected = _column_sig_str(rule.kind, rule.width)
        for name in matches:
            claimed.add(name)
            kind, width = column_signature(present[name])
            if (kind, width) != (rule.kind, rule.width):
                out.append(
                    Violation(
                        "column",
                        name,
                        "mismatch",
                        expected,
                        _column_sig_str(kind, width),
                    )
                )
        lo, hi = _cardinality(rule)
        if len(matches) < lo or (hi is not None and len(matches) > hi):
            out.append(
                Violation(
                    "column",
                    rule.name,
                    "count",
                    str(rule.count or lo),
                    str(len(matches)),
                )
            )

    if level in ("strict", "warn"):
        for name in present:
            if name not in claimed:
                kind, width = column_signature(present[name])
                out.append(
                    Violation(
                        "column", name, "extra", None, _column_sig_str(kind, width)
                    )
                )
    return out


def validate_frame(
    frame: Frame, compiled: CompiledSpec, level: Conformance
) -> list[Violation]:
    """Return every schema deviation in `frame`. Never raises. `extra` items are
    reported only under `strict`/`warn`; missing/mismatch/count are reported at
    all levels. Metadata and frame checks are added by later helpers."""

    return _validate_columns(frame, compiled, level)
