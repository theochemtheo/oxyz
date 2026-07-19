from __future__ import annotations

import fnmatch
import re
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from oxyz._rust import OxyzError
from oxyz._schema import Kind
from oxyz._schema_spec import (
    KIND_TO_LETTER,
    ColumnRule,
    FrameRule,
    MetadataRule,
    SchemaSpec,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from oxyz._frames import Frame

# How a schema deviation is handled. "required": missing/mismatch/count
# violations raise, extra fields are ignored. "strict": as required, but extra
# fields raise too. "warn": every violation (extra included) is emitted as a
# SchemaWarning instead of raising.
Conformance = Literal["strict", "required", "warn"]

# Which part of a frame a Violation concerns.
Axis = Literal["column", "metadata", "frame"]

# How a found value diverges from its rule: absent ("missing"), present but
# undeclared ("extra"), wrong kind/width/shape ("mismatch"), or a pattern rule
# matched too few/many fields ("count").
Deviation = Literal["missing", "extra", "mismatch", "count"]

_NUMPY_KIND: dict[str, Kind] = {
    "f": Kind.REAL,
    "i": Kind.INT,
    "u": Kind.INT,
    "b": Kind.BOOL,
}


@dataclass(frozen=True, slots=True)
class Violation:
    """One schema deviation found in a frame.

    Attributes
    ----------
    axis
        Which part of the frame the deviation concerns.
    name
        The column name, metadata key, or frame-rule field.
    deviation
        How `found` diverges from what the rule expected.
    expected
        The expected signature, value, or bound as text, or `None` for
        `extra`.
    found
        The observed signature, value, or count as text, or `None` for
        `missing`.
    frame_index
        Index of the frame the violation was found in. Left `None` by every
        producer in this module.
    line
        Source line of the frame. Left `None` by every producer in this
        module.
    """

    axis: Axis
    name: str
    deviation: Deviation
    expected: str | None
    found: str | None
    frame_index: int | None = None
    line: int | None = None


@dataclass(frozen=True, slots=True)
class CompiledSpec:
    """A `SchemaSpec` partitioned for matching.

    Literal rules are keyed by name; pattern rules are paired with their
    compiled matcher.

    Attributes
    ----------
    columns_literal
        Literal-name column rules, keyed by `ColumnRule.name`.
    columns_pattern
        Glob/regex column rules, each paired with its compiled matcher.
    metadata_literal
        Literal-key metadata rules, keyed by `MetadataRule.key`.
    metadata_pattern
        Glob/regex metadata rules, each paired with its compiled matcher.
    frame
        The spec's frame rule, or `None`.
    """

    columns_literal: dict[str, ColumnRule]
    columns_pattern: tuple[tuple[ColumnRule, re.Pattern[str]], ...]
    metadata_literal: dict[str, MetadataRule]
    metadata_pattern: tuple[tuple[MetadataRule, re.Pattern[str]], ...]
    frame: FrameRule | None


def _is_pattern(name: str) -> bool:
    return name.startswith("re:") or any(ch in name for ch in "*?[")


def _matcher(name: str) -> re.Pattern[str]:
    if name.startswith("re:"):
        return re.compile(name[3:])
    return re.compile(fnmatch.translate(name))


def _partition[Rule: (ColumnRule, MetadataRule)](
    rules: Iterable[Rule], identifier: Callable[[Rule], str]
) -> tuple[dict[str, Rule], tuple[tuple[Rule, re.Pattern[str]], ...]]:
    """Split rules into literals (keyed by identifier) and (rule, matcher) patterns.

    `identifier` reads a rule's name/key, differing between the two rule
    types (`ColumnRule.name`, `MetadataRule.key`).
    """
    literal: dict[str, Rule] = {}
    patterns: list[tuple[Rule, re.Pattern[str]]] = []
    for rule in rules:
        name = identifier(rule)
        if _is_pattern(name):
            patterns.append((rule, _matcher(name)))
        else:
            literal[name] = rule
    return literal, tuple(patterns)


def compile_spec(spec: SchemaSpec) -> CompiledSpec:
    columns_literal, columns_pattern = _partition(spec.columns, lambda r: r.name)
    metadata_literal, metadata_pattern = _partition(spec.metadata, lambda r: r.key)
    return CompiledSpec(
        columns_literal=columns_literal,
        columns_pattern=columns_pattern,
        metadata_literal=metadata_literal,
        metadata_pattern=metadata_pattern,
        frame=spec.frame,
    )


def column_signature(value: object) -> tuple[Kind, int]:
    """Derive `(kind, width)` from a built column value.

    A numpy array is 1-D for width 1 or 2-D with its second dim as width; a
    string column is a list of strings (width 1) or a list of lists.
    """
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


def metadata_signature(value: object) -> tuple[Kind, tuple[int, ...]]:
    """Derive `(kind, shape)` from a built metadata value.

    `bool` is checked before `int` because Python's `bool` is a subclass of
    `int`.
    """
    if isinstance(value, bool):
        return Kind.BOOL, ()
    if isinstance(value, int):
        return Kind.INT, ()
    if isinstance(value, float):
        return Kind.REAL, ()
    if isinstance(value, str):
        return Kind.STR, ()
    if isinstance(value, np.ndarray):
        return _NUMPY_KIND[value.dtype.kind], (value.shape[0],)
    if isinstance(value, list):
        # a string array (`list[str]`)
        return Kind.STR, (len(value),)
    raise TypeError(f"unsupported metadata value type: {type(value).__name__}")


def _metadata_sig_str(kind: Kind, shape: tuple[int, ...]) -> str:
    letter = KIND_TO_LETTER[kind]
    return letter if shape == () else f"{letter}[{shape[0]}]"


def _cardinality(rule: ColumnRule | MetadataRule) -> tuple[int, int | None]:
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
        if name not in present:
            if rule.required:
                expected = _column_sig_str(rule.kind, rule.width)
                out.append(Violation("column", name, "missing", expected, None))
            continue
        claimed.add(name)
        kind, width = column_signature(present[name])
        if (kind, width) != (rule.kind, rule.width):
            expected = _column_sig_str(rule.kind, rule.width)
            out.append(
                Violation(
                    "column", name, "mismatch", expected, _column_sig_str(kind, width)
                )
            )

    for rule, matcher in compiled.columns_pattern:
        matches = [n for n in present if n not in claimed and matcher.match(n)]
        for name in matches:
            claimed.add(name)
            kind, width = column_signature(present[name])
            if (kind, width) != (rule.kind, rule.width):
                expected = _column_sig_str(rule.kind, rule.width)
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


def _validate_metadata(
    frame: Frame, compiled: CompiledSpec, level: Conformance
) -> list[Violation]:
    out: list[Violation] = []
    claimed: set[str] = set()
    present = frame.metadata

    for name, rule in compiled.metadata_literal.items():
        if name not in present:
            if rule.required:
                expected = _metadata_sig_str(rule.kind, rule.shape)
                out.append(Violation("metadata", name, "missing", expected, None))
            continue
        claimed.add(name)
        kind, shape = metadata_signature(present[name])
        if (kind, shape) != (rule.kind, rule.shape):
            expected = _metadata_sig_str(rule.kind, rule.shape)
            out.append(
                Violation(
                    "metadata",
                    name,
                    "mismatch",
                    expected,
                    _metadata_sig_str(kind, shape),
                )
            )

    for rule, matcher in compiled.metadata_pattern:
        matches = [n for n in present if n not in claimed and matcher.match(n)]
        for name in matches:
            claimed.add(name)
            kind, shape = metadata_signature(present[name])
            if (kind, shape) != (rule.kind, rule.shape):
                expected = _metadata_sig_str(rule.kind, rule.shape)
                out.append(
                    Violation(
                        "metadata",
                        name,
                        "mismatch",
                        expected,
                        _metadata_sig_str(kind, shape),
                    )
                )
        lo, hi = _cardinality(rule)
        if len(matches) < lo or (hi is not None and len(matches) > hi):
            out.append(
                Violation(
                    "metadata",
                    rule.key,
                    "count",
                    str(rule.count or lo),
                    str(len(matches)),
                )
            )

    if level in ("strict", "warn"):
        for name in present:
            if name not in claimed:
                kind, shape = metadata_signature(present[name])
                out.append(
                    Violation(
                        "metadata", name, "extra", None, _metadata_sig_str(kind, shape)
                    )
                )
    return out


def _validate_frame_rule(frame: Frame, compiled: CompiledSpec) -> list[Violation]:
    rule = compiled.frame
    if rule is None:
        return []
    out: list[Violation] = []
    lo, hi = rule.n_atoms_min, rule.n_atoms_max
    if (lo is not None and frame.n_atoms < lo) or (
        hi is not None and frame.n_atoms > hi
    ):
        bounds = f"[{'' if lo is None else lo}, {'' if hi is None else hi}]"
        out.append(
            Violation("frame", "n_atoms", "mismatch", bounds, str(frame.n_atoms))
        )
    if rule.lattice_required and "Lattice" not in frame.metadata:
        out.append(Violation("frame", "Lattice", "missing", "required", None))
    return out


def validate_frame(
    frame: Frame, compiled: CompiledSpec, level: Conformance
) -> list[Violation]:
    """Return every schema deviation in `frame`.

    Never raises. `extra` items are reported only under `strict`/`warn`;
    `missing`/`mismatch`/`count` at all levels.
    """
    return (
        _validate_columns(frame, compiled, level)
        + _validate_metadata(frame, compiled, level)
        + _validate_frame_rule(frame, compiled)
    )


class SchemaError(OxyzError):
    """A frame failed schema validation.

    Carries the offending frame and entry as attributes, so callers need not
    parse the message.

    Attributes
    ----------
    frame_index
        Index of the offending frame, or `None` if not known.
    name
        The column name, metadata key, or frame-rule field that failed, or
        `None` if not known.
    """

    def __init__(
        self, message: str, *, frame_index: int | None = None, name: str | None = None
    ) -> None:
        super().__init__(message)
        self.frame_index = frame_index
        self.name = name


class SchemaWarning(UserWarning):
    """A schema deviation under `conformance="warn"`.

    Silence with `warnings.filterwarnings("ignore",
    category=oxyz.SchemaWarning)`.
    """


def body(violation: Violation) -> str:
    match violation.deviation:
        case "mismatch":
            if violation.axis == "frame":
                return f"expected in {violation.expected}, found {violation.found}"
            return f"expected {violation.expected}, found {violation.found}"
        case "missing":
            return "missing (required)"
        case "extra":
            return f"unexpected ({violation.found})"
        case "count":
            return f"expected {violation.expected}, found {violation.found}"


def message(violation: Violation, frame_index: int) -> str:
    return (
        f"frame {frame_index}: {violation.axis} '{violation.name}': {body(violation)}"
    )


def resolve(schema: SchemaSpec | str | Path) -> CompiledSpec:
    """Compile a `SchemaSpec` directly, or load one from a file path first."""
    spec = schema if isinstance(schema, SchemaSpec) else SchemaSpec.from_file(schema)
    return compile_spec(spec)


def enforce_frame(
    frame: Frame, compiled: CompiledSpec, level: Conformance, frame_index: int
) -> None:
    """Validate one frame and apply policy.

    Raises `SchemaError` on the first violation under `strict`/`required`;
    emits a `SchemaWarning` per violation under `warn`; does nothing when
    conformant.
    """
    violations = validate_frame(frame, compiled, level)
    if not violations:
        return
    if level == "warn":
        for v in violations:
            warnings.warn(message(v, frame_index), SchemaWarning, stacklevel=3)
        return
    first = violations[0]
    raise SchemaError(
        message(first, frame_index), frame_index=frame_index, name=first.name
    )
