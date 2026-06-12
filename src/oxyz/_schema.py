from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import oxyz._rust as _rust


class Kind(StrEnum):
    """The four extxyz value kinds, shared by columns and metadata."""

    REAL = "Real"
    INT = "Int"
    BOOL = "Bool"
    STR = "Str"


@dataclass(frozen=True, slots=True)
class ColumnVariant:
    """One observed (kind, width) combination for a per-atom column.

    `frames` counts how many frames used exactly this combination; the
    variant counts on a column sum to its `frames_present`.
    """

    kind: Kind
    width: int
    frames: int


@dataclass(frozen=True, slots=True)
class MetadataVariant:
    """One observed (kind, shape) combination for a metadata key.

    `shape` follows numpy: `()` for scalars, `(n,)` for arrays. `frames`
    counts how many frames used exactly this combination.
    """

    kind: Kind
    shape: tuple[int, ...]
    frames: int


@dataclass(frozen=True, slots=True)
class ColumnSchema:
    """Everything observed about one per-atom column across the file.

    More than one variant means the column changed kind or width between
    frames; `frames_present < n_frames` means some frames lack it entirely.
    `unified` is the single (kind, width) every frame's column can be read
    as — the sole variant, or the Real that an Int/Real pair of equal width
    promotes to — and None when the variants genuinely conflict.
    """

    name: str
    variants: tuple[ColumnVariant, ...]
    frames_present: int
    unified: tuple[Kind, int] | None


@dataclass(frozen=True, slots=True)
class MetadataSchema:
    """Everything observed about one metadata key across the file.

    Mirrors `ColumnSchema`; `unified` is (kind, shape) under the same
    Int/Real promotion rule, or None on a genuine conflict.
    """

    key: str
    variants: tuple[MetadataVariant, ...]
    frames_present: int
    unified: tuple[Kind, tuple[int, ...]] | None


@dataclass(frozen=True, slots=True)
class Schema:
    """Observed structure of a dataset: which columns and metadata keys
    appear, with what types and shapes, and how consistently.

    Built by `infer_schema` in one pass; records counts, not frame indices.
    `is_consistent` is strict: every column and key has a single variant and
    appears in every frame. Int/Real promotion does not count — consult
    each entry's `unified` for that looser reading. `min_atoms`/`max_atoms`
    are None only for an empty file.
    """

    n_frames: int
    total_atoms: int
    min_atoms: int | None
    max_atoms: int | None
    columns: tuple[ColumnSchema, ...]
    metadata: tuple[MetadataSchema, ...]
    is_consistent: bool
    _report: str = field(repr=False)

    def report(self) -> str:
        """Human-readable summary: one line per column and metadata key."""
        return self._report

    def __str__(self) -> str:
        return self._report


def _column_schema(data: _rust.ColumnSchemaData) -> ColumnSchema:
    unified = data["unified"]
    return ColumnSchema(
        name=data["name"],
        variants=tuple(
            ColumnVariant(Kind(v["kind"]), v["width"], v["frames"])
            for v in data["variants"]
        ),
        frames_present=data["frames_present"],
        unified=None if unified is None else (Kind(unified[0]), unified[1]),
    )


def _metadata_schema(data: _rust.MetadataSchemaData) -> MetadataSchema:
    unified = data["unified"]
    return MetadataSchema(
        key=data["key"],
        variants=tuple(
            MetadataVariant(Kind(v["kind"]), v["shape"], v["frames"])
            for v in data["variants"]
        ),
        frames_present=data["frames_present"],
        unified=None if unified is None else (Kind(unified[0]), unified[1]),
    )


def infer_schema(path: str | Path) -> Schema:
    """Infer the dataset's schema in a single pass over the file."""
    data = _rust.infer_schema(str(path))
    return Schema(
        n_frames=data["n_frames"],
        total_atoms=data["total_atoms"],
        min_atoms=data["min_atoms"],
        max_atoms=data["max_atoms"],
        columns=tuple(_column_schema(entry) for entry in data["columns"]),
        metadata=tuple(_metadata_schema(entry) for entry in data["metadata"]),
        is_consistent=data["is_consistent"],
        _report=data["report"],
    )
