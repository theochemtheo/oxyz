from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import oxyz._rust as _rust
from oxyz import _remote
from oxyz._stats import AtomCountStats

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from oxyz._frames import Compression
    from oxyz._schema_spec import SchemaSpec


class Kind(StrEnum):
    """The four extxyz value kinds, shared by columns and metadata.

    Attributes
    ----------
    REAL
        Floating-point; letter code `R` in a `ColumnRule`/`MetadataRule`
        `kind`.
    INT
        Integer; letter code `I`.
    BOOL
        Boolean; letter code `L`.
    STR
        String; letter code `S`.
    """

    REAL = "Real"
    INT = "Int"
    BOOL = "Bool"
    STR = "Str"


@dataclass(frozen=True, slots=True)
class ColumnVariant:
    """One observed (kind, width) combination for a per-atom column.

    `frames` counts how many frames used exactly this combination; the
    variant counts on a column sum to its `frames_present`.

    Attributes
    ----------
    kind
        Value kind observed.
    width
        Number of components per atom (1 for a scalar column).
    frames
        Number of frames that used this (kind, width) combination.
    """

    kind: Kind
    width: int
    frames: int


@dataclass(frozen=True, slots=True)
class MetadataVariant:
    """One observed (kind, shape) combination for a metadata key.

    `shape` follows numpy: `()` for scalars, `(n,)` for arrays. `frames`
    counts how many frames used exactly this combination.

    Attributes
    ----------
    kind
        Value kind observed.
    shape
        Array shape observed, `()` for a scalar.
    frames
        Number of frames that used this (kind, shape) combination.
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

    Attributes
    ----------
    name
        Column name, e.g. `"pos"`, `"forces"`.
    variants
        Every (kind, width) combination observed, in first-seen order.
    frames_present
        Number of frames that carried this column at all.
    unified
        The single (kind, width) every frame's column can be read as, or
        None on a genuine conflict.
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

    Attributes
    ----------
    key
        Metadata key, e.g. `"Lattice"`, `"energy"`.
    variants
        Every (kind, shape) combination observed, in first-seen order.
    frames_present
        Number of frames that carried this key at all.
    unified
        The single (kind, shape) every frame's value can be read as, or
        None on a genuine conflict.
    """

    key: str
    variants: tuple[MetadataVariant, ...]
    frames_present: int
    unified: tuple[Kind, tuple[int, ...]] | None


@dataclass(frozen=True, slots=True)
class Schema(AtomCountStats):
    """Observed structure of a dataset: columns, metadata, and consistency.

    Built by `infer_schema` in one pass; records counts, not frame indices.
    `is_consistent` is strict: every column and key has a single variant and
    appears in every frame. Int/Real promotion does not count — consult
    each entry's `unified` for that looser reading. `min_atoms`/`max_atoms`
    are None only for an empty file. The same single pass keeps the per-frame
    `n_atoms`, so `mean_atoms`/`median_atoms`/`std_atoms` (from
    `AtomCountStats`) match what a `scan` would report without a second read.

    Attributes
    ----------
    n_frames
        Number of frames in the file.
    total_atoms
        Sum of `n_atoms` across all frames.
    min_atoms
        Smallest per-frame atom count, or None if empty.
    max_atoms
        Largest per-frame atom count, or None if empty.
    n_atoms
        Declared atom count per frame.
    columns
        Every per-atom column observed, in first-seen order.
    metadata
        Every comment-line metadata key observed, in first-seen order.
    is_consistent
        Whether every column and key has a single variant and appears in
        every frame.
    """

    n_frames: int
    total_atoms: int
    min_atoms: int | None
    max_atoms: int | None
    n_atoms: np.ndarray
    columns: tuple[ColumnSchema, ...]
    metadata: tuple[MetadataSchema, ...]
    is_consistent: bool
    _report: str = field(repr=False)

    def report(self) -> str:
        """Human-readable summary: one line per column and metadata key."""
        return self._report

    def to_spec(self) -> SchemaSpec:
        """Best-effort prescriptive schema from what was observed.

        Partial-presence entries come out optional, enumerable column
        families collapse to `*` globs, and there are no `frame` bounds.
        Feed the result back to `read(..., schema=...)`.
        """
        from oxyz._schema_emit import spec_from_schema

        return spec_from_schema(self)

    def __str__(self) -> str:
        """Alias for `report()`."""
        return self._report


def _column_schema(data: _rust.ColumnSchemaData) -> ColumnSchema:
    """Build a `ColumnSchema` from the Rust scan's raw data for one column."""
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
    """Build a `MetadataSchema` from the Rust scan's raw data for one key."""
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


def infer_schema(
    path: str | Path,
    *,
    compression: Compression = "infer",
    member: str | None = None,
    storage_options: _remote.StorageOptions | None = None,
) -> Schema:
    """Fold the whole file into a `Schema` in a single pass.

    Records, per column and metadata key, the observed variants and how many
    frames used each, with a strict `is_consistent` and a promoted `unified`.
    Parses every frame; for the structure alone, use `scan`.

    A compressed path is decoded while streaming; `compression` and `member`
    work as in `read`.

    A remote URL (``s3://``, ``gs://``, ``az://``) streams the object through
    the same parser (needs the ``oxyz[s3]`` extra); ``storage_options`` passes
    endpoint/credentials to the store.

    Parameters
    ----------
    path
        File path or remote URL to fold into a schema.
    compression
        Forces a codec (`"infer"`, `"none"`, `"gzip"`, `"zstd"`, `"zip"`)
        instead of inferring it from `path`.
    member
        Selects one entry from a `.zip`/`.tar`/`.tar.gz` holding more than
        one.
    storage_options
        Endpoint/credentials for a remote store, falling back to `AWS_*`
        env vars.

    Returns
    -------
    Schema
        Observed columns, metadata, and atom-count statistics.

    Examples
    --------
    >>> import oxyz
    >>> oxyz.infer_schema(DATA / "water.extxyz").is_consistent
    True
    >>> oxyz.infer_schema(DATA / "mixed.extxyz").is_consistent
    False
    """
    if _remote.is_remote(path):
        src = _remote.open_source(
            path,
            compression=compression,
            member=member,
            storage_options=storage_options,
        )
        data = _rust.infer_schema_reader(src.obj, src.codec, src.member)
    else:
        data = _rust.infer_schema(str(path), compression, member)
    return Schema(
        n_frames=data["n_frames"],
        total_atoms=data["total_atoms"],
        min_atoms=data["min_atoms"],
        max_atoms=data["max_atoms"],
        n_atoms=data["n_atoms"],
        columns=tuple(_column_schema(entry) for entry in data["columns"]),
        metadata=tuple(_metadata_schema(entry) for entry in data["metadata"]),
        is_consistent=data["is_consistent"],
        _report=data["report"],
    )
