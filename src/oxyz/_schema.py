from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from oxyz import _html, _remote, _rust, _summary
from oxyz._lookup import NameLookup
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
        Floating-point, letter `R`.
    INT
        Integer, letter `I`.
    BOOL
        Boolean, letter `L`.
    STR
        String, letter `S`.
    """

    REAL = "Real"
    INT = "Int"
    BOOL = "Bool"
    STR = "Str"

    def __repr__(self) -> str:
        """Name the member as it is spelt in code, e.g. `Kind.REAL`."""
        return f"Kind.{self.name}"


LETTER_TO_KIND: dict[str, Kind] = {
    "R": Kind.REAL,
    "I": Kind.INT,
    "L": Kind.BOOL,
    "S": Kind.STR,
}
KIND_TO_LETTER: dict[Kind, str] = {
    kind: letter for letter, kind in LETTER_TO_KIND.items()
}


def column_sig(kind: Kind, width: int) -> str:
    """Render a column's type as extxyz writes it in `Properties=`: `R:3`."""
    return f"{KIND_TO_LETTER[kind]}:{width}"


def metadata_sig(kind: Kind, shape: tuple[int, ...]) -> str:
    """Render a metadata value's type: `R` for a scalar, else `R[9]`, `R[3, 3]`."""
    letter = KIND_TO_LETTER[kind]
    return letter if shape == () else f"{letter}[{', '.join(map(str, shape))}]"


@dataclass(frozen=True, slots=True)
class ColumnVariant:
    """One observed (kind, width) combination for a per-atom column.

    Attributes
    ----------
    kind
        Value kind observed.
    width
        Number of components per atom (1 for a scalar column).
    frames
        Number of frames that used this combination.
    """

    kind: Kind
    width: int
    frames: int


@dataclass(frozen=True, slots=True)
class MetadataVariant:
    """One observed (kind, shape) combination for a metadata key.

    Attributes
    ----------
    kind
        Value kind observed.
    shape
        Array shape observed, `()` for a scalar.
    frames
        Number of frames that used this combination.
    """

    kind: Kind
    shape: tuple[int, ...]
    frames: int


@dataclass(frozen=True, slots=True)
class ColumnSchema:
    """Everything observed about one per-atom column across the file.

    Attributes
    ----------
    name
        Column name, e.g. `"pos"`, `"forces"`.
    variants
        Every (kind, width) combination observed, in first-seen order.
    frames_present
        Number of frames that carried this column.
    unified
        The (kind, width) every frame's column can be read as, promoting an
        Int/Real pair of equal width to Real; `None` on a conflict.
    """

    name: str
    variants: tuple[ColumnVariant, ...]
    frames_present: int
    unified: tuple[Kind, int] | None


@dataclass(frozen=True, slots=True)
class MetadataSchema:
    """Everything observed about one metadata key across the file.

    Attributes
    ----------
    key
        Metadata key, e.g. `"Lattice"`, `"energy"`.
    variants
        Every (kind, shape) combination observed, in first-seen order.
    frames_present
        Number of frames that carried this key.
    unified
        The (kind, shape) every frame's value can be read as, promoting as
        `ColumnSchema.unified` does; `None` on a conflict.
    """

    key: str
    variants: tuple[MetadataVariant, ...]
    frames_present: int
    unified: tuple[Kind, tuple[int, ...]] | None


@dataclass(frozen=True, slots=True)
class Schema(AtomCountStats, NameLookup[ColumnSchema, MetadataSchema]):
    """Observed structure of a dataset: columns, metadata, and consistency.

    `schema[name]` returns the `ColumnSchema` or `MetadataSchema` so named.
    The atom-count statistics match what `scan` reports.

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
        every frame. Int/Real promotion does not count; see `unified`.
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

    def _sides(
        self,
    ) -> tuple[dict[str, ColumnSchema], dict[str, MetadataSchema]]:
        return (
            {c.name: c for c in self.columns},
            {m.key: m for m in self.metadata},
        )

    def _repr_html_(self) -> str:
        """Render as tables for a notebook, previewing a bounded slice of values."""
        return _html.schema(self)

    def report(self) -> str:
        """Human-readable summary: one line per column and metadata key."""
        return self._report

    def to_spec(self) -> SchemaSpec:
        """Best-effort prescriptive schema from what was observed.

        Partially present entries are optional, column families collapse to `*`
        globs, and there are no `frame` bounds.
        """
        from oxyz._schema_emit import spec_from_schema

        return spec_from_schema(self)

    def __str__(self) -> str:
        """Alias for `report()`."""
        return self._report

    def __repr__(self) -> str:
        """Summarise counts and each entry's type, in extxyz letters.

        Drifting variants join with `|`; `?` marks an entry some frames lack.
        """

        def absent(entry: ColumnSchema | MetadataSchema) -> str:
            return "?" if entry.frames_present < self.n_frames else ""

        columns = {
            c.name: "|".join(column_sig(v.kind, v.width) for v in c.variants)
            + absent(c)
            for c in self.columns
        }
        metadata = {
            m.key: "|".join(metadata_sig(v.kind, v.shape) for v in m.variants)
            + absent(m)
            for m in self.metadata
        }
        return (
            f"Schema(n_frames={self.n_frames}, total_atoms={self.total_atoms}"
            f"{_summary.atom_range(self.n_atoms)}, "
            f"columns={_summary.mapping_repr(columns, str)}, "
            f"metadata={_summary.mapping_repr(metadata, str)}, "
            f"is_consistent={self.is_consistent})"
        )


def _column_schema(data: _rust.ColumnSchemaData) -> ColumnSchema:
    """Build a `ColumnSchema` from the binding's raw data for one column."""
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
    """Build a `MetadataSchema` from the binding's raw data for one key."""
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

    Parses every frame; for the structure alone, use `scan`.

    Parameters
    ----------
    path
        File path, or a remote URL; see `read`.
    compression
        Forces a codec instead of inferring it from `path`; see `read`.
    member
        Selects one entry from an archive holding more than one; see `read`.
    storage_options
        Endpoint/credentials for a remote store; see `read`.

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
