"""Fast extxyz reading for atomistic machine learning.

A Rust parser behind a small, typed Python API. `read`/`iread` return frames as
numpy arrays; `read_batch`/`iread_batch` concatenate frames into batches; `scan`
and `infer_schema` report a file's structure. A
`SchemaSpec` supplied to the read functions validates frames against an
expected structure. ASE conversion lives in the optional `oxyz.ase`
submodule.

Columns and metadata are kept as written — no aliasing, no normalisation.
`Frame`'s derived accessors (`positions`, `numbers`, `symbols`, `cell`, `pbc`)
are opt-in views over those same dicts, computed on access.
"""

from __future__ import annotations

from oxyz._batch import Batch, MemoryScaling, iread_batch, read_batch
from oxyz._convert import FieldError
from oxyz._frames import (
    ColumnValues,
    Compression,
    Frame,
    MetadataValue,
    iread,
    read,
)
from oxyz._lookup import AmbiguousNameError
from oxyz._remote import StorageOptions
from oxyz._rust import OxyzError, ParseError
from oxyz._scan import FrameIndex, scan
from oxyz._schema import (
    ColumnSchema,
    ColumnVariant,
    Kind,
    MetadataSchema,
    MetadataVariant,
    Schema,
    infer_schema,
)
from oxyz._schema_match import (
    Conformance,
    SchemaError,
    SchemaWarning,
    Violation,
)
from oxyz._schema_spec import ColumnRule, FrameRule, MetadataRule, Mode, SchemaSpec
from oxyz._write import Writable, Writer, write

__all__ = [
    "AmbiguousNameError",
    "Batch",
    "ColumnRule",
    "ColumnSchema",
    "ColumnValues",
    "ColumnVariant",
    "Compression",
    "Conformance",
    "FieldError",
    "Frame",
    "FrameIndex",
    "FrameRule",
    "Kind",
    "MemoryScaling",
    "MetadataRule",
    "MetadataSchema",
    "MetadataValue",
    "MetadataVariant",
    "Mode",
    "OxyzError",
    "ParseError",
    "Schema",
    "SchemaError",
    "SchemaSpec",
    "SchemaWarning",
    "StorageOptions",
    "Violation",
    "Writable",
    "Writer",
    "infer_schema",
    "iread",
    "iread_batch",
    "read",
    "read_batch",
    "scan",
    "write",
]
