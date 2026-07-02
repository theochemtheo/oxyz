"""Fast extxyz reading for atomistic machine learning.

A Rust parser behind a small, typed Python API. `read_frames`/`iter_frames`
return frames as numpy arrays; `read_batch`/`iter_batches` concatenate frames
into batches; `scan` and `infer_schema` report a file's structure. A
`SchemaSpec` supplied to the read functions validates frames against an
expected structure. ASE conversion lives in the optional `oxyz.ase`
submodule.

Columns and metadata are kept as written — no aliasing, no normalisation.
"""

from __future__ import annotations

from oxyz._batch import Batch, MemoryScaling, iter_batches, read_batch
from oxyz._frames import (
    ColumnValues,
    Compression,
    Frame,
    MetadataValue,
    iter_frames,
    read_first,
    read_frames,
)
from oxyz._remote import StorageOptions
from oxyz._rust import ParseError
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
from oxyz._schema_spec import ColumnRule, FrameRule, MetadataRule, SchemaSpec
from oxyz._write import Writer, write

__all__ = [
    "Batch",
    "ColumnRule",
    "ColumnSchema",
    "ColumnValues",
    "ColumnVariant",
    "Compression",
    "Conformance",
    "Frame",
    "FrameIndex",
    "FrameRule",
    "Kind",
    "MemoryScaling",
    "MetadataRule",
    "MetadataSchema",
    "MetadataValue",
    "MetadataVariant",
    "ParseError",
    "Schema",
    "SchemaError",
    "SchemaSpec",
    "SchemaWarning",
    "StorageOptions",
    "Violation",
    "Writer",
    "infer_schema",
    "iter_batches",
    "iter_frames",
    "read_batch",
    "read_first",
    "read_frames",
    "scan",
    "write",
]
