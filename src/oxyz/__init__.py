"""Fast extxyz reading for atomistic machine learning.

`read`/`iread` return frames of numpy arrays, `read_batch`/`iread_batch`
concatenate them, and `scan` and `infer_schema` report a file's structure.
A `SchemaSpec` validates or reshapes frames as they are read. Columns and
metadata are kept as written. `oxyz.ase`, `oxyz.metatomic` and
`oxyz.torch_sim` convert to other libraries.
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
from oxyz._record import Cell, Column, FieldSpec, Matrix3, Meta, Record
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
    "Cell",
    "Column",
    "ColumnRule",
    "ColumnSchema",
    "ColumnValues",
    "ColumnVariant",
    "Compression",
    "Conformance",
    "FieldError",
    "FieldSpec",
    "Frame",
    "FrameIndex",
    "FrameRule",
    "Kind",
    "Matrix3",
    "MemoryScaling",
    "Meta",
    "MetadataRule",
    "MetadataSchema",
    "MetadataValue",
    "MetadataVariant",
    "Mode",
    "OxyzError",
    "ParseError",
    "Record",
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
