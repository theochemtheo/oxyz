"""Notebook rendering: the `_repr_html_` tables.

Plain `<table>` markup with no colours, scripts or styles, so a notebook's own
theme applies. Unlike `repr`, cells preview values, within fixed bounds: the
first `PREVIEW_ROWS` rows of an array, arrays in full up to `ARRAY_ELEMENTS`
elements, every cell cut at `CELL_CHARS`. Numbers are formatted here, element
by element, so numpy's global printoptions never change the output. Every name
and value is escaped: keys come from the file.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from oxyz._batch import Batch
    from oxyz._frames import Frame
    from oxyz._scan import FrameIndex
    from oxyz._schema import ColumnSchema, MetadataSchema, Schema
    from oxyz._schema_spec import ColumnRule, MetadataRule, SchemaSpec

PREVIEW_ROWS = 3
ARRAY_ELEMENTS = 12
CELL_CHARS = 80
DECIMALS = 4
DASH = "—"


def number(value: object) -> str:
    """Format one scalar: floats to `DECIMALS` places, strings as `repr`."""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return np.format_float_positional(float(value), precision=DECIMALS, trim="0")
    if isinstance(value, str):
        return repr(str(value))  # str() drops numpy 2's `np.str_(...)` repr
    raise TypeError(type(value).__name__)


def _size(value: object) -> int:
    if isinstance(value, np.ndarray):
        return value.size
    if isinstance(value, list):
        return sum(_size(item) for item in value)
    return 1


def value_text(value: object) -> str:
    """Format a scalar, array, or (nested) string list, at most 12 elements.

    Past `ARRAY_ELEMENTS`, keep the leading items whose elements fit, then
    `…`; each kept item is formatted, and so bounded, the same way.
    """
    if isinstance(value, np.ndarray) and value.ndim == 0:
        # Narrowing from `object` leaves numpy's stubs an unusable dtype.
        return number(cast("np.ndarray", value).tolist())
    if not isinstance(value, (np.ndarray, list)):
        return number(value)
    items = list(value)
    total = _size(value)
    if total <= ARRAY_ELEMENTS:
        return "[" + " ".join(value_text(item) for item in items) + "]"
    per_item = max(1, total // max(1, len(items)))
    kept = items[: max(1, ARRAY_ELEMENTS // per_item)]
    return "[" + " ".join(value_text(item) for item in kept) + " …]"


def rows_text(value: object) -> str:
    """Format the first `PREVIEW_ROWS` rows of a column, then `, …`."""
    rows = list(value[:PREVIEW_ROWS])  # ty: ignore[not-subscriptable]
    text = ", ".join(value_text(row) for row in rows)
    return text + ", …" if len(value) > PREVIEW_ROWS else text  # ty: ignore[invalid-argument-type]


def dtype_text(value: object) -> str:
    if isinstance(value, np.ndarray):
        return str(value.dtype)
    if isinstance(value, list):
        return "str"  # string columns and string metadata arrays, by contract
    return type(value).__name__


def shape_text(value: object) -> str:
    if isinstance(value, np.ndarray):
        return str(value.shape)
    if isinstance(value, list):
        first = value[0] if value else None
        return str(
            (len(value), len(first)) if isinstance(first, list) else (len(value),)
        )
    return "()"


def _cell(format_: Callable[[object], str], value: object) -> str:
    """Format a cell, falling back to the type name; cut at `CELL_CHARS`."""
    try:
        text = format_(value)
    except Exception:  # noqa: BLE001  a notebook repr must render whatever it holds
        text = type(value).__name__
    if len(text) > CELL_CHARS:
        text = text[: CELL_CHARS - 1] + "…"
    return html.escape(text)


def _table(caption: str, header: Iterable[str], rows: Iterable[Iterable[str]]) -> str:
    head = "".join(f"<th>{html.escape(name)}</th>" for name in header)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return (
        f"<table><caption>{html.escape(caption)}</caption>"
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    )


def _page(kind: str, facts: Iterable[str], *blocks: str) -> str:
    title = " · ".join([f"<strong>{kind}</strong>", *map(html.escape, facts)])
    return f'<div class="oxyz-repr"><p>{title}</p>{"".join(blocks)}</div>'


def _fields(
    caption: str,
    last: str,
    fields: Mapping[str, object],
    format_: Callable[[object], str],
) -> str:
    rows = [
        [
            html.escape(name),
            _cell(dtype_text, value),
            _cell(shape_text, value),
            _cell(format_, value),
        ]
        for name, value in fields.items()
    ]
    return _table(caption, ["name", "dtype", "shape", last], rows)


def _atoms(n_atoms: np.ndarray, total: int) -> str:
    if not n_atoms.size:
        return f"{total} atoms"
    return f"{total} atoms ({int(n_atoms.min())}..{int(n_atoms.max())} per frame)"


def frame(obj: Frame) -> str:
    return _page(
        "Frame",
        [f"{obj.n_atoms} atoms"],
        _fields("columns", "preview", obj.columns, rows_text),
        _fields("metadata", "value", obj.metadata, value_text),
    )


def batch(obj: Batch) -> str:
    return _page(
        "Batch",
        [f"{obj.n_frames} frames", _atoms(obj.n_atoms, obj.total_atoms)],
        _fields("columns", "preview", obj.columns, rows_text),
        _fields("metadata", "value", obj.metadata, rows_text),
    )


def schema(obj: Schema) -> str:
    from oxyz._schema import column_sig, metadata_sig

    def present(entry: ColumnSchema | MetadataSchema) -> str:
        return f"{entry.frames_present}/{obj.n_frames}"

    columns = [
        [
            html.escape(c.name),
            html.escape("|".join(column_sig(v.kind, v.width) for v in c.variants)),
            present(c),
            DASH if c.unified is None else html.escape(column_sig(*c.unified)),
        ]
        for c in obj.columns
    ]
    metadata = [
        [
            html.escape(m.key),
            html.escape("|".join(metadata_sig(v.kind, v.shape) for v in m.variants)),
            present(m),
            DASH if m.unified is None else html.escape(metadata_sig(*m.unified)),
        ]
        for m in obj.metadata
    ]
    header = ["name", "type", "frames", "unified"]
    return _page(
        "Schema",
        [
            f"{obj.n_frames} frames",
            _atoms(obj.n_atoms, obj.total_atoms),
            "consistent" if obj.is_consistent else "inconsistent",
        ],
        _table("columns", header, columns),
        _table("metadata", header, metadata),
    )


def spec(obj: SchemaSpec) -> str:
    from oxyz._schema import column_sig, metadata_sig

    def bounds(rule: ColumnRule | MetadataRule) -> list[str]:
        return [
            "" if value is None else _cell(number, value)
            for value in (rule.count, rule.min, rule.max, rule.fill)
        ]

    columns = [
        [
            html.escape(r.name),
            html.escape(column_sig(r.kind, r.width)),
            "yes" if r.required else "no",
            *bounds(r),
        ]
        for r in obj.columns
    ]
    metadata = [
        [
            html.escape(r.key),
            html.escape(metadata_sig(r.kind, r.shape)),
            "yes" if r.required else "no",
            *bounds(r),
        ]
        for r in obj.metadata
    ]
    header = ["name", "type", "required", "count", "min", "max", "fill"]
    frame_line = (
        "" if obj.frame is None else f"<p>frame: {html.escape(repr(obj.frame))}</p>"
    )
    return _page(
        "SchemaSpec",
        [obj.mode],
        frame_line,
        _table("columns", header, columns),
        _table("metadata", header, metadata),
    )


def frame_index(obj: FrameIndex) -> str:
    def stat(value: float | None) -> str:
        return DASH if value is None else _cell(number, value)

    atoms = _table(
        "atoms per frame",
        ["statistic", "value"],
        [
            ["min", stat(obj.min_atoms)],
            ["median", stat(obj.median_atoms)],
            ["mean", stat(obj.mean_atoms)],
            ["max", stat(obj.max_atoms)],
            ["std", stat(obj.std_atoms)],
        ],
    )
    blocks = [atoms]
    if obj.volumes is not None:
        known = obj.volumes[~np.isnan(obj.volumes)]
        blocks.append(
            _table(
                "volume",
                ["statistic", "value"],
                [
                    ["min", stat(float(known.min()) if known.size else None)],
                    ["mean", stat(float(known.mean()) if known.size else None)],
                    ["max", stat(float(known.max()) if known.size else None)],
                    ["frames without", str(obj.volumes.size - known.size)],
                ],
            )
        )
    return _page(
        "FrameIndex", [f"{obj.n_frames} frames", f"{obj.total_atoms} atoms"], *blocks
    )
