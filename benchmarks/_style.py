"""Reader palette, ordering and number formatting shared by every figure."""

from __future__ import annotations

from typing import Any

import seaborn as sns

# Fixed so every reader keeps its position and colour across figures. oxyz
# rows share fading shades of one hue; competitors stay muted.
OXYZ_BASE = "#d95f02"
OXYZ_ORDER = [
    "oxyz",
    "oxyz-serial",
    "oxyz-iter",
    "oxyz-batches",
    "oxyz-read-batch",
    "oxyz-to-ase",
    "oxyz-to-ase-serial",
    "oxyz-scan",
    "sequential-64-frames",
    "shuffled-2048-atoms",
]
# Blend towards a light-but-still-orange endpoint so no shade washes out.
OXYZ_SHADES = dict(
    zip(
        OXYZ_ORDER,
        sns.blend_palette([OXYZ_BASE, "#f7bd84"], len(OXYZ_ORDER)),
        strict=True,
    )
)
COMPETITOR_COLORS = {
    "ase": "#7570b3",
    "cextxyz": "#1b9e77",
    "cextxyz-to-ase": "#66a61e",
    "atompack-serial": "#1f78b4",
    "atompack-native": "#a6cee3",
    "lmdb-pickle": "#e377c2",
    "ase-sqlite": "#7570b3",
    "ase-lmdb": "#9e9ac8",
}
FALLBACK_COLOR = "#999999"


def reader_order(readers: set[str]) -> list[str]:
    """oxyz rows first in their fixed order, competitors after."""
    ours = sorted(
        (r for r in readers if r not in COMPETITOR_COLORS),
        key=lambda r: (OXYZ_ORDER.index(r) if r in OXYZ_ORDER else len(OXYZ_ORDER), r),
    )
    return ours + sorted(r for r in readers if r in COMPETITOR_COLORS)


def reader_color(reader: str) -> Any:
    if reader in OXYZ_SHADES:
        return OXYZ_SHADES[reader]
    return COMPETITOR_COLORS.get(reader, FALLBACK_COLOR)


def fmt_value(value: float) -> str:
    for cut, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if value >= cut:
            return f"{value / cut:.3g}{suffix}"
    return f"{value:.3g}"


# Legends name the call each line measures; all-core labels fill in the
# recording machine's core count.
ALL_CORE_LABELS = {
    "oxyz": "oxyz.read",
    "oxyz-to-ase": "oxyz.ase.read",
}

READER_LABELS = {
    "oxyz-serial": "oxyz.read(threads=1)",
    "oxyz-to-ase-serial": "oxyz.ase.read(threads=1)",
    "ase": 'ase.io.read(format="extxyz")',
    "cextxyz": "extxyz.read_dicts",
    "cextxyz-to-ase": 'ase.io.read(format="cextxyz")',
}


def reader_label(reader: str, ncores: int | None = None) -> str:
    """The function-call label for a reader in a scaling legend."""
    call = ALL_CORE_LABELS.get(reader)
    if call is not None:
        return f"{call}(threads={ncores})" if ncores else f"{call}()"
    return READER_LABELS.get(reader, reader)


# Tells the all-core oxyz.read line from the serial one, which shares its hue.
MARKERS = {"oxyz": "s"}


def reader_marker(reader: str) -> str:
    return MARKERS.get(reader, "o")
