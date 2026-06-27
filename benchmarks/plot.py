# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib>=3.8",
#     "seaborn>=0.13",
# ]
# ///
"""Render benchmarks/figures/ from a saved pytest-benchmark run.

Usage:
    uv run benchmarks/plot.py [SAVE.json]

Reads the same save JSON as report.py (newest under .benchmarks/ without
an argument). Saves are first flattened into tidy rows so styling changes
never touch data handling.

Bars show atoms/s (higher is better) where the save records the row's
workload shape — atoms/s compares across workloads where frames/s does
not, since a frame is whatever size the file made it. Panels without
shapes (first/last/scan) show mean time instead (lower is better, noted
on the axis). Parallel rows say how many cores they used; the count
comes from the machine the save was recorded on.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

# Supplied by this script's inline metadata, not the project venv that ty
# checks under.
import seaborn as sns
from matplotlib.ticker import LogLocator, NullFormatter

REPO = Path(__file__).resolve().parent.parent
FIGURES = Path(__file__).resolve().parent / "figures"

# Scenarios with one or two small groups share a figure; everything else
# (read_all now, the store comparison later) gets its own, one panel per
# group. scan is special-cased: its groups become columns of one panel.
SHARED_SCENARIOS = ["selective", "batches", "read_first", "read_last"]


def newest_save() -> Path:
    saves = sorted(
        (REPO / ".benchmarks").rglob("*.json"), key=lambda p: p.stat().st_mtime
    )
    if not saves:
        sys.exit("no saves under .benchmarks/ — run with --benchmark-autosave first")
    return saves[-1]


def row_id(bench: dict[str, Any]) -> str:
    if bench.get("param"):
        return bench["param"]
    name = bench["name"]
    return name[name.index("[") + 1 : -1] if "[" in name else name


def flatten(data: dict[str, Any]) -> list[dict[str, Any]]:
    """One tidy row per benchmark: scenario/workload from the group name,
    reader from the param id, everything else from stats and extra_info."""
    ncores = data["machine_info"].get("cpu", {}).get("count")
    rows = []
    for bench in data["benchmarks"]:
        group = bench["group"] or "ungrouped"
        scenario, _, workload = group.partition("/")
        info = bench.get("extra_info", {})
        # Swept rows record their thread count and carry a -Nt id suffix;
        # stripping it folds them into one reader with a bar per count.
        # Unswept parallel rows used every core the recording machine had.
        reader = row_id(bench)
        threads = info.get("threads")
        if threads is not None:
            reader = re.sub(r"-\d+t$", "", reader)
        else:
            threads = 1 if info.get("mode") == "serial" else ncores
        rows.append(
            {
                "group": group,
                "scenario": scenario,
                "workload": workload or scenario,
                "reader": reader,
                "threads": threads,
                "output": info.get("output", "?"),
                "mode": info.get("mode", "?"),
                "mean": bench["stats"]["mean"],
                "stddev": bench["stats"]["stddev"],
                "n_frames": info.get("n_frames"),
                "n_atoms": info.get("n_atoms"),
            }
        )
    return rows


# Reader order and colours are fixed globally so every reader keeps its
# position and colour across panels and figures. oxyz rows share the
# saturated base hue in fading shades; competitors stay muted.
OXYZ_BASE = "#d95f02"
OXYZ_ORDER = [
    "oxyz",
    "oxyz-serial",
    "oxyz-iter",
    "oxyz-batches",
    "oxyz-read-batch",
    "oxyz-to-ase",
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


def metric_for(rows: list[dict[str, Any]]) -> str:
    """Throughput where every row recorded its workload; time otherwise."""
    if all(r["n_atoms"] is not None for r in rows):
        return "atoms/s"
    return "mean time (ms)"


def value_of(row: dict[str, Any], metric: str) -> float:
    return row["n_atoms"] / row["mean"] if metric == "atoms/s" else row["mean"] * 1e3


def draw_bar(
    ax, pos: float, value: float, width: float, color, label: str, rotate: bool = False
) -> None:
    ax.bar(pos, value, width, color=color, edgecolor="#262626", linewidth=0.8)
    # Sub-bars in a thread sweep are too narrow for horizontal labels.
    ax.annotate(
        label,
        (pos, value),
        ha="center",
        va="bottom",
        fontsize=7 if rotate else 7.5,
        fontweight="bold",
        rotation=90 if rotate else 0,
        textcoords="offset points",
        xytext=(0, 2),
    )


def finish_log_axis(
    ax, values: list[float], tick_labels: list[str], top: float = 3.0
) -> None:
    ax.set_yscale("log")
    # Explicit limits: matplotlib otherwise zooms a log axis onto a sliver
    # when bars are close, inflating small differences and cluttering the
    # ticks with off-decade labels.
    ax.set_ylim(min(values) / 4, max(values) * top)
    ax.set_xticks(range(len(tick_labels)))
    # rotation_mode="anchor" pivots each label at its right end, keeping it
    # aligned under its tick once rotated.
    ax.set_xticklabels(
        tick_labels,
        rotation=20,
        ha="right",
        rotation_mode="anchor",
        fontsize=8,
        fontweight="bold",
    )
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", which="major", labelsize=8, length=4.5, width=1.1)
    ax.tick_params(axis="y", which="minor", length=2.5, width=0.8)


def panel(ax, rows: list[dict[str, Any]], metric: str) -> None:
    """Vertical bars for one group: readers along x, one bar per thread
    count within each reader's slot."""
    readers = reader_order({r["reader"] for r in rows})
    slot = 0.8  # total width available per reader group
    values = []
    tick_labels = []
    has_sweep = False

    for x, reader in enumerate(readers):
        bars = sorted(
            (r for r in rows if r["reader"] == reader),
            key=lambda r: r["threads"] or 0,
        )
        # A lone parallel bar names its core count in the reader label;
        # a thread sweep labels each bar instead.
        if len(bars) == 1 and bars[0]["mode"] == "parallel":
            tick_labels.append(f"{reader} ({bars[0]['threads']} cores)")
        else:
            tick_labels.append(reader)
        width = slot / len(bars)
        has_sweep = has_sweep or len(bars) > 1
        for i, r in enumerate(bars):
            value = value_of(r, metric)
            values.append(value)
            pos = x - slot / 2 + (i + 0.5) * width
            label = fmt_value(value)
            if len(bars) > 1:
                label = f"{r['threads']}c: {label}"
            draw_bar(
                ax,
                pos,
                value,
                width * 0.92,
                reader_color(reader),
                label,
                rotate=len(bars) > 1,
            )

    # Rotated sweep labels stand tall; give them headroom clear of the title.
    finish_log_axis(ax, values, tick_labels, top=9.0 if has_sweep else 3.0)


def metric_label(metric: str) -> str:
    if metric == "atoms/s":
        return f"{metric} — higher is better"
    return f"{metric} — lower is better"


def render_figure(
    name: str, rows: list[dict[str, Any]], groups: list[str] | None = None
) -> Path:
    """One panel per group. Each panel picks its own metric, so throughput
    and time panels can share a figure; the y label reappears whenever the
    metric changes from the panel before."""
    if groups is None:
        groups = sorted({r["group"] for r in rows})

    fig, axes = plt.subplots(
        1, len(groups), figsize=(0.9 + 3.6 * len(groups), 3.1), squeeze=False
    )
    prev_metric = None
    for ax, group in zip(axes[0], groups, strict=True):
        group_rows = [r for r in rows if r["group"] == group]
        metric = metric_for(group_rows)
        panel(ax, group_rows, metric)
        ax.set_title(group, fontsize=10)
        ax.set_xlabel("reader")
        if metric != prev_metric:
            ax.set_ylabel(metric_label(metric))
        prev_metric = metric

    fig.tight_layout()
    out = FIGURES / f"{name}.svg"
    fig.savefig(out)
    plt.close(fig)
    return out


def scan_figure(rows: list[dict[str, Any]]) -> Path:
    """The scan groups are one oxyz row each, so they make one panel with
    a column per workload rather than a figure of one-bar panels."""
    rows = sorted(rows, key=lambda r: r["workload"])
    metric = "mean time (ms)"
    values = [value_of(r, metric) for r in rows]

    fig, ax = plt.subplots(figsize=(0.9 + 3.6, 3.1))
    for x, (r, value) in enumerate(zip(rows, values, strict=True)):
        draw_bar(ax, x, value, 0.55, reader_color(r["reader"]), fmt_value(value))
    finish_log_axis(ax, values, [r["workload"] for r in rows])
    ax.set_title("scan (oxyz structural scan)", fontsize=10)
    ax.set_xlabel("workload")
    ax.set_ylabel(metric_label(metric))

    fig.tight_layout()
    out = FIGURES / "scan.svg"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    save = Path(sys.argv[1]) if len(sys.argv) > 1 else newest_save()
    rows = flatten(json.loads(save.read_text()))

    # "ticks" rather than "whitegrid" so tick marks (and the log subticks)
    # actually draw; the grid comes back via rc, paler than either default.
    sns.set_theme(
        style="ticks",
        context="paper",
        rc={
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#e9e9e9",
            "grid.linewidth": 0.7,
            "axes.linewidth": 1.2,
            "axes.edgecolor": "#262626",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "text.color": "#1a1a1a",
            "axes.labelcolor": "#1a1a1a",
            "xtick.color": "#1a1a1a",
            "ytick.color": "#1a1a1a",
        },
    )
    FIGURES.mkdir(exist_ok=True)

    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_scenario.setdefault(r["scenario"], []).append(r)

    outputs = []
    shared = [s for s in SHARED_SCENARIOS if s in by_scenario]
    scan_rows = by_scenario.pop("scan", None)
    for scenario in sorted(set(by_scenario) - set(shared)):
        outputs.append(render_figure(scenario, by_scenario[scenario]))
    if shared:
        shared_rows = [r for s in shared for r in by_scenario[s]]
        shared_groups = [
            g for s in shared for g in sorted({r["group"] for r in by_scenario[s]})
        ]
        outputs.append(render_figure("scenarios", shared_rows, shared_groups))
    if scan_rows:
        outputs.append(scan_figure(scan_rows))

    for out in outputs:
        print(f"wrote {out.relative_to(REPO)} from {save.name}")


if __name__ == "__main__":
    main()
