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
from _style import (
    fmt_value,
    reader_color,
    reader_order,
)
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


def sweep_size_rows(
    rows: list[dict[str, Any]], prefix: str
) -> dict[str, list[tuple[int, float, float]]]:
    """reader -> sorted [(size, mean, n_atoms)] for one sweep family. The size
    is parsed from the group suffix (e.g. scaling_dataset/10000 -> 10000)."""
    by_reader: dict[str, list[tuple[int, float, float]]] = {}
    for r in rows:
        if not r["group"].startswith(prefix + "/"):
            continue
        size = int(r["group"].split("/")[1])
        # Size-sweep param ids carry the size (e.g. "oxyz-1000"); strip it back
        # to the base reader so a reader's points across sizes form one curve.
        reader = r["reader"].removesuffix(f"-{size}")
        by_reader.setdefault(reader, []).append((size, r["mean"], r["n_atoms"]))
    for reader in by_reader:
        by_reader[reader].sort()
    return by_reader


def size_curve_figure(
    rows: list[dict[str, Any]], prefix: str, name: str, xlabel: str
) -> Path | None:
    """Two panels: read time vs size (loglog) and speedup vs ase (semilogx),
    overlaying every reader — the extxyz plot_bench shape."""
    by_reader = sweep_size_rows(rows, prefix)
    if not by_reader:
        return None

    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(9.5, 4.0))
    baseline = {s: m for s, m, _ in by_reader.get("ase", [])}

    for reader in reader_order(set(by_reader)):
        pts = by_reader[reader]
        sizes = [s for s, _, _ in pts]
        times = [m for _, m, _ in pts]
        ax_t.loglog(sizes, times, marker="o", label=reader, color=reader_color(reader))
        if baseline and reader != "ase":
            shared = [(s, baseline[s] / m) for s, m, _ in pts if s in baseline]
            if shared:
                ax_s.semilogx(
                    [s for s, _ in shared],
                    [sp for _, sp in shared],
                    marker="o",
                    label=reader,
                    color=reader_color(reader),
                )

    ax_t.set_xlabel(xlabel)
    ax_t.set_ylabel("read time (s) — lower is better")
    ax_t.set_title(f"{name.replace('scaling_', '')}: time vs size")
    ax_t.grid(True, which="both", alpha=0.3)
    ax_t.legend(fontsize=8)

    ax_s.axhline(1, color="gray", linewidth=0.8, linestyle="--")
    ax_s.set_xlabel(xlabel)
    ax_s.set_ylabel("speedup over ase.io")
    ax_s.set_title("speedup vs size")
    ax_s.grid(True, which="both", alpha=0.3)
    ax_s.legend(fontsize=8)

    fig.tight_layout()
    out = FIGURES / f"{name}.svg"
    fig.savefig(out)
    plt.close(fig)
    return out


def thread_curve_figure(rows: list[dict[str, Any]]) -> Path | None:
    """Throughput (atoms/s) vs thread count, one panel per family."""
    families = [
        ("scaling_threads/dataset", "dataset (many small frames)"),
        ("scaling_threads/system", "system (few large frames)"),
    ]
    present = [(g, t) for g, t in families if any(r["group"] == g for r in rows)]
    if not present:
        return None

    fig, axes = plt.subplots(
        1, len(present), figsize=(4.6 * len(present), 4.0), squeeze=False
    )
    for ax, (group, title) in zip(axes[0], present, strict=True):
        pts = sorted(
            (r["threads"], r["n_atoms"] / r["mean"])
            for r in rows
            if r["group"] == group
        )
        threads = [t for t, _ in pts]
        thru = [v for _, v in pts]
        ax.plot(threads, thru, marker="o", color=reader_color("oxyz"))
        # Linear-scaling reference from the 1-thread point.
        if thru:
            ideal = [thru[0] * t / threads[0] for t in threads]
            ax.plot(
                threads,
                ideal,
                linestyle="--",
                color="gray",
                linewidth=0.8,
                label="linear",
            )
        ax.set_xlabel("threads")
        ax.set_ylabel("atoms/s — higher is better")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    out = FIGURES / "scaling_threads.svg"
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

    # The scaling scenarios render as curves, not bars; keep them out of the
    # bar loop.
    for scaling in ("scaling_dataset", "scaling_system", "scaling_threads"):
        by_scenario.pop(scaling, None)

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

    outputs.append(
        size_curve_figure(rows, "scaling_dataset", "scaling_dataset", "frames")
    )
    outputs.append(
        size_curve_figure(rows, "scaling_system", "scaling_system", "atoms per frame")
    )
    outputs.append(thread_curve_figure(rows))
    outputs = [o for o in outputs if o is not None]

    for out in outputs:
        print(f"wrote {out.relative_to(REPO)} from {save.name}")


if __name__ == "__main__":
    main()
