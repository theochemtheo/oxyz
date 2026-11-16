# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib>=3.8",
#     "seaborn>=0.13",
#     "fonttools[woff]>=4.0",
# ]
# ///
"""Render benchmarks/figures/ from a saved pytest-benchmark run.

Usage:
    uv run benchmarks/plot.py [SAVE.json] [--output-dir DIR]

Without an argument, the newest save under .benchmarks/ is used. Pass an
explicit save for publication; --only real_data scaling_dataset selects figures.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

# Supplied by this script's inline metadata, not the venv ty checks under.
import seaborn as sns
from _style import (  # ty: ignore[import-not-found]  # sibling module, run as a script
    fmt_value,
    reader_color,
    reader_label,
    reader_marker,
    reader_order,
)
from fontTools.ttLib import TTFont
from matplotlib import font_manager
from matplotlib.ticker import LogLocator, NullFormatter

REPO = Path(__file__).resolve().parent.parent
FIGURES = Path(__file__).resolve().parent / "figures"

# Small scenarios share a figure; the rest get one panel per group.
SHARED_SCENARIOS = ["selective", "batches", "read_first", "read_last"]


def configure_style() -> None:
    """Use the site's bundled fonts without depending on installed fonts."""
    fonts = REPO / "docs" / "fonts"
    for weight in ("Regular", "Bold"):
        font_manager.fontManager.addfont(
            fonts / "atkinson" / f"AtkinsonHyperlegible-{weight}.ttf"
        )
    cache = REPO / ".cache" / "benchmark-fonts"
    cache.mkdir(parents=True, exist_ok=True)
    for weight in ("Regular", "Bold"):
        source = fonts / "iosevka" / f"Iosevka-{weight}.woff2"
        target = cache / f"Iosevka-{weight}.ttf"
        if not target.exists() or target.stat().st_mtime < source.stat().st_mtime:
            with TTFont(source) as font:
                font.flavor = None
                font.save(target)
        font_manager.fontManager.addfont(target)
    sns.set_theme(
        style="ticks",
        context="paper",
        font="Atkinson Hyperlegible",
        rc={
            "font.monospace": ["Iosevka"],
            "font.size": 10,
            "axes.grid": True,
            "grid.color": "#e9e9e9",
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "text.color": "#1a1a1a",
            "axes.labelcolor": "#1a1a1a",
            "svg.fonttype": "path",
            "svg.hashsalt": "oxyz-benchmarks",
        },
    )


def save_comparison(
    fig,
    name: str,
    title: str,
    notes: str,
    provenance: str,
    *,
    output_dir: Path | None = None,
) -> Path:
    fig.suptitle(title, x=0.025, ha="left", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 1), h_pad=3, w_pad=3)
    if provenance:
        fig.text(
            0.025,
            0.025,
            provenance.splitlines()[0],
            fontsize=8,
            color="#555555",
            va="top",
        )
    out = (output_dir or FIGURES) / f"{name}.svg"
    fig.savefig(
        out,
        metadata={
            "Title": title,
            "Description": f"{notes}\n{provenance}",
            "Date": None,
        },
    )
    out.write_text(
        "\n".join(line.rstrip() for line in out.read_text().splitlines()) + "\n"
    )
    plt.close(fig)
    return out


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
        # Swept rows carry a -Nt suffix; stripping it gives a bar per thread count.
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
    # matplotlib would otherwise zoom a log axis onto a sliver.
    ax.set_ylim(min(values) / 4, max(values) * top)
    ax.set_xticks(range(len(tick_labels)))
    ax.set_xticklabels(
        tick_labels,
        rotation=20,
        ha="right",
        rotation_mode="anchor",
        fontsize=8,
        fontfamily="Iosevka",
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
        tick_labels.append(reader_label(reader, threads=bars[0]["threads"]))
        width = slot / len(bars)
        has_sweep = has_sweep or len(bars) > 1
        for i, r in enumerate(bars):
            value = value_of(r, metric)
            values.append(value)
            pos = x - slot / 2 + (i + 0.5) * width
            label = fmt_value(value)
            if len(bars) > 1:
                label = f"{reader_label(reader, threads=r['threads'])}\n{label}"
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
    name: str,
    rows: list[dict[str, Any]],
    groups: list[str] | None = None,
    *,
    output_dir: Path | None = None,
) -> Path:
    """One panel per group, each with its own metric."""
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
    out = (output_dir or FIGURES) / f"{name}.svg"
    fig.savefig(out)
    plt.close(fig)
    return out


def real_data_figure(
    rows: list[dict[str, Any]],
    provenance: str = "",
    *,
    output_dir: Path | None = None,
) -> Path | None:
    """Compare full MAD reads without mixing output contracts or scans."""
    rows = [
        r
        for r in rows
        if r["group"] == "real_data/mad"
        and (r["reader"] != "oxyz" or r["threads"] in (1, 12))
    ]
    if not rows:
        return None
    contracts = [
        (output, title)
        for output, title in (
            ("numpy frames", "NumPy frames"),
            ("ase.Atoms", "ASE objects"),
        )
        if any(r["output"] == output for r in rows)
    ]
    fig, axes = plt.subplots(
        len(contracts), 1, figsize=(10.5, 6.7), squeeze=False, sharex=True
    )
    for ax, (output, title) in zip(axes[:, 0], contracts, strict=True):
        panel_rows = sorted(
            (r for r in rows if r["output"] == output), key=lambda r: r["mean"]
        )
        values = [r["mean"] for r in panel_rows]
        ax.barh(
            range(len(panel_rows)),
            values,
            color=[reader_color(r["reader"]) for r in panel_rows],
            height=0.65,
        )
        ax.set_yticks(
            range(len(panel_rows)),
            [reader_label(r["reader"], threads=r["threads"]) for r in panel_rows],
            fontfamily="Iosevka",
            fontsize=10,
        )
        ax.invert_yaxis()
        ax.set_xlim(0, max(r["mean"] for r in rows) * 1.16)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Full-trajectory read time (s)")
        ax.grid(axis="y", visible=False)
        for i, value in enumerate(values):
            ax.annotate(
                f"{value:.2f} s",
                (value, i),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
            )
    return save_comparison(
        fig,
        "real_data",
        "Reading the MAD dataset",
        "",
        provenance,
        output_dir=output_dir,
    )


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
        # Strip the size suffix so a reader's points form one curve.
        reader = r["reader"].removesuffix(f"-{size}")
        by_reader.setdefault(reader, []).append((size, r["mean"], r["n_atoms"]))
    for points in by_reader.values():
        points.sort()
    return by_reader


def size_curve_figure(
    rows: list[dict[str, Any]],
    prefix: str,
    name: str,
    xlabel: str,
    ncores: int | None = None,
    provenance: str = "",
    *,
    output_dir: Path | None = None,
) -> Path | None:
    """Plot time and same-output speedup without extrapolating references."""
    selected = [r for r in rows if r["group"].startswith(prefix + "/")]
    contracts = [
        (output, title, baseline)
        for output, title, baseline in (
            ("numpy frames", "NumPy frames", "cextxyz"),
            ("ase.Atoms", "ASE objects", "ase"),
        )
        if any(r["output"] == output for r in selected)
    ]
    if not contracts:
        return None
    fig, axes = plt.subplots(
        len(contracts), 2, figsize=(11.8, 4.0 * len(contracts) + 1.2), squeeze=False
    )
    for (ax_t, ax_s), (output, title, reference) in zip(axes, contracts, strict=True):
        output_rows = [r for r in selected if r["output"] == output]
        by_reader = sweep_size_rows(output_rows, prefix)
        baseline = {s: m for s, m, _ in by_reader.get(reference, [])}
        for reader in reader_order(set(by_reader)):
            pts = by_reader[reader]
            recorded = next(
                r
                for r in output_rows
                if r["reader"].removesuffix(f"-{r['workload']}") == reader
            )
            label = reader_label(reader, ncores, threads=recorded["threads"])
            ax_t.loglog(
                [s for s, _, _ in pts],
                [m for _, m, _ in pts],
                marker=reader_marker(reader),
                label=label,
                color=reader_color(reader),
            )
            if reader != reference:
                shared = [(s, baseline[s] / m) for s, m, _ in pts if s in baseline]
                if shared:
                    ax_s.semilogx(
                        [s for s, _ in shared],
                        [speedup for _, speedup in shared],
                        marker=reader_marker(reader),
                        label=label,
                        color=reader_color(reader),
                    )
        ax_t.set_xlabel(xlabel)
        ax_t.set_ylabel("Full-trajectory read time (s)")
        ax_t.set_title(title, loc="left")
        ax_t.grid(True, which="both", alpha=0.3)
        ax_t.legend(prop={"family": "Iosevka", "size": 8}, loc="upper left")
        if baseline:
            ax_s.axhline(
                1,
                color=reader_color(reference),
                linewidth=1,
                linestyle="--",
                label=reader_label(reference),
            )
            ax_s.legend(
                title="Calls · dashed line is reference",
                prop={"family": "Iosevka", "size": 8},
                loc="upper right",
            )
        else:
            ax_s.text(
                0.5,
                0.5,
                "No matching reference measurements",
                transform=ax_s.transAxes,
                ha="center",
            )
        ax_s.set_xlabel(xlabel)
        ax_s.set_ylabel("Speedup over reference (×)")
        ax_s.set_title("Relative read speed", loc="left")
        ax_s.grid(True, which="both", alpha=0.3)
    title = (
        "Reading increasingly large datasets"
        if prefix == "scaling_dataset"
        else "Reading increasingly large systems"
    )
    return save_comparison(
        fig,
        name,
        title,
        "Time panels: logarithmic axes; lower is better. "
        "Speedup panels: higher is better.\n"
        "References match the output type. "
        "Speedups stop where reference measurements stop.",
        provenance,
        output_dir=output_dir,
    )


def thread_curve_figure(
    rows: list[dict[str, Any]], *, output_dir: Path | None = None
) -> Path | None:
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
        ax.plot(
            threads,
            thru,
            marker=reader_marker("oxyz"),
            color=reader_color("oxyz"),
            label="oxyz.read",
        )
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
    out = (output_dir or FIGURES) / "scaling_threads.svg"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("save", nargs="?", type=Path)
    parser.add_argument("--only", nargs="+", help="Figure names to generate")
    parser.add_argument("--output-dir", type=Path, default=FIGURES)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    save = args.save if args.save is not None else newest_save()
    data = json.loads(save.read_text())
    rows = [r for r in flatten(data) if r["reader"] != "oxyz-scan"]
    ncores = data["machine_info"].get("cpu", {}).get("count")
    machine = data["machine_info"]
    cpu = machine.get("cpu", {}).get("brand_raw", machine.get("processor", "?"))
    revision = data.get("commit_info", {})
    dirty = " (uncommitted changes)" if revision.get("dirty") else ""
    version = machine.get("package_versions", {}).get("oxyz", "?")
    provenance = (
        f"Measured {data.get('datetime', '?')[:10]} · revision "
        f"{revision.get('id', '?')[:9]}{dirty} · {cpu} · "
        f"CPython {machine.get('python_version', '?')} · oxyz {version}\n"
        f"Source: {save.name}"
    )
    configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    def wanted(name: str) -> bool:
        return args.only is None or name in args.only

    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if r["scenario"] not in (
            "scaling_dataset",
            "scaling_system",
            "scaling_threads",
            "real_data",
        ):
            by_scenario.setdefault(r["scenario"], []).append(r)
    shared = [s for s in SHARED_SCENARIOS if s in by_scenario]
    outputs: list[Path | None] = [
        render_figure(scenario, by_scenario[scenario], output_dir=output_dir)
        for scenario in sorted(set(by_scenario) - set(shared))
        if wanted(scenario)
    ]
    if shared and wanted("scenarios"):
        shared_rows = [r for s in shared for r in by_scenario[s]]
        shared_groups = [
            g for s in shared for g in sorted({r["group"] for r in by_scenario[s]})
        ]
        outputs.append(
            render_figure(
                "scenarios", shared_rows, shared_groups, output_dir=output_dir
            )
        )
    if wanted("real_data"):
        outputs.append(real_data_figure(rows, provenance, output_dir=output_dir))
    for prefix, xlabel in (
        ("scaling_dataset", "Number of frames"),
        ("scaling_system", "Atoms per frame"),
    ):
        if wanted(prefix):
            outputs.append(
                size_curve_figure(
                    rows,
                    prefix,
                    prefix,
                    xlabel,
                    ncores,
                    provenance,
                    output_dir=output_dir,
                )
            )
    if wanted("scaling_threads"):
        outputs.append(thread_curve_figure(rows, output_dir=output_dir))

    for out in [o for o in outputs if o is not None]:
        print(f"wrote {out} from {save.name}")


if __name__ == "__main__":
    main()
