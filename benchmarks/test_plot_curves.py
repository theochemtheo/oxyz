"""The curve renderer produces the scaling figures from a save's rows."""

from __future__ import annotations

import pytest

# plot imports matplotlib/seaborn, absent from the benchmark-runner env.
plot = pytest.importorskip("plot")


def _row(group, reader, mean, n_atoms, threads=1):
    return {
        "group": group,
        "scenario": group.split("/")[0],
        "workload": group.split("/")[1],
        "reader": reader,
        "threads": threads,
        "output": "numpy frames",
        "mode": "parallel",
        "mean": mean,
        "stddev": 0.0,
        "n_frames": None,
        "n_atoms": n_atoms,
    }


# flatten() leaves the size-sweep param id (e.g. "oxyz-1000") on the reader —
# unlike thread rows, whose -Nt suffix it strips. sweep_size_rows must strip the
# size back off so a reader's points form one curve, so the test rows carry the
# suffix a real save would.
def _size_row(group, base_reader, mean, n_atoms):
    size = group.split("/")[1]
    return _row(group, f"{base_reader}-{size}", mean, n_atoms)


def test_sweep_size_rows_groups_by_reader():
    rows = [
        _size_row("scaling_dataset/1000", "oxyz", 0.01, 7000),
        _size_row("scaling_dataset/10000", "oxyz", 0.1, 70000),
        _size_row("scaling_dataset/1000", "ase", 0.2, 7000),
    ]
    grouped = plot.sweep_size_rows(rows, "scaling_dataset")
    assert [s for s, _, _ in grouped["oxyz"]] == [1000, 10000]
    assert "ase" in grouped


def test_size_curve_figure_writes_svg(tmp_path, monkeypatch):
    monkeypatch.setattr(plot, "FIGURES", tmp_path)
    rows = [
        _size_row("scaling_dataset/1000", "oxyz", 0.01, 7000),
        _size_row("scaling_dataset/10000", "oxyz", 0.1, 70000),
        _size_row("scaling_dataset/1000", "oxyz-to-ase", 0.05, 7000),
        _size_row("scaling_dataset/10000", "oxyz-to-ase", 0.5, 70000),
        _size_row("scaling_dataset/1000", "ase", 0.2, 7000),
        _size_row("scaling_dataset/10000", "ase", 2.0, 70000),
    ]
    out = plot.size_curve_figure(
        rows, "scaling_dataset", "scaling_dataset", "frames", ncores=12
    )
    assert out is not None and out.exists() and out.suffix == ".svg"


def test_thread_curve_figure_writes_svg(tmp_path, monkeypatch):
    monkeypatch.setattr(plot, "FIGURES", tmp_path)
    rows = [
        _row("scaling_threads/dataset", "oxyz", 0.1 / t, 70000, threads=t)
        for t in (1, 2, 4, 8)
    ] + [
        _row("scaling_threads/system", "oxyz", 0.1 / t, 600000, threads=t)
        for t in (1, 2, 4, 8)
    ]
    out = plot.thread_curve_figure(rows)
    assert out is not None and out.exists()
