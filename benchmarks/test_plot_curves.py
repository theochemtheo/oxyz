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


# Real saves keep the size suffix on the reader; sweep_size_rows strips it.
def _size_row(group, base_reader, mean, n_atoms, output="numpy frames"):
    size = group.split("/")[1]
    result = _row(group, f"{base_reader}-{size}", mean, n_atoms)
    result["output"] = output
    return result


def test_sweep_size_rows_groups_by_reader():
    rows = [
        _size_row("scaling_dataset/1000", "oxyz", 0.01, 7000),
        _size_row("scaling_dataset/10000", "oxyz", 0.1, 70000),
        _size_row("scaling_dataset/1000", "ase", 0.2, 7000),
    ]
    grouped = plot.sweep_size_rows(rows, "scaling_dataset")
    assert [s for s, _, _ in grouped["oxyz"]] == [1000, 10000]
    assert "ase" in grouped


@pytest.fixture
def figures(monkeypatch):
    captured = []
    close = plot.plt.close
    monkeypatch.setattr(plot.plt, "close", captured.append)
    yield captured
    for figure in captured:
        close(figure)


def test_scaling_references_match_output_and_do_not_extrapolate(
    tmp_path, monkeypatch, figures
):
    monkeypatch.setattr(plot, "FIGURES", tmp_path)
    rows = [
        _size_row("scaling_dataset/1000", "oxyz", 0.01, 7000),
        _size_row("scaling_dataset/10000", "oxyz", 0.1, 70000),
        _size_row("scaling_dataset/1000", "cextxyz", 0.08, 7000),
        _size_row("scaling_dataset/1000", "oxyz-to-ase", 0.05, 7000, "ase.Atoms"),
        _size_row("scaling_dataset/1000", "ase", 0.2, 7000, "ase.Atoms"),
    ]
    plot.size_curve_figure(
        rows, "scaling_dataset", "scaling_dataset", "frames", ncores=12
    )
    numpy_time, numpy_speed, ase_time, ase_speed = figures[0].axes
    assert len(numpy_time.lines) == 2
    assert len(ase_time.lines) == 2
    native = numpy_speed.lines[0]
    assert list(native.get_xdata()) == [1000]
    assert list(native.get_ydata()) == [8.0]
    converted = ase_speed.lines[0]
    assert list(converted.get_ydata()) == [4.0]


def test_real_data_excludes_scans_and_keeps_output_panels(
    tmp_path, monkeypatch, figures
):
    monkeypatch.setattr(plot, "FIGURES", tmp_path)
    native = _row("real_data/mad", "oxyz", 1.0, 100, threads=12)
    converted = _row("real_data/mad", "oxyz-to-ase", 3.0, 100, threads=4)
    converted["output"] = "ase.Atoms"
    scan = _row("real_data/mad_scan", "oxyz-scan", 0.01, 100)
    scan["output"] = "FrameIndex"
    serial = _row("real_data/mad", "oxyz", 2.0, 100, threads=1)
    iterator = _row("real_data/mad", "oxyz-iter", 1.5, 100)
    extxyz = _row("real_data/mad", "cextxyz", 4.0, 100)
    intermediate = [
        _row("real_data/mad", "oxyz", 0.9, 100, threads=threads)
        for threads in (2, 4, 8)
    ]
    plot.real_data_figure(
        [native, converted, scan, serial, iterator, extxyz, *intermediate],
        provenance="Historical measurement",
    )
    numpy_axis, ase_axis = figures[0].axes
    assert [bar.get_width() for bar in numpy_axis.patches] == [1.0, 1.5, 2.0, 4.0]
    assert [bar.get_width() for bar in ase_axis.patches] == [3.0]
    assert not numpy_axis.collections
    assert not ase_axis.collections
    for axis in (numpy_axis, ase_axis):
        assert axis.get_yticklabels()[0].get_fontfamily() == ["Iosevka"]


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
