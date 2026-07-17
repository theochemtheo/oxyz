"""Unit tests for the shared benchmark palette."""

from __future__ import annotations

import pytest

# _style pulls in seaborn, which the benchmark-runner env (run.py) omits; skip
# there so a full run.py collection does not abort on the plot-only modules.
_style = pytest.importorskip("_style")


def test_reader_color_is_stable_per_reader():
    assert _style.reader_color("oxyz") == _style.reader_color("oxyz")
    assert _style.reader_color("ase") == _style.COMPETITOR_COLORS["ase"]


def test_unknown_reader_falls_back():
    assert _style.reader_color("mystery-reader") == _style.FALLBACK_COLOR


def test_reader_order_puts_oxyz_first():
    order = _style.reader_order({"ase", "oxyz", "cextxyz"})
    assert order[0] == "oxyz"
    assert set(order) == {"ase", "oxyz", "cextxyz"}


def test_fmt_value_uses_si_suffixes():
    assert _style.fmt_value(5.56e6) == "5.56M"
    assert _style.fmt_value(140_800) == "141k"


def test_reader_label_shows_function_calls():
    # The all-core rows name the recording machine's core count.
    assert _style.reader_label("oxyz", 12) == "oxyz.read(threads=12)"
    assert _style.reader_label("oxyz-serial") == "oxyz.read(threads=1)"
    assert _style.reader_label("oxyz-to-ase", 12) == "oxyz.ase.read(threads=12)"
    assert _style.reader_label("oxyz-to-ase-serial") == "oxyz.ase.read(threads=1)"
    assert _style.reader_label("ase") == 'ase.io.read(format="extxyz")'
    assert _style.reader_label("cextxyz") == "extxyz.read_dicts"
    assert _style.reader_label("cextxyz-to-ase") == 'ase.io.read(format="cextxyz")'


def test_all_core_labels_fall_back_without_core_count():
    assert _style.reader_label("oxyz") == "oxyz.read()"
    assert _style.reader_label("oxyz-to-ase") == "oxyz.ase.read()"


def test_serial_ase_reader_keeps_an_oxyz_shade():
    # A new oxyz row absent from OXYZ_ORDER would render in the grey
    # fallback colour and sort after the competitors.
    assert _style.reader_color("oxyz-to-ase-serial") != _style.FALLBACK_COLOR
    assert _style.reader_order({"ase", "oxyz-to-ase-serial"})[0] == "oxyz-to-ase-serial"


def test_parallel_oxyz_has_distinct_marker():
    # The all-core oxyz.read line must be tellable from the serial one at a
    # glance, since both are orange.
    assert _style.reader_marker("oxyz") != _style.reader_marker("oxyz-serial")
    assert _style.reader_marker("ase") == _style.reader_marker("oxyz-serial")
