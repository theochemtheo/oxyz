"""Unit tests for the shared benchmark palette."""

from __future__ import annotations

import _style


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
