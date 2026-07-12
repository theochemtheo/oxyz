"""The scaling groups sort numerically, not lexically, in RESULTS.md."""

from __future__ import annotations

import report


def test_natural_sort_orders_sizes_numerically():
    groups = [
        "scaling_dataset/1000",
        "scaling_dataset/100000",
        "scaling_dataset/10000",
    ]
    assert report.natural_key("scaling_dataset/100000") > report.natural_key(
        "scaling_dataset/10000"
    )
    assert sorted(groups, key=report.natural_key) == [
        "scaling_dataset/1000",
        "scaling_dataset/10000",
        "scaling_dataset/100000",
    ]
