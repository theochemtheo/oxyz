"""RESULTS.md sections are chosen by group-name prefix."""

from __future__ import annotations

import report


def sample_groups() -> dict[str, list[dict[str, object]]]:
    return {
        "read_all/many_small_frames": [{"id": 1}],
        "stores/sequential": [{"id": 2}],
        "scaling_dataset/1000": [{"id": 3}],
        "real_data/mad": [{"id": 4}],
        "real_data/mad_scan": [{"id": 5}],
    }


def test_real_data_groups_get_their_own_section():
    buckets = report.bucket_groups(sample_groups())
    assert set(buckets["real_data"]) == {"real_data/mad", "real_data/mad_scan"}


def test_real_data_groups_are_not_plain_reader_groups():
    # Before real_data existed every non-stores/non-scaling group fell into the
    # readers section, which carries no provenance caveat; MAD must not land there.
    buckets = report.bucket_groups(sample_groups())
    assert set(buckets["readers"]) == {"read_all/many_small_frames"}


def test_existing_sections_still_bucket():
    buckets = report.bucket_groups(sample_groups())
    assert set(buckets["stores"]) == {"stores/sequential"}
    assert set(buckets["scaling"]) == {"scaling_dataset/1000"}


def test_every_group_lands_in_exactly_one_bucket():
    buckets = report.bucket_groups(sample_groups())
    placed = [group for bucket in buckets.values() for group in bucket]
    assert sorted(placed) == sorted(sample_groups())


def test_real_data_intro_carries_attribution_and_caveat():
    # CC-BY-4.0 attribution is a licence obligation, and the numbers must not
    # travel without the note that they cannot be regenerated from the repo.
    assert "materialscloud:ak-4p" in report.REAL_DATA_INTRO
    assert "cannot be regenerated" in report.REAL_DATA_INTRO
