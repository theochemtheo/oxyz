from __future__ import annotations

import json
from pathlib import Path

import pytest

from oxyz._cli import main

DATA = Path(__file__).parent / "data"


def test_scan_prints_stats_and_schema(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["scan", str(DATA / "two_frame_same_schema.xyz")])
    out = capsys.readouterr().out
    assert code == 0
    assert "frames:      2" in out
    assert "atoms/frame:" in out
    # The schema report follows the stats block.
    assert "frames" in out and "species" in out


def test_scan_empty_file_reports_zero_frames(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    empty = tmp_path / "empty.xyz"
    empty.write_text("")
    code = main(["scan", "--no-schema", str(empty)])
    out = capsys.readouterr().out
    assert code == 0
    assert "frames:      0" in out
    assert "atoms/frame:" not in out


def test_scan_no_schema_omits_schema(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["scan", "--no-schema", str(DATA / "two_frame_same_schema.xyz")])
    out = capsys.readouterr().out
    assert code == 0
    assert "atoms/frame:" in out
    assert "species" not in out


def test_scan_json_shape(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["scan", "--json", str(DATA / "varying_atom_counts.xyz")])
    out = capsys.readouterr().out
    assert code == 0
    payload = json.loads(out)
    assert set(payload) == {"stats", "schema"}
    assert payload["stats"]["n_frames"] == payload["schema"]["n_frames"]
    assert payload["schema"]["columns"]  # full detail present
    assert "variants" in payload["schema"]["columns"][0]
    # The raw per-frame counts stay out of JSON; the derived stats stand in.
    assert "n_atoms" not in payload["schema"]


def test_scan_json_no_schema(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["scan", "--json", "--no-schema", str(DATA / "simple.extxyz")])
    out = capsys.readouterr().out
    assert code == 0
    payload = json.loads(out)
    assert "schema" not in payload
    assert payload["stats"]["n_frames"] == 1


def test_scan_missing_file_exits_one(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["scan", str(DATA / "does_not_exist.extxyz")])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert captured.err.startswith("oxyz:")


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    code = main([])
    assert code == 2
    assert "scan" in capsys.readouterr().err
