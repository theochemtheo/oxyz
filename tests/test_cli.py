from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from oxyz._cli import main
from oxyz._schema_spec import SchemaSpec

if TYPE_CHECKING:
    import pytest

DATA = Path(__file__).parent / "data"


def test_scan_prints_stats_and_schema(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["scan", str(DATA / "two_frame_same_schema.xyz")])
    out = capsys.readouterr().out
    assert code == 0
    assert "frames:      2" in out
    assert "atoms/frame:" in out
    # The schema report follows the stats block.
    assert "frames" in out
    assert "species" in out


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


def test_scan_text_schema_block_is_valid_schema(capsys):
    assert main(["scan", str(DATA / "schema_conformant.extxyz")]) == 0
    out = capsys.readouterr().out
    block = out.split("columns:", 1)
    assert len(block) == 2  # a `columns:` section is present
    spec = SchemaSpec.from_yaml_text("columns:" + block[1])
    names = {rule.name for rule in spec.columns}
    assert {"species", "pos"} <= names


def test_scan_emit_schema_writes_file(tmp_path: Path):
    out = tmp_path / "schema.yaml"
    assert (
        main(
            ["scan", str(DATA / "schema_conformant.extxyz"), "--emit-schema", str(out)]
        )
        == 0
    )
    spec = SchemaSpec.from_file(out)
    assert any(rule.name == "pos" for rule in spec.columns)


def test_scan_emit_schema_json(tmp_path: Path):
    out = tmp_path / "schema.json"
    assert (
        main(
            ["scan", str(DATA / "schema_conformant.extxyz"), "--emit-schema", str(out)]
        )
        == 0
    )
    spec = SchemaSpec.from_file(out)
    assert any(rule.name == "pos" for rule in spec.columns)


def test_emit_schema_with_no_schema_errors(tmp_path, capsys):
    out = tmp_path / "s.yaml"
    code = main(
        [
            "scan",
            str(DATA / "schema_conformant.extxyz"),
            "--emit-schema",
            str(out),
            "--no-schema",
        ]
    )
    assert code == 1  # ValueError -> main() prints and returns 1


def _write_spec(tmp_path: Path) -> Path:
    p = tmp_path / "s.yaml"
    p.write_text(
        "columns:\n"
        "  species: {kind: S}\n"
        "  pos: {kind: R, width: 3}\n"
        # optional, so the conformant/extra files (no magmom) pass; a
        # present-but-mismatched magmom still fires on the drift fixture
        "  magmom: {kind: R, width: 3, required: false}\n"
        "metadata:\n  energy: {kind: R}\n"
    )
    return p


def test_check_conformant_exits_zero(tmp_path, capsys):
    spec = _write_spec(tmp_path)
    code = main(
        ["check", str(DATA / "schema_conformant.extxyz"), "--schema", str(spec)]
    )
    assert code == 0


def test_check_reports_all_and_exits_one(tmp_path, capsys):
    spec = _write_spec(tmp_path)
    code = main(
        ["check", str(DATA / "schema_drift_type.extxyz"), "--schema", str(spec)]
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "magmom" in out
    assert (
        "first at frame 1 (L5)" in out
    )  # frame 1 begins at line 5 (frame 0 = 4 lines)


def test_check_json_includes_line(tmp_path, capsys):
    spec = _write_spec(tmp_path)
    code = main(
        [
            "check",
            str(DATA / "schema_drift_type.extxyz"),
            "--schema",
            str(spec),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    violation = payload["violations"][0]
    assert violation["name"] == "magmom"
    assert violation["first_frame"] == 1
    assert violation["first_line"] == 5


def test_check_extra_only_errors_under_strict(tmp_path):
    spec = _write_spec(tmp_path)
    args = ["check", str(DATA / "schema_extra_column.extxyz"), "--schema", str(spec)]
    assert main(args) == 0  # required: extra column allowed
    assert main([*args, "--conformance", "strict"]) == 1


def test_freeze_writes_project_ready_schema(tmp_path: Path) -> None:
    data = tmp_path / "mixed.xyz"
    data.write_text(
        "1\nProperties=species:S:1:pos:R:3:d_0:R:1:d_1:R:1\nH 0 0 0 0.1 0.2\n"
        "1\nProperties=species:S:1:pos:R:3:d_0:R:1\nH 0 0 0 0.3\n"
    )
    in_schema = tmp_path / "in.yaml"
    in_schema.write_text('mode: project\ncolumns:\n  "d_*": {kind: R}\n')
    out_schema = tmp_path / "out.yaml"
    rc = main(
        ["freeze", str(data), "--schema", str(in_schema), "--out", str(out_schema)]
    )
    assert rc == 0
    text = out_schema.read_text()
    assert "mode: project" in text
    assert "d_0" in text
    assert "d_1" in text
    assert "d_*" not in text  # pattern expanded away


def test_scan_emit_schema_project_is_frozen(tmp_path: Path) -> None:
    data = tmp_path / "mixed.xyz"
    data.write_text(
        "1\nProperties=species:S:1:pos:R:3:d_0:R:1:d_1:R:1\nH 0 0 0 0.1 0.2\n"
        "1\nProperties=species:S:1:pos:R:3:d_0:R:1\nH 0 0 0 0.3\n"
    )
    out = tmp_path / "schema.yaml"
    rc = main(["scan", str(data), "--emit-schema", str(out), "--project"])
    assert rc == 0
    text = out.read_text()
    assert "mode: project" in text
    assert "*" not in text  # no glob families; frozen to literals


def test_freeze_rejects_toml_output(tmp_path: Path, capsys) -> None:
    data = tmp_path / "d.xyz"
    data.write_text("1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n")
    schema = tmp_path / "in.yaml"
    schema.write_text("mode: project\ncolumns:\n  species: {kind: S}\n")
    rc = main(
        [
            "freeze",
            str(data),
            "--schema",
            str(schema),
            "--out",
            str(tmp_path / "o.toml"),
        ]
    )
    assert rc == 1
    assert "TOML" in capsys.readouterr().err


def test_scan_project_without_emit_errors(tmp_path: Path, capsys) -> None:
    data = tmp_path / "d.xyz"
    data.write_text("1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n")
    rc = main(["scan", str(data), "--project"])
    assert rc == 1
    assert "project" in capsys.readouterr().err
