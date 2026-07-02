from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from oxyz import (
    Kind,
    iter_frames,
    read_first,
    read_frames,
)
from oxyz._frames import read_frames_sliced
from oxyz._schema_match import SchemaError, SchemaWarning
from oxyz._schema_spec import ColumnRule, MetadataRule, SchemaSpec

DATA = Path(__file__).parent / "data"

SPEC = SchemaSpec(
    columns=(ColumnRule("species", Kind.STR), ColumnRule("pos", Kind.REAL, width=3)),
    metadata=(MetadataRule("energy", Kind.REAL),),
)


def test_conformant_file_reads_all_frames():
    frames = read_frames(DATA / "schema_conformant.extxyz", schema=SPEC)
    assert len(frames) == 2


def test_no_schema_is_unchanged():
    assert len(read_frames(DATA / "schema_extra_column.extxyz")) == 2


def test_extra_column_strict_raises_with_frame_index():
    with pytest.raises(SchemaError) as excinfo:
        read_frames(
            DATA / "schema_extra_column.extxyz", schema=SPEC, conformance="strict"
        )
    assert excinfo.value.frame_index == 1
    assert excinfo.value.name == "charge"


def test_extra_column_required_allowed():
    frames = read_frames(
        DATA / "schema_extra_column.extxyz", schema=SPEC, conformance="required"
    )
    assert "charge" in frames[1].columns


def test_drift_type_required_raises_at_frame_1():
    with pytest.raises(SchemaError) as excinfo:
        read_frames(
            DATA / "schema_drift_type.extxyz", schema=SPEC, conformance="required"
        )
    assert excinfo.value.frame_index == 1
    assert excinfo.value.name == "pos"


def test_warn_conformance_warns_not_raises():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        frames = read_frames(
            DATA / "schema_drift_type.extxyz", schema=SPEC, conformance="warn"
        )
    assert len(frames) == 2
    assert any(issubclass(w.category, SchemaWarning) for w in caught)


def test_schema_from_yaml_path(tmp_path: Path):
    (tmp_path / "s.yaml").write_text(
        "columns:\n  species: {kind: S}\n  pos: {kind: R, width: 3}\n"
        "metadata:\n  energy: {kind: R}\n"
    )
    frames = read_frames(DATA / "schema_conformant.extxyz", schema=tmp_path / "s.yaml")
    assert len(frames) == 2


def test_read_first_validates_only_frame_zero():
    # frame 0 conforms even though frame 1 drifts, so read_first succeeds
    assert read_first(DATA / "schema_drift_type.extxyz", schema=SPEC) is not None


def test_iter_frames_raises_mid_stream_at_drift():
    it = iter_frames(
        DATA / "schema_drift_type.extxyz", schema=SPEC, conformance="required"
    )
    assert next(it) is not None
    with pytest.raises(SchemaError) as excinfo:
        next(it)
    assert excinfo.value.frame_index == 1


def test_read_frames_sliced_reports_original_index():
    with pytest.raises(SchemaError) as excinfo:
        read_frames_sliced(
            DATA / "schema_drift_type.extxyz",
            slice(1, None),
            schema=SPEC,
            conformance="required",
        )
    assert excinfo.value.frame_index == 1
