from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from oxyz import Kind
from oxyz._frames import Frame
from oxyz._schema_match import (
    SchemaError,
    SchemaWarning,
    Violation,
    body,
    compile_spec,
    enforce_frame,
    message,
    resolve,
    validate_frame,
)
from oxyz._schema_spec import ColumnRule, FrameRule, MetadataRule, SchemaSpec


def frame(columns=None, metadata=None, n_atoms=2) -> Frame:
    return Frame(n_atoms=n_atoms, columns=columns or {}, metadata=metadata or {})


def cols():
    return {
        "species": ["H", "O"],
        "pos": np.zeros((2, 3), dtype=np.float64),
        "charge": np.zeros(2, dtype=np.float64),
    }


def test_conformant_columns_have_no_violations():
    spec = SchemaSpec(
        columns=(
            ColumnRule("species", Kind.STR),
            ColumnRule("pos", Kind.REAL, width=3),
            ColumnRule("charge", Kind.REAL),
        )
    )
    assert validate_frame(frame(cols()), compile_spec(spec), "required") == []


def test_missing_required_column_flagged_at_all_levels():
    spec = SchemaSpec(
        columns=(
            ColumnRule("species", Kind.STR),
            ColumnRule("pos", Kind.REAL, width=3),
            ColumnRule("charge", Kind.REAL),
            ColumnRule("forces", Kind.REAL, width=3),
        )
    )
    for level in ("strict", "required", "warn"):
        result = validate_frame(frame(cols()), compile_spec(spec), level)
        assert result == [Violation("column", "forces", "missing", "R:3", None)]


def test_optional_missing_column_is_fine():
    spec = SchemaSpec(
        columns=(
            ColumnRule("species", Kind.STR),
            ColumnRule("pos", Kind.REAL, width=3),
            ColumnRule("charge", Kind.REAL),
            ColumnRule("forces", Kind.REAL, width=3, required=False),
        )
    )
    assert validate_frame(frame(cols()), compile_spec(spec), "strict") == []


def test_width_mismatch_flagged():
    spec = SchemaSpec(columns=(ColumnRule("pos", Kind.REAL, width=2),))
    result = validate_frame(frame(cols()), compile_spec(spec), "required")
    assert result == [Violation("column", "pos", "mismatch", "R:2", "R:3")]


def test_kind_mismatch_flagged():
    spec = SchemaSpec(columns=(ColumnRule("charge", Kind.INT),))
    result = validate_frame(frame(cols()), compile_spec(spec), "required")
    assert result == [Violation("column", "charge", "mismatch", "I:1", "R:1")]


def test_extra_column_flagged_only_under_strict():
    spec = SchemaSpec(
        columns=(ColumnRule("species", Kind.STR), ColumnRule("pos", Kind.REAL, width=3))
    )
    compiled = compile_spec(spec)
    assert validate_frame(frame(cols()), compiled, "required") == []
    assert validate_frame(frame(cols()), compiled, "strict") == [
        Violation("column", "charge", "extra", None, "R:1")
    ]


def test_glob_count_exact():
    columns = {f"descriptor_{i}": np.zeros(2, dtype=np.float64) for i in range(5)}
    spec = SchemaSpec(columns=(ColumnRule("descriptor_*", Kind.REAL, count=5),))
    assert validate_frame(frame(columns), compile_spec(spec), "strict") == []


def test_glob_count_wrong_flagged():
    columns = {f"descriptor_{i}": np.zeros(2, dtype=np.float64) for i in range(4)}
    spec = SchemaSpec(columns=(ColumnRule("descriptor_*", Kind.REAL, count=5),))
    result = validate_frame(frame(columns), compile_spec(spec), "required")
    assert result == [Violation("column", "descriptor_*", "count", "5", "4")]


def test_regex_entry_matches():
    columns = {"md_step": np.zeros(2, dtype=np.int64)}
    spec = SchemaSpec(columns=(ColumnRule("re:^md_.*$", Kind.INT, min=1),))
    assert validate_frame(frame(columns), compile_spec(spec), "strict") == []


def test_string_column_width_from_list_of_lists():
    columns = {"labels": [["a", "b"], ["c", "d"]]}
    spec = SchemaSpec(columns=(ColumnRule("labels", Kind.STR, width=2),))
    assert validate_frame(frame(columns), compile_spec(spec), "strict") == []


def test_metadata_scalar_and_array_conformant():
    meta = {"energy": -1.5, "stress": np.zeros(9, dtype=np.float64)}
    spec = SchemaSpec(
        metadata=(
            MetadataRule("energy", Kind.REAL),
            MetadataRule("stress", Kind.REAL, shape=(9,)),
        )
    )
    assert validate_frame(frame(cols(), meta), compile_spec(spec), "required") == []


def test_metadata_bool_not_confused_with_int():
    spec = SchemaSpec(metadata=(MetadataRule("periodic", Kind.BOOL),))
    result = validate_frame(
        frame(cols(), {"periodic": True}), compile_spec(spec), "required"
    )
    assert result == []


def test_metadata_shape_mismatch_flagged():
    meta = {"stress": np.zeros(6, dtype=np.float64)}
    spec = SchemaSpec(metadata=(MetadataRule("stress", Kind.REAL, shape=(9,)),))
    result = validate_frame(frame(cols(), meta), compile_spec(spec), "required")
    assert result == [Violation("metadata", "stress", "mismatch", "R[9]", "R[6]")]


def test_metadata_missing_required_flagged():
    spec = SchemaSpec(metadata=(MetadataRule("energy", Kind.REAL),))
    result = validate_frame(frame(cols(), {}), compile_spec(spec), "required")
    assert result == [Violation("metadata", "energy", "missing", "R", None)]


def test_frame_n_atoms_bounds():
    spec = SchemaSpec(frame=FrameRule(n_atoms_min=3))
    result = validate_frame(frame(cols(), n_atoms=2), compile_spec(spec), "required")
    assert result == [Violation("frame", "n_atoms", "mismatch", "[3, ]", "2")]


def test_frame_lattice_required():
    spec = SchemaSpec(frame=FrameRule(lattice_required=True))
    result = validate_frame(frame(cols(), {}), compile_spec(spec), "required")
    assert result == [Violation("frame", "Lattice", "missing", "required", None)]


def test_body_strings():
    assert body(Violation("column", "forces", "mismatch", "R:3", "R:1")) == (
        "expected R:3, found R:1"
    )
    assert (
        body(Violation("metadata", "energy", "missing", "R", None))
        == "missing (required)"
    )
    assert (
        body(Violation("column", "charge", "extra", None, "R:1")) == "unexpected (R:1)"
    )
    assert body(Violation("column", "descriptor_*", "count", "5", "4")) == (
        "expected 5, found 4"
    )


def test_message_prefixes_frame_and_name():
    v = Violation("column", "forces", "mismatch", "R:3", "R:1")
    assert message(v, 42) == "frame 42: column 'forces': expected R:3, found R:1"


def test_enforce_frame_raises_first_violation_strict():
    spec = SchemaSpec(columns=(ColumnRule("forces", Kind.REAL, width=3),))
    with pytest.raises(SchemaError) as excinfo:
        enforce_frame(frame(cols()), compile_spec(spec), "required", 7)
    assert excinfo.value.frame_index == 7
    assert excinfo.value.name == "forces"
    assert "frame 7" in str(excinfo.value)


def test_enforce_frame_warns_under_warn():
    # Columns must be fully declared so the only deviation is the missing
    # "forces" column — `_validate_columns` also flags undeclared present
    # columns as "extra" under "warn" (see Task 3), which would otherwise
    # produce extra warnings unrelated to what this test is checking.
    spec = SchemaSpec(
        columns=(
            ColumnRule("species", Kind.STR),
            ColumnRule("pos", Kind.REAL, width=3),
            ColumnRule("charge", Kind.REAL),
            ColumnRule("forces", Kind.REAL, width=3),
        )
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        enforce_frame(frame(cols()), compile_spec(spec), "warn", 3)
    assert len(caught) == 1
    assert issubclass(caught[0].category, SchemaWarning)


def test_resolve_accepts_spec_object_and_path(tmp_path: Path):
    spec = SchemaSpec(columns=(ColumnRule("pos", Kind.REAL, width=3),))
    assert resolve(spec) == compile_spec(spec)
    (tmp_path / "s.yaml").write_text("columns:\n  pos: {kind: R, width: 3}\n")
    assert resolve(tmp_path / "s.yaml") == compile_spec(spec)
