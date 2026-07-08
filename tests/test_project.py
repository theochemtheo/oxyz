from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import pytest

from oxyz._project import compile_projection, effective_mode, enforce_projection
from oxyz._schema import Kind
from oxyz._schema_match import SchemaError, SchemaWarning
from oxyz._schema_spec import ColumnRule, MetadataRule, SchemaSpec

if TYPE_CHECKING:
    from oxyz._rust import DeviationData


def test_effective_mode_override_beats_spec():
    spec = SchemaSpec(mode="project")
    assert effective_mode(spec, None) == "project"
    assert effective_mode(spec, "validate") == "validate"
    assert effective_mode(SchemaSpec(), None) == "validate"


def test_validate_mode_compiles_to_none():
    spec = SchemaSpec(columns=(ColumnRule("pos", Kind.REAL, width=3),))
    assert compile_projection(spec, "validate") is None


def test_project_plan_real_gets_nan_fill_by_default():
    spec = SchemaSpec(columns=(ColumnRule("charge", Kind.REAL, required=False),))
    plan = compile_projection(spec, "project")
    assert plan is not None
    columns, metadata = plan
    name, letter, width, required, fill = columns[0]
    assert (name, letter, width, required) == ("charge", "R", 1, False)
    assert math.isnan(fill)
    assert metadata == []


def test_project_plan_metadata_shape_crosses_as_tuple():
    spec = SchemaSpec(metadata=(MetadataRule("stress", Kind.REAL, shape=(9,)),))
    plan = compile_projection(spec, "project")
    assert plan is not None
    _columns, metadata = plan
    name, letter, shape, required, _fill = metadata[0]
    assert (name, letter, shape, required) == ("stress", "R", (9,), True)


def test_project_plan_optional_int_without_fill_is_spec_error():
    spec = SchemaSpec(columns=(ColumnRule("id", Kind.INT, required=False),))
    with pytest.raises(SchemaError, match="fill"):
        compile_projection(spec, "project")


def test_project_plan_required_int_without_fill_is_allowed():
    # Required non-REAL with no fill is fine: an absent one drops the frame.
    spec = SchemaSpec(columns=(ColumnRule("id", Kind.INT, required=True),))
    plan = compile_projection(spec, "project")
    assert plan is not None
    columns, _metadata = plan
    assert columns[0][4] is None  # no fill


def test_project_plan_pattern_rule_points_at_freeze():
    spec = SchemaSpec(columns=(ColumnRule("d_*", Kind.REAL),))
    with pytest.raises(SchemaError, match="freeze"):
        compile_projection(spec, "project")


def test_enforce_strict_raises_on_first_deviation():
    dev: list[DeviationData] = [
        {
            "axis": "column",
            "name": "pos",
            "deviation": "missing",
            "expected": "R:3",
            "found": None,
        }
    ]
    with pytest.raises(SchemaError) as exc:
        enforce_projection(dev, "required", frame_index=4, dropped=True)
    assert exc.value.frame_index == 4
    assert exc.value.name == "pos"


def test_enforce_warn_drops_and_warns():
    dev: list[DeviationData] = [
        {
            "axis": "column",
            "name": "id",
            "deviation": "missing",
            "expected": "I:1",
            "found": None,
        }
    ]
    with pytest.warns(SchemaWarning, match="id"):
        keep = enforce_projection(dev, "warn", frame_index=0, dropped=True)
    assert keep is False


def test_enforce_clean_frame_kept_silently():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an error
        assert enforce_projection([], "warn", 0, dropped=False) is True


def test_fill_kind_mismatch_is_spec_error():
    spec = SchemaSpec(columns=(ColumnRule("id", Kind.INT, fill="oops"),))
    with pytest.raises(SchemaError, match="fill"):
        compile_projection(spec, "project")
