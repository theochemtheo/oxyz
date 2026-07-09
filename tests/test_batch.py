from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import oxyz

DATA_DIR = Path(__file__).parent / "data"
VARYING = DATA_DIR / "varying_atom_counts.xyz"


def as_array(value: object) -> np.ndarray:
    """Same ty-limitation shim as test_extxyz.as_array; delete with the canary."""
    assert isinstance(value, np.ndarray)
    return value


def test_sequential_batches_chunk_the_file() -> None:
    batches = list(oxyz.iter_batches(VARYING, frames_per_batch=2))

    assert len(batches) == 2
    first, last = batches
    assert first.n_frames == 2
    assert first.total_atoms == 4
    assert_array_equal(first.offsets, [0, 3, 4])
    assert_array_equal(first.frame_indices, [0, 1])
    assert_array_equal(last.frame_indices, [2])


def test_batch_columns_concatenate_frames() -> None:
    frames = oxyz.read_frames(VARYING)
    (batch,) = oxyz.iter_batches(VARYING, frames_per_batch=3)

    stacked = np.vstack([as_array(frame.columns["pos"]) for frame in frames])
    assert_allclose(as_array(batch.columns["pos"]), stacked)
    assert batch.columns["species"] == [
        s for frame in frames for s in frame.columns["species"]
    ]
    assert_allclose(as_array(batch.metadata["energy"]), [-76.3, -13.6, -31.8])


def test_batch_derived_properties() -> None:
    (batch,) = oxyz.iter_batches(VARYING, frames_per_batch=3)

    assert_array_equal(batch.n_atoms, [3, 1, 2])
    assert_array_equal(batch.ptr, batch.offsets)
    assert_array_equal(batch.batch, [0, 0, 0, 1, 2, 2])


def test_atom_budget_packs_greedily() -> None:
    batches = list(oxyz.iter_batches(VARYING, atoms_per_batch=4))
    assert [list(b.frame_indices) for b in batches] == [[0, 1], [2]]

    # A frame above the budget still gets a batch to itself.
    batches = list(oxyz.iter_batches(VARYING, atoms_per_batch=2))
    assert [list(b.frame_indices) for b in batches] == [[0], [1], [2]]


def test_atom_budget_empty_file_yields_no_batches(tmp_path: Path) -> None:
    empty = tmp_path / "empty.xyz"
    empty.write_text("")
    assert list(oxyz.iter_batches(empty, atoms_per_batch=100)) == []


VARYING_DENSITY = DATA_DIR / "varying_density.extxyz"


def test_memory_binning_by_n_atoms_balances_best_fit() -> None:
    # VARYING has 3/1/2 atoms. Best-fit-decreasing at budget 3 packs the
    # 3-atom frame alone, then the 2- and 1-atom frames together.
    batches = list(
        oxyz.iter_batches(VARYING, memory_scales_with="n_atoms", max_scaler=3)
    )
    assert [sorted(b.frame_indices) for b in batches] == [[0], [1, 2]]
    # No bin exceeds the budget (each here totals 3 atoms).
    assert all(b.total_atoms <= 3 for b in batches)


def test_memory_binning_by_density_separates_dense_from_sparse() -> None:
    # varying_density: three 2-atom frames. Frame 0 is dense (vol 1 -> weight 4),
    # frames 1 and 2 sparse (vol 1000 -> ~0.004). By density the dense frame is
    # isolated and the two sparse frames share a bin; by raw atom count (all
    # weigh 2) the packing differs, proving the density weight is in play.
    by_density = [
        sorted(b.frame_indices)
        for b in oxyz.iter_batches(
            VARYING_DENSITY, memory_scales_with="n_atoms_x_density", max_scaler=4
        )
    ]
    by_atoms = [
        sorted(b.frame_indices)
        for b in oxyz.iter_batches(
            VARYING_DENSITY, memory_scales_with="n_atoms", max_scaler=4
        )
    ]
    assert by_density == [[0], [1, 2]]
    assert by_atoms == [[0, 1], [2]]


def test_density_weight_falls_back_to_atom_count_for_molecules() -> None:
    # A frame with no Lattice has NaN volume; its weight is the atom count, the
    # torch_sim where(volume > 0, n**2/v, n) fallback. Pinned at the weight
    # helper because a bin may not mix Lattice and Lattice-free schemas.
    from oxyz._batch import _memory_weights

    n_atoms = np.array([2, 3], dtype=np.intp)
    volumes = np.array([8.0, np.nan])
    weights = _memory_weights("n_atoms_x_density", n_atoms, volumes)
    assert_allclose(weights, [2 * 2 / 8, 3.0])


def test_memory_binning_isolates_a_frame_over_budget() -> None:
    # A frame whose weight exceeds the budget still gets its own bin.
    batches = list(
        oxyz.iter_batches(VARYING, memory_scales_with="n_atoms", max_scaler=1)
    )
    assert sorted(sorted(b.frame_indices) for b in batches) == [[0], [1], [2]]


def test_memory_binning_preserves_frame_provenance() -> None:
    # Reordered packing still records which file frame each entry came from.
    batches = list(
        oxyz.iter_batches(VARYING, memory_scales_with="n_atoms", max_scaler=3)
    )
    seen = sorted(i for b in batches for i in b.frame_indices)
    assert seen == [0, 1, 2]


def test_memory_binning_requires_max_scaler() -> None:
    with pytest.raises(ValueError, match="max_scaler"):
        list(oxyz.iter_batches(VARYING, memory_scales_with="n_atoms"))


def test_memory_binning_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError, match="memory_scales_with"):
        list(
            oxyz.iter_batches(
                VARYING,
                memory_scales_with="n_edges",  # ty: ignore[invalid-argument-type]
                max_scaler=4,
            )
        )


def test_shuffled_batches_are_seeded_and_partition_the_file() -> None:
    def plan(seed: int) -> list[list[int]]:
        return [
            list(b.frame_indices)
            for b in oxyz.iter_batches(
                VARYING, atoms_per_batch=4, shuffle=True, seed=seed
            )
        ]

    assert plan(0) == plan(0)

    flat = sorted(i for batch in plan(0) for i in batch)
    assert flat == [0, 1, 2]


def test_read_batch_gathers_in_requested_order() -> None:
    frames = oxyz.read_frames(VARYING)
    batch = oxyz.read_batch(VARYING, [2, 0])

    assert_array_equal(batch.frame_indices, [2, 0])
    assert_array_equal(batch.offsets, [0, 2, 5])
    stacked = np.vstack(
        [as_array(frames[2].columns["pos"]), as_array(frames[0].columns["pos"])]
    )
    assert_allclose(as_array(batch.columns["pos"]), stacked)
    assert_allclose(as_array(batch.metadata["energy"]), [-31.8, -76.3])


def test_read_batch_whole_file_concatenates_every_frame() -> None:
    frames = oxyz.read_frames(VARYING)
    batch = oxyz.read_batch(VARYING)

    assert batch.n_frames == len(frames)
    assert_array_equal(batch.frame_indices, range(len(frames)))
    stacked = np.vstack([as_array(frame.columns["pos"]) for frame in frames])
    assert_allclose(as_array(batch.columns["pos"]), stacked)


def test_read_batch_whole_file_matches_threads() -> None:
    serial = oxyz.read_batch(VARYING, threads=1)
    parallel = oxyz.read_batch(VARYING, threads=4)
    assert_array_equal(serial.offsets, parallel.offsets)
    assert_allclose(as_array(serial.columns["pos"]), as_array(parallel.columns["pos"]))


def test_read_batch_whole_file_empty_is_no_frames(tmp_path: Path) -> None:
    empty = tmp_path / "empty.xyz"
    empty.write_text("")
    batch = oxyz.read_batch(empty)
    assert batch.n_frames == 0
    assert_array_equal(batch.offsets, [0])
    assert batch.columns == {}


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"frames_per_batch": 2, "atoms_per_batch": 4},
        {"frames_per_batch": 0},
        {"atoms_per_batch": 0},
        {"frames_per_batch": 2, "seed": 0},
        {"frames_per_batch": 2, "threads": 0},
        {"atoms_per_batch": 4, "memory_scales_with": "n_atoms"},
        {"memory_scales_with": "n_atoms", "max_scaler": 0},
        {"memory_scales_with": "n_atoms", "max_scaler": 4, "shuffle": True},
        {"max_scaler": 4},
    ],
)
def test_invalid_batching_arguments(kwargs) -> None:
    with pytest.raises(ValueError):
        oxyz.iter_batches(VARYING, **kwargs)


def test_zero_threads_is_rejected() -> None:
    # threads=0 would read as "all cores" in rayon; require None or >= 1.
    with pytest.raises(ValueError, match="threads must be"):
        oxyz.read_batch(VARYING, [0], threads=0)
    with pytest.raises(ValueError, match="threads must be"):
        oxyz.read_frames(VARYING, threads=0)


def assert_batches_equal(left: oxyz.Batch, right: oxyz.Batch) -> None:
    assert_array_equal(left.frame_indices, right.frame_indices)
    assert_array_equal(left.offsets, right.offsets)
    assert set(left.columns) == set(right.columns)
    for name, values in right.columns.items():
        if isinstance(values, np.ndarray):
            assert_array_equal(as_array(left.columns[name]), values)
        else:
            assert left.columns[name] == values
    assert set(left.metadata) == set(right.metadata)
    for key, values in right.metadata.items():
        if isinstance(values, np.ndarray):
            assert_array_equal(as_array(left.metadata[key]), values)
        else:
            assert left.metadata[key] == values


def test_threads_never_change_batch_composition() -> None:
    """Same seed, same file: identical batches at any thread count."""

    def batches(threads: int | None) -> list[oxyz.Batch]:
        return list(
            oxyz.iter_batches(
                VARYING, atoms_per_batch=4, shuffle=True, seed=7, threads=threads
            )
        )

    serial = batches(1)
    for threads in (None, 4):
        for left, right in zip(batches(threads), serial, strict=True):
            assert_batches_equal(left, right)


def test_read_batch_ignores_damage_past_the_last_requested_frame(
    tmp_path: Path,
) -> None:
    """The partial-read promise: only the needed file prefix is inspected."""
    path = tmp_path / "tail.extxyz"
    good = "1\nProperties=species:S:1:pos:R:3 energy=-1\nH 0 0 0\n"
    path.write_text(good * 2 + "garbage\n")

    batch = oxyz.read_batch(path, [0, 1])
    assert batch.n_frames == 2

    # A whole-file read must still reject the damage.
    with pytest.raises(ValueError, match="invalid atom count"):
        oxyz.read_frames(path)


def test_read_batch_out_of_range_raises_index_error(tmp_path: Path) -> None:
    path = tmp_path / "short.extxyz"
    path.write_text("1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n")

    with pytest.raises(IndexError, match="frame index 3 out of range"):
        oxyz.read_batch(path, [0, 3])


def test_read_batch_negative_index_raises_index_error() -> None:
    # Negative indices are not supported; they raise the documented
    # IndexError rather than leaking pyo3's OverflowError.
    with pytest.raises(IndexError, match="frame index -1 out of range"):
        oxyz.read_batch(VARYING, [0, -1])


def test_read_batch_threads_are_equivalent() -> None:
    serial = oxyz.read_batch(VARYING, [2, 0, 1], threads=1)
    parallel = oxyz.read_batch(VARYING, [2, 0, 1], threads=4)
    assert_batches_equal(parallel, serial)


def test_sequential_batches_match_across_thread_counts() -> None:
    streamed = list(oxyz.iter_batches(VARYING, frames_per_batch=2, threads=1))
    planned = list(oxyz.iter_batches(VARYING, frames_per_batch=2))
    for left, right in zip(planned, streamed, strict=True):
        assert_batches_equal(left, right)


def test_int_real_metadata_promotes_to_float(tmp_path: Path) -> None:
    path = tmp_path / "promote.extxyz"
    path.write_text(
        "1\nProperties=species:S:1:pos:R:3 energy=-1\nH 0 0 0\n"
        "1\nProperties=species:S:1:pos:R:3 energy=-1.5\nH 0 0 0\n"
    )
    (batch,) = oxyz.iter_batches(path, frames_per_batch=2)

    energy = as_array(batch.metadata["energy"])
    assert energy.dtype == np.float64
    assert_allclose(energy, [-1.0, -1.5])


def test_schema_drift_within_a_batch_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "drift.extxyz"
    path.write_text(
        "1\nProperties=species:S:1:pos:R:3 energy=-1\nH 0 0 0\n"
        "1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n"
    )
    with pytest.raises(ValueError, match="missing metadata"):
        list(oxyz.iter_batches(path, frames_per_batch=2))

    # Batches that never span the drift are still readable.
    assert len(list(oxyz.iter_batches(path, frames_per_batch=1))) == 2


def test_projected_batch_binding_entries_exist():
    import oxyz._rust as _rust

    for name in (
        "read_batch_projected",
        "read_batch_projected_reader",
        "BatchIterProjected",
    ):
        assert hasattr(_rust, name), name
    assert hasattr(_rust.IndexedFrames, "get_batch_projected")


def _mixed_batchable(tmp_path):
    f = tmp_path / "mixed.xyz"
    # frame 1 has charge, frame 2 does not -> unbatchable without projection
    f.write_text(
        "1\nProperties=species:S:1:pos:R:3:charge:R:1\nH 0 0 0 0.5\n"
        "1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n"
    )
    return f


def test_projected_batch_is_readable(tmp_path):
    from oxyz._schema import Kind
    from oxyz._schema_spec import ColumnRule, SchemaSpec

    f = _mixed_batchable(tmp_path)
    spec = SchemaSpec(
        columns=(
            ColumnRule("species", Kind.STR),
            ColumnRule("pos", Kind.REAL, width=3),
            ColumnRule("charge", Kind.REAL, required=False),
        ),
        mode="project",
    )
    batch = oxyz.read_batch(f, schema=spec)
    assert set(batch.columns) == {"species", "pos", "charge"}
    assert batch.n_frames == 2
    assert np.isnan(batch.columns["charge"][1])
    assert batch.frame_indices.tolist() == [0, 1]


def test_warn_drops_unfillable_frame_from_batch(tmp_path):
    import warnings

    from oxyz._schema import Kind
    from oxyz._schema_match import SchemaWarning
    from oxyz._schema_spec import ColumnRule, SchemaSpec

    f = _mixed_batchable(tmp_path)
    # require an int 'id' with no fill -> both frames lack it -> all dropped
    spec = SchemaSpec(
        columns=(ColumnRule("pos", Kind.REAL, width=3), ColumnRule("id", Kind.INT)),
        mode="project",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SchemaWarning)
        batches = list(
            oxyz.iter_batches(f, frames_per_batch=2, schema=spec, conformance="warn")
        )
    # both frames dropped -> the single window is skipped, not yielded empty
    assert batches == []


def test_read_batch_all_dropped_is_empty_not_error(tmp_path):
    from oxyz._schema import Kind
    from oxyz._schema_spec import ColumnRule, SchemaSpec

    f = _mixed_batchable(tmp_path)
    spec = SchemaSpec(
        columns=(ColumnRule("pos", Kind.REAL, width=3), ColumnRule("id", Kind.INT)),
        mode="project",
    )
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        batch = oxyz.read_batch(f, schema=spec, conformance="warn")
    assert batch.n_frames == 0
    assert batch.frame_indices.tolist() == []


def test_batch_mode_without_schema_errors(tmp_path):
    f = _mixed_batchable(tmp_path)
    with pytest.raises(ValueError, match="mode"):
        oxyz.read_batch(f, mode="project")


def _assert_batches_identical(a, b):
    # assert_array_equal treats same-position NaNs as equal, so the NaN fills
    # are compared directly rather than masked away.
    assert a.frame_indices.tolist() == b.frame_indices.tolist()
    assert_array_equal(a.offsets, b.offsets)
    assert sorted(a.columns) == sorted(b.columns)
    for key in a.columns:
        assert_array_equal(np.asarray(a.columns[key]), np.asarray(b.columns[key]))
    for key in a.metadata:
        assert_array_equal(np.asarray(a.metadata[key]), np.asarray(b.metadata[key]))


def test_projected_batch_parity_serial_vs_parallel():
    from oxyz._schema import Kind
    from oxyz._schema_spec import ColumnRule, SchemaSpec

    fixture = DATA_DIR / "mixed_schema_optional_column.xyz"
    spec = SchemaSpec(
        columns=(
            ColumnRule("species", Kind.STR),
            ColumnRule("pos", Kind.REAL, width=3),
            ColumnRule("charge", Kind.REAL, required=False),
        ),
        mode="project",
    )
    _assert_batches_identical(
        oxyz.read_batch(fixture, schema=spec, threads=1),
        oxyz.read_batch(fixture, schema=spec, threads=None),
    )


def test_projected_batch_parity_with_warn_drops(tmp_path):
    import warnings

    from oxyz._schema import Kind
    from oxyz._schema_match import SchemaWarning
    from oxyz._schema_spec import ColumnRule, SchemaSpec

    # frames 0,2 carry id; frame 1 does not -> frame 1 drops under warn. Serial
    # and parallel must drop the *same* frame and agree on survivors/columns.
    f = tmp_path / "drops.xyz"
    f.write_text(
        "1\nProperties=species:S:1:pos:R:3:id:I:1\nH 0 0 0 10\n"
        "1\nProperties=species:S:1:pos:R:3\nH 1 0 0\n"
        "1\nProperties=species:S:1:pos:R:3:id:I:1\nH 2 0 0 12\n"
    )
    spec = SchemaSpec(
        columns=(ColumnRule("pos", Kind.REAL, width=3), ColumnRule("id", Kind.INT)),
        mode="project",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SchemaWarning)
        serial = oxyz.read_batch(f, schema=spec, threads=1, conformance="warn")
        parallel = oxyz.read_batch(f, schema=spec, threads=None, conformance="warn")
    assert serial.frame_indices.tolist() == [0, 2]  # frame 1 dropped
    _assert_batches_identical(serial, parallel)


def test_projected_batch_parity_which_error_wins(tmp_path):
    from oxyz._schema import Kind
    from oxyz._schema_spec import ColumnRule, SchemaSpec

    # A malformed frame 1: the serial and parallel reads must fail with the same
    # error (the parity promise covers which error wins on a bad file).
    f = tmp_path / "bad.xyz"
    f.write_text(
        "1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n"
        "1\nProperties=species:S:1:pos:R:3\nnot-a-count\n"
    )
    spec = SchemaSpec(columns=(ColumnRule("pos", Kind.REAL, width=3),), mode="project")
    errs = {}
    for threads in (1, None):
        try:
            oxyz.read_batch(f, schema=spec, threads=threads)
        except ValueError as exc:  # ParseError/SchemaError are ValueErrors
            errs[threads] = str(exc)
    assert errs[1] == errs[None]


def test_get_batch_projected_empty_indices_errors():
    import oxyz._rust as _rust

    idx = _rust.IndexedFrames(str(DATA_DIR / "mixed_schema_optional_column.xyz"))
    plan = ([("pos", "R", 3, True, float("nan"))], [])
    with pytest.raises(ValueError, match="empty"):
        idx.get_batch_projected([], plan)


def test_build_plan_rejects_multidim_metadata_shape():
    import oxyz._rust as _rust

    plan = ([], [("stress", "R", (2, 3), False, 0.0)])  # 2-D shape unsupported
    with pytest.raises(ValueError, match="dimension"):
        _rust.read_first_frame_projected(
            str(DATA_DIR / "mixed_schema_optional_column.xyz"), plan=plan
        )


def _batch_schema(mode="validate"):
    from oxyz._schema import Kind
    from oxyz._schema_spec import ColumnRule, SchemaSpec

    return SchemaSpec(
        columns=(
            ColumnRule("species", Kind.STR),
            ColumnRule("pos", Kind.REAL, width=3),
            ColumnRule("charge", Kind.REAL, width=1),  # required
        ),
        mode=mode,
    )


def test_validate_batch_matches_frame_reader(tmp_path):
    from oxyz._schema_match import SchemaError

    spec = _batch_schema("validate")
    # uniform-conforming: batch reads fine and honours the schema
    ok = tmp_path / "ok.xyz"
    ok.write_text(
        "1\nProperties=species:S:1:pos:R:3:charge:R:1\nH 0 0 0 0.5\n"
        "1\nProperties=species:S:1:pos:R:3:charge:R:1\nH 1 0 0 -0.5\n"
    )
    assert oxyz.read_batch(ok, schema=spec).n_frames == 2

    # uniform-violating: same frame-indexed SchemaError as read_frames
    bad = tmp_path / "bad.xyz"
    bad.write_text(
        "1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n"
        "1\nProperties=species:S:1:pos:R:3\nH 1 0 0\n"
    )
    with pytest.raises(SchemaError) as batch_exc:
        oxyz.read_batch(bad, schema=spec)
    with pytest.raises(SchemaError) as frame_exc:
        oxyz.read_frames(bad, schema=spec)
    assert str(batch_exc.value) == str(frame_exc.value)
    assert batch_exc.value.frame_index == 0

    # mixed file: a schema error naming the offending frame, not a raw BatchError
    mixed = tmp_path / "mixed.xyz"
    mixed.write_text(
        "1\nProperties=species:S:1:pos:R:3:charge:R:1\nH 0 0 0 0.5\n"
        "1\nProperties=species:S:1:pos:R:3\nH 1 0 0\n"
    )
    with pytest.raises(SchemaError) as exc:
        oxyz.read_batch(mixed, schema=spec)
    assert exc.value.frame_index == 1


def test_validate_batch_warn_keeps_uniform_batch(tmp_path):

    from oxyz._schema_match import SchemaWarning

    spec = _batch_schema("validate")
    bad = tmp_path / "bad.xyz"
    bad.write_text(
        "1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n"
        "1\nProperties=species:S:1:pos:R:3\nH 1 0 0\n"
    )
    with pytest.warns(SchemaWarning, match="charge"):
        batch = oxyz.read_batch(bad, schema=spec, conformance="warn")
    assert batch.n_frames == 2
    assert "charge" not in batch.columns  # validation does not reshape


def test_validate_batch_mode_via_iter_batches(tmp_path):
    from oxyz._schema_match import SchemaError

    spec = _batch_schema("validate")
    bad = tmp_path / "bad.xyz"
    bad.write_text(
        "1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n"
        "1\nProperties=species:S:1:pos:R:3\nH 1 0 0\n"
    )
    with pytest.raises(SchemaError, match="charge"):
        # threads=1 takes the sequential path; default threads the planned path
        list(oxyz.iter_batches(bad, frames_per_batch=1, schema=spec, threads=1))
    with pytest.raises(SchemaError, match="charge"):
        list(oxyz.iter_batches(bad, frames_per_batch=1, schema=spec))


def test_project_batch_enforces_frame_rule(tmp_path):
    from oxyz._schema import Kind
    from oxyz._schema_match import SchemaError
    from oxyz._schema_spec import ColumnRule, FrameRule, SchemaSpec

    f = tmp_path / "counts.xyz"
    f.write_text(
        "1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n"
        "2\nProperties=species:S:1:pos:R:3\nH 0 0 0\nH 1 0 0\n"
    )
    spec = SchemaSpec(
        columns=(
            ColumnRule("species", Kind.STR),
            ColumnRule("pos", Kind.REAL, width=3),
        ),
        frame=FrameRule(n_atoms_min=2),
        mode="project",
    )
    # frame 0 has 1 atom < 2 -> the frame rule fires on the projected batch
    with pytest.raises(SchemaError) as exc:
        oxyz.read_batch(f, schema=spec)
    assert exc.value.frame_index == 0
    assert "n_atoms" in str(exc.value)


def test_project_batch_frame_rule_lattice_warn(tmp_path):
    from oxyz._schema import Kind
    from oxyz._schema_match import SchemaWarning
    from oxyz._schema_spec import ColumnRule, FrameRule, SchemaSpec

    f = tmp_path / "nolattice.xyz"
    f.write_text(
        "1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n"
        "1\nProperties=species:S:1:pos:R:3\nH 1 0 0\n"
    )
    spec = SchemaSpec(
        columns=(
            ColumnRule("species", Kind.STR),
            ColumnRule("pos", Kind.REAL, width=3),
        ),
        frame=FrameRule(lattice_required=True),
        mode="project",
    )
    # No frame carries a Lattice: under warn the batch survives but every frame
    # warns (the lattice-missing frame-rule branch).
    with pytest.warns(SchemaWarning, match="Lattice"):
        batch = oxyz.read_batch(f, schema=spec, conformance="warn")
    assert batch.n_frames == 2


def test_validate_batch_warn_via_sequential_stream(tmp_path):
    """threads=1 iter_batches takes the sequential windowing path; warn keeps
    every frame, so the window is actually assembled and yielded."""
    from oxyz._schema_match import SchemaWarning

    spec = _batch_schema("validate")
    bad = tmp_path / "bad.xyz"
    bad.write_text(
        "1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n"
        "1\nProperties=species:S:1:pos:R:3\nH 1 0 0\n"
        "1\nProperties=species:S:1:pos:R:3\nH 2 0 0\n"
    )
    with pytest.warns(SchemaWarning, match="charge"):
        batches = list(
            oxyz.iter_batches(
                bad, frames_per_batch=2, schema=spec, threads=1, conformance="warn"
            )
        )
    # two windows (2 + 1), both assembled and yielded
    assert [b.n_frames for b in batches] == [2, 1]


def test_validate_batch_selection_and_out_of_range(tmp_path):
    """read_batch validate-mode with an explicit index list, and the
    out-of-range IndexError from the materialise-and-pick path."""
    spec = _batch_schema("validate")
    ok = tmp_path / "ok.xyz"
    ok.write_text(
        "1\nProperties=species:S:1:pos:R:3:charge:R:1\nH 0 0 0 0.5\n"
        "1\nProperties=species:S:1:pos:R:3:charge:R:1\nH 1 0 0 -0.5\n"
    )
    batch = oxyz.read_batch(ok, [1, 0, 1], schema=spec)
    assert batch.frame_indices.tolist() == [1, 0, 1]
    with pytest.raises(IndexError, match="out of range"):
        oxyz.read_batch(ok, [5], schema=spec)
