"""Reading compressed sources through the Python surface.

The fixtures under `data/compressed/` are compressed twins of
`two_frame_same_schema.xyz`; each codec must read back to the same frames as the
plain file. Random-access strategies, which cannot seek a stream, must refuse a
compressed source with a clear error.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import oxyz

DATA_DIR = Path(__file__).parent / "data"
PLAIN = DATA_DIR / "two_frame_same_schema.xyz"

# (fixture name, whether it is a multi-stream/archive form)
CODECS = [
    "compressed/two_frame.xyz.gz",
    "compressed/two_frame.xyz.zst",
    "compressed/two_frame.xyz.zip",
    "compressed/two_frame.tar.gz",
    "compressed/two_frame.tar",
]


def _frames_equal(a: oxyz.Frame, b: oxyz.Frame) -> bool:
    if a.n_atoms != b.n_atoms or a.columns.keys() != b.columns.keys():
        return False
    return all(np.array_equal(a.columns[k], b.columns[k]) for k in a.columns)


@pytest.fixture
def plain_frames() -> list[oxyz.Frame]:
    return oxyz.read_frames(PLAIN)


@pytest.mark.parametrize("name", CODECS)
def test_read_frames_matches_plain(name: str, plain_frames: list[oxyz.Frame]) -> None:
    frames = oxyz.read_frames(DATA_DIR / name)
    assert len(frames) == len(plain_frames)
    assert all(_frames_equal(a, b) for a, b in zip(frames, plain_frames, strict=True))


@pytest.mark.parametrize("name", CODECS)
def test_serial_and_parallel_agree(name: str) -> None:
    serial = oxyz.read_frames(DATA_DIR / name, threads=1)
    parallel = oxyz.read_frames(DATA_DIR / name, threads=None)
    assert len(serial) == len(parallel) == 2


@pytest.mark.parametrize("name", CODECS)
def test_streaming_and_scan_and_schema(name: str) -> None:
    path = DATA_DIR / name
    assert len(list(oxyz.iter_frames(path))) == 2
    assert oxyz.scan(path).n_frames == 2
    assert oxyz.infer_schema(path).n_frames == 2
    assert oxyz.read_first(path).n_atoms == 2
    assert oxyz.read_batch(path).n_frames == 2
    # A streamed selection works on a compressed source.
    assert oxyz.read_batch(path, [1, 0]).n_frames == 2


def test_concatenated_gzip_reads_all_members() -> None:
    # concat.xyz.gz holds two gzip members; both must be read.
    assert len(oxyz.read_frames(DATA_DIR / "compressed/concat.xyz.gz")) == 4


def test_concatenated_zstd_reads_all_frames() -> None:
    # concat.xyz.zst holds two zstd frames; both must be read.
    assert len(oxyz.read_frames(DATA_DIR / "compressed/concat.xyz.zst")) == 4


def test_member_selects_one_from_archive(plain_frames: list[oxyz.Frame]) -> None:
    frames = oxyz.read_frames(DATA_DIR / "compressed/multi_member.zip", member="a.xyz")
    assert len(frames) == len(plain_frames)


def test_ambiguous_archive_raises_listing_members() -> None:
    with pytest.raises(ValueError, match="multiple extxyz members"):
        oxyz.read_frames(DATA_DIR / "compressed/multi_member.zip")


def test_missing_member_raises() -> None:
    with pytest.raises(ValueError, match="not found"):
        oxyz.read_frames(DATA_DIR / "compressed/multi_member.zip", member="nope.xyz")


def test_member_on_plain_file_raises() -> None:
    with pytest.raises(ValueError, match="non-archive"):
        oxyz.read_frames(PLAIN, member="x.xyz")


def test_compression_override_forces_and_disables() -> None:
    gz = DATA_DIR / "compressed/two_frame.xyz.gz"
    # Forcing the right codec on an unhelpfully-named file works.
    assert len(oxyz.read_frames(gz, compression="gzip")) == 2
    # Disabling decompression makes the gzip bytes fail to parse as text.
    with pytest.raises(oxyz.ParseError):
        oxyz.read_frames(gz, compression="none")


def test_unknown_compression_raises() -> None:
    with pytest.raises(ValueError, match="unknown compression"):
        oxyz.read_frames(PLAIN, compression="lz4")  # ty: ignore[invalid-argument-type]


class TestBatching:
    def test_frames_per_batch_streams_on_compressed(self) -> None:
        gz = DATA_DIR / "compressed/two_frame.xyz.gz"
        batches = list(oxyz.iter_batches(gz, frames_per_batch=1))
        assert len(batches) == 2
        assert all(batch.n_frames == 1 for batch in batches)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"frames_per_batch": 1, "shuffle": True},
            {"atoms_per_batch": 4},
            {"memory_scales_with": "n_atoms", "max_scaler": 10.0},
        ],
    )
    def test_random_access_strategies_raise_on_compressed(
        self, kwargs: dict[str, object]
    ) -> None:
        gz = DATA_DIR / "compressed/two_frame.xyz.gz"
        with pytest.raises(ValueError, match="randomly accessed"):
            list(oxyz.iter_batches(gz, **kwargs))  # ty: ignore[invalid-argument-type]


class TestAse:
    def test_default_index_reads_last_frame(self) -> None:
        ase = pytest.importorskip("oxyz.ase")
        gz = DATA_DIR / "compressed/two_frame.xyz.gz"
        last = ase.read(gz)  # default index -1, needs the in-memory fallback
        assert last.get_global_number_of_atoms() == 2

    def test_slice_and_reverse_index(self) -> None:
        ase = pytest.importorskip("oxyz.ase")
        gz = DATA_DIR / "compressed/two_frame.xyz.gz"
        assert len(ase.read(gz, ":")) == 2
        assert len(ase.read(gz, "::-1")) == 2

    def test_archive_member(self) -> None:
        ase = pytest.importorskip("oxyz.ase")
        atoms = ase.read(DATA_DIR / "compressed/multi_member.zip", 0, member="a.xyz")
        assert atoms.get_global_number_of_atoms() == 2
