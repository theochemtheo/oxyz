"""End-to-end write benchmarks: serialise an in-memory corpus back to disk.

Record a run: uv run benchmarks/run.py --benchmark-autosave
Smoke only:   uv run benchmarks/run.py --benchmark-disable
Render:       uv run benchmarks/report.py

The corpus is read once from a fixture, then each writer serialises it to
a throwaway path. Only serialisation parallelises — the output stream and
any compression stay serial — so the thread sweep measures the serialise
win, identical output bytes at every count. The `@row(output, mode,
threads)` tags and recorded n_frames/n_atoms drive report.py exactly as
the read suite's do; there are no comparison libraries here (ASE writes
serially through a different contract), so the rows are oxyz-only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import oxyz
from oxyz import Frame

THREAD_SWEEP = (1, 2, 4, 8)


def row(output: str, mode: str, threads: int | None = None):
    def tag(fn):
        info: dict[str, str | int] = {"output": output, "mode": mode}
        if threads is not None:
            info["threads"] = threads
        fn.row_info = info
        return fn

    return tag


def run(benchmark, write, frames: list[Frame], out: Path) -> None:
    """Benchmark `write(frames, out)`, recording the row tags and the work
    done (frames and atoms serialised) so report.py derives throughput."""
    benchmark.extra_info.update(write.row_info)
    benchmark.extra_info.update(
        n_frames=len(frames), n_atoms=sum(f.n_atoms for f in frames)
    )
    benchmark(write, frames, out)


@pytest.fixture(scope="session")
def small_corpus(many_small_frames: Path) -> list[Frame]:
    return oxyz.read_frames(many_small_frames)


@pytest.fixture(scope="session")
def metadata_corpus(metadata_heavy: Path) -> list[Frame]:
    return oxyz.read_frames(metadata_heavy)


@pytest.fixture(scope="session")
def large_corpus(large_frames: Path) -> list[Frame]:
    return oxyz.read_frames(large_frames)


def oxyz_write_with(threads: int, suffix: str):
    output = "extxyz.gz" if suffix.endswith("gz") else "extxyz"

    @row(output, "serial" if threads == 1 else "parallel", threads=threads)
    def write(frames: list[Frame], out: Path) -> None:
        oxyz.write(out.with_suffix(f".{suffix}"), frames, threads=threads)

    return write


PLAIN = [
    pytest.param(oxyz_write_with(t, "extxyz"), id=f"oxyz-{t}t") for t in THREAD_SWEEP
]
GZIP = [
    pytest.param(oxyz_write_with(t, "extxyz.gz"), id=f"oxyz-gz-{t}t")
    for t in THREAD_SWEEP
]


@pytest.mark.benchmark(group="write_all/many_small_frames")
@pytest.mark.parametrize("write", PLAIN)
def test_write_many_small_frames(benchmark, write, small_corpus, tmp_path):
    run(benchmark, write, small_corpus, tmp_path / "out")


@pytest.mark.benchmark(group="write_all/metadata_heavy")
@pytest.mark.parametrize("write", PLAIN)
def test_write_metadata_heavy(benchmark, write, metadata_corpus, tmp_path):
    run(benchmark, write, metadata_corpus, tmp_path / "out")


@pytest.mark.benchmark(group="write_all/large_frames")
@pytest.mark.parametrize("write", PLAIN)
def test_write_large_frames(benchmark, write, large_corpus, tmp_path):
    run(benchmark, write, large_corpus, tmp_path / "out")


@pytest.mark.benchmark(group="write_gzip/many_small_frames")
@pytest.mark.parametrize("write", GZIP)
def test_write_gzip_many_small_frames(benchmark, write, small_corpus, tmp_path):
    run(benchmark, write, small_corpus, tmp_path / "out")
