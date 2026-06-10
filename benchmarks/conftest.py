"""Deterministic trajectory fixtures, generated once and cached on disk.

Shapes mirror crates/oxyz-core/benches/parse.rs (many small frames vs few
large frames) so PyO3 binding overhead can be read off against the cargo
bench numbers for the same workload.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

import oxyz._rust


def pytest_configure(config) -> None:
    # Debug-build timings are ~9x off and look like real regressions.
    if oxyz._rust.__build_profile__ != "release":
        raise pytest.UsageError(
            "the installed oxyz._rust extension is a "
            f"{oxyz._rust.__build_profile__!r} build; benchmarks need "
            "release. Run: uv run maturin develop --release"
        )


CACHE_DIR = Path(__file__).parent / ".cache"

# Bump to invalidate cached files when the generator changes.
GENERATOR_VERSION = 1

SPECIES = ["H", "C", "N", "O", "Si"]


def _write_frame(out, rng: random.Random, n_atoms: int) -> None:
    a = 5.0 + 10.0 * rng.random()
    energy = -10.0 * rng.random()

    out.write(f"{n_atoms}\n")
    out.write(
        f'Lattice="{a:.6f} 0.0 0.0 0.0 {a:.6f} 0.0 0.0 0.0 {a:.6f}" '
        f'Properties=species:S:1:pos:R:3:forces:R:3 energy={energy:.6f} pbc="T T T"\n'
    )
    for _ in range(n_atoms):
        species = rng.choice(SPECIES)
        out.write(
            f"{species}"
            f" {a * rng.random():.6f} {a * rng.random():.6f} {a * rng.random():.6f}"
            f" {rng.random() - 0.5:.6f}"
            f" {rng.random() - 0.5:.6f}"
            f" {rng.random() - 0.5:.6f}\n"
        )


def _trajectory_file(
    name: str, n_frames: int, atoms_lo: int, atoms_hi: int, seed: int
) -> Path:
    path = CACHE_DIR / f"{name}-v{GENERATOR_VERSION}.extxyz"
    if path.exists():
        return path

    CACHE_DIR.mkdir(exist_ok=True)
    rng = random.Random(seed)
    # Write-then-rename so an interrupted run never leaves a half file.
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as out:
        for _ in range(n_frames):
            _write_frame(out, rng, rng.randrange(atoms_lo, atoms_hi))
    tmp.rename(path)
    return path


@pytest.fixture(scope="session")
def many_small_frames() -> Path:
    return trajectory_files()["many_small_frames"]


@pytest.fixture(scope="session")
def large_frames() -> Path:
    return trajectory_files()["large_frames"]


def trajectory_files() -> dict[str, Path]:
    return {
        "many_small_frames": _trajectory_file(
            "many_small_frames", 2_000, 16, 64, seed=0x5EED
        ),
        "large_frames": _trajectory_file(
            "large_frames", 4, 100_000, 100_001, seed=0x5EED2
        ),
    }
