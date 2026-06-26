"""Deterministic trajectory fixtures, generated once and cached on disk.

Shapes mirror crates/oxyz-core/benches/parse.rs (many small frames vs few
large frames) so PyO3 binding overhead can be read off against the cargo
bench numbers for the same workload.

The store-comparison suite adds a much larger dataset (`store_dataset`,
100k frames of 64 atoms, ~0.4 GB of text): random reads on the 2k-frame
fixtures measure fixed overheads, not the stores. Each store backend
ingests it once into benchmarks/.cache/ as well — expect ~2 GB of cache
and a slow first run.
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


def pytest_benchmark_update_machine_info(config, machine_info) -> None:
    """Record library versions in the saved JSON, so report.py describes
    the environment the numbers came from rather than the one it runs in."""
    import importlib.metadata as metadata

    versions = {}
    for dist in (
        "oxyz",
        "numpy",
        "ase",
        "extxyz",
        "ase-extxyz",
        "atompack-db",
        "lmdb",
        "torch",
        "metatomic-torch",
    ):
        try:
            versions[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            versions[dist] = None
    machine_info["package_versions"] = versions


CACHE_DIR = Path(__file__).parent / ".cache"

# Bump to invalidate cached files when the generator changes.
GENERATOR_VERSION = 2

SPECIES = ["H", "C", "N", "O", "Si"]


def _floats(rng: random.Random, n: int) -> str:
    return " ".join(f"{rng.random() - 0.5:.6f}" for _ in range(n))


def _write_atom_lines(out, rng: random.Random, n_atoms: int, a: float) -> None:
    for _ in range(n_atoms):
        species = rng.choice(SPECIES)
        out.write(
            f"{species}"
            f" {a * rng.random():.6f} {a * rng.random():.6f} {a * rng.random():.6f}"
            f" {_floats(rng, 3)}\n"
        )


def _write_frame(out, rng: random.Random, n_atoms: int) -> None:
    a = 5.0 + 10.0 * rng.random()
    energy = -10.0 * rng.random()

    out.write(f"{n_atoms}\n")
    out.write(
        f'Lattice="{a:.6f} 0.0 0.0 0.0 {a:.6f} 0.0 0.0 0.0 {a:.6f}" '
        f'Properties=species:S:1:pos:R:3:forces:R:3 energy={energy:.6f} pbc="T T T"\n'
    )
    _write_atom_lines(out, rng, n_atoms, a)


def _write_metadata_heavy_frame(out, rng: random.Random, n_atoms: int) -> None:
    """A comment line shaped like real DFT output: ~16 keys of every kind
    (scalars, quoted float arrays, bracket arrays, booleans, sentences)."""
    a = 5.0 + 10.0 * rng.random()
    out.write(f"{n_atoms}\n")
    out.write(
        f'Lattice="{a:.6f} 0.0 0.0 0.0 {a:.6f} 0.0 0.0 0.0 {a:.6f}" '
        f"Properties=species:S:1:pos:R:3:forces:R:3 "
        f"energy={-10.0 * rng.random():.8f} "
        f"free_energy={-10.0 * rng.random():.8f} "
        f'stress="{_floats(rng, 9)}" '
        f'virial="{_floats(rng, 9)}" '
        f'dipole="{_floats(rng, 3)}" '
        f"config_type=bulk_amorphous "
        f"md_step={rng.randrange(100_000)} "
        f"time={1000.0 * rng.random():.4f} "
        f"temperature={300.0 + 50.0 * rng.random():.4f} "
        f"pressure={rng.random():.6f} "
        f"converged=T "
        f"smearing=0.01 "
        f"kpoints=[4,4,4] "
        f'comment="generated fixture, not physical" '
        f'pbc="T T T"\n'
    )
    _write_atom_lines(out, rng, n_atoms, a)


def _write_mace_mixed(out, rng: random.Random) -> None:
    """MACE-style training file: a handful of isolated-atom E0 frames first,
    then the bulk training frames, with deliberate schema drift between the
    two kinds. ``config_type`` rides only the isolated-atom frames; per-config
    training weights (``config_*_weight``) and ``stress`` ride only the bulk
    frames. Real training sets vary in which other keys appear, so this
    captures the shape of the drift, not any one file's exact columns."""
    for species in SPECIES:
        out.write("1\n")
        out.write(
            f'Lattice="15.0 0.0 0.0 0.0 15.1 0.0 0.0 0.0 15.2" '
            f"Properties=species:S:1:pos:R:3:forces:R:3 "
            f"energy={-1.0 * rng.random():.8f} config_type=IsolatedAtom "
            f'pbc="T T T"\n'
        )
        out.write(f"{species} 0.0 0.0 0.0 0.0 0.0 0.0\n")

    for _ in range(995):
        n_atoms = rng.randrange(32, 96)
        a = 5.0 + 10.0 * rng.random()
        out.write(f"{n_atoms}\n")
        # Bulk frame: no config_type, but per-frame training weights instead.
        out.write(
            f'Lattice="{a:.6f} 0.0 0.0 0.0 {a:.6f} 0.0 0.0 0.0 {a:.6f}" '
            f"Properties=species:S:1:pos:R:3:forces:R:3 "
            f"energy={-10.0 * rng.random():.8f} "
            f'stress="{_floats(rng, 9)}" '
            f"config_energy_weight={rng.random():.6f} "
            f"config_forces_weight={rng.random():.6f} "
            f"config_stress_weight={10.0 * rng.random():.6f} "
            f'pbc="T T T"\n'
        )
        _write_atom_lines(out, rng, n_atoms, a)


def _cached_file(name: str, seed: int, write) -> Path:
    path = CACHE_DIR / f"{name}-v{GENERATOR_VERSION}.extxyz"
    if path.exists():
        return path

    CACHE_DIR.mkdir(exist_ok=True)
    rng = random.Random(seed)
    # Write-then-rename so an interrupted run never leaves a half file.
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as out:
        write(out, rng)
    tmp.rename(path)
    return path


def _trajectory_file(
    name: str, n_frames: int, atoms_lo: int, atoms_hi: int, seed: int
) -> Path:
    def write(out, rng: random.Random) -> None:
        for _ in range(n_frames):
            _write_frame(out, rng, rng.randrange(atoms_lo, atoms_hi))

    return _cached_file(name, seed, write)


@pytest.fixture(scope="session")
def many_small_frames() -> Path:
    return trajectory_files()["many_small_frames"]


@pytest.fixture(scope="session")
def large_frames() -> Path:
    return trajectory_files()["large_frames"]


@pytest.fixture(scope="session")
def metadata_heavy() -> Path:
    return trajectory_files()["metadata_heavy"]


@pytest.fixture(scope="session")
def mace_mixed() -> Path:
    return trajectory_files()["mace_mixed"]


# The store-comparison dataset. Fixed frame size keeps the maths obvious:
# every read of k frames is k * 64 atoms.
STORE_N_FRAMES = 100_000
STORE_ATOMS_PER_FRAME = 64


@pytest.fixture(scope="session")
def store_dataset() -> Path:
    return store_dataset_file()


def store_dataset_file() -> Path:
    def write(out, rng: random.Random) -> None:
        for _ in range(STORE_N_FRAMES):
            _write_frame(out, rng, STORE_ATOMS_PER_FRAME)

    return _cached_file("store_100k", 0x5EED5, write)


def trajectory_files() -> dict[str, Path]:
    def write_metadata_heavy(out, rng: random.Random) -> None:
        for _ in range(2_000):
            _write_metadata_heavy_frame(out, rng, rng.randrange(16, 64))

    return {
        "many_small_frames": _trajectory_file(
            "many_small_frames", 2_000, 16, 64, seed=0x5EED
        ),
        "large_frames": _trajectory_file(
            "large_frames", 4, 100_000, 100_001, seed=0x5EED2
        ),
        "metadata_heavy": _cached_file("metadata_heavy", 0x5EED3, write_metadata_heavy),
        "mace_mixed": _cached_file("mace_mixed", 0x5EED4, _write_mace_mixed),
    }
