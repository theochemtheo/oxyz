"""Shared ingest plumbing for the store backends."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

import oxyz


def key(index: int) -> bytes:
    """LMDB key for a frame index: fixed-width big-endian, so lexicographic
    cursor order is frame order."""
    return index.to_bytes(8, "big")


def build_once(dest: Path, build: Callable[[Path], None]) -> Path:
    """Run `build` into a temporary path and rename, so an interrupted
    ingest never leaves a half-written store behind."""
    if dest.exists():
        return dest
    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    build(tmp)
    tmp.rename(dest)
    # LMDB leaves a lock file next to the environment it built.
    tmp_lock = tmp.with_name(tmp.name + "-lock")
    if tmp_lock.exists():
        tmp_lock.unlink()
    return dest


def frame_record(frame: oxyz.Frame) -> dict[str, Any]:
    """The arrays every store keeps per frame: what a training pipeline
    actually consumes, with species as atomic numbers."""
    from ase.data import atomic_numbers

    # The column and metadata types are unions over everything extxyz can
    # hold; the generated dataset pins them (species strings, float energy).
    metadata = frame.metadata
    return {
        "numbers": np.array(
            [atomic_numbers[s] for s in frame.columns["species"]],  # ty: ignore[invalid-argument-type]
            dtype=np.uint8,
        ),
        "positions": np.asarray(frame.columns["pos"]),
        "forces": np.asarray(frame.columns["forces"]),
        "energy": float(metadata["energy"]),  # ty: ignore[invalid-argument-type]
        "cell": np.asarray(metadata["Lattice"]).reshape(3, 3, order="F"),
        "pbc": np.asarray(metadata["pbc"], dtype=bool),
    }
