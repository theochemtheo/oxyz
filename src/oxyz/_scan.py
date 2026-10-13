from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from oxyz import _html, _remote, _rust, _summary
from oxyz._stats import AtomCountStats

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from oxyz._frames import Compression


@dataclass(frozen=True, slots=True)
class FrameIndex(AtomCountStats):
    """Structural facts from a scan: frame offsets and declared atom counts.

    All statistics are None for an empty file; `std_atoms` is the population
    standard deviation.

    Attributes
    ----------
    offsets
        Byte offset of each frame's first line.
    n_atoms
        Declared atom count per frame, dtype `intp`.
    volumes
        Per-frame cell volume `|det(Lattice)|` with
        `scan(..., with_volume=True)`, `NaN` without a `Lattice`; else None.
    """

    offsets: np.ndarray
    n_atoms: np.ndarray
    volumes: np.ndarray | None = None

    def __repr__(self) -> str:
        """Summarise the counts without printing the per-frame arrays."""
        volumes = (
            ""
            if self.volumes is None
            else f", volumes={_summary.summarise(self.volumes)}"
        )
        return (
            f"FrameIndex(n_frames={self.n_frames}, total_atoms={self.total_atoms}"
            f"{_summary.atom_range(self.n_atoms)}{volumes})"
        )

    def _repr_html_(self) -> str:
        """Render as tables for a notebook, previewing a bounded slice of values."""
        return _html.frame_index(self)

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return len(self.n_atoms)

    @property
    def total_atoms(self) -> int:
        """Sum of `n_atoms` across all frames."""
        return int(self.n_atoms.sum())

    @property
    def min_atoms(self) -> int | None:
        """Smallest per-frame atom count, or None if empty."""
        return int(self.n_atoms.min()) if self.n_frames else None

    @property
    def max_atoms(self) -> int | None:
        """Largest per-frame atom count, or None if empty."""
        return int(self.n_atoms.max()) if self.n_frames else None


def scan(
    path: str | Path,
    *,
    with_volume: bool = False,
    compression: Compression = "infer",
    member: str | None = None,
    storage_options: _remote.StorageOptions | None = None,
) -> FrameIndex:
    """Scan a file's structure without parsing any frame contents.

    For column and metadata types too, at the cost of a full parse, use
    `infer_schema`. On a compressed source the offsets index the decompressed
    stream, so they do not speed up a re-read.

    Parameters
    ----------
    path
        File path, or a remote URL; see `read`.
    with_volume
        Also record each frame's cell volume in `volumes`, reading each
        comment line.
    compression
        Forces a codec instead of inferring it from `path`; see `read`.
    member
        Selects one entry from an archive holding more than one; see `read`.
    storage_options
        Endpoint/credentials for a remote store; see `read`.

    Returns
    -------
    FrameIndex
        Per-frame byte offsets and declared atom counts.

    Examples
    --------
    >>> import oxyz
    >>> oxyz.scan(DATA / "water.extxyz").n_frames
    3
    """
    if _remote.is_remote(path):
        src = _remote.open_source(
            path,
            compression=compression,
            member=member,
            storage_options=storage_options,
        )
        data = _rust.scan_reader(src.obj, src.codec, with_volume, src.member)
    else:
        data = _rust.scan(str(path), with_volume, compression, member)
    return FrameIndex(
        offsets=data["offsets"],
        n_atoms=data["n_atoms"],
        volumes=data.get("volumes"),
    )
