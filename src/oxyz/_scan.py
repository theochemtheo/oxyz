from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import oxyz._rust as _rust
from oxyz import _remote
from oxyz._stats import AtomCountStats

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from oxyz._frames import Compression


@dataclass(frozen=True, slots=True)
class FrameIndex(AtomCountStats):
    """Structural facts from a scan: frame offsets and declared atom counts.

    Nothing is parsed beyond the count lines, so this is cheap even for files
    where a full read is not. Statistics are derived from the stored counts;
    `mean_atoms`/`median_atoms`/`std_atoms` come from `AtomCountStats`, with
    `std_atoms` the population standard deviation. All statistics are None for
    an empty file.
    """

    offsets: np.ndarray
    n_atoms: np.ndarray
    volumes: np.ndarray | None = None
    """Per-frame cell volume `|det(Lattice)|`, only for `scan(..., with_volume=True)`;
    `NaN` for a frame with no `Lattice`. `None` when volume was not requested."""

    @property
    def n_frames(self) -> int:
        return len(self.n_atoms)

    @property
    def total_atoms(self) -> int:
        return int(self.n_atoms.sum())

    @property
    def min_atoms(self) -> int | None:
        return int(self.n_atoms.min()) if self.n_frames else None

    @property
    def max_atoms(self) -> int | None:
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

    Returns a `FrameIndex` of per-frame byte offsets and declared atom counts.
    `n_atoms` is `intp` so arithmetic with it does not promote to float64.
    The atom-count statistics (`min_atoms`/`max_atoms`/`mean_atoms`/
    `median_atoms`/`std_atoms`) are `None` for an empty file — the only
    optionals in the result. For column and metadata types too, at the cost of
    a full parse, use `infer_schema`.

    `with_volume=True` reads one extra line per frame (the comment line) to
    record each frame's cell volume `|det(Lattice)|` in `volumes`; a frame with
    no `Lattice` gets `NaN`. It backs density-aware batch binning
    (`iter_batches(memory_scales_with="n_atoms_x_density")`).

    A compressed path is scanned by streaming through the decoder; the recorded
    offsets are into the decompressed stream, so they give no random-access
    speedup on a re-read (which decompresses afresh). `compression` and `member`
    work as in `read`.

    A remote URL (``s3://``, ``gs://``, ``az://``) streams the object through
    the same scanner (needs the ``oxyz[s3]`` extra); ``storage_options`` passes
    endpoint/credentials to the store.
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
