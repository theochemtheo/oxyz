"""Read frames straight into numpy arrays — no per-atom Python objects."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import oxyz

DATA = Path(__file__).parent / "data" / "water.extxyz"


def main() -> None:
    frames = oxyz.read(DATA)  # list[Frame]; ":" is the default index
    first = frames[0]
    print("frames:", len(frames))
    pos = np.asarray(first.columns["pos"])  # string columns come back as lists
    print("pos:", pos.shape, pos.dtype)
    print("species:", list(first.columns["species"]))
    print("energy:", first.metadata["energy"])


if __name__ == "__main__":
    main()
