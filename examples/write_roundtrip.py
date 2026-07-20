"""Write frames back to extxyz and read them again — a lossless round-trip."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import oxyz

DATA = Path(__file__).parent / "data" / "water.extxyz"


def main() -> None:
    frames = oxyz.read(DATA)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "roundtrip.extxyz"
        oxyz.write(out, frames)
        reread = oxyz.read(out)
        same = all(
            np.array_equal(a.columns["pos"], b.columns["pos"])
            for a, b in zip(frames, reread, strict=True)
        )
        print("frames:", len(reread), "positions identical:", same)


if __name__ == "__main__":
    main()
