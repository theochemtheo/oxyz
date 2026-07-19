"""Use oxyz where you would use ase.io — same call, same fields, faster."""

from __future__ import annotations

import importlib.util
from pathlib import Path

DATA = Path(__file__).parent / "data" / "water.extxyz"


def main() -> None:
    if importlib.util.find_spec("ase") is None:
        print("ase not installed; skipping")
        return
    import oxyz.ase

    last = oxyz.ase.read(DATA)  # last frame, like ase.io.read
    images = oxyz.ase.read(DATA, ":")  # every frame
    print("last:", last.get_chemical_formula(), "images:", len(images))


if __name__ == "__main__":
    main()
