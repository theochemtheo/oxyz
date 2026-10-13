"""Ask an unfamiliar file what it contains, and whether its schema is consistent."""

from __future__ import annotations

from pathlib import Path

import oxyz

DATA = Path(__file__).parent / "data" / "water.extxyz"
MIXED = Path(__file__).parent / "data" / "mixed.extxyz"


def main() -> None:
    index = oxyz.scan(DATA)  # structural only — parses no values
    print(f"{DATA.name}: {index.n_frames} frames")

    schema = oxyz.infer_schema(DATA)
    print(schema)
    print("consistent:", schema.is_consistent)

    mixed = oxyz.infer_schema(MIXED)
    print("mixed consistent:", mixed.is_consistent)  # False — forces drifts


if __name__ == "__main__":
    main()
