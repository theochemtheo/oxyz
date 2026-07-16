#!/usr/bin/env python3
"""Slice every Nth frame of a large extxyz file into a small committed sample.

Byte-level: frames are copied verbatim, so the sample keeps the source's exact
formatting rather than a re-serialised approximation. Stdlib only; extxyz is
self-delimiting (count line, comment line, then `count` atom rows).

Usage: slice_mad_sample.py SOURCE DEST --stride N [--max-bytes B]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import TextIO


def frames(handle: TextIO) -> Iterator[str]:
    """Yield each frame of an extxyz stream as a verbatim string."""
    while True:
        count_line = handle.readline()
        if not count_line:
            return
        if not count_line.strip():
            continue
        n_atoms = int(count_line.strip())
        comment = handle.readline()
        rows = [handle.readline() for _ in range(n_atoms)]
        yield count_line + comment + "".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("dest", type=Path)
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--max-bytes", type=int, default=500_000)
    args = parser.parse_args()

    written = kept = 0
    with args.source.open() as src, args.dest.open("w") as out:
        for index, frame in enumerate(frames(src)):
            if index % args.stride:
                continue
            if written + len(frame) > args.max_bytes:
                break
            out.write(frame)
            written += len(frame)
            kept += 1

    print(f"kept {kept} frames ({written} bytes) at stride {args.stride}")


if __name__ == "__main__":
    main()
