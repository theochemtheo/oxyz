"""Stream atom-budgeted batches in the PyG layout, zero-copy into torch."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

import oxyz

DATA = Path(__file__).parent / "data" / "water.extxyz"


def main() -> None:
    for batch in oxyz.iread_batch(DATA, atoms_per_batch=6, shuffle=True, seed=0):
        pos = np.asarray(batch.columns["pos"])  # (total_atoms, 3), atom-major
        energy = np.asarray(batch.metadata["energy"])  # (n_frames,)
        print(
            "batch:",
            pos.shape,
            "frames:",
            energy.shape,
            "provenance:",
            list(batch.frame_indices),
        )
        if importlib.util.find_spec("torch") is not None:
            import torch

            tensor = torch.from_numpy(pos)  # zero-copy view
            print("  torch:", tuple(tensor.shape))


if __name__ == "__main__":
    main()
