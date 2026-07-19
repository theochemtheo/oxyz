"""Read straight into metatomic.torch.System and torch_sim.SimState."""

from __future__ import annotations

import importlib.util
from pathlib import Path

DATA = Path(__file__).parent / "data" / "water.extxyz"


def main() -> None:
    have_torch = importlib.util.find_spec("torch") is not None
    if have_torch and importlib.util.find_spec("metatomic") is not None:
        import torch

        import oxyz.metatomic

        systems = oxyz.metatomic.read(DATA, dtype=torch.float64)  # list[System]
        print("metatomic systems:", len(systems))
    else:
        print("metatomic not installed; skipping")

    if have_torch and importlib.util.find_spec("torch_sim") is not None:
        import oxyz.torch_sim

        state = oxyz.torch_sim.read(DATA)  # one batched SimState
        print("torch_sim atoms:", state.positions.shape[0])
    else:
        print("torch_sim not installed; skipping")


if __name__ == "__main__":
    main()
