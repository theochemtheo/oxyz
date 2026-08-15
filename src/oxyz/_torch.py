"""Torch conversion helpers shared by `oxyz.metatomic` and `oxyz.torch_sim`.

Both targets turn the same untouched core arrays into torch tensors: numeric
arrays to tensors of a resolved dtype, and the `dtype=None` ->
`torch.get_default_dtype()` rule that `systems_to_torch` and `atoms_to_state`
both follow. The species/cell *policy* that differs between the two
(metatomic's per-frame transpose-and-zero cell versus torch_sim's batched
column-convention cell, and each target's own error type) stays in the target
modules; only the target-neutral mechanics live here.

`torch` is imported eagerly: this module is only reached after a target module's
own guarded import of torch has already succeeded. That is why the well-known
field resolvers live in `oxyz._convert` instead — `Frame.numbers` must work
without the torch extra installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import numpy as np


def resolve_dtype(dtype: torch.dtype | None) -> torch.dtype:
    return torch.get_default_dtype() if dtype is None else dtype


def to_tensor(
    array: np.ndarray,
    key: str,
    dtype: torch.dtype | None,
    device: torch.device | None,
) -> torch.Tensor:
    """Tensor from a numeric/bool array, or a clear error naming the key.

    `dtype` is passed straight to `torch.tensor`: `None` infers from the array
    (so a float64 column stays float64), as both `systems_to_torch` and
    `atoms_to_state` do for their extras. The two targets differ in whether a
    `None` *default* resolves to `torch.get_default_dtype()`; that choice stays
    in the caller, which resolves before calling when it wants to.

    `torch.tensor` on a string or object array raises a cryptic TypeError, so
    reject non-numeric columns up front — a per-atom `species` column or a
    string metadata value is not a target.
    """
    if array.dtype.kind not in "biuf":
        raise ValueError(
            f"{key!r} is not numeric (dtype {array.dtype}); cannot make a tensor"
        )
    return torch.tensor(array, dtype=dtype, device=device)
