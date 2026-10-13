"""Torch helpers shared by `oxyz.metatomic` and `oxyz.torch_sim`.

Imports torch eagerly: it is only reached after a target's guarded import.
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
    """Tensor from a numeric or bool array, or a clear error naming the key.

    `dtype=None` infers from the array; resolving a default dtype is the
    caller's choice.
    """
    if array.dtype.kind not in "biuf":
        raise ValueError(
            f"{key!r} is not numeric (dtype {array.dtype}); cannot make a tensor"
        )
    return torch.tensor(array, dtype=dtype, device=device)
