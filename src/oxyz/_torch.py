"""Torch conversion helpers shared by `oxyz.metatomic` and `oxyz.torch_sim`.

Both targets turn the same untouched core arrays into torch tensors: atomic
numbers from a `Z`/`numbers`/`species` column, numeric arrays to tensors of a
resolved dtype, and the `dtype=None` -> `torch.get_default_dtype()` rule that
`systems_to_torch` and `atoms_to_state` both follow. The species/cell *policy*
that differs between the two (metatomic's per-frame transpose-and-zero cell
versus torch_sim's batched column-convention cell, and each target's own error
type) stays in the target modules; only the target-neutral mechanics live here.

`torch` is imported eagerly: this module is only reached after a target module's
own guarded import of torch has already succeeded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from oxyz._convert import species_to_numbers

if TYPE_CHECKING:
    from oxyz._frames import ColumnValues


class MissingSpeciesError(Exception):
    """Columns carry no `Z`, `numbers`, or `species` to derive atomic numbers.

    Raised neutrally so each target wraps it in its own conversion error with
    its own wording (a metatomic `Frame` versus a batched torch_sim state).
    """


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


def numbers(columns: dict[str, ColumnValues]) -> np.ndarray:
    """Return atomic numbers (int32) from columns.

    An explicit `Z`/`numbers` column wins, else the `species` column is
    mapped via the shared element table. Works on a single frame's per-atom
    columns or a batch's concatenated
    columns alike. Raises `MissingSpeciesError` when no usable column is
    present and `oxyz._convert.UnknownSpeciesError` when a species token has no
    chemical symbol; the caller maps both to its own error type.
    """
    for name in ("Z", "numbers"):
        column = columns.get(name)
        if column is not None:
            values = np.asarray(column)
            if np.issubdtype(values.dtype, np.floating):
                values = np.rint(values)  # a float Z column: round, don't truncate
            return values.astype(np.int32, copy=False)

    species = columns.get("species")
    if species is None:
        raise MissingSpeciesError
    return species_to_numbers(species)
