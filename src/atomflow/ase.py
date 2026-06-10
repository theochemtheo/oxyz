"""ASE conversion layer: all ASE knowledge in atomflow lives in this module.

Mirrors the mapping of `ase.io.extxyz`'s reader, reusing its routing tables
and `set_calc_and_arrays` so the result agrees with `ase.io.read` by
construction; golden tests in tests/test_ase.py hold the two readers equal.

Conversion is last-moment: the Rust core and `Frame` know nothing of ASE.
ASE is an optional dependency pinned `<4` — ASEv4 may change the data model,
so new majors are opted into deliberately.

Known divergence from `ase.io.read`: 6-component (Voigt) `stress` metadata
is accepted and routed to the calculator; ASE's comment parser rejects it.
"""

from __future__ import annotations

from pathlib import Path
from typing import overload

import numpy as np

from atomflow._frames import Frame, read_frames

try:
    from ase import Atoms
    from ase.constraints import FixAtoms, FixCartesian
    from ase.data import atomic_numbers

    # Internal-but-stable pieces of ase.io.extxyz, reused deliberately so the
    # key routing cannot drift from ASE's own reader. Revisit at ASEv4.
    from ase.io.extxyz import (
        REV_PROPERTY_NAME_MAP,
        SPECIAL_3_3_KEYS,
        set_calc_and_arrays,
    )
except ImportError as error:
    raise ImportError(
        "atomflow.ase requires the optional dependency 'ase'; "
        "install it with: pip install atomflow[ase]"
    ) from error

__all__ = ["ToAseError", "read", "to_atoms"]


class ToAseError(ValueError):
    """The frame has no faithful `ase.Atoms` representation (strict: no repair)."""


def to_atoms(frame: Frame) -> Atoms:
    """Convert one `Frame` to `ase.Atoms`, mirroring `ase.io.read`'s mapping."""
    info: dict = dict(frame.metadata)

    for key in SPECIAL_3_3_KEYS:
        value = info.get(key)
        if value is None:
            continue
        value = np.asarray(value)
        if value.shape == (9,):
            # extxyz stores these matrices flattened in Fortran order.
            info[key] = value.reshape((3, 3), order="F")
        elif key == "Lattice":
            raise ToAseError(f"Lattice must have 9 components, got shape {value.shape}")

    pbc = info.pop("pbc", None)
    cell = None
    lattice = info.pop("Lattice", None)
    if lattice is not None:
        cell = lattice.T
        if pbc is None:
            pbc = [True, True, True]

    arrays: dict = {}
    for name, values in frame.columns.items():
        ase_name = REV_PROPERTY_NAME_MAP.get(name, name)
        arrays[ase_name] = np.asarray(values) if isinstance(values, list) else values

    numbers = arrays.pop("numbers", None)
    symbols = arrays.pop("symbols", None)
    if symbols is not None:
        symbols = [str(s).capitalize() for s in symbols]
        unknown = sorted({s for s in symbols if s not in atomic_numbers})
        if unknown:
            raise ToAseError(f"species are not chemical symbols: {unknown}")
    if numbers is None and symbols is None:
        raise ToAseError("frame has neither a 'species' nor a 'Z' column")

    atoms = Atoms(
        numbers if numbers is not None else symbols,
        positions=arrays.pop("positions", None),
        charges=arrays.pop("initial_charges", None),
        cell=cell,
        pbc=pbc,
        info=info,
    )

    if "move_mask" in arrays:
        move_mask = np.asarray(arrays.pop("move_mask")).astype(bool)
        if move_mask.ndim == 2 and move_mask.shape[1] == 3:
            constraints = [
                FixCartesian(a, mask=~move_mask[a]) for a in range(frame.n_atoms)
            ]
            atoms.set_constraint(constraints)
        elif move_mask.ndim == 1:
            atoms.set_constraint(FixAtoms(mask=~move_mask))
        else:
            raise ToAseError(f"move_mask must have width 1 or 3, got {move_mask.shape}")

    # ASE's own routing: known results -> SinglePointCalculator, rest -> arrays.
    set_calc_and_arrays(atoms, arrays)
    return atoms


@overload
def read(
    path: str | Path, index: int | None = ..., *, format: str | None = ...
) -> Atoms: ...


@overload
def read(
    path: str | Path, index: str | slice, *, format: str | None = ...
) -> list[Atoms]: ...


def read(
    path: str | Path,
    index: int | str | slice | None = None,
    *,
    format: str | None = None,
) -> Atoms | list[Atoms]:
    """Drop-in for `ase.io.read` on extxyz files.

    Like ASE, the default index is -1: the last frame.
    """
    if format not in (None, "extxyz", "xyz"):
        raise ValueError(f"atomflow.ase.read only reads extxyz, got format={format!r}")

    # TODO(step 2): stream via the FrameIter binding instead of materialising.
    frames = read_frames(path)
    if index is None:
        index = -1
    if isinstance(index, int):
        return to_atoms(frames[index])
    return [to_atoms(frame) for frame in frames[_as_slice(index)]]


def _as_slice(index: str | slice) -> slice:
    """Subset of ASE's index grammar: ':' and 'start:stop'. TODO: step, int strings."""
    if isinstance(index, str):
        parts = index.split(":")
        if len(parts) == 1:
            raise NotImplementedError(
                f"string index {index!r} not supported yet; pass an int instead"
            )
        if len(parts) > 2:
            raise NotImplementedError(f"slice step not supported yet: {index!r}")
        start, stop = (int(part) if part else None for part in parts)
        return slice(start, stop)

    if index.step not in (None, 1):
        raise NotImplementedError(f"slice step not supported yet: {index!r}")
    return slice(index.start, index.stop)
