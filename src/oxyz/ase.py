"""ASE conversion layer: all ASE knowledge in oxyz lives in this module.

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

from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import overload

import numpy as np

from oxyz._frames import Frame, IndexedFrames, iter_frames

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
        "oxyz.ase requires the optional dependency 'ase'; "
        "install it with: pip install oxyz[ase]"
    ) from error

__all__ = ["ToAseError", "iread", "read", "to_atoms"]


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
    path: str | Path, index: slice, *, format: str | None = ...
) -> list[Atoms]: ...


@overload
def read(
    path: str | Path, index: str, *, format: str | None = ...
) -> Atoms | list[Atoms]: ...


def read(
    path: str | Path,
    index: int | str | slice | None = None,
    *,
    format: str | None = None,
) -> Atoms | list[Atoms]:
    """Drop-in for `ase.io.read` on extxyz files; full ASE index grammar.

    Like ASE, the default index is -1: the last frame. Forward selections
    stream; negative or reverse ones resolve via a structural scan and seek,
    never a full parse. Only requested frames have their contents read —
    whole-file validation is `oxyz.infer_schema`'s job.
    """
    _check_format(format)
    index = _parse_index(-1 if index is None else index)
    if isinstance(index, int):
        return to_atoms(_nth_frame(path, index))
    return [to_atoms(frame) for frame in _sliced_frames(path, index)]


def iread(
    path: str | Path,
    index: int | str | slice = ":",
    *,
    format: str | None = None,
) -> Iterator[Atoms]:
    """Drop-in for `ase.io.iread` on extxyz files: yields one Atoms at a time."""
    _check_format(format)
    index = _parse_index(index)
    if isinstance(index, int):
        return iter((to_atoms(_nth_frame(path, index)),))
    return (to_atoms(frame) for frame in _sliced_frames(path, index))


def _check_format(format: str | None) -> None:
    if format not in (None, "extxyz", "xyz"):
        raise ValueError(f"oxyz.ase only reads extxyz, got format={format!r}")


def _parse_index(index: int | str | slice) -> int | slice:
    """ASE's index grammar: an int, an int string, or a slice string."""
    if not isinstance(index, str):
        return index
    if ":" not in index:
        return int(index)
    parts = index.split(":")
    if len(parts) > 3:
        raise ValueError(f"invalid slice string: {index!r}")
    start, stop, step = (int(part) if part else None for part in (*parts, "", "")[:3])
    return slice(start, stop, step)


def _nth_frame(path: str | Path, index: int) -> Frame:
    """Frame `index`; negatives resolve via a scan and seek, not a full parse."""
    if index < 0:
        frames = IndexedFrames(path)
        if index + len(frames) < 0:
            raise IndexError(
                f"frame {index} out of range: file has {len(frames)} frames"
            )
        return frames.get(index + len(frames))

    frame = next(islice(iter_frames(path), index, None), None)
    if frame is None:
        raise IndexError(f"frame {index} out of range")
    return frame


def _sliced_frames(path: str | Path, frames: slice) -> Iterator[Frame]:
    """Forward slices stream; negative bounds or steps go via the index."""
    start, stop, step = frames.start, frames.stop, frames.step
    forward = all(bound is None or bound >= 0 for bound in (start, stop)) and (
        step is None or step > 0
    )
    if forward:
        return islice(iter_frames(path), start, stop, step)

    indexed = IndexedFrames(path)
    selected = range(*frames.indices(len(indexed)))
    return (indexed.get(i) for i in selected)
