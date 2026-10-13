"""Convert between extxyz frames and `ase.Atoms`.

Reuses `ase.io.extxyz`'s routing tables and `set_calc_and_arrays`, so results
agree with `ase.io.read`. One divergence: 6-component (Voigt) `stress`
metadata is accepted; the README lists the rest.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np

from oxyz._convert import UnknownSpeciesError, species_to_numbers
from oxyz._frames import (
    ColumnValues,
    Compression,
    Frame,
    MetadataValue,
    _require_schema_for_mode,
)
from oxyz._rust import OxyzError
from oxyz._select import frames_for_read, nth_frame, parse_index, sliced_frames

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from oxyz._remote import StorageOptions
    from oxyz._schema_match import Conformance
    from oxyz._schema_spec import Mode, SchemaSpec

try:
    from ase import Atoms
    from ase.constraints import FixAtoms, FixCartesian

    # Reused so key routing matches ASE's own reader.
    from ase.io.extxyz import (
        PROPERTY_NAME_MAP,
        REV_PROPERTY_NAME_MAP,
        SPECIAL_3_3_KEYS,
        save_calc_results,
        set_calc_and_arrays,
    )
except ImportError as error:
    raise ImportError(
        "oxyz.ase requires the optional dependency 'ase'; "
        "install it with: pip install oxyz[ase]"
    ) from error

__all__ = ["FromAtomsError", "ToAseError", "from_atoms", "iread", "read", "to_atoms"]

_N_CARTESIAN = 3


class ToAseError(OxyzError):
    """The frame has no faithful `ase.Atoms` representation (strict: no repair)."""


class FromAtomsError(OxyzError):
    """The `ase.Atoms` carries something a `Frame` cannot represent faithfully."""


def to_atoms(frame: Frame) -> Atoms:  # noqa: C901, PLR0912  flat field-by-field mapping to ase.Atoms
    """Convert one `Frame` to `ase.Atoms`, mirroring `ase.io.read`'s mapping.

    Parameters
    ----------
    frame
        The frame to convert.

    Returns
    -------
    ase.Atoms
        The converted structure.

    Raises
    ------
    ToAseError
        If `frame` has no faithful `Atoms` representation: no `pos` column, an
        unmapped species, a malformed `Lattice`, or a malformed `move_mask`.
    """
    info: dict = dict(frame.metadata)

    # Unlike `oxyz._convert.cell`, this follows ASE's tables for virial and stress too.
    for key in SPECIAL_3_3_KEYS:
        value = info.get(key)
        if value is None:
            continue
        value = np.asarray(value)
        if value.shape == (9,):
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
        # A `Z` column, if present, still wins, as in ASE's reader.
        try:
            mapped = species_to_numbers(symbols)
        except UnknownSpeciesError as error:
            raise ToAseError(str(error)) from None
        if numbers is None:
            numbers = mapped
    if numbers is None:
        raise ToAseError("frame has neither a 'species' nor a 'Z' column")

    positions = arrays.pop("positions", None)
    if positions is None:
        # ASE would otherwise put every atom at the origin.
        raise ToAseError("frame has no 'pos' column to use as positions")

    atoms = Atoms(
        numbers=numbers,
        positions=positions,
        charges=arrays.pop("initial_charges", None),
        cell=cell,
        pbc=pbc,
        info=info,
    )

    if "move_mask" in arrays:
        move_mask = np.asarray(arrays.pop("move_mask")).astype(bool)
        if move_mask.shape[1:] == (_N_CARTESIAN,):
            constraints = [
                FixCartesian(a, mask=~move_mask[a]) for a in range(frame.n_atoms)
            ]
            atoms.set_constraint(constraints)
        elif move_mask.ndim == 1:
            atoms.set_constraint(FixAtoms(mask=~move_mask))
        else:
            raise ToAseError(f"move_mask must have width 1 or 3, got {move_mask.shape}")

    set_calc_and_arrays(atoms, arrays)
    return atoms


def from_atoms(atoms: Atoms) -> Frame:
    """Convert one `ase.Atoms` to a `Frame`, the inverse of `to_atoms`.

    Follows `ase.io.write`'s mapping. Calculator results are folded in
    unprefixed, so they re-read as a `SinglePointCalculator`; `FixAtoms`
    becomes a `move_mask` column.

    Parameters
    ----------
    atoms
        The structure to convert.

    Returns
    -------
    Frame
        The converted frame.

    Raises
    ------
    FromAtomsError
        If `atoms` carries a constraint other than `FixAtoms`.
    """
    # `Atoms.copy()` drops the calculator, so keep the original's.
    calc = atoms.calc
    atoms = atoms.copy()
    if calc is not None:
        save_calc_results(atoms, calc=calc, calc_prefix="")

    columns: dict[str, ColumnValues] = {
        "species": list(atoms.get_chemical_symbols()),
        "pos": atoms.get_positions(),
    }
    for name, values in atoms.arrays.items():
        if name not in ("numbers", "positions"):
            columns[PROPERTY_NAME_MAP.get(name, name)] = values

    move_mask = _move_mask_column(atoms)
    if move_mask is not None:
        columns["move_mask"] = move_mask

    metadata: dict[str, MetadataValue] = {}
    if atoms.cell.rank > 0 or atoms.pbc.any():
        # The inverse of to_atoms's Fortran-order reshape and transpose.
        metadata["Lattice"] = np.asarray(atoms.cell).reshape(9)
        metadata["pbc"] = np.asarray(atoms.pbc)
    for key, value in atoms.info.items():
        array = np.asarray(value)
        if key in SPECIAL_3_3_KEYS and array.shape == (3, 3):
            metadata[key] = array.reshape(9, order="F")
        else:
            metadata[key] = value

    return Frame(n_atoms=len(atoms), columns=columns, metadata=metadata)


def _move_mask_column(atoms: Atoms) -> np.ndarray | None:
    """Per-atom `move_mask` from `FixAtoms` constraints (True where free)."""
    if not atoms.constraints:
        return None
    mask = np.ones(len(atoms), dtype=bool)
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            mask[constraint.index] = False
        else:
            raise FromAtomsError(
                f"cannot represent constraint {type(constraint).__name__} as a "
                "column; only FixAtoms is supported"
            )
    return mask


@overload
def read(
    path: str | Path,
    index: int | None = ...,
    *,
    format: str | None = ...,
    threads: int | None = ...,
    schema: SchemaSpec | str | Path | None = ...,
    conformance: Conformance = ...,
    mode: Mode | None = ...,
    compression: Compression = ...,
    member: str | None = ...,
    storage_options: StorageOptions | None = ...,
) -> Atoms: ...


@overload
def read(
    path: str | Path,
    index: slice,
    *,
    format: str | None = ...,
    threads: int | None = ...,
    schema: SchemaSpec | str | Path | None = ...,
    conformance: Conformance = ...,
    mode: Mode | None = ...,
    compression: Compression = ...,
    member: str | None = ...,
    storage_options: StorageOptions | None = ...,
) -> list[Atoms]: ...


@overload
def read(
    path: str | Path,
    index: str,
    *,
    format: str | None = ...,
    threads: int | None = ...,
    schema: SchemaSpec | str | Path | None = ...,
    conformance: Conformance = ...,
    mode: Mode | None = ...,
    compression: Compression = ...,
    member: str | None = ...,
    storage_options: StorageOptions | None = ...,
) -> Atoms | list[Atoms]: ...


def read(  # noqa: PLR0913  the index/schema/projection/source options are the contract
    path: str | Path,
    index: int | str | slice | None = None,
    *,
    format: str | None = None,
    threads: int | None = None,
    schema: SchemaSpec | str | Path | None = None,
    conformance: Conformance = "required",
    mode: Mode | None = None,
    compression: Compression = "infer",
    member: str | None = None,
    storage_options: StorageOptions | None = None,
) -> Atoms | list[Atoms]:
    """Drop-in for `ase.io.read` on extxyz files, with ASE's index grammar.

    As in ASE, the default index is -1, the last frame. A negative or reverse
    index seeks after a structural scan, except on a compressed or remote
    source, or with a `schema`, where the whole file is read.

    Parameters
    ----------
    path
        File path or a remote URL; see `oxyz.read`.
    index
        ASE index grammar. `None` means -1; an `int` returns one `Atoms`; a
        slice or slice-string (`"1:10:2"`) returns a list.
    format
        For `ase.io.read` compatibility; only `None`, `"extxyz"`, and `"xyz"`
        are valid.
    threads
        Parallel parse for a whole-file or forward read (`None`: all cores,
        `1`: serial).
    schema
        A `SchemaSpec`, or a path to a schema file, applied to the frames read.
        See `oxyz.SchemaSpec`.
    conformance
        How a schema deviation is handled: `"strict"`, `"required"`
        (default), or `"warn"`.
    mode
        Overrides the schema's own `mode`.
    compression
        Forces a codec instead of inferring it from `path`; see `oxyz.read`.
    member
        Selects one entry from an archive holding more than one.
    storage_options
        Endpoint/credentials for a remote store.

    Returns
    -------
    Atoms or list[Atoms]
        A single `Atoms` for an integer (or default) index, otherwise a list.

    Examples
    --------
    >>> import oxyz.ase
    >>> images = oxyz.ase.read(DATA / "water.extxyz", ":")
    >>> len(images)
    3
    """
    _check_format(format)
    _require_schema_for_mode(schema, mode)
    index = parse_index(-1 if index is None else index)
    if isinstance(index, int):
        return to_atoms(
            nth_frame(
                path,
                index,
                schema=schema,
                conformance=conformance,
                mode=mode,
                compression=compression,
                member=member,
                storage_options=storage_options,
            )
        )
    return [
        to_atoms(frame)
        for frame in frames_for_read(
            path,
            index,
            threads,
            schema=schema,
            conformance=conformance,
            mode=mode,
            compression=compression,
            member=member,
            storage_options=storage_options,
        )
    ]


def iread(  # noqa: PLR0913  the index/schema/projection/source options are the contract
    path: str | Path,
    index: int | str | slice = ":",
    *,
    format: str | None = None,
    schema: SchemaSpec | str | Path | None = None,
    conformance: Conformance = "required",
    mode: Mode | None = None,
    compression: Compression = "infer",
    member: str | None = None,
    storage_options: StorageOptions | None = None,
) -> Iterator[Atoms]:
    """Drop-in for `ase.io.iread` on extxyz files, yielding one `Atoms` at a time.

    Parameters
    ----------
    path
        File path or a remote URL; see `oxyz.read`.
    index
        ASE index grammar; see `read`. Default `":"` yields every frame.
    format
        For `ase.io.iread` compatibility; see `read`.
    schema
        A `SchemaSpec`, or a path to a schema file; see `read`.
    conformance
        How a schema deviation is handled; see `read`.
    mode
        Overrides the schema's own `mode`; see `read`.
    compression
        Forces a codec instead of inferring it from `path`; see `read`.
    member
        Selects one entry from an archive holding more than one; see `read`.
    storage_options
        Endpoint/credentials for a remote store.

    Returns
    -------
    Iterator[Atoms]
        Frames in file order.
    """
    _check_format(format)
    _require_schema_for_mode(schema, mode)
    index = parse_index(index)
    if isinstance(index, int):
        return iter(
            (
                to_atoms(
                    nth_frame(
                        path,
                        index,
                        schema=schema,
                        conformance=conformance,
                        mode=mode,
                        compression=compression,
                        member=member,
                        storage_options=storage_options,
                    )
                ),
            )
        )
    return (
        to_atoms(frame)
        for frame in sliced_frames(
            path,
            index,
            schema=schema,
            conformance=conformance,
            mode=mode,
            compression=compression,
            member=member,
            storage_options=storage_options,
        )
    )


def _check_format(format: str | None) -> None:
    """Reject a `format` other than the ones `oxyz.ase` reads."""
    if format not in (None, "extxyz", "xyz"):
        raise ValueError(f"oxyz.ase only reads extxyz, got format={format!r}")
