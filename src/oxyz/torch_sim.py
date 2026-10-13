"""Convert extxyz frames to batched `torch_sim.SimState`.

`SimState` holds many systems in one batch, so this target builds on
`oxyz.read_batch` rather than reading frame by frame. `read` returns one
state for the whole selection, `iread` streams states with the strategies of
`oxyz.iread_batch`, and `SimStateSource` parses once and serves the state and
per-key tensors.

The result matches `torch_sim.io.atoms_to_state(ase.io.read(...))`: cells in
torch_sim's column-vector convention, one pbc for the batch, and masses from a
`masses` column or the atomic numbers.

Needs `pip install oxyz[torch-sim]`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import numpy as np

from oxyz._batch import Batch, MemoryScaling, iread_batch, read_batch
from oxyz._convert import (
    MissingSpeciesError,
    UnknownSpeciesError,
    numbers,
    numbers_to_masses,
)
from oxyz._frames import _require_schema_for_mode
from oxyz._rust import OxyzError
from oxyz._scan import scan
from oxyz._select import parse_index

try:
    import torch
    from torch_sim import SimState
except ImportError as error:
    raise ImportError(
        "oxyz.torch_sim requires the optional dependencies 'torch' and "
        "'torch-sim-atomistic'; install them with: pip install oxyz[torch-sim]"
    ) from error

from typing import TYPE_CHECKING

from oxyz._torch import to_tensor

if TYPE_CHECKING:
    from pathlib import Path

    from oxyz._frames import Compression
    from oxyz._remote import StorageOptions
    from oxyz._schema_match import Conformance
    from oxyz._schema_spec import Mode, SchemaSpec

__all__ = ["SimStateSource", "ToSimStateError", "iread", "read"]

ExtrasMap = Mapping[str, str]


class ToSimStateError(OxyzError):
    """The frames have no faithful `SimState` representation (strict: no repair)."""


def read(  # noqa: PLR0913  keyword options mirror the SimState data model
    path: str | Path,
    index: int | str | slice = ":",
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    positions_requires_grad: bool = False,
    system_extras: ExtrasMap | None = None,
    atom_extras: ExtrasMap | None = None,
    threads: int | None = None,
    schema: SchemaSpec | str | Path | None = None,
    conformance: Conformance = "required",
    mode: Mode | None = None,
    compression: Compression = "infer",
    member: str | None = None,
    storage_options: StorageOptions | None = None,
) -> SimState:
    """Read selected frames into one batched `SimState`.

    For a model-aware split, pass the result to torch_sim's
    `BinningAutoBatcher`.

    Parameters
    ----------
    path
        File path or a remote URL; see `oxyz.read`.
    index
        `":"` (default) reads every frame; an `int` gives a state with
        `n_systems == 1`; a slice or slice-string selects several.
    dtype
        Float dtype for positions/cell/masses. `None` keeps the data's
        float64, as `atoms_to_state` does; pass `torch.float32` for ML use.
    device
        Target device for the returned tensors.
    positions_requires_grad
        Whether the returned `positions` tensor requires grad.
    system_extras
        `{simstate_key: oxyz_key}` map pulling frame metadata into the
        state's `_system_extras`.
    atom_extras
        `{simstate_key: oxyz_key}` map pulling per-atom columns into the
        state's `_atom_extras`.
    threads
        Parallel parse (`None`: all cores).
    schema
        A `SchemaSpec`, or a path to a schema file. See `oxyz.SchemaSpec`.
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
        Endpoint/credentials for a remote store; see `oxyz.read`.

    Returns
    -------
    SimState
        The selected frames, batched into one state.

    Raises
    ------
    ToSimStateError
        If the frames have no faithful `SimState` representation.

    Examples
    --------
    >>> import oxyz.torch_sim
    >>> state = oxyz.torch_sim.read(DATA / "water.extxyz")
    >>> int(state.positions.shape[0])
    9
    """
    _require_schema_for_mode(schema, mode)
    indices = _plan_indices(
        path,
        index,
        compression=compression,
        member=member,
        storage_options=storage_options,
    )
    batch = read_batch(
        path,
        ":" if indices is None else indices,
        threads=threads,
        schema=schema,
        conformance=conformance,
        mode=mode,
        compression=compression,
        member=member,
        storage_options=storage_options,
    )
    return _to_state(
        batch,
        dtype,
        device,
        positions_requires_grad=positions_requires_grad,
        system_extras=system_extras,
        atom_extras=atom_extras,
    )


def iread(  # noqa: PLR0913  batching options plus the SimState data model
    path: str | Path,
    *,
    frames_per_batch: int | None = None,
    atoms_per_batch: int | None = None,
    memory_scales_with: MemoryScaling | None = None,
    max_scaler: float | None = None,
    shuffle: bool = False,
    seed: int | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    positions_requires_grad: bool = False,
    system_extras: ExtrasMap | None = None,
    atom_extras: ExtrasMap | None = None,
    threads: int | None = None,
    schema: SchemaSpec | str | Path | None = None,
    conformance: Conformance = "required",
    mode: Mode | None = None,
    compression: Compression = "infer",
    member: str | None = None,
    storage_options: StorageOptions | None = None,
) -> Iterator[SimState]:
    """Stream the file as `SimState` batches, one per step.

    Exactly one strategy must be given, as in `oxyz.iread_batch`, which also
    describes what a compressed source supports.

    Parameters
    ----------
    path
        File path or a remote URL; see `oxyz.read`.
    frames_per_batch
        Fixed frame count per batch.
    atoms_per_batch
        Atom budget per batch.
    memory_scales_with
        `"n_atoms"` or `"n_atoms_x_density"`, packed under `max_scaler`.
    max_scaler
        The per-batch weight budget for `memory_scales_with`.
    shuffle
        Draw frames in a seeded random order.
    seed
        Seeds `shuffle`.
    dtype
        Float dtype for positions/cell/masses; see `read`.
    device
        Target device for the returned tensors.
    positions_requires_grad
        Whether each `positions` tensor requires grad.
    system_extras
        `{simstate_key: oxyz_key}` map pulling frame metadata into each
        state's `_system_extras`.
    atom_extras
        `{simstate_key: oxyz_key}` map pulling per-atom columns into each
        state's `_atom_extras`.
    threads
        Parallel parse (`None`: all cores).
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

    Yields
    ------
    SimState
        One batch per step.

    Raises
    ------
    ToSimStateError
        If a batch's frames have no faithful `SimState` representation.
    """
    _require_schema_for_mode(schema, mode)
    for batch in iread_batch(
        path,
        frames_per_batch=frames_per_batch,
        atoms_per_batch=atoms_per_batch,
        memory_scales_with=memory_scales_with,
        max_scaler=max_scaler,
        shuffle=shuffle,
        seed=seed,
        threads=threads,
        schema=schema,
        conformance=conformance,
        mode=mode,
        compression=compression,
        member=member,
        storage_options=storage_options,
    ):
        yield _to_state(
            batch,
            dtype,
            device,
            positions_requires_grad=positions_requires_grad,
            system_extras=system_extras,
            atom_extras=atom_extras,
        )


class SimStateSource:
    """Parse a file once; serve a batched `SimState` and per-key tensors from it.

    The frames must share a schema, as in any `Batch`.

    Parameters
    ----------
    path
        File path or a remote URL; see `oxyz.read`.
    threads
        Parallel parse (`None`: all cores).
    compression
        Forces a codec instead of inferring it from `path`.
    member
        Selects one entry from an archive holding more than one.
    storage_options
        Endpoint/credentials for a remote store.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        threads: int | None = None,
        compression: Compression = "infer",
        member: str | None = None,
        storage_options: StorageOptions | None = None,
    ) -> None:
        self._batch: Batch = read_batch(
            path,
            threads=threads,
            compression=compression,
            member=member,
            storage_options=storage_options,
        )

    def __len__(self) -> int:
        """Return the number of frames in the source."""
        return self._batch.n_frames

    def __repr__(self) -> str:
        """Summarise the batch the source serves from, under its own name."""
        return "SimStateSource" + repr(self._batch).removeprefix("Batch")

    def state(
        self,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        positions_requires_grad: bool = False,
        system_extras: ExtrasMap | None = None,
        atom_extras: ExtrasMap | None = None,
    ) -> SimState:
        """Convert the whole file to one batched `SimState`.

        Parameters
        ----------
        dtype
            Float dtype for positions/cell/masses; see `read`.
        device
            Target device for the returned tensors.
        positions_requires_grad
            Whether the returned `positions` tensor requires grad.
        system_extras
            `{simstate_key: oxyz_key}` map pulling frame metadata into
            `_system_extras`.
        atom_extras
            `{simstate_key: oxyz_key}` map pulling per-atom columns into
            `_atom_extras`.

        Returns
        -------
        SimState
            The whole file, batched into one state.

        Raises
        ------
        ToSimStateError
            If the frames have no faithful `SimState` representation.
        """
        return _to_state(
            self._batch,
            dtype,
            device,
            positions_requires_grad=positions_requires_grad,
            system_extras=system_extras,
            atom_extras=atom_extras,
        )

    def per_config(
        self,
        key: str,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """One metadata value across frames as a tensor: `(n_frames, *shape)`.

        Parameters
        ----------
        key
            The metadata key to extract.
        dtype
            Target tensor dtype.
        device
            Target device.

        Returns
        -------
        torch.Tensor
            Shape `(n_frames, *shape)`.

        Raises
        ------
        ValueError
            If `key` is missing from the source.
        """
        value = self._batch.metadata.get(key)
        if value is None:
            raise ValueError(f"metadata {key!r} missing from the source")
        return to_tensor(np.asarray(value), key, dtype, device)

    def per_atom(
        self,
        key: str,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, np.ndarray]:
        """One per-atom column across frames as `(values, offsets)`.

        Parameters
        ----------
        key
            The column key to extract.
        dtype
            Target tensor dtype.
        device
            Target device.

        Returns
        -------
        values : torch.Tensor
            Shape `(total_atoms, *width)`.
        offsets : numpy.ndarray
            Length `n_frames + 1`; frame `i` is
            `values[offsets[i]:offsets[i + 1]]`.

        Raises
        ------
        ValueError
            If `key` is missing from the source.
        """
        value = self._batch.columns.get(key)
        if value is None:
            raise ValueError(f"column {key!r} missing from the source")
        return to_tensor(np.asarray(value), key, dtype, device), self._batch.offsets


def _plan_indices(
    path: str | Path,
    index: int | str | slice,
    *,
    compression: Compression = "infer",
    member: str | None = None,
    storage_options: StorageOptions | None = None,
) -> list[int] | None:
    """Resolve an ASE-style index to frame indices, or `None` for the whole file."""
    selection = parse_index(index)
    if isinstance(selection, slice) and selection == slice(None, None, None):
        return None
    universe = range(
        scan(
            path,
            compression=compression,
            member=member,
            storage_options=storage_options,
        ).n_frames
    )
    if isinstance(selection, int):
        try:
            return [universe[selection]]
        except IndexError:
            raise IndexError(
                f"frame {selection} out of range: file has {len(universe)} frames"
            ) from None
    return [int(i) for i in universe[selection]]


def _to_state(
    batch: Batch,
    dtype: torch.dtype | None,
    device: torch.device | None,
    *,
    positions_requires_grad: bool,
    system_extras: ExtrasMap | None,
    atom_extras: ExtrasMap | None,
) -> SimState:
    # dtype=None infers float64, as atoms_to_state does.
    n_frames = batch.n_frames

    pos = batch.columns.get("pos")
    if pos is None:
        raise ToSimStateError("no 'pos' column to use as positions")
    positions = torch.tensor(
        np.asarray(pos),
        dtype=dtype,
        device=device,
        requires_grad=positions_requires_grad,
    )

    atomic_numbers_np = _numbers(batch)
    atomic_numbers = torch.tensor(atomic_numbers_np, dtype=torch.int32, device=device)
    masses = torch.tensor(_masses(batch, atomic_numbers_np), dtype=dtype, device=device)

    system_idx = torch.tensor(
        np.repeat(np.arange(n_frames), np.diff(batch.offsets)),
        dtype=torch.int64,
        device=device,
    )

    cell_np, pbc_np = _cell_and_pbc(batch)
    cell = torch.tensor(cell_np, dtype=dtype, device=device)
    pbc = torch.tensor(pbc_np, dtype=torch.bool, device=device)

    return SimState(
        positions=positions,
        masses=masses,
        cell=cell,
        pbc=pbc,
        atomic_numbers=atomic_numbers,
        system_idx=system_idx,
        _system_extras=_extras(batch.metadata, system_extras, dtype, device),
        _atom_extras=_extras(batch.columns, atom_extras, dtype, device),
    )


def _numbers(batch: Batch) -> np.ndarray:
    """Atomic numbers, with the missing/unknown cases as `ToSimStateError`s."""
    try:
        return numbers(batch.columns)
    except MissingSpeciesError:
        raise ToSimStateError(
            "no 'species' or 'Z' column to derive atomic numbers"
        ) from None
    except UnknownSpeciesError as error:
        raise ToSimStateError(str(error)) from None


def _masses(batch: Batch, atomic_numbers: np.ndarray) -> np.ndarray:
    """Per-atom masses: a `masses` column, else standard atomic weights."""
    column = batch.columns.get("masses")
    if column is None:
        return numbers_to_masses(atomic_numbers)
    values = np.asarray(column, dtype=float).reshape(-1)
    if values.shape[0] != atomic_numbers.shape[0]:
        raise ToSimStateError(
            f"'masses' column has {values.shape[0]} values for "
            f"{atomic_numbers.shape[0]} atoms"
        )
    return values


# torch_sim's column-vector convention, unlike `oxyz._convert.cell`.
def _cell_and_pbc(batch: Batch) -> tuple[np.ndarray, np.ndarray]:
    """Per-system `(n_frames, 3, 3)` cells and the batch's shared `(3,)` pbc.

    A frame with no `Lattice` has a zero cell.
    """
    n_frames = batch.n_frames
    lattice = batch.metadata.get("Lattice")
    pbc = batch.metadata.get("pbc")
    if lattice is not None:
        flat = np.asarray(lattice, dtype=float)
        if flat.shape != (n_frames, 9):
            raise ToSimStateError(
                f"Lattice must have 9 components per frame, got shape {flat.shape}"
            )
        cell = flat.reshape(n_frames, 3, 3).transpose(0, 2, 1)
        pbc_array = _resolve_pbc(pbc, n_frames, default=True)
    else:
        cell = np.zeros((n_frames, 3, 3))
        pbc_array = _resolve_pbc(pbc, n_frames, default=False)
    return np.ascontiguousarray(cell), pbc_array


def _resolve_pbc(pbc: object, n_frames: int, *, default: bool) -> np.ndarray:
    """Return the one `(3,)` pbc every system shares, or raise if they differ."""
    if pbc is None:
        return np.full(3, default, dtype=bool)
    array = np.asarray(pbc)
    if array.ndim == 1 and array.shape[0] == n_frames:
        per_frame = np.repeat(array.astype(bool)[:, None], 3, axis=1)
    elif array.shape == (n_frames, 3):
        per_frame = array.astype(bool)
    else:
        raise ToSimStateError(
            f"pbc must be a scalar or 3 booleans per frame, got shape {array.shape}"
        )
    if n_frames and not np.all(per_frame == per_frame[0]):
        raise ToSimStateError(
            "frames disagree on pbc; a SimState shares one pbc across all systems"
        )
    return per_frame[0] if n_frames else np.full(3, default, dtype=bool)


def _extras(
    source: dict,
    mapping: ExtrasMap | None,
    dtype: torch.dtype | None,
    device: torch.device | None,
) -> dict[str, torch.Tensor]:
    """Build a `{simstate_key: tensor}` dict from a `{simstate_key: oxyz_key}` map."""
    if not mapping:
        return {}
    extras: dict[str, torch.Tensor] = {}
    for state_key, source_key in mapping.items():
        value = source.get(source_key)
        if value is None:
            raise ToSimStateError(f"extra source {source_key!r} missing from frames")
        extras[state_key] = to_tensor(np.asarray(value), source_key, dtype, device)
    return extras
