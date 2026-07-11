"""torch_sim conversion layer: extxyz frames to `torch_sim.SimState`.

`SimState` is natively batched — one state holds many systems with their atoms
concatenated — so this layer maps onto oxyz's batched reader rather than the
per-frame path the `oxyz.ase` / `oxyz.metatomic` targets use:

- `read` returns a *single* batched `SimState` (the whole selection as one
  batch). With a model and a GPU, hand it straight to `torch_sim`'s
  `BinningAutoBatcher`, which sizes memory-aware batches by probing the model.
- `iread` streams the file as a sequence of `SimState` batches, one per step,
  for files too large to materialise at once. It forwards the binning knobs of
  `oxyz.iread_batch` (`frames_per_batch` / `atoms_per_batch` /
  `memory_scales_with` + `max_scaler`); pick exactly one.
- `SimStateSource` parses a file once and serves the state plus array-native
  `per_config` / `per_atom` tensor extraction.

The conversion reproduces `torch_sim.io.atoms_to_state(ase.io.read(...))`: cells
in `torch_sim`'s column-vector convention (ASE's cell transposed), a single pbc
shared by every system (frames that disagree are an error), masses from a
`masses` column or, failing that, derived from atomic numbers via this package's
ASE-parity table. Extras are opt-in, mirroring `atoms_to_state`'s maps.

Optional dependencies `torch` and `torch-sim-atomistic`, installed with
`pip install oxyz[torch-sim]`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import numpy as np

from oxyz._batch import Batch, MemoryScaling, iread_batch, read_batch
from oxyz._convert import UnknownSpeciesError, numbers_to_masses
from oxyz._frames import _require_schema_for_mode
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

from oxyz._torch import MissingSpeciesError, numbers, to_tensor

if TYPE_CHECKING:
    from pathlib import Path

    from oxyz._frames import Compression
    from oxyz._schema_match import Conformance
    from oxyz._schema_spec import Mode, SchemaSpec

__all__ = ["SimStateSource", "ToSimStateError", "iread", "read"]

ExtrasMap = Mapping[str, str]


class ToSimStateError(ValueError):
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
) -> SimState:
    """Read selected frames into one batched `SimState`; default reads the file.

    An int selects one frame (a state with `n_systems == 1`); a slice or
    slice-string selects several, batched into a single state. `dtype=None`
    infers from the data (so positions are float64), matching `atoms_to_state`
    rather than `torch.get_default_dtype()`; pass `torch.float32` for ML use.
    `system_extras`
    / `atom_extras` are `{simstate_key: oxyz_key}` maps pulling frame metadata
    into `_system_extras` and per-atom columns into `_atom_extras`. `threads`
    sets the parallel parse (`None`: all cores).

    Compressed paths are read too (any index: the scan and the selecting read
    both stream); `compression` and `member` are as in `oxyz.read`.
    """
    _require_schema_for_mode(schema, mode)
    indices = _plan_indices(path, index, compression=compression, member=member)
    batch = read_batch(
        path,
        ":" if indices is None else indices,
        threads=threads,
        schema=schema,
        conformance=conformance,
        mode=mode,
        compression=compression,
        member=member,
    )
    return _to_state(
        batch, dtype, device, positions_requires_grad, system_extras, atom_extras
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
) -> Iterator[SimState]:
    """Stream the file as `SimState` batches, one per step.

    Pick exactly one binning strategy, as in `oxyz.iread_batch`: a fixed
    `frames_per_batch`, a greedy `atoms_per_batch`, or balanced memory-aware
    bins (`memory_scales_with` + `max_scaler`). For a model-aware split prefer
    `read` plus `torch_sim`'s `BinningAutoBatcher`, which probes the model.

    A compressed source supports only `frames_per_batch` without `shuffle`
    (it cannot be randomly accessed); see `oxyz.iread_batch`. `compression` and
    `member` are as in `oxyz.read`.
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
    ):
        yield _to_state(
            batch, dtype, device, positions_requires_grad, system_extras, atom_extras
        )


class SimStateSource:
    """Read a file once; serve a batched `SimState` and per-key tensors from it.

    Built for the case where a caller needs both the state and several target
    arrays (energy, forces, stress) from one file: one parse backs every
    accessor, so the file is never re-read. Like every oxyz `Batch`, the frames
    must share a schema — a file whose metadata keys drift between frames cannot
    form one batch.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        threads: int | None = None,
        compression: Compression = "infer",
        member: str | None = None,
    ) -> None:
        self._batch: Batch = read_batch(
            path, threads=threads, compression=compression, member=member
        )

    def __len__(self) -> int:
        return self._batch.n_frames

    def state(
        self,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        positions_requires_grad: bool = False,
        system_extras: ExtrasMap | None = None,
        atom_extras: ExtrasMap | None = None,
    ) -> SimState:
        """Convert the whole file to one batched `SimState`."""
        return _to_state(
            self._batch,
            dtype,
            device,
            positions_requires_grad,
            system_extras,
            atom_extras,
        )

    def per_config(
        self,
        key: str,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """One metadata value across frames as a tensor: `(n_frames, *shape)`."""
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

        `values` is `(total_atoms, *width)` and `offsets` (length `n_frames + 1`)
        marks each frame's rows, so frame `i` is `values[offsets[i]:offsets[i+1]]`.
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
) -> list[int] | None:
    """Resolve an ASE-style index to a list of frame indices, or `None` for the
    whole file (read in a single pass)."""
    selection = parse_index(index)
    if isinstance(selection, slice) and selection == slice(None, None, None):
        return None
    universe = range(scan(path, compression=compression, member=member).n_frames)
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
    positions_requires_grad: bool,
    system_extras: ExtrasMap | None,
    atom_extras: ExtrasMap | None,
) -> SimState:
    # dtype=None infers from the (float64) arrays, matching atoms_to_state; it
    # does not resolve to torch.get_default_dtype() the way systems_to_torch does.
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
    """Per-atom masses: a `masses` column wins, else standard atomic weights.

    `SimState` requires masses; extxyz rarely carries them, so they are derived
    from atomic numbers via the ASE-parity table when no column is present.
    """
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


def _cell_and_pbc(batch: Batch) -> tuple[np.ndarray, np.ndarray]:
    """Per-system cells `(n_frames, 3, 3)` in torch_sim's column convention, and
    a single pbc `(3,)` shared by every system.

    torch_sim stores the ASE cell transposed; an extxyz `Lattice` is the ASE
    cell flattened in Fortran order, so the column-convention cell is the plain
    C-order reshape transposed per frame. A frame with no `Lattice` contributes
    a zero cell.
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
    """The single `(3,)` pbc every system shares, or a `ToSimStateError` when
    frames disagree (a `SimState` has one pbc for the whole batch)."""
    if pbc is None:
        return np.full(3, default, dtype=bool)
    array = np.asarray(pbc)
    if array.ndim == 1 and array.shape[0] == n_frames:
        # one scalar pbc per frame; each broadcasts to all three axes, as ASE has it
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
    """Build a `{simstate_key: tensor}` extras dict from a `{simstate_key:
    oxyz_key}` map over a batch's metadata or columns."""
    if not mapping:
        return {}
    extras: dict[str, torch.Tensor] = {}
    for state_key, source_key in mapping.items():
        value = source.get(source_key)
        if value is None:
            raise ToSimStateError(f"extra source {source_key!r} missing from frames")
        extras[state_key] = to_tensor(np.asarray(value), source_key, dtype, device)
    return extras
