"""metatomic conversion layer: extxyz frames to `metatomic.torch.System`.

`read`/`iread` mirror `oxyz.ase`'s entry points but yield `System`s; the
conversion reproduces `metatomic.torch.systems_to_torch(ase.io.read(...))`
without an ASE round-trip — species map to atomic numbers via this package's
own element table (`oxyz._convert`), and the cell follows the same Fortran-order
`Lattice` reshape and pbc-masked zeroing `systems_to_torch` applies.

`SystemSource` is the read-once handle: it parses a file once and serves both
`systems()` and array-native `per_config`/`per_atom` extraction of arbitrary
metadata/columns as torch tensors. It is built on per-frame `Frame`s rather than
a single concatenated `Batch`, so files whose metadata keys drift between frames
(common in training sets) still convert — each frame stands alone.

Optional dependencies `torch` and `metatomic-torch`, installed with
`pip install oxyz[metatomic]`.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, overload

import numpy as np

from oxyz._convert import UnknownSpeciesError
from oxyz._frames import (
    Compression,
    Frame,
    _read_all,
    _require_schema_for_mode,
)
from oxyz._rust import OxyzError
from oxyz._select import frames_for_read, nth_frame, parse_index, sliced_frames

try:
    import torch
    from metatomic.torch import System
except ImportError as error:
    raise ImportError(
        "oxyz.metatomic requires the optional dependencies 'torch' and "
        "'metatomic-torch'; install them with: pip install oxyz[metatomic]"
    ) from error

from oxyz._torch import MissingSpeciesError, numbers, resolve_dtype, to_tensor

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from oxyz._remote import StorageOptions
    from oxyz._schema_match import Conformance
    from oxyz._schema_spec import Mode, SchemaSpec

__all__ = ["SystemSource", "ToSystemError", "iread", "read"]

# systems_to_torch's warning, on the same condition, with "a frame" in place of
# its "an `ase.Atoms` object" — oxyz has no Atoms object to name. The shared
# "non-zero cell vectors" phrasing is what the parity test matches on.
_PBC_CELL_MISMATCH = (
    "A conversion to `System` was requested for a frame with one or more "
    "non-zero cell vectors but where the corresponding boundary conditions are "
    "set to `False`. The corresponding cell vectors will be set to zero."
)


class ToSystemError(OxyzError):
    """The frame has no faithful `System` representation (strict: no repair)."""


@overload
def read(
    path: str | Path,
    index: int,
    *,
    dtype: torch.dtype | None = ...,
    device: torch.device | None = ...,
    positions_requires_grad: bool = ...,
    cell_requires_grad: bool = ...,
    threads: int | None = ...,
    schema: SchemaSpec | str | Path | None = ...,
    conformance: Conformance = ...,
    mode: Mode | None = ...,
    compression: Compression = ...,
    member: str | None = ...,
    storage_options: StorageOptions | None = ...,
) -> System: ...


@overload
def read(
    path: str | Path,
    index: slice = ...,
    *,
    dtype: torch.dtype | None = ...,
    device: torch.device | None = ...,
    positions_requires_grad: bool = ...,
    cell_requires_grad: bool = ...,
    threads: int | None = ...,
    schema: SchemaSpec | str | Path | None = ...,
    conformance: Conformance = ...,
    mode: Mode | None = ...,
    compression: Compression = ...,
    member: str | None = ...,
    storage_options: StorageOptions | None = ...,
) -> list[System]: ...


@overload
def read(
    path: str | Path,
    index: str,
    *,
    dtype: torch.dtype | None = ...,
    device: torch.device | None = ...,
    positions_requires_grad: bool = ...,
    cell_requires_grad: bool = ...,
    threads: int | None = ...,
    schema: SchemaSpec | str | Path | None = ...,
    conformance: Conformance = ...,
    mode: Mode | None = ...,
    compression: Compression = ...,
    member: str | None = ...,
    storage_options: StorageOptions | None = ...,
) -> System | list[System]: ...


def read(  # noqa: PLR0913  keyword options mirror the System data model
    path: str | Path,
    index: int | str | slice = ":",
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    positions_requires_grad: bool = False,
    cell_requires_grad: bool = False,
    threads: int | None = None,
    schema: SchemaSpec | str | Path | None = None,
    conformance: Conformance = "required",
    mode: Mode | None = None,
    compression: Compression = "infer",
    member: str | None = None,
    storage_options: StorageOptions | None = None,
) -> System | list[System]:
    """Read frames into `System`s; default `index=":"` reads the whole file.

    An int selects one frame (returned bare); a slice or slice-string returns a
    list. Compressed paths are read too; a remote URL (``s3://``, ``gs://``,
    ``az://``) is read through the same parser.

    Parameters
    ----------
    path
        File path or an S3-compatible URL.
    index
        `":"` (default) reads every frame; an `int` returns one `System`; a
        slice or slice-string returns a list.
    dtype
        Float dtype for positions/cell. `None` follows
        `torch.get_default_dtype()`, as `systems_to_torch` does.
    device
        Target device for the returned tensors.
    positions_requires_grad
        Whether the returned `positions` tensor requires grad.
    cell_requires_grad
        Accepted for signature parity with `systems_to_torch`; not applied.
    threads
        Parallel parse for the whole-file read (`None`: all cores). No effect
        on bounded or reverse selections, which stream or seek.
    schema
        A `SchemaSpec`, or a path to a schema file, applied to the frames
        actually read. See `oxyz.SchemaSpec`.
    conformance
        How a schema deviation is handled: `"strict"`, `"required"`
        (default), or `"warn"`.
    mode
        Overrides the schema's own `mode`.
    compression
        Forces a codec instead of inferring it from `path`; as in `oxyz.read`.
    member
        Selects one entry from an archive holding more than one.
    storage_options
        Endpoint/credentials for a remote store (needs the ``oxyz[s3]``
        extra).

    Returns
    -------
    System or list[System]
        A single `System` for an integer index, otherwise a list.

    Raises
    ------
    ToSystemError
        If a frame has no faithful `System` representation.

    Examples
    --------
    >>> import torch
    >>> import oxyz.metatomic
    >>> systems = oxyz.metatomic.read("examples/data/water.extxyz", dtype=torch.float64)
    >>> len(systems)
    3
    """
    _require_schema_for_mode(schema, mode)
    options = (dtype, device, positions_requires_grad, cell_requires_grad)
    selection = parse_index(index)
    if isinstance(selection, int):
        frame = nth_frame(
            path,
            selection,
            schema=schema,
            conformance=conformance,
            mode=mode,
            compression=compression,
            member=member,
            storage_options=storage_options,
        )
        return _to_system(frame, *options)
    return [
        _to_system(frame, *options)
        for frame in frames_for_read(
            path,
            selection,
            threads,
            schema=schema,
            conformance=conformance,
            mode=mode,
            compression=compression,
            member=member,
            storage_options=storage_options,
        )
    ]


def iread(  # noqa: PLR0913  keyword options mirror the System data model
    path: str | Path,
    index: int | str | slice = ":",
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    positions_requires_grad: bool = False,
    cell_requires_grad: bool = False,
    schema: SchemaSpec | str | Path | None = None,
    conformance: Conformance = "required",
    mode: Mode | None = None,
    compression: Compression = "infer",
    member: str | None = None,
    storage_options: StorageOptions | None = None,
) -> Iterator[System]:
    """Stream `System`s one at a time, in constant memory (serial parse).

    Parameters
    ----------
    path
        File path or an S3-compatible URL.
    index
        `":"` (default) yields every frame; see `read`.
    dtype
        Float dtype for positions/cell; see `read`.
    device
        Target device for the returned tensors.
    positions_requires_grad
        Whether each `positions` tensor requires grad.
    cell_requires_grad
        Accepted for signature parity; not applied.
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
    Iterator[System]
        Systems in file order.

    Raises
    ------
    ToSystemError
        If a frame has no faithful `System` representation.
    """
    _require_schema_for_mode(schema, mode)
    options = (dtype, device, positions_requires_grad, cell_requires_grad)
    selection = parse_index(index)
    if isinstance(selection, int):
        frame = nth_frame(
            path,
            selection,
            schema=schema,
            conformance=conformance,
            mode=mode,
            compression=compression,
            member=member,
            storage_options=storage_options,
        )
        return iter((_to_system(frame, *options),))
    return (
        _to_system(frame, *options)
        for frame in sliced_frames(
            path,
            selection,
            schema=schema,
            conformance=conformance,
            mode=mode,
            compression=compression,
            member=member,
            storage_options=storage_options,
        )
    )


class SystemSource:
    """Read a file once; serve `System`s and per-key tensors from the result.

    Built for the case where a caller needs both the structures and several
    target arrays from one file (e.g. energy, forces, stress): one parse backs
    every accessor, so the file is never re-read.

    Parameters
    ----------
    path
        File path or an S3-compatible URL.
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
        self._frames: list[Frame] = _read_all(
            path,
            threads=threads,
            compression=compression,
            member=member,
            storage_options=storage_options,
        )

    def __len__(self) -> int:
        """Return the number of frames in the source."""
        return len(self._frames)

    def systems(
        self,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        positions_requires_grad: bool = False,
        cell_requires_grad: bool = False,
    ) -> list[System]:
        """Convert every frame to a `System`.

        Parameters
        ----------
        dtype
            Float dtype for positions/cell. `None` follows
            `torch.get_default_dtype()`.
        device
            Target device for the returned tensors.
        positions_requires_grad
            Whether each `positions` tensor requires grad.
        cell_requires_grad
            Accepted for signature parity; not applied.

        Returns
        -------
        list[System]
            One `System` per frame, in file order.

        Raises
        ------
        ToSystemError
            If a frame has no faithful `System` representation.
        """
        return [
            _to_system(
                frame, dtype, device, positions_requires_grad, cell_requires_grad
            )
            for frame in self._frames
        ]

    def per_config(
        self,
        key: str,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Stack one metadata value across frames: `(n_frames, *value_shape)`.

        Scalars give a 1-D tensor; arrays keep their width.

        Parameters
        ----------
        key
            The metadata key to extract.
        dtype
            Target tensor dtype. `None` resolves via `oxyz._torch.resolve_dtype`.
        device
            Target device.

        Returns
        -------
        torch.Tensor
            Shape `(n_frames, *value_shape)`.

        Raises
        ------
        ValueError
            If the source has no frames, the key is absent from any frame, or
            its shape drifts between frames.
        """
        self._require_frames(key)
        values = [
            _require(frame.metadata, key, "metadata", i)
            for i, frame in enumerate(self._frames)
        ]
        try:
            stacked = np.array(values)
        except ValueError:
            # numpy 2 raises on ragged input rather than building an object
            # array; surface the same message either way.
            raise ValueError(
                f"metadata {key!r} has inconsistent shapes across frames"
            ) from None
        if stacked.dtype == object:
            raise ValueError(f"metadata {key!r} has inconsistent shapes across frames")
        return to_tensor(stacked, key, resolve_dtype(dtype), device)

    def per_atom(
        self,
        key: str,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Concatenate one per-atom column across frames, with frame offsets.

        Parameters
        ----------
        key
            The column key to extract.
        dtype
            Target tensor dtype. `None` resolves via `oxyz._torch.resolve_dtype`.
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
            If the source has no frames or the key is absent from any frame.
        """
        self._require_frames(key)
        columns = [
            np.asarray(_require(frame.columns, key, "column", i))
            for i, frame in enumerate(self._frames)
        ]
        offsets = np.zeros(len(self._frames) + 1, dtype=np.intp)
        offsets[1:] = np.cumsum([frame.n_atoms for frame in self._frames])
        values = np.concatenate(columns, axis=0)
        return to_tensor(values, key, resolve_dtype(dtype), device), offsets

    def _require_frames(self, key: str) -> None:
        """Raise if the source has no frames to extract `key` from."""
        if not self._frames:
            raise ValueError(f"cannot extract {key!r}: the source has no frames")


def _require(mapping: dict, key: str, kind: str, frame_index: int) -> object:
    """Look up `key` in `mapping`, raising with `kind`/`frame_index` context."""
    if key not in mapping:
        raise ValueError(f"{kind} {key!r} missing from frame {frame_index}")
    return mapping[key]


def _to_system(
    frame: Frame,
    dtype: torch.dtype | None,
    device: torch.device | None,
    positions_requires_grad: bool,
    cell_requires_grad: bool,
) -> System:
    """Convert one frame, reproducing `systems_to_torch`'s cell/pbc handling."""
    resolved = resolve_dtype(dtype)

    pos = frame.columns.get("pos")
    if pos is None:
        raise ToSystemError("frame has no 'pos' column to use as positions")
    positions = torch.tensor(
        np.asarray(pos),
        dtype=resolved,
        device=device,
        requires_grad=positions_requires_grad,
    )

    types = torch.tensor(_numbers(frame), dtype=torch.int32, device=device)

    cell, pbc = _cell_and_pbc(frame)
    if not np.all(np.any(cell != 0, axis=1) == pbc):
        # stacklevel=3 (as systems_to_torch uses): point past _to_system and the
        # read/systems comprehension at the user's call.
        warnings.warn(_PBC_CELL_MISMATCH, stacklevel=3)
    pbc_tensor = torch.tensor(pbc, dtype=torch.bool, device=device)
    cell_tensor = torch.zeros((3, 3), dtype=resolved, device=device)
    # Mirror systems_to_torch: keep only the periodic cell vectors. (Like it, we
    # accept cell_requires_grad for signature parity but do not apply it.)
    cell_tensor[pbc_tensor] = torch.tensor(cell[pbc], dtype=resolved, device=device)

    return System(types=types, positions=positions, cell=cell_tensor, pbc=pbc_tensor)


def _numbers(frame: Frame) -> np.ndarray:
    """Atomic numbers, with the missing/unknown cases as `ToSystemError`s."""
    try:
        return numbers(frame.columns)
    except MissingSpeciesError:
        raise ToSystemError("frame has neither a 'species' nor a 'Z' column") from None
    except UnknownSpeciesError as error:
        raise ToSystemError(str(error)) from None


def _cell_and_pbc(frame: Frame) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct ASE's `(cell, pbc)`.

    A Fortran-order `Lattice`; pbc inferred from `Lattice` presence when not
    given explicitly.
    """
    pbc = frame.metadata.get("pbc")
    lattice = frame.metadata.get("Lattice")
    if lattice is not None:
        flat = np.asarray(lattice)
        if flat.shape != (9,):
            raise ToSystemError(
                f"Lattice must have 9 components, got shape {flat.shape}"
            )
        cell = flat.reshape((3, 3), order="F").T
        if pbc is None:
            pbc = np.array([True, True, True])
    else:
        cell = np.zeros((3, 3))
        if pbc is None:
            pbc = np.array([False, False, False])
    pbc_array = np.asarray(pbc, dtype=bool)
    if pbc_array.ndim == 0:
        # A scalar pbc (e.g. `pbc=T`) broadcasts to all three axes, as ASE does.
        pbc_array = np.full(3, bool(pbc_array))
    elif pbc_array.shape != (3,):
        raise ToSystemError(
            f"pbc must be a scalar or 3 booleans, got shape {pbc_array.shape}"
        )
    return np.ascontiguousarray(cell, dtype=float), pbc_array
