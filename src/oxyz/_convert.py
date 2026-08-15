"""ASE-independent chemistry and well-known-field helpers.

`oxyz.metatomic` must map species to atomic numbers without importing ASE — the
whole point is to read into torch without an ASE round-trip. The element table
here is the same construction ASE uses (symbol index = atomic number, with `"X"`
the dummy at 0), so it agrees with `ase.data.atomic_numbers` by construction; a
parity test pins the two equal.

The well-known-field resolvers (`positions`, `numbers`, `symbols`, `cell`,
`pbc`) take the plain `columns`/`metadata` dicts rather than a `Frame`, so a
`Batch`'s concatenated columns resolve through the same functions.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, cast

import numpy as np

from oxyz._rust import OxyzError

if TYPE_CHECKING:
    from oxyz._frames import ColumnValues, MetadataValue

# Index = atomic number; `"X"` (0) is ASE's dummy atom. Element order is the
# periodic table through Og (118).
CHEMICAL_SYMBOLS: tuple[str, ...] = (
    "X",
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
    "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra",
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc",
    "Lv", "Ts", "Og",
)  # fmt: skip

SYMBOL_TO_Z: dict[str, int] = {symbol: z for z, symbol in enumerate(CHEMICAL_SYMBOLS)}

# Standard atomic weights, index = atomic number, matching `ase.data.atomic_masses`
# (the IUPAC 2016 table ASE defaults to). The dummy atom `X` (Z=0) is 1.0, as ASE
# has it. `oxyz.torch_sim` needs masses without an ASE round-trip — extxyz rarely
# carries a `masses` column, so it derives them from atomic numbers here; a parity
# test pins this equal to `ase.data.atomic_masses`.
ATOMIC_MASSES: np.ndarray = np.array(
    [
        1.0, 1.008, 4.002602, 6.94, 9.0121831, 10.81,
        12.011, 14.007, 15.999, 18.998403163, 20.1797, 22.98976928,
        24.305, 26.9815385, 28.085, 30.973761998, 32.06, 35.45,
        39.948, 39.0983, 40.078, 44.955908, 47.867, 50.9415,
        51.9961, 54.938044, 55.845, 58.933194, 58.6934, 63.546,
        65.38, 69.723, 72.63, 74.921595, 78.971, 79.904,
        83.798, 85.4678, 87.62, 88.90584, 91.224, 92.90637,
        95.95, 97.90721, 101.07, 102.9055, 106.42, 107.8682,
        112.414, 114.818, 118.71, 121.76, 127.6, 126.90447,
        131.293, 132.90545196, 137.327, 138.90547, 140.116, 140.90766,
        144.242, 144.91276, 150.36, 151.964, 157.25, 158.92535,
        162.5, 164.93033, 167.259, 168.93422, 173.054, 174.9668,
        178.49, 180.94788, 183.84, 186.207, 190.23, 192.217,
        195.084, 196.966569, 200.592, 204.38, 207.2, 208.9804,
        208.98243, 209.98715, 222.01758, 223.01974, 226.02541, 227.02775,
        232.0377, 231.03588, 238.02891, 237.04817, 244.06421, 243.06138,
        247.07035, 247.07031, 251.07959, 252.083, 257.09511, 258.09843,
        259.101, 262.11, 267.122, 268.126, 271.134, 270.133,
        269.1338, 278.156, 281.165, 281.166, 285.177, 286.182,
        289.19, 289.194, 293.204, 293.208, 294.214,
    ]
)  # fmt: skip


def numbers_to_masses(atomic_numbers: np.ndarray) -> np.ndarray:
    """Return standard atomic weights for an array of atomic numbers.

    Looks up `ATOMIC_MASSES`. Callers pass numbers from `species_to_numbers`
    or a `Z` column, in range `0..118` for real elements.
    """
    return ATOMIC_MASSES[atomic_numbers]


def numbers_to_symbols(atomic_numbers: np.ndarray) -> list[str]:
    """Return chemical symbols for an array of atomic numbers.

    The inverse of `species_to_numbers` on the same `CHEMICAL_SYMBOLS` table.
    Validates explicitly rather than letting the table index raise, because a
    negative atomic number would silently wrap to the far end of the table.
    """
    values = atomic_numbers.tolist()
    unknown = sorted({z for z in values if not 0 <= z < len(CHEMICAL_SYMBOLS)})
    if unknown:
        raise UnknownSpeciesError(f"atomic numbers have no chemical symbol: {unknown}")
    return [CHEMICAL_SYMBOLS[z] for z in values]


class FieldError(OxyzError):
    """A well-known field is absent, or present in a shape it cannot hold.

    Raised by the derived `Frame` accessors (`positions`, `numbers`, `symbols`,
    `cell`, `pbc`) and by `oxyz.write` when a column's row count disagrees with
    the frame's `n_atoms`.
    """


class UnknownSpeciesError(FieldError):
    """A species token has no chemical-symbol mapping to an atomic number."""


class MissingSpeciesError(FieldError):
    """Columns carry no `Z`, `numbers`, or `species` to derive atomic numbers.

    Carries a default message so it reads sensibly when it escapes a `Frame`
    accessor unwrapped; each conversion target catches it and re-raises with
    its own wording.
    """

    def __init__(self, message: str = "no 'species', 'Z', or 'numbers' column") -> None:
        super().__init__(message)


@functools.cache
def _atomic_number(symbol: str) -> int:
    """Atomic number for a species token, capitalised (so `si` -> `Si`).

    Cached: a file repeats the same handful of species across every atom, so the
    capitalise and dict lookup happen once per distinct token, not once per atom.
    Raises `KeyError` for non-symbols, surfaced as `UnknownSpeciesError` by the
    caller.
    """
    return SYMBOL_TO_Z[symbol.capitalize()]


def species_to_numbers(symbols: np.ndarray | list) -> np.ndarray:
    """Map a species column to int32 atomic numbers via the cached per-token lookup.

    The capitalising lookup both conversion layers share: `oxyz.ase` and
    `oxyz.metatomic` call this rather than ASE's, on this module's own table so
    no ASE import is needed. Accepts the column as stored: a 1-D string array or
    `list[str]` maps; a multi-component `list[list[str]]` has no symbol mapping
    and raises.
    """
    # tolist() yields plain str, faster to iterate than ndarray's np.str_ scalars.
    species: list = symbols if isinstance(symbols, list) else symbols.tolist()
    try:
        return np.fromiter(
            (_atomic_number(s) for s in species),
            dtype=np.int32,
            count=len(species),
        )
    except (KeyError, TypeError):
        # KeyError: a token that is not a chemical symbol. TypeError: a non-scalar
        # species cell (a multi-component `species:S:2` column is list[list[str]],
        # unhashable, so the cached lookup cannot key on it). Either way the column
        # has no faithful symbol mapping.
        tokens = (str(s).capitalize() for s in species)
        unknown = sorted({t for t in tokens if t not in SYMBOL_TO_Z})
        raise UnknownSpeciesError(
            f"species are not chemical symbols: {unknown}"
        ) from None


def positions(columns: dict[str, ColumnValues]) -> np.ndarray:
    """Return the `pos` column as an array.

    The stored array itself, not a copy. Raises `FieldError` when the frame
    carries no `pos` column.
    """
    column = columns.get("pos")
    if column is None:
        raise FieldError("no 'pos' column to use as positions")
    return np.asarray(column)


def numbers(columns: dict[str, ColumnValues]) -> np.ndarray:
    """Return atomic numbers (int32) from columns.

    An explicit `Z`/`numbers` column wins, else the `species` column is
    mapped via the shared element table. Works on a single frame's per-atom
    columns or a batch's concatenated
    columns alike. Raises `MissingSpeciesError` when no usable column is
    present, `UnknownSpeciesError` when a species token has no chemical
    symbol, and `FieldError` when a `Z`/`numbers` column holds something that
    is not a number (a `Z:S:1` string column, say); the conversion layers map
    these to their own error type.
    """
    for name in ("Z", "numbers"):
        column = columns.get(name)
        if column is not None:
            try:
                values = np.asarray(column)
                if np.issubdtype(values.dtype, np.floating):
                    values = np.rint(values)  # a float Z column: round, don't truncate
                return values.astype(np.int32, copy=False)
            except (ValueError, TypeError) as error:
                raise FieldError(
                    f"column {name!r} holds no atomic numbers: {error}"
                ) from None

    species = columns.get("species")
    if species is None:
        raise MissingSpeciesError
    return species_to_numbers(species)


def symbols(columns: dict[str, ColumnValues]) -> list[str]:
    """Return chemical symbols for the frame's atoms.

    A `species` column wins and is returned as stored (the list itself when it
    is already a `list[str]`); otherwise the symbols are mapped back from
    `numbers`. Raises `MissingSpeciesError` when neither is present, and
    `FieldError` for a multi-component `species:S:2`-style column, which has
    no one symbol per atom, or for a `species` that is not a per-atom column
    at all.
    """
    species = columns.get("species")
    if species is None:
        return numbers_to_symbols(numbers(columns))
    values = species if isinstance(species, list) else np.asarray(species).tolist()
    if not isinstance(values, list):
        raise FieldError(
            f"'species' is not a per-atom column; got {type(species).__name__}"
        )
    if values and not isinstance(values[0], str):
        raise FieldError("'species' is multi-component; it has no symbol per atom")
    return cast("list[str]", values)


def cell(metadata: dict[str, MetadataValue]) -> np.ndarray:
    """Return the 3x3 cell in ASE's row-vector convention.

    The flat, Fortran-order `Lattice` reshaped and transposed; all-zero when
    `Lattice` is absent. Raises `FieldError` when `Lattice` is present with a
    shape other than `(9,)`, or holding something that is not numeric.
    """
    lattice = metadata.get("Lattice")
    if lattice is None:
        return np.zeros((3, 3))
    try:
        flat = np.asarray(lattice)
    except (ValueError, TypeError) as error:
        raise FieldError(f"Lattice is not an array: {error}") from None
    if flat.shape != (9,):
        raise FieldError(f"Lattice must have 9 components, got shape {flat.shape}")
    try:
        return np.ascontiguousarray(flat.reshape((3, 3), order="F").T, dtype=float)
    except (ValueError, TypeError) as error:
        raise FieldError(f"Lattice holds no numeric cell: {error}") from None


def pbc(metadata: dict[str, MetadataValue]) -> np.ndarray:
    """Return the three periodic-boundary flags.

    An explicit `pbc` defaults to all-true when `Lattice` is present and
    all-false when it is not; a scalar broadcasts to all three axes, as ASE
    does. Raises `FieldError` for any other shape.
    """
    value = metadata.get("pbc")
    if value is None:
        return np.full(3, metadata.get("Lattice") is not None, dtype=bool)
    try:
        array = np.asarray(value, dtype=bool)
    except (ValueError, TypeError) as error:
        raise FieldError(f"pbc is not a boolean value: {error}") from None
    if array.ndim == 0:
        return np.full(3, bool(array))
    if array.shape != (3,):
        raise FieldError(f"pbc must be a scalar or 3 booleans, got shape {array.shape}")
    return array
