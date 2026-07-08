"""ASE-independent chemistry helpers shared by the conversion layers.

`oxyz.metatomic` must map species to atomic numbers without importing ASE — the
whole point is to read into torch without an ASE round-trip. The element table
here is the same construction ASE uses (symbol index = atomic number, with `"X"`
the dummy at 0), so it agrees with `ase.data.atomic_numbers` by construction; a
parity test pins the two equal.
"""

from __future__ import annotations

import functools

import numpy as np

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
    """Standard atomic weights for an array of atomic numbers, via `ATOMIC_MASSES`.

    Callers pass numbers from `species_to_numbers` or a `Z` column, in range
    `0..118` for real elements.
    """
    return ATOMIC_MASSES[atomic_numbers]


class UnknownSpeciesError(ValueError):
    """A species token has no chemical-symbol mapping to an atomic number."""


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
