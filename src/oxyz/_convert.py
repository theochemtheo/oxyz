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
    species: list = symbols.tolist() if isinstance(symbols, np.ndarray) else symbols
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
