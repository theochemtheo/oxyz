from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from oxyz._convert import (
    SYMBOL_TO_Z,
    UnknownSpeciesError,
    species_to_numbers,
)

needs_ase = pytest.mark.skipif(
    importlib.util.find_spec("ase") is None, reason="ase not installed"
)


@needs_ase
def test_element_table_matches_ase() -> None:
    # The metatomic path maps species without importing ASE; the table must
    # nonetheless agree with ASE's, element for element.
    from ase.data import atomic_numbers

    assert atomic_numbers == SYMBOL_TO_Z


def test_species_to_numbers_maps_and_capitalises() -> None:
    out = species_to_numbers(["H", "si", "FE", "o"])
    assert out.dtype == np.int32
    assert out.tolist() == [1, 14, 26, 8]


def test_species_to_numbers_accepts_ndarray() -> None:
    out = species_to_numbers(np.array(["C", "C", "O"]))
    assert out.tolist() == [6, 6, 8]


def test_species_to_numbers_rejects_non_elements() -> None:
    with pytest.raises(UnknownSpeciesError, match="Zz"):
        species_to_numbers(["H", "Zz"])
