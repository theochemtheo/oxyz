from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from oxyz._convert import (
    ATOMIC_MASSES,
    SYMBOL_TO_Z,
    UnknownSpeciesError,
    numbers_to_masses,
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


@needs_ase
def test_atomic_masses_match_ase() -> None:
    # The torch_sim path derives masses without importing ASE; the table must
    # agree with ASE's, element for element.
    from ase.data import atomic_masses

    np.testing.assert_array_equal(ATOMIC_MASSES, atomic_masses)


def test_numbers_to_masses_indexes_by_atomic_number() -> None:
    masses = numbers_to_masses(np.array([1, 8, 6]))
    np.testing.assert_allclose(masses, [1.008, 15.999, 12.011])


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
