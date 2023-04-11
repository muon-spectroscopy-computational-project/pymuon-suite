"""Tests for utils"""

import unittest

from ase import Atoms

import numpy as np

from pymuonsuite.utils import make_muonated_supercell


class TestUtils(unittest.TestCase):
    def assert_supercell(
        self, supercell_atoms: Atoms, custom_symbol: str, x: int, y: int, z: int
    ):
        si_symbols = ["Si"] * x * y * z
        self.assertEqual(supercell_atoms.get_chemical_symbols(), si_symbols + ["H"])
        positions = []
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    positions.append([2 * i, 2 * j, 2 * k])
        positions += [[1, 1, 1]]
        self.assertTrue((supercell_atoms.get_positions() == positions).all())
        species = supercell_atoms.get_array("castep_custom_species")
        self.assertTrue(
            (species == si_symbols + [custom_symbol]).all(),
            f"Expected {si_symbols + [custom_symbol]} but got {species}",
        )

    def test_make_muonated_supercell(self):
        # Test equality when not changing cell size
        atoms = Atoms(["Si", "H"], [[0, 0, 0], [1, 1, 1]], cell=[2, 2, 2])
        supercell_atoms = make_muonated_supercell(atoms, 1, "H:mu")
        atoms.set_array("castep_custom_species", np.array(["Si", "H:mu"]))
        self.assertEqual(atoms, supercell_atoms)

        # Test int argument
        supercell_atoms = make_muonated_supercell(atoms, 2, "muon")
        self.assert_supercell(supercell_atoms, "muon", 2, 2, 2)

        # Test list argument
        supercell_atoms = make_muonated_supercell(atoms, [1, 1, 2], "list")
        self.assert_supercell(supercell_atoms, "list", 1, 1, 2)

        # Test 3*3 matrix (ndarray) argument
        supercell_atoms = make_muonated_supercell(atoms, np.diag([1, 1, 3]), "matrix")
        self.assert_supercell(supercell_atoms, "matr", 1, 1, 3)


if __name__ == "__main__":
    unittest.main()
