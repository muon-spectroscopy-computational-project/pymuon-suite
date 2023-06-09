"""Tests for hyperfine methods"""

import unittest

from ase import Atoms

import numpy as np

from pymuonsuite.calculate.hfine import compute_hfine_mullpop, compute_hfine_tensor


class TestHyperfine(unittest.TestCase):
    def test_compute_muon(self):
        hyperfine_tensor = compute_hfine_tensor(
            [[0.0, 0.0, 0.0]], [0.5], species=["mu"]
        )
        self.assertTrue(
            np.all(
                hyperfine_tensor == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            )
        )

    def test_compute_hydrogen(self):
        hyperfine_tensor = compute_hfine_tensor([[0.0, 0.0, 0.0]], [0], species=["H"])
        self.assertTrue(
            np.all(
                hyperfine_tensor == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            )
        )

    def test_mullpop_pbc(self):
        atoms = Atoms(["H"], pbc=True)
        hyperfine_tensor = compute_hfine_mullpop(
            atoms,
            [
                {
                    "q": 0,
                    "pop": 0,
                    "spin": 0,
                    "pop_orbital": {},
                    "spin_orbital": {},
                }
            ],
            fermi_neigh=True,
        )
        self.assertTrue(
            np.all(
                hyperfine_tensor == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            )
        )

    def test_mullpop_pbc_any(self):
        atoms = Atoms(pbc=[True, False, False])
        with self.assertRaises(ValueError) as e:
            compute_hfine_mullpop(atoms, None)
        self.assertIn("Partially periodic systems not implemented", str(e.exception))
