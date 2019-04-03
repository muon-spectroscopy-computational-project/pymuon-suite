"""Tests for quantum averaging methods"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
import scipy.constants as cnst

from pymuonsuite.quantum.vibrational.schemes import (
    IndependentDisplacements,)


class TestDisplacements(unittest.TestCase):

    def test_independent(self):

        # Create mock eigenvalues and eigenvectors
        evals = (np.ones(3)*cnst.hbar/cnst.u/(2*np.pi*cnst.c) *
                 1e18)  # Value calculated to give a sigma of 1 Ang
        evecs = np.eye(3)[:, None, :]
        masses = np.ones(3)

        scheme = IndependentDisplacements(evals, evecs, masses, 0)
        self.assertTrue(np.isclose(scheme._sigmas, cnst.u**0.5).all())

        # A test quantity to average
        def A(xyz):
            return 1.3+xyz[:, 0]**2+1.5*xyz[:, 1]**4+np.exp(0.3*xyz[:, 2])

        # Volumetric averaging
        grid = np.array(np.meshgrid(
            *[np.linspace(-5, 5, 51)]*3,
            indexing='ij')).reshape((3, -1)).T
        Avol = A(grid)
        Gvol = np.exp(-np.sum(grid**2, axis=1))
        avgvol = np.sum(Avol*Gvol)/np.sum(Gvol)

        # Test even number of points
        scheme.recalc_displacements(n=20)
        scheme.recalc_weights()
        displ = scheme.displacements
        weights = scheme.weights
        Adspl = A(displ[:, 0])
        avgdispl = np.sum(Adspl*weights)

        self.assertAlmostEqual(avgvol, avgdispl, 2)

        # Test odd number of points
        scheme.recalc_displacements(n=21)
        scheme.recalc_weights()
        displ = scheme.displacements
        weights = scheme.weights
        Adspl = A(displ[:, 0])
        avgdispl = np.sum(Adspl*weights)

        self.assertAlmostEqual(avgvol, avgdispl, 2)


if __name__ == "__main__":

    unittest.main()
