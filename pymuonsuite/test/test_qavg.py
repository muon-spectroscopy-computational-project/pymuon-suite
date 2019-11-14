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
    IndependentDisplacements, MonteCarloDisplacements)


class TestDisplacements(unittest.TestCase):

    # Create mock eigenvalues and eigenvectors
    evals = (np.ones(3)*cnst.hbar/cnst.u/(2*np.pi*cnst.c) *
             1e18)  # Value calculated to give a sigma of 1 Ang
    evecs = np.eye(3)[:, None, :]
    masses = np.ones(1)

    # A test quantity to average
    @staticmethod
    def _A(xyz):
        return 1.3+xyz[:, 0]**2+1.5*xyz[:, 1]**4+np.exp(0.3*xyz[:, 2])

    # Volume averaged value (for reference)
    @classmethod
    def _A_expect(self, gridX=5, gridN=51, sigmas=np.ones(3)):
        grid = np.array(np.meshgrid(
            *[np.linspace(-gridX, gridX, gridN)]*3,
            indexing='ij')).reshape((3, -1)).T
        Avol = self._A(grid)
        Gvol = np.exp(-np.sum((grid/sigmas[None,:])**2, axis=1))
        avgvol = np.sum(Avol*Gvol)/np.sum(Gvol)

        return avgvol

    def test_independent(self):

        scheme = IndependentDisplacements(self.evals, self.evecs,
                                          self.masses, 0)
        self.assertTrue(np.isclose(scheme._sigmas, cnst.u**0.5).all())

        # Volumetric averaging
        avgvol = self._A_expect(5, 51, scheme._sigmas*1e10/cnst.u**0.5)

        # Test even number of points
        scheme.recalc_displacements(n=20)
        scheme.recalc_weights()
        displ = scheme.displacements
        weights = scheme.weights
        Adspl = self._A(displ[:, 0])
        avgdispl = np.sum(Adspl*weights)

        self.assertAlmostEqual(avgvol, avgdispl, 2)

        # Test odd number of points
        scheme.recalc_displacements(n=21)
        scheme.recalc_weights()
        displ = scheme.displacements
        weights = scheme.weights
        Adspl = self._A(displ[:, 0])
        avgdispl = np.sum(Adspl*weights)

        self.assertAlmostEqual(avgvol, avgdispl, 2)

    def test_montecarlo(self):

        scheme = MonteCarloDisplacements(self.evals, self.evecs,
                                         self.masses)

        avgvol = self._A_expect(5, 51, scheme._sigmas*1e10/cnst.u**0.5)

        displ = scheme.recalc_displacements(n=100000)
        weights = scheme.recalc_weights()
        Adspl = self._A(displ[:, 0])
        avgdispl = np.sum(Adspl*weights)

        # MonteCarlo accuracy is low
        self.assertAlmostEqual(avgvol, avgdispl, 1)



if __name__ == "__main__":

    unittest.main()
