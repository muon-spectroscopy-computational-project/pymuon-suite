"""Tests for quantum averaging methods"""


import unittest
import numpy as np
import scipy.constants as cnst

from pymuonsuite.quantum.vibrational.schemes import (
    IndependentDisplacements,
    MonteCarloDisplacements,
)
from pymuonsuite.quantum.vibrational.harmonic import (
    harmonic_rho,
    harmonic_rho_sum,
)


class TestDisplacements(unittest.TestCase):

    # Create mock eigenvalues and eigenvectors
    evals = (
        np.ones(3) * cnst.hbar / cnst.u / (2 * np.pi * cnst.c) * 1e18
    )  # Value calculated to give a sigma of 1 Ang

    evecs = np.eye(3)[:, None, :]
    masses = np.ones(1)

    # A test quantity to average
    @staticmethod
    def _A(xyz):
        return 1.3 + xyz[:, 0] ** 2 + 1.5 * xyz[:, 1] ** 4 + np.exp(0.3 * xyz[:, 2])

    # Volume averaged value (for reference)
    @classmethod
    def _A_expect(self, gridX=5, gridN=51, sigmas=np.ones(3), xi=np.zeros(3)):
        grid = (
            np.array(
                np.meshgrid(*[np.linspace(-gridX, gridX, gridN)] * 3, indexing="ij")
            )
            .reshape((3, -1))
            .T
        )
        Avol = self._A(grid)
        Gvol = np.exp(
            -np.sum(
                (grid / sigmas[None, :]) ** 2
                * ((1.0 - xi**2) / (1 + xi**2))[None, :],
                axis=1,
            )
        )
        avgvol = np.sum(Avol * Gvol) / np.sum(Gvol)

        return avgvol

    def assertSmallRelativeError(self, x0, x1, tol):
        err = abs((x1 - x0) / x0)
        if err > tol:
            raise AssertionError("Error {0} is bigger than {1}".format(err, tol))

    def testHarmonic(self):
        # Simple tests for the core theory functions

        x = np.linspace(-5.0, 5, 100) * 1e-10
        m = cnst.u
        om = 1e20 * cnst.hbar / m

        rhos = harmonic_rho_sum(x, m, om)
        rhot = harmonic_rho(x, m, om)

        self.assertAlmostEqual(np.trapz(rhos, x), 1)
        self.assertAlmostEqual(np.trapz(rhot, x), 1)

        self.assertTrue(np.average((rhos - rhot) ** 2) < 1e-3)

        # Now with non-zero temperature...
        T = 0.7 * cnst.hbar * om / cnst.k

        rhos = harmonic_rho_sum(x, m, om, T)
        rhot = harmonic_rho(x, m, om, T)

        self.assertAlmostEqual(np.trapz(rhos, x), 1)
        self.assertAlmostEqual(np.trapz(rhot, x), 1)

        self.assertTrue(np.average((rhos - rhot) ** 2) < 1e-3)

    def testIndependent(self):

        scheme = IndependentDisplacements(
            self.evals, self.evecs, self.masses, 0, sigma_n=5
        )
        self.assertTrue(np.isclose(scheme._sigmas, cnst.u**0.5).all())

        # Volumetric averaging
        avgvol = self._A_expect(5, 51, scheme._sigmas * 1e10 / cnst.u**0.5)

        # Test even number of points
        scheme.recalc_displacements(n=20)
        scheme.recalc_weights()
        displ = scheme.displacements
        weights = scheme.weights
        Adspl = self._A(displ[:, 0])
        avgdispl = np.sum(Adspl * weights)

        self.assertSmallRelativeError(avgvol, avgdispl, 1e-2)

        # Test odd number of points
        scheme.recalc_displacements(n=21)
        scheme.recalc_weights()
        displ = scheme.displacements
        weights = scheme.weights
        Adspl = self._A(displ[:, 0])
        avgdispl = np.sum(Adspl * weights)

        self.assertSmallRelativeError(avgvol, avgdispl, 1e-2)

        # Non-zero temperature
        T = 50
        E = cnst.hbar * self.evals * 1e2 * cnst.c * 2 * np.pi
        xi = np.exp(-0.5 * E / (cnst.k * T))
        avgvolT = self._A_expect(5, 51, scheme._sigmas * 1e10 / cnst.u**0.5, xi)

        weights = scheme.recalc_weights(T=T)
        avgdisplT = np.sum(Adspl * weights)

        self.assertSmallRelativeError(avgvolT, avgdisplT, 1e-2)

    def testMontecarlo(self):

        scheme = MonteCarloDisplacements(self.evals, self.evecs, self.masses)

        avgvol = self._A_expect(5, 51, scheme._sigmas * 1e10 / cnst.u**0.5)

        displ = scheme.recalc_displacements(n=100000)
        weights = scheme.recalc_weights()
        Adspl = self._A(displ[:, 0])
        avgdispl = np.sum(Adspl * weights)

        # MonteCarlo accuracy is low
        self.assertSmallRelativeError(avgvol, avgdispl, 1e-1)

        # Non-zero temperature
        T = 50
        E = cnst.hbar * self.evals * 1e2 * cnst.c * 2 * np.pi
        xi = np.exp(-0.5 * E / (cnst.k * T))
        avgvolT = self._A_expect(5, 51, scheme._sigmas * 1e10 / cnst.u**0.5, xi)

        displ = scheme.recalc_displacements(n=100000, T=T)
        weights = scheme.recalc_weights(T=T)
        Adspl = self._A(displ[:, 0])
        avgdisplT = np.sum(Adspl * weights)

        self.assertSmallRelativeError(avgvolT, avgdisplT, 1e-1)

        # Reweighed
        weights = scheme.recalc_weights()
        Adspl = self._A(displ[:, 0])
        avgdisplRW = np.sum(Adspl * weights)

        self.assertSmallRelativeError(avgvol, avgdisplRW, 1e-1)


if __name__ == "__main__":

    unittest.main()
