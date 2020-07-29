"""Tests for quantum averaging methods with a toy EMT system"""

import unittest
import numpy as np
import scipy.constants as cnst

from ase import Atoms
from ase.build import bulk
from ase.build.supercells import make_supercell
from ase.calculators.emt import EMT
from ase.dft.kpoints import monkhorst_pack
from ase.phonons import Phonons
from pymuonsuite.constants import m_mu_amu
from pymuonsuite.quantum.vibrational.phonons import ase_phonon_calc
from pymuonsuite.quantum.vibrational.schemes import (
    IndependentDisplacements, MonteCarloDisplacements)
from pymuonsuite.quantum.vibrational.harmonic import (harmonic_rho,
                                                      harmonic_rho_sum)


class TestQuantumEMT(unittest.TestCase):

    crystals = {}
    phonons = {}

    @classmethod
    def setUpClass(cls):

        N = 7

        # Define a bulk
        cls.crystals['Al'] = make_supercell(bulk('Al', 'fcc', a=4.05),
                                            np.eye(3)*2)
        for k, a in cls.crystals.items():

            # Add a muon
            a += Atoms('H', cell=a.get_cell(),
                       scaled_positions=[[0.5, 0.5, 0.5]])
            m = a.get_masses()
            m[-1] = m_mu_amu
            a.set_masses(m)

            a.calc = EMT(atoms=a)
            a.calc.initialize(a)

            phdata = ase_phonon_calc(a, a.calc, force_clean=2,
                                     name='test-{0}'.format(k))

            cls.crystals[k] = phdata.structure
            cls.crystals[k].calc = a.calc
            cls.phonons[k] = phdata

    def testIndependent(self):

        for k, a in self.crystals.items():
            phdata = self.phonons[k]
            displ = IndependentDisplacements(phdata.frequencies[0],
                                             phdata.modes[0],
                                             a.get_masses(), -1,
                                             sigma_n=0.1)
            displ.recalc_displacements()

            calcE = []

            for dxyz in displ.displacements:
                adispl = a.copy()
                adispl.calc = a.calc
                adispl.set_positions(a.get_positions()+dxyz)
                calcE.append(adispl.get_potential_energy())

            # Refer to equilibrium configuration
            calcE = np.array(calcE)-calcE[0]

            # Predicted harmonic energies
            harmE = displ.E

            # print(calcE)
            # print(harmE)
            # print(np.average(np.abs((harmE-calcE)/calcE)[1:]))
            self.assertLess(np.average(np.abs((harmE-calcE)/calcE)[1:]),
                            0.1)

    def testMontecarlo(self):

        for k, a in self.crystals.items():
            phdata = self.phonons[k]
            displ = MonteCarloDisplacements(phdata.frequencies[0],
                                            phdata.modes[0],
                                            a.get_masses())
            displ.recalc_displacements()

            calcE = []

            for dxyz in displ.displacements:
                adispl = a.copy()
                adispl.calc = a.calc
                adispl.set_positions(a.get_positions()+dxyz)
                calcE.append(adispl.get_potential_energy())

            # Refer to equilibrium configuration
            optE = a.get_potential_energy()
            calcE = np.array(calcE)-optE

            # Predicted harmonic energies
            harmE = displ.E

            # print(calcE)
            # print(harmE)
            # print(np.average(np.abs((harmE-calcE)/calcE)))
            self.assertLess(np.average(np.abs((harmE-calcE)/calcE)),
                            0.1)


if __name__ == "__main__":

    unittest.main()
