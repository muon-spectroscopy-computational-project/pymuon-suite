"""Tests for ReadWriteGaussian methods"""

import unittest
import numpy as np

import os
import shutil
import copy

from ase import Atoms, io
from ase.calculators.gaussian import Gaussian

from pymuonsuite import constants
from pymuonsuite.io.gaussian import ReadWriteGaussian


_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTDATA_DIR = os.path.join(_TEST_DIR, "test_data/ethyleneMu")
os.chdir(_TESTDATA_DIR)

_positions = [[0.241132,  0.,        0.708084],
              [-0.04628,  -0.,       -0.747689],
              [-0.164565, 0.885014,  1.20288],
              [-0.164565, -0.885014,  1.20288],
              [-0.094178,  0.924421, -1.305393],
              [-0.094178, -0.924421, -1.305393],
              [1.322635,  0.,        0.91211]]

_cell = [[10, 0, 0], [0, 10, 0], [0, 0, 10]]


class TestReadWriteGaussian(unittest.TestCase):

    def test_read(self):
        ''' Tests reading a gaussian output file, checking
        that the positions, atomic numbers, pbc and fermi
        contact term have been correctly read'''
        reader = ReadWriteGaussian()
        atoms = Atoms('C2H5', positions=_positions)
        atoms_read = reader.read(".", 'ethylene-mu')

        assert np.all(atoms_read.numbers == atoms.numbers)
        assert np.allclose(atoms_read.positions, atoms.positions, atol=1e-3)
        assert np.all(atoms_read.pbc == atoms.pbc)
        assert np.allclose(atoms_read.cell, atoms.cell)

        expected_hyperfine = '163.07409'
        hyperfine = reader.read(".", 'ethylene-mu',
                                read_hyperfine=True).get_array('hyperfine')[-1]
        assert(hyperfine == expected_hyperfine)

    def test_write(self):
        try:
            ''' Tests writing a gaussian input file, then reads this back in,
            checking that the structure has been set correctly, along with the
            muon mass and magnetic moment.
            Checks all of the calculator's properties have been set as we
            expect.'''

            os.chdir(_TESTDATA_DIR)
            out_folder = "test_gaussian"
            os.mkdir(out_folder)
            sname = out_folder
            atoms = Atoms('C2H5', positions=_positions, cell=_cell, pbc=True)
            atoms.calc = Gaussian()
            params = {'chk': 'ethylene-sp.chk', 'nprocshared': '16',
                      'output_type': 'p', 'b3lyp': None, 'epr-iii': None,
                      'charge': 0, 'mult': 2}

            gaussian_io = ReadWriteGaussian(
                params={'gaussian_input': 'ethylene-SP.com'})

            atoms_copy = atoms.copy()
            atoms_copy.calc = copy.copy(atoms.calc)
            gaussian_io.write(atoms_copy, out_folder, sname)

            # add muon properties to atoms to simulate what the write method
            # should be doing:

            masses = [None] * len(atoms.numbers)
            masses[-1] = constants.m_mu_amu
            atoms.set_masses(masses)

            NMagMs = [None] * len(atoms.numbers)
            NMagMs[-1] = constants.mu_nmagm
            params['nmagmlist'] = NMagMs
            atoms.calc.parameters = params

            # check if OG atoms should be modified if we don't save a copy
            atoms_read = io.read(os.path.join(
                out_folder, 'test_gaussian.com'), attach_calculator=True)

            # The masses that are read get saved in the 'isolist' property
            # of the calculator. We have to retrieve them:
            atoms_read.set_masses(atoms_read.calc.parameters.pop('isolist'))

            # Checks properties of the atoms are written out correctly
            assert np.all(atoms_read.numbers == atoms.numbers)
            assert np.allclose(atoms_read.positions,
                               atoms.positions, atol=1e-3)
            assert np.all(atoms_read.pbc == atoms.pbc)
            assert np.allclose(atoms_read.cell, atoms.cell)

            # checks that the muon properties have been correctly added by the
            # ReadWrite class:

            assert np.allclose(atoms_read.get_masses(), atoms.get_masses())

            new_params = atoms_read.calc.parameters

            matching_params = {k: new_params[k] for k in new_params
                               if k in params and new_params[k] == params[k]}

            assert (len(params) == len(matching_params))

        finally:
            shutil.rmtree('test_gaussian')


if __name__ == "__main__":

    unittest.main()
