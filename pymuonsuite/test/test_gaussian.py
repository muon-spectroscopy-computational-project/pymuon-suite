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


def _test_write(params, input_file=None):
    ''' Tests writing a gaussian input file. If input_file (str) has been set,
        then the input file is written out with the settings that have been
        read from the file named input_file.
        Then reads this back in, checking that the structure
        has been set correctly, along with the muon mass and magnetic
        moment.
        Checks that the parameters of the resulting calculator (from reading)
        the file we have generated are equal to the params (dict)

        '''
    try:
        os.chdir(_TESTDATA_DIR)
        out_folder = "test_gaussian"
        os.mkdir(out_folder)
        sname = out_folder

        # These are the atoms we will write to our input file - ethylene
        # plus an extra H, which will be a muon:
        atoms = Atoms('C2H5', positions=_positions, cell=_cell, pbc=True)
        atoms.calc = Gaussian()

        atoms_copy = atoms.copy()
        atoms_copy.calc = copy.copy(atoms.calc)

        # We will read the parameters from the file: ethylene-SP.com
        gaussian_io = ReadWriteGaussian(
            params={'gaussian_input': input_file})

        gaussian_io.write(atoms, sname)

        # add muon properties to atoms to simulate what the write method
        # should be doing:

        masses = [None] * len(atoms.numbers)
        masses[-1] = constants.m_mu_amu
        atoms.set_masses(masses)

        NMagMs = [None] * len(atoms.numbers)
        NMagMs[-1] = constants.mu_nmagm
        params['nmagmlist'] = NMagMs
        atoms.calc.parameters = params

        # Read back in the gaussian input file that we wrote out.
        atoms_read = io.read(os.path.join(
            out_folder, 'test_gaussian.com'), attach_calculator=True)

        # The masses that are read get saved in the 'isolist' property
        # of the calculator. We have to retrieve them:
        atoms_read.set_masses(atoms_read.calc.parameters.pop('isolist'))

        # Checks properties of the atoms are written out correctly - if
        # they were then the atoms we have read should be the same as the
        #  atoms we originally created:
        assert np.all(atoms_read.numbers == atoms.numbers)
        assert np.allclose(atoms_read.positions,
                           atoms.positions, atol=1e-3)
        assert np.all(atoms_read.pbc == atoms.pbc)
        assert np.allclose(atoms_read.cell, atoms.cell)

        # checks that the muon properties have been correctly added by the
        # ReadWrite class:

        assert np.allclose(atoms_read.get_masses(), atoms.get_masses())

        new_params = atoms_read.calc.parameters

        # checks that the other settings from ethylene-SP.com have been
        # correctly read and written to the new input file:
        matching_params = {k: new_params[k] for k in new_params
                           if k in params and new_params[k] == params[k]}

        assert (len(params) == len(matching_params))
    finally:
        shutil.rmtree('test_gaussian')


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
        ''' Tests writing a gaussian input file, with the same settings as
        an existing file: ethylene-SP.com.
        Then reads this back in, checking that the structure and settings
        have been set correctly, along with the muon mass and magnetic
        moment.
        '''

        # These are the params that are in our file: ethylene-SP.com
        # We expect, and will be checking that, these have been written to
        # the gaussian input file we create.
        params = {'chk': 'ethylene-sp.chk', 'nprocshared': '16',
                  'output_type': 'p', 'b3lyp': None, 'epr-iii': None,
                  'charge': 0, 'mult': 2}

        _test_write(params, input_file='ethylene-SP.com')

    def test_write_default(self):
        ''' Tests writing a gaussian input file, without providing an input
            file from which to take the settings. Therefore, here we check
            that the default settings we expect have been applied.
            Then reads this back in, checking that the structure and settings
            have been set correctly, along with the muon mass and magnetic
            moment.
            '''

        # We expect that the following parameters should be written out to
        # the gaussian input file:
        sname = "test_gaussian"
        params = {'chk': '{}.chk'.format(sname), 'method': 'ub3lyp',
                  'basis': 'epr-iii',
                  'opt': 'tight,maxcyc=100', 'charge': 0, 'mult': 2}
        _test_write(params, input_file=None)


if __name__ == "__main__":

    unittest.main()
