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

# Set to False by default because we haven't included a gaussian
# output file in pymuonsuite:
_TEST_READ = False


def _test_write(expected_params, input_file=None, calc=None, atoms_calc=None):
    ''' Tests writing a gaussian input file. If input_file (str) has been set,
        then the input file is written out with the settings that have been
        read from the file named input_file.
        If calc (gaussian calculator) is set, this is given as input to the
        ReadWrite class and its parameters should be written out to the input
        file.
        If atoms_calc (gaussian calculator) is set, this is attached to the
        atoms object given to the write method in the ReadWrite class, and its
        parameters sould be written out to the output file (same behaviour as
        above)
        Then reads this back in, checking that the structure
        has been set correctly, along with the muon mass and magnetic
        moment.
        Checks that the parameters of the resulting calculator (from reading
        the file we have generated) are equal to the expected_params (dict)

        '''
    try:
        os.chdir(_TESTDATA_DIR)
        out_folder = "test_gaussian"
        os.mkdir(out_folder)
        sname = out_folder

        # These are the atoms we will write to our input file - ethylene
        # plus an extra H, which will be a muon:
        atoms = Atoms('C2H5', positions=_positions, cell=_cell, pbc=True)
        if atoms_calc is not None:
            atoms.calc = atoms_calc

        atoms_copy = atoms.copy()
        atoms_copy.calc = copy.copy(atoms.calc)

        # We will read the parameters from the file: ethylene.com
        gaussian_io = ReadWriteGaussian(
            params={'gaussian_input': input_file}, calc=calc)

        gaussian_io.write(atoms, sname)

        # add muon properties to atoms to simulate what the write method
        # should be doing:

        atoms.calc = Gaussian()

        masses = [None] * len(atoms.numbers)
        masses[-1] = constants.m_mu_amu
        atoms.set_masses(masses)

        NMagMs = [None] * len(atoms.numbers)
        NMagMs[-1] = constants.mu_nmagm
        expected_params['nmagmlist'] = NMagMs
        atoms.calc.parameters = expected_params

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

        for key, value in expected_params.items():
            params_equal = expected_params.get(
                key) == new_params.get(key)
            if isinstance(params_equal, np.ndarray):
                assert((expected_params.get(
                    key) == new_params.get(key)).all())
            else:
                assert(expected_params.get(
                    key) == new_params.get(key))

    finally:
        shutil.rmtree('test_gaussian')


class TestReadWriteGaussian(unittest.TestCase):

    def test_read(self):
        ''' Tests reading a gaussian output file, checking
        that the positions, atomic numbers, pbc and fermi
        contact term have been correctly read'''
        if _TEST_READ:
            reader = ReadWriteGaussian()
            atoms = Atoms('C2H5', positions=_positions)
            atoms_read = reader.read(".", 'ethylene-mu')

            assert np.all(atoms_read.numbers == atoms.numbers)
            assert np.allclose(atoms_read.positions,
                               atoms.positions, atol=1e-3)
            assert np.all(atoms_read.pbc == atoms.pbc)
            assert np.allclose(atoms_read.cell, atoms.cell)

            expected_hyperfine = '163.07409'
            hyperfine = reader.read(".", 'ethylene-mu',
                                    read_hyperfine=True).get_array(
                                        'hyperfine')[-1]
            assert(hyperfine == expected_hyperfine)

    def test_write(self):
        ''' Tests writing a gaussian input file, with the same settings as
        an existing file: ethylene.com.
        Then reads this back in, checking that the structure and settings
        have been set correctly, along with the muon mass and magnetic
        moment.
        '''

        # These are the params that are in our file: ethylene.com
        # We expect, and will be checking that, these have been written to
        # the gaussian input file we create.
        params = {'chk': 'ethylene.chk', 'nprocshared': '16',
                  'output_type': 'p', 'b3lyp': None, 'epr-iii': None,
                  'charge': 0, 'mult': 2}

        _test_write(params, input_file='ethylene.com')

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

    def test_calc(self):
        ''' Tests writing a gaussian input file when a Gaussian calculator is
        provided. '''

        # First test the case where we give the calculator as an input to
        # the ReadWrite class.
        params = {'chk': 'ethylene.chk', 'nprocshared': '16',
                  'output_type': 'p', 'b3lyp': None, 'epr-iii': None,
                  'charge': 0, 'mult': 2}
        calc = Gaussian(**params)
        _test_write(params, calc=calc)

        # Next test the case where we attach the calculator to the atoms
        # object.
        calc = Gaussian(**params)
        _test_write(params, atoms_calc=calc)

        # In the case where we set both a calculator and an input file, we
        # expect the input file to take precedence, and the parameters will
        # be set from there instead of from the calculator.
        params = {'chk': 'ethylene.chk', 'nprocshared': '16',
                  'output_type': 'p', 'b3lyp': None, 'epr-iii': None,
                  'charge': 0, 'mult': 2}

        _test_write(params, input_file='ethylene.com', calc=calc)


if __name__ == "__main__":
    unittest.main()
