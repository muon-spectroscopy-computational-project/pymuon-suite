"""Tests for ReadWriteCastep methods"""

import unittest
import numpy as np

import os
import sys
import shutil

from copy import deepcopy

from ase import Atoms, io
from ase.io.castep import read_param

from pymuonsuite.utils import list_to_string
from pymuonsuite.io.castep import ReadWriteCastep
from pymuonsuite.schemas import load_input_file, MuAirssSchema


_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTDATA_DIR = os.path.join(_TEST_DIR, "test_data")
_TESTSAVE_DIR = os.path.join(_TEST_DIR, "test_save")


class TestReadWriteCastep(unittest.TestCase):

    def test_read(self):
        seednames = ['ethyleneMu', None]
        for sname in seednames:
            folder = _TESTDATA_DIR  # does not contain any castep files
            reader = ReadWriteCastep()
            # test that we do not get any result for trying to read
            # an empty folder:
            self.assertFalse(reader.read(folder, sname, calc_type="GEOM_OPT"))
            self.assertFalse(reader.read(folder, sname, calc_type="MAGRES"))
            self.assertFalse(reader.read(folder, sname, calc_type="PHONONS"))

            folder = os.path.join(_TESTDATA_DIR, "castep")
            # tests castep file being read:
            self.assertTrue(reader.read(folder, sname, calc_type="GEOM_OPT"))
            atom_arrays_castep = deepcopy(reader.atoms.arrays)

            # tests hyperfine being read:
            self.assertNotIn('hyperfine', atom_arrays_castep)
            self.assertTrue(reader.read(folder, sname, calc_type="MAGRES"))
            atom_arrays_hyperfine = deepcopy(reader.atoms.arrays)
            # checks if loading .magres file has added hyperfine to atom array:
            self.assertIn('hyperfine', atom_arrays_hyperfine.keys())

            # tests phonons being read:
            self.assertTrue(reader.read(folder, sname, calc_type="PHONONS"))

            # contains magres but not castep file:
            folder = os.path.join(folder, "magres_only")
            # check for magres not being read without castep
            reader = ReadWriteCastep()
            self.assertFalse(reader.read_castep_hyperfine_magres(folder))

    def test_create_calc(self):
        params = {"mu_symbol": "mu", "k_points_grid": [2, 2, 2]}
        folder = _TESTDATA_DIR  # does not contain any castep files
        reader = ReadWriteCastep()
        calc_geom_opt = reader.create_castep_calculator(params,
                                                        calc_type="GEOM_OPT")
        calc_magres = reader.create_castep_calculator(params,
                                                      calc_type="MAGRES")

        # Tests that the calculators have the correct tasks set:
        self.assertEqual(calc_geom_opt.param.task.value,
                         "GeometryOptimization")
        self.assertEqual(calc_magres.param.task.value, "Magres")
        self.assertEqual(calc_magres.param.magres_task.value, "Hyperfine")

        # Tests that k_points_grid gets set in both calculators
        self.assertEqual(calc_geom_opt.cell.kpoint_mp_grid.value,
                         list_to_string(params['k_points_grid']))
        self.assertEqual(calc_magres.cell.kpoint_mp_grid.value,
                         list_to_string(params['k_points_grid']))

    def test_write(self):
        # read in cell file to get atom

        input_folder = _TESTDATA_DIR
        output_folder = _TESTSAVE_DIR

        atoms = io.read(os.path.join(_TESTDATA_DIR, "srtio3.cell"))

        # test writing geom_opt output
        reader = ReadWriteCastep()
        reader.write(atoms, output_folder, sname="srtio3_geom_opt",
                     calc_type="GEOM_OPT")

        reader.write(atoms, output_folder, sname="srtio3_magres",
                     calc_type="MAGRES")

        # read back in and check that atom locations are preserved
        geom_opt_atoms = io.read(os.path.join(output_folder,
                                 "srtio3_geom_opt.cell"))
        magres_atoms = io.read(os.path.join(output_folder,
                               "srtio3_magres.cell"))
        equal = atoms.positions == geom_opt_atoms.positions
        self.assertTrue(equal.all())
        equal = atoms.positions == magres_atoms.positions
        self.assertTrue(equal.all())

        # Test if parameters file have correct tasks:
        geom_params = read_param(os.path.join(output_folder,
                                 "srtio3_geom_opt.param"))
        magres_params = read_param(os.path.join(output_folder,
                                   "srtio3_magres.param"))
        self.assertEqual(geom_params.param.task.value,
                         "GeometryOptimization")
        self.assertEqual(magres_params.param.task.value, "Magres")
        self.assertEqual(magres_params.param.magres_task.value, "Hyperfine")

        # TODO:
        # test setting up a parameter file and seeing if output occurs:
        # geom_force_tol, k points, etc.


if __name__ == "__main__":

    unittest.main()
