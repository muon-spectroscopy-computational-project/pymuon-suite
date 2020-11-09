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
            try:
                reader.read(folder, sname)
            except Exception as e:
                print(e)

            folder = os.path.join(_TESTDATA_DIR, "castep")
            # tests castep file being read:
            self.assertTrue(reader.read(folder, sname))
            atoms = reader.read(folder, sname)
            atom_arrays_castep = deepcopy(reader.read(folder, sname).arrays)

            #  checks if phonon info has been loaded into atom object:
            self.assertIn('ph_evecs', atoms.info.keys())
            self.assertIn('ph_evals', atoms.info.keys())

            # # tests hyperfine being read:
            # checks if loading .magres file has added hyperfine to atom array:
            self.assertIn('hyperfine', atoms.arrays.keys())

            # tests phonons being read:
            self.assertTrue(reader.read(folder, sname))

    def test_create_calc(self):
        params = {"mu_symbol": "mu", "k_points_grid": [7, 7, 7]}
        folder = os.path.join(_TESTDATA_DIR, "castep")  # does not contain any castep files
        reader = ReadWriteCastep(params=params)
        calc = reader._ReadWriteCastep__create_calculator()
        self.assertTrue(calc)
        calc_geom = reader._ReadWriteCastep__update_calculator("GEOM_OPT")
        self.assertTrue(calc_geom)

        self.assertEqual(calc_geom.param.task.value,
                         "GeometryOptimization")

        calc_magres = reader._ReadWriteCastep__update_calculator(calc_type="MAGRES")

        # Tests that the calculators have the correct tasks set:

        self.assertEqual(calc_magres.param.task.value, "Magres")
        self.assertEqual(calc_magres.param.magres_task.value, "Hyperfine")

        # Tests that k_points_grid gets set in both calculators
        self.assertEqual(calc_geom.cell.kpoint_mp_grid.value,
                         list_to_string(params['k_points_grid']))
        self.assertEqual(calc_magres.cell.kpoint_mp_grid.value,
                         list_to_string(params['k_points_grid']))

    def test_write(self):
        # read in cell file to get atom

        input_folder = _TESTDATA_DIR +"/castep"
        output_folder = _TESTSAVE_DIR

        atoms = io.read(os.path.join(input_folder, "srtio3.cell"))

        # test writing geom_opt output
        reader = ReadWriteCastep()
        print(atoms.calc)
        reader.write(atoms, output_folder, sname="srtio3_geom_opt",
                     calc_type="GEOM_OPT")

        reader.write(atoms, output_folder, sname="srtio3_magres",
                     calc_type="MAGRES")

        # # read back in and check that atom locations are preserved
        geom_opt_atoms = io.read(os.path.join(output_folder,
                                 "srtio3_geom_opt.cell"))
        magres_atoms = io.read(os.path.join(output_folder,
                               "srtio3_magres.cell"))
        equal = atoms.positions == geom_opt_atoms.positions
        self.assertTrue(equal.all())
        equal = atoms.positions == magres_atoms.positions
        self.assertTrue(equal.all())

        # # Test if parameters file have correct tasks:
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

        os.remove(os.path.join(output_folder,
                                 "srtio3_geom_opt.param"))
        os.remove(os.path.join(output_folder,
                                   "srtio3_magres.param"))
        os.remove(os.path.join(output_folder,
                               "srtio3_magres.cell"))
        os.remove(os.path.join(output_folder,
                                 "srtio3_geom_opt.cell"))


if __name__ == "__main__":

    unittest.main()
