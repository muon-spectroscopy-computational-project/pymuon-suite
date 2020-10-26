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
from pymuonsuite.io.dftb import ReadWriteDFTB
from pymuonsuite.schemas import load_input_file, MuAirssSchema, AsePhononsSchema

import argparse as ap


_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTDATA_DIR = os.path.join(_TEST_DIR, "test_data")
_TESTSAVE_DIR = os.path.join(_TEST_DIR, "test_save")


class TestReadWriteDFTB(unittest.TestCase):

    def test_read(self):
        seednames = ['ethyleneMu_opt']
        for sname in seednames:
            folder = _TESTDATA_DIR  # does not contain any castep files
            reader = ReadWriteDFTB()
            # test that we do not get any result for trying to read
            # an empty folder:
            self.assertFalse(reader.read(folder, avg_prop=None))
            self.assertFalse(reader.read(folder, avg_prop="hyperfine"))

            folder = os.path.join(_TESTDATA_DIR, "dftb2")
            # tests castep file being read:
            self.assertTrue(reader.read(folder))
            atom_arrays = deepcopy(reader.read(folder).arrays)

            # tests hyperfine being read:
            self.assertNotIn('hyperfine', atom_arrays)
            self.assertTrue(reader.read(folder, avg_prop="hyperfine"))
            atom_arrays_hyperfine = deepcopy(reader.read(folder, avg_prop="hyperfine").arrays)
            # checks if added hyperfine to atom array:
            self.assertIn('hyperfine', atom_arrays_hyperfine.keys())

            folder = os.path.join(_TESTDATA_DIR, "dftb")
            # tests phonons being read:
            self.assertTrue(reader.read(folder, sname, calc_type="PHONONS"))
            # can't read phonons without a seed name
            self.assertFalse(reader.read(folder, calc_type="PHONONS"))

    def test_create_calc(self):
        params = {"mu_symbol": "mu", "k_points_grid": [2, 2, 2],
                  "geom_force_tol": 0.01, 'dftb_set': '3ob-3-1',
                  'dftb_optionals': [], 'geom_steps': 500,
                  "max_scc_steps": 20, "charged": False, "dftb_pbc": False}
        folder = _TESTDATA_DIR  # does not contain any castep files
        reader = ReadWriteDFTB()
        #self.assertFalse(reader.create_calculator(params=params))
        calc_geom_opt = reader.create_calculator(params = params, calc_type="muairss")
        calc_magres = reader.create_calculator(params = params, calc_type="spinpol")
        self.assertTrue(calc_geom_opt)
        self.assertTrue(calc_magres)

        self.assertEqual(calc_geom_opt.kpts, None)
        self.assertEqual(calc_magres.kpts, params["k_points_grid"])

    def test_write(self):
        params = {"mu_symbol": "mu", "k_points_grid": [2, 2, 2],
                  "geom_force_tol": 0.01, 'dftb_set': '3ob-3-1',
                  'dftb_optionals': [], 'geom_steps': 500,
                  "max_scc_steps": 20, "charged": False, "dftb_pbc": False,
                  'pbc': False, "force_tol": 0.01, "force_clean": False,
                  'name': 'test'}

        # read in cell file to get atom

        input_folder = os.path.join(_TESTDATA_DIR, "dftb")
        output_folder = _TESTSAVE_DIR

        atoms = io.read(os.path.join(_TESTDATA_DIR, "srtio3.cell"))

        # test writing geom_opt output
        reader = ReadWriteDFTB()
        reader.write(atoms, output_folder, sname="srtio3_geom_opt", params=params, calc_type="muairss")
        atoms2 = reader.read(output_folder, avg_prop=None)
        self.assertTrue(reader.read(output_folder, avg_prop=None))
        self.assertEqual(atoms, atoms2)

        # test phonons output:
        
        os.chdir(input_folder)
        print("FOLDER: ", input_folder)
        atoms = io.read(os.path.join(input_folder, "ethyleneMu.xyz"))
        params = load_input_file("phonons.yaml", AsePhononsSchema)

        cell_file = "ethyleneMu.xyz"
        param_file = "phonons.yaml"
        sys.argv[1:] = [cell_file, param_file]

        parser = ap.ArgumentParser(description="Compute phonon modes with ASE and"
                               " DFTB+ for reuse in quantum effects "
                               "calculations.")
        parser.add_argument('structure_file', type=str,
                            help="Structure for which to compute the phonons")
        parser.add_argument('parameter_file', type=str,
                            help="YAML file containing relevant input parameters")

        args = parser.parse_args()

        reader.write(atoms, input_folder, params=params, calc_type="PHONONS", args = args)

        # TODO:
        # More tests of write outputs being correct


if __name__ == "__main__":

    unittest.main()
