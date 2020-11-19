"""Tests for ReadWriteDFTB methods"""

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
from pymuonsuite.schemas import (load_input_file, MuAirssSchema,
                                 AsePhononsSchema)

import argparse as ap


_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTDATA_DIR = os.path.join(_TEST_DIR, "test_data")
_RUN_DFTB = False
#  Setting _RUN_DFTB to True requires dftb+ to be installed locally
#  and will test generating dftb phonons files


class TestReadWriteDFTB(unittest.TestCase):

    def test_read(self):
        folder = _TESTDATA_DIR + "/ethyleneMu"
        # does not contain any dftb files
        reader = ReadWriteDFTB()
        # test that we do not get any result for trying to read
        # an empty folder:
        try:
            reader.read(folder)
        except Exception as e:
            print(e)

        folder = os.path.join(folder,
                              "dftb-nq-results/ethyleneMu_opt_displaced/ethyleneMu_opt_displaced_0")
        # tests dftb files being read:
        self.assertTrue(reader.read(folder))
        atom_arrays = deepcopy(reader.read(folder).arrays)

        # tests hyperfine being read:
        atoms = reader.read(folder)
        # checks if added hyperfine to atom array:
        self.assertIn('hyperfine', atoms.arrays.keys())

        # checks if phonon info has been loaded into atom object:
        # No phonon file in this folder
        self.assertNotIn('ph_evecs', atoms.info.keys())
        self.assertNotIn('ph_evals', atoms.info.keys())

        # phonon file in this folder:
        folder = os.path.join(_TESTDATA_DIR, "ethyleneMu/dftb-phonons")
        self.assertTrue(reader.read(folder))
        atoms2 = reader.read(folder)
        self.assertIn('ph_evecs', atoms2.info.keys())
        self.assertIn('ph_evals', atoms2.info.keys())

    def test_create_calc(self):
        params = {"mu_symbol": "mu", "k_points_grid": [2, 2, 2],
                  "geom_force_tol": 0.01, 'dftb_set': '3ob-3-1',
                  'dftb_optionals': [], 'geom_steps': 500,
                  "max_scc_steps": 20, "charged": False, "dftb_pbc": False}
        folder = _TESTDATA_DIR  # does not contain any castep files
        reader = ReadWriteDFTB(params=params)

        calc_geom_opt = reader._create_calculator(calc_type="GEOM_OPT")
        calc_magres = reader._create_calculator(calc_type="SPINPOL")
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
        output_folder = os.path.join(_TESTDATA_DIR, "test_save")
        os.mkdir(output_folder)

        atoms = io.read(os.path.join(_TESTDATA_DIR, "Si2/Si2.cell"))

        # test writing geom_opt output
        reader = ReadWriteDFTB(params=params)
        reader.write(atoms, output_folder, sname="Si2_geom_opt",
                     calc_type="GEOM_OPT")
        atoms_read = reader.read(output_folder)
        self.assertTrue(reader.read(output_folder))
        self.assertEqual(atoms, atoms_read)

        # only test phonons output if we are able to run dftb:
        if _RUN_DFTB:
            input_folder = os.path.join(_TESTDATA_DIR,
                                        "ethyleneMu/dftb-phonons")
            os.chdir(input_folder)
            atoms = io.read(os.path.join(input_folder, "ethyleneMu.xyz"))
            params = load_input_file("phonons.yaml", AsePhononsSchema)

            cell_file = "ethyleneMu.xyz"
            param_file = "phonons.yaml"
            sys.argv[1:] = [cell_file, param_file]

            parser = ap.ArgumentParser()
            parser.add_argument('structure_file', type=str,
                                help="Structure for which to compute the"
                                "phonons")
            parser.add_argument('parameter_file', type=str,
                                help="YAML file containing relevant input"
                                "parameters")
            args = parser.parse_args()

            reader.write(atoms, output_folder, calc_type="PHONONS", args=args)
            self.assertTrue(os.path.exists('test_opt.phonons.pkl'))

            for filename in os.listdir(input_folder):
                if 'test' in filename:
                    os.remove(os.path.join(input_folder, filename))

        shutil.rmtree(output_folder)


if __name__ == "__main__":

    unittest.main()
