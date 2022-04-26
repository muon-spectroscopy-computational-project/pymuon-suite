"""Tests for ReadWriteUEP methods"""

import os
import shutil
import unittest

import numpy as np
from ase import Atoms, io
from scipy.constants import physical_constants as pcnst
from pymuonsuite.io.uep import ReadWriteUEP

from pymuonsuite.schemas import load_input_file, MuAirssSchema, UEPOptSchema
from pymuonsuite.utils import get_element_from_custom_symbol
from soprano.utils import silence_stdio

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTDATA_DIR = os.path.join(_TEST_DIR, "test_data")


class TestReadWriteUEP(unittest.TestCase):
    def test_read_fails_when_folder_empty(self):
        sname = "Si2_1"
        folder = _TESTDATA_DIR  # does not contain any uep files
        reader = ReadWriteUEP()
        # test that we do not get any result for trying to read
        # an empty folder:
        with self.assertRaises(OSError) as context:
            reader.read(folder, sname)

        self.assertIn("could not read UEP file", str(context.exception))

    def test_read(self):
        sname = "Si2_1"
        reader = ReadWriteUEP()

        folder = os.path.join(_TESTDATA_DIR, "Si2/uep-result")
        # tests uep file being read, and compares structure to
        # that in the xyz file - these should be equal
        read_uep = reader.read(folder, sname)
        read_xyz = io.read(os.path.join(folder, sname + ".xyz"))

        self.assertTrue(np.all(read_uep.numbers == read_xyz.numbers))
        self.assertTrue(np.allclose(read_uep.positions, read_xyz.positions, atol=1e-3))
        self.assertTrue(np.all(read_uep.pbc == read_xyz.pbc))
        self.assertTrue(np.allclose(read_uep.cell, read_xyz.cell))

        # These are results contained in the uep pickle file
        Eclass = -8.843094140155303
        Ezp = 0.11128549781255458
        Etot = -8.731808642342749
        # Check these have been read correctly:
        self.assertEqual(read_uep.calc._Eclass, Eclass)
        self.assertEqual(read_uep.calc._Ezp, Ezp)
        self.assertEqual(read_uep.calc._Etot, Etot)

    def test_create_calc(self):
        folder = os.path.join(_TESTDATA_DIR, "Si2")

        def check_geom_opt_params(calc, params):
            self.assertEqual(calc.label, params["name"])
            self.assertEqual(calc.geom_steps, params["geom_steps"])
            self.assertEqual(calc.gw_factor, params["uep_gw_factor"])
            self.assertEqual(calc.opt_tol, params["geom_force_tol"])
            self.assertEqual(calc.save_structs, params["uep_save_structs"])

        # In the case that a params dict is provided, the values for the
        # parameters should be taken from here:
        params = {
            "name": "Si2",
            "charged": True,
            "geom_steps": 300,
            "uep_gw_factor": 4.0,
            "geom_force_tol": 0.05,
            "uep_save_structs": False,
            "uep_chden": "Si2.den_fmt",
        }

        reader = ReadWriteUEP(params=params)
        with silence_stdio():
            a = io.read(os.path.join(folder, "Si2.cell"))

        calc = reader._create_calculator(a, folder, "Si2")
        check_geom_opt_params(calc, params)

        # In the case that we do not supply a params dict or a calculator,
        # the new calculator should get the default settings:
        reader = ReadWriteUEP()

        params = {
            "name": "Si2",
            "geom_steps": 30,
            "uep_gw_factor": 5.0,
            "geom_force_tol": 1e-5,
            "uep_save_structs": True,
        }

        calc = reader._create_calculator(a, folder, "Si2")
        check_geom_opt_params(calc, params)

    def test_write_succeeds_when_charged_true(self):
        # read in cell file to get atom
        try:
            input_folder = _TESTDATA_DIR + "/Si2"
            os.chdir(input_folder)

            output_folder = "test_save"
            os.mkdir(output_folder)

            with silence_stdio():
                atoms = io.read("Si2.cell")

            # test writing geom_opt output
            param_file = "Si2-muairss-uep.yaml"
            params = load_input_file(param_file, MuAirssSchema)

            reader = ReadWriteUEP(params=params)

            reader.write(atoms, output_folder)

            self.assertTrue(
                os.path.exists(os.path.join(output_folder, "test_save.yaml"))
            )
        finally:
            shutil.rmtree("test_save")

    def test_write_fails_when_charged_false(self):
        # read in cell file to get atom
        try:
            input_folder = _TESTDATA_DIR + "/Si2"
            os.chdir(input_folder)

            output_folder = "test_save"
            os.mkdir(output_folder)

            with silence_stdio():
                atoms = io.read("Si2.cell")

            # test writing geom_opt output
            param_file = "Si2-muairss-uep.yaml"
            params = load_input_file(param_file, MuAirssSchema)

            reader = ReadWriteUEP(params=params)

            reader.write(atoms, output_folder)

            self.assertTrue(
                os.path.exists(os.path.join(output_folder, "test_save.yaml"))
            )

            params["charged"] = False

            reader = ReadWriteUEP(params=params)

            with self.assertRaises(RuntimeError) as context:
                reader.write(atoms, output_folder)

            self.assertIn(
                "Can't use UEP method for neutral system", str(context.exception)
            )

        finally:
            shutil.rmtree("test_save")

    def test_write_uses_correct_particle_mass(self):

        # read in cell file to get atom
        try:
            input_folder = _TESTDATA_DIR + "/Si2"
            os.chdir(input_folder)

            output_folder = "test_save"
            os.mkdir(output_folder)

            with silence_stdio():
                atoms = io.read("Si2.cell")

            # test writing geom_opt output
            param_file = "Si2-muairss-uep-Li8.yaml"
            params = load_input_file(param_file, MuAirssSchema)

            mu_symbol_element = get_element_from_custom_symbol(params["mu_symbol"])

            atoms += Atoms(
                mu_symbol_element,
                positions=[(0, 0, 0)],
                masses=[params["particle_mass_amu"]],
            )
            reader = ReadWriteUEP(params=params)

            reader.write(atoms, output_folder)

            expected_file = os.path.join(output_folder, "test_save.yaml")
            self.assertTrue(os.path.exists(expected_file))
            uep_params = load_input_file(expected_file, UEPOptSchema)

            self.assertEqual(
                uep_params["particle_mass"],
                params["particle_mass_amu"] * pcnst["atomic mass constant"][0],
            )

        finally:
            shutil.rmtree("test_save")


if __name__ == "__main__":

    unittest.main()
