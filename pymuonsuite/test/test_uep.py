"""Tests for ReadWriteUEP methods"""

import glob
import os
import shutil
import sys
import unittest

from ase import Atoms, io

import numpy as np

from pymuonsuite.calculate.uep.__main__ import plot_entry
from pymuonsuite.calculate.uep.charged import ChargeDistribution
from pymuonsuite.io.uep import ReadWriteUEP
from pymuonsuite.schemas import MuAirssSchema, UEPOptSchema, load_input_file
from pymuonsuite.utils import get_element_from_custom_symbol

from scipy.constants import physical_constants as pcnst

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


class TestPlotUEP(unittest.TestCase):
    def tearDown(self):
        for f in glob.glob("Si8.*.*.dat"):
            os.remove(f)

    def test_plot(self):
        input_folder = _TESTDATA_DIR + "/Si8"
        os.chdir(input_folder)
        sys.argv[1:] = ["Si8.yaml"]
        plot_entry()
        for i in (1, 2, 3):
            self.assertTrue(os.path.exists(f"Si8.line.{i}.dat"))
        for i in (1, 2):
            self.assertTrue(os.path.exists(f"Si8.plane.{i}.dat"))

    def test_plot_invalid_line(self):
        input_folder = _TESTDATA_DIR + "/Si8"
        os.chdir(input_folder)
        sys.argv[1:] = ["Si8-invalid-line.yaml"]
        with self.assertRaises(SystemExit):
            plot_entry()

    def test_plot_invalid_plane(self):
        input_folder = _TESTDATA_DIR + "/Si8"
        os.chdir(input_folder)
        sys.argv[1:] = ["Si8-invalid-plane.yaml"]
        with self.assertRaises(SystemExit):
            plot_entry()


class TestChargeDistribution(unittest.TestCase):
    def tearDown(self):
        for f in glob.glob("test.*"):
            os.remove(f)

    def test_validate_points(self):
        input_folder = _TESTDATA_DIR + "/Si2"
        os.chdir(input_folder)
        charge_distribution = ChargeDistribution("Si2")
        self.assertTrue(
            np.all(charge_distribution._validate_points([0, 0, 0]) == [[0, 0, 0]])
        )
        self.assertTrue(
            np.all(charge_distribution._validate_points([[0, 0, 0]]) == [[0, 0, 0]])
        )
        self.assertTrue(
            np.all(
                charge_distribution._validate_points([[0, 0, 0], [1, 1, 1]])
                == [[0, 0, 0], [1, 1, 1]]
            )
        )

    def test_basic(self):
        input_folder = _TESTDATA_DIR + "/Si2"
        os.chdir(input_folder)
        charge_distribution = ChargeDistribution("Si2")

        self.assertTrue(
            np.allclose(
                charge_distribution.cell,
                [
                    [2.6954645, 2.6954645, 0.0],
                    [2.6954645, 0.0, 2.6954645],
                    [0.0, 2.6954645, 2.6954645],
                ],
            )
        )
        self.assertAlmostEqual(charge_distribution.volume, 39.1679503)
        self.assertTrue(charge_distribution.chemical_symbols == ["Si", "Si"])
        self.assertTrue(
            np.allclose(
                charge_distribution.positions,
                [[0.0, 0.0, 0.0], [1.34773225, 1.34773225, 1.34773225]],
            )
        )
        self.assertTrue(
            np.allclose(
                charge_distribution.scaled_positions, [[0, 0, 0], [0.25, 0.25, 0.25]]
            )
        )
        self.assertFalse(charge_distribution.has_spin)
        self.assertAlmostEqual(charge_distribution.thomasFermiE, 0.3015282)

    def test_spin_si8(self):
        """
        This is intended as a stop gap test to replace test_spin_yb_cu_as2, which
        does not work with the latest versions of ase. It is not yet clear whether this
        is an ase bug introduced in version 3.23.0, or if the .castep file is broken.
        """
        input_folder = _TESTDATA_DIR + "/Si8"
        os.chdir(input_folder)
        charge_distribution = ChargeDistribution("Si8")

        self.assertTrue(
            np.allclose(
                charge_distribution.cell,
                [[5.4754511, 0.0, 0.0], [0.0, 5.4754511, 0.0], [0.0, 0.0, 5.4754511]],
            )
        )
        self.assertAlmostEqual(charge_distribution.volume, 164.1571162307474)
        self.assertEqual(charge_distribution.chemical_symbols, ["Si"] * 8)
        self.assertTrue(
            np.allclose(
                charge_distribution.positions,
                [
                    [0.0, 0.0, 0.0],
                    [4.10658832, 4.10658832, 1.36886277],
                    [2.73772555, 0.0, 2.73772555],
                    [4.10658832, 1.36886277, 4.10658832],
                    [0.0, 2.73772555, 2.73772555],
                    [1.36886277, 1.36886277, 1.36886277],
                    [1.36886277, 4.10658832, 4.10658832],
                    [2.73772555, 2.73772555, 0.0],
                ],
            )
        )
        self.assertTrue(
            np.allclose(
                charge_distribution.scaled_positions,
                [
                    [0.0, 0.0, 0.0],
                    [0.75, 0.75, 0.25],
                    [0.5, 0.0, 0.5],
                    [0.75, 0.25, 0.75],
                    [0.0, 0.5, 0.5],
                    [0.25, 0.25, 0.25],
                    [0.25, 0.75, 0.75],
                    [0.5, 0.5, 0.0],
                ],
            )
        )
        self.assertFalse(charge_distribution.has_spin)
        self.assertAlmostEqual(charge_distribution.thomasFermiE, 0.2334152597010807)

    @unittest.skip("ase>=3.23.0 castep_reader._add_atoms can't read YbCuAs2.castep")
    def test_spin_yb_cu_as2(self):
        input_folder = _TESTDATA_DIR + "/YbCuAs2"
        os.chdir(input_folder)
        charge_distribution = ChargeDistribution("YbCuAs2")

        self.assertTrue(
            np.allclose(
                charge_distribution.cell,
                [[3.842, 0.0, 0.0], [0.0, 3.842, 0.0], [0.0, 0.0, 9.743]],
            )
        )
        self.assertAlmostEqual(charge_distribution.volume, 143.8160723)
        self.assertTrue(
            charge_distribution.chemical_symbols
            == ["Cu", "Cu", "As", "As", "As", "As", "Yb", "Yb"]
        )
        self.assertTrue(
            np.allclose(
                charge_distribution.positions,
                [
                    [2.8815, 0.9605, 4.8715],
                    [0.9605, 2.8815, 4.8715],
                    [2.8815, 0.9605, 0.0],
                    [0.9605, 2.8815, 0.0],
                    [2.8815, 2.8815, 3.32962154],
                    [0.9605, 0.9605, 6.41337847],
                    [0.9605, 0.9605, 2.30306983],
                    [2.8815, 2.8815, 7.43993017],
                ],
            )
        )
        self.assertTrue(
            np.allclose(
                charge_distribution.scaled_positions,
                [
                    [0.75, 0.25, 0.5],
                    [0.25, 0.75, 0.5],
                    [0.75, 0.25, 0.0],
                    [0.25, 0.75, 0.0],
                    [0.75, 0.75, 0.341745],
                    [0.25, 0.25, 0.658255],
                    [0.25, 0.25, 0.236382],
                    [0.75, 0.75, 0.763618],
                ],
            )
        )
        self.assertTrue(charge_distribution.has_spin)
        self.assertAlmostEqual(charge_distribution.thomasFermiE, 6.2406736)

    def test_charged(self):
        input_folder = _TESTDATA_DIR + "/Si2"
        os.chdir(input_folder)
        with self.assertRaises(RuntimeError) as e:
            ChargeDistribution("Si2-charged")
        self.assertIn("Cell is not neutral", str(e.exception))


if __name__ == "__main__":

    unittest.main()
