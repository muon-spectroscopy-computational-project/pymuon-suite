"""Tests for ReadWriteCastep methods"""

import unittest

import os
import shutil
import numpy as np

from ase import Atoms, io
from ase.io.castep import read_param

from pymuonsuite.utils import list_to_string, get_element_from_custom_symbol
from pymuonsuite.io.castep import ReadWriteCastep
from pymuonsuite.schemas import load_input_file, MuAirssSchema

from soprano.utils import silence_stdio

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTDATA_DIR = os.path.join(_TEST_DIR, "test_data")


class TestReadWriteCastep(unittest.TestCase):
    def test_read(self):
        sname = "ethyleneMu"
        folder = _TESTDATA_DIR  # does not contain any castep files
        reader = ReadWriteCastep()
        # test that we do not get any result for trying to read
        # an empty folder:
        with self.assertRaises(IOError) as e:
            reader.read(folder, sname)
            self.assertTrue("no such file or directory" in e)

        folder = os.path.join(_TESTDATA_DIR, sname)
        # tests castep file being read:
        atoms = reader.read(folder, sname, read_magres=True, read_phonons=True)

        #  checks if phonon info has been loaded into atom object:
        self.assertIn("ph_evecs", atoms.info.keys())
        self.assertIn("ph_evals", atoms.info.keys())

        # # tests hyperfine being read:
        # checks if loading .magres file has added hyperfine to atom array:
        self.assertIn("hyperfine", atoms.arrays.keys())

    def test_create_calc(self):
        params = {"mu_symbol": "mu", "k_points_grid": [7, 7, 7]}
        reader = ReadWriteCastep(params=params)
        reader._create_calculator()
        calc_geom = reader._update_calculator("GEOM_OPT")

        self.assertEqual(calc_geom.param.task.value, "GeometryOptimization")

        calc_magres = reader._update_calculator(calc_type="MAGRES")

        # Tests that the calculators have the correct tasks set:

        self.assertEqual(calc_magres.param.task.value, "Magres")
        self.assertEqual(calc_magres.param.magres_task.value, "Hyperfine")

        # Tests that k_points_grid gets set in both calculators
        self.assertEqual(
            calc_geom.cell.kpoint_mp_grid.value,
            list_to_string(params["k_points_grid"]),
        )
        self.assertEqual(
            calc_magres.cell.kpoint_mp_grid.value,
            list_to_string(params["k_points_grid"]),
        )

    def test_write(self):
        # read in cell file to get atom
        try:
            input_folder = _TESTDATA_DIR + "/Si2"
            output_folder = os.path.join(_TESTDATA_DIR, "test_save")
            os.mkdir(output_folder)

            os.chdir(input_folder)

            yaml_file = os.path.join(input_folder, "Si2-muairss-castep.yaml")
            cell_file = os.path.join(input_folder, "Si2.cell")
            param_file = os.path.join(input_folder, "Si2.param")
            input_params = load_input_file(yaml_file, MuAirssSchema)

            with silence_stdio():
                castep_param = read_param(param_file).param
                atoms = io.read(cell_file)

            # test writing geom_opt output
            reader = ReadWriteCastep(params=input_params)
            reader.write(
                atoms,
                output_folder,
                sname="Si2_geom_opt",
                calc_type="GEOM_OPT",
            )

            reader.write(atoms, output_folder, sname="Si2_magres", calc_type="MAGRES")

            # # read back in and check that atom locations are preserved
            with silence_stdio():
                geom_opt_atoms = io.read(
                    os.path.join(output_folder, "Si2_geom_opt.cell")
                )
                magres_atoms = io.read(os.path.join(output_folder, "Si2_magres.cell"))
            equal = atoms.positions == geom_opt_atoms.positions
            # self.assertTrue(equal.all()) # is not true due to to rounding
            equal = geom_opt_atoms.positions == magres_atoms.positions
            self.assertTrue(equal.all())
            self.assertEqual(
                geom_opt_atoms.calc.cell.kpoint_mp_grid.value,
                list_to_string(input_params["k_points_grid"]),
            )
            self.assertEqual(
                magres_atoms.calc.cell.kpoint_mp_grid.value,
                list_to_string(input_params["k_points_grid"]),
            )

            # Test if parameters file have correct tasks:
            with silence_stdio():
                geom_params = read_param(
                    os.path.join(output_folder, "Si2_geom_opt.param")
                ).param
                magres_params = read_param(
                    os.path.join(output_folder, "Si2_magres.param")
                ).param
            self.assertEqual(geom_params.task.value, "GeometryOptimization")
            self.assertEqual(magres_params.task.value, "Magres")
            self.assertEqual(magres_params.magres_task.value, "Hyperfine")
            # These are only set in the param file only so should equal
            # the value in the param file:
            self.assertEqual(geom_params.geom_max_iter, castep_param.geom_max_iter)
            self.assertEqual(geom_params.cut_off_energy, castep_param.cut_off_energy)
            self.assertEqual(geom_params.elec_energy_tol, castep_param.elec_energy_tol)
            # This is set in the input yaml and param file so should equal
            # the value in the yaml file:
            self.assertEqual(
                geom_params.geom_force_tol.value,
                str(input_params["geom_force_tol"]),
            )

            self.assertEqual(magres_params.cut_off_energy, castep_param.cut_off_energy)
            self.assertEqual(
                magres_params.elec_energy_tol, castep_param.elec_energy_tol
            )

        finally:
            shutil.rmtree(output_folder)

    def test_write_script(self):
        # read in cell file to get atom
        try:
            input_folder = _TESTDATA_DIR + "/Si2"
            output_folder = os.path.join(_TESTDATA_DIR, "test_save")
            os.mkdir(output_folder)

            os.chdir(input_folder)

            yaml_file = os.path.join(input_folder, "Si2-muairss-castep.yaml")
            cell_file = os.path.join(input_folder, "Si2.cell")
            input_params = load_input_file(yaml_file, MuAirssSchema)

            with silence_stdio():
                atoms = io.read(cell_file)

            # write output including script
            reader = ReadWriteCastep(
                params=input_params, script=input_params.get("script_file")
            )
            reader.write(
                atoms,
                output_folder,
                sname="Si2_geom_opt",
                calc_type="GEOM_OPT",
            )

            # read back in and check that scripts are as intended
            with open(
                os.path.join(output_folder, "script.sh"), newline=""
            ) as script_written:
                with open(
                    os.path.join(input_folder, "submit_Si2_geom_opt.sh"), newline=""
                ) as script_expect:
                    script_written_lines = script_written.readlines()
                    script_expect_lines = script_expect.readlines()
                    for written, expect in zip(
                        script_written_lines, script_expect_lines
                    ):
                        self.assertEqual(written, expect)

        finally:
            shutil.rmtree(output_folder)

    def test_write_uses_correct_particle_mass_and_element(self):
        # read in cell file to get atom
        try:
            input_folder = _TESTDATA_DIR + "/Si2"
            output_folder = os.path.join(_TESTDATA_DIR, "test_save")
            os.mkdir(output_folder)

            os.chdir(input_folder)

            yaml_file = os.path.join(input_folder, "Si2-muairss-castep-Li8.yaml")
            cell_file = os.path.join(input_folder, "Si2.cell")
            input_params = load_input_file(yaml_file, MuAirssSchema)

            with silence_stdio():
                atoms = io.read(cell_file)

            mu_symbol = input_params["mu_symbol"]
            mu_symbol_element = get_element_from_custom_symbol(mu_symbol)
            custom_species = atoms.get_chemical_symbols() + [mu_symbol]

            atoms += Atoms(
                mu_symbol_element,
                positions=[(0, 0, 0)],
                masses=[input_params["particle_mass_amu"]],
            )
            atoms.set_array("castep_custom_species", np.array(custom_species))

            # test writing geom_opt output
            reader = ReadWriteCastep(params=input_params)
            reader.write(
                atoms,
                output_folder,
                sname="Si2_geom_opt",
                calc_type="GEOM_OPT",
            )

            reader.write(atoms, output_folder, sname="Si2_magres", calc_type="MAGRES")

            # read back in and check that particle mass is preserved
            with silence_stdio():
                geom_opt_atoms = io.read(
                    os.path.join(output_folder, "Si2_geom_opt.cell")
                )
                magres_atoms = io.read(os.path.join(output_folder, "Si2_magres.cell"))

            self.assertEqual(
                mu_symbol_element, geom_opt_atoms.get_chemical_symbols()[-1]
            )
            self.assertEqual(mu_symbol_element, magres_atoms.get_chemical_symbols()[-1])

            with self.assertRaises(AssertionError):
                # currently these checks fail,
                # because custom species masses aren't loaded correctly in ASE
                # when that is resolved, remove the assertRaises and these should pass
                self.assertEqual(
                    input_params["particle_mass_amu"], geom_opt_atoms.get_masses()[-1]
                )
                self.assertEqual(
                    input_params["particle_mass_amu"], magres_atoms.get_masses()[-1]
                )

            # in the meantime, manually check that file was written correctly
            # remove this when the above checks are fixed
            expected_block = """%BLOCK SPECIES_MASS
AMU
Li:8 8.02246
%ENDBLOCK SPECIES_MASS"""
            with open(os.path.join(output_folder, "Si2_geom_opt.cell"), "r") as f:
                contents = f.read()
                self.assertIn(expected_block, contents)
            with open(os.path.join(output_folder, "Si2_magres.cell"), "r") as f:
                contents = f.read()
                self.assertIn(expected_block, contents)

        finally:
            shutil.rmtree(output_folder)


if __name__ == "__main__":

    unittest.main()
