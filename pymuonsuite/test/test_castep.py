"""Tests for ReadWriteCastep methods"""

import tempfile
import unittest

import os
import shutil
import numpy as np

from ase import Atoms, io
from ase.calculators.castep import Castep
from ase.io.castep import read_param

from pymuonsuite.utils import list_to_string, get_element_from_custom_symbol
from pymuonsuite.io.castep import (
    CastepError,
    ReadWriteCastep,
    add_to_castep_block,
    parse_castep_bands,
    parse_castep_gamma_block,
    parse_castep_mass_block,
    parse_castep_masses,
    parse_hyperfine_magres,
    parse_hyperfine_oldblock,
)
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
        with self.assertRaises(OSError) as e:
            reader.read(folder, sname)
        self.assertIn("No such file or directory", str(e.exception))

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
            input_params["max_scc_steps"] = 31

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
            self.assertEqual(reader._calc.param.max_scf_cycles.value, "31")

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

    def test_calc_overwrite(self):
        # Ensure that if a calculator is provided, it is initialised properly
        calculator = Castep()
        read_write_castep = ReadWriteCastep(calc=calculator)

        # Original calculator should have been deep copied, so remain as it was
        self.assertNotEqual(
            "radsectesla\nH:mu 851615456.5978916\n",
            calculator.cell.species_gamma.value,
        )
        self.assertNotEqual(
            "AMU\nH:mu 0.1134289259\n",
            calculator.cell.species_mass.value,
        )

        # Copy stored should have appropriate muon values set
        self.assertEqual(
            "radsectesla\nH:mu 851615456.5978916\n",
            read_write_castep._calc.cell.species_gamma.value,
        )
        self.assertEqual(
            "AMU\nH:mu 0.1134289259\n",
            read_write_castep._calc.cell.species_mass.value,
        )

    def test_set_params(self):
        # Ensure set_params changes the calculator object
        read_write_castep = ReadWriteCastep()
        read_write_castep.set_params({"mu_symbol": "H:nu"})

        self.assertEqual(
            "radsectesla\nH:nu 851615456.5978916\n",
            read_write_castep._calc.cell.species_gamma.value,
        )
        self.assertEqual(
            "AMU\nH:nu 0.1134289259\n",
            read_write_castep._calc.cell.species_mass.value,
        )

    def test_read_error(self):
        # Ensure user receives an error if .castep output file is missing
        read_write_castep = ReadWriteCastep()
        si_folder = _TESTDATA_DIR + "/Si2"
        ethylene_mu_folder = _TESTDATA_DIR + "/ethyleneMu"

        with self.assertRaises(IOError) as e:
            read_write_castep.read(".")
        self.assertIn("ERROR: No .castep files found in", str(e.exception))

        with self.assertRaises(IOError) as e:
            read_write_castep.read(folder=si_folder, read_magres=True)
        self.assertIn("No .magres files found in", str(e.exception))

        with self.assertRaises(IOError) as e:
            read_write_castep.read(folder=si_folder, read_phonons=True)
        self.assertIn("No .phonon files found in", str(e.exception))

        with self.assertRaises(IOError) as e:
            read_write_castep.read(
                folder=ethylene_mu_folder,
                sname="ethyleneMu_no_gamma",
                read_phonons=True,
            )
        self.assertIn(
            "Could not find gamma point phonons in CASTEP phonon file",
            str(e.exception),
        )

    def test_write_cell_calc_type(self):
        # Ensure calc_type is propagated by write_cell
        read_write_castep = ReadWriteCastep()

        with tempfile.TemporaryDirectory() as tmp_dir:
            read_write_castep.write_cell(
                a=Atoms(["H"]), folder=tmp_dir, sname="H", calc_type="MAGRES"
            )
        self.assertEqual("Magres", read_write_castep._calc.param.task.value)

    def test_write_cell_calc_type_error(self):
        # Ensure user receives an error if using a bad calc_type
        read_write_castep = ReadWriteCastep()

        with self.assertRaises(NotImplementedError) as e:
            read_write_castep.write_cell(
                a=None, folder=None, sname=None, calc_type=None
            )
        self.assertIn(
            "Calculation type None is not implemented. "
            "Please choose 'GEOM_OPT' or 'MAGRES'",
            str(e.exception),
        )

    def test_parse_castep_bands(self):
        # Test bands are parsed correctly
        bands = parse_castep_bands(os.path.join(_TESTDATA_DIR, "io", "spin_1.bands"))
        k_1 = [
            -1.30719545,
            -1.30518607,
            -1.30518607,
            -0.42212589,
            -0.00236496,
            0.12800951,
            0.12800951,
            0.43121450,
            0.59058383,
            0.74760086,
            0.74760086,
        ]
        k_2 = [
            -1.30681854,
            -1.30542366,
            -1.30535437,
            -0.42441211,
            0.01913897,
            0.10899750,
            0.12109690,
            0.46278464,
            0.57008484,
            0.67600432,
            0.77227510,
        ]

        self.assertTrue(np.allclose(np.array([k_1, k_2]), bands))

    def test_parse_castep_bands_header(self):
        # Test bands header is parsed correctly
        infile = os.path.join(_TESTDATA_DIR, "io", "spin_1.bands")
        n_kpts, n_evals = parse_castep_bands(infile=infile, header=True)

        self.assertEqual(2, n_kpts)
        self.assertEqual(11, n_evals)

    def test_parse_castep_bands_error(self):
        # Test error raised when spin components != 1
        infile = os.path.join(_TESTDATA_DIR, "io", "spin_2.bands")
        expected = (
            "Either incorrect file format detected or greater than 1 spin component "
            "used (parse_castep_bands only works with 1 spin component)."
        )

        with self.assertRaises(ValueError) as e:
            parse_castep_bands(infile=infile)
        self.assertEqual(expected, str(e.exception))

    def test_parse_castep_masses(self):
        # Test when present, the BLOCK SPECIES_MASS is read correctly
        filename = os.path.join(
            _TESTDATA_DIR, "Si2", "castep-results", "castep", "Si2_1", "Si2_1.cell"
        )
        cell = io.read(filename)
        returned_masses = parse_castep_masses(cell=cell)
        cell_masses = cell.get_masses()

        for masses in [returned_masses, cell_masses]:
            self.assertEqual(3, len(masses))
            self.assertTrue(np.allclose([28.085, 28.085, 0.1134289259], masses))

    def test_parse_castep_masses_no_block(self):
        # Test when not present, we get default masses back
        filename = os.path.join(_TESTDATA_DIR, "Si2", "Si2.cell")
        cell = io.read(filename)
        returned_masses = parse_castep_masses(cell=cell)
        cell_masses = cell.get_masses()

        for masses in [returned_masses, cell_masses]:
            self.assertEqual(2, len(masses))
            self.assertTrue(np.allclose([28.085, 28.085], masses))

    def test_parse_castep_mass_block_invalid_mass_unit(self):
        # Test we raise an error for bad units
        mass_block = "bad_unit_name"

        with self.assertRaises(CastepError) as e:
            parse_castep_mass_block(mass_block=mass_block)
        self.assertEqual("Invalid mass unit in species_mass block", str(e.exception))

    def test_parse_castep_mass_block_invalid_line(self):
        # Test we raise an error for bad line
        mass_block = "amu\nbad_line"

        with self.assertRaises(CastepError) as e:
            parse_castep_mass_block(mass_block=mass_block)
        self.assertEqual("Invalid line in species_mass block", str(e.exception))

    def test_parse_castep_gamma_block(self):
        # Test we raise an error for bad line
        gamma_block = "radsectesla\nH:mu 851615456.5978916"
        custom_gammas = parse_castep_gamma_block(gamma_block=gamma_block)

        self.assertEqual(1, len(custom_gammas))
        self.assertAlmostEqual(851615456.5978916, custom_gammas["H:mu"])

    def test_parse_castep_gamma_block_invalid_gamma_unit(self):
        # Test we raise an error for bad units
        gamma_block = "bad_unit_name"

        with self.assertRaises(CastepError) as e:
            parse_castep_gamma_block(gamma_block=gamma_block)
        self.assertEqual("Invalid gamma unit in species_gamma block", str(e.exception))

    def test_parse_castep_gamma_block_invalid_line(self):
        # Test we raise an error for bad line
        gamma_block = "radsectesla\nbad_line"

        with self.assertRaises(CastepError) as e:
            parse_castep_gamma_block(gamma_block=gamma_block)
        self.assertEqual("Invalid line in species_gamma block", str(e.exception))

    def test_add_to_castep_block(self):
        # Test can add to existing blocks
        cblock = add_to_castep_block("AMU", "H:mu", 0.11)

        self.assertEqual("AMU\nH:mu 0.11\n", cblock)

    def test_parse_hyperfine_magres_no_old(self):
        # Test error is raised when magresblock_magres_old is missing
        infile = os.path.join(_TESTDATA_DIR, "io", "no_old.magres")
        with self.assertRaises(RuntimeError) as e:
            parse_hyperfine_magres(infile)

        self.assertEqual(".magres file has no hyperfine information", str(e.exception))

    def test_parse_hyperfine_oldblock_invalid(self):
        # Test error is raised when magres block does not have Atom definitions
        with self.assertRaises(RuntimeError) as e:
            parse_hyperfine_oldblock(
                "TOTAL tensor\n"
                "\n"
                "-31.9305 -0.3111 -0.0650\n"
                "-0.3111 -36.3412 -5.7021\n"
                "-0.0650 -5.7021 -42.9359\n"
            )

        self.assertEqual("Invalid block in magres hyperfine file", str(e.exception))


if __name__ == "__main__":

    unittest.main()
