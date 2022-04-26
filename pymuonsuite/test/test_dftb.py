"""Tests for ReadWriteDFTB methods"""

import unittest

import os
import shutil

from ase import Atoms, io
from ase.calculators.dftb import Dftb

from pymuonsuite.data.dftb_pars import DFTBArgs
from pymuonsuite.io.dftb import ReadWriteDFTB
from pymuonsuite.utils import get_element_from_custom_symbol


_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTDATA_DIR = os.path.join(_TEST_DIR, "test_data")


class TestReadWriteDFTB(unittest.TestCase):
    def test_read(self):
        folder = _TESTDATA_DIR + "/ethyleneMu"
        # does not contain any dftb files
        reader = ReadWriteDFTB()
        # test that we do not get any result for trying to read
        # an empty folder:
        with self.assertRaises(OSError) as e:
            reader.read(folder)
            self.assertTrue("no such file or directory" in e)

        folder = os.path.join(
            folder,
            "dftb-nq-results/ethyleneMu_opt_displaced/" "ethyleneMu_opt_displaced_0",
        )
        # tests dftb files being read:
        self.assertTrue(reader.read(folder))

        # tests hyperfine being read:
        atoms = reader.read(folder, read_spinpol=True)
        # checks if added hyperfine to atom array:
        self.assertIn("hyperfine", atoms.arrays.keys())

        # checks if phonon info has been loaded into atom object:
        # No phonon file in this folder
        self.assertNotIn("ph_evecs", atoms.info.keys())
        self.assertNotIn("ph_evals", atoms.info.keys())

        # phonon file in this folder:
        folder = os.path.join(_TESTDATA_DIR, "ethyleneMu/dftb-phonons")
        self.assertTrue(reader.read(folder))
        atoms2 = reader.read(folder, read_phonons=True)
        self.assertIn("ph_evecs", atoms2.info.keys())
        self.assertIn("ph_evals", atoms2.info.keys())

    def test_create_calc(self):
        # Tests whether the correct values of the parameters are set
        # when creating a calculator that would be used for writing
        # a dftb input file.

        def check_geom_opt_params(params, calc_params):
            self.assertEqual(calc_params["Hamiltonian_Charge"], params["charged"] * 1.0)
            self.assertEqual(calc_params["Driver_MaxSteps"], params["geom_steps"])
            self.assertEqual(
                calc_params["Driver_MaxForceComponent [eV/AA]"],
                params["geom_force_tol"],
            )
            self.assertEqual(
                calc_params["Hamiltonian_MaxSccIterations"],
                params["max_scc_steps"],
            )

        # In the case that a params dict is provided, the values for the
        # parameters should be taken from here.
        params = {
            "k_points_grid": [2, 2, 2],
            "geom_force_tol": 0.01,
            "dftb_set": "pbc-0-3",
            "dftb_optionals": [],
            "geom_steps": 500,
            "max_scc_steps": 150,
            "charged": False,
            "dftb_pbc": True,
        }
        reader = ReadWriteDFTB(params=params)
        # First test a geom opt calculator:
        calc_geom_opt = reader._create_calculator(calc_type="GEOM_OPT")
        calc_params = calc_geom_opt.parameters
        check_geom_opt_params(params, calc_params)
        self.assertEqual(calc_geom_opt.kpts, params["k_points_grid"])

        # Next a spinpol calculator:
        calc_magres = reader._create_calculator(calc_type="SPINPOL")
        self.assertEqual(calc_magres.kpts, params["k_points_grid"])

        # In the case that a calculator is provided, the new calculator should
        # retain the same properties.
        args = {
            "Hamiltonian_Charge": params["charged"] * 1.0,
            "Driver_MaxSteps": params["geom_steps"],
            "Driver_MaxForceComponent [eV/AA]": params["geom_force_tol"],
            "Hamiltonian_MaxSccIterations": params["max_scc_steps"],
        }
        dargs = DFTBArgs(params["dftb_set"])
        dargs.set_optional("spinpol.json", True)
        args.update(dargs.args)
        calc = Dftb(kpts=params["k_points_grid"], **args)
        reader = ReadWriteDFTB(calc=calc)
        # First test a geom opt calculator:
        calc_geom_opt = reader._create_calculator(calc_type="GEOM_OPT")
        calc_params = calc_geom_opt.parameters
        check_geom_opt_params(params, calc_params)
        self.assertEqual(calc_geom_opt.kpts, params["k_points_grid"])
        # Next a spinpol calculator:
        calc_magres = reader._create_calculator(calc_type="SPINPOL")
        self.assertEqual(calc_magres.kpts, params["k_points_grid"])

        # In the case that we do not supply a params dict or a calculator,
        # the new calculator should get the default settings:
        params = {
            "k_points_grid": None,
            "geom_force_tol": 0.05,
            "dftb_set": "3ob-3-1",
            "geom_steps": 30,
            "max_scc_steps": 200,
            "charged": False,
        }
        reader = ReadWriteDFTB()
        # First test a geom opt calculator:
        calc_geom_opt = reader._create_calculator(calc_type="GEOM_OPT")
        calc_params = calc_geom_opt.parameters
        self.assertEqual(calc_geom_opt.kpts, params["k_points_grid"])
        check_geom_opt_params(params, calc_params)
        # Next a spinpol calculator:
        calc_magres = reader._create_calculator(calc_type="SPINPOL")
        self.assertEqual(calc_magres.kpts, params["k_points_grid"])

    def test_write(self):
        # Tests writing DFTB+ input files, and checks that the
        # atoms read from those input files is the same as the
        # atoms used to generate them.
        try:
            params = {
                "geom_force_tol": 0.01,
                "dftb_set": "3ob-3-1",
                "geom_steps": 10,
                "max_scc_steps": 200,
                "dftb_pbc": True,
                "kpoints_grid": [2, 2, 2],
            }

            output_folder = os.path.join(_TESTDATA_DIR, "test_save")
            os.mkdir(output_folder)

            # read in cell file to get atom:
            atoms = io.read(os.path.join(_TESTDATA_DIR, "ethyleneMu/ethyleneMu.xyz"))

            # test writing input files
            reader = ReadWriteDFTB(params=params)
            reader.write(
                atoms,
                output_folder,
                sname="ethylene_geom_opt",
                calc_type="SPINPOL",
            )
            atoms_read = reader.read(output_folder)
            self.assertEqual(atoms, atoms_read)
            reader.write(
                atoms,
                output_folder,
                sname="ethylene_geom_opt",
                calc_type="GEOM_OPT",
            )
            atoms_read = reader.read(output_folder)
            self.assertEqual(atoms, atoms_read)
        finally:
            shutil.rmtree(output_folder)

    def test_write_uses_correct_particle_mass_and_element(self):

        # Tests writing DFTB+ input files, and checks that the
        # atoms read from those input files is the same as the
        # atoms used to generate them.
        try:
            params = {
                "geom_force_tol": 0.01,
                "dftb_set": "3ob-3-1",
                "geom_steps": 10,
                "max_scc_steps": 200,
                "dftb_pbc": True,
                "kpoints_grid": [2, 2, 2],
                "particle_mass_amu": 8.02246,
                "mu_symbol": "Li:8",
            }

            output_folder = os.path.join(_TESTDATA_DIR, "test_save")
            os.mkdir(output_folder)

            # read in cell file to get atom:
            atoms = io.read(os.path.join(_TESTDATA_DIR, "ethyleneMu/ethyleneMu.xyz"))

            mu_symbol_element = get_element_from_custom_symbol(params["mu_symbol"])

            atoms += Atoms(
                mu_symbol_element,
                positions=[(0, 0, 0)],
                masses=[params["particle_mass_amu"]],
            )

            # test writing input files
            reader = ReadWriteDFTB(params=params)
            reader.write(
                atoms,
                output_folder,
                sname="ethylene_geom_opt",
                calc_type="GEOM_OPT",
            )
            atoms_read = reader.read(output_folder)
            self.assertEqual(mu_symbol_element, atoms_read.get_chemical_symbols()[-1])
            with self.assertRaises(AssertionError):
                self.assertEqual(
                    params["particle_mass_amu"], atoms_read.get_masses()[-1]
                )  # doesn't work as mass is defined in dftb_in.hsd, which isn't read in

            # in the meantime, manually check that file was written correctly
            # remove this when the above checks are fixed
            expected_line = "MassPerAtom [amu] = 8.02246"
            with open(os.path.join(output_folder, "dftb_in.hsd"), "r") as f:
                contents = f.read()
                self.assertIn(expected_line, contents)

        finally:
            shutil.rmtree(output_folder)


if __name__ == "__main__":

    unittest.main()
