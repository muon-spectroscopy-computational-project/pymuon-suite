"""Tests for muairss with uep, dftb+ and castep"""

import glob
import os
import platform
import shutil
import subprocess
import sys
import unittest

from ase import Atoms, io
from ase.calculators.castep import CastepParam
from ase.io.castep import read_param

from pymuonsuite.muairss import main as run_muairss
from pymuonsuite.schemas import MuAirssSchema, UEPOptSchema, load_input_file
from pymuonsuite.utils import get_element_from_custom_symbol, list_to_string

from scipy.constants import physical_constants as pcnst

from soprano.utils import silence_stdio

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTDATA_DIR = os.path.join(_TEST_DIR, "test_data/Si2")

#  Setting _RUN_DFTB to True requires dftb+ to be installed locally
#  and will test running dftb using the files output by muairss
#  otherwise, the test will use previously generated dftb results
_RUN_DFTB = False


def _clean_testdata_dir():

    os.chdir(_TESTDATA_DIR)

    folders = [
        "Si2_clusters",
        "muon-airss-out-all",
        "muon-airss-out-uep",
        "muon-airss-out-castep",
        "muon-airss-out-dftb",
    ]

    files = [
        "Si2_clusters.txt",
        "Si2_Si2_uep_clusters.dat",
        "Si2_Si2_castep_clusters.dat",
        "Si2_Si2_dftb+_clusters.dat",
        "all.cell",
    ]

    for f in files:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass

    for f in folders:
        try:
            shutil.rmtree(f)
        except FileNotFoundError:
            pass


class TestMuairss(unittest.TestCase):
    def setUp(self):
        _clean_testdata_dir()

    def check_write_castep(
        self, out_dir: str, input_params: dict, castep_param: CastepParam
    ):
        for (rootDir, subDirs, files) in os.walk(out_dir):
            for s in subDirs:
                expected_file = os.path.join(out_dir, s, s + ".cell")
                script_file = input_params["script_file"]
                if script_file is not None:
                    expected_script = os.path.join(out_dir, s, "script.sh")
                    self.assertTrue(os.path.exists(expected_script))

                self.assertTrue(os.path.exists(expected_file))
                with silence_stdio():
                    atoms = io.read(expected_file)
                self.assertEqual(
                    atoms.calc.cell.kpoint_mp_grid.value,
                    list_to_string(input_params["k_points_grid"]),
                )
                expected_param_file = os.path.join(out_dir, s, s + ".param")
                self.assertTrue(os.path.exists(expected_param_file))
                with silence_stdio():
                    output_castep_param = read_param(expected_param_file).param
                self.assertEqual(
                    output_castep_param.cut_off_energy,
                    castep_param.cut_off_energy,
                )
                self.assertEqual(
                    output_castep_param.elec_energy_tol,
                    castep_param.elec_energy_tol,
                )
                # below test didn't work as cell positions get rounded...
                # equal = atoms.cell == input_atoms.cell
                # self.assertTrue(equal.all())

    def check_write_dftb(self, out_dir: str, input_atoms: Atoms):
        for rootDir, subDirs, files in os.walk(os.path.abspath(out_dir)):
            expected_files = ["geo_end.gen", "dftb_in.hsd"]
            for s in subDirs:
                count = 0
                for f in expected_files:
                    f = os.path.join(out_dir, s, f)
                    self.assertTrue(os.path.exists(f))
                    if count == 0:
                        with silence_stdio(True, True):
                            atoms = io.read(f)
                        equal = atoms.cell == input_atoms.cell
                        self.assertTrue(equal.all())
                    count += 1

    def check_write_uep(self, out_dir: str, input_params: dict):
        for (rootDir, subDirs, files) in os.walk(out_dir):
            for s in subDirs:
                expected_file = os.path.join(out_dir, s, s + ".yaml")
                self.assertTrue(os.path.exists(expected_file))
                params = load_input_file(expected_file, UEPOptSchema)
                self.assertEqual(params["geom_steps"], input_params["geom_steps"])
                self.assertEqual(params["opt_tol"], input_params["geom_force_tol"])
                self.assertEqual(params["gw_factor"], input_params["uep_gw_factor"])

    def run_uep(self, out_dir: str, uep_save_structs: bool = False):
        if platform.system() == "Windows":
            script_path = os.path.join(_TESTDATA_DIR, "script-uep-windows.ps1")
            subprocess.run(
                ["powershell", "-File", os.path.normpath(script_path), out_dir]
            )
        else:
            subprocess.run([os.path.join(_TESTDATA_DIR, "script-uep"), out_dir])

        # Check all folders contain UEP file
        for (rootDir, subDirs, files) in os.walk(out_dir):
            for s in subDirs:
                expected_file = os.path.join(out_dir, s, s + ".uep")
                exists = os.path.exists(expected_file)
                self.assertTrue(exists, f"{expected_file} does not exist")
                if uep_save_structs:
                    expected_file = os.path.join(out_dir, s, s + ".xyz")
                    exists = os.path.exists(expected_file)
                    self.assertTrue(exists, f"{expected_file} does not exist")

    def testBatchAll(self):
        try:
            yaml_file = os.path.join(_TESTDATA_DIR, "Si2-muairss-all.yaml")
            cell_dir = os.path.join(_TESTDATA_DIR, "batch")
            cell_file = os.path.join(_TESTDATA_DIR, "batch", "Si2.cell")
            param_file = os.path.join(_TESTDATA_DIR, "Si2.param")
            input_params = load_input_file(yaml_file, MuAirssSchema)
            with silence_stdio(True, True):
                castep_param = read_param(param_file).param
            with silence_stdio(True, True):
                input_atoms = io.read(cell_file)

            # Run Muairss write:
            sys.argv[1:] = ["-tw", cell_dir, yaml_file]
            os.chdir(_TESTDATA_DIR)
            run_muairss()
            # Check all folders contain a yaml file
            uep_dir = "muon-airss-out-all/Si2/uep"
            castep_dir = "muon-airss-out-all/Si2/castep"
            dftb_dir = "muon-airss-out-all/Si2/dftb+"

            self.check_write_uep(uep_dir, input_params)
            self.check_write_castep(castep_dir, input_params, castep_param)
            self.check_write_dftb(dftb_dir, input_atoms)

            # Run calculators (mocking where needed)
            self.run_uep(uep_dir, True)
            src = os.path.join(_TESTDATA_DIR, "castep-results", "castep")
            shutil.rmtree(castep_dir)
            os.symlink(src, castep_dir)
            if _RUN_DFTB:
                subprocess.run([os.path.join(_TESTDATA_DIR, "script-dftb"), dftb_dir])
            else:
                src = os.path.join(_TESTDATA_DIR, "dftb-results", "dftb+")
                shutil.rmtree(dftb_dir)
                os.symlink(src, dftb_dir)

            sys.argv[1:] = [cell_dir, yaml_file]
            run_muairss()

            self.assertTrue(os.path.exists("Si2_clusters.txt"))
            self.assertTrue(os.path.exists("Si2_Si2_uep_clusters.dat"))
        finally:
            #  Remove all created files and folders
            _clean_testdata_dir()

    def testUEP(self):
        try:
            yaml_file = os.path.join(_TESTDATA_DIR, "Si2-muairss-uep.yaml")
            cell_file = os.path.join(_TESTDATA_DIR, "Si2.cell")
            input_params = load_input_file(yaml_file, MuAirssSchema)

            # Run Muairss write:
            sys.argv[1:] = ["-tw", cell_file, yaml_file]
            os.chdir(_TESTDATA_DIR)
            run_muairss()
            # Check all folders contain a yaml file
            out_dir = "muon-airss-out-uep/uep"
            self.check_write_uep(out_dir, input_params)

            # Run UEP
            self.run_uep(out_dir)

            # Run Muairss read:
            sys.argv[1:] = [cell_file, yaml_file]
            run_muairss()

            self.assertTrue(os.path.exists("Si2_clusters.txt"))
            self.assertTrue(os.path.exists("Si2_Si2_uep_clusters.dat"))
        finally:
            #  Remove all created files and folders
            _clean_testdata_dir()

    def testUEP_custom_particle(self):
        try:
            yaml_file = os.path.join(_TESTDATA_DIR, "Si2-muairss-uep-Li8.yaml")
            cell_file = os.path.join(_TESTDATA_DIR, "Si2.cell")
            input_params = load_input_file(yaml_file, MuAirssSchema)

            # Run Muairss write:
            sys.argv[1:] = ["-tw", cell_file, yaml_file]
            os.chdir(_TESTDATA_DIR)
            run_muairss()
            # Check all folders contain a yaml file
            for (rootDir, subDirs, files) in os.walk("muon-airss-out-uep/uep/"):
                for s in subDirs:
                    expected_file = os.path.join(
                        "muon-airss-out-uep/uep/" + s, s + ".yaml"
                    )
                    self.assertTrue(os.path.exists(expected_file))
                    params = load_input_file(expected_file, UEPOptSchema)
                    self.assertEqual(
                        params["particle_mass"],
                        input_params["particle_mass_amu"]
                        * pcnst["atomic mass constant"][0],
                    )

            # Run UEP
            if platform.system() == "Windows":
                script_path = os.path.join(_TESTDATA_DIR, "script-uep-windows.ps1")
                subprocess.call(["powershell", "-File", os.path.normpath(script_path)])
            else:
                subprocess.call(os.path.join(_TESTDATA_DIR, "script-uep"))

            # Check all folders contain UEP file
            for (rootDir, subDirs, files) in os.walk("muon-airss-out-uep/uep/"):
                for s in subDirs:
                    expected_file = os.path.join(
                        "muon-airss-out-uep/uep/" + s, s + ".uep"
                    )
                    self.assertTrue(os.path.exists(expected_file))

            # Run Muairss read:
            sys.argv[1:] = [cell_file, yaml_file]
            run_muairss()

            all_path = input_params["allpos_filename"]
            self.assertTrue(os.path.exists(all_path))
            with open(all_path) as all_file:
                self.assertIn(input_params["mu_symbol"], all_file.read())
            cluster_path = os.path.join(
                "Si2_clusters", "uep", "Si2_uep_min_cluster_1.cell"
            )
            self.assertTrue(os.path.exists(cluster_path))
            with open(cluster_path) as cluster_file:
                self.assertIn(input_params["mu_symbol"], cluster_file.read())
        finally:
            #  Remove all created files and folders
            _clean_testdata_dir()

    def testCASTEP(self):
        try:
            yaml_file = os.path.join(_TESTDATA_DIR, "Si2-muairss-castep.yaml")
            cell_file = os.path.join(_TESTDATA_DIR, "Si2.cell")
            param_file = os.path.join(_TESTDATA_DIR, "Si2.param")
            input_params = load_input_file(yaml_file, MuAirssSchema)
            with silence_stdio(True, True):
                castep_param = read_param(param_file).param

            # Run Muairss write:
            sys.argv[1:] = ["-tw", cell_file, yaml_file]
            os.chdir(_TESTDATA_DIR)
            run_muairss()
            # Check all folders contain a yaml file
            out_dir = "muon-airss-out-castep/castep"
            self.check_write_castep(out_dir, input_params, castep_param)

            yaml_file = os.path.join(_TESTDATA_DIR, "Si2-muairss-castep-read.yaml")
            sys.argv[1:] = [cell_file, yaml_file]
            run_muairss()

            self.assertTrue(os.path.exists("Si2_clusters.txt"))
            self.assertTrue(os.path.exists("Si2_Si2_castep_clusters.dat"))

            # Test clustering_write_input has produced files we expect:
            self.assertTrue(os.path.exists("Si2_clusters"))
            calc_folder = "Si2_clusters/castep/"
            for (rootDir, subDirs, files) in os.walk(calc_folder):
                for s in subDirs:
                    expected_file = os.path.join(calc_folder + s, s + ".cell")
                    self.assertTrue(os.path.exists(expected_file))
                    with silence_stdio():
                        atoms = io.read(expected_file)
                    self.assertEqual(
                        atoms.calc.cell.kpoint_mp_grid.value,
                        list_to_string(input_params["k_points_grid"]),
                    )
                    expected_param_file = os.path.join(calc_folder + s, s + ".param")
                    self.assertTrue(os.path.exists(expected_param_file))
                    with silence_stdio():
                        output_castep_param = read_param(expected_param_file).param
                    self.assertEqual(
                        output_castep_param.cut_off_energy,
                        castep_param.cut_off_energy,
                    )
                    self.assertEqual(
                        output_castep_param.elec_energy_tol,
                        castep_param.elec_energy_tol,
                    )
        finally:
            # Remove all created files and folders
            _clean_testdata_dir()

    def testCASTEP_custom_particle(self):
        try:
            yaml_file = os.path.join(_TESTDATA_DIR, "Si2-muairss-castep-Li8.yaml")
            cell_file = os.path.join(_TESTDATA_DIR, "Si2.cell")
            input_params = load_input_file(yaml_file, MuAirssSchema)

            # Run Muairss write:
            sys.argv[1:] = ["-tw", cell_file, yaml_file]
            os.chdir(_TESTDATA_DIR)
            run_muairss()
            # Check all folders contain a yaml file
            for (rootDir, subDirs, files) in os.walk("muon-airss-out-castep/castep/"):
                for s in subDirs:
                    expected_file = os.path.join(
                        "muon-airss-out-castep/castep/" + s, s + ".cell"
                    )
                    self.assertTrue(os.path.exists(expected_file))
                    with silence_stdio():
                        atoms = io.read(expected_file)
                    self.assertEqual(
                        atoms.get_array("castep_custom_species")[-1],
                        input_params["mu_symbol"],
                    )
                    # masses broken in ASE
                    # self.assertEqual(
                    #     atoms.get_masses()[-1],
                    #     input_params["particle_mass_amu"]
                    # )
                    # use alternative
                    with open(expected_file) as exp_file:
                        self.assertIn(
                            str(input_params["particle_mass_amu"]), exp_file.read()
                        )

            yaml_file = os.path.join(_TESTDATA_DIR, "Si2-muairss-castep-read-Li8.yaml")
            sys.argv[1:] = [cell_file, yaml_file]
            run_muairss()

            self.assertTrue(os.path.exists("Si2_clusters.txt"))
            self.assertTrue(os.path.exists("Si2_Si2_castep_clusters.dat"))

            # Test clustering_write_input has produced files we expect:
            all_path = input_params["allpos_filename"]
            self.assertTrue(os.path.exists(all_path))
            with open(all_path) as all_file:
                self.assertIn(input_params["mu_symbol"], all_file.read())

            self.assertTrue(os.path.exists("Si2_clusters"))
            calc_folder = "Si2_clusters/castep/"
            for (rootDir, subDirs, files) in os.walk(calc_folder):
                for s in subDirs:
                    expected_file = os.path.join(calc_folder + s, s + ".cell")
                    self.assertTrue(os.path.exists(expected_file))
                    with silence_stdio():
                        atoms = io.read(expected_file)
                    self.assertEqual(
                        atoms.get_chemical_symbols()[-1],
                        get_element_from_custom_symbol(input_params["mu_symbol"]),
                    )
                    self.assertEqual(
                        atoms.get_array("castep_custom_species")[-1],
                        input_params["mu_symbol"],
                    )
                    # masses broken in ASE
                    # self.assertEqual(
                    #     atoms.get_masses()[-1],
                    #     input_params["particle_mass_amu"]
                    # )
                    # use alternative
                    with open(expected_file) as exp_file:
                        self.assertIn(
                            str(input_params["particle_mass_amu"]), exp_file.read()
                        )

        finally:
            # Remove all created files and folders
            _clean_testdata_dir()

    def testDFTB(self):
        try:
            yaml_file = os.path.join(_TESTDATA_DIR, "Si2-muairss-dftb.yaml")
            cell_file = os.path.join(_TESTDATA_DIR, "Si2.cell")
            input_params = load_input_file(yaml_file, MuAirssSchema)
            with silence_stdio(True, True):
                input_atoms = io.read(cell_file)

            # Run Muairss write:
            sys.argv[1:] = ["-tw", cell_file, yaml_file]
            os.chdir(_TESTDATA_DIR)
            run_muairss()
            # Check all folders contain a dftb_in.hsd and geo_end.gen
            out_dir = "muon-airss-out-dftb/dftb+"
            self.check_write_dftb(out_dir, input_atoms)

            # Run DFTB
            if _RUN_DFTB:
                subprocess.run([os.path.join(_TESTDATA_DIR, "script-dftb"), out_dir])
            else:
                yaml_file = os.path.join(_TESTDATA_DIR, "Si2-muairss-dftb-read.yaml")

            input_params = load_input_file(yaml_file, MuAirssSchema)

            sys.argv[1:] = [cell_file, yaml_file]
            run_muairss()
            clust_folder = "Si2_clusters/dftb+"
            self.assertTrue(os.path.exists(clust_folder))
            self.assertTrue(
                len(
                    glob.glob(
                        os.path.join(
                            clust_folder,
                            "*.{0}".format(input_params["clustering_save_format"]),
                        )
                    )
                )
                > 0
            )

            self.assertTrue(os.path.exists("Si2_clusters.txt"))
            self.assertTrue(os.path.exists("Si2_Si2_dftb+_clusters.dat"))
        finally:
            #  Remove all created files and folders
            _clean_testdata_dir()

    def testDFTB_custom_particle(self):
        try:
            yaml_file = os.path.join(_TESTDATA_DIR, "Si2-muairss-dftb-Li8.yaml")
            cell_file = os.path.join(_TESTDATA_DIR, "Si2.cell")
            input_params = load_input_file(yaml_file, MuAirssSchema)

            # Run Muairss write:
            sys.argv[1:] = ["-tw", cell_file, yaml_file]
            os.chdir(_TESTDATA_DIR)
            run_muairss()
            # Check all folders contain a dftb_in.hsd and geo_end.gen
            for rootDir, subDirs, files in os.walk(
                os.path.abspath("muon-airss-out-dftb/dftb+")
            ):
                expected_files = ["geo_end.gen", "dftb_in.hsd"]

                for s in subDirs:
                    count = 0
                    for f in expected_files:
                        f = os.path.join("muon-airss-out-dftb/dftb+/" + s, f)
                        self.assertTrue(os.path.exists(f))
                        if count == 1:
                            with open(f) as hsd_file:
                                self.assertIn(
                                    str(input_params["particle_mass_amu"]),
                                    hsd_file.read(),
                                )
                        count += 1

            # Run DFTB
            if _RUN_DFTB:
                subprocess.call(os.path.join(_TESTDATA_DIR, "script-dftb"))
            else:
                yaml_file = os.path.join(
                    _TESTDATA_DIR, "Si2-muairss-dftb-read-Li8.yaml"
                )

            input_params = load_input_file(yaml_file, MuAirssSchema)

            sys.argv[1:] = [cell_file, yaml_file]
            run_muairss()

            all_path = input_params["allpos_filename"]
            self.assertTrue(os.path.exists(all_path))
            with open(all_path) as all_file:
                self.assertIn(input_params["mu_symbol"], all_file.read())
            cluster_path = os.path.join(
                "Si2_clusters", "dftb+", "Si2_dftb+_min_cluster_1.cell"
            )
            self.assertTrue(os.path.exists(cluster_path))
            with open(cluster_path) as cluster_file:
                self.assertIn(input_params["mu_symbol"], cluster_file.read())

        finally:
            #  Remove all created files and folders
            _clean_testdata_dir()

    def tearDown(self):
        _clean_testdata_dir()


if __name__ == "__main__":

    unittest.main()
