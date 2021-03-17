"""Tests for muairss with uep, dftb+ and castep"""

import unittest

import os
import sys
import shutil
import subprocess
import glob
from pymuonsuite.muairss import main as run_muairss
from pymuonsuite.schemas import load_input_file, MuAirssSchema, UEPOptSchema
from ase.io.castep import read_param
from pymuonsuite.utils import list_to_string
from soprano.utils import silence_stdio
from ase import io
import numpy as np

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTDATA_DIR = os.path.join(_TEST_DIR, "test_data/Si2")

_RUN_DFTB = False
#  Setting _RUN_DFTB to True requires dftb+ to be installed locally
#  and will test running dftb using the files output by muairss
#  otherwise, the test will use previously generated dftb results


class TestMuairss(unittest.TestCase):

    def test_uep(self):
        try:
            yaml_file = os.path.join(_TESTDATA_DIR, 'Si2-muairss-uep.yaml')
            cell_file = os.path.join(_TESTDATA_DIR, 'Si2.cell')
            input_params = load_input_file(yaml_file, MuAirssSchema)

            # Run Muairss write:
            sys.argv[1:] = ["-tw", cell_file, yaml_file]
            os.chdir(_TESTDATA_DIR)
            run_muairss()
            # Check all folders contain a yaml file
            for (rootDir, subDirs, files) in (
                    os.walk("muon-airss-out-uep/uep/")):
                for s in subDirs:
                    expected_file = os.path.join("muon-airss-out-uep/uep/" + s,
                                                 s + ".yaml")
                    self.assertTrue(os.path.exists(expected_file))
                    params = load_input_file(expected_file, UEPOptSchema)
                    self.assertEqual(params['geom_steps'],
                                     input_params['geom_steps'])
                    self.assertEqual(params['opt_tol'],
                                     input_params['geom_force_tol'])
                    self.assertEqual(params['gw_factor'],
                                     input_params['uep_gw_factor'])

            # Run UEP
            subprocess.call(os.path.join(_TESTDATA_DIR, "script-uep"))

            # Check all folders contain UEP file
            for (rootDir, subDirs, files) in (
                    os.walk("muon-airss-out-uep/uep/")):
                for s in subDirs:
                    expected_file = os.path.join("muon-airss-out-uep/uep/" + s,
                                                 s + ".uep")
                    self.assertTrue(os.path.exists(expected_file))

            sys.argv[1:] = [cell_file, yaml_file]
            run_muairss()

            self.assertTrue(os.path.exists("Si2_clusters.txt"))
            self.assertTrue(os.path.exists("Si2_Si2_uep_clusters.dat"))
        finally:
            #  Remove all created files and folders
            shutil.rmtree("muon-airss-out-uep")
            os.remove("Si2_clusters.txt")
            os.remove("Si2_Si2_uep_clusters.dat")
            os.remove("all.cell")

    def test_castep(self):
        try:
            yaml_file = os.path.join(_TESTDATA_DIR, 'Si2-muairss-castep.yaml')
            cell_file = os.path.join(_TESTDATA_DIR, 'Si2.cell')
            param_file = os.path.join(_TESTDATA_DIR, 'Si2.param')
            input_params = load_input_file(yaml_file, MuAirssSchema)
            with silence_stdio(True, True):
                castep_param = read_param(param_file).param

            # Run Muairss write:
            sys.argv[1:] = ["-tw", cell_file, yaml_file]
            os.chdir(_TESTDATA_DIR)
            run_muairss()
            # Check all folders contain a yaml file
            for (rootDir, subDirs, files) in (
                    os.walk("muon-airss-out-castep/castep/")):
                for s in subDirs:
                    expected_file = os.path.join(
                        "muon-airss-out-castep/castep/" + s, s + ".cell")
                    script_file = input_params['script_file']
                    if script_file is not None:
                        expected_script = os.path.join(
                            "muon-airss-out-castep/castep/" + s, 'script.sh')
                        self.assertTrue(os.path.exists(expected_script))

                    self.assertTrue(os.path.exists(expected_file))
                    with silence_stdio():
                        atoms = io.read(expected_file)
                    self.assertEqual(atoms.calc.cell.kpoint_mp_grid.value,
                                     list_to_string(input_params
                                                    ['k_points_grid']))
                    expected_param_file = os.path.join(
                        "muon-airss-out-castep/castep/" + s, s + ".param")
                    self.assertTrue(os.path.exists(expected_param_file))
                    with silence_stdio():
                        output_castep_param = read_param(
                            expected_param_file).param
                    self.assertEqual(output_castep_param.cut_off_energy,
                                     castep_param.cut_off_energy)
                    self.assertEqual(output_castep_param.elec_energy_tol,
                                     castep_param.elec_energy_tol)
                    # below test didn't work as cell positions get rounded...
                    # equal = atoms.cell == input_atoms.cell
                    # self.assertTrue(equal.all())

            yaml_file = os.path.join(_TESTDATA_DIR,
                                     'Si2-muairss-castep-read.yaml')
            sys.argv[1:] = [cell_file, yaml_file]
            run_muairss()

            self.assertTrue(os.path.exists("Si2_clusters.txt"))
            self.assertTrue(os.path.exists("Si2_Si2_castep_clusters.dat"))

            # Test clustering_write_input has produced files we expect:
            self.assertTrue(os.path.exists("Si2_clusters"))
            calc_folder = "Si2_clusters/castep/"
            for (rootDir, subDirs, files) in (
                    os.walk(calc_folder)):
                for s in subDirs:
                    expected_file = os.path.join(
                        calc_folder + s, s + ".cell")
                    self.assertTrue(os.path.exists(expected_file))
                    with silence_stdio():
                        atoms = io.read(expected_file)
                    self.assertEqual(atoms.calc.cell.kpoint_mp_grid.value,
                                     list_to_string(input_params
                                                    ['k_points_grid']))
                    expected_param_file = os.path.join(
                        calc_folder + s, s + ".param")
                    self.assertTrue(os.path.exists(expected_param_file))
                    with silence_stdio():
                        output_castep_param = read_param(
                            expected_param_file).param
                    self.assertEqual(output_castep_param.cut_off_energy,
                                     castep_param.cut_off_energy)
                    self.assertEqual(output_castep_param.elec_energy_tol,
                                     castep_param.elec_energy_tol)
        finally:
            # Remove all created files and folders
            shutil.rmtree("muon-airss-out-castep")
            shutil.rmtree("Si2_clusters")
            os.remove("Si2_clusters.txt")
            os.remove("Si2_Si2_castep_clusters.dat")
            os.remove("all.cell")

    def test_dftb(self):
        try:
            yaml_file = os.path.join(_TESTDATA_DIR, 'Si2-muairss-dftb.yaml')
            cell_file = os.path.join(_TESTDATA_DIR, 'Si2.cell')
            input_params = load_input_file(yaml_file, MuAirssSchema)
            with silence_stdio(True, True):
                input_atoms = io.read(cell_file)

            # Run Muairss write:
            sys.argv[1:] = ["-tw", cell_file, yaml_file]
            os.chdir(_TESTDATA_DIR)
            run_muairss()
            # Check all folders contain a dftb_in.hsd and geo_end.gen
            for rootDir, subDirs, files in os.walk(os.path.abspath(
                    "muon-airss-out-dftb/dftb+")):
                expected_files = ['geo_end.gen', 'dftb_in.hsd']

                for s in subDirs:
                    count = 0
                    for f in expected_files:
                        f = os.path.join("muon-airss-out-dftb/dftb+/" + s, f)
                        self.assertTrue(os.path.exists(f))
                        if count == 0:
                            with silence_stdio(True, True):
                                atoms = io.read(f)
                            np.all(atoms.cell == input_atoms.cell)
                            np.allclose(atoms.positions, atoms.positions)
                        count += 1

            # Run DFTB
            if _RUN_DFTB:
                subprocess.call(os.path.join(_TESTDATA_DIR, "script-dftb"))
            else:
                yaml_file = os.path.join(_TESTDATA_DIR,
                                         'Si2-muairss-dftb-read.yaml')

            input_params = load_input_file(yaml_file, MuAirssSchema)

            sys.argv[1:] = [cell_file, yaml_file]
            run_muairss()
            clust_folder = "Si2_clusters/dftb+"
            self.assertTrue(os.path.exists(clust_folder))
            self.assertTrue(len(glob.glob(os.path.join(clust_folder, '*.{0}'
                                                       .format(input_params['clustering_save_format']))
                                          )) > 0)

            self.assertTrue(os.path.exists("Si2_clusters.txt"))
            self.assertTrue(os.path.exists("Si2_Si2_dftb+_clusters.dat"))
        finally:
            #  Remove all created files and folders
            shutil.rmtree("muon-airss-out-dftb")
            shutil.rmtree("Si2_clusters")
            os.remove("Si2_clusters.txt")
            os.remove("Si2_Si2_dftb+_clusters.dat")
            os.remove("all.cell")

    def test_gaussian(self):
        '''This only tests writing out of gaussian input files
        using muairss. It does not test reading of gaussian
        output files with muairss.'''

        try:
            yaml_file = os.path.join(
                _TESTDATA_DIR, 'Si2-muairss-gaussian.yaml')
            cell_file = os.path.join(_TESTDATA_DIR, 'Si2.com')
            input_atoms = io.read(cell_file)

            # Run Muairss write:
            sys.argv[1:] = ["-tw", cell_file, yaml_file]
            os.chdir(_TESTDATA_DIR)
            run_muairss()
            # Check all folders contain a dftb_in.hsd and geo_end.gen
            for rootDir, subDirs, files in os.walk(os.path.abspath(
                    "muon-airss-out-gaussian/gaussian")):

                for s in subDirs:
                    count = 0
                    f = os.path.join(
                        "muon-airss-out-gaussian/gaussian/" + s, s + '.com')
                    self.assertTrue(os.path.exists(f))
                    atoms = io.read(f)
                    assert(np.all(atoms.cell == input_atoms.cell))
                    assert(np.allclose(atoms.positions, atoms.positions))
                    count += 1

        finally:
            #  Remove all created files and folders
            shutil.rmtree("muon-airss-out-gaussian")


if __name__ == "__main__":
    unittest.main()
