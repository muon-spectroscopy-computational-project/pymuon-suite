"""Tests for quantum averaging methods"""

import unittest

import os
import sys
import shutil
import subprocess

from pymuonsuite.quantum.__main__ import asephonons_entry, nq_entry


_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTDATA_DIR = os.path.join(_TEST_DIR, "test_data/ethyleneMu")

_RUN_DFTB = False
#  Setting _RUN_DFTB to True requires dftb+ to be installed locally
#  and will test generating dftb phonons files, and running the
#  dftb+ files generated using pm-nq


class TestQuantum(unittest.TestCase):
    def remove_phonons_data(self, dir_files):
        for f in dir_files:
            if not (
                f.endswith(".yaml")
                or f.endswith("Mu.xyz")
                or f.endswith(".sh")
                or f.endswith(".castep")
                or f.endswith(".magres")
                or f.endswith(".phonon")
            ):
                if os.path.isfile(f):
                    os.remove(f)

    def test_asephonons(self, remove_data=True):
        if _RUN_DFTB:
            yaml_file = os.path.join(_TESTDATA_DIR, "phonons.yaml")
            struct_file = os.path.join(_TESTDATA_DIR, "ethyleneMu.xyz")
            sys.argv[1:] = [struct_file, yaml_file]
            os.chdir(_TESTDATA_DIR)
            asephonons_entry()
            self.assertTrue(os.path.exists("ethyleneMu_opt.phonons.pkl"))
            if remove_data:
                self.remove_phonons_data(os.listdir(_TESTDATA_DIR))

    def test_nq(self):
        folder = _TESTDATA_DIR + "/dftb-phonons/"
        os.chdir(folder)
        yaml_file = "quantum.yaml"
        structure_file = "ethyleneMu_opt.xyz"
        sys.argv[1:] = ["-tw", structure_file, yaml_file]
        nq_entry()

        if _RUN_DFTB:
            subprocess.call(os.path.join(folder, "run_dftb.sh"))
        else:
            os.chdir(_TESTDATA_DIR + "/dftb-nq-results")

        sys.argv[1:] = [structure_file, yaml_file]
        nq_entry()
        self.assertTrue(os.path.exists("averages.dat"))

        os.remove("averages.dat")
        if not _RUN_DFTB:
            os.chdir(folder)
        os.remove("ethyleneMu_opt_allconf.xyz")
        shutil.rmtree("ethyleneMu_opt_displaced")


if __name__ == "__main__":

    unittest.main()
