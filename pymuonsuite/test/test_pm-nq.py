"""Tests for quantum averaging methods"""

# Python 2-to-3 compatibility code
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals

import unittest
import numpy as np
import scipy.constants as cnst

import argparse
import os
import sys
import shutil
import subprocess

from pymuonsuite.quantum.__main__ import asephonons_entry, nq_entry
from pymuonsuite.schemas import load_input_file, MuAirssSchema, UEPOptSchema
from ase import io

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTDATA_DIR = os.path.join(_TEST_DIR, "test_data/ethyleneMu")


class TestQuantum(unittest.TestCase):

    def test_asephonons(self, remove_data=True):
        yaml_file = os.path.join(_TESTDATA_DIR, "phonons.yaml")
        struct_file = os.path.join(_TESTDATA_DIR, "ethyleneMu.xyz")
        sys.argv[1:] = [struct_file, yaml_file]
        os.chdir(_TESTDATA_DIR)
        asephonons_entry()
        self.assertTrue(os.path.exists("ethyleneMu_opt.phonons.pkl"))
        dir_files = os.listdir(_TESTDATA_DIR)
        if remove_data:
            for f in dir_files:
                if not (f.endswith(".yaml") or f.endswith("Mu.xyz") 
                        or f.endswith(".sh")):
                    if os.path.isfile(f):
                        os.remove(f)

    def test_nq(self):
        yaml_file = os.path.join(_TESTDATA_DIR, "quantum.yaml")
        self.test_asephonons(remove_data=False)
        os.chdir(_TESTDATA_DIR)
        sys.argv[1:] = ["-w", yaml_file]
        nq_entry()

        subprocess.call(os.path.join(_TESTDATA_DIR,"run_dftb.sh"))
        sys.argv[1:] = [yaml_file]
        nq_entry()
        self.assertTrue(os.path.exists("averages.dat"))

        shutil.rmtree("ethyleneMu_opt_displaced")
        
        dir_files = os.listdir(_TESTDATA_DIR)
        for f in dir_files:
            if not (f.endswith(".yaml") or f.endswith("Mu.xyz")
                    or f.endswith(".sh")):
                if os.path.isfile(f):
                    os.remove(f)
        


if __name__ == "__main__":

    unittest.main()