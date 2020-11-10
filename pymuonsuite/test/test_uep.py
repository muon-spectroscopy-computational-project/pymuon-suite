"""Tests for ReadWriteUEP methods"""

import unittest
import numpy as np

import os
import sys
import shutil

from copy import deepcopy

from ase import Atoms, io
from ase.io.castep import read_param

from pymuonsuite.utils import list_to_string
from pymuonsuite.io.uep import ReadWriteUEP
from pymuonsuite.schemas import load_input_file, MuAirssSchema


_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTDATA_DIR = os.path.join(_TEST_DIR, "test_data")
_TESTSAVE_DIR = os.path.join(_TEST_DIR, "test_save")


class TestReadWriteUEP(unittest.TestCase):

    def test_read(self):
        sname = 'Si2_1'
        folder = _TESTDATA_DIR  # does not contain any uep files
        reader = ReadWriteUEP()
        # test that we do not get any result for trying to read
        # an empty folder:
        #self.assertFalse(reader.read(folder, sname))
        try:
            reader.read(folder, sname)
        except Exception as e:
            print(e)

        #TODO: check we got the exception above

        folder = os.path.join(_TESTDATA_DIR, "uep")
        # # tests uep file being read:
        self.assertTrue(reader.read(folder, sname))

    def test_create_calc(self):
        folder = os.path.join(_TESTDATA_DIR, "uep")
        
        param_file = os.path.join(_TESTDATA_DIR, "uep/Si2.yaml")
        params = load_input_file(param_file, MuAirssSchema)

        reader = ReadWriteUEP(params=params)
        a = io.read(os.path.join(_TESTDATA_DIR, "uep/Si2.cell"))

        self.assertTrue(reader._ReadWriteUEP__create_calculator(a, folder, "Si2"))
        calc = reader._ReadWriteUEP__create_calculator(a, folder, "Si2")

        self.assertEqual(calc.gw_factor, params['uep_gw_factor'])
        self.assertEqual(calc.geom_steps, params['geom_steps'])

    def test_write(self):
        # read in cell file to get atom

        input_folder = _TESTDATA_DIR + "/uep"
        output_folder = _TESTSAVE_DIR

        atoms = io.read(os.path.join(input_folder, "Si2.cell"))

        # test writing geom_opt output
        param_file = os.path.join(input_folder, "Si2.yaml")
        params = load_input_file(param_file, MuAirssSchema)

        reader = ReadWriteUEP(params=params)

        reader.write(atoms, output_folder)

        self.assertTrue(os.path.exists(os.path.join(output_folder,
                        "test_save.yaml")))

        try:
            params['charged'] = False

            reader = ReadWriteUEP(params=params)

            reader.write(atoms, output_folder)
        except Exception as e:
            print(e)

        os.remove(os.path.join(output_folder,
                  "test_save.yaml"))


if __name__ == "__main__":

    unittest.main()
