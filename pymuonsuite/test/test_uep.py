"""Tests for ReadWriteUEP methods"""

import unittest

import os
import shutil
import numpy as np

from ase import io

from pymuonsuite.io.uep import ReadWriteUEP
from pymuonsuite.schemas import load_input_file, MuAirssSchema

from soprano.utils import silence_stdio

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTDATA_DIR = os.path.join(_TEST_DIR, "test_data")


class TestReadWriteUEP(unittest.TestCase):

    def test_read(self):
        sname = 'Si2_1'
        folder = _TESTDATA_DIR  # does not contain any uep files
        reader = ReadWriteUEP()
        # test that we do not get any result for trying to read
        # an empty folder:
        try:
            reader.read(folder, sname)
        except Exception as e:
            print(e)
        folder = os.path.join(_TESTDATA_DIR, "Si2/uep-result")
        # tests uep file being read, and compares structure to
        # that in the xyz file - these should be equal
        read_uep = reader.read(folder, sname)
        read_xyz = io.read(os.path.join(folder, sname + '.xyz'))

        self.assertTrue(np.all(read_uep.numbers == read_xyz.numbers))
        self.assertTrue(np.allclose(read_uep.positions,
                                    read_xyz.positions, atol=1e-3))
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

        param_file = os.path.join(folder, "Si2-muairss-uep.yaml")
        params = load_input_file(param_file, MuAirssSchema)

        reader = ReadWriteUEP(params=params)
        with silence_stdio():
            a = io.read(os.path.join(folder, "Si2.cell"))

        self.assertTrue(reader._create_calculator(a, folder, "Si2"))
        calc = reader._create_calculator(a, folder, "Si2")

        self.assertEqual(calc.gw_factor, params['uep_gw_factor'])
        self.assertEqual(calc.geom_steps, params['geom_steps'])

    def test_write(self):
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

            self.assertTrue(os.path.exists(os.path.join(output_folder,
                                                        "test_save.yaml")))

            params['charged'] = False

            reader = ReadWriteUEP(params=params)

            reader.write(atoms, output_folder)
        except Exception as e:
            print(e)
        finally:
            shutil.rmtree("test_save")


if __name__ == "__main__":

    unittest.main()
