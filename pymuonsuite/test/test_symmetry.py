"""Tests for pm-symmetry"""

import contextlib
import io
import os
import sys
import unittest

from pymuonsuite.symmetry import main as run_symmetry

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTDATA_DIR = os.path.join(_TEST_DIR, "test_data/Si2")

EXPECTED_REGEX = r"""Wyckoff points symmetry report for .*
Space Group International Symbol: Fd-3m
Space Group Hall Number: 525
Absolute\t\tFractional\t\tHessian constraints\tOccupied
\[2\.695 2\.695 5\.391\]\t\[0\.000 1\.000 1\.000\]\tisotropic\t\t\t
\[0\.674 0\.674 0\.674\]\t\[0\.125 0\.125 0\.125\]\tnone\t\t\t
\[0\.674 2\.022 2\.022\]\t\[0\.125 0\.125 0\.625\]\tnone\t\t\t
\[2\.022 0\.674 2\.022\]\t\[0\.125 0\.625 0\.125\]\tnone\t\t\t
\[2\.022 2\.022 3\.369\]\t\[0\.125 0\.625 0\.625\]\tnone\t\t\t
\[1\.348 1\.348 1\.348\]\t\[0\.250 0\.250 0\.250\]\tisotropic\t\t\tX
\[2\.695 2\.695 2\.695\]\t\[0\.500 0\.500 0\.500\]\tisotropic\t\t\t
\[2\.022 2\.022 0\.674\]\t\[0\.625 0\.125 0\.125\]\tnone\t\t\t
\[2\.022 3\.369 2\.022\]\t\[0\.625 0\.125 0\.625\]\tnone\t\t\t
\[3\.369 2\.022 2\.022\]\t\[0\.625 0\.625 0\.125\]\tnone\t\t\t
\[3\.369 3\.369 3\.369\]\t\[0\.625 0\.625 0\.625\]\tnone\t\t\t
\[4\.043 4\.043 4\.043\]\t\[0\.750 0\.750 0\.750\]\tisotropic\t\t\t
\[2\.695 5\.391 2\.695\]\t\[1\.000 0\.000 1\.000\]\tisotropic\t\t\t
\[5\.391 2\.695 2\.695\]\t\[1\.000 1\.000 0\.000\]\tisotropic\t\t\t
\[5\.391 5\.391 5\.391\]\t\[1\.000 1\.000 1\.000\]\tisotropic\t\t\t
"""


class TestSymmetry(unittest.TestCase):
    def test_symmetry(self):
        cell_file = os.path.join(_TESTDATA_DIR, "Si2.cell")
        sys.argv[1:] = [cell_file]
        os.chdir(_TESTDATA_DIR)
        with contextlib.redirect_stdout(io.StringIO()) as f:
            run_symmetry()
        self.assertRegex(f.getvalue(), EXPECTED_REGEX)


if __name__ == "__main__":

    unittest.main()
