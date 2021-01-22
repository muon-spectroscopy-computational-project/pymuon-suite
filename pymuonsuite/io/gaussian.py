# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import glob
import numpy as np
# import warnings

from copy import deepcopy

from ase import io
from ase.calculators.gaussian import Gaussian

# from soprano.utils import customize_warnings
from soprano.utils import seedname

from pymuonsuite import constants
from pymuonsuite.io.readwrite import ReadWrite

# customize_warnings()


class ReadWriteGaussian(ReadWrite):
    def __init__(self, params={}, script=None, calc=None):
        '''
        |   params (dict):          Contains gaussian input file and whether
        |                           to make the muon charged
        |   script (str):           Path to a file containing a submission
        |                           script to copy to the input folder. The
        |                           script can contain the argument
        |                           {seedname} in curly braces, and it will
        |                           be appropriately replaced.
        |   calc (ase.Calculator):  Calculator to attach to Atoms. If
        |                           present, the pre-existent one will
        |                           be ignored.
        '''
        if not (isinstance(params, dict)):
            raise ValueError('params should be a dict, not ', type(params))
            return
        else:
            self.set_params(params)
        self.script = script
        # if calc is not None and self.params != {}:
        #     self._create_calculator()

    def set_params(self, params):
        '''
        |   params (dict)           Contains muon symbol, parameter file,
        |                           k_points_grid.
        '''
        if not (isinstance(params, dict)):
            raise ValueError('params should be a dict, not ', type(params))
            return
        else:
            self.params = params
        # if the params have been changed, the calc has to be remade
        # from scratch:
        self._calc = None
        self._create_calculator()

    def set_script(self, script):
        '''
        |   script (str):           Path to a file containing a submission
        |                           script to copy to the input folder. The
        |                           script can contain the argument
        |                           {seedname} in curly braces, and it will
        |                           be appropriately replaced.
        '''
        self.script = script

    def read(self, folder, sname=None):
        """Reads Gaussian output files.

        | Args:
        |   folder (str):           Path to folder from which to read files.
        |   sname (str):            Seedname to save the files with. If not
        |                           given, use the name of the folder.
        """
        atoms = self._read_gaussian(folder, sname)
        return atoms

    def _read_gaussian(self, folder, sname=None):
        try:
            if sname is not None:
                gfile = os.path.join(folder, sname + '.out')
            else:
                gfile = glob.glob(os.path.join(folder, '*.out'))[0]
                sname = seedname(gfile)
            atoms = io.read(gfile)
            atoms.info['name'] = sname
            return atoms

        except IndexError:
            raise IOError("ERROR: No .out files found in {}."
                          .format(os.path.abspath(folder)))
        except OSError as e:
            raise IOError("ERROR: {}".format(e))
        except (io.formats.UnknownFileTypeError, ValueError, TypeError,
                Exception):
            raise IOError("ERROR: Invalid file: {file}"
                          .format(file=sname + '.out'))

    def write(self, a, folder, sname=None, calc_type=None):
        """Writes input files for an Atoms object with a Gaussian
        calculator.

        | Args:
        |   a (ase.Atoms):          Atoms object to write. Can have a Gaussian
        |                           calculator attached to carry
        |                           keywords.
        |   folder (str):           Path to save the input files to.
        |   sname (str):            Seedname to save the files with. If not
        |                           given, use the name of the folder.
        """

        if sname is None:
            sname = os.path.split(folder)[-1]  # Same as folder name

        # print("SEEDNAME: ", sname)
        # print("FOLDER: ", folder)
        # print("Atoms calc:", a.calc)

        a = self._add_muon_properties(a)

        self._calc = deepcopy(self._calc)

        # We only use the calculator attached to the atoms object if a calc
        # has not been set when initialising the ReadWrite object OR we
        # have not called write() and made a calculator before.

        if self._calc is None:
            if isinstance(a.calc, Gaussian):
                self._calc = deepcopy(a.calc)
        self._create_calculator()

        a.set_calculator(self._calc)

        io.write(os.path.join(folder, sname + '.com'), a, **self._calc.parameters)

        if self.script is not None:
            stxt = open(self.script).read()
            stxt = stxt.format(seedname=sname)
            with open(os.path.join(folder, 'script.sh'), 'w') as sf:
                sf.write(stxt)

    def _add_muon_properties(self, a):
        # the muon is in the final position:
        mu_index = len(a.positions)-1
        masses = a.get_masses()
        NMagMs = a.arrays.get('gaussian_NMagM', None)
        masses[mu_index] = str(constants.m_mu_amu)
        a.set_masses(masses)
        if NMagMs is None:
            NMagMs = [None]*len(masses)
            a.set_array('gaussian_NMagM', np.array(NMagMs))
        NMagMs[mu_index] = str(constants.mu_nmagm)
        a.arrays['gaussian_NMagM'] = NMagMs

        return a

    def _create_calculator(self):
        if self._calc is not None and isinstance(self._calc, Gaussian):
            self._calc = deepcopy(self._calc)
        else:
            self._calc = Gaussian()

        # read the gaussian input file:
        if 'gaussian_input' in self.params:
            in_file = self.params['gaussian_input']
            self._calc.parameters = io.read(
                in_file, get_calculator=True).calc.parameters

        return self._calc

