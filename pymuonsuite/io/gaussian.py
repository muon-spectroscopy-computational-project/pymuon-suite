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
from ase import Atoms
from ase.calculators.gaussian import Gaussian

# from soprano.utils import customize_warnings
from soprano.utils import seedname

from pymuonsuite import constants
from pymuonsuite.io.readwrite import ReadWrite

# customize_warnings()


class ReadWriteGaussian(ReadWrite):
    def __init__(self, params={}, script=None, calc=None):
        '''
        |   params (dict):           Contains basis set
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
            self.params = params
        self.script = script
        self._calc = calc
        if calc is not None and self.params != {}:
            self._create_calculator()

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
                Exception) as e:
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

        print("SEEDNAME: ", sname)
        print("FOLDER: ", folder)

        a = self._reposition_muon(a)
        # We will no longer need to reposition the muon, when the changes to
        # ASE have been made
        # TODO: replace with:
        a = self._add_muon_properties(a)

        self._calc = deepcopy(self._calc)

        # We only use the calculator attached to the atoms object if a calc
        # has not been set when initialising the ReadWrite object OR we
        # have not called write() and made a calculator before.

        if self._calc is None:
            if isinstance(a.calc, Gaussian):
                self._calc = deepcopy(a.calc)
        self._create_calculator(folder, sname)

        a.set_calculator(self._calc)

        print("LABEL: ", a.calc.label)

        a.calc.write_input(a)  # TODO: test this compared to below
        #io.write(os.path.join(folder, sname + '.com'), a)

        if self.script is not None:
            stxt = open(self.script).read()
            stxt = stxt.format(seedname=sname)
            with open(os.path.join(folder, 'script.sh'), 'w') as sf:
                sf.write(stxt)

    def _reposition_muon(self, a):
        # TODO: delete this method when the changes to ASE have been made
        mu_symbol = self.params.get('mu_symbol', 'H:mu')
        mu_index = list(a.arrays['castep_custom_species']).index(mu_symbol)
        # move muons to front of atoms, so we can set the mass
        # correctly in the input file:
        mu = a.pop(mu_index)
        a = Atoms('H', [mu.position], magmoms=[mu.magmom]) + a
        a.pbc = None
        a.cell = None
        return a

    def _add_muon_properties(self, a):
        mu_symbol = self.params.get('mu_symbol', 'H:mu')
        if mu_symbol in a.arrays['castep_custom_species']:
            mu_index = list(a.arrays['castep_custom_species']).index(mu_symbol)
        else:
            # otherwise, the muon is in the final position:
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

    def _create_calculator(self, folder, sname):
        if self._calc is not None and isinstance(self._calc, Gaussian):
            calc = deepcopy(self._calc)
        else:
            calc = Gaussian()
            # Read the parameters
            pfile = self.params.get('gaussian_input', None)
            if pfile is not None:
                route = self._read_route_section(pfile)
                calc.parameters = {'method': '',
                                   'extra': route}
            else:
                atoms = io.read("example_in.com")
                calc.parameters = {'method': 'uB3LYP',
                                   'basis': 'EPR-III',
                                   'opt': 'Tight, MaxCyc=100',
                                   'integral': "Ultrafine"}

            calc.parameters.update({'nprocshared': 16,
                                    'charge': 0,
                                    'mult': 2,
                                    'freq': 'ReadIso',
                                    'addsec': str(constants.m_mu_amu)})

        calc.parameters.update({'chk': sname + '.chk'})
        calc.label = os.path.join(folder, sname)

        print("PARAMS", calc.parameters)

        self._calc = calc

        return self._calc

    def _read_route_section(self, in_file):
        route = ""
        route_section = False
        with open(in_file) as gaussian_input:
            for line in gaussian_input:
                if str(line)[:1] == '#':
                    route += str(line[3:])
                    route_section = True
                elif route_section and str(line) != '\n':
                    route += str(line)
                elif route_section and str(line) == '\n':
                    return route.strip("\n")
        return route.strip("\n")
