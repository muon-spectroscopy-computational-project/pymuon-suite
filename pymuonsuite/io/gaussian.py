# Python 2-to-3 compatibility code
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import glob
import os
from copy import deepcopy

import numpy as np
from ase import io
from ase.calculators.gaussian import Gaussian
from pymuonsuite import constants
from pymuonsuite.io.readwrite import ReadWrite
from soprano.utils import seedname


class ReadWriteGaussian(ReadWrite):
    def __init__(self, params={}, script=None, calc=None):
        '''
        |   params (dict):          Contains name of gaussian input file (str)
        |                           to read parameters from,
        |                           and whether to make the muon charged (bool)
        |                           e.g. {'gaussian_input': 'ethylene-SP.com',
        |                           'charged': False}
        |   script (str):           Path to a file containing a submission
        |                           script to copy to the input folder. The
        |                           script can contain the argument
        |                           {seedname} in curly braces, and it will
        |                           be appropriately replaced.
        |   calc (ase.Calculator):  Gaussian calculator to attach to Atoms. If
        |                           present, the pre-existent one will
        |                           be ignored.

        '''
        self.params = self._validate_params(params)
        self.script = script
        self._calc = calc

    def _validate_params(self, params):
        if not (isinstance(params, dict)):
            raise ValueError('params should be a dict, not ', type(params))
            return
        else:
            return params

    def set_params(self, params):
        '''
        |   params (dict)           Contains name of gaussian input file to
        |                           read parameters from,
        |                           and whether to make the muon charged
        |                           e.g. {'gaussian_input': 'ethylene-SP.com',
        |                           'charged': False}
        '''
        self.params = self._validate_params(params)
        # if the params have been changed, the calc has to be remade
        # from scratch:
        self._calc = None
        self._create_calculator()

    def read(self, folder, sname=None, read_hyperfine=False):
        """Reads Gaussian output files.

        | Args:
        |   folder (str):           Path to folder from which to read files.
        |   sname (str):            Seedname to save the files with. If not
        |                           given, use the name of the folder.
        |   read_hyperfine (bool):  If true, reads the fermi contact terms
        |                           (MHz) for the atoms from the output file
        |                           and attaches these to the atoms as a
        |                           custom array called 'hyperfine'.
        |
        """
        atoms = self._read_gaussian(folder, sname, read_hyperfine)
        return atoms

    def _read_gaussian(self, folder, sname=None, read_hyperfine=False):
        try:
            if sname is not None:
                gfile = os.path.join(folder, sname + '.out')
            else:
                gfile = glob.glob(os.path.join(folder, '*.out'))[0]
                sname = seedname(gfile)
            atoms = io.read(gfile)
            atoms.info['name'] = sname
            if read_hyperfine:
                self._read_gaussian_hyperfine(gfile, atoms)
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

    def _read_gaussian_hyperfine(self, filename, a):
        '''Reads fermi contact terms (MHz) from filename and attaches these to
        the atoms (a) as a custom array called hyperfine'''
        first_line = -1
        target_line = -1
        fermi_contact_terms = None
        with open(filename) as fd:
            for i, line in enumerate(fd):
                if 'Isotropic Fermi Contact Couplings' in line:
                    fermi_contact_terms = []
                    first_line = i+2
                    target_line = i+len(a.symbols)+1
                if first_line <= i <= target_line:
                    fermi_contact_terms.append(line.split()[3])

        if fermi_contact_terms:
            a.set_array('hyperfine', np.array(
                fermi_contact_terms))

        return a

    def write(self, a, folder, sname=None, calc_type=None):
        """Writes input files for an Atoms object with a Gaussian
        calculator. This assumes that the muon is in the final
        position, and adds to this atom's properties the muon
        mass and nuclear magnetic moment.

        | Args:
        |   a (ase.Atoms):          Atoms object to write. Can have a Gaussian
        |                           calculator attached to carry
        |                           keywords.
        |   folder (str):           Path to save the input files to.
        |   sname (str):            Seedname to save the files with. If not
        |                           given, use the name of the folder.


        Note: the settings from the gaussian input file take precedence if a
        gaussian input file has been set in the params of the ReadWriteGaussian
        instance, as does the charge setting in the params dict.
        Otherwise, the settings to be written out are taken from the
        calculator, and if no calculator has been set, default settings will
        be written.
        """

        if sname is None:
            sname = os.path.split(folder)[-1]  # Same as folder name

        self._calc = deepcopy(self._calc)

        # We only use the calculator attached to the atoms object if a calc
        # has not been set when initialising the ReadWrite object OR we
        # have not called write() and made a calculator before.

        if self._calc is None:
            if isinstance(a.calc, Gaussian):
                self._calc = deepcopy(a.calc)
        self._create_calculator(sname)

        a.set_calculator(self._calc)

        a = self._add_muon_properties(a)

        io.write(os.path.join(folder, sname + '.com'),
                 a, **self._calc.parameters)

        if self.script is not None:
            stxt = open(self.script).read()
            stxt = stxt.format(seedname=sname)
            with open(os.path.join(folder, 'script.sh'), 'w') as sf:
                sf.write(stxt)

    def _add_muon_properties(self, a):
        # the muon is in the final position:
        masses = a.get_masses()
        masses[-1] = str(constants.m_mu_amu)
        a.set_masses(masses)
        NMagMs = a.calc.parameters.get('nmagmlist', None)
        if NMagMs is None:
            NMagMs = [None]*len(masses)
        NMagMs[-1] = str(constants.mu_nmagm)
        a.calc.parameters['nmagmlist'] = NMagMs

        return a

    def _create_calculator(self, sname=None):
        ''' Create a calculator with the parameters we want to write to
        the gaussian input file '''
        if self._calc is not None and isinstance(self._calc, Gaussian):
            self._calc = deepcopy(self._calc)
            calc_given = True
        else:
            self._calc = Gaussian()
            calc_given = False

        # read the gaussian input file:
        in_file = self.params.get('gaussian_input')
        if in_file is not None:
            self._calc.parameters = io.read(
                in_file, attach_calculator=True).calc.parameters

        else:
            # Only fall back to setting default values if the user has
            # not provided a gaussian input file or a gaussian calculator.
            if not calc_given:
                if sname is None:
                    sname = 'gaussian'
                parameters = {'chk': '{}.chk'.format(sname),
                              'method': 'uB3LYP', 'basis': 'EPR-III',
                              'opt': ['Tight', 'MaxCyc=100'], 'mult': 2}
                self._calc.parameters = parameters

        charge_param = self.params.get('charged')

        if charge_param is not None:
            self._calc.parameters['charge'] = charge_param*1.0
        else:
            if self._calc.parameters.get('charge') is None:
                self._calc.parameters['charge'] = False*1.0

        return self._calc
