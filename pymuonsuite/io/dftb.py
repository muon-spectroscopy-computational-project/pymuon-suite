# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import json
import pickle
import glob
import warnings
import numpy as np

from copy import deepcopy

from soprano.utils import customize_warnings

from ase import io
from ase.calculators.dftb import Dftb
from ase.calculators.singlepoint import SinglePointCalculator

from pymuonsuite.utils import BackupFile
from pymuonsuite.calculate.hfine import compute_hfine_mullpop
from pymuonsuite import constants

from pymuonsuite.quantum.vibrational.phonons import ase_phonon_calc
from pymuonsuite.io.output import write_phonon_report
from pymuonsuite.io.readwrite import ReadWrite

_geom_opt_args = {'Driver_': 'ConjugateGradient', 'Driver_Masses_': '',
                  'Driver_Masses_Mass_': '', 'Driver_Masses_Mass_Atoms': '-1',
                  'Driver_Masses_Mass_MassPerAtom [amu]':
                  str(constants.m_mu_amu)}

_spinpol_args = {'Hamiltonian_SpinPolarisation_': 'Colinear',
                 'Hamiltonian_SpinPolarisation_UnpairedElectrons': 1,
                 'Hamiltonian_SpinPolarisation_InitialSpins_': '',
                 'Hamiltonian_SpinPolarisation_InitialSpins_Atoms': '-1',
                 'Hamiltonian_SpinPolarisation_InitialSpins_SpinPerAtom': 1}

customize_warnings()


class ReadWriteDFTB(ReadWrite):
    def __init__(self, params={}, script=None, calc=None):
        '''
        |   Args:
        |   params (dict):          Contains dftb_set, k_points_grid,
        |                           geom_force_tol and dftb_optionals.
        |                           geom_steps, max_scc_steps, and
        |                           charged are also required in the case
        |                           of writing geom_opt input files
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

        self.set_params(params)
        self.script = script
        self._calc = calc
        self._calc_type = None

    def set_script(self, script):
        '''
        |   Args:
        |   script (str):           Path to a file containing a submission
        |                           script to copy to the input folder. The
        |                           script can contain the argument
        |                           {seedname} in curly braces, and it will
        |                           be appropriately replaced.
        '''
        self.script = script

    def set_params(self, params):
        '''
        |   Args:
        |   params (dict):          Contains dftb_set, k_points_grid,
        |                           geom_force_tol and dftb_optionals.
        |                           geom_steps, max_scc_steps, and
        |                           charged are also required in the case
        |                           of writing geom_opt input files
        '''
        if not (isinstance(params, dict)):
            raise ValueError('params should be a dict, not ', type(params))
            return

        if params == {}:
            params = {'dftb_set': '3ob-3-1', 'k_points_grid': None,
                      'geom_force_tol': 0.01, 'dftb_optionals': []}

        self.params = params
        # resetting this to None makes sure that the calc is recreated after
        # the params are updated:
        self._calc_type = None

    def read(self, folder, sname=None):
        ''' Read a DFTB+ output non-destructively.
        |
        |   Args:
        |   folder (str) :          path to a directory to load DFTB+ results
        |   sname (str):            name to label the atoms with and/or of the
        |                           .phonons.pkl file to be read
        |   Returns:
        |   atoms (ase.Atoms):      an atomic structure with the results
        |                           attached in a SinglePointCalculator
        '''

        try:
            atoms = io.read(os.path.join(folder, 'geo_end.gen'))

        except IOError:
            raise IOError("ERROR: No geo_end.gen file found in {}."
                          .format(os.path.abspath(folder)))
        except Exception as e:
            raise IOError("ERROR: Could not read {file}, due to error: {error}"
                          .format(file='geo_end.gen', error=e))
        if sname is None:
            atoms.info['name'] = os.path.split(folder)[-1]
        else:
            atoms.info['name'] = sname
        results_file = os.path.join(folder, "results.tag")
        if os.path.isfile(results_file):
            # DFTB+ was used to perform the optimisation
            temp_file = os.path.join(folder, "results.tag.bak")

            # We need to backup the results file here because
            # .read_results() will remove the results file
            with BackupFile(results_file, temp_file):
                calc = Dftb(atoms=atoms)
                calc.atoms_input = atoms
                calc.directory = folder
                calc.do_forces = True
                calc.read_results()

            energy = calc.get_potential_energy()
            forces = calc.get_forces()
            charges = calc.get_charges(atoms)

            calc = SinglePointCalculator(atoms, energy=energy,
                                         forces=forces, charges=charges)

            atoms.calc = calc

        try:
            pops = parse_spinpol_dftb(folder)
            hfine = []
            for i in range(len(atoms)):
                hf = compute_hfine_mullpop(atoms, pops, self_i=i, fermi=True,
                                           fermi_neigh=True)
                hfine.append(hf)
            atoms.set_array('hyperfine', np.array(hfine))
        except (IndexError, IOError) as e:
            warnings.warn('Could not read hyperfine details due to error: '
                          '{0}'.format(e))

        try:
            if sname is not None:
                phonon_source_file = os.path.join(folder, sname +
                                                  '.phonons.pkl')
            else:
                print("Phonons filename was not given, searching for any"
                      " .phonons.pkl file.")
                phonon_source_file = glob.glob(os.path.join(folder,
                                               '*.phonons.pkl'))[0]
            self._read_dftb_phonons(atoms, phonon_source_file)
        except IndexError:
            warnings.warn("No .phonons.pkl files found in {}."
                          .format(os.path.abspath(folder)))
        except IOError:
            warnings.warn("{} could not be found."
                          .format(phonon_source_file))
        except Exception as e:
            warnings.warn('Could not read {file} due to error: {error}'
                          .format(file=phonon_source_file, error=e))

        return atoms

    def _read_dftb_phonons(self, atoms, phonon_source_file):
        with open(phonon_source_file, 'rb') as f:
            phdata = pickle.load(f)
            # Find the gamma point
            gamma_i = None
            for i, p in enumerate(phdata.path):
                if (p == 0).all():
                    gamma_i = i
                    break
            try:
                ph_evals = phdata.frequencies[gamma_i]
                ph_evecs = phdata.modes[gamma_i]
                atoms.info['ph_evals'] = ph_evals
                atoms.info['ph_evecs'] = ph_evecs
            except TypeError:
                raise RuntimeError(('Phonon file {0} does not contain gamma '
                                    'point data').format(phonon_source_file))

    def write(self, a, folder, sname=None, calc_type="GEOM_OPT", args=None):

        """Writes input files for an Atoms object with a Dftb+
        calculator.

        | Args:
        |   a (ase.Atoms):          Atoms object to write. Can have a Dftb
        |                           calculator attached to carry
        |                           arguments.
        |   folder (str):           Path to save the input files to.
        |   sname (str):            Seedname to save the files with. If not
        |                           given, use the name of the folder.
        |   calc_type (str):        Calculation which will be performed:
        |                           "GEOM_OPT", "SPINPOL" or "PHONONS"
        |   args:                   Input arguments for writing phonons
        """

        if calc_type == "PHONONS":
            self._write_phonons(a, args)

        elif calc_type == "GEOM_OPT" or calc_type == "SPINPOL":
            if sname is None:
                sname = os.path.split(folder)[-1]  # Same as folder name

            if self._calc is None and isinstance(a.calc, Dftb):
                self._calc = a.calc

            self._calc = deepcopy(self._calc)

            # only create a new calc if the calc type requested is different
            # to that already saved.
            if calc_type != self._calc_type:
                self._create_calculator(calc_type=calc_type)

            a.set_calculator(self._calc)
            a.calc.label = sname
            a.calc.directory = folder
            a.calc.write_input(a)

            if self.script is not None:
                stxt = open(self.script).read()
                stxt = stxt.format(seedname=sname)
                with open(os.path.join(folder, 'script.sh'), 'w') as sf:
                    sf.write(stxt)
        else:
            raise(NotImplementedError("Calculation type {} is not implemented."
                  " Please choose 'PHONONS', 'GEOM_OPT' or 'SPINPOL'"
                                      .format(calc_type)))

    def _write_phonons(self, a, args):
        from pymuonsuite.data.dftb_pars import DFTBArgs

        dargs = DFTBArgs(self.params['dftb_set'])
        # Is it periodic?
        if self.params['pbc']:
            a.set_pbc(True)
            self._calc = Dftb(atoms=a, label='asephonons',
                              kpts=self.params['kpoint_grid'],
                              **dargs.args)
            ph_kpts = self.params['phonon_kpoint_grid']
        else:
            a.set_pbc(False)
            self._calc = Dftb(atoms=a, label='asephonons',
                              **dargs.args)
            ph_kpts = None
        a.set_calculator(self._calc)
        try:
            phdata = ase_phonon_calc(a, kpoints=ph_kpts,
                                     ftol=self.params['force_tol'],
                                     force_clean=self.params['force_clean'],
                                     name=self.params['name'])
        except Exception as e:
            print(e)
            print("Error: Could not write phonons file, see asephonons.out for"
                  " details.")
            return

        fext = os.path.splitext(args.structure_file)[-1]

        # Save optimised structure
        io.write(self.params['name'] + '_opt' + fext, phdata.structure)

        # And write out the phonons
        outf = self.params['name'] + '_opt.phonons.pkl'
        pickle.dump(phdata, open(outf, 'wb'))
        write_phonon_report(args, self.params, phdata)

    def _create_calculator(self, calc_type="GEOM_OPT"):
        from pymuonsuite.data.dftb_pars.dftb_pars import DFTBArgs

        if not isinstance(self._calc, Dftb):
            args = {}
        else:
            args = self._calc.todict()

        dargs = DFTBArgs(self.params['dftb_set'])

        if calc_type == "SPINPOL":
            if 'dftb_optionals' not in self.params:
                self.params['dftb_optionals'] = []
            self.params['dftb_optionals'].append('spinpol.json')
            if self.params['k_points_grid'] is not None:
                self.params['dftb_pbc'] = True

        if self.params['dftb_pbc']:
            self._calc = Dftb(kpts=self.params['k_points_grid'])
        else:
            self._calc = Dftb()

        for opt in self.params['dftb_optionals']:
            try:
                dargs.set_optional(opt, True)
            except KeyError:
                if opt == 'spinpol.json':
                    raise ValueError('DFTB+ parameter set does not allow spin'
                                     'polarised calculations')
                else:
                    warnings.warn('Warning: optional DFTB+ file {0} not'
                                  'available for {1}'
                                  ' parameter set, skipping').format(
                                  opt, self.params['dftb_set'])
        args.update(dargs.args)

        if calc_type == "GEOM_OPT":
            args.update(_geom_opt_args)
            geom_opt_param_args = {'Driver_MaxForceComponent [eV/AA]':
                                   self.params['geom_force_tol'],
                                   'Driver_MaxSteps':
                                   self.params['geom_steps'],
                                   'Driver_MaxSccIterations':
                                   self.params['max_scc_steps'],
                                   'Hamiltonian_Charge': 1.0 if
                                   self.params['charged'] else 0.0}

            args.update(geom_opt_param_args)

        elif calc_type == "SPINPOL":
            del(args['Hamiltonian_SpinPolarisation'])
            args.update(_spinpol_args)
            self._calc.do_forces = True
        else:
            raise(NotImplementedError("Calculation type {} is not implemented."
                  " Please choose 'GEOM_OPT' or 'SPINPOL'".format(calc_type)))

        self._calc.parameters.update(args)

        self._calc_type = calc_type

        return self._calc


def parse_spinpol_dftb(folder):
    """Parse atomic spin populations from a detailed.out DFTB+ file."""

    with open(os.path.join(folder, 'detailed.out')) as f:
        lines = f.readlines()

    # Find the atomic populations blocks
    spinpol = {
        'up': [],
        'down': [],
    }

    charges = {}

    for i, l in enumerate(lines):
        if 'Atomic gross charges (e)' in l:
            for ll in lines[i+2:]:
                lspl = ll.split()[:2]
                try:
                    a_i, q = int(lspl[0]), float(lspl[1])
                except (IndexError, ValueError):
                    break
                charges[a_i-1] = q

        if 'Orbital populations' in l:
            s = l.split()[2][1:-1]
            if s not in spinpol:
                raise RuntimeError('Invalid detailed.out file')
            for ll in lines[i+2:]:
                lspl = ll.split()[:5]
                try:
                    a_i, n, l, m, pop = map(float, lspl)
                except ValueError:
                    break
                a_i, n, l, m = map(int, [a_i, n, l, m])
                if len(spinpol[s]) < a_i:
                    spinpol[s].append({})
                spinpol[s][a_i-1][(n, l, m)] = pop

    # Build population and net spin
    N = len(spinpol['up'])
    if N == 0:
        raise RuntimeError('No atomic populations found in detailed.out')

    pops = [{} for i in range(N)]

    # Start with total populations and total spin
    for i in range(N):
        pops[i] = {
            'q': charges[i],
            'pop': 0,
            'spin': 0,
            'pop_orbital': {},
            'spin_orbital': {}
        }
        for s, sign in {'up': 1, 'down': -1}.items():
            for nlm, p in spinpol[s][i].items():
                pops[i]['pop'] += p
                pops[i]['spin'] += sign*p
                pops[i]['pop_orbital'][nlm] = pops[i]['pop_orbital'].get(
                    nlm, 0.0)+p
                pops[i]['spin_orbital'][nlm] = pops[i]['spin_orbital'].get(
                    nlm, 0.0)+p*sign

    return pops
