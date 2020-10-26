# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import json
import pickle

import numpy as np

from ase import io
from ase.calculators.dftb import Dftb
from ase.calculators.singlepoint import SinglePointCalculator

from pymuonsuite.utils import BackupFile
from pymuonsuite.calculate.hfine import compute_hfine_mullpop
from pymuonsuite import constants

from pymuonsuite.quantum.vibrational.phonons import ase_phonon_calc
from pymuonsuite.io.output import write_phonon_report


class ReadWriteDFTB(object):

    def read(self, folder, sname=None, calc_type=None, avg_prop=None):
        # dftb doesn't do seednames
        """Read a DFTB+ output non-destructively.

        Args:
        directory (str): path to a directory to load DFTB+ results

        Returns:
        atoms (ase.Atoms): an atomic structure with the results attached in a
        SinglePointCalculator
        """
        if calc_type == "PHONONS":
            if sname is not None:
                phonon_source_file = os.path.join(folder, sname + '.phonons.pkl')
            else:
                print("Read error: {0}".format("Phonons filename was not given."))
                return
            try:
                return self.read_dftb_phonons(phonon_source_file)
            except IOError as e:
                print('Read/write error: {0}'.format(e))
                return

        else:
            try:
                atoms = io.read(os.path.join(folder, 'geo_end.gen'))
            except IOError as e:
                print('Read/write error: {0}'.format(e))
                return

            atoms.info['name'] = os.path.split(folder)[-1]
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

            if calc_type == "MAGRES" and avg_prop == 'hyperfine':
                try:
                    pops = parse_spinpol_dftb(folder)
                    hfine = []
                    for i in range(len(atoms)):
                        hf = compute_hfine_mullpop(atoms, pops, self_i=i, fermi=True,
                                                fermi_neigh=True)
                        hfine.append(hf)
                    atoms.set_array('hyperfine', np.array(hfine))
                except IOError as e:
                    print('Read/write error: {0}'.format(e))
                    return

            return atoms

    def read_dftb_phonons(self, phonon_source_file):
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
                return ph_evals, ph_evecs
            except TypeError:
                raise RuntimeError(('Phonon file {0} does not contain gamma '
                                    'point data').format(phonon_source_file))

    def write(self, a, folder, sname=None,
              params={'dftb_set': '3ob-3-1', 'k_points_grid': None},
              calc=None, calc_type="GEOM_OPT", script=None, args=None):

        """Writes input files for an Atoms object with a Dftb+
        calculator.

        | Args:
        |   a (ase.Atoms):          Atoms object to write. Can have a Dftb
        |                           calculator attached to carry
        |                           arguments.
        |   folder (str):           Path to save the input files to.
        |   calc (ase.Calculator):  Calculator to attach to Atoms. If
        |                           present, the pre-existent one will
        |                           be ignored.
        |   name (str):             Seedname to save the files with. If not
        |                           given, use the name of the folder.
        """

        if calc_type == "PHONONS":
            self.write_phonons(a, params, args)

        else:
            if sname is None:
                sname = os.path.split(folder)[-1]  # Same as folder name

            if calc is not None:
                a.calc = calc

            if not isinstance(a.calc, Dftb):
                a = a.copy()
                calc = Dftb(label=sname, atoms=a)

            calc = self.create_calculator(calc, calc_type, params)
            print("PRODUCED CALC:", calc)

            a.set_calculator(calc)
            a.calc.label = sname
            a.calc.directory = folder
            a.calc.write_input(a)

            if script is not None:
                stxt = open(script).read()
                stxt = stxt.format(seedname=sname)
                with open(os.path.join(folder, 'script.sh'), 'w') as sf:
                    sf.write(stxt)

    def write_phonons(self, a, params, args):
        from pymuonsuite.data.dftb_pars import DFTBArgs

        dargs = DFTBArgs(params['dftb_set'])
        # Is it periodic?
        if params['pbc']:
            a.set_pbc(True)
            calc = Dftb(atoms=a, label='asephonons',
                        kpts=params['kpoint_grid'],
                        **dargs.args)
            ph_kpts = params['phonon_kpoint_grid']
        else:
            a.set_pbc(False)
            calc = Dftb(atoms=a, label='asephonons',
                        **dargs.args)
            ph_kpts = None
        a.set_calculator(calc)
        phdata = ase_phonon_calc(a, kpoints=ph_kpts,
                                 ftol=params['force_tol'],
                                 force_clean=params['force_clean'],
                                 name=params['name'])

        fext = os.path.splitext(args.structure_file)[-1]

        # Save optimised structure
        io.write(params['name'] + '_opt' + fext, phdata.structure)

        # And write out the phonons
        outf = params['name'] + '_opt.phonons.pkl'
        pickle.dump(phdata, open(outf, 'wb'))
        write_phonon_report(args, params, phdata)

    def create_calculator(self, calc=None, calc_type="muairss", params={'dftb_set': '3ob-3-1', 'k_points_grid': None, 'dftb_optionals': []}):
        from pymuonsuite.data.dftb_pars.dftb_pars import DFTBArgs

        if not isinstance(calc, Dftb):
            args = {}
        else:
            args = calc.todict()

        if calc_type == "muairss":
            calc = self.create_muairss_dftb_calculator(args, params)

        elif calc_type == "spinpol":
            calc = self.create_spinpol_dftbp_calculator(args, params)

        else:
            calc = None

        return calc

    def create_muairss_dftb_calculator(self, args={}, params={'dftb_set': '3ob-3-1',
                                        'k_points_grid': None, 'geom_force_tol': 0.01}):

        from pymuonsuite.data.dftb_pars.dftb_pars import DFTBArgs

        dargs = DFTBArgs(params['dftb_set'])
        print("dargs: ", dargs)

        for opt in params['dftb_optionals']:
            try:
                dargs.set_optional(opt, True)
            except KeyError:
                print(('WARNING: optional DFTB+ file {0} not available for {1}'
                    ' parameter set, skipping').format(opt, params['dftb_set'])
                    )

        args.update(dargs.args)
        args = dargs.args
        args['Driver_'] = 'ConjugateGradient'
        args['Driver_Masses_'] = ''
        args['Driver_Masses_Mass_'] = ''
        args['Driver_Masses_Mass_Atoms'] = '-1'
        args['Driver_Masses_Mass_MassPerAtom [amu]'] = str(constants.m_mu_amu)

        args['Driver_MaxForceComponent [eV/AA]'] = params['geom_force_tol']
        args['Driver_MaxSteps'] = params['geom_steps']
        args['Driver_MaxSccIterations'] = params['max_scc_steps']
        args['Hamiltonian_Charge'] = 1.0 if params['charged'] else 0.0

        if params['dftb_pbc']:
            calc = Dftb(kpts=params['k_points_grid'],
                        **args)
        else:
            calc = Dftb(**args)

        print("CALC:", calc)

        return calc

    def create_spinpol_dftbp_calculator(self, calc=None, params={'dftb_set': '3ob-3-1',
                                        'k_points_grid': None}):
        """Create a calculator containing all necessary parameters for a DFTB+
        SCC spin polarised calculation"""
        from pymuonsuite.data.dftb_pars import DFTBArgs
        # Create the arguments
        dargs = DFTBArgs(params['dftb_set'])
        # Make it spin polarised
        try:
            dargs.set_optional('spinpol.json', True)
        except KeyError:
            raise ValueError('DFTB+ parameter set does not allow spin polarised'
                            ' calculations')
        # Fix a few things, and add a spin on the muon
        args = dargs.args
        del(args['Hamiltonian_SpinPolarisation'])
        args['Hamiltonian_SpinPolarisation_'] = 'Colinear'
        args['Hamiltonian_SpinPolarisation_UnpairedElectrons'] = 1
        args['Hamiltonian_SpinPolarisation_InitialSpins_'] = ''
        args['Hamiltonian_SpinPolarisation_InitialSpins_Atoms'] = '-1'
        args['Hamiltonian_SpinPolarisation_InitialSpins_SpinPerAtom'] = 1

        if params['k_points_grid'] is not None:
            calc = Dftb(kpts=params['k_points_grid'],
                        **args)
        else:
            calc = Dftb(**args)

        calc.do_forces = True

        return calc


def dftb_write_input(a, folder, calc=None, name=None, script=None):
    """Writes input files for an Atoms object with a Dftb+
    calculator.

    | Args:
    |   a (ase.Atoms):          Atoms object to write. Can have a Dftb
    |                           calculator attached to carry
    |                           arguments.
    |   folder (str):           Path to save the input files to.
    |   calc (ase.Calculator):  Calculator to attach to Atoms. If
    |                           present, the pre-existent one will
    |                           be ignored.
    |   name (str):             Seedname to save the files with. If not
    |                           given, use the name of the folder.
    |   script (str):           Path to a file containing a submission script
    |                           to copy to the input folder. The script can 
    |                           contain the argument {seedname} in curly braces,
    |                           and it will be appropriately replaced.
    """

    if name is None:
        name = os.path.split(folder)[-1]  # Same as folder name

    if calc is not None:
        calc.atoms = a
        a.set_calculator(calc)

    if not isinstance(a.calc, Dftb):
        a = a.copy()
        calc = Dftb(label=name, atoms=a)
        a.set_calculator(calc)

    a.calc.label = name
    a.calc.directory = folder
    a.calc.write_input(a)

    if script is not None:
        stxt = open(script).read()
        stxt = stxt.format(seedname=name)
        with open(os.path.join(folder, 'script.sh'), 'w') as sf:
            sf.write(stxt)


def dftb_read_input(folder):
    """Read a DFTB+ output non-destructively.

    Args:
      directory (str): path to a directory to load DFTB+ results

    Returns:
      atoms (ase.Atoms): an atomic structure with the results attached in a
      SinglePointCalculator
    """

    atoms = io.read(os.path.join(folder, 'geo_end.gen'))
    atoms.info['name'] = os.path.split(folder)[-1]
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

    return atoms


def load_muonconf_dftb(folder):
    """Duplicate of dftb_read_input.
    Implemented here for backwards compatibility.
    """
    return dftb_read_input(folder)


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

# Deprecated, left in for compatibility


def save_muonconf_dftb(a, folder, params, dftbargs={}):

    from pymuonsuite.data.dftb_pars import DFTBArgs

    name = os.path.split(folder)[-1]

    a.set_pbc(params['dftb_pbc'])

    dargs = DFTBArgs(params['dftb_set'])

    custom_species = a.get_array('castep_custom_species')
    muon_index = np.where(custom_species == params['mu_symbol'])[0][0]

    is_spinpol = params.get('spin_polarized', False)
    if is_spinpol:
        dargs.set_optional('spinpol.json', True)

    # Add muon mass
    args = dargs.args
    args['Driver_'] = 'ConjugateGradient'
    args['Driver_Masses_'] = ''
    args['Driver_Masses_Mass_'] = ''
    args['Driver_Masses_Mass_Atoms'] = '{}'.format(muon_index)
    args['Driver_Masses_Mass_MassPerAtom [amu]'] = '0.1138'

    args['Driver_MaxForceComponent [eV/AA]'] = params['geom_force_tol']
    args['Driver_MaxSteps'] = params['geom_steps']
    args['Driver_MaxSccIterations'] = params['max_scc_steps']

    if is_spinpol:
        # Configure initial spins
        spins = np.array(a.get_initial_magnetic_moments())
        args['Hamiltonian_SpinPolarisation_InitialSpins'] = '{'
        args['Hamiltonian_SpinPolarisation_' +
             'InitialSpins_AllAtomSpins'] = '{' + '\n'.join(
            map(str, spins)) + '}'
        args['Hamiltonian_SpinPolarisation_UnpairedElectrons'] = str(
            np.sum(spins))

    # Add any custom arguments
    if isinstance(dftbargs, DFTBArgs):
        args.update(dftbargs.args)
    else:
        args.update(dftbargs)

    if params['dftb_pbc']:
        dcalc = Dftb(label=name, atoms=a,
                     kpts=params['k_points_grid'], 
                     **args)
    else:
        dcalc = Dftb(label=name, atoms=a, **args)

    dcalc.directory = folder
    dcalc.write_input(a)
