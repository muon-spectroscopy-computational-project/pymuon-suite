# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import json

import numpy as np

from ase import io
from ase.calculators.dftb import Dftb
from ase.calculators.singlepoint import SinglePointCalculator

from pymuonsuite.utils import BackupFile


def dftb_write_input(a, folder, calc=None, name=None):
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

    if name is None:
        name = os.path.split(folder)[-1]  # Same as folder name

    if calc is not None:
        calc.atoms = a
        a.set_calculator(calc)

    if not isinstance(a.calc, Dftb):
        a = a.copy()
        calc = Dftb(label=name, atoms=a, run_manyDftb_steps=True)
        a.set_calculator(calc)

    a.calc.label = name
    a.calc.directory = folder
    a.calc.write_input(a)


def load_muonconf_dftb(folder):
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
            calc.read_results()

        energy = calc.get_potential_energy()
        forces = calc.get_forces()
        charges = calc.get_charges(atoms)

        calc = SinglePointCalculator(atoms, energy=energy,
                                     forces=forces, charges=charges)

        atoms.set_calculator(calc)

    return atoms


def parse_spinpol_dftb(folder):
    """Parse atomic spin populations from a detailed.out DFTB+ file."""

    with open(os.path.join(folder, 'detailed.out')) as f:
        lines = f.readlines()

    # Find the atomic populations blocks
    spinpol = {
        'up': [],
        'down': [],
    }

    for i, l in enumerate(lines):
        if 'Atom populations' in l:
            s = l.split()[2][1:-1]
            if s not in spinpol:
                raise RuntimeError('Invalid detailed.out file')
            for ll in lines[i+2:]:
                lspl = ll.split()
                try:
                    n, pop = map(float, lspl)
                except ValueError:
                    break
                spinpol[s].append(pop)

    # Build population and net spin
    N = len(spinpol['up'])
    if N == 0:
        raise RuntimeError('No atomic populations found in detailed.out')

    pops = np.zeros((N, 2))
    pops[:, 0] = spinpol['up']

    if len(spinpol['down']) == 0:
        return pops
    elif len(spinpol['down']) == N:
        pops[:, 1] = pops[:, 0]-spinpol['down']
        pops[:, 0] += spinpol['down']
        return pops
    else:
        raise RuntimeError('Incomplete populations in detailed.out')

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
                     run_manyDftb_steps=True, **args)
    else:
        dcalc = Dftb(label=name, atoms=a, run_manyDftb_steps=True, **args)

    dcalc.directory = folder
    dcalc.write_input(a)
