# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import shutil
import json

import numpy as np

from ase import io
from ase.calculators.dftb import Dftb
from ase.calculators.singlepoint import SinglePointCalculator

from pymuonsuite.data.dftb_pars import DFTBArgs


def save_muonconf_dftb(a, folder, params, dftbargs={}):

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
    args.update(dftbargs)

    if params['dftb_pbc']:
        dcalc = Dftb(label=name, atoms=a,
                     kpts=params['k_points_grid'],
                     run_manyDftb_steps=True, **args)
    else:
        dcalc = Dftb(label=name, atoms=a, run_manyDftb_steps=True, **args)

    dcalc.directory = folder
    dcalc.write_input(a)
