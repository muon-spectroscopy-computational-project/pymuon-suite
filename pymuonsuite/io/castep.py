# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import yaml
from ase import io
from ase.calculators.castep import Castep

from pymuonsuite.utils import list_to_string


def save_muonconf_castep(a, folder, params):
    # Muon mass and gyromagnetic ratio
    mass_block = 'AMU\n{0}       0.1138'
    gamma_block = 'radsectesla\n{0}        851586494.1'

    if isinstance(a.calc, Castep):
        ccalc = a.calc
    else:
        ccalc = Castep(castep_command=params['castep_command'])

    ccalc.cell.kpoint_mp_grid.value = list_to_string(params['k_points_grid'])
    ccalc.cell.species_mass = mass_block.format(params['mu_symbol']
                                                ).split('\n')
    ccalc.cell.species_gamma = gamma_block.format(params['mu_symbol']
                                                  ).split('\n')
    ccalc.cell.fix_all_cell = True  # To make sure for older CASTEP versions

    a.set_calculator(ccalc)

    name = os.path.split(folder)[-1]
    io.write(os.path.join(folder, '{0}.cell'.format(name)), a)
    ccalc.atoms = a

    if params['castep_param'] is not '':
        castep_params = yaml.load(open(params['castep_param'], 'r'))
    else:
        castep_params = {}

    castep_params['task'] = "GeometryOptimization"

    # Parameters from .yaml will overwrite parameters from .param
    castep_params['geom_max_iter'] = params['geom_steps']
    castep_params['geom_force_tol'] = params['geom_force_tol']
    castep_params['max_scf_cycles'] = params['max_scc_steps']

    parameter_file = os.path.join(folder, '{0}.param'.format(name))
    yaml.safe_dump(castep_params, open(parameter_file, 'w'),
                   default_flow_style=False)
