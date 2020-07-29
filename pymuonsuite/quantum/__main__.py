"""
Author: Adam Laverack
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import pickle
import numpy as np
import argparse as ap

from pymuonsuite.quantum.vibrational.phonons import ase_phonon_calc
from pymuonsuite.quantum.vibrational.average import (muon_vibrational_average_write,
                                                     muon_vibrational_average_read,
                                                     create_spinpol_dftbp_calculator)
from pymuonsuite.schemas import (load_input_file, MuonHarmonicSchema,
                                 AsePhononsSchema)
from pymuonsuite.io.output import write_phonon_report
from pymuonsuite.quantum.vibrational.utils import apply_muon_mass


def nq_entry():
    parser = ap.ArgumentParser()
    parser.add_argument('parameter_file', type=str,
                        help="YAML file containing relevant input parameters")
    parser.add_argument('-w',   action='store_true', default=False,
                        help="Create and write input files instead of parsing the results")

    args = parser.parse_args()

    # Load parameters
    params = load_input_file(args.parameter_file, MuonHarmonicSchema)

    # Temperature
    if params['average_T'] is None:
        params['average_T'] = params['displace_T']

    if args.w:
        muon_vibrational_average_write(**params)
    else:
        try:
            muon_vibrational_average_read(**params)
        except IOError as e:
            print('Read/write error: {0}'.format(e))
            print('\nThis could mean it was impossible to find the displaced '
                  'structure folder: maybe you wanted to run with the '
                  '-w option to write it?\n')


def asephonons_entry():

    from ase import io
    from ase.calculators.dftb import Dftb
    from pymuonsuite.data.dftb_pars import DFTBArgs

    parser = ap.ArgumentParser(description="Compute phonon modes with ASE and"
                               " DFTB+ for reuse in quantum effects "
                               "calculations.")
    parser.add_argument('structure_file', type=str,
                        help="Structure for which to compute the phonons")
    parser.add_argument('parameter_file', type=str,
                        help="YAML file containing relevant input parameters")

    args = parser.parse_args()

    # Load parameters
    params = load_input_file(args.parameter_file, AsePhononsSchema)

    fname, fext = os.path.splitext(args.structure_file)
    if params['name'] is None:
        params['name'] = fname

    # Load structure
    a = io.read(args.structure_file)
    # Apply mass
    a, mu_index = apply_muon_mass(a, params['mu_symbol'], params['mu_index'])
    # Create a Dftb calculator
    dargs = DFTBArgs(params['dftb_set'])

    # Is it periodic?
    if params['pbc']:
        # Only valid if there's a good cell!
        if np.linalg.det(a.get_cell()) == 0:
            # Cell parameters are dependent or zero!
            raise RuntimeError('Can not apply Periodic Boundary Conditions'
                               ' to system with non-independent lattice '
                               'parameters. Set pbc: false in the YAML file'
                               ' or fix the lattice parameters in the cell')
        # calc = Dftb(atoms=a, label='asephonons',
        #             kpts=params['kpoint_grid'],
        #             **dargs.args)
        calc = create_spinpol_dftbp_calculator(param_set=params['dftb_set'],
                                               kpts=params['kpoint_grid'])
        ph_kpts = params['phonon_kpoint_grid']
    else:
        a.set_pbc(False)
        # calc = Dftb(atoms=a, label='asephonons',
        #             **dargs.args)
        calc = create_spinpol_dftbp_calculator(param_set=params['dftb_set'])
        ph_kpts = None
    calc.label = 'asephonons'
    a.set_calculator(calc)
    phdata = ase_phonon_calc(a, kpoints=ph_kpts,
                             ftol=params['force_tol'],
                             force_clean=params['force_clean'],
                             name=params['name'])

    # Save optimised structure
    io.write(params['name'] + '_opt' + fext, phdata.structure)

    # And write out the phonons
    outf = params['name'] + '_opt.phonons.pkl'
    pickle.dump(phdata, open(outf, 'wb'))
    write_phonon_report(args, params, phdata)


if __name__ == "__main__":
    nq_entry()
