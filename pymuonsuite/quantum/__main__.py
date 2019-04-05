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
import argparse as ap

from pymuonsuite.quantum.vibrational.phonons import ase_phonon_calc
from pymuonsuite.quantum.vibrational.average import (muon_vibrational_average_write,
                                                     muon_vibrational_average_read)
from pymuonsuite.schemas import (load_input_file, MuonHarmonicSchema,
                                 AsePhononsSchema)


def nq_entry():
    parser = ap.ArgumentParser()
    parser.add_argument('parameter_file', type=str,
                        help="YAML file containing relevant input parameters")
    parser.add_argument('-w',   action='store_true', default=False,
                        help="Create and write input files instead of parsing the results")

    args = parser.parse_args()

    # Load parameters
    params = load_input_file(args.parameter_file, MuonHarmonicSchema)

    if args.w:
        muon_vibrational_average_write(**params)
    else:
        muon_vibrational_average_read(**params)


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

    # Load structure
    a = io.read(args.structure_file)
    # Create a Dftb calculator
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
                             ftol=params['force_tol'])

    # Save optimised structure
    fname, fext = os.path.splitext(args.structure_file)
    io.write(fname + '_opt' + fext, a)

    # And write out the phonons
    pickle.dump(phdata, open(params['output_file'], 'w'))

if __name__ == "__main__":
    nq_entry()
