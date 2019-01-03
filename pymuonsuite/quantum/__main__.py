"""
Author: Adam Laverack
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse as ap

from pymuonsuite.quantum.vibrational.programs import vib_avg
from pymuonsuite.schemas import load_input_file, MuonHarmonicSchema


def nq_entry():
    parser = ap.ArgumentParser()
    parser.add_argument('calculation_type', type=str,
                        help="""Type of calculation to be performed, currently supports:
                'vib_avg': Nuclear quantum effects of atoms simulated
                by treating atoms as a particles in a quantum harmonic oscillator""")
    parser.add_argument('parameter_file', type=str,
                        help="YAML file containing relevant input parameters")
    parser.add_argument('-w',   action='store_true', default=False,
                        help="Create and write input files instead of parsing the results")

    args = parser.parse_args()

    # Load parameters
    params = load_input_file(args.parameter_file, MuonHarmonicSchema)

    # Check that input is valid
    if params['property'].strip().lower() != 'hyperfine':
        raise ValueError("Invalid value entered for property ('{0}')".format(
        params['property']))

    if params['value_type'].strip().lower() != 'scalar' and \
       params['value_type'].strip().lower() != 'vector' and \
       params['value_type'].strip().lower() != 'matrix':
        raise ValueError("Invalid value entered for value_type ('{0}')".format(
        params['value_type']))

    if params['weight'].strip().lower() != 'harmonic':
        raise ValueError("Invalid value entered for weight ('{0}')".format(
        params['weight']))

    if args.calculation_type == "vib_avg":
        vib_avg(params['cell_file'], params['muon_symbol'], params['grid_n'],
                    params['atom_indices'], params['property'], params['value_type'],
                    params['weight'], params['param_file'], params['ignore_ipsoH'],
                    params['numerical_solver'], args.w, params['ase_phonons'],
                    params['dftb_phonons'])
    else:
        raise RuntimeError("""Invalid calculation type entered, please use
                              python -h flag to see currently supported types""")


if __name__ == "__main__":
    nq_entry()
