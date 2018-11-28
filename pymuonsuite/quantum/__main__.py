"""
Author: Adam Laverack
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse as ap

from pymuonsuite.quantum.vibrational.phonons import phonon_hfcc
from pymuonsuite.schemas import load_input_file, PhononHfccSchema


def nq_entry():
    parser = ap.ArgumentParser()
    parser.add_argument('calculation_type', type=str,
                        help="""Type of calculation to be performed, currently supports:
                'phonon_hfcc': Nuclear quantum effects simulated by phonons""")
    parser.add_argument('parameter_file', type=str,
                        help="YAML file containing relevant input parameters")
    parser.add_argument('-w',   action='store_true', default=False,
                        help="Create and write input files instead of parsing the results")

    args = parser.parse_args()

    # Load parameters
    params = load_input_file(args.parameter_file, PhononHfccSchema)

    if args.calculation_type == "phonon_hfcc":
        phonon_hfcc(params['cell_file'], params['muon_symbol'], params['grid_n'],
                    params['calculator'], params['param_file'], params['ignore_ipsoH'],
                    params['numerical_solver'], args.w, params['ase_phonons'],
                    params['dftb_phonons'])
    else:
        raise RuntimeError("""Invalid calculation type entered, please use
                              python -h flag to see currently supported types""")


if __name__ == "__main__":
    nq_entry()
