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
from pymuonsuite.quantum.vibrational.average import muon_vibrational_average_write
from pymuonsuite.schemas import load_input_file, MuonHarmonicSchema


def nq_entry():
    parser = ap.ArgumentParser()
    parser.add_argument('parameter_file', type=str,
                        help="YAML file containing relevant input parameters")
    parser.add_argument('-w',   action='store_true', default=False,
                        help="Create and write input files instead of parsing the results")
    # We comment this bit out until we have other calculation types
    #
    # parser.add_argument('calculation_type', type=str,
    #                     help="""Type of calculation to be performed, currently supports:
    #             'vib_avg': Nuclear quantum effects of atoms simulated
    #             by treating atoms as a particles in a quantum harmonic oscillator""")

    args = parser.parse_args()

    # Load parameters
    params = load_input_file(args.parameter_file, MuonHarmonicSchema)

    if args.w:
        muon_vibrational_average_write(**params)
    else:
        pass

    # Call functions
    # vib_avg(params['cell_file'], params['method'], params['muon_symbol'],
    #         params['grid_n'], params['property'], params['selection'],
    #         params['weight'], params['param_file'],
    #         args.w, params['ase_phonons'])


if __name__ == "__main__":
    nq_entry()
