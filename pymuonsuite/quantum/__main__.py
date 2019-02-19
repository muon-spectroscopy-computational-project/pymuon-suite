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
    if params['method'] != 'wavefunction' and \
       params['method'] != 'thermal':
        raise ValueError("""Invalid value entered for method ('{0}'). Remember
        that this is case sensitive.""".format(params['method']))

    if params['property'] != 'hyperfine' and \
       params['property'] != 'bandstructure':
        raise ValueError("""Invalid value entered for weight ('{0}'). Remember
        that this is case sensitive.""".format(params['property']))

    if params['weight'] != 'harmonic':
        raise ValueError("""Invalid value entered for weight ('{0}'). Remember
        that this is case sensitive.""".format(params['weight']))

    # Call functions
    if args.calculation_type == "vib_avg":
        vib_avg(params['cell_file'], params['method'], params['muon_symbol'],
                    params['grid_n'], params['property'],
                    params['atom_indices'], params['weight'], params['param_file'],
                    args.w, params['ase_phonons'],
                    params['dftb_phonons'])

    else:
        raise RuntimeError("""Invalid calculation type entered, please use
                              python -h flag to see currently supported types""")


if __name__ == "__main__":
    nq_entry()
