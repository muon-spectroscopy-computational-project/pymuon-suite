"""
Author: Adam Laverack
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse as ap
try:
    from casteppy.data.phonon import PhononData
except ImportError:
    raise ImportError("""
Can't use castep phonon interface due to casteppy not being installed.
Please download and install casteppy from Bitbucket:

HTTPS:  https://bitbucket.org/casteppy/casteppy.git
SSH:    git@bitbucket.org:casteppy/casteppy.git

and try again.""")

from pymuonsuite.quantum.vibrational.programs import vib_avg
from pymuonsuite.schemas import load_input_file, MuonHarmonicSchema


def read_castep_phonons(phonon_file):
    # Parse CASTEP phonon data into casteppy object
    pd = PhononData(sname)
    # Convert frequencies back to cm-1
    pd.convert_e_units('1/cm')
    # Get phonon frequencies+modes
    evals = np.array(pd.freqs)
    evecs = np.array(pd.eigenvecs)

    return evals, evecs


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

    # Call functions
    if args.calculation_type == "vib_avg":
        vib_avg(params['cell_file'], params['method'], params['muon_symbol'],
                params['grid_n'], params['property'], params['selection'],
                params['weight'], params['param_file'],
                args.w, params['ase_phonons'])

    else:
        raise RuntimeError("""Invalid calculation type entered, please use
                              python -h flag to see currently supported types""")


if __name__ == "__main__":
    nq_entry()
