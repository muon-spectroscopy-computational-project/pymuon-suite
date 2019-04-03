"""
Author: Adam Laverack
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse as ap

from pymuonsuite.quantum.vibrational.average import (muon_vibrational_average_write,
                                                     muon_vibrational_average_read)
from pymuonsuite.schemas import load_input_file, MuonHarmonicSchema


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


if __name__ == "__main__":
    nq_entry()
