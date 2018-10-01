"""
Author: Adam Laverack
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse as ap

from pymuonsuite.calculate.nuclearquantum.phonon import phonon_hfcc
from pymuonsuite.schemas import load_input_file, PhononHfccSchema

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument('calculation_type', type=str,
        help="""Type of calculation to be performed, currently supports:
                'phonon_hfcc': Nuclear quantum effects simulated by phonons""")
    parser.add_argument('parameter_file', type=str,
        help="YAML file containing relevant input parameters")

    args = parser.parse_args()

    #Load parameters
    params = load_input_file(args.parameter_file, PhononHfccSchema)

    if args.calculation_type == "phonon_hfcc":
        phonon_hfcc(params)
    else:
        raise RuntimeError("""Invalid calculation type entered, please use
                              python -h flag to see currently supported types""")
