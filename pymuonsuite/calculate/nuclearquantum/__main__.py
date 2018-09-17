# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse as ap

from pymuonsuite.calculate.nuclearquantum.phonon import phonon_hfcc

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument('calculation_type', type=str,
        help="""Type of calculation to be performed, currently supports:
                Quantum effects as phonons: phonon_hfcc""")
    parser.add_argument('parameter_file', type=str,
        help="YAML file containing relevant input parameters")

    args = parser.parse_args()

    if args.calculation_type == "phonon_hfcc":
        phonon_hfcc(args.parameter_file)
    else:
        raise RuntimeError("""Invalid calculation type entered, please use
                              python -h flag to see currently supported types""")
