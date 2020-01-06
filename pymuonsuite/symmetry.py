"""symmetry.py

Produce a report on a given structure file detailing the high symmetry
points that could constitute muon stopping sites
"""

import numpy as np
from ase import io
import argparse as ap
from soprano.utils import silence_stdio
from soprano.properties.symmetry import (SymmetryDataset, WyckoffPoints)


def print_symmetry_report():

    parser = ap.ArgumentParser()
    parser.add_argument('structure', type=str, default=None,
                        help="A structure file in an ASE readable format")
    args = parser.parse_args()

    with silence_stdio():
        a = io.read(args.structure)

    symdata = SymmetryDataset.get(a)
    wpoints = WyckoffPoints.get(a)
    fpos = a.get_scaled_positions()

    print("Wyckoff points symmetry report for {0}".format(args.structure))
    print("Space Group International Symbol: "
          "{0}".format(symdata['international']))
    print("Space Group Hall Number: "
          "{0}".format(symdata['hall_number']))
    print("Absolute\t\t\tFractional\t\tHessian constraints")

    # List any Wyckoff point that does not already have an atom in it
    for wp in wpoints:
        if np.any(np.isclose(np.linalg.norm(fpos-wp.fpos, axis=1), 0)):
            continue
        print("{0}\t{1}\t{2}".format(wp.pos, wp.fpos, wp.hessian))
