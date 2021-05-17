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
    parser.add_argument('-sp', '--symprec', type=float, default=1e-3,
                        help="Symmetry precision to use in spglib")
    args = parser.parse_args()

    with silence_stdio():
        a = io.read(args.structure)

    symdata = SymmetryDataset.get(a, symprec=args.symprec)
    wpoints = WyckoffPoints.get(a, symprec=args.symprec)
    fpos = a.get_scaled_positions()

    print("Wyckoff points symmetry report for {0}".format(args.structure))
    print("Space Group International Symbol: "
          "{0}".format(symdata['international']))
    print("Space Group Hall Number: "
          "{0}".format(symdata['hall_number']))
    print("Absolute\t\t\tFractional\t\tHessian constraints")

    # List any Wyckoff point that does not already have an atom in it
    vformat = '[{0:.3f} {1:.3f} {2:.3f}]'
    for wp in wpoints:
        if np.any(np.isclose(np.linalg.norm(fpos-wp.fpos, axis=1), 0,
                             atol=args.symprec)):
            continue
        ps = vformat.format(*wp.pos)
        fps = vformat.format(*wp.fpos)
        print("{0}\t{1}\t{2}".format(ps, fps, wp.hessian))
