"""symmetry.py

Produce a report on a given structure file detailing the high symmetry
points that could constitute muon stopping sites
"""

from ase import io
import argparse as ap
from soprano.utils import silence_stdio
from soprano.properties.symmetry import SymmetryDataset, WyckoffPoints
from pymuonsuite.io.output import write_symmetry_report


def main():

    parser = ap.ArgumentParser()
    parser.add_argument(
        "structure",
        type=str,
        default=None,
        help="A structure file in an ASE readable format",
    )
    parser.add_argument(
        "-sp",
        "--symprec",
        type=float,
        default=1e-3,
        help="Symmetry precision to use in spglib",
    )
    args = parser.parse_args()

    with silence_stdio():
        a = io.read(args.structure)

    symdata = SymmetryDataset.get(a, symprec=args.symprec)
    wpoints = WyckoffPoints.get(a, symprec=args.symprec)
    fpos = a.get_scaled_positions()

    write_symmetry_report(args, symdata, wpoints, fpos)
