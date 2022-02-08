"""
Author: Adam Laverack
"""


import os
import sys
import pickle
import argparse as ap

from pymuonsuite.quantum.vibrational.phonons import ase_phonon_calc
from pymuonsuite.quantum.vibrational.average import (
    muon_vibrational_average_write,
    muon_vibrational_average_read,
)
from pymuonsuite.schemas import (
    load_input_file,
    MuonHarmonicSchema,
    AsePhononsSchema,
)
from pymuonsuite.io.output import write_phonon_report
from soprano.utils import silence_stdio


def nq_entry():
    parser = ap.ArgumentParser()
    parser.add_argument(
        "structure",
        type=str,
        default=None,
        help="A structure file in an ASE readable format",
    )
    parser.add_argument(
        "parameter_file",
        type=str,
        help="YAML file containing relevant input parameters",
    )
    parser.add_argument(
        "-t",
        type=str,
        default="r",
        choices=["r", "w"],
        dest="task",
        help="""Task to be run by pm-nq. Can be either 'w'
                        (=generate and WRITE structures) or 'r' (=READ and
                        analyse results). Default is READ.""",
    )

    args = parser.parse_args()

    # Load parameters
    params = load_input_file(args.parameter_file, MuonHarmonicSchema)

    # Temperature
    if params["average_T"] is None:
        params["average_T"] = params["displace_T"]

    if args.task == "w":
        try:
            muon_vibrational_average_write(args.structure, **params)
        except IOError as e:
            print(e)
    else:
        try:
            muon_vibrational_average_read(args.structure, **params)
        except IOError as e:
            print("Read/write error: {0}".format(e))
            print(
                "\nThis could mean it was impossible to find the displaced "
                "structure folder: maybe you wanted to run with the "
                "-t w option to write it?\n"
            )


def asephonons_entry():

    from ase import io
    from ase.calculators.dftb import Dftb
    from pymuonsuite.data.dftb_pars import DFTBArgs

    parser = ap.ArgumentParser(
        description="Compute phonon modes with ASE and"
        " DFTB+ for reuse in quantum effects "
        "calculations."
    )
    parser.add_argument(
        "structure_file",
        type=str,
        help="Structure for which to compute the phonons",
    )
    parser.add_argument(
        "parameter_file",
        type=str,
        help="YAML file containing relevant input parameters",
    )

    args = parser.parse_args()

    # Load parameters
    params = load_input_file(args.parameter_file, AsePhononsSchema)

    fname, fext = os.path.splitext(args.structure_file)
    if params["name"] is None:
        params["name"] = fname

    # Load structure
    with silence_stdio():
        a = io.read(args.structure_file)

    # Create a Dftb calculator
    dargs = DFTBArgs(params["dftb_set"])
    # Is it periodic?
    if params["pbc"]:
        a.set_pbc(True)
        calc = Dftb(
            atoms=a, label="asephonons", kpts=params["kpoint_grid"], **dargs.args
        )
        ph_kpts = params["phonon_kpoint_grid"]
    else:
        a.set_pbc(False)
        calc = Dftb(atoms=a, label="asephonons", **dargs.args)
        ph_kpts = None
    a.calc = calc
    try:
        phdata = ase_phonon_calc(
            a,
            kpoints=ph_kpts,
            ftol=params["force_tol"],
            force_clean=params["force_clean"],
            name=params["name"],
        )
    except Exception as e:
        print(e)
        print("Error: Could not write phonons file, see asephonons.out for" " details.")
        sys.exit(1)

    # Save optimised structure
    with silence_stdio():
        io.write(params["name"] + "_opt" + fext, phdata.structure)

    # And write out the phonons
    outf = params["name"] + "_opt.phonons.pkl"
    pickle.dump(phdata, open(outf, "wb"))
    write_phonon_report(args, params, phdata)


if __name__ == "__main__":
    nq_entry()
