import os
import sys
import pickle
import numpy as np
import argparse as ap
import datetime

from ase import io
from ase import Atoms
from scipy.optimize import minimize
from scipy import constants as cnst


from pymuonsuite.schemas import UEPOptSchema, UEPPlotSchema, load_input_file
from pymuonsuite.calculate.uep.charged import ChargeDistribution

from parsefmt.fmtreader import FMTError

_header = """
*********************************
|   UU  UU   EEEEEE    PPPP     |
|   UU  UU   EE        PP  PP   |
|   UU  UU   EEEEEE    PPPP     |
|   UU  UU   EE        PP       |
|    UUUU    EEEEEE    PP       |
*********************************

Unperturbed Electrostatic Potential
optimiser for mu+ stopping site finding

by Simone Sturniolo (2018)

Calculations started on {0}

""".format(
    datetime.datetime.now()
)


def _make_chdistr(params):
    try:
        return ChargeDistribution(
            seedname=params["chden_seed"],
            path=params["chden_path"],
            gw_fac=params["gw_factor"],
        )
    except FileNotFoundError as e:
        raise FileNotFoundError("ERROR: {}.".format(e))
    except FMTError as e:
        raise FMTError(e)
    except (io.formats.UnknownFileTypeError, ValueError, TypeError) as e:
        raise IOError(
            "ERROR due to invalid file {}: {} ".format(
                params["chden_seed"] + ".castep", e
            )
        )


def geomopt(params, outf=None):

    t0 = datetime.datetime.now()

    chdistr = _make_chdistr(params)

    a = chdistr.atoms
    m = params["particle_mass"]

    if outf is not None:
        outf.write(_header)
        outf.write(
            "Charge distribution loaded from {0}\n".format(
                os.path.join(params["chden_path"], params["chden_seed"])
            )
        )
        outf.write("Gaussian width factor used: {0}\n".format(params["gw_factor"]))
        outf.write("Particle mass: {0} kg\n".format(m))

    # Objective functions
    def f(x):
        return chdistr.V([x])[0][0]

    def fprime(x):
        return chdistr.dV([x])[0][0] * 1e-10

    def fhess(x):
        return chdistr.d2V([x])[0][0] * 1e-20

    if outf is not None:
        outf.write("\n---------\n\n")
        outf.write(
            "Performing optimisation with method {0}\n".format(params["opt_method"])
        )
        outf.write(
            "Tolerance required for convergence: {0} eV\n".format(params["opt_tol"])
        )
        outf.write("Maximum number of steps: {0}\n".format(params["geom_steps"]))
        outf.write(
            "Defect starting position: {0} {1} {2} Ang\n".format(*params["mu_pos"])
        )

    sol = minimize(
        f,
        params["mu_pos"],
        jac=fprime,
        hess=fhess,
        tol=params["opt_tol"],
        method=params["opt_method"],
        options={"maxiter": params["geom_steps"]},
    )

    xsol = sol.x
    fxsol = np.linalg.solve(a.get_cell(complete=True), xsol)
    Eclass = sol.fun
    # Zero point energy
    hess = fhess(sol.x) * 1e20 * cnst.e
    evals, _ = np.linalg.eigh(hess)
    if (evals < 0).any():
        # The minimum is not stable!
        Ezp = np.nan
    else:
        Ezp = np.sum(0.5 * cnst.hbar * (evals / m) ** 0.5) / cnst.e
    Etot = Eclass + Ezp

    if outf is not None:
        outf.write("\n---------\n\n")
        outf.write("Optimisation stopped after {0} steps\n".format(sol.nit))
        if not sol.success:
            outf.write("Optimisation failed:\n{0}\n\n".format(sol.message))

        outf.write("\n\n")
        outf.write("Final coordinates: {0} {1} {2} Ang\n".format(*xsol))
        outf.write("Final fractional coordinates: {0} {1} {2}\n".format(*fxsol))
        outf.write("Classical energy: {0} eV\n".format(Eclass))

        if np.isnan(Ezp):
            outf.write(
                "Extremum is saddle point, impossible to compute " "zero-point energy"
            )
        else:
            outf.write("Zero-point energy: {0} eV\n".format(Ezp))
            outf.write("Quantum total energy: {0} eV\n".format(Etot))

        dt = datetime.datetime.now() - t0

        outf.write("\n\n")
        outf.write(
            "Calculation time: {0} s".format(dt.seconds + dt.microseconds * 1e-6)
        )

    # Finally, return the solution's values
    results = {
        "x": xsol,
        "fx": fxsol,
        "Eclass": Eclass,
        "Ezp": Ezp,
        "Etot": Etot,
        "sol": sol,  # For any additional details about the calculation
        "struct": a,  # For reference, the structure
    }

    return results


def _interpret_line(ldef, cell, pos):
    # Interpret a line definition for plotting
    # The final format has to be:
    # [[starting point], [end point], number of points]

    # Get a signature
    sig = "".join(["l" if type(ld) is list else "n" for ld in ldef])
    if sig == "lln":
        p0 = np.array(ldef[0])
        p1 = np.array(ldef[1])
        n = int(ldef[2])
    elif sig == "llnn":
        p0 = np.array(ldef[1])
        v = np.dot(ldef[0], cell)
        v /= np.linalg.norm(v)
        p1 = p0 + ldef[2] * v
        n = int(ldef[3])
    elif sig == "nnn":
        p0 = np.array(pos[ldef[0]])
        p1 = np.array(pos[ldef[1]])
        n = int(ldef[2])
    else:
        raise ValueError("Invalid line definition in input file")

    return [p0, p1, n]


def _interpret_plane(pdef, cell, pos):
    # Interpret a plane definition for plotting
    # The final format has to be:
    # [[corner 1], [corner 2], [corner 3],
    #  points along width, points along height]

    # Get a signature
    sig = "".join(["l" if type(pd) is list else "n" for pd in pdef])

    if sig == "lllnn":
        p0 = np.array(pdef[0])
        p1 = np.array(pdef[1])
        p2 = np.array(pdef[2])
        nw = int(pdef[3])
        nh = int(pdef[4])
    elif sig == "nnnnn":
        p0 = pos[pdef[0]]
        p1 = pos[pdef[1]]
        p2 = pos[pdef[2]]
        nw = int(pdef[3])
        nh = int(pdef[4])
    else:
        raise ValueError("Invalid plane definition in input file")

    return [p0, p1, p2, nw, nh]


def plot(params, prefix="uepplot"):

    chdistr = _make_chdistr(params)

    a = chdistr.atoms
    cell = a.get_cell()
    pos = a.get_positions()

    # Plot lines
    for i, ldef in enumerate(params["line_plots"]):
        ldef = _interpret_line(ldef, cell, pos)
        # Make a range
        dx = ldef[1] - ldef[0]
        lrange = np.linspace(0, 1.0, ldef[2])
        lrange = dx * lrange[:, None] + ldef[0]
        V, Ve, Vi = chdistr.V(lrange)
        rho, rhoe, rhoi = chdistr.rho(lrange)
        r = np.linalg.norm(lrange - lrange[0], axis=1)

        outf = open("{0}.line.{1}.dat".format(prefix, i + 1), "w")
        for j, x in enumerate(r):
            outf.write(
                "\t".join(map(str, [x, V[j], Ve[j], Vi[j], rho[j], rhoe[j], rhoi[j]]))
                + "\n"
            )
        outf.close()

    for i, pdef in enumerate(params["plane_plots"]):
        pdef = _interpret_plane(pdef, cell, pos)
        dx = pdef[1] - pdef[0]
        dy = pdef[2] - pdef[0]
        lxrange = np.linspace(0, 1.0, pdef[3])
        lxrange = dx * lxrange[:, None] + pdef[0]
        lyrange = np.linspace(0, 1.0, pdef[4])
        lyrange = dy * lyrange[:, None]
        lxyrange = lxrange[:, None, :] + lyrange[None, :, :]

        dynorm = dy - np.dot(dy, dx) * dx / np.linalg.norm(dx) ** 2
        outf = open("{0}.plane.{1}.dat".format(prefix, i + 1), "w")

        for j, lrange in enumerate(lxyrange):
            rx = np.dot(lrange, dx) / np.linalg.norm(dx)
            ry = np.dot(lrange, dynorm) / np.linalg.norm(dynorm)
            V, Ve, Vi = chdistr.V(lrange)
            rho, rhoe, rhoi = chdistr.rho(lrange)
            for k, (x, y) in enumerate(zip(rx, ry)):
                outf.write(
                    "\t".join(
                        map(
                            str,
                            [
                                x,
                                y,
                                V[k],
                                Ve[k],
                                Vi[k],
                                rho[k],
                                rhoe[k],
                                rhoi[k],
                            ],
                        )
                    )
                    + "\n"
                )
            outf.write("\n")

        outf.close()


def geomopt_entry():
    parser = ap.ArgumentParser()
    parser.add_argument("input", type=str, help="Input YAML file for the calculation")
    args = parser.parse_args()

    params = load_input_file(args.input, UEPOptSchema)

    # Some output to give feedback to the user
    print("Performing UEP optimisation for {0}".format(args.input))

    seedpath = os.path.splitext(args.input)[0]
    seedname = os.path.split(seedpath)[1]

    if params["chden_seed"] is None:
        params["chden_seed"] = seedname  # Default is the same

    with open(seedpath + ".uep", "w") as outf:
        try:
            results = geomopt(params, outf)
        except Exception as e:
            print("Error: ", e)
            sys.exit(1)

    print("Optimisation complete")

    # Now dump results
    if params["save_pickle"]:
        pickle.dump(results, open(seedpath + ".uep.pkl", "wb"))
        muon = Atoms("H", positions=[results["x"]])
    if params["save_structs"]:
        io.write(seedpath + ".xyz", results["struct"] + muon)


def plot_entry():
    parser = ap.ArgumentParser()
    parser.add_argument("input", type=str, help="Input YAML file for the calculation")
    args = parser.parse_args()

    params = load_input_file(args.input, UEPPlotSchema)

    seedpath = os.path.splitext(args.input)[0]
    seedname = os.path.split(seedpath)[1]

    if params["chden_seed"] is None:
        params["chden_seed"] = seedname  # Default is the same

    try:
        plot(params, seedname)
    except Exception as e:
        print("Error: ", e)
        sys.exit(1)


if __name__ == "__main__":
    geomopt_entry()
