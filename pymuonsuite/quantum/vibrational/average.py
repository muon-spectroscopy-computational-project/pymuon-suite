"""
average.py

Quantum vibrational averages
"""


import os
import numpy as np

from ase import io
from soprano.utils import seedname
from soprano.collection import AtomsCollection
from soprano.utils import silence_stdio

# Internal imports
from pymuonsuite import constants
from pymuonsuite.io.castep import parse_castep_masses, ReadWriteCastep
from pymuonsuite.io.dftb import ReadWriteDFTB
from pymuonsuite.quantum.vibrational.schemes import (
    IndependentDisplacements,
    MonteCarloDisplacements,
)


class MuonAverageError(Exception):
    pass


def muon_vibrational_average_write(
    structure,
    method="independent",
    mu_index=-1,
    mu_symbol="H:mu",
    grid_n=20,
    sigma_n=3,
    avgprop="hyperfine",
    calculator="castep",
    displace_T=0,
    phonon_source_file=None,
    phonon_source_type="castep",
    **kwargs
):
    """
    Write input files to compute a vibrational average for a quantity on a muon
    in a given system.

    | Pars:
    |   structure (str):    Filename for input structure file
    |   method (str):       Method to use for the average. Options are
    |                       'independent', 'montecarlo'.
    |                       Default is 'independent'.
    |   mu_index (int):     Position of the muon in the given cell file.
    |                       Default is -1.
    |   mu_symbol (str):    Use this symbol to look for the muon among
    |                       CASTEP custom species. Overrides muon_index if
    |                       present in cell.
    |   grid_n (int):       Number of configurations used for sampling.
    |                       Applies slightly
    |                       differently to different schemes.
    |   sigma_n (int):      Number of sigmas of the harmonic wavefunction used
    |                       for sampling.
    |   avgprop (str):      Property to calculate and average. Default is
    |                       'hyperfine'.
    |   calculator (str):   Source of the property to calculate and average.
    |                       Can be 'castep' or 'dftb+'. Default is 'castep'.
    |   phonon_source (str):Source of the phonon data. Can be 'castep' or
    |                       'asedftbp'. Default is 'castep'.
    |   **kwargs:           Other arguments (such as specific arguments for
    |                       the given phonon method)
    """

    # Open the structure file
    with silence_stdio():
        cell = io.read(structure)
    path = os.path.split(structure)[0]
    sname = seedname(structure)

    cell.info["name"] = sname

    # Fetch species
    try:
        species = cell.get_array("castep_custom_species")
    except KeyError:
        species = np.array(cell.get_chemical_symbols())

    mu_indices = np.where(species == mu_symbol)[0]
    if len(mu_indices) > 1:
        raise MuonAverageError("More than one muon found in the system")
    elif len(mu_indices) == 1:
        mu_index = mu_indices[0]
    else:
        species = list(species)
        species[mu_index] = mu_symbol
        species = np.array(species)

    cell.set_array("castep_custom_species", species)

    io_formats = {"castep": ReadWriteCastep, "dftb+": ReadWriteDFTB}

    # Load the phonons
    if phonon_source_file is not None:
        phpath, phfile = os.path.split(phonon_source_file)
        phfile = seedname(seedname(phfile))  # have to do twice for dftb case
    else:
        phpath = path
        phfile = sname

    try:
        rw = io_formats[phonon_source_type]()
        atoms = rw.read(phpath, phfile, read_phonons=True)
        ph_evals = atoms.info["ph_evals"]
        ph_evecs = atoms.info["ph_evecs"]
    except IOError:
        raise
        return
    except KeyError:
        phonon_source_file = os.path.join(phpath, phfile + ".phonon")
        if phonon_source_type == "dftb+":
            phonon_source_file = phonon_source_file + "s.pkl"
        raise (IOError("Phonon file {0} could not be read.".format(phonon_source_file)))
        return

    # Fetch masses
    try:
        masses = parse_castep_masses(cell)
    except AttributeError:
        # Just fall back on ASE standard masses if not available
        masses = cell.get_masses()
    masses[mu_index] = constants.m_mu_amu
    cell.set_masses(masses)

    # Now create the distribution scheme
    if method == "independent":
        displsch = IndependentDisplacements(
            ph_evals, ph_evecs, masses, mu_index, sigma_n
        )
    elif method == "montecarlo":
        # Set seed
        np.random.seed(kwargs["random_seed"])
        displsch = MonteCarloDisplacements(ph_evals, ph_evecs, masses)

    displsch.recalc_displacements(n=grid_n, T=displace_T)

    # Make it a collection
    pos = cell.get_positions()
    displaced_cells = []
    for i, d in enumerate(displsch.displacements):
        dcell = cell.copy()
        dcell.set_positions(pos + d)
        if calculator == "dftb" and not kwargs["dftb_pbc"]:
            dcell.set_pbc(False)
        dcell.info["name"] = sname + "_displaced_{0}".format(i)
        displaced_cells.append(dcell)

    if kwargs["write_allconf"]:
        # Write a global configuration structure
        allconf = sum(displaced_cells, cell.copy())
        with silence_stdio():
            if all(allconf.get_pbc()):
                io.write(sname + "_allconf.cell", allconf)
            else:
                io.write(sname + "_allconf.xyz", allconf)

    # Get a calculator
    if calculator == "castep":
        params = {
            "castep_param": kwargs["castep_param"],
            "k_points_grid": kwargs["k_points_grid"],
            "mu_symbol": mu_symbol,
        }
        io_format = ReadWriteCastep(
            params=params, calc=cell.calc, script=kwargs["script_file"]
        )
        opt_args = {"calc_type": "MAGRES"}

    elif calculator == "dftb+":
        params = {
            "dftb_set": kwargs["dftb_set"],
            "dftb_pbc": kwargs["dftb_pbc"],
            "k_points_grid": kwargs["k_points_grid"] if kwargs["dftb_pbc"] else None,
        }
        io_format = ReadWriteDFTB(
            params=params, calc=cell.calc, script=kwargs["script_file"]
        )
        opt_args = {"calc_type": "SPINPOL"}

    displaced_coll = AtomsCollection(displaced_cells)
    displaced_coll.info["displacement_scheme"] = displsch
    displaced_coll.info["muon_index"] = mu_index
    displaced_coll.save_tree(sname + "_displaced", io_format.write, opt_args=opt_args)


def muon_vibrational_average_read(
    structure,
    calculator="castep",
    avgprop="hyperfine",
    average_T=0,
    average_file="averages.dat",
    **kwargs
):
    # Open the structure file
    sname = seedname(structure)

    io_formats = {"castep": ReadWriteCastep, "dftb+": ReadWriteDFTB}

    try:
        displaced_coll = AtomsCollection.load_tree(
            sname + "_displaced",
            io_formats[calculator]().read,
            opt_args={"read_magres": True},
            safety_check=2,
        )
    except Exception:
        raise
        return

    mu_i = displaced_coll.info["muon_index"]
    displsch = displaced_coll.info["displacement_scheme"]

    to_avg = []

    for a in displaced_coll:
        if avgprop == "hyperfine":
            to_avg.append(a.get_array("hyperfine")[mu_i])
        elif avgprop == "charge":
            # Used mostly as test
            try:
                to_avg.append(a.get_charges()[mu_i])
            except RuntimeError:
                raise (IOError("Could not read charges."))

    to_avg = np.array(to_avg)
    displsch.recalc_weights(T=average_T)
    # New shape
    N = len(displaced_coll)
    shape = tuple([slice(N)] + [None] * (len(to_avg.shape) - 1))
    weights = displsch.weights[shape]
    avg = np.sum(weights * to_avg, axis=0)

    # Print output report
    with open(average_file, "w") as f:
        avgname = {"hyperfine": "hyperfine tensor", "charge": "charge"}[avgprop]
        f.write(
            """
Quantum average of {property} calculated on {cell}.
Scheme details:

{scheme}

Averaged value:

{avg}

All values, by configuration:

{vals}

        """.format(
                property=avgname,
                cell=structure,
                scheme=displsch,
                avg=avg,
                vals="\n".join(
                    [
                        "Conf: {0} (Weight = {1})\n{2}\n".format(
                            i, displsch.weights[i], v
                        )
                        for i, v in enumerate(to_avg)
                    ]
                ),
            )
        )
