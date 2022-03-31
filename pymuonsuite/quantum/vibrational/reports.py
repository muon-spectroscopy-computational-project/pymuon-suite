"""
Author: Adam Laverack and Simone Sturniolo
"""


import numpy as np
import scipy.constants as cnst


def harm_potential_report(R, grid_n, mass, freqs, E_table, filename):
    """
    Calculate the harmonic potential at all displacements on the grid for an
    atom and write out to file in a format that can be plotted.

    | Args:
    |   R(Numpy float array, shape:(axes)): Displacement amplitude along each
    |       axis
    |   grid_n(int): Number of grid points along each axis
    |   mass(float): Mass of atom
    |   freqs(Numpy float array, shape:(axes)): Frequencies of harmonic
    |       oscillator along each axis
    |   E_table(Numpy float array, shape:(np.size(R), grid_n)): Table of CASTEP
    |       final system energies.
    |   filename(str): Filename to be used for file
    |
    | Returns: Nothing
    """
    R_axes = np.array([np.linspace(-3 * Ri, 3 * Ri, grid_n) for Ri in R])
    # Now the potential, measured vs. theoretical
    harm_K = mass * freqs**2
    harm_V = (0.5 * harm_K[:, None] * (R_axes * 1e-10) ** 2) / cnst.electron_volt
    # Normalise E_table
    if E_table.shape[1] % 2 == 1:
        E_table -= (E_table[:, E_table.shape[1] // 2])[:, None]
    else:
        E_table -= (
            E_table[:, E_table.shape[1] // 2] + E_table[:, E_table.shape[1] // 2 - 1]
        )[:, None] / 2.0
    all_table = np.concatenate((R_axes, harm_V, E_table), axis=0)
    np.savetxt(filename, all_table.T)


def hfine_report(total_grid_n, tensors, hfine_tens_avg, weight, filename, atoms):
    """Write a report on a selection of atom's hyperfine coupling constants and
    their hyperfine tensor dipolar components based on a vibrational averaging
    calculation.

    | Args:
    |   total_grid_n(int): Total number of grid points
    |   tensors(Numpy float array, shape:(total_grid_n, num_atoms, 3, 3)):
    |       Array of hyperfine tensors for each atom at each grid point
    |   hfine_tens_avg(Numpy float array, shape:(num_atoms,3,3)): Average
    |       tensors of atoms over grid
    |   weight (Numpy float, shape:(total_grid_n)): Weighting of each grid
    |                                               point
    |   filename(str): Filename to be used for file
    |   atoms(dict, {index(int):symbol(str)}): Dictionary containing indices
    |                                          and symbols of atoms to write
    |                                          hyperfine coupling report about
    |
    | Returns:
    |   Nothing
    """
    ofile = open(filename, "w")
    for index in atoms:
        hfine_table = np.trace(tensors[:, index], axis1=1, axis2=2) / 3
        hfine_avg = np.sum(weight * hfine_table) / np.sum(weight)
        ofile.write(
            "Predicted hyperfine coupling on labeled atom ({1}): {0} MHz\n".format(
                hfine_avg, atoms[index]
            )
        )

        evals, evecs = np.linalg.eigh(hfine_tens_avg[index])
        evals, evecs = zip(*sorted(zip(evals, evecs), key=lambda x: abs(x[0])))
        evals_notr = -np.array(evals) + np.average(evals)

        if abs(evals_notr[2]) > abs(evals_notr[0]):
            D1 = evals_notr[2]
            D2 = evals_notr[1] - evals_notr[0]
        else:
            D1 = evals_notr[0]
            D2 = evals_notr[2] - evals_notr[1]

        ofile.write(
            (
                "Predicted dipolar hyperfine components on labeled atom ({2}):\n"
                "D1:\t{0} MHz\nD2:\t{1} MHz\n"
            ).format(D1, D2, atoms[index])
        )
