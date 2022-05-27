"""
Author: Simone Sturniolo and Adam Laverack
"""


import random

import numpy as np
from soprano.collection.generate import linspaceGen


# def calc_wavefunction(sigmas, grid_n, sigmaN=3):
#     """
#     Calculate harmonic oscillator wavefunction

#     | Args:
#     |   sigmas(Numpy float array, shape:(axes)): Displacement amplitude along
#     |                                            each axis
#     |   grid_n(int): Number of grid points along each axis
#     |   sigmaN(int): Number of standard deviations to cover in range
#     |                (default is 3)
#     | Returns:
#     |   prob_dens (Numpy float array, shape:(grid_n*3)): Probability density of
#     |       harmonic oscillator at each displacement
#     """
#     R_axes = np.array([np.linspace(-sigmas * Ri, sigmas * Ri, grid_n) for Ri in R])

#     # Wavefunction
#     psi_norm = (1.0 / (np.prod(R) ** 2 * np.pi ** 3)) ** 0.25
#     # And along the three axes
#     psi = psi_norm * np.exp(-((R_axes / R[:, None]) ** 2) / 2.0)
#     # And average
#     r2psi2 = R_axes ** 2 * np.abs(psi) ** 2

#     # Convert to portable output format
#     prob_dens = np.zeros((grid_n * 3))
#     for i, mode in enumerate(r2psi2):
#         for j, point in enumerate(mode):
#             prob_dens[j + i * grid_n] = point

#     return prob_dens


def create_displaced_cell(cell, displacements):
    """
    Take a set of atomic displacements and a cell and return an ASE Atoms object
    with atoms displaced appropriately.

    | Args:
    |   cell(ASE Atoms object): Cell containing original atomic positions.
    |   displacements(Numpy float array(num_atoms, 3)): Array containing a
    |       displacement vector for each atom in the system.
    |
    | Returns:
    |   disp_cell(ASE Atoms object): Cell containing displaced atomic positions.
    """
    disp_pos = cell.get_positions()
    disp_cell = cell.copy()
    disp_pos += displacements
    disp_cell.set_positions(disp_pos)
    disp_cell.calc = cell.calc

    return disp_cell


def displaced_cell_range(cell, a_i, grid_n, disp):
    """Return a generator of ASE Atoms objects with the displacement of the atom
    at index a_i varying between -disp and +disp with grid_n increments

    | Args:
    |   cell (ASE Atoms object): Object containing atom to be displaced
    |   a_i (int): Index of atom to be displaced
    |   grid_n (int): Number of increments/objects to create
    |   disp (float): Maximum displacement from original position
    |
    | Returns:
    |   lg(Soprano linspaceGen object): Generator of displaced cells
    """
    pos = cell.get_positions()
    cell_L = cell.copy()
    pos_L = pos.copy()
    pos_L[a_i] -= disp
    cell_L.set_positions(pos_L)
    cell_R = cell.copy()
    pos_R = pos.copy()
    pos_R[a_i] += disp
    cell_R.set_positions(pos_R)
    lg = linspaceGen(cell_L, cell_R, steps=grid_n, periodic=True)
    return lg


def tl_disp_generator(norm_coords, evecs, num_atoms):
    """
    Calculate a set of displacements of atoms in a system by generating a set of
    random thermal lines at T=0.

    | Args:
    |   norm_coords(Numpy float array(num_atoms, num_modes)): Array containing
    |       the normal mode coordinates of each real mode of the system.
    |   evecs(Numpy float array(size(norm_coords), num_atoms)): Array containing
    |       the eigenvectors of all real phonon modes for each atom in the
    |       system in the format evecs[modes][atoms].
    |   num_atoms(int): Number of atoms in the system.
    |
    | Returns:
    |   displacements(Numpy float array(num_atoms, 3)): Array containing the
    |       appropriate displacements of atoms for a randomly generated thermal
    |       line.
    """
    displacements = np.zeros((num_atoms, 3))
    coefficients = np.zeros(np.size(norm_coords, 1))
    for i in range(np.size(coefficients)):
        coefficients[i] = random.choice([-1, 1])
    norm_coords = norm_coords * coefficients

    for atom in range(num_atoms):
        for mode in range(np.size(norm_coords, 1)):
            displacements[atom] += (
                norm_coords[atom][mode] * evecs[mode][atom].real * 1e10
            )

    return displacements


def weighted_tens_avg(tensors, weight):
    """
    Given a set of tensors resulting from the sampling of a property over a
    set of different displacements, calculate a weighted average of the tensors
    for each atom or the whole system, using a given weight for each
    displacement.

    | Args:
    |   tensors(Numpy float array, shape:(N,No. of atoms,x,y)): For each grid
    |       point, a set of x by y tensors for each atom or the whole system.
    |       "No. of atoms" should be 1 in the latter case.
    |   weight(Numpy float array, shape:(N)): A weighting for each point
    |       on the grid.
    |
    | Returns:
    |   tens_avg(Numpy float array, shape:(Atoms,x,y)): The averaged tensor for
    |       each atom.
    """
    num_atoms = np.size(tensors, 1)
    x = np.size(tensors, 2)
    y = np.size(tensors, 3)
    tens_avg = np.zeros((num_atoms, x, y))
    tensors = tensors * weight[:, None, None, None]
    for i in range(num_atoms):
        tens_avg[i] = np.sum(tensors[:, i], axis=0) / np.sum(weight)
    return tens_avg


def wf_disp_generator(disp_factor, maj_evecs, grid_n):
    """
    Generate a set of displacements of an atom for the wavefunction sampling
    method.

    | Args:
    |   disp_factor(float(3)): A displacement factor for each of the 3 major
    |       phonon modes of the atom.
    |   maj_evecs(Numpy float array(3, 3)): The eigenvectors of the 3 major
    |       phonon modes of the atom.
    |   grid_n(int): The number of desired grid points along each mode.
    |
    | Returns:
    |   displacements(Numpy float array(grid_n*3, 3)): Array containing the
    |       amount the atom should be displaced by at each grid point. The first
    |       <grid_n> elements are for the first mode, the second <grid_n> for
    |       the second mode, etc.
    """
    displacements = np.zeros((grid_n * 3, 3))
    max_disp = 3 * maj_evecs * disp_factor[:, None]
    for mode in range(3):
        for n, t in enumerate(np.linspace(-1, 1, grid_n)):
            displacements[n + mode * grid_n] = t * max_disp[mode]

    return displacements
