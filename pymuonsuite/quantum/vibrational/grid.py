"""
Author: Simone Sturniolo and Adam Laverack
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import scipy.constants as cnst
from ase import Atoms

def calc_wavefunction(R, grid_n, write_table = True, filename = ''):
    """
    Calculate harmonic oscillator wavefunction

    | Args:
    |   R(Numpy float array, shape:(axes)): Displacement amplitude along each
    |       axis
    |   grid_n(int): Number of grid points along each axis
    |   write_table: Write out table of probability densities in format:
    |       Displacement | Prob. Density
    |   filename (str): Filename of file to write to, required for write_table
    |
    | Returns:
    |   prob_dens (Numpy float, shape:(grid_n*3)): Probability density of
    |       harmonic oscillator at each displacement
    """
    R_axes = np.array([np.linspace(-3*Ri, 3*Ri, grid_n)
                       for Ri in R])

    # Wavefunction
    psi_norm = (1.0/(np.prod(R)**2*np.pi**3))**0.25
    # And along the three axes
    psi = psi_norm*np.exp(-(R_axes/R[:, None])**2/2.0)
    # Save the densities
    if write_table:
        psi_table = np.concatenate(
            (R_axes, psi**2), axis=0)
        np.savetxt(filename, psi_table.T)
    # And average
    r2psi2 = R_axes**2*np.abs(psi)**2

    # Convert to portable output format
    prob_dens = np.zeros((grid_n*3))
    for i, axis in enumerate(r2psi2):
        for j, point in enumerate(axis):
            prob_dens[j + i*grid_n] = point

    return prob_dens

def weighted_tens_avg(tensors, weight):
    """
    Given a set of 3x3 tensors resulting from the sampling of a property on an
    N point grid for a set of atoms, calculate a weighted average of the tensors
    for each atom using a given weight for each grid point.

    | Args:
    |   tensors(Numpy float array, shape:(N,Atoms,3,3)): For each grid point,
    |       a set of 3x3 tensors for each atom.
    |   weight(Numpy float array, shape:(N)): A weighting for each point
    |       on the grid.
    |
    | Returns:
    |   tens_avg(Numpy float array, shape:(Atoms,3,3)): The averaged tensor for
    |       each atom.
    """
    num_atoms = np.size(tensors, 1)
    tens_avg = np.zeros((num_atoms, 3, 3))
    tensors = tensors*weight[:, None, None, None]
    for i in range(num_atoms):
        tens_avg[i] = np.sum(tensors[:, i], axis=0)/np.sum(weight)
    return tens_avg
