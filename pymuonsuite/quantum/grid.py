"""
Author: Simone Sturniolo(Functionality) and Adam Laverack(Interface)
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import scipy.constants as cnst
from ase import Atoms


def calc_wavefunction(R, grid_n, num_solve = False, atom_mass = None,
                      E_table = None, write_table = True, value_table = None,
                      sname = None):
    """
    Calculate harmonic oscillator wavefunction

    | Args:
    |   R (Numpy float array): Displacement amplitudes along phonon axes
    |   grid_n (int): Number of displacements per axis
    |   num_solve (bool): Solve schroedinger equation numerically using qlab
    |   atom_mass (float): Mass of atom, required for num_solve
    |   E_table (Numpy float, shape:(size(R), grid_n)): Array of final
    |       system energies at each displacement on the grid, required for
    |       num_solve and write_table
    |   write_table: Write out table of wavefunction values in format:
    |       Displacement | Energy | Prob. Density | Coup. const.
    |   value_table (Numpy float, shape:(size(R), grid_n)): Array of
    |       coupling constants for atom at each displacement on the grid,
    |       required for write_table
    |   sname (str): Seedname of file to write to, required for write_table
    |
    | Returns:
    |   r2psi2 (Numpy float, shape:(size(R), grid_n)): Probability density of
    |       harmonic oscillator at each displacement
    """
    R_axes = np.array([np.linspace(-3*Ri, 3*Ri, grid_n)
                       for Ri in R])

    if not num_solve:
        # Wavefunction
        psi_norm = (1.0/(np.prod(R)**2*np.pi**3))**0.25
        # And along the three axes
        psi = psi_norm*np.exp(-(R_axes/R[:, None])**2/2.0)
    else:
        # Import qlab
        try:
            from qlab.solve import QSolution
        except ImportError:
            raise RuntimeError('QLab not present on this system, '
                               '-num option is invalid')
        if write_table:
            sname += '_num'
        psi = []
        for i, Ri in enumerate(R):
            qSol = QSolution([(-3e-10*Ri, 3e-10*Ri)], grid_n,
                             E_table[i]*cnst.electron_volt, atom_mass)
            psi.append(qSol.evec_grid(0))
        psi = np.array(psi)
    # Oh, and save the densities!
    if write_table:
        psi_table = np.concatenate(
            (R_axes, E_table, psi**2, value_table), axis=0)
        np.savetxt(sname + '_psi.dat', psi_table.T)
    # And average
    r2psi2 = R_axes**2*np.abs(psi)**2

    return r2psi2

def weighted_tens_avg(tensors, weight):
    """
    Given a set of 3x3 tensors resulting from the sampling of a property on an
    NxM grid for a set of atoms, calculate a weighted average of the tensors for
    each atom using a given weight for each grid point.

    | Args:
    |   tensors(Numpy float array, shape:(Atoms,N,M,3,3)): For each atom, an NxM
    |       set of shape 3x3 tensors.
    |   weight(Numpy float array, shape:(N,M)): A weighting for each point
    |       on the NxM grid.
    |
    | Returns:
    |   tens_avg(Numpy float array, shape:(Atoms,3,3)): The averaged tensor for
    |       each atom.
    """
    tens_avg = np.zeros((np.size(tensors, 0), 3, 3))
    for i in range(np.size(tensors, 0)):
        tens_avg[i] = np.sum(
            weight[:, :, None, None]*tensors[i], axis=(0, 1))/np.sum(weight)
    return tens_avg
