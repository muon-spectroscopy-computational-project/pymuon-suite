"""
Author: Adam Laverack and Simone Sturniolo
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

def hfine_report(R, total_grid_n, tensors, hfine_tens_avg, r2psi2, filename, atoms):
    """Write a report on a selection of atom's hyperfine coupling constants and
    their hyperfine tensor dipolar components based on a vibrational averaging
    calculation.

    | Args:
    |   R(Numpy float array, shape:(axes)): Displacement amplitude along each
    |       phonon axis
    |   total_grid_n(int): Total number of grid points
    |   tensors(Numpy float array, shape:(total_grid_n, num_atoms, 3, 3)):
    |       Array of hyperfine tensors for each atom at each grid point
    |   hfine_tens_avg(Numpy float array, shape:(num_atoms,3,3)): Average
    |       tensors of atoms over grid
    |   r2psi2 (Numpy float, shape:(total_grid_n)): Probability density of
    |       harmonic oscillator at each displacement
    |   filename(str): Filename to be used for file
    |   atoms(dict, {index(int):symbol(str)}): Dictionary containing indices and
    |       symbols of atoms to write hyperfine coupling report about
    |
    | Returns: Nothing
    """
    ofile = open(filename, 'w')
    for index in atoms:
        hfine_table = np.zeros((total_grid_n))
        for i, tensor in enumerate(tensors[:][index]):
            hfine_table[i] = np.trace(tensor)/3.0

        hfine_avg = np.sum(r2psi2*hfine_table)/np.sum(r2psi2)
        ofile.write('Predicted hyperfine coupling on labeled atom ({1}): {0} MHz\n'.format(
            hfine_avg, atoms[index]))

        evals, evecs = np.linalg.eigh(hfine_tens_avg[index])
        evals, evecs = zip(*sorted(zip(evals, evecs), key=lambda x: abs(x[0])))
        evals_notr = -np.array(evals)+np.average(evals)

        if abs(evals_notr[2]) > abs(evals_notr[0]):
            D1 = evals_notr[2]
            D2 = evals_notr[1]-evals_notr[0]
        else:
            D1 = evals_notr[0]
            D2 = evals_notr[2]-evals_notr[1]

        ofile.write(('Predicted dipolar hyperfine components on labeled atom ({2}):\n'
                     'D1:\t{0} MHz\nD2:\t{1} MHz\n').format(
            D1, D2, atoms[index]))
