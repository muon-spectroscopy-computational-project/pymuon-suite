"""
Author: Adam Laverack and Simone Sturniolo
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

def hfine_report(R, grid_n, tensors, hfine_tens_avg, r2psi2, filename, symbol):
    """Write a report on an atom's hyperfine coupling constant and its
    hyperfine tensor dipolar components based on a vibrational averaging
    calculation.

    | Args:
    |   R(Numpy float array, shape:(axes)): Displacement amplitude along each
    |       phonon axis
    |   grid_n(int): Number of grid points along each axis
    |   tensors(Numpy float array, shape:(np.size(R), grid_n, 3, 3)): Array of
    |       hyperfine tensors for atom at each grid point
    |   hfine_tens_avg(float): Average tensor of desired atom over grid
    |   r2psi2 (Numpy float, shape:(size(R), grid_n)): Probability density of
    |       harmonic oscillator at each displacement
    |   filename(str): Filename to be used for file
    |   symbol(str): Symbol of atom
    |
    | Returns: Nothing
    """
    hfine_table = np.zeros((np.size(R), grid_n))
    for i, axis in enumerate(tensors):
        for j, tensor in enumerate(axis):
            hfine_table[i][j] = np.trace(tensor)/3.0

    ofile = open(filename, 'a')

    hfine_avg = np.sum(r2psi2*hfine_table)/np.sum(r2psi2)
    ofile.write('Predicted hyperfine coupling on labeled atom ({1}): {0} MHz\n'.format(
        hfine_avg, symbol))

    evals, evecs = np.linalg.eigh(hfine_tens_avg)
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
        D1, D2, symbol))
