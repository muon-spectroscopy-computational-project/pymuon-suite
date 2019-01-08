"""
Author: Adam Laverack and Simone Sturniolo
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import shutil

import numpy as np
import scipy.constants as cnst
from ase import Atoms
from ase.calculators.dftb import Dftb
from ase.dft import kpoints
from ase.phonons import Phonons

def ase_phonon_calc(cell, dftb_phonons):
    """Calculate phonon modes of a molecule using ASE. If dftb_phonons is true,
    DFTB+ will be used as the calculator. Otherwise, the input cell's calculator
    will be used. A report of the phonon modes will be written to a file and
    arrays of the eigenvectors and eigenvalues returned.

    | Args:
    |   cell(ASE Atoms object): Atoms object with geometry to calculate modes
    |   for.
    |   dftb_phonons(bool): If True, use DFTB+ to calculate. If False, use
    |   cell's default calculator.
    | Returns:
    |   evals(float[k-points][modes]): Eigenvalues of phonon modes
    |   evecs(float[k-points][modes][ions][3]): Eigenvectors of phonon modes
    """
    if dftb_phonons:
        phonon_calc = Dftb(kpts=[1,1,1])
    else:
        phonon_calc = cell.get_calculator()
    ph = Phonons(cell, phonon_calc)
    ph.run()
    ph.read(acoustic=True)
    path = kpoints.monkhorst_pack((1,1,1))
    evals, evecs = ph.band_structure(path, True)
    evals *= 8065.5 #Convert from eV to cm-1

    #Write phonon report
    filename = "ase_phonons.dat"
    phonfile = open(filename, 'a')
    print("Writing phonon report in location: ", filename)
    phonfile.write("Eigenvalues\n")
    for i, kpt in enumerate(evals):
        phonfile.write("Mode Frequency(cm-1) k-point = {0}\n".format(i))
        for j, value in enumerate(kpt):
            phonfile.write("{0} \t{1}\n".format(j, value))
    phonfile.write("Eigenvectors\n")
    phonfile.write("Mode Ion Vector\n")
    for i, mode in enumerate(evecs[0]):
        for j, ion in enumerate(mode):
            phonfile.write("{0} {1} \t{2}\n".format(i, j, ion))

    return evals, evecs

def calc_harm_potential(R, grid_n, mass, freqs, E_table, filename):
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
    R_axes = np.array([np.linspace(-3*Ri, 3*Ri, grid_n)
                       for Ri in R])
    # Now the potential, measured vs. theoretical
    harm_K = mass*freqs**2
    harm_V = (0.5*harm_K[:, None]*(R_axes*1e-10)**2)/cnst.electron_volt
    # Normalise E_table
    if E_table.shape[1] % 2 == 1:
        E_table -= (E_table[:, E_table.shape[1]//2])[:, None]
    else:
        E_table -= (E_table[:, E_table.shape[1]//2] +
                    E_table[:, E_table.shape[1]//2-1])[:, None]/2.0
    all_table = np.concatenate((R_axes, harm_V, E_table), axis=0)
    np.savetxt(filename, all_table.T)

def get_major_emodes(evecs, i):
    """Find the normalized phonon modes of the atom at index i

    | Args:
    |   evecs (Numpy float array, shape: (num_modes, num_ions, 3)):
    |                                   Eigenvectors of phonon modes of molecule
    |   i (int): Index of atom in position array
    |
    | Returns:
    |   major_evecs_i (int[3]): Indices of atom's phonon eigenvectors in evecs
    |   major_evecs (float[3, 3]): Normalized eigenvectors of atom's phonon modes
    |   major_evecs_ortho (float[3, 3]): Orthogonalised phonon modes
    """
    # First, find the eigenmodes whose amplitude is greater for ion i
    evecs_amp = np.linalg.norm(evecs, axis=-1)
    ipr = evecs_amp**4/np.sum(evecs_amp**2, axis=-1)[:, None]**2
    evecs_order = np.argsort(ipr[:, i])

    # How many?
    major_evecs_i = evecs_order[-3:]
    major_evecs = evecs[major_evecs_i, i]
    major_evecs_ortho = np.linalg.qr(major_evecs.T)[0].T

    return major_evecs_i, major_evecs, major_evecs_ortho
