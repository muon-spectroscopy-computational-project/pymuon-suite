"""
Author: Adam Laverack and Simone Sturniolo
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import os
import shutil

import ase.io as ase_io
import numpy as np
import scipy.constants as cnst

from ase import Atoms
from ase.calculators.dftb import Dftb
from ase.dft import kpoints
from ase.optimize import BFGS
from ase.phonons import Phonons

def ase_phonon_calc(cell, calc=None, fname='ase_phonons', ftol=0.01):
    """Calculate phonon modes of a molecule using ASE and a given calculator.
    The system will be geometry optimized before calculating the modes. A
    report of the phonon modes will be written to a file and arrays of the
    eigenvectors and eigenvalues returned.

    | Args:
    |   cell (ase.Atoms):       Atoms object with to calculate modes for.
    |   calc (ase.Calculator):  Calculator for energies and forces (if not 
    |                           present, use the one from the cell)
    |   fname (str):            Name for the .dat file that will hold the 
    |                           phonon report (default is ase_phonons)
    |   ftol (float):           Tolerance for geometry optimisation (default
    |                           is 0.01 eV/Ang)
    | Returns:
    |   evals (float[k-points][modes]):          Eigenvalues of phonon modes
    |   evecs (float[k-points][modes][ions][3]): Eigenvectors of phonon modes
    |   cell (ase.Atoms):                        Optimised cell
    """

    if calc is None:
        calc = cell.calc
    cell = cell.copy()
    calc.atoms = cell
    cell.set_calculator(calc)
    dyn = BFGS(cell, trajectory='geom_opt.traj')
    dyn.run(fmax=ftol)

    # Calculate phonon modes
    ph = Phonons(cell, calc)
    ph.run()
    ph.read(acoustic=True)
    path = kpoints.monkhorst_pack((1,1,1))
    evals, evecs = ph.band_structure(path, True)

    # eV to cm^-1
    evals *= ((cnst.electron_volt/cnst.h)/cnst.c)/100.0

    # Write phonon report
    filename = fname + '.dat'
    with open(filename, 'w') as phonfile:
        print('Writing phonon report in location: ', filename)
        phonfile.write('Eigenvalues\n')
        for i, kpt in enumerate(evals):
            phonfile.write('Mode Frequency(cm-1) k-point = {0}\n'.format(i))
            for j, value in enumerate(kpt):
                phonfile.write('{0} \t{1}\n'.format(j, value))
        phonfile.write('Eigenvectors\n')
        phonfile.write('Mode Ion Vector\n')
        for i, mode in enumerate(evecs[0]):
            for j, ion in enumerate(mode):
                phonfile.write("{0} {1} \t{2}\n".format(i, j, ion))

    return evals, evecs, cell

def get_major_emodes(evecs, i, ortho=False):
    """Find the phonon modes of the atom at index i. Return orthogonalized and
    normalized modes if ortho == True.

    | Args:
    |   evecs (Numpy float array, shape: (num_modes, num_ions, 3)):
    |                                   Eigenvectors of phonon modes of molecule
    |   i (int): Index of atom in position array
    |   ortho (bool): If true, orthogonalize and normalize major modes before
    |       returning.
    |
    | Returns:
    |   major_evecs_i (int[3]): Indices of atom's phonon eigenvectors in evecs
    |   major_evecs (float[3, 3]): Eigenvectors of atom's phonon modes
    """
    # First, find the eigenmodes whose amplitude is greater for ion i
    evecs_amp = np.linalg.norm(evecs, axis=-1)
    ipr = evecs_amp**4/np.sum(evecs_amp**2, axis=-1)[:, None]**2
    evecs_order = np.argsort(ipr[:, i])

    # How many?
    major_evecs_i = evecs_order[-3:]
    major_evecs = evecs[major_evecs_i, i]

    if ortho == True:
        #Orthogonolize and normalize
        major_evecs[0] = major_evecs[0]/np.linalg.norm(major_evecs[0])
        major_evecs[1] = major_evecs[1] - \
                           np.dot(major_evecs[0], major_evecs[1])*major_evecs[0]
        major_evecs[1] = major_evecs[1]/np.linalg.norm(major_evecs[1])
        major_evecs[2] = major_evecs[2] - \
                         np.dot(major_evecs[0], major_evecs[2])*major_evecs[0] - \
                         np.dot(major_evecs[2], major_evecs[1])*major_evecs[1]
        major_evecs[2] = major_evecs[2]/np.linalg.norm(major_evecs[2])

    return major_evecs_i, major_evecs.real
