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
from ase import Atoms
from ase.calculators.dftb import Dftb
from ase.dft import kpoints
from ase.optimize import BFGS
from ase.phonons import Phonons

def ase_phonon_calc(cell):
    """Calculate phonon modes of a molecule using ASE and DFTB+. The system
    will be geometry optimized using DFTB+ before calculating the modes. A
    report of the phonon modes will be written to a file and arrays of the
    eigenvectors and eigenvalues returned.

    | Args:
    |   cell(ASE Atoms object): Atoms object with to calculate modes for.
    | Returns:
    |   evals(float[k-points][modes]): Eigenvalues of phonon modes
    |   evecs(float[k-points][modes][ions][3]): Eigenvectors of phonon modes
    """
    dftb_cell = copy.deepcopy(cell)
    # Relax structure using DFTB+
    calc = Dftb(kpts=[1,1,1])
    dftb_cell.set_calculator(calc)
    dyn = BFGS(dftb_cell, trajectory='geom_opt.traj')
    dyn.run(fmax=0.01)
    dftb_cell.set_positions((ase_io.read("geo_end.xyz")).get_positions())

    # Calculate phonon modes
    ph = Phonons(dftb_cell, calc)
    ph.run()
    ph.read(acoustic=True)
    path = kpoints.monkhorst_pack((1,1,1))
    evals, evecs = ph.band_structure(path, True)
    evals *= 8065.5 #Convert from eV to cm-1

    # Write phonon report
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
        major_evecs[1] = major_evecs[1] - \
                           np.dot(major_evecs[0], major_evecs[1])*major_evecs[0]
        major_evecs[1] = major_evecs[1]/np.linalg.norm(major_evecs[1])
        major_evecs[2] = major_evecs[2] - \
                         np.dot(major_evecs[0], major_evecs[2])*major_evecs[0] - \
                         np.dot(major_evecs[2], major_evecs[1])*major_evecs[1]
        major_evecs[2] = major_evecs[2]/np.linalg.norm(major_evecs[2])

    return major_evecs_i, major_evecs.real
