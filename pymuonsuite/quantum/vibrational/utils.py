"""
Author: Simone Sturniolo(Functionality) and Adam Laverack(Interface)
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


def get_major_emodes(evecs, i):
    """Find the normalized phonon modes of the atom at index i

    | Args:
    |   evecs (Numpy float array, shape: (num_modes, num_ions, 3)):
    |                                   Eigenvectors of phonon modes of molecule
    |   i (int): Index of atom in position array
    |
    | Returns:
    |   major_evecs_i (int[3]): Indices of atom's phonon eigenvectors in evecs
    |   major_evecs (float[3]): Normalized eigenvectors of atom's phonon modes
    |   major_evecs_ortho (float[3]): Orthogonalised phonon modes
    """
    # First, find the eigenmodes whose amplitude is greater for ion i
    evecs_amp = np.linalg.norm(evecs, axis=-1)
    ipr = evecs_amp**4/np.sum(evecs_amp**2, axis=-1)[:, None]**2
    evecs_order = np.argsort(ipr[:, i])

    # How many?
    major_evecs_i = evecs_order[-3:]
    major_evecs = evecs[major_evecs_i, i]
    major_evecs_ortho = np.linalg.qr(major_evecs.T)[0].T

    major_evecs = major_evecs/np.linalg.norm(major_evecs, axis=-1, keepdims=True)
    major_evecs = np.real(major_evecs)

    return major_evecs_i, major_evecs, major_evecs_ortho
