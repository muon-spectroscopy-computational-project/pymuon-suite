#!/usr/bin/env python

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import numpy as np
import scipy.constants as cnst
from ase import Atoms
from soprano.utils import minimum_periodic
from pymuonsuite.io.parse_phonon_file import parse_phonon_file

def get_major_emodes(evecs, i):

    # First, find the eigenmodes whose amplitude is greater for ion i
    evecs_amp = np.linalg.norm(evecs, axis=-1)
    ipr = evecs_amp**4/np.sum(evecs_amp**2, axis=-1)[:,None]**2
    evecs_order = np.argsort(ipr[:,i])

    # How many?
    major_evecs_i = evecs_order[-3:]
    major_evecs = evecs[major_evecs_i,i]
    major_evecs_ortho = np.linalg.qr(major_evecs.T)[0].T

    return major_evecs_i, major_evecs, major_evecs_ortho


if __name__ == "__main__":


    struct, ph_evals, ph_evecs = parse_phonon_file(sys.argv[1])
    mu_sym = sys.argv[2]

    pos = struct.get_positions()
    chems = struct.get_array('castep_custom_species')

    # Ok, so first, find the muon
    mu_i = np.where(chems == mu_sym)[0]
    if len(mu_i) != 1:
        raise RuntimeError('Invalid muon symbol - {0} muons found'.format(len(mu_i)))
    mu_i = mu_i[0]

    # Check against the masses
    if mu_i != np.argmin(struct.get_masses()):
        raise RuntimeError('Invalid muon symbol - not the lightest particle in the system!')

    # Ok, on to work. Find ipso hydrogen
    iH = np.where(chems == 'H')[0]
    posH = pos[iH]
    distH = np.linalg.norm(minimum_periodic(posH - pos[mu_i], struct.get_cell())[0], axis=-1)
    # Which one is the closest?
    ipso_i = iH[np.argmin(distH)]

    # And there we go. Find the best triplet
    # For mu_i:
    em_i, em, em_o = get_major_emodes(ph_evecs, mu_i)
    print("#Muon MODES:")
    print("{0}   # Mass of the particle in kg".format(struct.get_masses()[mu_i]*cnst.u))
    print("{0} {1} {2} # Eigenvalues, in cm^-1".format(*ph_evals[em_i]))
    print("#### Eigenvector matrix, rows = eigenvectors ####")
    for e in em:
        print("{0} {1} {2}".format(*np.real(e)))

    print("")

    em_i, em, em_o = get_major_emodes(ph_evecs, ipso_i)
    print("#Ipso hydrogen MODES:")
    print("{0}   # Mass of the particle in kg".format(struct.get_masses()[ipso_i]*cnst.u))
    print("{0} {1} {2} # Eigenvalues, in cm^-1".format(*ph_evals[em_i]))
    print("#### Eigenvector matrix, rows = eigenvectors ####")
    for e in em:
        print("{0} {1} {2}".format(*np.real(e)))
