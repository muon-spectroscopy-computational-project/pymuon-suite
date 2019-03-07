"""
Author: Simone Sturniolo

Functions to provide various possible displacement schemes for quantum effects
approximations.
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import scipy.constants as cnst
from pymuonsuite.quantum.vibrational.phonons import get_major_emodes

# Cm^-1 to rad/s
_wnum2om = 2*np.pi*1e2*cnst.c


class DisplacementScheme(object):

    def __init__(self, evals, evecs, masses):

        evals = np.array(evals)
        evecs = np.array(evecs)
        masses = np.array(masses)

        self._evals = evals
        self._evecs = evecs
        self._masses = masses*cnst.u                        # amu to kg
        self._sigmas = (cnst.hbar/(_wnum2om*evals))**0.5

        self._M = evecs.shape[0]  # Number of modes (should be 3N)
        self._N = evecs.shape[1]  # Number of atoms

    def displacements(self, n=100, sigma_n=3):
        pass

    def weights(self, dx, T=0):
        pass


class MollerDisplacements(DisplacementScheme):

    def __init__(self, evals, evecs, masses, i):
        super(MollerDisplacements, self).__init__(evals, evecs, masses)

        # Find the major eigenmodes for the atom of interest
        self._i = i
        self._majev = get_major_emodes(evecs, masses, i, ortho=True)

    def displacements(self, n=100, sigma_n=3):

        displ = np.zeros((3*n, self._N, 3))

        mi, mvs = self._majev

        # Range for q, normal mode variable
        qrng = np.concatenate([np.linspace(-sigma_n, sigma_n, n)[:, None] *
                               v[None, :] * self._sigmas[i]
                               for i, v in zip(mi, mvs)])

        # To x (Ang)
        xrng = (qrng/self._masses[self._i]**0.5)*1e10

        displ[:, self._i, :] = xrng

        return displ

    def weights(self, dx, T=0):

        mi, mvs = self._majev

        om = _wnum2om*self._evals[mi]
        if T > 0:
            xi = np.exp(-cnst.hbar*om/(2*cnst.k*T))
        else:
            xi = np.ones(3)
        
        
