"""
Author: Simone Sturniolo

Functions and classes to provide various possible displacement schemes for 
different averaging methods meant to approximate nuclear quantum effects.
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pickle
import numpy as np
import scipy.constants as cnst
from pymuonsuite.quantum.vibrational.phonons import get_major_emodes

# Cm^-1 to rad/s
_wnum2om = 2*np.pi*1e2*cnst.c


class DisplacementScheme(object):

    """DisplacementScheme

    A generic class template for various quantum averaging displacement
    schemes. Meant to store the displacements, be saved/loaded as a pickle,
    and calculate the weights as a function of temperature.
    This class is not meant to be used directly: rather, the derived classes
    will use it as a template to implement the actual schemes.
    """

    def __init__(self, evals, evecs, masses):

        evals = np.array(evals)
        evecs = np.array(evecs)
        masses = np.array(masses)

        self._evals = evals
        self._evecs = evecs
        self._masses = masses*cnst.u                        # amu to kg
        self._sigmas = (cnst.hbar/(_wnum2om*evals))**0.5

        self._n = 0               # Grid points
        self._sigma_n = 3         # Number of sigmas covered
        self._M = evecs.shape[0]  # Number of modes (should be 3N)
        self._N = evecs.shape[1]  # Number of atoms

        self._dq = None
        self._dx = None
        self._w = None

    @property
    def evals(self):
        return self._evals.copy()

    @property
    def evecs(self):
        return self._evecs.copy()

    @property
    def masses(self):
        return self._masses.copy()

    @property
    def sigmas(self):
        return self._sigmas.copy()

    @property
    def displacements_q(self):
        return self._dq.copy()

    @property
    def displacements(self):
        return self._dx.copy()

    @property
    def weights(self):
        return self._w.copy()

    @property
    def n(self):
        return self._n

    @property
    def sigma_n(self):
        return self._sigma_n

    def save(self, file):
        pickle.dump(self, open(file, 'w'))

    @staticmethod
    def load(file):
        return pickle.load(open(file))

    def recalc_displacements(self, n=20, sigma_n=3):
        raise NotImplementedError('DisplacementScheme has no implementation'
                                  ' of this method; use one of the derived '
                                  'classes.')

    def recalc_weights(self, T=0):
        raise NotImplementedError('DisplacementScheme has no implementation'
                                  ' of this method; use one of the derived '
                                  'classes.')


class IndependentDisplacements(DisplacementScheme):
    """IndependentDisplacements

    Compute displacements and weights for an averaging method based on 
    independent motion along three axes. This method is an improved version of
    the one used by Moller et al., Phys. Rev. B 87, 121108(R) (2013). 
    The only modes to matter are considered to be the three ones with the 
    highest APR for the ion of interest and the quantity to average is assumed 
    to be separable as a sum of these three variables:

    f(x1, x2, x3) = f_1(x1) + f_2(x2) + f_3(x3)

    so that the three averages can be effectively performed separately in one
    dimension each:

    <f> = <f_1> + <f_2> + <f_3>
    """

    def __init__(self, evals, evecs, masses, i):
        super(self.__class__, self).__init__(evals, evecs, masses)

        # Find the major eigenmodes for the atom of interest
        self._i = i
        self._majev = get_major_emodes(evecs, masses, i, ortho=True)

        self._T = 0

    @property
    def i(self):
        return self._i

    @property
    def major_evecs(self):
        return self._majev[1].copy()

    @property
    def major_evecs_inds(self):
        return self._majev[0].copy()

    @property
    def major_evals(self):
        return self._evals[self._majev[0]]

    @property
    def major_sigmas(self):
        return self._sigmas[self._majev[0]]

    @property
    def T(self):
        return self._T

    def recalc_displacements(self, n=20, sigma_n=3):

        self._n = n
        self._sigma_n = sigma_n
        # Displacements along the three normal modes of choice
        dz = np.linspace(-sigma_n, sigma_n, n)

        sx = self.major_sigmas
        self._dq = np.zeros((3*n, 3))
        for i in range(3):
            self._dq[n*i:n*(i+1), i] = dz*sx[i]

        # Turn these into position displacements
        dx = np.dot(self._dq, self.major_evecs)
        dx *= 1e10/self.masses[self.i]**0.5

        self._dx = np.zeros((3*n, self._N, 3))
        self._dx[:, self.i, :] = dx

        return self.displacements

    def recalc_weights(self, T=0):

        self._T = T

        om = self.major_evals*1e2*cnst.c*2*np.pi
        xi = np.exp(-cnst.hbar*om/(cnst.k*T))
        tfac = (1.0-xi**2)/(1+xi**2)

        # Now for the weights
        sx = self.major_sigmas

        dz = np.linspace(-self.sigma_n, self.sigma_n, self.n)

        rho = np.exp(-dz**2)
        rhoall = [rho**tf/np.sum(rho**tf) for tf in tfac]
        self._w = np.concatenate(rhoall)

        return self.weights
