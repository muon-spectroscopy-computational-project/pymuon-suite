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
from soprano.collection import AtomsCollection
from pymuonsuite.quantum.vibrational.phonons import get_major_emodes

# Cm^-1 to rad/s
_wnum2om = 2*np.pi*1e2*cnst.c


class PhononDisplacementError(Exception):
    pass


class DisplacementScheme(object):

    """DisplacementScheme

    A generic class template for various quantum averaging displacement
    schemes. Meant to store the displacements, be saved/loaded as a pickle,
    and calculate the weights as a function of temperature.
    This class is not meant to be used directly: rather, the derived classes
    will use it as a template to implement the actual schemes.
    """

    def __init__(self, evals, evecs, masses, cut_imaginary=True):

        evals = np.real(evals)
        evecs = np.real(evecs)
        masses = np.array(masses)

        if (evals <= 0).any():
            if cut_imaginary:
                print('Warning: removing imaginary frequency eigenmodes')
                evals_i = np.where(evals > 0)[0]
                evals = evals[evals_i]
                evecs = evecs[evals_i]
            else:
                raise PhononDisplacementError('Imaginary frequency eigenmodes')

        self._evals = evals
        self._evecs = evecs
        self._masses = masses*cnst.u                        # amu to kg

        self._sigmas = np.real((cnst.hbar/(_wnum2om*evals))**0.5)

        self._n = 0               # Grid points
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

    def save(self, file):
        pickle.dump(self, open(file, 'w'))

    @staticmethod
    def load(file):
        return pickle.load(open(file))

    def recalc_all(self, displ_args={}, weights_args={}):
        self.recalc_displacements(**displ_args)
        self.recalc_weights(**weights_args)

        return self.displacements, self.weights

    def recalc_displacements(self):
        raise NotImplementedError('DisplacementScheme has no implementation'
                                  ' of this method; use one of the derived '
                                  'classes.')

    def recalc_weights(self):
        raise NotImplementedError('DisplacementScheme has no implementation'
                                  ' of this method; use one of the derived '
                                  'classes.')

    def __str__(self):
        return 'Generic DisplacementScheme'


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
        self._majev = get_major_emodes(self._evecs, masses, i, ortho=True)

        self._T = 0
        self._sigma_n = 3         # Number of sigmas covered

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

    @property
    def sigma_n(self):
        return self._sigma_n

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

        # We need also the central configuration,
        # so depending on whether n is even or odd
        # we get two different situations
        self._dx = np.zeros((3*n+1, self._N, 3))
        self._dx[1:, self.i, :] = dx
        if n % 2 == 1:
            ci = int((n-1)/2)
            # Remove the superfluous zeroes
            self._dx = np.delete(self._dx, np.arange(3)*n+ci+1,
                                 axis=0)

        return self.displacements

    def recalc_weights(self, T=0):

        self._T = T

        om = self.major_evals*1e2*cnst.c*2*np.pi
        if T > 0:
            xi = np.exp(-cnst.hbar*om/(cnst.k*T))
        else:
            xi = om*0
        tfac = (1.0-xi**2)/(1+xi**2)

        # Now for the weights
        dz = np.linspace(-self.sigma_n, self.sigma_n, self.n)
        w0 = -2.0  # Weight of the central configuration

        rho = np.exp(-dz**2)
        rhoall = np.array([rho**tf/np.sum(rho**tf) for tf in tfac])

        if self.n % 2 == 1:
            ci = int((self.n-1)/2)
            # Fix the central configuration's weight
            w0 += np.sum(rhoall[:, ci])
            rhoall = np.delete(rhoall, ci, axis=1)

        self._w = np.zeros(np.prod(rhoall.shape)+1)
        self._w[0] = w0
        self._w[1:] = np.concatenate(rhoall)

        return self.weights

    def __str__(self):
        return """Independent Displacements Scheme
Displaces one single atom of index i along the
three phonon modes with greatest Atomic Participation Ratio (APR).

-------------------------

Atom index: {i}

Phonon frequencies: \n{evals} cm^-1

Displacement vectors: \n{evecs}

Temperature: \n{T} K

Max sigma N: \n{sN}

Weights: \n{w}

-------------------------
        """.format(i=self.i, evals='\t'.join(map(str, self.major_evals)),
                   evecs='\n'.join(map(str, self.major_evecs)),
                   T=self.T, w=self.weights, sN=self.sigma_n,
                   sigmas=self.major_sigmas)
