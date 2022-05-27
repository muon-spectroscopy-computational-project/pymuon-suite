"""
Author: Simone Sturniolo

Functions and classes to provide various possible displacement schemes for
different averaging methods meant to approximate nuclear quantum effects.
"""


import pickle
import numpy as np
import scipy.constants as cnst
from pymuonsuite.quantum.vibrational.phonons import get_major_emodes

# Cm^-1 to rad/s
_wnum2om = 2 * np.pi * 1e2 * cnst.c
# Cm^-1 to J
_wnum2E = _wnum2om * cnst.hbar
# Cm^-1 to K
_wnum2T = _wnum2E / cnst.k


class PhononDisplacementError(Exception):
    pass


def _wnumSigmaEnhance(wnums, T=0):
    # Enhancement factor for the sigmas, given the energies in wave number
    if T > 0:
        xi = np.exp(-_wnum2T * wnums / T)
    else:
        xi = 0 * wnums
    tf = (1.0 - xi) / (1 + xi)
    return tf


class DisplacementScheme(object):

    """DisplacementScheme

    A generic class template for various quantum averaging displacement
    schemes. Meant to store the displacements, be saved/loaded as a pickle,
    and calculate the weights as a function of temperature.
    This class is not meant to be used directly: rather, the derived classes
    will use it as a template to implement the actual schemes.
    """

    def __init__(self, evals, evecs, masses, evals_threshold=1e-3):

        evals = np.real(evals)
        evecs = np.real(evecs)
        masses = np.array(masses)

        if (evals <= max(0, evals_threshold)).any():
            if evals_threshold > 0:
                print(
                    "Warning: removing eigenmodes with frequency "
                    "< {0}".format(evals_threshold)
                )
                evals_i = np.where(evals > evals_threshold)[0]
                evals = evals[evals_i]
                evecs = evecs[evals_i]
            else:
                raise PhononDisplacementError("Imaginary frequency eigenmodes")

        self._evals = evals
        self._evecs = evecs
        self._masses = masses * cnst.u  # amu to kg

        self._sigmas = np.real((cnst.hbar / (_wnum2om * evals)) ** 0.5)

        self._n = 0  # Grid points
        self._M = evecs.shape[0]  # Number of modes (should be 3N)
        self._N = evecs.shape[1]  # Number of atoms

        self._dq = None
        self._dx = None
        self._w = None

        self._Td = 0  # Temperature for displacements
        self._Tw = 0  # Temperature for weights

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
    def Td(self):
        return self._Td

    @property
    def Tw(self):
        return self._Tw

    def save(self, file):
        pickle.dump(self, open(file, "w"))

    @staticmethod
    def load(file):
        return pickle.load(open(file))

    def recalc_all(self, displ_args={}, weights_args={}):
        self.recalc_displacements(**displ_args)
        self.recalc_weights(**weights_args)

        return self.displacements, self.weights

    def recalc_displacements(self):
        raise NotImplementedError(
            "DisplacementScheme has no implementation"
            " of this method; use one of the derived "
            "classes."
        )

    def recalc_weights(self):
        raise NotImplementedError(
            "DisplacementScheme has no implementation"
            " of this method; use one of the derived "
            "classes."
        )

    def __str__(self):
        return "Generic DisplacementScheme"


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

    def __init__(self, evals, evecs, masses, i, sigma_n=3):
        super(self.__class__, self).__init__(evals, evecs, masses)

        # Find the major eigenmodes for the atom of interest
        self._i = i
        self._majev = get_major_emodes(self._evecs, masses, i, ortho=True)

        self._sigma_n = sigma_n  # Number of sigmas covered

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
    def sigma_n(self):
        return self._sigma_n

    def recalc_displacements(self, n=20, T=0):

        self._Td = T
        self._n = n
        # Displacements along the three normal modes of choice
        dz = np.linspace(-self.sigma_n, self.sigma_n, n)

        sx = self.major_sigmas
        self._dq = np.zeros((3 * n, 3))
        for i in range(3):
            self._dq[n * i : n * (i + 1), i] = dz * sx[i]

        # Turn these into position displacements
        dx = np.dot(self._dq, self.major_evecs)
        dx *= 1e10 / self.masses[self.i] ** 0.5

        # We need also the central configuration,
        # so depending on whether n is even or odd
        # we get two different situations
        self._dx = np.zeros((3 * n + 1, self._N, 3))
        self._dx[1:, self.i, :] = dx
        if n % 2 == 1:
            ci = int((n - 1) / 2)
            # Remove the superfluous zeroes
            self._dx = np.delete(self._dx, np.arange(3) * n + ci + 1, axis=0)

        return self.displacements

    def recalc_weights(self, T=0):

        self._Tw = T

        tfac = _wnumSigmaEnhance(self.major_evals, T)

        # Now for the weights
        dz = np.linspace(-self.sigma_n, self.sigma_n, self.n)
        w0 = -2.0  # Weight of the central configuration

        rho = np.exp(-(dz**2))
        rhoall = rho[None, :] ** tfac[:, None]
        rhoall /= np.sum(rhoall, axis=1)[:, None]

        if self.n % 2 == 1:
            ci = int((self.n - 1) / 2)
            # Fix the central configuration's weight
            w0 += np.sum(rhoall[:, ci])
            rhoall = np.delete(rhoall, ci, axis=1)

        self._w = np.zeros(np.prod(rhoall.shape) + 1)
        self._w[0] = w0
        self._w[1:] = np.concatenate(rhoall)

        return self.weights

    def __str__(self):
        msg = """Independent Displacements Scheme
Displaces one single atom of index i along the
three phonon modes with greatest Atomic Participation Ratio (APR).

-------------------------

Atom index: {i}

Phonon frequencies: \n{evals} cm^-1

Displacement vectors: \n{evecs}

Temperature: \n{Td} K (displacements), {Tw} K (weights)

Max sigma N: \n{sN}

Weights: \n{w}

-------------------------
        """.format(
            i=self.i,
            evals="\t".join(map(str, self.major_evals)),
            evecs="\n".join(map(str, self.major_evecs)),
            Td=self.Td,
            Tw=self.Tw,
            w=self.weights,
            sN=self.sigma_n,
        )

        if self.Tw != self.Td:
            msg += """
WARNING: Temperatures for displacements and weights are different.
This can be a cause of inaccuracy in averaging.

------------------------
"""
        if self.Tw > self.Td:
            msg += """
WARNING: Temperatures for weights is higher than for displacements.
This is very likely to cause inaccuracy in averaging.

------------------------
"""

        return msg


class MonteCarloDisplacements(DisplacementScheme):
    """MonteCarloDisplacements

    Compute displacements and weights for an averaging method based on
    sampling the thermal distribution for multiple modes at the same time.
    This method is described in B. Monserrat et al., Jour. Chem. Phys. 141,
    134113 (2014).

    The selected modes are sampled at the same time, producing configurations
    with a normal distribution around the equilibrium configuration.
    """

    def __init__(self, evals, evecs, masses, modes=None):
        super(self.__class__, self).__init__(evals, evecs, masses)

        # Selected modes
        if modes is None:
            modes = np.arange(self._M)  # All of them

        self._modes = modes

    @property
    def modes(self):
        return self._modes.copy()

    @property
    def T(self):
        return self._T

    def recalc_displacements(self, n=50, T=0):

        self._n = n
        self._Td = T

        tfac = _wnumSigmaEnhance(self._evals, T)

        dz = np.random.normal(size=(n, len(self._modes)), scale=0.5**0.5)
        self._dq = np.zeros((n, self._M))
        self._dq[:, self._modes] = dz * (self._sigmas / tfac**0.5)[None, self._modes]

        # Turn these into position displacements
        dx = self._dq[:, self._modes, None, None] * self._evecs[None, self._modes]
        dx = np.sum(dx, axis=1)
        dx *= 1e10 / self.masses[None, :, None] ** 0.5

        self._dx = dx

        return self.displacements

    def recalc_weights(self, T=0):

        self._Tw = T

        if self._Tw > self._Td:
            print(
                "WARNING: reweighing temperature is higher than displacement"
                " temperature in MonteCarlo displacements scheme."
                " This is likely to cause major errors on averages."
            )

        tfacw = _wnumSigmaEnhance(self._evals, self._Tw)
        tfacd = _wnumSigmaEnhance(self._evals, self._Td)

        sd_sw = (tfacw / tfacd) ** 0.5
        dz = self._dq / self._sigmas[None, self._modes]
        qw = sd_sw[None, self._modes] * np.exp(
            dz**2 * (tfacd - tfacw)[None, self._modes]
        )

        self._w = np.prod(qw, axis=1) / self._n

        return self.weights

    def __str__(self):
        msg = """Monte Carlo Normal Displacements Scheme
Displaces all atoms along chosen phonon modes, randomly,
with amplitude determined by the desired temperature. Since they already
follow a normal distribution, all points have equal weight.

-------------------------

Modes: \n{modes}

Phonon frequencies: \n{evals} cm^-1

Displacement vectors: \n{evecs}

Temperature: \n{Td} K (displacements), {Tw} K (weights)

-------------------------
        """.format(
            modes=self._modes,
            evals="\t".join(map(str, self._evals)),
            evecs="\n".join(map(str, self._evecs)),
            Td=self.Td,
            Tw=self.Tw,
        )

        if self.Tw != self.Td:
            msg += """
WARNING: Temperatures for displacements and weights are different.
This can be a cause of inaccuracy in averaging.

------------------------
"""
        if self.Tw > self.Td:
            msg += """
WARNING: Temperatures for weights is higher than for displacements.
This is very likely to cause inaccuracy in averaging.

------------------------
"""

        return msg
