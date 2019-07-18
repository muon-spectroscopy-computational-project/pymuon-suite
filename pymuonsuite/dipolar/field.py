"""
field.py

DipolarField class, computing dipolar field distributions at a muon position
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings
import numpy as np
import scipy.constants as cnst
from scipy.integrate import quad, romberg
from soprano.utils import minimum_supcell, supcell_gridgen
from soprano.calculate.powder import ZCW, SHREWD, TriAvg
from soprano.properties.nmr.utils import _get_isotope_data, _dip_constant
from pymuonsuite.constants import m_gamma

# try:
#     from numba import jit
# except ImportError:
#     warnings.warn('Numba not found - install for boost in performance')

#     def jit(nopython=True):
#         def dummy(f):
#             def wrapf(*args, **kwargs):
#                 return f(*args, **kwargs)
#             return wrapf
#         return dummy

# Dipolar line functions


def _distr_D(x, D):
    x = np.where((x > -D/2)*(x < D), x, np.inf)
    return 1/(3*D*((2*(x/D))/3+1.0/3.0)**0.5)


def _distr_eta(x, x0, D, eta):
    den = eta**2*(2-2*x0/D)**2-9*(x-x0)**2
    den = np.where(den > 0, den, np.inf)
    y = 1.0/den**0.5 * 3/np.pi
    return y


def _distr_spec(x, D, eta, nsteps=3000):
    x0min = np.expand_dims((x - 2/3.0*eta)/(1-2/3.0*eta/D), 0)
    x0min = np.where(x0min > -D/2, x0min, -D/2)
    x0max = np.expand_dims((x + 2/3.0*eta)/(1+2/3.0*eta/D), 0)
    xd = np.expand_dims(x, 0)
    f0 = np.expand_dims(np.linspace(0, 1, nsteps), 1)
    phi0 = (x0max-x0min)*(10*f0**3-15 *
                          f0**4+6*f0**5)+x0min
    """
    Note on this integral:

    ok, this is a bit tricky. Basically, we take _distr_D (the eta = 0 pattern),
    then for each frequency there we broaden one delta into a line that represents
    the distribution one gets by changing phi from 0 to 2pi.

    This would be all nice and good, but requires two fixes to work smoothly:
    1) make sure that x0min is properly picked, because the kernel of the 
    integral diverges at the boundaries and you don't want that to fall inside
    your interval of definition for x0, or it'll cause numerical noise. Hence
    the np.where clause above
    2) condition the kernel of the integral to tame the singularities at the
    boundaries. This is done by picking a function phi(t) such that x = phi(t)
    and phi(a) = a and phi(b) = b, while phi'(a) = phi'(b) = 0, so that

    dx = phi'(t) dt 

    and we can integrate in t instead than x and the derivative kindly kills
    off the divergence for us. Our choice of function here can be seen above 
    in the definition of phi0.
    """
    ker = _distr_D(phi0/D, 1.0)*_distr_eta(xd/D, phi0/D,
                                           1.0, eta/D)*(30*f0**2*(1-2*f0+f0**2))

    dx0 = 1.0/(nsteps-1)*(x0max[0]-x0min[0])
    return np.sum(ker, axis=0)*dx0


class DipolarField(object):

    def __init__(self, atoms, mu_pos, isotopes={}, isotope_list=None,
                 cutoff=10, overlap_eps=1e-3):

        # Get positions, cell, and species, only things we care about
        self.cell = np.array(atoms.get_cell())
        pos = atoms.get_positions()
        el = np.array(atoms.get_chemical_symbols())

        self.mu_pos = np.array(mu_pos)

        scell = minimum_supcell(cutoff, self.cell)
        grid_f, grid = supcell_gridgen(self.cell, scell)

        self.grid_f = grid_f

        r = (pos[:, None, :]+grid[None, :, :] -
             self.mu_pos[None, None, :]).reshape((-1, 3))
        rnorm = np.linalg.norm(r, axis=1)
        sphere = np.where(rnorm <= cutoff)[0]
        sphere = sphere[np.argsort(rnorm[sphere])[::-1]]  # Sort by length
        r = r[sphere]
        rnorm = rnorm[sphere]

        self._r = r
        self._rn = rnorm
        self._rn = np.where(self._rn > overlap_eps, self._rn, np.inf)
        self._ri = sphere
        self._dT = (3*r[:, :, None]*r[:, None, :] /
                    self._rn[:, None, None]**2-np.eye(3)[None, :, :])/2

        self._an = len(pos)
        self._gn = self.grid_f.shape[0]
        self._a_i = self._ri//self._gn
        self._ijk = self.grid_f[self._ri % self._gn]

        # Get gammas
        self.gammas = _get_isotope_data(el, 'gamma', isotopes, isotope_list)
        self.gammas = self.gammas[self._a_i]
        Dn = _dip_constant(self._rn*1e-10, m_gamma, self.gammas)
        De = _dip_constant(self._rn*1e-10, m_gamma,
                           cnst.physical_constants['electron gyromag. ratio'][0])

        self._D = {'n': Dn, 'e': De}

        # Start with all zeros
        self.spins = rnorm*0

    def set_moments(self, moments, moment_type='e'):

        spins = np.array(moments)
        if spins.shape != (self._an,):
            raise ValueError('Invalid moments array shape')

        try:
            self.spins = spins[self._a_i]*self._D[moment_type]
        except KeyError:
            raise ValueError('Invalid moment type')

    def dipten(self):
        return np.sum(self.spins[:, None, None]*self._dT, axis=0)

    def frequency(self, axis=[0, 0, 1]):

        D = self.dipten()
        return np.sum(np.dot(D, axis)*axis)

    def pwd_spec(self, width=None, h_steps=100):

        dten = self.dipten()
        evals, evecs = np.linalg.eigh(dten)
        evals = np.sort(evals)
        D = evals[2]
        eta = (evals[1]-evals[0])/2

        if width is None:
            width = D

        om = np.linspace(-width, width, 2*h_steps+1)
        if np.isclose(eta/D, 0):
            spec = _distr_D(om, D)
        else:
            spec = _distr_spec(om, D, eta)
        spec = (spec+spec[::-1])/2
        spec /= np.trapz(spec, om)  # Normalize

        return om, spec

    # def get_single_field(self, moments, ext_field_dir=[0, 0, 1.0], moment_type='e'):

    #     s = np.array(moments)[self._a_i]
    #     D = self._D[moment_type]

    #     n = np.array(ext_field_dir).astype(float)
    #     n /= np.linalg.norm(n)

    #     DT = np.sum((D*s)[:, None, None]*self._dT, axis=0)
    #     return np.dot(n, np.dot(DT, n))

    # def get_pwd_distribution(self, moment_gen, orients, moment_type='e'):

    #     s = moment_gen(self._a_i, self._ijk)

    #     D = self._D[moment_type]
    #     DT = np.sum((D*s)[:, None, None]*self._dT, axis=0)

    #     return np.sum(np.tensordot(DT, orients, axes=(1, 1)).T*orients, axis=1)

    # def get_zf_distribution(self, moment_gen, moment_type='e'):

    #     s = moment_gen(self._a_i, self._ijk)

    #     D = self._D[moment_type]
    #     DT = D[:, None, None]*self._dT
    #     return np.sum(DT*s[:, None, :]*s[:, :, None], axis=(1, 2))
