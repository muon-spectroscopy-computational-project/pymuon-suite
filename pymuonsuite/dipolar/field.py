"""
field.py

DipolarField class, computing dipolar field distributions at a muon position
"""


import numpy as np
import scipy.constants as cnst
from soprano.utils import minimum_supcell, supcell_gridgen
from soprano.calculate.powder import ZCW
from soprano.nmr.utils import _get_isotope_data, _dip_constant
from pymuonsuite.constants import m_gamma


# Dipolar line functions


def _distr_D(x, D):
    x = np.where((x > -D / 2) * (x < D), x, np.inf)
    return 1 / (3 * D * ((2 * (x / D)) / 3 + 1.0 / 3.0) ** 0.5)


def _distr_eta(x, x0, D, eta):
    den = eta**2 * (2 - 2 * x0 / D) ** 2 - 9 * (x - x0) ** 2
    den = np.where(den > 0, den, np.inf)
    y = 1.0 / den**0.5 * 3 / np.pi
    return y


def _distr_spec(x, D, eta, nsteps=3000):
    x0min = np.expand_dims((x - 2 / 3.0 * eta) / (1 - 2 / 3.0 * eta / D), 0)
    x0min = np.where(x0min > -D / 2, x0min, -D / 2)
    x0max = np.expand_dims((x + 2 / 3.0 * eta) / (1 + 2 / 3.0 * eta / D), 0)
    xd = np.expand_dims(x, 0)
    f0 = np.expand_dims(np.linspace(0, 1, nsteps), 1)
    phi0 = (x0max - x0min) * (10 * f0**3 - 15 * f0**4 + 6 * f0**5) + x0min
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
    ker = (
        _distr_D(phi0 / D, 1.0)
        * _distr_eta(xd / D, phi0 / D, 1.0, eta / D)
        * (30 * f0**2 * (1 - 2 * f0 + f0**2))
    )

    dx0 = 1.0 / (nsteps - 1) * (x0max[0] - x0min[0])
    return np.sum(ker, axis=0) * dx0


class DipolarField(object):
    def __init__(
        self,
        atoms,
        mu_pos,
        isotopes={},
        isotope_list=None,
        cutoff=10,
        overlap_eps=1e-3,
    ):

        # Get positions, cell, and species, only things we care about
        self.cell = np.array(atoms.get_cell())
        pos = atoms.get_positions()
        el = np.array(atoms.get_chemical_symbols())

        self.mu_pos = np.array(mu_pos)

        # Is it periodic?
        if np.any(atoms.get_pbc()):
            scell = minimum_supcell(cutoff, self.cell)
        else:
            scell = [1, 1, 1]

        grid_f, grid = supcell_gridgen(self.cell, scell)

        self.grid_f = grid_f

        r = (pos[:, None, :] + grid[None, :, :] - self.mu_pos[None, None, :]).reshape(
            (-1, 3)
        )
        rnorm = np.linalg.norm(r, axis=1)
        sphere = np.where(rnorm <= cutoff)[0]
        sphere = sphere[np.argsort(rnorm[sphere])[::-1]]  # Sort by length
        r = r[sphere]
        rnorm = rnorm[sphere]

        self._r = r
        self._rn = rnorm
        self._rn = np.where(self._rn > overlap_eps, self._rn, np.inf)
        self._ri = sphere
        self._dT = (
            3 * r[:, :, None] * r[:, None, :] / self._rn[:, None, None] ** 2
            - np.eye(3)[None, :, :]
        ) / 2

        self._an = len(pos)
        self._gn = self.grid_f.shape[0]
        self._a_i = self._ri // self._gn
        self._ijk = self.grid_f[self._ri % self._gn]

        # Get gammas
        self.gammas = _get_isotope_data(el, "gamma", isotopes, isotope_list)
        self.gammas = self.gammas[self._a_i]
        Dn = _dip_constant(self._rn * 1e-10, m_gamma, self.gammas)
        De = _dip_constant(
            self._rn * 1e-10,
            m_gamma,
            cnst.physical_constants["electron gyromag. ratio"][0],
        )

        self._D = {"n": Dn, "e": De}

        # Start with all zeros
        self.spins = rnorm * 0

    def set_moments(self, moments, moment_type="e"):

        spins = np.array(moments)
        if spins.shape != (self._an,):
            raise ValueError("Invalid moments array shape")

        try:
            self.spins = spins[self._a_i] * self._D[moment_type]
        except KeyError:
            raise ValueError("Invalid moment type")

    def dipten(self):
        return np.sum(self.spins[:, None, None] * self._dT, axis=0)

    def frequency(self, axis=[0, 0, 1]):

        D = self.dipten()
        return np.sum(np.dot(D, axis) * axis)

    def pwd_spec(self, width=None, h_steps=100, nsteps=100):

        dten = self.dipten()
        evals, evecs = np.linalg.eigh(dten)
        evals = np.sort(evals)
        D = evals[2]
        eta = (evals[1] - evals[0]) / 2

        if width is None:
            width = D

        om = np.linspace(-width, width, 2 * h_steps + 1)
        if np.isclose(eta / D, 0):
            spec = _distr_D(om, D)
        else:
            spec = _distr_spec(om, D, eta, nsteps=nsteps)
        spec = (spec + spec[::-1]) / 2
        spec /= np.trapz(spec, om)  # Normalize

        return om, spec

    def random_spec_uniaxial(self, axis=[0, 0, 1], width=None, h_steps=100, occ=1.0):

        # Consider individual dipolar constants
        DD = self.spins[:, None, None] * self._dT
        Ds = np.dot(np.tensordot(DD, axis, axes=(1, 0)), axis)

        if width is None:
            width = np.sum(np.abs(Ds))

        """

        To generate the spectrum we consider the probability distribution
        for each of these. Basically it's

        rho(d) = 1/2*delta(d-D)+1/2*delta(d+D)

        with the Dirac delta. So the characteristic function is

        f(t) = cos(D*t)

        which means we can generate the total distribution function as
        an inverse FFT of a product:

        rho_tot(d) = IFFT[Prod(cos(D_i*t))]

        In case of occupancy < 1.0 all we need to do is include the possibility
        of field = 0, so:

        rho(d) = (1/2*delta(d-D)+1/2*delta(d+D))*occ + delta(d)*(1-occ)

        and the rest follows easily.

        """

        dt = h_steps / (2 * h_steps + 1.0) * 2 * np.pi / width

        t = np.linspace(-h_steps * dt, h_steps * dt, 2 * h_steps + 1)
        chfun = np.prod(np.cos(Ds[:, None] * t[None, :]) * occ + (1 - occ), axis=0)

        spec = np.abs(np.fft.fftshift(np.fft.ifft(chfun)))
        om = np.linspace(-width, width, 2 * h_steps + 1)

        spec /= np.trapz(spec, om)

        return om, spec

    def random_spec_pwd(self, width=None, h_steps=100, pwdN=50, occ=1.0):

        if width is None:
            width = np.sum(np.abs(self.spins))

        pwd = ZCW("sphere")
        orients, weights = pwd.get_orient_points(pwdN)
        specs = []

        for n in orients:
            specs.append(
                self.random_spec_uniaxial(n, width=width, h_steps=h_steps, occ=occ)
            )

        specs = np.array(specs)
        om = specs[0, 0]
        spec = np.sum(specs[:, 1] * weights[:, None], axis=0)

        return om, spec

    def random_spec_zf(self, width=None, h_steps=100):

        if width is None:
            width = np.sum(np.abs(self.spins))

        dt = h_steps / (2 * h_steps + 1.0) * 2 * np.pi / width
        t = np.linspace(-h_steps * dt, h_steps * dt, 2 * h_steps + 1)

        """
        Same logic as random_spec_pwd, except here we know that each
        tensor contributes to one component of the field with a uniform
        probability distribution, with width defined by the tensor, which we
        compute here. Then the characteristic function of the uniform
        probability distribution is sinc.

        Returns three spectra, corresponding to three projected components
        of the distribution: Bx, By, and Bz.
        """

        # Find the maximal component values for each tensor
        phi = np.arctan(self._dT[:, 1] / self._dT[:, 0])
        phi = np.where(np.isnan(phi), 0, phi)
        theta = np.arctan(
            (self._dT[:, 0] * np.cos(phi) + self._dT[:, 1] * np.sin(phi))
            / self._dT[:, 2]
        )
        ct = np.cos(theta)
        st = np.sin(theta)
        cp = np.cos(phi)
        sp = np.sin(phi)
        maxdir = np.swapaxes(np.array([st * cp, st * sp, ct]), 0, 1)
        maxw = np.abs(np.sum(self._dT * maxdir, axis=1)) * self.spins[:, None]

        chfun = np.prod(
            np.sin(maxw[:, :, None] * t[None, None, :])
            / (maxw[:, :, None] * t[None, None, :]),
            axis=0,
        )
        chfun = np.where(np.isnan(chfun), 1, chfun)
        spec = np.abs(np.fft.fftshift(np.fft.ifft(chfun, axis=-1), axes=(-1)))
        om = np.linspace(-width, width, 2 * h_steps + 1)

        spec /= np.trapz(spec, om, axis=-1)[:, None]

        return om, spec
