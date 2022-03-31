"""harmonic.py

Functions to compute the solution to the quantum harmonic oscillator.
Author: Simone Sturniolo
"""

import numpy as np
import scipy.constants as cnst
from scipy.special import hermite, factorial


def harmonic_psi(x, m, om, n=0):
    s = (cnst.hbar / (m * om)) ** 0.5
    Hn = hermite(n)

    return (
        1.0
        / (2**n * factorial(n)) ** 0.5
        * (np.pi**-0.25)
        * (s**-0.5)
        * np.exp(-0.5 * (x / s) ** 2)
        * Hn(x / s)
    )


def harmonic_partfunc(om, T=0, nmax=20):
    Z = np.zeros(nmax)
    if T == 0:
        Z[0] = 1
    else:
        Z = np.exp(-cnst.hbar * om * np.arange(nmax) / (cnst.k * T))
        Z /= np.sum(Z)

    return Z


def harmonic_rho_sum(x, m, om, T=0, nmax=20):
    """Returns the density for an harmonic oscillator at T > 0
    by explicitly summing over all wavefunctions. Used
    only for debugging purposes."""

    Z = harmonic_partfunc(om, T, nmax)
    psis = np.array([harmonic_psi(x, m, om, n) for n in range(nmax)])

    rho = np.sum(Z[:, None] * psis**2, axis=0)

    return rho


def harmonic_rho(x, m, om, T=0):
    """Returns the density for an harmonic oscillator at T > 0
    using the proper formula that includes the sum."""

    rho0 = harmonic_psi(x, m, om) ** 2
    if T > 0:
        s = (cnst.hbar / (m * om)) ** 0.5
        xi = np.exp(-0.5 * cnst.hbar * om / (cnst.k * T))
        tf = (1.0 - xi**2) / (1.0 + xi**2)
        rho = tf**0.5 * rho0**tf * (np.pi**0.5 * s) ** (tf - 1)
    else:
        rho = rho0

    return rho
