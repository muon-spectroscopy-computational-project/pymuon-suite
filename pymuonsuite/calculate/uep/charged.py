"""
charged.py

ChargeDistribution class for Unperturbed Electrostatic Potential
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
from ase import io
from scipy import constants as cnst
from parsefmt.fmtreader import FMTReader

from pymuonsuite.utils import make_process_slices
from pymuonsuite.io.castep import parse_castep_ppots


class ChargeDistribution(object):

    def __init__(self, seedname, gw_fac=3, path=''):

        # Load the electronic density
        seedpath = os.path.join(path, seedname)

        self._elec_den = FMTReader(seedpath + '.den_fmt')
        self._struct = io.read(seedpath + '.castep')
        ppots = parse_castep_ppots(seedpath + '.castep')

        # FFT grid
        lattice = np.array(self._elec_den.real_lattice)
        grid = np.array(self._elec_den.grid)

        dx = [np.linalg.norm(lattice[i])/grid[i] for i in range(3)]
        inv_latt = np.linalg.inv(lattice.T)*2*np.pi

        fft_grid = np.array(np.meshgrid(*[np.fft.fftfreq(grid[i])*grid[i]
                                          for i in range(3)], indexing='ij'))
        # Uses the g-vector convention in formulas used
        self._g_grid = np.tensordot(inv_latt, fft_grid, axes=(0, 0))

        # Information for the elements, and guarantee zero net charge
        elems = self._struct.get_chemical_symbols()
        pos = self._struct.get_positions()
        q = np.array([ppots[el][0] for el in elems])
        gw = np.array([ppots[el][1]/gw_fac for el in elems])

        # Here we find the Fourier components of the potential due to
        # the valence electrons
        self._rho = self._elec_den.data[:, :, :, 0]
        gvol = np.prod(self._rho.shape)
        if not np.isclose(np.sum(self._rho)/gvol, sum(q), 1e-4):
            raise RuntimeError('Cell is not neutral')
        # Put the minus sign for electrons
        self._rho *= -gvol*sum(q)/np.sum(self._rho)  # Normalise charge
        self._rhoe_G = np.fft.fftn(self._rho)
        Gnorm = np.linalg.norm(self._g_grid, axis=0)
        Gnorm_fixed = np.where(Gnorm > 0, Gnorm, np.inf)

        cell = np.array(self._elec_den.real_lattice)
        vol = abs(np.dot(np.cross(cell[:, 0], cell[:, 1]), cell[:, 2]))
        self._vol = vol

        self._Ve_G = 4*np.pi/Gnorm_fixed**2*(self._rhoe_G / (vol*gvol))

        # Now on to doing the same for ionic components
        self._rhoi_G = (q[None, None, None, :] *
                        np.exp(-1.0j*np.sum(self._g_grid[:, :, :, :, None] *
                                            pos.T[:, None, None, None, :],
                                            axis=0) -
                               0.5*(gw[None, None, None, :] *
                                    Gnorm[:, :, :, None])**2))

        pregrid = (4*np.pi/Gnorm_fixed**2*1.0/vol)
        self._Vi_G = (pregrid[:, :, :, None] * self._rhoi_G)

    @property
    def atoms(self):
        return self._struct.copy()

    @property
    def cell(self):
        return self._struct.get_cell()

    @property
    def volume(self):
        return self._vol

    @property
    def chemical_symbols(self):
        return self._struct.get_chemical_symbols()

    @property
    def positions(self):
        return self._struct.get_positions()

    @property
    def scaled_positions(self):
        return self._struct.get_scaled_positions()

    def rho(self, p, max_process_p=20):
        # Return charge density at a point or list of points
        p = np.array(p)
        if len(p.shape) == 1:
            p = p[None, :]   # Make it into a list of points

        # The point list is sliced for convenience, to avoid taking too much
        # memory
        N = p.shape[0]
        rhoe = np.zeros(N)
        rhoi = np.zeros(N)

        slices = make_process_slices(N, max_process_p)

        for s in slices:
            # Fourier transform kernel
            ftk = np.exp(1.0j*np.tensordot(self._g_grid, p[s].T, axes=(0, 0)))
            rhoe[s] = np.real(np.sum(self._rhoe_G[:, :, :, None]*ftk,
                                     axis=(0, 1, 2)))
            rhoi[s] = np.real(np.sum(self._rhoi_G[:, :, :, :, None] *
                                     ftk[:, :, :, None],
                                     axis=(0, 1, 2, 3)))
        # Convert units to e/Ang^3
        rhoe /= np.prod(self._elec_den.grid)*self._vol
        rhoi /= self._vol
        rho = rhoe+rhoi

        return rho, rhoe, rhoi

    def V(self, p, max_process_p=20):
        # Return potential at a point or list of points
        p = np.array(p)
        if len(p.shape) == 1:
            p = p[None, :]   # Make it into a list of points

        # The point list is sliced for convenience, to avoid taking too much
        # memory
        N = p.shape[0]
        Ve = np.zeros(N)
        Vi = np.zeros(N)

        slices = make_process_slices(N, max_process_p)

        for s in slices:
            # Fourier transform kernel
            ftk = np.exp(1.0j*np.tensordot(self._g_grid, p[s].T, axes=(0, 0)))
            # Compute the electronic potential
            Ve[s] = np.real(np.sum(self._Ve_G[:, :, :, None]*ftk,
                                   axis=(0, 1, 2)))
            # Now add the ionic one
            Vi[s] = np.real(np.sum(self._Vi_G[:, :, :, :, None] *
                                   ftk[:, :, :, None],
                                   axis=(0, 1, 2, 3)))

        # Coulomb constant
        cK = 1.0/(4.0*np.pi*cnst.epsilon_0)
        Ve *= cK*cnst.e*1e10  # Moving to SI units
        Vi *= cK*cnst.e*1e10

        V = Ve + Vi

        return V, Ve, Vi

    def dV(self, p, max_process_p=20):
        # Return potential gradient at a point or list of points
        p = np.array(p)
        if len(p.shape) == 1:
            p = p[None, :]   # Make it into a list of points

        # The point list is sliced for convenience, to avoid taking too much
        # memory
        N = p.shape[0]
        dVe = np.zeros((N, 3))
        dVi = np.zeros((N, 3))

        slices = make_process_slices(N, max_process_p)

        for s in slices:
            # Fourier transform kernel
            ftk = np.exp(1.0j*np.tensordot(self._g_grid, p[s].T, axes=(0, 0)))
            dftk = 1.0j*self._g_grid[:, :, :, :, None]*ftk[None, :, :, :, :]
            # Compute the electronic potential
            dVe[s] = np.real(
                np.sum(self._Ve_G[None, :, :, :, None]*dftk,
                       axis=(1, 2, 3))).T
            # Now add the ionic one
            dVi[s] = np.real(np.sum(self._Vi_G[None, :, :, :, :, None] *
                                    dftk[:, :, :, :, None],
                                    axis=(1, 2, 3, 4))).T

        # Coulomb constant
        cK = 1.0/(4.0*np.pi*cnst.epsilon_0)
        dVe *= cK*cnst.e*1e20  # Moving to SI units
        dVi *= cK*cnst.e*1e20

        dV = dVe + dVi

        return dV, dVe, dVi

    def d2V(self, p, max_process_p=20):
        # Return potential Hessian at a point or a list of points

        p = np.array(p)
        if len(p.shape) == 1:
            p = p[None, :]   # Make it into a list of points

        # The point list is sliced for convenience, to avoid taking too much
        # memory
        N = p.shape[0]
        d2Ve = np.zeros((N, 3, 3))
        d2Vi = np.zeros((N, 3, 3))

        slices = make_process_slices(N, max_process_p)

        for s in slices:
            # Fourier transform kernel
            ftk = np.exp(1.0j*np.tensordot(self._g_grid, p[s].T, axes=(0, 0)))
            g2_mat = (self._g_grid[:, None, :, :, :] *
                      self._g_grid[None, :, :, :, :])
            d2ftk = -g2_mat[:, :, :, :, :, None]*ftk[None, None, :, :, :, :]
            # Compute the electronic potential
            d2Ve[s] = np.real(
                np.sum(self._Ve_G[None, None, :, :, :, None]*d2ftk,
                       axis=(2, 3, 4))).T
            # Now add the ionic one
            d2Vi[s] = np.real(np.sum(self._Vi_G[None, None, :, :, :, :, None] *
                                     d2ftk[:, :, :, :, :, None],
                                     axis=(2, 3, 4, 5))).T

        # Coulomb constant
        cK = 1.0/(4.0*np.pi*cnst.epsilon_0)
        d2Ve *= cK*cnst.e*1e30  # Moving to SI units
        d2Vi *= cK*cnst.e*1e30

        d2V = d2Ve + d2Vi

        return d2V, d2Ve, d2Vi
