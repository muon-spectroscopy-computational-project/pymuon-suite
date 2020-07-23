"""
charged.py

ChargeDistribution class for Perturbed Electrostatic Potential
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
from scipy import constants as cnst
from soprano.utils import (max_distance_in_cell, minimum_supcell,
                           supcell_gridgen)

from pymuonsuite.utils import make_process_slices
from pymuonsuite.calculate.uep import ChargeDistribution
from pymuonsuite.calculate.uep.charged import _cK


def _fk(x, k=0):
    if k <= 0:
        return x
    else:
        fkm = _fk(x, k-1)
        return 0.5*fkm*(3-fkm**2)


class PEPChargeDistribution(ChargeDistribution):

    def __init__(self, seedname, gw_fac=3, path=''):

        super().__init__(seedname, gw_fac, path)

        # Partition the charges using the Becke method
        # Becke, A. D. ‘A multicenter numerical integration scheme for polyatomic molecules’
        # J. Chem. Phys. 1988, 88, p 2547-2553. http://dx.doi.org/10.1063/1.454033
        xyz = self._elec_den.xyz
        maxR = max_distance_in_cell(self.cell)
        scell = minimum_supcell(maxR, self.cell)
        scgfrac, scgxyz = supcell_gridgen(self.cell, scell)

        # For each point, find their minimum periodic distance from each ion
        pos = self.positions.T
        scgion = pos[:, :, None]+(scgxyz.T)[:, None, :]

        # Grid
        xn, yn, zn = xyz.shape[1:]
        N = len(self.positions)

        rA = np.zeros((xn, yn, zn, N))

        for i in range(xn):
            for j in range(yn):
                rA[i, j, :, :] = np.amin(np.linalg.norm(xyz[:, i, j, :, None,
                                                            None]
                                                        - scgion[:, None, :, :],
                                                        axis=0), axis=-1)

        rAB = np.linalg.norm(pos[:, :, None]-pos[:, None, :], axis=0)
        rAB += np.diag([np.inf]*len(rAB))
        muAB = ((rA[:, :, :, :, None]-rA[:, :, :, None, :]) /
                rAB[None, None, None, :, :])

        sk = 0.5*(1.0-_fk(muAB, 3))
        # Set the diagonal to 1 so that it doesn't affect the products
        ii = range(len(self.positions))
        sk[:, :, :, ii, ii] = 1

        # Now weight functions
        wA = np.prod(sk, axis=-1)
        self._wA = wA/np.sum(wA, axis=-1)[:, :, :, None]

        # Finally, partition the density
        self._rhopart = self._rho[:, :, :, None]*self._wA
        self._rhopart_G = np.fft.fftn(self._rhopart, axes=(0, 1, 2))

        Gnorm = np.linalg.norm(self._g_grid, axis=0)
        Gnorm_fixed = np.where(Gnorm > 0, Gnorm, np.inf)

        vol = self.volume

        self._Vpart_G = 4*np.pi/Gnorm_fixed[:, :, :, None]**2*(self._rhopart_G /
                                                               vol)

        # The individual ion density must be redefined too
        self._rhoipart_G = np.zeros((xn, yn, zn, N))*0.j
        pos = self.atoms.get_positions()
        for i, p in enumerate(pos):
            self._rhoipart_G[:, :, :, i] = (self._q[i] *
                                        np.exp(-1.0j*np.sum(self._g_grid[:, :, :, :] *
                                                            p[:, None, None, None],
                                                            axis=0) -
                                               0.5*(self._gw[i] * Gnorm)**2))

        # # Now compute interaction energy of ions
        # self._rhoion_G = self._rhopart_G + self._rhoi_G
        # self._Vion_G = self._Vpart_G+self._Vi_G

        # self._ionE = np.real(np.sum(self._rhoion_G *
        #                             np.conj(np.sum(self._Vion_G, axis=-1)
        #                                     )[:, :, :, None])
        #                      )*_cK*cnst.e*1e10

    # Commented out right now as it does not work
    # def ionE(self, dr=None):

    #     if dr is None:
    #         return self._ionE

    #     # Otherwise, we need to recalculate by including the effects of the
    #     # displacements

    #     edr = np.exp(-1.0j*np.tensordot(self._g_grid, dr.T, axes=(0, 0)))

    #     return np.real(np.sum(self._rhoion_G * edr *
    #                           np.conj(np.sum(self._Vion_G*edr, axis=-1)
    #                                   )[:, :, :, None])
    #                    )*_cK*cnst.e*1e10

    def rhopart(self, p, dr=None, max_process_p=20):
        # Return partitioned charge density at a point or list of points
        p = np.array(p)
        if len(p.shape) == 1:
            p = p[None, :]   # Make it into a list of points

        # The point list is sliced for convenience, to avoid taking too much
        # memory
        N = p.shape[0]
        I = len(self.positions)
        rhoe = np.zeros((I, N))
        rhoi = np.zeros((I, N))

        if dr is None:
            dr = np.zeros((I, 3))
        else:
            dr = np.array(dr)
            if dr.shape != (I, 3):
                raise ValueError('Invalid ionic displacement vector')
        edr = np.exp(-1.0j*np.tensordot(self._g_grid, dr.T, axes=(0, 0)))

        slices = make_process_slices(N, max_process_p)

        for s in slices:
            # Fourier transform kernel
            ftk = np.exp(1.0j*np.tensordot(self._g_grid, p[s].T, axes=(0, 0)))
            rhoe[:, s] = np.real(np.sum(self._rhopart_G[:, :, :, :, None] *
                                        ftk[:, :, :, None] *
                                        edr[:, :, :, :, None],
                                        axis=(0, 1, 2)))
            rhoi[:, s] = np.real(np.sum(self._rhoipart_G[:, :, :, :, None] *
                                        ftk[:, :, :, None] *
                                        edr[:, :, :, :, None],
                                        axis=(0, 1, 2)))
        # Convert units to e/Ang^3
        rhoe /= self._vol
        rhoi /= self._vol
        rho = rhoe+rhoi

        return rho, rhoe, rhoi

    def Vpart(self, p, dr=None, max_process_p=20):
        # Return potential at a point or list of points
        p = np.array(p)
        if len(p.shape) == 1:
            p = p[None, :]   # Make it into a list of points

        # The point list is sliced for convenience, to avoid taking too much
        # memory
        N = p.shape[0]
        I = len(self.positions)
        Ve = np.zeros((I, N))
        Vi = np.zeros((I, N))

        if dr is None:
            dr = np.zeros((I, 3))
        else:
            dr = np.array(dr)
            if dr.shape != (I, 3):
                raise ValueError('Invalid ionic displacement vector')
        edr = np.exp(-1.0j*np.tensordot(self._g_grid, dr.T, axes=(0, 0)))

        slices = make_process_slices(N, max_process_p)

        for s in slices:
            # Fourier transform kernel
            ftk = np.exp(1.0j*np.tensordot(self._g_grid, p[s].T, axes=(0, 0)))
            # Compute the electronic potential
            Ve[:, s] = np.real(np.sum(self._Vpart_G[:, :, :, :, None] *
                                      ftk[:, :, :, None, :] *
                                      edr[:, :, :, :, None],
                                      axis=(0, 1, 2)))
            # Now add the ionic one
            Vi[:, s] = np.real(np.sum(self._Vi_G[:, :, :, :, None] *
                                      ftk[:, :, :, None, :] *
                                      edr[:, :, :, :, None],
                                      axis=(0, 1, 2)))

        # Coulomb constant
        Ve *= _cK*cnst.e*1e10  # Moving to SI units
        Vi *= _cK*cnst.e*1e10

        V = Ve + Vi

        return V, Ve, Vi

    def dVpart(self, p, dr=None, max_process_p=20):
        # Return potential at a point or list of points
        p = np.array(p)
        if len(p.shape) == 1:
            p = p[None, :]   # Make it into a list of points

        # The point list is sliced for convenience, to avoid taking too much
        # memory
        N = p.shape[0]
        I = len(self.positions)
        dVe = np.zeros((3, I, N))
        dVi = np.zeros((3, I, N))

        if dr is None:
            dr = np.zeros((I, 3))
        else:
            dr = np.array(dr)
            if dr.shape != (I, 3):
                raise ValueError('Invalid ionic displacement vector')
        edr = np.exp(-1.0j*np.tensordot(self._g_grid, dr.T, axes=(0, 0)))

        slices = make_process_slices(N, max_process_p)

        for s in slices:
            # Fourier transform kernel
            ftk = np.exp(1.0j*np.tensordot(self._g_grid, p[s].T, axes=(0, 0)))
            dftk = 1.0j*self._g_grid[:, :, :, :, None]*ftk[None, :, :, :, :]
            # Compute the electronic potential
            dVe[:, :, s] = np.real(np.sum(self._Vpart_G[None, :, :, :, :, None] *
                                          dftk[:, :, :, :, None, :] *
                                          edr[None, :, :, :, :, None],
                                          axis=(1, 2, 3)))
            # Now add the ionic one
            dVi[:, :, s] = np.real(np.sum(self._Vi_G[None, :, :, :, :, None] *
                                          dftk[:, :, :, :, None, :] *
                                          edr[None, :, :, :, :, None],
                                          axis=(1, 2, 3)))

        # Swap axes for convenience
        dVe = np.swapaxes(dVe, 0, 1)
        dVi = np.swapaxes(dVi, 0, 1)

        dVe *= _cK*cnst.e*1e20  # Moving to SI units
        dVi *= _cK*cnst.e*1e20

        dV = dVe + dVi

        return dV, dVe, dVi

    def d2Vpart(self, p, dr=None, max_process_p=20):
        # Return potential at a point or list of points
        p = np.array(p)
        if len(p.shape) == 1:
            p = p[None, :]   # Make it into a list of points

        # The point list is sliced for convenience, to avoid taking too much
        # memory
        N = p.shape[0]
        I = len(self.positions)
        d2Ve = np.zeros((3, 3, I, N))
        d2Vi = np.zeros((3, 3, I, N))

        if dr is None:
            dr = np.zeros((I, 3))
        else:
            dr = np.array(dr)
            if dr.shape != (I, 3):
                raise ValueError('Invalid ionic displacement vector')
        edr = np.exp(-1.0j*np.tensordot(self._g_grid, dr.T, axes=(0, 0)))

        slices = make_process_slices(N, max_process_p)

        for s in slices:
            # Fourier transform kernel
            ftk = np.exp(1.0j*np.tensordot(self._g_grid, p[s].T, axes=(0, 0)))
            dftk = 1.0j*self._g_grid[:, :, :, :, None]*ftk[None, :, :, :, :]
            g2_mat = (self._g_grid[:, None, :, :, :] *
                      self._g_grid[None, :, :, :, :])
            d2ftk = -g2_mat[:, :, :, :, :, None]*ftk[None, None, :, :, :, :]
            # Compute the electronic potential
            d2Ve[:, :, :, s] = np.real(
                np.sum(self._Vpart_G[None, None, :, :, :, :, None] *
                       d2ftk[:, :, :, :, :, None, :] *
                       edr[None, None, :, :, :, :, None],
                       axis=(2, 3, 4)))
            # Now add the ionic one
            d2Vi[:, :, :, s] = np.real(
                np.sum(self._Vi_G[None, None, :, :, :, :, None] *
                       d2ftk[:, :, :, :, :, None] *
                       edr[None, None, :, :, :, :, None],
                       axis=(2, 3, 4)))

        # Swap axes for convenience
        d2Ve = np.moveaxis(d2Ve, 2, 0)
        d2Vi = np.moveaxis(d2Vi, 2, 0)

        d2Ve *= _cK*cnst.e*1e30  # Moving to SI units
        d2Vi *= _cK*cnst.e*1e30

        d2V = d2Ve + d2Vi

        return d2V, d2Ve, d2Vi
